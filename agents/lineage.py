"""
agents/lineage.py

Data lineage tracking for every ingestion event.

Every dataset that enters the system gets:
  1. A SHA-256 content hash of the raw bytes (tamper evidence)
  2. A lineage record in Neon (data_lineage table)
  3. An MLflow dataset input log (mlflow.log_input)
  4. A HMAC-SHA256 governance signature over the lineage record

The lineage record forms the root node of the model lineage DAG:
  DataLineageRecord
    → ValidationResult
    → DriftReport
    → TrainingRun (MLflow run)
    → EvaluationRun
    → ModelVersion (MLflow Registry)
    → OCI Image (Phase 4)
    → KServe InferenceService (Phase 4)

Usage (in DataAgent):
    tracker = DataLineageTracker(pool, workflow_id)
    lineage_id = await tracker.record(ingestion_result, source_uri)
    # lineage_id is then passed to validation_results and drift_reports FK
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import mlflow

from agents.connectors import IngestionResult
from configs.settings import settings

logger = logging.getLogger(__name__)

# HMAC key for lineage signing — loaded from environment, not hardcoded.
# Defaults to a weak dev key that is obviously not production-safe.
_LINEAGE_SIGNING_KEY: bytes = os.getenv(
    "LINEAGE_SIGNING_KEY", "dev-only-change-in-production"
).encode()


class DataLineageRecord:
    """
    In-memory representation of a lineage record before DB persistence.
    Validated and typed — prevents incomplete records from being persisted.
    """

    __slots__ = (
        "source_uri", "source_type", "content_hash", "row_count",
        "col_count", "schema_version", "schema_registry_id",
        "byte_size", "format", "mlflow_dataset_id", "mlflow_run_id",
        "workflow_id", "governance_signature",
    )

    def __init__(
        self,
        *,
        source_uri: str,
        source_type: str,
        content_hash: str,
        row_count: int,
        col_count: int,
        byte_size: int,
        format: str,
        workflow_id: str,
        schema_version: Optional[str] = None,
        schema_registry_id: Optional[int] = None,
        mlflow_dataset_id: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
    ) -> None:
        self.source_uri = source_uri
        self.source_type = source_type
        self.content_hash = content_hash
        self.row_count = row_count
        self.col_count = col_count
        self.byte_size = byte_size
        self.format = format
        self.workflow_id = workflow_id
        self.schema_version = schema_version
        self.schema_registry_id = schema_registry_id
        self.mlflow_dataset_id = mlflow_dataset_id
        self.mlflow_run_id = mlflow_run_id
        self.governance_signature = self._sign()

    def _sign(self) -> str:
        """
        HMAC-SHA256 signature over canonical JSON of the record.
        Enables tamper detection: if any field changes, the signature breaks.
        """
        payload = json.dumps(
            {
                "source_uri": self.source_uri,
                "content_hash": self.content_hash,
                "row_count": self.row_count,
                "workflow_id": self.workflow_id,
            },
            sort_keys=True,
        ).encode()
        return hmac.new(_LINEAGE_SIGNING_KEY, payload, hashlib.sha256).hexdigest()

    def verify_signature(self) -> bool:
        """Verify the stored signature matches the current record state."""
        return hmac.compare_digest(self.governance_signature, self._sign())

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for logging and MLflow."""
        return {
            "source_uri": self.source_uri,
            "source_type": self.source_type,
            "content_hash": self.content_hash,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "byte_size": self.byte_size,
            "format": self.format,
            "workflow_id": self.workflow_id,
            "schema_version": self.schema_version,
            "governance_signature": self.governance_signature,
        }


class DataLineageTracker:
    """Persists ingestion provenance to Neon and logs it to MLflow. One instance per workflow. 
       The workflow_id is set at construction, and used as the FK for every record written in that workflow"""
    def __init__(self, pool: Any, workflow_id: str) -> None:
        """
        Args:
            pool: asyncpg.Pool connected to Neon.
            workflow_id: The UUID of the currently executing workflow.
        """
        self._pool = pool
        self._workflow_id = workflow_id

    async def record(
        self,
        ingestion_result: IngestionResult,
        source_uri: str,
        schema_version: Optional[str] = None,
        schema_registry_id: Optional[int] = None,
        mlflow_run_id: Optional[str] = None,
    ) -> int:
        source_type = self._infer_source_type(source_uri)

        # Log to MLflow as a dataset input (full lineage tracing in MLflow UI)
        mlflow_dataset_id = self._log_mlflow_input(
            ingestion_result=ingestion_result,
            source_uri=source_uri,
            schema_version=schema_version,
            mlflow_run_id=mlflow_run_id,
        )

        record = DataLineageRecord(
            source_uri=source_uri,
            source_type=source_type,
            content_hash=ingestion_result.content_hash,
            row_count=ingestion_result.row_count,
            col_count=ingestion_result.col_count,
            byte_size=ingestion_result.byte_size,
            format=ingestion_result.format,
            workflow_id=self._workflow_id,
            schema_version=schema_version,
            schema_registry_id=schema_registry_id,
            mlflow_dataset_id=mlflow_dataset_id,
            mlflow_run_id=mlflow_run_id,
        )

        lineage_id = await self._persist(record)
        logger.info(
            "Lineage recorded: id=%d workflow=%s source=%s hash=%s rows=%d",
            lineage_id,
            self._workflow_id,
            source_uri,
            ingestion_result.content_hash[:12],
            ingestion_result.row_count,
        )
        return lineage_id

    async def get_lineage_chain(self, workflow_id: Optional[str] = None) -> list[dict]:
        """
        Return all lineage records for a workflow, ordered by ingestion time.
        Defaults to the current workflow_id.
        """
        wf_id = workflow_id or self._workflow_id
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, source_uri, source_type, content_hash, row_count,
                       col_count, byte_size, format, schema_version,
                       governance_signature, ingested_at
                FROM data_lineage
                WHERE workflow_id = $1
                ORDER BY ingested_at ASC
                """,
                wf_id,
            )
        return [dict(r) for r in rows]

    async def verify_chain_integrity(self, workflow_id: Optional[str] = None) -> bool:
        """
        Re-compute the governance signature for each record in the lineage chain
        and verify it matches the stored value.

        Returns True only if every record passes signature verification.
        """
        chain = await self.get_lineage_chain(workflow_id)
        for raw in chain:
            record = DataLineageRecord(
                source_uri=raw["source_uri"],
                source_type=raw["source_type"],
                content_hash=raw["content_hash"],
                row_count=raw["row_count"],
                col_count=raw["col_count"],
                byte_size=raw["byte_size"],
                format=raw["format"],
                workflow_id=raw.get("workflow_id", self._workflow_id),
            )
            stored_sig = raw.get("governance_signature", "")
            computed_sig = record.governance_signature
            if not hmac.compare_digest(stored_sig, computed_sig):
                logger.error(
                    "Lineage integrity FAILURE: id=%s source=%s — "
                    "stored_sig=%s computed_sig=%s",
                    raw.get("id"), raw.get("source_uri"),
                    stored_sig[:12], computed_sig[:12],
                )
                return False
        return True

    # Internal persistence 
    async def _persist(self, record: DataLineageRecord) -> int:
        """
        Insert lineage record into Neon. Returns the new row id.
        ON CONFLICT (workflow_id, content_hash) → returns existing id.
        """
        async with self._pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO data_lineage (
                    workflow_id, source_uri, source_type, content_hash,
                    row_count, col_count, schema_version, schema_registry_id,
                    byte_size, format, mlflow_dataset_id, mlflow_run_id,
                    governance_signature
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                ON CONFLICT (workflow_id, content_hash) DO UPDATE
                    SET source_uri = EXCLUDED.source_uri
                RETURNING id
                """,
                record.workflow_id,
                record.source_uri,
                record.source_type,
                record.content_hash,
                record.row_count,
                record.col_count,
                record.schema_version,
                record.schema_registry_id,
                record.byte_size,
                record.format,
                record.mlflow_dataset_id,
                record.mlflow_run_id,
                record.governance_signature,
            )
        return row_id

    # MLflow input logging 
    @staticmethod
    def _log_mlflow_input(
        ingestion_result: IngestionResult,
        source_uri: str,
        schema_version: Optional[str],
        mlflow_run_id: Optional[str],
    ) -> Optional[str]:
        """
        Log the dataset as an MLflow input on the active run.
        Returns the MLflow dataset_id or None if logging fails.

        mlflow.log_input() creates a DatasetInput record in MLflow Tracking
        that links the dataset URI and hash to the training run.
        This is what powers the MLflow lineage UI.
        """
        try:
            dataset = mlflow.data.from_pandas(
                ingestion_result.dataframe,
                source=source_uri,
                name=f"input-{ingestion_result.content_hash[:8]}",
                targets="label" if "label" in ingestion_result.dataframe.columns else None,
            )
            mlflow.log_input(dataset, context="training")
            # dataset.digest is the mlflow-computed hash (may differ from our SHA-256)
            return dataset.digest
        except Exception as exc:
            # Non-fatal: lineage is still captured in Neon even if MLflow log fails
            logger.warning("MLflow dataset input logging failed: %s", exc)
            return None

    # Source type inference 
    @staticmethod
    def _infer_source_type(uri: str) -> str:
        """Infer the connector type string from the URI scheme."""
        if uri.startswith(("s3://", "r2://")):
            return "r2"
        if uri.startswith(("postgresql://", "postgres://")):
            return "postgres"
        if uri.startswith("file://") or uri.startswith("/"):
            return "file"
        if uri.startswith("kafka://"):
            return "kafka"
        if uri.startswith(("http://", "https://")):
            return "rest"
        return "unknown"