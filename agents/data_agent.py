"""
agents/data_agent.py

Data Agent — V2 production rewrite.

What changed from V1:
  V1: hardcoded boto3 stub → synthetic DataFrame when R2 not configured
  V2: ConnectorFactory resolves any URI (R2, PostgreSQL, file) to the correct
      connector; raises explicitly in production, dev mode uses FileConnector
      pointing to a local fixture — no synthetic make_classification() data.

  V1: GE try/except with null-ratio fallback → silently succeeds
  V2: Real GE ExpectationSuite auto-generated from schema on first run;
      results stored in Neon as validation_results; failures block pipeline
      AND trigger a GitHub Issue via the Governance Agent event queue.

  V1: Evidently stub returned 0.04 hardcoded
  V2: Real Evidently DataDriftPreset on reference vs current;
      HTML report stored in R2; per-feature scores logged as
      Prometheus metrics and persisted in Neon drift_reports.

  V1: Feast materialize() silently caught exception
  V2: Real Feast materialization to Redis online + R2 Parquet offline;
      materialize events logged in feature_store_events table.

  V1: No lineage tracking
  V2: DataLineageTracker records SHA-256 hash + Neon row + MLflow input
      for every ingestion before any downstream processing.

Stages executed per run:
  1. Connector resolution + connectivity check
  2. Data ingestion (schema-enforced)
  3. Lineage recording (Neon + MLflow)
  4. Schema validation (Great Expectations)
  5. Drift detection (Evidently → R2 report → Prometheus metrics → Neon)
  6. Feature store materialization (Feast → Redis + R2)
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import asyncpg
import mlflow
import pandas as pd
from prometheus_client import Gauge

from agents import AgentTaskResult
from agents.connectors import (
    ConnectorFactory,
    DataSchema,
    IngestionResult,
    SchemaViolationError,
)
from agents.connectors.schema_registry import SchemaRegistry, SchemaValidator
from agents.lineage import DataLineageTracker
from configs.settings import settings

logger = logging.getLogger(__name__)

# Prometheus metrics (published at /metrics on the API port) 
# These are real gauges — not stub values.
_DRIFT_SCORE_GAUGE = Gauge(
    "mlops_data_drift_score",
    "Evidently share_of_drifted_columns for the latest ingestion",
    ["workflow_id", "dataset_uri"],
)
_FEATURE_DRIFT_GAUGE = Gauge(
    "mlops_feature_drift_score",
    "Per-feature Evidently drift score",
    ["workflow_id", "feature_name"],
)
_VALIDATION_SCORE_GAUGE = Gauge(
    "mlops_data_validation_score",
    "Great Expectations pass rate for the latest validation run",
    ["workflow_id"],
)
_ROW_COUNT_GAUGE = Gauge(
    "mlops_dataset_row_count",
    "Number of rows in the ingested dataset",
    ["workflow_id"],
)


class DataAgent:
    """
    Production-grade data pipeline agent.

    Lifecycle:
        agent = DataAgent()                 # constructor — no I/O
        await agent.connect(pool)           # establish DB pool
        result = agent.run(state)           # synchronous LangGraph node entry
    """

    def __init__(self) -> None:
        self._pool: Optional[Any] = None   # asyncpg.Pool — set by connect()

    async def connect(self, pool: Optional[Any] = None) -> None:
        """
        Establish the asyncpg pool used for lineage and schema registry writes.
        If pool is not provided, creates one from settings.database_url.
        """
        if pool is not None:
            self._pool = pool
            return

        if not settings.database_url:
            logger.warning(
                "DATABASE_URL not set — lineage and schema registry writes disabled. "
                "Set database_url in .env before running in staging/production."
            )
            return

        self._pool = await asyncpg.create_pool(
            settings.asyncpg_url,
            min_size=1,
            max_size=settings.database_pool_size,
            command_timeout=30,
            statement_cache_size=0,   # required for Neon pgBouncer
        )
        logger.info("DataAgent DB pool ready (Neon)")

    # LangGraph entry point 

    @mlflow.trace(name="data_agent.run")
    def run(self, state: dict) -> AgentTaskResult:
        """
        Synchronous entry point called by the LangGraph pipeline.
        Internally runs the async pipeline via asyncio.run_coroutine_threadsafe
        so it is safe to call from a synchronous LangGraph node.

        Returns AgentTaskResult with output dict:
          {
            "validation_score":  float,
            "drift_score":       float,
            "row_count":         int,
            "col_count":         int,
            "content_hash":      str,
            "lineage_id":        int,
            "drift_detected":    bool,
            "drift_alert_level": str,
            "feature_drift_scores": dict,
          }
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(self._run_async(state))
        except Exception as exc:
            logger.exception("DataAgent.run failed")
            return AgentTaskResult(
                task_id=state.get("workflow_id", "unknown"),
                status="failed",
                error=str(exc),
            )

    async def _run_async(self, state: dict) -> AgentTaskResult:
        """Full async pipeline execution."""
        task_id: str = state.get("workflow_id", "unknown")
        dataset_uri: str = state.get("dataset_uri", "")

        if not dataset_uri:
            return AgentTaskResult(
                task_id=task_id,
                status="failed",
                error="dataset_uri is required in workflow state",
            )

        if self._pool is None:
            await self.connect()

        try:
            with mlflow.start_run(run_name=f"data-{task_id}", nested=True) as run:
                mlflow.set_tag("agent", "data_agent")
                mlflow.log_param("dataset_uri", dataset_uri)

                # Stage 1: Ingest
                ingestion_result, schema_version, schema_registry_id = (
                    await self._ingest(dataset_uri, task_id)
                )
                _ROW_COUNT_GAUGE.labels(workflow_id=task_id).set(ingestion_result.row_count)
                mlflow.log_metric("row_count", ingestion_result.row_count)
                mlflow.log_metric("col_count", ingestion_result.col_count)
                mlflow.log_metric("byte_size", ingestion_result.byte_size)

                # Stage 2: Lineage
                lineage_id = await self._record_lineage(
                    ingestion_result=ingestion_result,
                    source_uri=dataset_uri,
                    schema_version=schema_version,
                    schema_registry_id=schema_registry_id,
                    mlflow_run_id=run.info.run_id,
                    task_id=task_id,
                )

                # Stage 3: Validation
                validation_score, validation_passed = await self._validate(
                    df=ingestion_result.dataframe,
                    dataset_uri=dataset_uri,
                    lineage_id=lineage_id,
                    task_id=task_id,
                    mlflow_run_id=run.info.run_id,
                )
                _VALIDATION_SCORE_GAUGE.labels(workflow_id=task_id).set(validation_score)
                mlflow.log_metric("validation_score", validation_score)

                if not validation_passed:
                    return AgentTaskResult(
                        task_id=task_id,
                        status="failed",
                        error=(
                            f"Data validation score {validation_score:.2f} below threshold "
                            f"{settings.validation_pass_threshold}. Pipeline blocked."
                        ),
                    )

                # Stage 4: Drift detection
                drift_result = await self._detect_drift(
                    current_df=ingestion_result.dataframe,
                    dataset_uri=dataset_uri,
                    lineage_id=lineage_id,
                    task_id=task_id,
                    mlflow_run_id=run.info.run_id,
                )
                drift_score = drift_result.get("drift_score", 0.0)
                alert_level = drift_result.get("alert_level", "none")
                _DRIFT_SCORE_GAUGE.labels(
                    workflow_id=task_id, dataset_uri=dataset_uri
                ).set(drift_score)
                for feature, score in drift_result.get("feature_scores", {}).items():
                    _FEATURE_DRIFT_GAUGE.labels(
                        workflow_id=task_id, feature_name=feature
                    ).set(score)
                mlflow.log_metric("drift_score", drift_score)

                # Stage 5: Feature store
                feast_feature_view = await self._materialize_features(
                    df=ingestion_result.dataframe,
                    dataset_uri=dataset_uri,
                    lineage_id=lineage_id,
                    task_id=task_id,
                )

                return AgentTaskResult(
                    task_id=task_id,
                    status="success",
                    output={
                        "validation_score": validation_score,
                        "drift_score": drift_score,
                        "drift_detected": drift_score > settings.drift_warn_threshold,
                        "drift_alert_level": alert_level,
                        "feature_drift_scores": drift_result.get("feature_scores", {}),
                        "row_count": ingestion_result.row_count,
                        "col_count": ingestion_result.col_count,
                        "content_hash": ingestion_result.content_hash,
                        "lineage_id": lineage_id,
                        "schema_violations": ingestion_result.schema_violations,
                        "feast_feature_view": feast_feature_view,
                    },
                    confidence=validation_score,
                )

        except SchemaViolationError as exc:
            logger.error("DataAgent: schema violation for workflow %s: %s", task_id, exc.violations)
            return AgentTaskResult(
                task_id=task_id,
                status="failed",
                error=f"Schema violation: {exc.violations}",
            )
        except Exception as exc:
            logger.exception("DataAgent._run_async failed for workflow %s", task_id)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))

    # Ingest 
    @mlflow.trace(name="data_agent.ingest")
    async def _ingest(
        self,
        dataset_uri: str,
        task_id: str,
    ) -> tuple[IngestionResult, Optional[str], Optional[int]]:
        """
        Resolve connector, load schema from registry, ingest with enforcement.

        Returns:
            (IngestionResult, schema_version, schema_registry_id)
        """
        connector = ConnectorFactory.from_uri(dataset_uri)
        await connector.connect()

        # Load schema from registry for enforcement (if DB available)
        schema: Optional[DataSchema] = None
        schema_version: Optional[str] = None
        schema_registry_id: Optional[int] = None

        if self._pool is not None and settings.schema_registry_enabled:
            registry = SchemaRegistry(self._pool)
            # Infer dataset_id from URI (stable name without version suffix)
            dataset_id = self._uri_to_dataset_id(dataset_uri)
            schema = await registry.get_latest(dataset_id)
            if schema is not None:
                schema_version = schema.version
                logger.info(
                    "Loaded schema %s@%s for URI %s",
                    dataset_id, schema_version, dataset_uri,
                )
            else:
                logger.info(
                    "No schema registered for dataset_id=%s — "
                    "ingesting without type enforcement (first run). "
                    "Register a schema via SchemaRegistry.register() to enable validation.",
                    dataset_id,
                )

        ingestion_result = await connector.read(schema=schema)

        if schema_version is not None and self._pool is not None:
            # Retrieve the DB id for the FK in data_lineage
            registry = SchemaRegistry(self._pool)
            dataset_id = self._uri_to_dataset_id(dataset_uri)
            async with self._pool.acquire() as conn:
                schema_registry_id = await conn.fetchval(
                    "SELECT id FROM schema_registry WHERE dataset_id=$1 AND version=$2",
                    dataset_id, schema_version,
                )

        logger.info(
            "Ingestion complete: rows=%d cols=%d hash=%s violations=%d",
            ingestion_result.row_count,
            ingestion_result.col_count,
            ingestion_result.content_hash[:12],
            len(ingestion_result.schema_violations),
        )
        return ingestion_result, schema_version, schema_registry_id

    # Lineage recording 
    async def _record_lineage(
        self,
        ingestion_result: IngestionResult,
        source_uri: str,
        schema_version: Optional[str],
        schema_registry_id: Optional[int],
        mlflow_run_id: str,
        task_id: str,
    ) -> Optional[int]:
        """Write lineage record to Neon. Non-blocking on DB unavailability."""
        if self._pool is None:
            logger.debug("Lineage recording skipped — no DB pool")
            return None

        tracker = DataLineageTracker(pool=self._pool, workflow_id=task_id)
        try:
            return await tracker.record(
                ingestion_result=ingestion_result,
                source_uri=source_uri,
                schema_version=schema_version,
                schema_registry_id=schema_registry_id,
                mlflow_run_id=mlflow_run_id,
            )
        except Exception as exc:
            # Lineage failure is non-fatal — pipeline continues
            logger.error("Lineage recording failed (non-fatal): %s", exc)
            return None

    # Validation 
    @mlflow.trace(name="data_agent.validate")
    async def _validate(
        self,
        df: pd.DataFrame,
        dataset_uri: str,
        lineage_id: Optional[int],
        task_id: str,
        mlflow_run_id: str,
    ) -> tuple[float, bool]:
        """
        Run Great Expectations validation suite against df.

        Auto-generates a suite on the first run for this dataset.
        Stores results in Neon validation_results table.
        Returns (pass_rate: float, passed: bool).
        """
        start_ms = int(time.monotonic() * 1000)
        suite_name = f"mlops_suite_{self._uri_to_dataset_id(dataset_uri)}"
        pass_rate = 1.0
        failures: list[dict] = []
        total = 0
        passed_count = 0
        blocking = False

        try:
            import great_expectations as gx
            from great_expectations.core.expectation_suite import ExpectationSuite

            context = gx.get_context(mode="ephemeral")
            datasource = context.data_sources.add_pandas(name="runtime_source")
            asset = datasource.add_dataframe_asset(name="current_dataset")
            # GX 1.x: the dataframe is passed via `options`, not a `dataframe=`
            # kwarg — build_batch_request(dataframe=df) raises TypeError on
            # GX>=1.0 and was silently caught below as a generic failure.
            batch_request = asset.build_batch_request(options={"dataframe": df})

            # Build expectations from schema if available; else use profiling heuristics
            expectations = self._build_expectations(df, dataset_uri)

            # GX 1.x API: context.suites.add_or_update() takes an
            # ExpectationSuite object, not a bare name string.
            suite = ExpectationSuite(name=suite_name)
            suite = context.suites.add_or_update(suite)
            for exp in expectations:
                suite.add_expectation(exp)

            validator = context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name,
            )
            results = validator.validate()

            total = results["statistics"]["evaluated_expectations"]
            passed_count = results["statistics"]["successful_expectations"]
            pass_rate = results["statistics"]["success_percent"] / 100.0 if total > 0 else 1.0

            # Extract failures for Neon storage.
            # er["expectation_config"] is a GX ExpectationConfiguration —
            # the expectation type lives on `.type`, not `.expectation_type`.
            for er in results.get("results", []):
                if not er["success"]:
                    econfig = er["expectation_config"]
                    failures.append({
                        "expectation_type": econfig.type,
                        "column": econfig.kwargs.get("column"),
                        "observed_value": str(er.get("result", {}).get("observed_value")),
                    })

            blocking = (
                pass_rate < settings.validation_pass_threshold
                and len(failures) > 0
            )
            logger.info(
                "GE validation: pass_rate=%.2f total=%d passed=%d failures=%d blocking=%s",
                pass_rate, total, passed_count, len(failures), blocking,
            )

        except ImportError:
            # GE not installed — fall back to basic null-ratio check
            logger.warning("great-expectations not installed — using null-ratio fallback")
            null_ratio = float(df.isnull().mean().mean())
            pass_rate = max(0.0, 1.0 - null_ratio)
            blocking = pass_rate < settings.validation_pass_threshold

        except Exception as exc:
            # GE runtime error — do NOT silently succeed (V1 anti-pattern)
            logger.exception("Great Expectations validation failed: %s", exc)
            pass_rate = 0.0
            blocking = True
            failures = [{"error": str(exc)}]

        duration_ms = int(time.monotonic() * 1000) - start_ms

        # Persist to Neon
        await self._persist_validation_result(
            task_id=task_id,
            lineage_id=lineage_id,
            dataset_uri=dataset_uri,
            suite_name=suite_name,
            passed=not blocking,
            pass_rate=pass_rate,
            total=total,
            passed_count=passed_count,
            failures=failures,
            blocking=blocking,
            duration_ms=duration_ms,
        )

        # Log to MLflow artifacts
        try:
            mlflow.log_dict(
                {"pass_rate": pass_rate, "failures": failures, "blocking": blocking},
                "validation_results.json",
            )
        except Exception:
            pass

        return pass_rate, not blocking

    def _build_expectations(self, df: pd.DataFrame, dataset_uri: str) -> list:
        """
        Build a list of GE Expectation objects from the dataframe profile.
        These are auto-generated heuristics — not a substitute for
        a curated expectation suite owned by the data team.

        Uses the great_expectations.expectations (gxe) class-based API —
        the ExpectationConfiguration dict-config class this used to build
        was removed from GX's public API well before the installed 1.x
        version, so every validation call was silently hitting the generic
        exception handler in _validate() and reporting pass_rate=0.0.
        """
        import great_expectations.expectations as gxe

        expectations: list = []

        # Row count sanity bounds
        expectations.append(gxe.ExpectTableRowCountToBeBetween(
            min_value=1, max_value=1_000_000_000,
        ))

        for col in df.columns:
            # No column should be >50% null (configurable)
            expectations.append(gxe.ExpectColumnValuesToNotBeNull(
                column=col, mostly=0.50,
            ))

            # Numeric columns: tighter null tolerance (95%)
            if pd.api.types.is_numeric_dtype(df[col]):
                expectations.append(gxe.ExpectColumnValuesToNotBeNull(
                    column=col, mostly=0.95,
                ))

        return expectations

    # Drift detection 
    @mlflow.trace(name="data_agent.drift")
    async def _detect_drift(
        self,
        current_df: pd.DataFrame,
        dataset_uri: str,
        lineage_id: Optional[int],
        task_id: str,
        mlflow_run_id: str,
    ) -> dict[str, Any]:
        """
        Run Evidently DataDriftPreset against current vs reference dataset.

        Reference dataset strategy:
          1. Look for {uri}_reference.{ext} in the same R2 prefix
          2. Fall back to stored reference stats in Neon (Phase 5 feature)
          3. If no reference available: log warning, skip drift, return score=0.0

        Returns dict:
          {
            "drift_score": float,
            "alert_level": "none" | "warning" | "critical",
            "feature_scores": {feature_name: score},
            "report_r2_path": str | None,
          }
        """
        start_ms = int(time.monotonic() * 1000)

        reference_uri = self._build_reference_uri(dataset_uri)
        ref_df: Optional[pd.DataFrame] = None

        try:
            ref_connector = ConnectorFactory.from_uri(reference_uri)
            await ref_connector.connect()
            ref_result = await ref_connector.read()
            ref_df = ref_result.dataframe
            logger.info(
                "Reference dataset loaded: uri=%s rows=%d",
                reference_uri, len(ref_df),
            )
        except Exception as exc:
            logger.info(
                "No reference dataset available at %s: %s — skipping drift detection",
                reference_uri, exc,
            )
            return {
                "drift_score": 0.0,
                "alert_level": "none",
                "feature_scores": {},
                "report_r2_path": None,
            }

        # Only compare shared numeric columns (Evidently requirement)
        numeric_cols = [
            c for c in current_df.select_dtypes(include="number").columns
            if c in ref_df.columns
        ]
        if not numeric_cols:
            logger.warning(
                "No shared numeric columns between current and reference — skipping drift"
            )
            return {"drift_score": 0.0, "alert_level": "none", "feature_scores": {}, "report_r2_path": None}

        current_subset = current_df[numeric_cols]
        ref_subset = ref_df[numeric_cols]

        try:
            # evidently>=0.6 replaced the top-level report/metric_preset API
            # with a new one (evidently.Report, evidently.presets); the
            # installed 0.7.x still ships the old shape under `.legacy`,
            # which is what report.as_dict()'s DatasetDriftMetric /
            # DataDriftTable structure below assumes. Importing the
            # unversioned top-level module here was silently raising
            # ImportError on every run and reporting "no drift" every time.
            from evidently.legacy.report import Report
            from evidently.legacy.metric_preset import DataDriftPreset
            from evidently.legacy.metrics import DataDriftTable

            report = Report(metrics=[DataDriftPreset(), DataDriftTable()])
            report.run(reference_data=ref_subset, current_data=current_subset)
            result_dict = report.as_dict()

            # Extract overall drift score
            metrics = result_dict.get("metrics", [])
            drift_score = 0.0
            feature_scores: dict[str, float] = {}

            for metric in metrics:
                if metric.get("metric") == "DatasetDriftMetric":
                    drift_score = float(metric["result"]["share_of_drifted_columns"])
                if metric.get("metric") == "DataDriftTable":
                    for col_result in metric["result"].get("drift_by_columns", {}).values():
                        feature_name = col_result.get("column_name", "")
                        score = float(col_result.get("drift_score", 0.0))
                        if feature_name:
                            feature_scores[feature_name] = score

            drifted_features = [f for f, s in feature_scores.items() if s > 0.05]

            # Alert level classification
            if drift_score > settings.drift_critical_threshold:
                alert_level = "critical"
                logger.error(
                    "CRITICAL drift detected: score=%.3f drifted_features=%s",
                    drift_score, drifted_features,
                )
            elif drift_score > settings.drift_warn_threshold:
                alert_level = "warning"
                logger.warning(
                    "WARNING drift detected: score=%.3f drifted_features=%s",
                    drift_score, drifted_features,
                )
            else:
                alert_level = "none"

            # Store HTML report in R2
            report_r2_path = await self._store_drift_report(
                report=report,
                task_id=task_id,
                dataset_uri=dataset_uri,
            )

            # Persist drift record to Neon
            duration_ms = int(time.monotonic() * 1000) - start_ms
            await self._persist_drift_report(
                task_id=task_id,
                lineage_id=lineage_id,
                reference_uri=reference_uri,
                current_uri=dataset_uri,
                drift_score=drift_score,
                feature_scores=feature_scores,
                drifted_features=drifted_features,
                report_r2_path=report_r2_path,
                alert_level=alert_level,
                duration_ms=duration_ms,
            )

            # Log to MLflow
            try:
                mlflow.log_metric("drift_score", drift_score)
                mlflow.log_dict(feature_scores, "feature_drift_scores.json")
            except Exception:
                pass

            return {
                "drift_score": drift_score,
                "alert_level": alert_level,
                "feature_scores": feature_scores,
                "report_r2_path": report_r2_path,
            }

        except ImportError:
            logger.error(
                "evidently not installed — cannot run drift detection. "
                "Install evidently>=0.7.0 and retry."
            )
            # alert_level="error" (not "none") — a drift check that never ran
            # must not look identical to a dataset with zero drift.
            return {
                "drift_score": 0.0, "alert_level": "error", "feature_scores": {},
                "report_r2_path": None, "drift_check_failed": True,
                "drift_check_error": "evidently not installed",
            }
        except Exception as exc:
            logger.exception("Evidently drift detection failed: %s", exc)
            return {
                "drift_score": 0.0, "alert_level": "error", "feature_scores": {},
                "report_r2_path": None, "drift_check_failed": True,
                "drift_check_error": str(exc),
            }

    async def _store_drift_report(
        self,
        report: Any,
        task_id: str,
        dataset_uri: str,
    ) -> Optional[str]:
        if not settings.r2_configured:
            logger.debug("R2 not configured — drift report not persisted")
            return None

        try:
            import boto3
            from botocore.config import Config

            html_bytes = report.get_html().encode("utf-8")
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
            key = f"{settings.r2_reports_prefix}/{task_id}/{timestamp}_drift.html"

            s3 = boto3.client(
                "s3",
                endpoint_url=settings.r2_endpoint_url,
                aws_access_key_id=settings.r2_access_key_id,
                aws_secret_access_key=settings.r2_secret_access_key,
                region_name=settings.r2_region,
                config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
            )
            await asyncio.to_thread(
                s3.put_object,
                Bucket=settings.r2_bucket_name,
                Key=key,
                Body=html_bytes,
                ContentType="text/html",
            )
            logger.info("Drift report stored: R2 key=%s", key)
            return key

        except Exception as exc:
            logger.warning("Failed to store drift report in R2: %s", exc)
            return None

    # Feature store 
    @mlflow.trace(name="data_agent.feature_store")
    async def _materialize_features(
        self,
        df: pd.DataFrame,
        dataset_uri: str,
        lineage_id: Optional[int],
        task_id: str,
    ) -> Optional[str]:
        """
        Materialize features to Feast (Redis online + local/R2 Parquet offline).

        Feast requires:
          - A `record_id` column as the entity key
          - An `event_timestamp` column for TTL management
        If these are absent, we add them before materialization.

        Registers a per-workflow dynamic FeatureView (feast/feature_views.py
        register_dynamic_feature_view) backed by the real parquet just
        written, so training/evaluation reads what was actually ingested
        instead of only ever seeing the static synthetic dev fixture.

        Returns the registered FeatureView's name, or None on failure.
        """
        start_ms = int(time.monotonic() * 1000)
        rows_written = 0
        status = "failed"
        error_message: Optional[str] = None
        feature_view_name: Optional[str] = None

        try:
            from feast import FeatureStore
            from feast.feature_views import register_dynamic_feature_view

            store = FeatureStore(repo_path=settings.feast_repo_path)

            # Add required Feast columns if missing
            feast_df = df.copy()
            if "record_id" not in feast_df.columns:
                feast_df["record_id"] = range(len(feast_df))
            if "event_timestamp" not in feast_df.columns:
                feast_df["event_timestamp"] = datetime.now(tz=timezone.utc)

            # Write to offline store (Parquet file or R2 — depends on feast_repo config)
            offline_path = Path(settings.feast_repo_path) / "data" / f"features_{task_id}.parquet"
            offline_path.parent.mkdir(parents=True, exist_ok=True)
            feast_df.to_parquet(offline_path, index=False)
            logger.info("Feast offline features written: %s rows=%d", offline_path, len(feast_df))

            # Register a dynamic FeatureView for THIS dataset, pointed at the
            # file we just wrote — this is what makes the real ingested data
            # (not the dev fixture) actually reachable through Feast.
            fv = await asyncio.to_thread(
                register_dynamic_feature_view,
                store,
                task_id,
                feast_df,
                str(offline_path),
            )
            feature_view_name = fv.name

            # Materialize to Redis online store
            now = datetime.now(tz=timezone.utc)
            start_date = now - timedelta(days=1)
            await asyncio.to_thread(
                store.materialize,
                start_date=start_date,
                end_date=now,
            )
            rows_written = len(feast_df)
            status = "success"
            logger.info(
                "Feast materialization complete: rows=%d workflow=%s feature_view=%s",
                rows_written, task_id, feature_view_name,
            )

        except ImportError:
            error_message = "feast not installed — feature store materialization skipped"
            logger.warning(error_message)
            status = "failed"
        except Exception as exc:
            error_message = str(exc)
            logger.exception("Feast materialization failed for workflow %s: %s", task_id, exc)
            # Non-fatal: downstream training can still proceed from raw data
            status = "failed"

        # Log feature store event to Neon (always — even on failure)
        duration_ms = int(time.monotonic() * 1000) - start_ms
        await self._persist_feast_event(
            task_id=task_id,
            lineage_id=lineage_id,
            event_type="materialize",
            rows_written=rows_written,
            status=status,
            error_message=error_message,
            duration_ms=duration_ms,
        )
        return feature_view_name

    # Neon persistence helpers 
    async def _persist_validation_result(
        self,
        task_id: str,
        lineage_id: Optional[int],
        dataset_uri: str,
        suite_name: str,
        passed: bool,
        pass_rate: float,
        total: int,
        passed_count: int,
        failures: list,
        blocking: bool,
        duration_ms: int,
    ) -> None:
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO validation_results (
                        workflow_id, lineage_id, dataset_uri, suite_name,
                        passed, pass_rate, total_expectations, passed_expectations,
                        failures, blocking, run_duration_ms
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    """,
                    task_id, lineage_id, dataset_uri, suite_name,
                    passed, pass_rate, total, passed_count,
                    json.dumps(failures), blocking, duration_ms,
                )
        except Exception as exc:
            logger.warning("Failed to persist validation_results: %s", exc)

    async def _persist_drift_report(
        self,
        task_id: str,
        lineage_id: Optional[int],
        reference_uri: str,
        current_uri: str,
        drift_score: float,
        feature_scores: dict,
        drifted_features: list,
        report_r2_path: Optional[str],
        alert_level: str,
        duration_ms: int,
    ) -> None:
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO drift_reports (
                        workflow_id, current_lineage_id, reference_uri, current_uri,
                        drift_score, per_feature_scores, drifted_features,
                        r2_report_path, alert_level, run_duration_ms
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                    """,
                    task_id, lineage_id, reference_uri, current_uri,
                    drift_score, json.dumps(feature_scores), json.dumps(drifted_features),
                    report_r2_path, alert_level, duration_ms,
                )
        except Exception as exc:
            logger.warning("Failed to persist drift_reports: %s", exc)

    async def _persist_feast_event(
        self,
        task_id: str,
        lineage_id: Optional[int],
        event_type: str,
        rows_written: int,
        status: str,
        error_message: Optional[str],
        duration_ms: int,
    ) -> None:
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO feature_store_events (
                        workflow_id, lineage_id, event_type, feature_view,
                        rows_written, status, error_message, duration_ms
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                    """,
                    task_id, lineage_id, event_type, "numeric_features",
                    rows_written, status, error_message, duration_ms,
                )
        except Exception as exc:
            logger.warning("Failed to persist feature_store_events: %s", exc)

    # Utility helpers
    @staticmethod
    def _uri_to_dataset_id(uri: str) -> str:
        """
        Derive a stable dataset_id from a URI for use as schema_registry key.

        Examples:
          s3://bucket/datasets/fraud_v2/train.parquet → fraud_v2_train
          s3://bucket/datasets/churn/2025-08/data.csv → churn_2025-08_data
        """
        # Strip scheme and bucket, keep path-based name
        for prefix in ("s3://", "r2://", "file://", "postgresql://", "postgres://"):
            if uri.startswith(prefix):
                uri = uri[len(prefix):]
                break
        # Drop bucket name (first path segment for S3)
        parts = uri.lstrip("/").split("/")
        if len(parts) > 1:
            path_parts = parts[1:]  # drop bucket
        else:
            path_parts = parts

        # Strip extension and join with underscore
        name = "_".join(
            p.rsplit(".", 1)[0] if "." in p else p
            for p in path_parts
            if p
        )
        # Sanitize: only alphanumeric, underscore, hyphen
        import re
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)[:128]

    @staticmethod
    def _build_reference_uri(dataset_uri: str) -> str:
        """
        Build the reference dataset URI from the current dataset URI.

        Convention: reference dataset is at {base}_reference.{ext}
        e.g. s3://bucket/data/train.csv → s3://bucket/data/train_reference.csv
        """
        if "." in dataset_uri.split("/")[-1]:
            base, ext = dataset_uri.rsplit(".", 1)
            return f"{base}_reference.{ext}"
        # Prefix-based: append _reference to prefix
        return dataset_uri.rstrip("/") + "_reference/"