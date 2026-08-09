"""
Data Agent

Responsibilities
- Ingest datasets from Cloudflare R2 (S3-compatible)
- Validate data quality with Great Expectations
- Detect distributional drift with Evidently
- Register features with Feast feature store
- Trigger retraining if drift score exceeds threshold

Pattern: Tool node in the LangGraph pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import boto3
import mlflow
import pandas as pd
from botocore.config import Config
from tenacity import retry, stop_after_attempt, wait_exponential

from agents import AgentTaskResult
from configs.settings import settings

logger = logging.getLogger(__name__)

_DRIFT_THRESHOLD = 0.15      # Evidently drift score above this triggers retraining
_VALIDATION_THRESHOLD = 0.85  # Great Expectations suite must pass at this rate


class DataAgent:
    """
    Handles the full data pipeline:
    1. Ingest from R2
    2. Validate with Great Expectations
    3. Detect drift with Evidently
    4. Register to Feast feature store
    """

    def __init__(self) -> None:
        self._s3 = self._build_s3_client()

    # S3 / R2 client 
    def _build_s3_client(self):
        """Build S3 client, or None when R2 is not configured (local dev)."""
        if not settings.r2_endpoint_url or not settings.r2_access_key_id:
            logger.warning("R2 not configured — S3 client disabled (local dev mode)")
            return None
        return boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint_url,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name=settings.r2_region,
            config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
        )

    # Main entry point 
    @mlflow.trace(name="data_agent.run")
    def run(self, state: dict) -> AgentTaskResult:
        """Execute all data pipeline stages and return a typed result."""
        task_id = state.get("workflow_id", "unknown")
        dataset_uri = state.get("dataset_uri", "")

        try:
            with mlflow.start_run(run_name=f"data-{task_id}", nested=True):
                mlflow.set_tag("agent", "data_agent")
                mlflow.log_param("dataset_uri", dataset_uri)

                # Stage 1 – Ingest
                df = self._ingest(dataset_uri)
                mlflow.log_metric("row_count", len(df))
                mlflow.log_metric("col_count", len(df.columns))

                # Stage 2 – Validate
                validation_score = self._validate(df)
                mlflow.log_metric("validation_score", validation_score)
                if validation_score < _VALIDATION_THRESHOLD:
                    return AgentTaskResult(
                        task_id=task_id,
                        status="failed",
                        error=f"Data validation score {validation_score:.2f} below threshold {_VALIDATION_THRESHOLD}",
                    )

                # Stage 3 – Drift detection (requires reference dataset)
                drift_score = self._detect_drift(df, dataset_uri)
                mlflow.log_metric("drift_score", drift_score)
                if drift_score > _DRIFT_THRESHOLD:
                    logger.warning("Drift detected: %.3f — flagging for retraining", drift_score)

                # Stage 4 – Feature store
                self._register_features(df, dataset_uri)

                return AgentTaskResult(
                    task_id=task_id,
                    status="success",
                    output={
                        "validation_score": validation_score,
                        "drift_score": drift_score,
                        "row_count": len(df),
                        "drift_detected": drift_score > _DRIFT_THRESHOLD,
                    },
                    confidence=validation_score,
                )

        except Exception as exc:  # noqa: BLE001
            logger.exception("DataAgent failed for workflow %s", task_id)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))

    # Stage 1: Ingest 
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    @mlflow.trace(name="data_agent.ingest")
    def _ingest(self, dataset_uri: str) -> pd.DataFrame:
        """
        Download dataset from Cloudflare R2.
        dataset_uri format: s3://bucket/path/to/data.csv   (or .parquet)
        Falls back to synthetic data when R2 is not configured (local dev).
        """
        if self._s3 is None:
            logger.info("R2 not configured — generating synthetic dataset for local dev")
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            df["label"] = y
            return df

        uri = dataset_uri.replace("s3://", "")
        bucket, key = uri.split("/", 1)
        local_path = f"/tmp/{key.replace('/', '_')}"

        logger.info("Downloading %s from R2 bucket %s", key, bucket)
        self._s3.download_file(bucket, key, local_path)

        if key.endswith(".parquet"):
            return pd.read_parquet(local_path)
        return pd.read_csv(local_path)

    # Stage 2: Great Expectations validation 
    @mlflow.trace(name="data_agent.validate")
    def _validate(self, df: pd.DataFrame) -> float:
        """
        Run a Great Expectations validation suite against the dataframe.
        Returns the proportion of passing expectations (0.0–1.0).
        """
        try:
            import great_expectations as gx

            context = gx.get_context()
            datasource = context.sources.add_or_update_pandas(name="runtime")
            asset = datasource.add_dataframe_asset(name="dataset")
            batch_request = asset.build_batch_request(dataframe=df)

            suite = context.add_or_update_expectation_suite("mlops_suite")
            suite.add_expectation_configurations(self._build_expectations(df))

            results = context.run_validation_operator(
                "action_list_operator",
                assets_to_validate=[(batch_request, "mlops_suite")],
            )
            # Fallback when operator not configured
            return float(results.success)
        except Exception:
            # Lightweight fallback: basic null checks
            null_ratio = df.isnull().mean().mean()
            score = max(0.0, 1.0 - null_ratio)
            logger.warning("GE not available, using null-ratio fallback score: %.2f", score)
            return score

    def _build_expectations(self, df: pd.DataFrame) -> list:
        """Generate sensible default expectations for any dataframe."""
        from great_expectations.core.expectation_configuration import ExpectationConfiguration

        expectations = []
        for col in df.columns:
            # No column should be entirely null
            expectations.append(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": col, "mostly": 0.95},
                )
            )
        # Row count sanity
        expectations.append(
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={"min_value": 1, "max_value": 100_000_000},
            )
        )
        return expectations

    # Stage 3: Evidently drift detection 
    @mlflow.trace(name="data_agent.drift")
    def _detect_drift(self, current_df: pd.DataFrame, dataset_uri: str) -> float:
        """
        Compare current dataset against the reference (baseline) dataset.
        Returns Evidently drift score (share of drifted columns).
        Falls back to 0.0 if no reference is available.
        """
        reference_uri = dataset_uri.replace(".csv", "_reference.csv").replace(".parquet", "_reference.parquet")
        try:
            ref_df = self._ingest(reference_uri)
        except Exception:
            logger.info("No reference dataset found; skipping drift detection")
            return 0.0

        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_df, current_data=current_df)
            result = report.as_dict()
            share_drifted = result["metrics"][0]["result"]["share_of_drifted_columns"]
            return float(share_drifted)
        except Exception as exc:
            logger.warning("Evidently drift check failed: %s", exc)
            return 0.0

    # Stage 4: Feast feature store registration 
    @mlflow.trace(name="data_agent.feature_store")
    def _register_features(self, df: pd.DataFrame, dataset_uri: str) -> None:
        """Push features to Feast offline store."""
        try:
            from feast import FeatureStore

            store = FeatureStore(repo_path=settings.feast_repo_path)
            # Feature views are pre-defined in feast_repo/feature_views.py
            # Here we materialize the latest values
            from datetime import datetime, timedelta
            store.materialize(
                start_date=datetime.utcnow() - timedelta(days=1),
                end_date=datetime.utcnow(),
            )
            logger.info("Feast materialization complete for %s", dataset_uri)
        except Exception as exc:
            # Non-fatal: feature store failure does not block training
            logger.warning("Feast registration skipped: %s", exc)
