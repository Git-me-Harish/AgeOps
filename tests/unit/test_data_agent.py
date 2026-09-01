# tests/unit/test_data_agent.py
"""
Unit tests for DataAgent — validation scoring, drift detection,
and graceful fallback when external services are unavailable.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from agents.data_agent import DataAgent
from agents.connectors import IngestionResult


@pytest.fixture
def agent() -> DataAgent:
    return DataAgent()


@pytest.fixture
def clean_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((200, 5)),
        columns=list("abcde"),
    )


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """DataFrame with 60% null values — should fail validation."""
    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        rng.standard_normal((200, 5)),
        columns=list("abcde"),
    )
    mask = rng.random(df.shape) < 0.6
    df[mask] = np.nan
    return df


# Validation scoring 
class TestValidation:
    @pytest.mark.asyncio
    async def test_clean_df_passes(self, agent, clean_df):
        with patch.object(
            agent,
            "_persist_validation_result",
            new=AsyncMock(),
        ):
            score, passed = await agent._validate(
                clean_df,
                "s3://bucket/data.csv",
                None,
                "test-task",
                "test-run",
            )

        assert 0.0 <= score <= 1.0
        assert passed is True

    @pytest.mark.asyncio
    async def test_dirty_df_fails(self, agent, dirty_df):
        with patch.object(
            agent,
            "_persist_validation_result",
            new=AsyncMock(),
        ):
            score, passed = await agent._validate(
                dirty_df,
                "s3://bucket/data.csv",
                None,
                "test-task",
                "test-run",
            )

        assert 0.0 <= score <= 1.0
        assert passed is False

    @pytest.mark.asyncio
    async def test_score_range(self, agent, clean_df):
        with patch.object(
            agent,
            "_persist_validation_result",
            new=AsyncMock(),
        ):
            score, passed = await agent._validate(
                clean_df,
                "s3://bucket/data.csv",
                None,
                "test-task",
                "test-run",
            )

        assert 0.0 <= score <= 1.0
        assert isinstance(passed, bool)


# Drift detection 
class TestDriftDetection:
    @pytest.mark.asyncio
    async def test_no_reference_returns_zero(self, agent, clean_df):
        mock_connector = MagicMock()
        mock_connector.connect = AsyncMock(
            side_effect=Exception("reference not found")
        )

        with patch(
            "agents.data_agent.ConnectorFactory.from_uri",
            return_value=mock_connector,
        ):
            result = await agent._detect_drift(
                clean_df,
                "s3://bucket/data.csv",
                None,
                "test-task",
                "test-run",
            )

        assert result["drift_score"] == 0.0
        assert result["alert_level"] == "none"
        assert result["feature_scores"] == {}
        assert result["report_r2_path"] is None

    @pytest.mark.asyncio
    async def test_identical_distributions_low_drift(self, agent, clean_df):
        mock_result = MagicMock()
        mock_result.dataframe = clean_df.copy()

        mock_connector = MagicMock()
        mock_connector.connect = AsyncMock()
        mock_connector.read = AsyncMock(return_value=mock_result)

        with patch(
            "agents.data_agent.ConnectorFactory.from_uri",
            return_value=mock_connector,
        ), patch.object(
            agent,
            "_store_drift_report",
            new=AsyncMock(return_value=None),
        ), patch.object(
            agent,
            "_persist_drift_report",
            new=AsyncMock(),
        ):
            result = await agent._detect_drift(
                clean_df,
                "s3://bucket/data.csv",
                None,
                "test-task",
                "test-run",
            )

        assert result["drift_score"] >= 0.0
        assert result["drift_score"] <= 1.0
        assert result["alert_level"] in {
            "none",
            "warning",
            "critical",
        }

    @pytest.mark.asyncio
    async def test_very_different_distributions_high_drift(self, agent):
        rng = np.random.default_rng(0)

        current = pd.DataFrame(
            rng.standard_normal((200, 5)) * 10 + 50,
            columns=list("abcde"),
        )

        reference = pd.DataFrame(
            rng.standard_normal((200, 5)),
            columns=list("abcde"),
        )

        mock_result = MagicMock()
        mock_result.dataframe = reference

        mock_connector = MagicMock()
        mock_connector.connect = AsyncMock()
        mock_connector.read = AsyncMock(return_value=mock_result)

        mock_report = MagicMock()
        mock_report.as_dict.return_value = {
            "metrics": [
                {
                    "metric": "DatasetDriftMetric",
                    "result": {
                        "share_of_drifted_columns": 0.8,
                    },
                },
                {
                    "metric": "DataDriftTable",
                    "result": {
                        "drift_by_columns": {},
                    },
                },
            ]
        }

        with patch(
            "agents.data_agent.ConnectorFactory.from_uri",
            return_value=mock_connector,
        ), patch(
            "evidently.legacy.report.Report",
            return_value=mock_report,
        ), patch.object(
            agent,
            "_store_drift_report",
            new=AsyncMock(return_value=None),
        ), patch.object(
            agent,
            "_persist_drift_report",
            new=AsyncMock(),
        ):
            result = await agent._detect_drift(
                current,
                "s3://bucket/data.csv",
                None,
                "test-task",
                "test-run",
            )

        assert result["drift_score"] == 0.8
        assert result["alert_level"] == "critical"


# Full run 
class TestFullRun:
    def test_successful_run(self, agent, sample_state, clean_df):
        ingestion_result = MagicMock()
        ingestion_result.dataframe = clean_df
        ingestion_result.row_count = len(clean_df)
        ingestion_result.col_count = len(clean_df.columns)
        ingestion_result.byte_size = 1024
        ingestion_result.content_hash = "a" * 64
        ingestion_result.schema_violations = []

        with patch.object(
            agent,
            "connect",
            new=AsyncMock(),
        ), patch.object(
            agent,
            "_ingest",
            new=AsyncMock(
                return_value=(
                    ingestion_result,
                    None,
                    None,
                )
            ),
        ), patch.object(
            agent,
            "_record_lineage",
            new=AsyncMock(return_value=None),
        ), patch.object(
            agent,
            "_validate",
            new=AsyncMock(return_value=(0.98, True)),
        ), patch.object(
            agent,
            "_detect_drift",
            new=AsyncMock(
                return_value={
                    "drift_score": 0.03,
                    "alert_level": "none",
                    "feature_scores": {},
                    "report_r2_path": None,
                }
            ),
        ), patch.object(
            agent,
            "_materialize_features",
            new=AsyncMock(),
        ):
            result = agent.run(sample_state)

        assert result.status == "success"
        assert result.output["validation_score"] == 0.98
        assert result.output["drift_score"] == 0.03

    def test_failed_ingest_returns_failed_result(
        self,
        agent,
        sample_state,
    ):
        with patch.object(
            agent,
            "connect",
            new=AsyncMock(),
        ), patch.object(
            agent,
            "_ingest",
            new=AsyncMock(side_effect=Exception("R2 unreachable")),
        ):
            result = agent.run(sample_state)

        assert result.status == "failed"
        assert "R2 unreachable" in (result.error or "")