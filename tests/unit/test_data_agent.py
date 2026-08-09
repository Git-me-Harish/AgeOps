# tests/unit/test_data_agent.py
"""
Unit tests for DataAgent — validation scoring, drift detection,
and graceful fallback when external services are unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from agents.data_agent import DataAgent, _VALIDATION_THRESHOLD, _DRIFT_THRESHOLD


@pytest.fixture
def agent() -> DataAgent:
    with patch("agents.data_agent.boto3.client"):
        return DataAgent()


@pytest.fixture
def clean_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((200, 5)), columns=list("abcde"))


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """DataFrame with 60 % null values — should fail validation."""
    rng = np.random.default_rng(99)
    df = pd.DataFrame(rng.standard_normal((200, 5)), columns=list("abcde"))
    mask = rng.random(df.shape) < 0.6
    df[mask] = np.nan
    return df


# ── Validation scoring ────────────────────────────────────────────────────────

class TestValidation:
    def test_clean_df_passes(self, agent, clean_df):
        score = agent._validate(clean_df)
        assert score >= _VALIDATION_THRESHOLD

    def test_dirty_df_fails(self, agent, dirty_df):
        score = agent._validate(dirty_df)
        assert score < _VALIDATION_THRESHOLD

    def test_score_range(self, agent, clean_df):
        score = agent._validate(clean_df)
        assert 0.0 <= score <= 1.0


# ── Drift detection ───────────────────────────────────────────────────────────

class TestDriftDetection:
    def test_no_reference_returns_zero(self, agent, clean_df):
        with patch.object(agent, "_ingest", side_effect=Exception("not found")):
            score = agent._detect_drift(clean_df, "s3://bucket/data.csv")
        assert score == 0.0

    def test_identical_distributions_low_drift(self, agent, clean_df):
        with patch.object(agent, "_ingest", return_value=clean_df.copy()):
            score = agent._detect_drift(clean_df, "s3://bucket/data.csv")
        assert score < _DRIFT_THRESHOLD

    def test_very_different_distributions_high_drift(self, agent):
        rng = np.random.default_rng(0)
        current = pd.DataFrame(rng.standard_normal((200, 5)) * 10 + 50, columns=list("abcde"))
        reference = pd.DataFrame(rng.standard_normal((200, 5)),          columns=list("abcde"))
        with patch.object(agent, "_ingest", return_value=reference):
            with patch("agents.data_agent.Report") as mock_report_cls:
                mock_report = MagicMock()
                mock_report.as_dict.return_value = {
                    "metrics": [{"result": {"share_of_drifted_columns": 0.8}}]
                }
                mock_report_cls.return_value = mock_report
                score = agent._detect_drift(current, "s3://bucket/data.csv")
        assert score >= _DRIFT_THRESHOLD


# ── Full run with mocked ingest ───────────────────────────────────────────────

class TestFullRun:
    def test_successful_run(self, agent, sample_state, clean_df):
        with patch.object(agent, "_ingest", return_value=clean_df), \
             patch.object(agent, "_detect_drift", return_value=0.03), \
             patch.object(agent, "_register_features"):
            result = agent.run(sample_state)
        assert result.status == "success"
        assert result.output["validation_score"] >= _VALIDATION_THRESHOLD

    def test_failed_ingest_returns_failed_result(self, agent, sample_state):
        with patch.object(agent, "_ingest", side_effect=Exception("R2 unreachable")):
            result = agent.run(sample_state)
        assert result.status == "failed"
        assert "R2 unreachable" in (result.error or "")
