# tests/unit/test_monitoring_agent.py
"""
Unit tests for MonitoringAgent (Phase 5).

Focus: the parts that used to be hardcoded stubs in V1 — threshold
classification, fail-closed behavior on check failures (never silently
"no drift"/"healthy"), and retraining-trigger dedup — plus the
_check_drift/_check_serving_metrics/_check_accuracy contracts each return.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import mlflow
import pytest

from agents.monitoring_agent import MonitoringAgent, _severity_rank


@pytest.fixture
def agent(reset_mlflow, tmp_path) -> MonitoringAgent:
    # MonitoringAgent.__init__ calls mlflow.set_tracking_uri(settings...),
    # which would otherwise stomp on conftest's per-test sqlite tracking URI
    # if fixture resolution ever runs it after reset_mlflow — reassert it
    # explicitly so tests are never order-dependent on that.
    a = MonitoringAgent()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-experiment")
    return a


def _make_pool(fetchrow_side_effect=None, fetchval_return=None):
    conn = AsyncMock()
    if fetchrow_side_effect is not None:
        conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)
    conn.fetchval = AsyncMock(return_value=fetchval_return)
    conn.execute = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool, conn


class TestSeverityRank:
    def test_critical_outranks_warning(self):
        assert _severity_rank("critical") > _severity_rank("warning")

    def test_warning_and_error_same_rank(self):
        assert _severity_rank("warning") == _severity_rank("error")

    def test_none_is_lowest(self):
        assert _severity_rank("none") == 0


class TestCheckAsyncThresholds:
    """
    These exercise _check_async's threshold classification directly by
    mocking the three sub-checks — the exact logic that replaced V1's
    hardcoded 0.83/0.04 return values.
    """

    async def test_clean_run_reports_none(self, agent):
        with patch.object(agent, "_check_drift", new=AsyncMock(return_value={
            "alert_level": "none", "drift_score": 0.02, "feature_scores": {}, "sample_count": 100,
        })), patch.object(agent, "_check_serving_metrics", new=AsyncMock(return_value={
            "alert_level": "none", "p99_ms": 50.0, "error_rate": 0.001,
        })), patch.object(agent, "_check_accuracy", new=AsyncMock(return_value={
            "alert_level": "none", "accuracy": 0.9, "baseline_accuracy": 0.89,
        })), patch.object(agent, "_persist_event", new=AsyncMock()):
            event = await agent._check_async("test-model")

        assert event["alert_level"] == "none"
        assert event["alerts"] == []

    async def test_drift_above_critical_threshold_triggers_retraining(self, agent):
        with patch.object(agent, "_check_drift", new=AsyncMock(return_value={
            "alert_level": "none", "drift_score": 0.35, "feature_scores": {"f0": 0.4}, "sample_count": 100,
        })), patch.object(agent, "_check_serving_metrics", new=AsyncMock(return_value={
            "alert_level": "none", "p99_ms": 50.0, "error_rate": 0.001,
        })), patch.object(agent, "_check_accuracy", new=AsyncMock(return_value={
            "alert_level": "none", "accuracy": 0.9, "baseline_accuracy": 0.89,
        })), patch.object(agent, "_persist_event", new=AsyncMock()), patch.object(
            agent, "_trigger_retraining", new=AsyncMock()
        ) as mock_trigger:
            event = await agent._check_async("test-model")

        assert event["alert_level"] == "critical"
        assert any("drift" in a.lower() for a in event["alerts"])
        mock_trigger.assert_awaited_once()

    async def test_drift_between_warn_and_critical_is_warning_only(self, agent):
        with patch.object(agent, "_check_drift", new=AsyncMock(return_value={
            "alert_level": "none", "drift_score": 0.20, "feature_scores": {}, "sample_count": 100,
        })), patch.object(agent, "_check_serving_metrics", new=AsyncMock(return_value={
            "alert_level": "none", "p99_ms": 50.0, "error_rate": 0.001,
        })), patch.object(agent, "_check_accuracy", new=AsyncMock(return_value={
            "alert_level": "none", "accuracy": 0.9, "baseline_accuracy": 0.89,
        })), patch.object(agent, "_persist_event", new=AsyncMock()), patch.object(
            agent, "_trigger_retraining", new=AsyncMock()
        ) as mock_trigger:
            event = await agent._check_async("test-model")

        assert event["alert_level"] == "warning"
        mock_trigger.assert_not_awaited()

    async def test_accuracy_drop_beyond_threshold_is_critical(self, agent):
        with patch.object(agent, "_check_drift", new=AsyncMock(return_value={
            "alert_level": "none", "drift_score": 0.0, "feature_scores": {}, "sample_count": 100,
        })), patch.object(agent, "_check_serving_metrics", new=AsyncMock(return_value={
            "alert_level": "none", "p99_ms": 50.0, "error_rate": 0.001,
        })), patch.object(agent, "_check_accuracy", new=AsyncMock(return_value={
            "alert_level": "none", "accuracy": 0.70, "baseline_accuracy": 0.90,
        })), patch.object(agent, "_persist_event", new=AsyncMock()), patch.object(
            agent, "_trigger_retraining", new=AsyncMock()
        ) as mock_trigger:
            event = await agent._check_async("test-model")

        assert event["alert_level"] == "critical"
        assert any("accuracy" in a.lower() for a in event["alerts"])
        mock_trigger.assert_awaited_once()

    async def test_failed_drift_check_reports_error_not_none(self, agent):
        """
        This is the direct regression test for the V1 anti-pattern: a check
        that could not run must never look identical to "no drift found".
        """
        with patch.object(agent, "_check_drift", new=AsyncMock(return_value={
            "alert_level": "error", "check_error": "Loki unreachable",
        })), patch.object(agent, "_check_serving_metrics", new=AsyncMock(return_value={
            "alert_level": "none", "p99_ms": 50.0, "error_rate": 0.001,
        })), patch.object(agent, "_check_accuracy", new=AsyncMock(return_value={
            "alert_level": "none", "accuracy": None, "baseline_accuracy": None,
        })), patch.object(agent, "_persist_event", new=AsyncMock()):
            event = await agent._check_async("test-model")

        assert event["alert_level"] == "error"
        assert event["drift_score"] is None
        assert any("drift check unavailable" in e for e in event["check_errors"])

    async def test_no_ground_truth_leaves_accuracy_none_not_fabricated(self, agent):
        """Direct regression test for the V1 hardcoded 0.83 accuracy stub."""
        with patch.object(agent, "_check_drift", new=AsyncMock(return_value={
            "alert_level": "none", "drift_score": 0.0, "feature_scores": {}, "sample_count": 100,
        })), patch.object(agent, "_check_serving_metrics", new=AsyncMock(return_value={
            "alert_level": "none", "p99_ms": 50.0, "error_rate": 0.001,
        })), patch.object(agent, "_check_accuracy", new=AsyncMock(return_value={
            "alert_level": "none", "accuracy": None, "baseline_accuracy": 0.9,
        })), patch.object(agent, "_persist_event", new=AsyncMock()):
            event = await agent._check_async("test-model")

        assert event["accuracy"] is None
        assert event["alert_level"] == "none"

    async def test_latency_breach_is_warning_not_critical(self, agent):
        with patch.object(agent, "_check_drift", new=AsyncMock(return_value={
            "alert_level": "none", "drift_score": 0.0, "feature_scores": {}, "sample_count": 100,
        })), patch.object(agent, "_check_serving_metrics", new=AsyncMock(return_value={
            "alert_level": "none", "p99_ms": 900.0, "error_rate": 0.001,
        })), patch.object(agent, "_check_accuracy", new=AsyncMock(return_value={
            "alert_level": "none", "accuracy": 0.9, "baseline_accuracy": 0.89,
        })), patch.object(agent, "_persist_event", new=AsyncMock()):
            event = await agent._check_async("test-model")

        assert event["alert_level"] == "warning"
        assert any("latency" in a.lower() for a in event["alerts"])


class TestTriggerRetraining:
    async def test_inserts_pending_trigger(self, agent):
        pool, conn = _make_pool(fetchval_return=None)  # no existing pending trigger
        agent._pool = pool
        with patch.object(agent, "_resolve_training_dataset_uri", new=AsyncMock(return_value="s3://bucket/data.csv")):
            await agent._trigger_retraining("test-model", "3", "CRITICAL drift: score=0.4")
        conn.execute.assert_awaited_once()

    async def test_skips_when_pending_trigger_already_exists(self, agent):
        pool, conn = _make_pool()
        conn.fetchval = AsyncMock(return_value=1)  # existing pending trigger id
        agent._pool = pool
        with patch.object(agent, "_resolve_training_dataset_uri", new=AsyncMock(return_value="s3://bucket/data.csv")):
            await agent._trigger_retraining("test-model", "3", "CRITICAL drift")
        conn.execute.assert_not_awaited()

    async def test_no_pool_does_not_raise(self, agent):
        agent._pool = None
        # Must not raise — a missing DB pool should degrade, not crash the tick.
        await agent._trigger_retraining("test-model", "3", "reason")


class TestCheckAccuracy:
    async def test_no_rows_reports_none_not_zero(self, agent):
        pool, conn = _make_pool()
        conn.fetchrow = AsyncMock(side_effect=[{"acc": None, "n": 0}, None])
        agent._pool = pool
        result = await agent._check_accuracy("test-model", "3")
        assert result["accuracy"] is None
        assert result["alert_level"] == "none"

    async def test_no_pool_reports_error(self, agent):
        agent._pool = None
        result = await agent._check_accuracy("test-model", "3")
        assert result["alert_level"] == "error"


class TestCheckDriftReportStorage:
    """
    Regression coverage for a real cost-control bug: _check_drift used to
    call _store_report() (an R2 upload of the full Evidently HTML report)
    unconditionally on every tick, including routine "all clear" checks.
    On a real production monitoring cadence that's an unbounded-cost bug —
    found when the project owner's free-tier R2 bucket filled up from a
    45s local demo loop. Now gated: only alert-worthy drift scores get the
    heavy HTML artifact stored; the numeric drift_score is always
    persisted to Postgres regardless (unaffected by this gate).
    """

    @staticmethod
    def _mock_report(drift_score: float):
        mock_report = MagicMock()
        mock_report.as_dict.return_value = {
            "metrics": [
                {"metric": "DatasetDriftMetric", "result": {"share_of_drifted_columns": drift_score}},
                {"metric": "DataDriftTable", "result": {"drift_by_columns": {}}},
            ]
        }
        return mock_report

    async def test_below_threshold_does_not_store_report(self, agent):
        import pandas as pd

        agent._resolve_training_dataset_uri = AsyncMock(return_value="postgresql://x/y?table=ref")
        agent._sample_recent_inference_inputs = AsyncMock(
            return_value=pd.DataFrame({"f0": range(40)})
        )
        agent._store_report = AsyncMock(return_value="reports/should-not-be-called.html")

        mock_connector = MagicMock()
        mock_connector.connect = AsyncMock()
        mock_connector.read = AsyncMock(return_value=MagicMock(dataframe=pd.DataFrame({"f0": range(40)})))

        with patch("agents.connectors.ConnectorFactory.from_uri", return_value=mock_connector), \
             patch("evidently.legacy.report.Report", return_value=self._mock_report(0.02)):
            result = await agent._check_drift("test-model", "3")

        assert result["drift_score"] == 0.02
        assert result["report_r2_path"] is None
        agent._store_report.assert_not_awaited()

    async def test_above_threshold_stores_report(self, agent):
        import pandas as pd

        agent._resolve_training_dataset_uri = AsyncMock(return_value="postgresql://x/y?table=ref")
        agent._sample_recent_inference_inputs = AsyncMock(
            return_value=pd.DataFrame({"f0": range(40)})
        )
        agent._store_report = AsyncMock(return_value="reports/evidently/serving/test-model/drift.html")

        mock_connector = MagicMock()
        mock_connector.connect = AsyncMock()
        mock_connector.read = AsyncMock(return_value=MagicMock(dataframe=pd.DataFrame({"f0": range(40)})))

        with patch("agents.connectors.ConnectorFactory.from_uri", return_value=mock_connector), \
             patch("evidently.legacy.report.Report", return_value=self._mock_report(0.5)):
            result = await agent._check_drift("test-model", "3")

        assert result["drift_score"] == 0.5
        assert result["report_r2_path"] == "reports/evidently/serving/test-model/drift.html"
        agent._store_report.assert_awaited_once()


class TestCheckServingMetricsNaN:
    """
    Regression test for a real bug found while verifying the Prometheus
    query fix against a live instance: histogram_quantile()/division over a
    window with no real traffic returns NaN, a valid Python float where
    `nan > threshold` is always False — an unguarded p99/error_rate would
    silently read as "healthy" instead of "unknown".
    """

    @staticmethod
    def _mock_urlopen_sequence(values: list):
        """values: list of floats/None/'NaN' to return for each successive query."""
        calls = {"i": 0}

        def _urlopen(url, timeout=5):
            i = calls["i"]
            calls["i"] += 1
            val = values[i]
            if val is None:
                result = []
            else:
                result = [{"metric": {}, "value": [0, str(val)]}]
            body = json.dumps({"status": "success", "data": {"result": result}}).encode()
            cm = MagicMock()
            cm.__enter__.return_value.read.return_value = body
            cm.__exit__.return_value = False
            return cm

        return _urlopen

    async def test_nan_p99_is_treated_as_unavailable_not_zero(self, agent):
        # error_query, total_query, p99_query — p99 comes back NaN
        mock_urlopen = self._mock_urlopen_sequence([0.0, 5.0, "NaN"])
        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = await agent._check_serving_metrics("test-model", "3")
        assert result["p99_ms"] is None

    async def test_nan_error_rate_is_treated_as_unavailable_not_zero(self, agent):
        mock_urlopen = self._mock_urlopen_sequence(["NaN", 5.0, 50.0])
        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = await agent._check_serving_metrics("test-model", "3")
        assert result["error_rate"] is None

    async def test_real_numbers_pass_through_normally(self, agent):
        mock_urlopen = self._mock_urlopen_sequence([1.0, 5.0, 42.0])
        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = await agent._check_serving_metrics("test-model", "3")
        assert result["error_rate"] == pytest.approx(0.2)
        assert result["p99_ms"] == 42.0
