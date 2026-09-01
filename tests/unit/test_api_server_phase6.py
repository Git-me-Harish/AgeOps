# tests/unit/test_api_server_phase6.py
"""
Unit tests for the remaining Phase 6 additions to mcp_servers/api_server.py:
rollback, agent activity, governance audit log, and the real
mlflow.search_runs()/search_traces() backed experiment endpoints that
replace the V1 UI's 100%-mocked ExperimentsPage.

Kubernetes/DB/mlflow are mocked at their client boundaries — the request
handling and response-shaping logic in api_server.py runs for real.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
from fastapi.testclient import TestClient

import mcp_servers.api_server as api_server

client = TestClient(api_server.app)


class TestRollback:
    def test_missing_reason_is_rejected(self):
        resp = client.post("/api/deployments/svc-a/rollback", json={"reason": "  "})
        assert resp.status_code == 422

    def test_valid_reason_calls_manual_rollback(self):
        with patch.object(api_server, "_deployment_agent") as mock_agent:
            mock_agent.manual_rollback = MagicMock()
            resp = client.post(
                "/api/deployments/svc-a/rollback",
                json={"reason": "elevated error rate", "model_name": "fraud-model", "model_version": "3"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "rolled_back"
        mock_agent.manual_rollback.assert_called_once_with("svc-a", "elevated error rate", "fraud-model", "3")

    def test_agent_failure_surfaces_502(self):
        with patch.object(api_server, "_deployment_agent") as mock_agent:
            mock_agent.manual_rollback = MagicMock(side_effect=RuntimeError("k8s unreachable"))
            resp = client.post("/api/deployments/svc-a/rollback", json={"reason": "test"})
        assert resp.status_code == 502


class TestAgentActivity:
    def test_no_pool_returns_empty_not_fabricated(self):
        with patch.object(api_server, "_orchestrator") as mock_orch:
            mock_orch._ensure_connected = AsyncMock()
            mock_orch._pool = None
            resp = client.get("/api/agents/planner/activity")
        assert resp.status_code == 200
        body = resp.json()
        assert body["recent_calls"] == []
        assert body["summary"] is None

    def test_real_rows_shape_response(self):
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[
            {
                "workflow_id": "wf-1", "model": "gpt-4o-mini", "was_fallback": False, "was_cache_hit": True,
                "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost_usd": 0.001,
                "latency_ms": 220, "created_at": None,
            },
        ])
        conn.fetchrow = AsyncMock(side_effect=[
            {"call_count": 1, "avg_latency_ms": 220.0, "total_tokens": 150, "total_cost_usd": 0.001, "cache_hit_rate": 1.0},
            {"ok_count": 1, "total_count": 1},
        ])
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(api_server, "_orchestrator") as mock_orch:
            mock_orch._ensure_connected = AsyncMock()
            mock_orch._pool = pool
            resp = client.get("/api/agents/planner/activity")

        body = resp.json()
        assert body["summary"]["call_count"] == 1
        assert body["summary"]["success_rate"] == 1.0
        assert body["recent_calls"][0]["model"] == "gpt-4o-mini"


class TestGovernanceAuditLog:
    def test_no_pool_returns_empty(self):
        with patch.object(api_server, "_orchestrator") as mock_orch:
            mock_orch._ensure_connected = AsyncMock()
            mock_orch._pool = None
            resp = client.get("/api/governance/audit-log")
        assert resp.status_code == 200
        assert resp.json()["decisions"] == []

    def test_real_rows_shape_response(self):
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[
            {
                "workflow_id": "wf-1", "policy_name": "promote_to_production", "policy_package": "mlops.governance",
                "rego_policy_version": "1.0", "decision": False, "deny_reasons": ["f1 below threshold"],
                "agent_role": "governance", "was_sidecar": True, "evaluation_time_ms": 0.8, "created_at": None,
            },
        ])
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch.object(api_server, "_orchestrator") as mock_orch:
            mock_orch._ensure_connected = AsyncMock()
            mock_orch._pool = pool
            resp = client.get("/api/governance/audit-log")

        decisions = resp.json()["decisions"]
        assert decisions[0]["decision"] is False
        assert decisions[0]["deny_reasons"] == ["f1 below threshold"]


class TestListWorkflowsEndpoint:
    def test_passes_limit_and_offset_through_to_the_orchestrator(self):
        with patch.object(api_server, "_orchestrator") as mock_orch:
            mock_orch.list_workflows = AsyncMock(return_value={
                "workflows": [], "total_count": 0, "running_count": 0, "awaiting_approval_count": 0,
            })
            resp = client.get("/api/workflows?limit=20&offset=40")
        assert resp.status_code == 200
        mock_orch.list_workflows.assert_awaited_once_with(limit=20, offset=40)

    def test_default_page_size_is_20_not_50(self):
        """
        Regression guard: the endpoint used to default to a 50-row single
        fetch with no offset at all — the UI now expects 20-per-page with
        real Previous/Next pagination.
        """
        with patch.object(api_server, "_orchestrator") as mock_orch:
            mock_orch.list_workflows = AsyncMock(return_value={
                "workflows": [], "total_count": 0, "running_count": 0, "awaiting_approval_count": 0,
            })
            client.get("/api/workflows")
        mock_orch.list_workflows.assert_awaited_once_with(limit=20, offset=0)


class TestExperimentRuns:
    def test_empty_dataframe_returns_empty_list(self):
        with patch.object(api_server.mlflow, "search_runs", return_value=pd.DataFrame()):
            resp = client.get("/api/experiments/runs")
        assert resp.status_code == 200
        assert resp.json()["runs"] == []

    def test_real_dataframe_shapes_response(self):
        df = pd.DataFrame([
            {
                "run_id": "abc123", "tags.mlflow.runName": "demo-run", "status": "FINISHED",
                "start_time": pd.Timestamp("2026-01-01T00:00:00Z"), "end_time": pd.Timestamp("2026-01-01T00:01:00Z"),
                "metrics.accuracy": 0.91, "params.lr": "0.01",
            },
        ])
        with patch.object(api_server.mlflow, "search_runs", return_value=df):
            resp = client.get("/api/experiments/runs")
        runs = resp.json()["runs"]
        assert runs[0]["run_name"] == "demo-run"
        assert runs[0]["metrics"]["accuracy"] == 0.91
        assert runs[0]["params"]["lr"] == "0.01"
        assert resp.json()["total_count"] == 1

    def test_pagination_slices_a_real_dataframe_and_reports_true_total(self):
        """
        Regression coverage for the runs-table pagination the UI added —
        total_count must reflect the whole fetched window, not just the
        page returned, or 'Next' would never enable past page 1.
        """
        df = pd.DataFrame([
            {
                "run_id": f"run-{i}", "tags.mlflow.runName": f"run-{i}", "status": "FINISHED",
                "start_time": pd.Timestamp("2026-01-01T00:00:00Z"), "end_time": pd.Timestamp("2026-01-01T00:01:00Z"),
            }
            for i in range(45)
        ])
        with patch.object(api_server.mlflow, "search_runs", return_value=df):
            page1 = client.get("/api/experiments/runs?limit=20&offset=0").json()
            page2 = client.get("/api/experiments/runs?limit=20&offset=20").json()
            page3 = client.get("/api/experiments/runs?limit=20&offset=40").json()

        assert page1["total_count"] == page2["total_count"] == page3["total_count"] == 45
        assert len(page1["runs"]) == 20
        assert len(page2["runs"]) == 20
        assert len(page3["runs"]) == 5  # last page, partial
        assert page1["runs"][0]["run_id"] == "run-0"
        assert page2["runs"][0]["run_id"] == "run-20"


class TestRunTrace:
    def test_no_experiment_returns_empty(self):
        with patch.object(api_server.mlflow, "get_experiment_by_name", return_value=None), \
             patch.object(api_server.mlflow, "search_traces", return_value=pd.DataFrame()):
            resp = client.get("/api/experiments/runs/abc123/trace")
        assert resp.status_code == 200
        assert resp.json()["spans"] == []

    def test_real_span_dicts_are_parsed(self):
        # Exact shape verified against a real mlflow.search_traces() call —
        # a plain dict per span, not a Span object (see api_server.py's comment).
        df = pd.DataFrame([{
            "spans": [
                {"name": "outer", "context": {"span_id": "0xAAA"}, "parent_id": None,
                 "start_time": 1, "end_time": 2, "status_code": "OK"},
                {"name": "inner", "context": {"span_id": "0xBBB"}, "parent_id": "0xAAA",
                 "start_time": 1, "end_time": 2, "status_code": "OK"},
            ],
        }])
        mock_exp = MagicMock(experiment_id="1")
        with patch.object(api_server.mlflow, "get_experiment_by_name", return_value=mock_exp), \
             patch.object(api_server.mlflow, "search_traces", return_value=df):
            resp = client.get("/api/experiments/runs/abc123/trace")
        spans = resp.json()["spans"]
        assert len(spans) == 2
        assert spans[0]["span_id"] == "0xAAA"
        assert spans[1]["parent_id"] == "0xAAA"
