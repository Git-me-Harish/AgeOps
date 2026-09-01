# tests/unit/test_orchestrator_routing.py
"""
Unit tests for OrchestratorAgent's pure routing functions (Phase 5 wiring).

These don't need a live graph/checkpointer — each _route_after_* function is
a pure function of state. The pause/resume mechanism itself (interrupt_before
+ aget_state + aupdate_state/ainvoke(None, ...)) was verified separately
against a real Postgres checkpointer while building this — see the fix pass
notes; it isn't practical to exercise langgraph's AsyncPostgresSaver inside
this repo's sqlite-backed unit test harness, so these tests cover the
decision logic that sits around it instead.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.orchestrator import OrchestratorAgent


@pytest.fixture
def orch() -> OrchestratorAgent:
    # No DB conn string — constructor skips real I/O, so this is safe to
    # instantiate directly for pure routing-function tests.
    return OrchestratorAgent(db_conn_string="")


class TestRouteAfterPlan:
    def test_ok_when_no_errors_and_not_pending(self, orch):
        assert orch._route_after_plan({"current_stage": "data", "errors": []}) == "ok"

    def test_needs_approval_when_plan_pending(self, orch):
        state = {"current_stage": "plan_pending_approval", "errors": []}
        assert orch._route_after_plan(state) == "needs_approval"

    def test_error_takes_priority_over_pending(self, orch):
        state = {"current_stage": "plan_pending_approval", "errors": ["boom"]}
        assert orch._route_after_plan(state) == "error"


class TestRouteAfterPlanApproval:
    def test_approved(self, orch):
        assert orch._route_after_plan_approval({"plan_approval_decision": "approved"}) == "approved"

    def test_rejected(self, orch):
        assert orch._route_after_plan_approval({"plan_approval_decision": "rejected"}) == "rejected"

    def test_missing_decision_defaults_to_rejected(self, orch):
        # Fail-closed: an unresolved/garbled decision must never be treated
        # as an implicit approval.
        assert orch._route_after_plan_approval({}) == "rejected"


class TestRouteAfterApproval:
    def test_approved(self, orch):
        assert orch._route_after_approval({"approval_decision": "approved"}) == "approved"

    def test_rejected(self, orch):
        assert orch._route_after_approval({"approval_decision": "rejected"}) == "rejected"

    def test_missing_decision_defaults_to_rejected(self, orch):
        assert orch._route_after_approval({}) == "rejected"


class TestOtherRouting:
    def test_route_after_registration_ok(self, orch):
        assert orch._route_after_registration({"errors": []}) == "ok"

    def test_route_after_registration_error(self, orch):
        assert orch._route_after_registration({"errors": ["missing dataset_hash"]}) == "error"

    def test_route_after_packaging_ok(self, orch):
        assert orch._route_after_packaging({"errors": []}) == "ok"

    def test_route_after_promotion_error(self, orch):
        assert orch._route_after_promotion({"errors": ["gate=oci reason=no image"]}) == "error"


class TestPlanNodeStatusHandling:
    """_plan_node's own branching on PlannerAgent's AgentTaskResult.status."""

    def test_pending_approval_status_sets_stage(self, orch):
        mock_result = MagicMock()
        mock_result.status = "pending_approval"
        mock_result.output = {"plan_db_id": 42, "plan_summary": {"frameworks": ["xgboost"]}}
        mock_result.model_dump.return_value = {"status": "pending_approval"}
        orch._planner_agent.run = MagicMock(return_value=mock_result)

        state = orch._plan_node({"workflow_id": "wf-1", "dataset_uri": "s3://x", "agent_decisions": []})

        assert state["current_stage"] == "plan_pending_approval"
        assert state["plan_db_id"] == 42
        assert orch._route_after_plan(state) == "needs_approval"

    def test_failed_status_records_error(self, orch):
        mock_result = MagicMock()
        mock_result.status = "failed"
        mock_result.error = "LLM budget exceeded"
        mock_result.output = None
        mock_result.model_dump.return_value = {"status": "failed"}
        orch._planner_agent.run = MagicMock(return_value=mock_result)

        state = orch._plan_node({"workflow_id": "wf-1", "dataset_uri": "s3://x", "agent_decisions": [], "errors": []})

        assert state["errors"] == ["LLM budget exceeded"]
        assert orch._route_after_plan(state) == "error"

    def test_success_status_carries_execution_plan_and_rl_recs(self, orch):
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.output = {
            "execution_plan": {"frameworks": ["xgboost"], "parallel_experiments": 1},
            "rl_recommendations": {"hyperparameter_adjustments": {"learning_rate": 0.05}},
        }
        mock_result.model_dump.return_value = {"status": "success"}
        orch._planner_agent.run = MagicMock(return_value=mock_result)

        state = orch._plan_node({"workflow_id": "wf-1", "dataset_uri": "s3://x", "agent_decisions": []})

        assert state["current_stage"] == "data"
        assert state["execution_plan"]["frameworks"] == ["xgboost"]
        assert state["rl_recommendations"]["hyperparameter_adjustments"]["learning_rate"] == 0.05
        assert orch._route_after_plan(state) == "ok"


def _make_pool(*, total_count: int, agg_row: dict, rows: list[dict]):
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=total_count)
    conn.fetchrow = AsyncMock(return_value=agg_row)
    conn.fetch = AsyncMock(return_value=rows)
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool, conn


class TestListWorkflows:
    """
    Real pagination + real global aggregates (Command Center table now
    paginates instead of dumping every workflow into one scroll) — the
    Postgres client is mocked at the asyncpg boundary, same convention as
    tests/unit/test_monitoring_agent.py.
    """

    async def test_no_pool_degrades_to_empty_result(self, orch):
        orch._connected = True  # skip _ensure_connected's real connection attempt
        orch._pool = None
        result = await orch.list_workflows(limit=20, offset=0)
        assert result == {"workflows": [], "total_count": 0, "running_count": 0, "awaiting_approval_count": 0}

    async def test_passes_limit_and_offset_to_the_query_and_reports_real_aggregates(self, orch):
        pool, conn = _make_pool(
            total_count=45,
            agg_row={"running_count": 3, "awaiting_approval_count": 1},
            rows=[{
                "id": "wf-1", "status": "running", "current_stage": "training", "dataset_uri": "s3://x",
                "model_uri": None, "metrics": "{}", "errors": "[]", "awaiting_approval": False,
                "source": "manual", "trigger_type": None, "created_at": None, "updated_at": None,
            }],
        )
        orch._connected = True
        orch._pool = pool

        result = await orch.list_workflows(limit=20, offset=40)

        # LIMIT/OFFSET actually reached the query, not silently dropped.
        _, call_args = conn.fetch.call_args
        assert conn.fetch.call_args[0][-2:] == (20, 40)
        assert result["total_count"] == 45
        # Aggregates come from the real COUNT(*) FILTER query, not derived
        # from the single (paginated) row returned above.
        assert result["running_count"] == 3
        assert result["awaiting_approval_count"] == 1
        assert len(result["workflows"]) == 1
