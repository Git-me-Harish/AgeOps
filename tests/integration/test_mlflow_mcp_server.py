# tests/integration/test_mlflow_mcp_server.py
"""
Integration tests for the MLflow MCP Server FastAPI endpoints.
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock

from mcp_servers.mlflow_server import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Tool discovery ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tools_endpoint_returns_all_tools(client):
    resp = await client.get("/tools")
    assert resp.status_code == 200
    tools = resp.json()["tools"]
    names = {t["name"] for t in tools}
    expected = {"get_model", "log_metrics", "search_runs", "register_model",
                "transition_stage", "log_agent_decision", "get_experiment_metrics"}
    assert expected == names


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Log metrics ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_log_metrics(client):
    with patch("mcp_servers.mlflow_server.mlflow.tracking.MlflowClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        resp = await client.post("/mlflow/v1/log_metrics", json={
            "run_id": "fake-run-id",
            "metrics": {"accuracy": 0.91, "f1_score": 0.88},
        })
    assert resp.status_code == 200
    assert resp.json()["success"] is True


# ── Search runs ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_runs_returns_list(client):
    import pandas as pd
    mock_df = pd.DataFrame({
        "run_id": ["r1", "r2"],
        "metrics.accuracy": [0.85, 0.90],
    })
    with patch("mcp_servers.mlflow_server.mlflow.search_runs", return_value=mock_df):
        resp = await client.post("/mlflow/v1/search_runs", json={
            "experiment_name": "test",
            "max_results": 10,
        })
    assert resp.status_code == 200
    assert resp.json()["count"] == 2


# ── Register model ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_model(client):
    mock_result = MagicMock()
    mock_result.name = "mlops-model"
    mock_result.version = "5"
    mock_result.status = "READY"
    with patch("mcp_servers.mlflow_server.mlflow.register_model", return_value=mock_result):
        resp = await client.post("/mlflow/v1/register_model", json={
            "model_uri": "runs:/fake_run/model",
            "model_name": "mlops-model",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "mlops-model"
    assert data["version"] == "5"


# ── Rate limiting ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rate_limit_header_not_hit_under_limit(client):
    # First call should pass (well below 100/min)
    resp = await client.get("/health", headers={"X-Agent-Id": "test-agent"})
    assert resp.status_code == 200


# ── Transition stage ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transition_stage(client):
    with patch("mcp_servers.mlflow_server.mlflow.tracking.MlflowClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        resp = await client.post("/mlflow/v1/transition_stage", json={
            "model_name": "mlops-model",
            "version": "3",
            "stage": "Production",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["new_stage"] == "Production"
