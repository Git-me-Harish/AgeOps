# tests/integration/test_api_server.py
"""
Integration tests for the FastAPI REST API.
Requires a running MLflow server (provided as a pytest fixture
or via the CI service container).
"""
from __future__ import annotations

import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock, MagicMock

from mcp_servers.api_server import app
from agents import AgentTaskResult


# ── Async test client ─────────────────────────────────────────────────────────

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


# ── Start workflow ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_workflow_accepted(client):
    mock_state = {
        "workflow_id": "wf-test-001",
        "current_stage": "done",
        "status": "completed",
        "metrics": {"accuracy": 0.88},
        "errors": [],
        "awaiting_approval": False,
    }
    with patch("mcp_servers.api_server._orchestrator") as mock_orch:
        mock_orch.run_workflow = AsyncMock(return_value=mock_state)
        resp = await client.post("/api/workflows", json={
            "dataset_uri": "s3://test-bucket/data.csv"
        })
    assert resp.status_code == 202
    body = resp.json()
    assert "workflow_id" in body
    assert body["status"] == "running"


@pytest.mark.asyncio
async def test_start_workflow_invalid_body(client):
    resp = await client.post("/api/workflows", json={})
    # dataset_uri is required
    assert resp.status_code == 422


# ── Get workflow ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_workflow_not_found(client):
    resp = await client.get("/api/workflows/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_workflow_found(client):
    from mcp_servers.api_server import _workflows
    wf_id = "wf-integration-test"
    _workflows[wf_id] = {
        "status": "completed",
        "current_stage": "done",
        "metrics": {"accuracy": 0.90},
        "errors": [],
        "awaiting_approval": False,
    }
    resp = await client.get(f"/api/workflows/{wf_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["metrics"]["accuracy"] == 0.90


# ── Approve workflow ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_approve_workflow_not_awaiting(client):
    from mcp_servers.api_server import _workflows
    wf_id = "wf-not-waiting"
    _workflows[wf_id] = {
        "status": "running",
        "current_stage": "training",
        "metrics": {},
        "errors": [],
        "awaiting_approval": False,
    }
    resp = await client.post(f"/api/workflows/{wf_id}/approve", json={"approved": True, "reviewer": "harish"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_approve_workflow_grants_access(client):
    from mcp_servers.api_server import _workflows
    wf_id = "wf-awaiting"
    _workflows[wf_id] = {
        "status": "running",
        "current_stage": "human_approval",
        "metrics": {},
        "errors": [],
        "awaiting_approval": True,
    }
    resumed_state = {
        "current_stage": "deployment",
        "awaiting_approval": False,
        "metrics": {},
        "errors": [],
        "status": "running",
    }
    with patch("mcp_servers.api_server._orchestrator") as mock_orch:
        mock_orch.resume_workflow = AsyncMock(return_value=resumed_state)
        resp = await client.post(f"/api/workflows/{wf_id}/approve", json={"approved": True, "reviewer": "harish"})
    assert resp.status_code == 200
    assert resp.json()["awaiting_approval"] is False


# ── Agents ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_agents(client):
    resp = await client.get("/api/agents")
    assert resp.status_code == 200
    data = resp.json()
    assert "agents" in data
    # Should have agents from the JSON card file
    assert len(data["agents"]) > 0


# ── Models ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_models(client):
    mock_rm = MagicMock()
    mock_rm.name = "mlops-model"
    mock_rm.description = "test"
    mock_rm.tags = {}
    mock_v = MagicMock()
    mock_v.version = "1"
    mock_v.current_stage = "Production"
    mock_v.run_id = "abc123"
    mock_rm.latest_versions = [mock_v]

    with patch("mcp_servers.api_server.mlflow.tracking.MlflowClient") as mock_client_cls:
        mock_client_cls.return_value.search_registered_models.return_value = [mock_rm]
        resp = await client.get("/api/models")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 1
    assert data["models"][0]["name"] == "mlops-model"


# ── Metrics summary ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_metrics_summary(client):
    import pandas as pd
    mock_df = pd.DataFrame({
        "metrics.accuracy":         [0.85, 0.87, 0.90],
        "metrics.f1_score":         [0.82, 0.84, 0.88],
        "metrics.drift_score":      [0.05, 0.07, 0.03],
        "metrics.validation_score": [0.92, 0.94, 0.95],
    })
    with patch("mcp_servers.api_server.mlflow.search_runs", return_value=mock_df):
        resp = await client.get("/api/metrics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert data["summary"]["accuracy"]["mean"] == pytest.approx(0.8733, rel=1e-2)
