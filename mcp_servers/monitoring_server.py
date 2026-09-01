"""
mcp_servers/monitoring_server.py

Phase 5 — Monitoring & RL Recommendations MCP Server.

Exposes Phase 5 observability and closed-loop-retraining operations.
Follows the exact same FastAPI + rate-limiting + tool-discovery pattern as
mlflow_server.py / registry_server.py / packaging_server.py.

Registered tools (discoverable at GET /tools):
  get_monitoring_events   — health-check history for a model (monitoring_events)
  get_drift_reports       — serving-time drift check history (serving_drift_reports)
  trigger_retraining      — manually enqueue a retraining workflow_trigger
  list_pending_triggers   — workflow_triggers awaiting processing
  list_recommendations    — RL Optimizer suggestions (rl_recommendations)
  decide_recommendation   — accept/reject an RL recommendation (Phase 5 scope
                             is the API only — a UI panel to drive this is
                             Phase 6 per the plan's own sequencing)

Design:
  - No agent logic lives here — this server only reads/writes the Phase 5
    tables (migration 0006_monitoring.py) and enqueues triggers. The actual
    monitoring loop is agents/monitoring_agent.py (run via scripts/
    monitor_loop.py); the actual trigger consumer is
    OrchestratorAgent.process_pending_triggers() (run via scripts/
    process_triggers.py).
  - DB pool is created lazily (lifespan pattern), same as the other servers.
  - Rate limit: 100 requests/minute per X-Agent-Id header.

Run alongside the other MCP servers:
  uvicorn mcp_servers.monitoring_server:app --host 0.0.0.0 --port 8004
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from configs.settings import settings

logger = logging.getLogger(__name__)

# DB pool (module-level, initialised lazily)
_pool: Optional[asyncpg.Pool] = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url.replace("+asyncpg", ""),
            min_size=2,
            max_size=settings.database_pool_size,
            command_timeout=30,
        )
    return _pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    try:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url.replace("+asyncpg", ""),
            min_size=2,
            max_size=settings.database_pool_size,
            command_timeout=30,
        )
        logger.info("Monitoring MCP server: DB pool initialised")
    except Exception as exc:
        logger.warning("Monitoring MCP server: DB pool init failed (%s) — will retry per-request", exc)
        _pool = None
    yield
    if _pool:
        await _pool.close()
        logger.info("Monitoring MCP server: DB pool closed")


app = FastAPI(
    title="Monitoring MCP Server",
    description="Model Context Protocol tools for Phase 5 observability and closed-loop retraining",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter (same pattern as the other MCP servers)
_request_counts: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT = 100
_RATE_WINDOW = 60


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    agent_id = request.headers.get("X-Agent-Id", "anonymous")
    now = time.time()
    window_start = now - _RATE_WINDOW
    calls = [t for t in _request_counts[agent_id] if t > window_start]
    if len(calls) >= _RATE_LIMIT:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    _request_counts[agent_id] = calls + [now]
    return await call_next(request)


@app.get("/tools")
async def list_tools():
    return {
        "tools": [
            {"name": "get_monitoring_events", "endpoint": "/monitoring/v1/events/{model_name}", "version": "1",
             "description": "Health-check history for a model — drift/accuracy/latency per tick."},
            {"name": "get_drift_reports", "endpoint": "/monitoring/v1/drift_reports/{model_name}", "version": "1",
             "description": "Serving-time drift check history, including R2 HTML report links."},
            {"name": "trigger_retraining", "endpoint": "/monitoring/v1/trigger_retraining", "version": "1",
             "description": "Manually enqueue a retraining workflow_trigger for OrchestratorAgent.process_pending_triggers() to pick up."},
            {"name": "list_pending_triggers", "endpoint": "/monitoring/v1/triggers", "version": "1",
             "description": "List workflow_triggers by status."},
            {"name": "list_recommendations", "endpoint": "/monitoring/v1/recommendations", "version": "1",
             "description": "List RL Optimizer suggestions, optionally filtered by status."},
            {"name": "decide_recommendation", "endpoint": "/monitoring/v1/recommendations/{id}/decide", "version": "1",
             "description": "Accept or reject a pending RL recommendation."},
        ]
    }


# Request / response schemas
class MonitoringEventOut(BaseModel):
    id: int
    model_name: str
    model_version: Optional[str]
    drift_score: Optional[float]
    accuracy: Optional[float]
    baseline_accuracy: Optional[float]
    p99_latency_ms: Optional[float]
    error_rate: Optional[float]
    alert_level: str
    alerts: list[str]
    created_at: str


class DriftReportOut(BaseModel):
    id: int
    model_name: str
    model_version: Optional[str]
    sample_count: int
    drift_score: Optional[float]
    alert_level: str
    check_error: Optional[str]
    r2_report_path: Optional[str]
    created_at: str


class TriggerRetrainingRequest(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    dataset_uri: Optional[str] = None
    reason: str = Field(default="manual trigger via API")


class TriggerOut(BaseModel):
    id: int
    trigger_type: str
    source: str
    model_name: str
    model_version: Optional[str]
    status: str
    reason: str
    launched_workflow_id: Optional[str]
    created_at: str


class RecommendationOut(BaseModel):
    id: int
    workflow_id: Optional[str]
    agent_role: str
    recommendation: dict[str, Any]
    confidence: Optional[float]
    status: str
    reviewed_by: Optional[str]
    created_at: str


class DecideRecommendationRequest(BaseModel):
    decision: str = Field(..., pattern="^(accepted|rejected)$")
    reviewed_by: str = Field(..., min_length=1)


# Endpoints
@app.get("/monitoring/v1/events/{model_name}", response_model=list[MonitoringEventOut])
async def get_monitoring_events(model_name: str, limit: int = 50) -> list[MonitoringEventOut]:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, model_name, model_version, drift_score, accuracy,
                   baseline_accuracy, p99_latency_ms, error_rate, alert_level,
                   alerts, created_at
            FROM monitoring_events WHERE model_name=$1
            ORDER BY created_at DESC LIMIT $2
            """,
            model_name, limit,
        )
    return [
        MonitoringEventOut(
            id=r["id"], model_name=r["model_name"], model_version=r["model_version"],
            drift_score=r["drift_score"], accuracy=r["accuracy"],
            baseline_accuracy=r["baseline_accuracy"], p99_latency_ms=r["p99_latency_ms"],
            error_rate=r["error_rate"], alert_level=r["alert_level"],
            alerts=list(json.loads(r["alerts"]) if isinstance(r["alerts"], str) else r["alerts"] or []),
            created_at=r["created_at"].isoformat() if r["created_at"] else "",
        )
        for r in rows
    ]


@app.get("/monitoring/v1/drift_reports/{model_name}", response_model=list[DriftReportOut])
async def get_drift_reports(model_name: str, limit: int = 50) -> list[DriftReportOut]:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, model_name, model_version, sample_count, drift_score,
                   alert_level, check_error, r2_report_path, created_at
            FROM serving_drift_reports WHERE model_name=$1
            ORDER BY created_at DESC LIMIT $2
            """,
            model_name, limit,
        )
    return [
        DriftReportOut(
            id=r["id"], model_name=r["model_name"], model_version=r["model_version"],
            sample_count=r["sample_count"], drift_score=r["drift_score"],
            alert_level=r["alert_level"], check_error=r["check_error"],
            r2_report_path=r["r2_report_path"],
            created_at=r["created_at"].isoformat() if r["created_at"] else "",
        )
        for r in rows
    ]


@app.post("/monitoring/v1/trigger_retraining", response_model=TriggerOut)
async def trigger_retraining(request: TriggerRetrainingRequest) -> TriggerOut:
    """
    Manually enqueue a workflow_trigger — same table MonitoringAgent writes
    to on a CRITICAL breach, so a manually-triggered retrain goes through
    the identical OrchestratorAgent.process_pending_triggers() path.
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO workflow_triggers (trigger_type, source, model_name, model_version, dataset_uri, reason)
            VALUES ('manual', 'user', $1, $2, $3, $4)
            RETURNING id, trigger_type, source, model_name, model_version, status, reason, launched_workflow_id, created_at
            """,
            request.model_name, request.model_version, request.dataset_uri, request.reason,
        )
    return TriggerOut(
        id=row["id"], trigger_type=row["trigger_type"], source=row["source"],
        model_name=row["model_name"], model_version=row["model_version"],
        status=row["status"], reason=row["reason"],
        launched_workflow_id=row["launched_workflow_id"],
        created_at=row["created_at"].isoformat() if row["created_at"] else "",
    )


@app.get("/monitoring/v1/triggers", response_model=list[TriggerOut])
async def list_pending_triggers(status: Optional[str] = None, limit: int = 50) -> list[TriggerOut]:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, trigger_type, source, model_name, model_version, status,
                   reason, launched_workflow_id, created_at
            FROM workflow_triggers
            WHERE ($1::text IS NULL OR status=$1)
            ORDER BY created_at DESC LIMIT $2
            """,
            status, limit,
        )
    return [
        TriggerOut(
            id=r["id"], trigger_type=r["trigger_type"], source=r["source"],
            model_name=r["model_name"], model_version=r["model_version"],
            status=r["status"], reason=r["reason"],
            launched_workflow_id=r["launched_workflow_id"],
            created_at=r["created_at"].isoformat() if r["created_at"] else "",
        )
        for r in rows
    ]


@app.get("/monitoring/v1/recommendations", response_model=list[RecommendationOut])
async def list_recommendations(status: Optional[str] = None, limit: int = 50) -> list[RecommendationOut]:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, workflow_id, agent_role, recommendation, confidence,
                   status, reviewed_by, created_at
            FROM rl_recommendations
            WHERE ($1::text IS NULL OR status=$1)
            ORDER BY created_at DESC LIMIT $2
            """,
            status, limit,
        )
    return [
        RecommendationOut(
            id=r["id"], workflow_id=r["workflow_id"], agent_role=r["agent_role"],
            recommendation=json.loads(r["recommendation"]) if isinstance(r["recommendation"], str) else r["recommendation"],
            confidence=r["confidence"], status=r["status"], reviewed_by=r["reviewed_by"],
            created_at=r["created_at"].isoformat() if r["created_at"] else "",
        )
        for r in rows
    ]


@app.post("/monitoring/v1/recommendations/{rec_id}/decide", response_model=RecommendationOut)
async def decide_recommendation(rec_id: int, request: DecideRecommendationRequest) -> RecommendationOut:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE rl_recommendations
            SET status=$2, reviewed_by=$3, reviewed_at=NOW()
            WHERE id=$1 AND status='pending'
            RETURNING id, workflow_id, agent_role, recommendation, confidence, status, reviewed_by, created_at
            """,
            rec_id, request.decision, request.reviewed_by,
        )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Recommendation {rec_id} not found or already decided")
    return RecommendationOut(
        id=row["id"], workflow_id=row["workflow_id"], agent_role=row["agent_role"],
        recommendation=json.loads(row["recommendation"]) if isinstance(row["recommendation"], str) else row["recommendation"],
        confidence=row["confidence"], status=row["status"], reviewed_by=row["reviewed_by"],
        created_at=row["created_at"].isoformat() if row["created_at"] else "",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
