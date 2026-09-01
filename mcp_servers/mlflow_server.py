"""
MLflow MCP Server
─────────────────
FastAPI server that exposes MLflow operations as MCP-compatible tools.
Agents discover available tools at runtime via GET /tools.

Versioned endpoints: /mlflow/v1/<tool>
All inputs/outputs are Pydantic-validated.
Rate limit: 100 req/min per agent (enforced via middleware).
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any, Optional

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from configs.settings import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLflow MCP Server",
    description="Model Context Protocol tools for MLflow operations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

# ── Simple in-memory rate limiter ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Tool discovery endpoint (MCP specification)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/tools")
async def list_tools():
    """Dynamic tool discovery — agents call this at startup."""
    return {
        "tools": [
            {"name": "get_model", "endpoint": "/mlflow/v1/get_model", "version": "1"},
            {"name": "log_metrics", "endpoint": "/mlflow/v1/log_metrics", "version": "1"},
            {"name": "search_runs", "endpoint": "/mlflow/v1/search_runs", "version": "1"},
            {"name": "register_model", "endpoint": "/mlflow/v1/register_model", "version": "1"},
            {"name": "transition_stage", "endpoint": "/mlflow/v1/transition_stage", "version": "1"},
            {"name": "log_agent_decision", "endpoint": "/mlflow/v1/log_agent_decision", "version": "1"},
            {"name": "get_experiment_metrics", "endpoint": "/mlflow/v1/get_experiment_metrics", "version": "1"},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class GetModelRequest(BaseModel):
    model_uri: str = Field(..., description="MLflow model URI, e.g. runs:/<run_id>/model")

class GetModelResponse(BaseModel):
    model_uri: str
    run_id: str
    metrics: dict[str, float]
    params: dict[str, str]
    tags: dict[str, str]


class LogMetricsRequest(BaseModel):
    run_id: str
    metrics: dict[str, float] = Field(..., description="Metric name → value")
    step: Optional[int] = None

class LogMetricsResponse(BaseModel):
    success: bool
    run_id: str


class SearchRunsRequest(BaseModel):
    experiment_name: str = settings.mlflow_experiment_name
    filter_string: str = ""
    max_results: int = Field(default=20, ge=1, le=100)
    order_by: list[str] = ["start_time DESC"]

class SearchRunsResponse(BaseModel):
    runs: list[dict[str, Any]]
    count: int


class RegisterModelRequest(BaseModel):
    model_uri: str
    model_name: str = "mlops-model"
    tags: Optional[dict[str, str]] = None

class RegisterModelResponse(BaseModel):
    name: str
    version: str
    status: str


class TransitionStageRequest(BaseModel):
    model_name: str
    version: str
    stage: str = Field(..., pattern="^(Staging|Production|Archived|None)$")
    archive_existing: bool = True

class TransitionStageResponse(BaseModel):
    model_name: str
    version: str
    new_stage: str


class AgentDecisionRequest(BaseModel):
    decision: dict[str, Any] = Field(..., description="Structured agent decision for audit trail")
    workflow_id: Optional[str] = None

class AgentDecisionResponse(BaseModel):
    logged: bool


class ExperimentMetricsRequest(BaseModel):
    experiment_name: str = settings.mlflow_experiment_name
    metric_keys: list[str] = ["accuracy", "f1_score", "drift_score"]

class ExperimentMetricsResponse(BaseModel):
    metrics_summary: dict[str, dict[str, float]]   # metric → {min, max, mean, latest}


# ─────────────────────────────────────────────────────────────────────────────
# Tool endpoints (v1)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/mlflow/v1/get_model", response_model=GetModelResponse)
async def get_model(request: GetModelRequest) -> GetModelResponse:
    """Retrieve model metadata from MLflow."""
    try:
        info = mlflow.models.get_model_info(request.model_uri)
        run = mlflow.get_run(info.run_id)
        return GetModelResponse(
            model_uri=request.model_uri,
            run_id=info.run_id,
            metrics=run.data.metrics,
            params=run.data.params,
            tags=run.data.tags,
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Model not found: {exc}")


@app.post("/mlflow/v1/log_metrics", response_model=LogMetricsResponse)
async def log_metrics(request: LogMetricsRequest) -> LogMetricsResponse:
    """Log metrics to an existing MLflow run."""
    try:
        client = mlflow.tracking.MlflowClient()
        for key, value in request.metrics.items():
            client.log_metric(request.run_id, key, value, step=request.step or 0)
        return LogMetricsResponse(success=True, run_id=request.run_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/mlflow/v1/search_runs", response_model=SearchRunsResponse)
async def search_runs(request: SearchRunsRequest) -> SearchRunsResponse:
    """Search MLflow runs with optional filter."""
    try:
        df = mlflow.search_runs(
            experiment_names=[request.experiment_name],
            filter_string=request.filter_string,
            max_results=request.max_results,
            order_by=request.order_by,
        )
        runs = df.to_dict("records") if not df.empty else []
        return SearchRunsResponse(runs=runs, count=len(runs))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/mlflow/v1/register_model", response_model=RegisterModelResponse)
async def register_model(request: RegisterModelRequest) -> RegisterModelResponse:
    """
    REMOVED (Phase 3): this endpoint used to call mlflow.register_model()
    directly, with no tag validation — a hard bypass of RegistrationGate's
    18-mandatory-tag check. Registration now goes exclusively through
    POST /registry/v1/register_model (mcp_servers/registry_server.py), which
    validates tags before ever touching the MLflow Registry. See
    agents/registry/promotion_state_machine.py's module docstring: "no direct
    mlflow.transition_model_version_stage() calls anywhere else in the codebase"
    — the same rule applies to register_model.
    """
    raise HTTPException(
        status_code=410,
        detail=(
            "This endpoint is retired. Use POST /registry/v1/register_model "
            "on the registry MCP server — it enforces the Phase 3 mandatory "
            "tag gate that this endpoint bypassed."
        ),
    )


@app.post("/mlflow/v1/transition_stage", response_model=TransitionStageResponse)
async def transition_stage(request: TransitionStageRequest) -> TransitionStageResponse:
    """
    REMOVED (Phase 3): this endpoint used to call
    client.transition_model_version_stage() directly, bypassing every
    PromotionStateMachine gate (evaluation, HITL, security, governance, OCI).
    Promotion now goes exclusively through POST /registry/v1/promote_model.
    """
    raise HTTPException(
        status_code=410,
        detail=(
            "This endpoint is retired. Use POST /registry/v1/promote_model "
            "on the registry MCP server — it enforces the promotion gates "
            "(evaluation, HITL, security, governance, OCI) that this endpoint bypassed."
        ),
    )


@app.post("/mlflow/v1/log_agent_decision", response_model=AgentDecisionResponse)
async def log_agent_decision_endpoint(request: AgentDecisionRequest) -> AgentDecisionResponse:
    """Append an agent decision to the MLflow run's audit log tag."""
    try:
        log_agent_decision(request.decision, run_id=None)
        return AgentDecisionResponse(logged=True)
    except Exception:
        return AgentDecisionResponse(logged=False)


@app.post("/mlflow/v1/get_experiment_metrics", response_model=ExperimentMetricsResponse)
async def get_experiment_metrics(request: ExperimentMetricsRequest) -> ExperimentMetricsResponse:
    """Aggregate metrics across all runs in an experiment."""
    try:
        df = mlflow.search_runs(
            experiment_names=[request.experiment_name],
            max_results=100,
        )
        summary: dict[str, dict[str, float]] = {}
        for key in request.metric_keys:
            col = f"metrics.{key}"
            if col in df.columns:
                vals = df[col].dropna()
                if not vals.empty:
                    summary[key] = {
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                        "mean": float(vals.mean()),
                        "latest": float(vals.iloc[-1]),
                    }
        return ExperimentMetricsResponse(metrics_summary=summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "mlflow_uri": settings.mlflow_tracking_uri}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function used by agents directly (non-HTTP path)
# ─────────────────────────────────────────────────────────────────────────────

def log_agent_decision(decision: dict, run_id: str | None = None) -> None:
    """Log an agent decision to the active MLflow run's tags."""
    try:
        active_run = mlflow.active_run()
        target_run_id = run_id or (active_run.info.run_id if active_run else None)
        if target_run_id:
            client = mlflow.tracking.MlflowClient()
            existing = client.get_run(target_run_id).data.tags.get("agent_decisions", "[]")
            decisions = json.loads(existing)
            decisions.append(decision)
            client.set_tag(target_run_id, "agent_decisions", json.dumps(decisions[-50:]))  # cap at 50
    except Exception as exc:
        logger.debug("Could not log agent decision to MLflow: %s", exc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
