"""
mcp_servers/registry_server.py

Phase 3 — Model Registry MCP Server.

Exposes Phase 3 registry operations as MCP-compatible tools to all agents.
Follows the exact same FastAPI + rate-limiting + tool-discovery pattern as
mlflow_server.py — agents discover tools via GET /tools and call them via POST.

Registered tools (discoverable at GET /tools):
  validate_tags          — validate a tag dict without registering (dry-run)
  register_model         — RegistrationGate: validate + register + tag + persist
  promote_model          — PromotionStateMachine: enforce state transition + gates
  get_lineage            — full DAG for a model version (UI lineage graph)
  compare_models         — ModelComparator: side-by-side metric comparison
  get_promotion_history  — audit trail of all state transitions for a model

Design:
  - No direct MLflow calls here — all mutations go through RegistrationGate and
    PromotionStateMachine so they get the full gate checking + audit trail.
  - DB pool is created lazily on first request (lifespan pattern — avoids connecting
    at import time, which breaks tests and CLI usage).
  - Rate limit: 100 requests/minute per X-Agent-Id header (same as mlflow_server).
  - All error responses follow {detail: str} shape matching FastAPI defaults.

Run alongside mlflow_server:
  uvicorn mcp_servers.registry_server:app --host 0.0.0.0 --port 8002
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Optional

import asyncpg
import mlflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agents.registry import (
    ModelComparator,
    PromotionRequest,
    PromotionStateMachine,
    RegistrationGate,
    LineageGraphQuerier,
    ModelStage,
    TagValidator,
)
from agents.registry.metadata_schema import RegistrationError
from agents.registry.promotion_state_machine import PromotionGateError
from configs.settings import settings

logger = logging.getLogger(__name__)

# DB pool (module-level, initialised on first request) 
_pool: Optional[asyncpg.Pool] = None
async def _get_pool() -> asyncpg.Pool:
    """Lazy-init asyncpg pool. Called on every request that needs the DB."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url.replace("+asyncpg", ""),
            min_size=2,
            max_size=settings.database_pool_size,
            command_timeout=30,
        )
    return _pool


# Application lifespan 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the DB pool on startup; close it on shutdown."""
    global _pool
    try:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url.replace("+asyncpg", ""),
            min_size=2,
            max_size=settings.database_pool_size,
            command_timeout=30,
        )
        logger.info("Registry MCP server: DB pool initialised")
    except Exception as exc:
        logger.warning("Registry MCP server: DB pool init failed (%s) — will retry per-request", exc)
        _pool = None
    yield
    if _pool:
        await _pool.close()
        logger.info("Registry MCP server: DB pool closed")


app = FastAPI(
    title="Registry MCP Server",
    description="Model Context Protocol tools for Phase 3 Model Registry operations",
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

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

# Rate limiter (same pattern as mlflow_server) 
_request_counts: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT  = 100
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

# Tool discovery (MCP spec)
@app.get("/tools")
async def list_tools():
    """Dynamic tool discovery — agents call this at startup."""
    return {
        "tools": [
            {
                "name":     "validate_tags",
                "endpoint": "/registry/v1/validate_tags",
                "version":  "1",
                "description": (
                    "Dry-run tag validation. Validates a tag dict against the 15 "
                    "staging-required or all 18 production-required mandatory tags "
                    "without performing any MLflow registration. Returns validated "
                    "fields or structured error diagnostics."
                ),
            },
            {
                "name":     "register_model",
                "endpoint": "/registry/v1/register_model",
                "version":  "1",
                "description": (
                    "Register a trained model in the MLflow Model Registry via "
                    "RegistrationGate. Enforces all 15 staging tags, applies them to "
                    "the MLflow version, persists to Neon, and writes an audit trail "
                    "entry. Returns the version string."
                ),
            },
            {
                "name":     "promote_model",
                "endpoint": "/registry/v1/promote_model",
                "version":  "1",
                "description": (
                    "Trigger a model lifecycle state transition via "
                    "PromotionStateMachine. Enforces the transition graph and runs all "
                    "gates (Evaluation, HITL, Security, Governance, OCI) for "
                    "Staging→Production. All transitions are immutably logged."
                ),
            },
            {
                "name":     "get_lineage",
                "endpoint": "/registry/v1/lineage/{model_name}/{model_version}",
                "version":  "1",
                "description": (
                    "Return the full provenance DAG (nodes + edges) for a model "
                    "version. Used by the UI lineage graph and AI engineer audit queries."
                ),
            },
            {
                "name":     "compare_models",
                "endpoint": "/registry/v1/compare_models",
                "version":  "1",
                "description": (
                    "Side-by-side metric comparison for 2–4 model versions. Computes "
                    "pairwise diffs (challenger vs baseline), recommends the best "
                    "version by F1, and persists the session to model_comparisons."
                ),
            },
            {
                "name":     "get_promotion_history",
                "endpoint": "/registry/v1/promotion_history/{model_name}",
                "version":  "1",
                "description": (
                    "Return the immutable promotion event log for a model, optionally "
                    "filtered by version. Used by the UI's historical timeline and "
                    "compliance reports."
                ),
            },
        ]
    }

# Request / response schemas
class ValidateTagsRequest(BaseModel):
    tags:       dict[str, str]
    checkpoint: str = Field(
        default="staging",
        pattern="^(staging|production)$",
        description="'staging' checks 15 tags; 'production' checks all 18.",
    )


class ValidateTagsResponse(BaseModel):
    valid:       bool
    checkpoint:  str
    parsed_tags: Optional[dict[str, Any]] = None   # typed values on success
    error:       Optional[dict[str, Any]] = None   # RegistrationError detail on failure


class RegisterModelRequest(BaseModel):
    run_id:        str = Field(..., description="MLflow run ID that produced the model artifact.")
    model_name:    str = Field(..., min_length=1)
    tags:          dict[str, str] = Field(..., description="Dict of mlops.* tag keys to string values.")
    artifact_path: str = Field(default="model", description="Path within the MLflow run artifacts.")
    workflow_id:   Optional[str] = None


class RegisterModelResponse(BaseModel):
    model_name:   str
    version:      str
    framework:    str
    eval_accuracy: float
    eval_f1:      float


class PromoteModelRequest(BaseModel):
    model_name:       str
    model_version:    str
    target_stage:     str = Field(
        ...,
        description="Target stage: 'Staging', 'Production', 'Archived', or 'Rejected'.",
    )
    triggered_by:     str = Field(..., description="GitHub username or agent name.")
    trigger_type:     str = Field(default="automatic", pattern="^(automatic|human)$")
    workflow_id:      Optional[str] = None
    rejection_reason: Optional[str] = None


class PromoteModelResponse(BaseModel):
    model_name:       str
    model_version:    str
    from_stage:       str
    to_stage:         str
    gates_passed:     list[str]
    gates_failed:     list[str]
    success:          bool
    promotion_db_id:  int
    rejection_reason: Optional[str] = None


class CompareModelsRequest(BaseModel):
    model_name:   str
    versions:     list[str] = Field(..., min_length=2, description="2–4 version strings. First = baseline.")
    initiated_by: str = Field(default="system")
    workflow_id:  Optional[str] = None


class CompareModelsResponse(BaseModel):
    comparison_id:       str
    model_name:          str
    recommended_version: str
    versions:            list[dict[str, Any]]
    metric_diffs:        list[dict[str, Any]]

# Endpoints
@app.post("/registry/v1/validate_tags", response_model=ValidateTagsResponse)
async def validate_tags(request: ValidateTagsRequest) -> ValidateTagsResponse:
    """
    Dry-run tag validation.  No MLflow or DB writes.

    Returns 200 with valid=False (not 422) so callers can inspect diagnostics
    programmatically without try/except around the HTTP call.
    """
    try:
        if request.checkpoint == "production":
            parsed = TagValidator.validate_for_production(request.tags)
        else:
            parsed = TagValidator.validate_for_staging(request.tags)

        return ValidateTagsResponse(
            valid=True,
            checkpoint=request.checkpoint,
            parsed_tags=parsed.to_mlflow_tags(),
        )
    except RegistrationError as exc:
        return ValidateTagsResponse(
            valid=False,
            checkpoint=request.checkpoint,
            error=exc.to_dict(),
        )


@app.post("/registry/v1/register_model", response_model=RegisterModelResponse)
async def register_model(request: RegisterModelRequest) -> RegisterModelResponse:
    """
    Register a model via RegistrationGate.

    Validates all 15 staging tags, registers in MLflow, applies tags to the
    version, persists to Neon, and writes an agent_audit_trail row.
    """
    try:
        pool = await _get_pool()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")

    gate = RegistrationGate(pool=pool)
    try:
        version, tags = await gate.register(
            run_id=request.run_id,
            model_name=request.model_name,
            raw_tags=request.tags,
            artifact_path=request.artifact_path,
            workflow_id=request.workflow_id,
        )
    except RegistrationError as exc:
        raise HTTPException(status_code=422, detail=exc.to_dict())
    except mlflow.exceptions.MlflowException as exc:
        raise HTTPException(status_code=500, detail=f"MLflow error: {exc}")

    return RegisterModelResponse(
        model_name=request.model_name,
        version=version,
        framework=tags.framework,
        eval_accuracy=tags.eval_accuracy,
        eval_f1=tags.eval_f1,
    )


@app.post("/registry/v1/promote_model", response_model=PromoteModelResponse)
async def promote_model(request: PromoteModelRequest) -> PromoteModelResponse:
    """
    Trigger a state machine transition via PromotionStateMachine.

    For Staging → Production, all 5 gates are run in sequence.  Gate failures
    return HTTP 422 with the gate name and reason in the error detail.
    Illegal transitions (e.g. Archived → Staging) return HTTP 400.
    """
    try:
        target = ModelStage(request.target_stage)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target stage '{request.target_stage}'. "
                   f"Valid values: {[s.value for s in ModelStage]}",
        )

    try:
        pool = await _get_pool()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")

    sm = PromotionStateMachine(pool=pool)
    promo_request = PromotionRequest(
        model_name=request.model_name,
        model_version=request.model_version,
        target_stage=target,
        triggered_by=request.triggered_by,
        trigger_type=request.trigger_type,  # type: ignore[arg-type]
        workflow_id=request.workflow_id,
        rejection_reason=request.rejection_reason,
    )

    try:
        result = await sm.promote(promo_request)
    except ValueError as exc:
        # Illegal transition
        raise HTTPException(status_code=400, detail=str(exc))
    except RegistrationError as exc:
        raise HTTPException(status_code=422, detail=exc.to_dict())
    except PromotionGateError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "gate":       exc.gate.value,
                "reason":     exc.reason,
                "model_name": exc.model_name,
                "version":    exc.version,
            },
        )

    return PromoteModelResponse(
        model_name=result.model_name,
        model_version=result.model_version,
        from_stage=result.from_stage.value,
        to_stage=result.to_stage.value,
        gates_passed=result.gates_passed,
        gates_failed=result.gates_failed,
        success=result.success,
        promotion_db_id=result.promotion_db_id,
        rejection_reason=result.rejection_reason,
    )


@app.get("/registry/v1/lineage/{model_name}/{model_version}")
async def get_lineage(model_name: str, model_version: str) -> dict[str, Any]:
    """
    Return the full provenance DAG for a model version.

    Response shape:
        {model_name, model_version, nodes: [...], edges: [...]}
    Returns an empty nodes/edges list if no lineage has been recorded yet.
    """
    try:
        pool = await _get_pool()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")

    querier = LineageGraphQuerier(pool=pool)
    return await querier.get_full_lineage(model_name, model_version)


@app.post("/registry/v1/compare_models", response_model=CompareModelsResponse)
async def compare_models(request: CompareModelsRequest) -> CompareModelsResponse:
    """
    Build a side-by-side comparison for 2–4 model versions.

    The first version is the baseline; all MetricDiff.delta values are expressed
    as challenger − baseline.  The response includes a recommended version (highest F1)
    and the full metric diff matrix.
    """
    if len(request.versions) < 2:
        raise HTTPException(status_code=422, detail="At least 2 versions required for comparison.")
    if len(request.versions) > 4:
        raise HTTPException(status_code=422, detail="Maximum 4 versions per comparison.")

    try:
        pool = await _get_pool()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")

    comparator = ModelComparator(pool=pool)
    try:
        comparison = await comparator.compare(
            model_name=request.model_name,
            versions=request.versions,
            initiated_by=request.initiated_by,
            workflow_id=request.workflow_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return CompareModelsResponse(
        comparison_id=comparison.comparison_id,
        model_name=comparison.model_name,
        recommended_version=comparison.recommended_version,
        versions=[v.to_dict() for v in comparison.versions],
        metric_diffs=[d.to_dict() for d in comparison.metric_diffs],
    )


@app.get("/registry/v1/promotion_history/{model_name}")
async def get_promotion_history(
    model_name: str,
    version: Optional[str] = None,
) -> dict[str, Any]:
    """
    Return the immutable promotion event log for a model.

    Query params:
        version (optional): filter to a specific version string.

    Response:
        {model_name, version, promotions: [...]}
    """
    try:
        pool = await _get_pool()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")

    sm = PromotionStateMachine(pool=pool)
    history = await sm.get_promotion_history(model_name, version)
    return {
        "model_name": model_name,
        "version":    version,
        "promotions": history,
    }


# Health 
@app.get("/health")
async def health():
    pool_ok = _pool is not None
    return {
        "status":      "ok" if pool_ok else "degraded",
        "db_pool":     "connected" if pool_ok else "not initialised",
        "mlflow_uri":  settings.mlflow_tracking_uri,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)