"""
Main REST API Server
────────────────────
Exposes HTTP endpoints consumed by the React UI and
external clients (CI/CD, GitHub Actions, etc.).

Routes
  POST /api/workflows          – start a new workflow
  GET  /api/workflows/{id}     – get workflow status
  POST /api/workflows/{id}/approve  – resume a human-approval gate
  GET  /api/agents             – list registered agents
  GET  /api/models             – list MLflow model registry entries
  GET  /api/metrics/summary    – aggregated metrics from MLflow
  GET  /api/security/report    – latest security scan summary
  GET  /api/health             – liveness probe
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

import mlflow
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.orchestrator import OrchestratorAgent
from configs.a2a_registry.registry import A2ARegistry
from configs.settings import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Agent MLOps API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in production to your UI domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
_orchestrator = OrchestratorAgent()
_registry = A2ARegistry()

# In-memory workflow store (replace with Redis or Postgres in production)
_workflows: dict[str, dict[str, Any]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class StartWorkflowRequest(BaseModel):
    dataset_uri: str
    model_uri: Optional[str] = None
    workflow_id: Optional[str] = None

class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    current_stage: Optional[str] = None
    metrics: dict[str, Any] = {}
    errors: list[str] = []
    awaiting_approval: bool = False

class ApproveWorkflowRequest(BaseModel):
    approved: bool
    reviewer: str = "anonymous"
    reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Workflow endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/workflows", response_model=WorkflowResponse, status_code=202)
async def start_workflow(body: StartWorkflowRequest, background_tasks: BackgroundTasks):
    """Launch a new end-to-end MLOps workflow asynchronously."""
    wf_id = body.workflow_id or str(uuid.uuid4())
    _workflows[wf_id] = {"status": "running", "current_stage": "idle", "metrics": {}, "errors": []}

    async def _run():
        try:
            final = await _orchestrator.run_workflow(
                dataset_uri=body.dataset_uri,
                model_uri=body.model_uri or "",
                workflow_id=wf_id,
            )
            _workflows[wf_id].update({
                "status": "completed" if final.get("current_stage") != "error" else "failed",
                **final,
            })
        except Exception as exc:
            _workflows[wf_id].update({"status": "failed", "errors": [str(exc)]})

    background_tasks.add_task(_run)
    return WorkflowResponse(workflow_id=wf_id, status="running")


@app.get("/api/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str):
    """Get current state of a workflow."""
    state = _workflows.get(workflow_id)
    if not state:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return WorkflowResponse(
        workflow_id=workflow_id,
        status=state.get("status", "unknown"),
        current_stage=state.get("current_stage"),
        metrics=state.get("metrics", {}),
        errors=state.get("errors", []),
        awaiting_approval=state.get("awaiting_approval", False),
    )


@app.post("/api/workflows/{workflow_id}/approve", response_model=WorkflowResponse)
async def approve_workflow(workflow_id: str, body: ApproveWorkflowRequest):
    """
    Resume a workflow paused at the human-approval gate.
    """
    state = _workflows.get(workflow_id)
    if not state:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if not state.get("awaiting_approval"):
        raise HTTPException(status_code=400, detail="Workflow is not awaiting approval")

    logger.info("Workflow %s %s by %s: %s", workflow_id, "approved" if body.approved else "rejected", body.reviewer, body.reason)
    final = await _orchestrator.resume_workflow(workflow_id, body.approved)
    _workflows[workflow_id].update(final)
    return WorkflowResponse(
        workflow_id=workflow_id,
        status=_workflows[workflow_id].get("status", "running"),
        current_stage=final.get("current_stage"),
        metrics=final.get("metrics", {}),
        errors=final.get("errors", []),
        awaiting_approval=final.get("awaiting_approval", False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent registry
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/agents")
async def list_agents():
    """Return all registered agent cards."""
    return {"agents": [c.model_dump() for c in _registry.list_all()]}


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/models")
async def list_models(stage: Optional[str] = Query(None, pattern="^(Staging|Production|Archived)$")):
    """List models from MLflow Model Registry, optionally filtered by stage."""
    try:
        client = mlflow.tracking.MlflowClient()
        registered = client.search_registered_models()
        result = []
        for rm in registered:
            for v in rm.latest_versions:
                if stage and v.current_stage != stage:
                    continue
                result.append({
                    "name": rm.name,
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "description": rm.description,
                    "tags": dict(rm.tags),
                })
        return {"models": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Metrics aggregation
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/metrics/summary")
async def metrics_summary():
    """Return aggregated metrics across recent MLflow experiments."""
    try:
        df = mlflow.search_runs(
            experiment_names=[settings.mlflow_experiment_name],
            max_results=50,
        )
        if df.empty:
            return {"summary": {}}

        def safe_stat(col):
            if col not in df.columns:
                return None
            vals = df[col].dropna()
            if vals.empty:
                return None
            return {"min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean())}

        return {
            "summary": {
                "accuracy": safe_stat("metrics.accuracy"),
                "f1_score": safe_stat("metrics.f1_score"),
                "drift_score": safe_stat("metrics.drift_score"),
                "validation_score": safe_stat("metrics.validation_score"),
            },
            "total_runs": len(df),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "env": settings.app_env}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.debug)
