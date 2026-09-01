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
import json
import logging
import time
import uuid
from typing import Any, Optional

import mlflow
from fastapi import (
    BackgroundTasks, FastAPI, HTTPException, Query, Request, Response,
    WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest

from agents import opa_admin
from agents.deployment_agent import DeploymentAgent
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
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation — configs/monitoring/prometheus.yaml already
# scrapes orchestrator:8000/metrics (job=orchestrator-api) and
# configs/monitoring/grafana-dashboard.json's "API Request Rate"/"API
# Latency p99" panels already query http_requests_total /
# http_request_duration_seconds_bucket — but nothing in this file ever
# exposed a /metrics endpoint or these metric names, so both panels have
# never had real data behind them. Real middleware, not a stub.
_HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total", "Total HTTP requests handled by the API server.",
    ["method", "path", "status"],
)
_HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", "HTTP request duration in seconds.",
    ["method", "path"],
)


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    _HTTP_REQUESTS_TOTAL.labels(request.method, path, str(response.status_code)).inc()
    _HTTP_REQUEST_DURATION.labels(request.method, path).observe(duration)
    return response


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
_orchestrator = OrchestratorAgent()
_registry = A2ARegistry()
_deployment_agent = DeploymentAgent()

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


@app.get("/api/workflows")
async def list_workflows(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)):
    """
    Real, offset-paginated workflow history/stream for the Phase 6 Command
    Center page — backed by the `workflows` table
    (agents/orchestrator.py.list_workflows), not the in-memory _workflows
    dict below, which only reflects workflows launched by this particular
    process since its last restart. total_count is a real COUNT(*), so the
    UI can render real "page N of M" / disable-Next-at-the-end controls.
    """
    return await _orchestrator.list_workflows(limit=limit, offset=offset)


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
    final = await _orchestrator.resume_workflow(
        workflow_id, body.approved, approved_by=body.reviewer or "unknown",
    )
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


_k8s_client_module: Any = None


def _init_k8s_client() -> Any:
    """
    Lazily initialise the Kubernetes client — same in-cluster/kubeconfig
    pattern as agents/training_agent.py._init_k8s. Returns None (not a
    raised exception) when unavailable, e.g. running api_server.py locally
    without a kubeconfig — callers degrade to reporting "unknown" pod
    status rather than crashing the request.
    """
    global _k8s_client_module
    if _k8s_client_module is not None:
        return _k8s_client_module
    try:
        from kubernetes import client as k8s_client, config as k8s_config
        if settings.k8s_in_cluster:
            k8s_config.load_incluster_config()
        else:
            k8s_config.load_kube_config()
        _k8s_client_module = k8s_client
    except Exception as exc:
        logger.warning("Kubernetes client init failed — pod status will report 'unknown': %s", exc)
    return _k8s_client_module


@app.get("/api/agents/status")
async def agent_pod_status():
    """
    Real per-agent pod health from the Kubernetes API — replaces the V1 UI's
    AgentStatusGrid, which defaulted every agent to a hardcoded 'online'.
    Matches pods in settings.k8s_namespace against each registered
    AgentCard's `app` label convention (agent_id with underscores replaced
    by hyphens, e.g. orchestrator_agent -> orchestrator-agent), and reports
    real phase/ready/restart-count — 'unknown' (not 'online') for any agent
    with no matching pod or when the Kubernetes client itself is
    unavailable, so a missing pod is never visually indistinguishable from
    a healthy one.
    """
    k8s = _init_k8s_client()
    cards = _registry.list_all()
    if k8s is None:
        return {
            "agents": [
                {"agent_id": c.agent_id, "status": "unknown", "reason": "kubernetes client unavailable"}
                for c in cards
            ]
        }
    try:
        core_v1 = k8s.CoreV1Api()
        pods = await asyncio.to_thread(core_v1.list_namespaced_pod, namespace=settings.k8s_namespace)
    except Exception as exc:
        logger.warning("Pod status query failed: %s", exc)
        return {
            "agents": [
                {"agent_id": c.agent_id, "status": "unknown", "reason": f"pod list query failed: {exc}"}
                for c in cards
            ]
        }

    by_app_label: dict[str, list] = {}
    for pod in pods.items:
        app_label = (pod.metadata.labels or {}).get("app")
        if app_label:
            by_app_label.setdefault(app_label, []).append(pod)

    result = []
    for card in cards:
        app_name = card.agent_id.replace("_", "-")
        matching = by_app_label.get(app_name, [])
        if not matching:
            result.append({"agent_id": card.agent_id, "status": "unknown", "reason": "no matching pod found"})
            continue
        phases = [p.status.phase for p in matching]
        ready_counts = [
            sum(1 for cs in (p.status.container_statuses or []) if cs.ready)
            for p in matching
        ]
        total_containers = [len(p.status.container_statuses or []) for p in matching]
        restarts = sum(
            sum(cs.restart_count for cs in (p.status.container_statuses or []))
            for p in matching
        )
        all_ready = all(r == t and t > 0 for r, t in zip(ready_counts, total_containers))
        status = "online" if all_ready and all(ph == "Running" for ph in phases) else "degraded"
        result.append({
            "agent_id": card.agent_id,
            "status": status,
            "pod_count": len(matching),
            "phases": phases,
            "restarts": restarts,
        })
    return {"agents": result}


# ─────────────────────────────────────────────────────────────────────────────
# Real-time events (Phase 6 §6.3) — WebSocket relay over Redis pub/sub
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    """
    Streams JSON events published by agents/events.py (workflow status
    changes, alerts, deployment progress) to the connected browser client.
    One Redis pub/sub connection per WebSocket client, subscribed via
    psubscribe to every channel agents/events.py publishes — no polling,
    and no fan-out server needed since Redis already does that.

    settings.ws_max_connections_per_client has existed since it was first
    stubbed in during Phase 6 planning but was never actually enforced —
    this endpoint accepted unlimited connections per client regardless of
    the setting. Enforced here via a Redis counter (not in-process memory)
    so the cap is correct across every orchestrator-agent replica, not
    just whichever pod happens to receive a given connection.
    """
    await websocket.accept()
    try:
        import redis.asyncio as aioredis
    except Exception as exc:
        await websocket.close(code=1011, reason=f"redis client unavailable: {exc}")
        return

    try:
        redis_client = aioredis.from_url(
            settings.redis_url, max_connections=settings.redis_max_connections, decode_responses=True,
        )
        await redis_client.ping()
    except Exception as exc:
        logger.warning("WS client could not reach Redis: %s", exc)
        await websocket.close(code=1011, reason="event bus unavailable")
        return

    client_ip = (
        websocket.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (websocket.client.host if websocket.client else "unknown")
    )
    conn_key = f"ws_conn_count:{client_ip}"
    current = await redis_client.incr(conn_key)
    await redis_client.expire(conn_key, 3600)  # safety TTL in case decrement is ever missed (crash, kill -9)
    if current > settings.ws_max_connections_per_client:
        await redis_client.decr(conn_key)
        await websocket.close(
            code=1008,
            reason=f"Too many connections from {client_ip} "
                   f"(limit {settings.ws_max_connections_per_client})",
        )
        await redis_client.close()
        return

    pubsub = redis_client.pubsub()
    await pubsub.psubscribe("workflow.*", "alert.*", "model.*")

    async def _heartbeat():
        while True:
            await asyncio.sleep(settings.ws_heartbeat_interval_seconds)
            await websocket.send_json({"channel": "_heartbeat", "ts": time.time()})

    heartbeat_task = asyncio.create_task(_heartbeat())
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message is not None and message.get("type") == "pmessage":
                try:
                    await websocket.send_json(json.loads(message["data"]))
                except Exception:
                    await websocket.send_text(str(message["data"]))
            # Yield so a client-initiated close (WebSocketDisconnect) is noticed promptly.
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WS event stream error: %s", exc)
    finally:
        heartbeat_task.cancel()
        await pubsub.punsubscribe()
        await pubsub.close()
        try:
            await redis_client.decr(conn_key)
        except Exception:
            pass  # TTL on conn_key is the backstop if this itself fails
        await redis_client.close()


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

_EXPERIMENT_RUNS_LOOKBACK_CAP = 500  # see docstring


@app.get("/api/experiments/runs")
async def list_experiment_runs(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)):
    """
    Real, paginated per-run listing for the Phase 6 Experiment Lab page —
    the V1 UI's ExperimentsPage generated its entire run table with
    Math.random() and never called any API at all. This is a genuine
    mlflow.search_runs() call against the real tracking server, every
    field taken straight from the DataFrame it returns.

    MLflow's fluent search_runs() has no cheap "count all runs" call and
    its own pagination is cursor-based (page tokens), not offset-based, so
    true offset pagination over an MLflow-backed table means picking a
    bound: this fetches the most recent _EXPERIMENT_RUNS_LOOKBACK_CAP runs
    in one real, sorted query and paginates within that window server-side
    (real data, real order — not a client-side slice of a much smaller
    fetch). total_count is honestly capped at that window size rather than
    claiming to know about runs beyond it.
    """
    try:
        df = mlflow.search_runs(
            experiment_names=[settings.mlflow_experiment_name],
            max_results=_EXPERIMENT_RUNS_LOOKBACK_CAP,
            order_by=["start_time DESC"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    total_count = len(df)
    df = df.iloc[offset:offset + limit]
    if df.empty:
        return {"runs": [], "total_count": total_count}

    def safe(v: Any) -> Any:
        try:
            if v is None or (isinstance(v, float) and v != v):  # NaN
                return None
        except Exception:
            pass
        return v

    runs = []
    for _, row in df.iterrows():
        metrics = {
            col.removeprefix("metrics."): safe(row[col])
            for col in df.columns if col.startswith("metrics.") and safe(row[col]) is not None
        }
        params = {
            col.removeprefix("params."): safe(row[col])
            for col in df.columns if col.startswith("params.") and safe(row[col]) is not None
        }
        runs.append({
            "run_id": row.get("run_id"),
            "run_name": row.get("tags.mlflow.runName") or row.get("run_id"),
            "status": row.get("status"),
            "start_time": row["start_time"].isoformat() if safe(row.get("start_time")) is not None else None,
            "end_time": row["end_time"].isoformat() if safe(row.get("end_time")) is not None else None,
            "metrics": metrics,
            "params": params,
        })
    return {"runs": runs, "total_count": total_count}


@app.get("/api/experiments/runs/{run_id}/trace")
async def get_run_trace(run_id: str):
    """
    Real span tree for a run, sourced from MLflow's trace store (the same
    @mlflow.trace decorators every agent already uses — configs/telemetry.py,
    agents/*.py). mlflow.search_traces()'s `spans` column holds a list of
    plain dicts shaped like {name, context: {span_id, trace_id}, parent_id,
    start_time, end_time, status_code, attributes, events} — verified
    directly against a real MLflow trace rather than assumed. Returns an
    empty list (not fabricated spans) if the run predates instrumentation.
    """
    try:
        # run_id alone (no experiment_ids) silently returns zero rows in
        # mlflow 2.17 — verified directly against a real trace store, not
        # assumed from the SDK's type signature. search_traces() (unlike
        # search_runs()) has no experiment_names kwarg, so the experiment
        # id has to be resolved first.
        experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
        experiment_ids = [experiment.experiment_id] if experiment else None
        traces = mlflow.search_traces(experiment_ids=experiment_ids, run_id=run_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if traces is None or traces.empty:
        return {"spans": []}
    spans_out = []
    for _, trow in traces.iterrows():
        for span in (trow.get("spans") or []):
            context = span.get("context") or {}
            spans_out.append({
                "span_id": str(context.get("span_id", "")),
                "name": span.get("name", ""),
                "start_time_ns": span.get("start_time"),
                "end_time_ns": span.get("end_time"),
                "parent_id": str(span.get("parent_id") or ""),
                "status": span.get("status_code", ""),
            })
    return {"spans": spans_out}


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
# Agent Intelligence (Phase 6) — real per-agent activity from token_usage,
# real OPA audit log from policy_decisions. Both tables have existed since
# the Phase 2 migration (0003_llm_gateway.py) but nothing ever read them —
# LLMGateway/GovernanceAgent write to them, no UI or endpoint surfaced them.
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/agents/{agent_role}/activity")
async def agent_activity(agent_role: str, limit: int = Query(10, ge=1, le=100)):
    await _orchestrator._ensure_connected()
    pool = _orchestrator._pool
    if pool is None:
        return {"agent_role": agent_role, "recent_calls": [], "summary": None}
    async with pool.acquire() as conn:
        calls = await conn.fetch(
            """
            SELECT workflow_id, model, was_fallback, was_cache_hit, prompt_tokens,
                   completion_tokens, total_tokens, cost_usd, latency_ms, created_at
            FROM token_usage WHERE agent_role=$1 ORDER BY created_at DESC LIMIT $2
            """,
            agent_role, limit,
        )
        summary = await conn.fetchrow(
            """
            SELECT COUNT(*) AS call_count,
                   AVG(latency_ms) AS avg_latency_ms,
                   SUM(total_tokens) AS total_tokens,
                   SUM(cost_usd) AS total_cost_usd,
                   AVG(CASE WHEN was_cache_hit THEN 1.0 ELSE 0.0 END) AS cache_hit_rate
            FROM token_usage WHERE agent_role=$1
            """,
            agent_role,
        )
        success = await conn.fetchrow(
            """
            SELECT COUNT(*) FILTER (WHERE w.status != 'error') AS ok_count, COUNT(*) AS total_count
            FROM (SELECT DISTINCT workflow_id FROM token_usage WHERE agent_role=$1 AND workflow_id IS NOT NULL) t
            JOIN workflows w ON w.id = t.workflow_id
            """,
            agent_role,
        )
    success_rate = (
        success["ok_count"] / success["total_count"] if success and success["total_count"] else None
    )
    return {
        "agent_role": agent_role,
        "recent_calls": [
            {
                "workflow_id": r["workflow_id"], "model": r["model"], "was_fallback": r["was_fallback"],
                "was_cache_hit": r["was_cache_hit"], "total_tokens": r["total_tokens"],
                "cost_usd": r["cost_usd"], "latency_ms": r["latency_ms"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in calls
        ],
        "summary": {
            "call_count": summary["call_count"] if summary else 0,
            "avg_latency_ms": float(summary["avg_latency_ms"]) if summary and summary["avg_latency_ms"] is not None else None,
            "total_tokens": summary["total_tokens"] if summary else 0,
            "total_cost_usd": float(summary["total_cost_usd"]) if summary and summary["total_cost_usd"] is not None else None,
            "cache_hit_rate": float(summary["cache_hit_rate"]) if summary and summary["cache_hit_rate"] is not None else None,
            "success_rate": success_rate,
        } if summary else None,
    }


@app.get("/api/governance/audit-log")
async def governance_audit_log(limit: int = Query(50, ge=1, le=200)):
    await _orchestrator._ensure_connected()
    pool = _orchestrator._pool
    if pool is None:
        return {"decisions": []}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT workflow_id, policy_name, policy_package, rego_policy_version, decision,
                   deny_reasons, agent_role, was_sidecar, evaluation_time_ms, created_at
            FROM policy_decisions ORDER BY created_at DESC LIMIT $1
            """,
            limit,
        )
    return {
        "decisions": [
            {
                "workflow_id": r["workflow_id"], "policy_name": r["policy_name"],
                "policy_package": r["policy_package"], "rego_policy_version": r["rego_policy_version"],
                "decision": r["decision"],
                "deny_reasons": json.loads(r["deny_reasons"]) if isinstance(r["deny_reasons"], str) else r["deny_reasons"],
                "agent_role": r["agent_role"], "was_sidecar": r["was_sidecar"],
                "evaluation_time_ms": r["evaluation_time_ms"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Live serving — manual rollback (Phase 6 Live Serving Dashboard)
# ─────────────────────────────────────────────────────────────────────────────

class RollbackRequest(BaseModel):
    reason: str
    model_name: str = ""
    model_version: str = ""


@app.post("/api/deployments/{service_name}/rollback")
async def rollback_deployment(service_name: str, body: RollbackRequest):
    if not body.reason.strip():
        raise HTTPException(status_code=422, detail="A rollback reason is required")
    try:
        await asyncio.to_thread(
            _deployment_agent.manual_rollback, service_name, body.reason, body.model_name, body.model_version,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Rollback failed: {exc}")
    return {"service_name": service_name, "status": "rolled_back", "reason": body.reason}


# ─────────────────────────────────────────────────────────────────────────────
# OPA policy editor (Phase 6 Security & Compliance page)
# ─────────────────────────────────────────────────────────────────────────────

class OpaPolicyBody(BaseModel):
    rego: str


@app.get("/api/opa/policy")
async def get_opa_policy():
    try:
        rego = await asyncio.to_thread(opa_admin.read_policy)
        return {"rego": rego}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not read policy ConfigMap: {exc}")


@app.put("/api/opa/policy")
async def put_opa_policy(body: OpaPolicyBody):
    try:
        result = await asyncio.to_thread(opa_admin.write_policy, body.rego)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not write policy ConfigMap: {exc}")
    if not result.valid:
        raise HTTPException(status_code=422, detail=result.to_dict())
    return result.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "env": settings.app_env}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.debug)
