"""
Orchestrator Agent

The root node of the LangGraph workflow.

Responsibilities
- Discover specialist agents via the A2A registry
- Decompose incoming requests into a phased execution plan
- Route control between Data → Training → Security scan → Evaluation →
  Registration (Staging) → Human approval → Packaging (OCI) →
  Promotion (Production) → Deployment → Monitoring
- Handle human-in-the-loop gates before high-risk transitions
- Surface errors and initiate rollback / retry sequences
- Own the single shared Neon connection pool and wire it into every agent
  that needs it (this used to never happen — see connect() calls below)

Pattern: Orchestrator-Worker (see agentic-systems.md § Multi-Agent Coordination)

Registry/packaging wiring (Phase 3 + 4):
  Registration (None → Staging) and promotion (Staging → Production) are
  never performed ad hoc by an agent — they go exclusively through
  RegistrationGate and PromotionStateMachine, invoked from the
  "registration" and "promotion" nodes below. Packaging (Dockerfile
  generation → Kaniko build → Trivy scan → SBOM → Cosign sign) runs between
  human approval and the Production promotion gate, so the promotion gate's
  OCI check has something real to verify, and so DeploymentAgent has a real
  OCI image to hand to KServe instead of falling back to the MLflow runtime.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg
import mlflow
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from agents import AgentState, AgentTaskResult, make_initial_state
from agents.planner_agent import PlannerAgent
from agents.data_agent import DataAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.deployment_agent import DeploymentAgent
from agents.monitoring_agent import MonitoringAgent
from agents.governance_agent import GovernanceAgent
from agents.security_agent import SecurityAgent
from agents.events import publish_workflow_status
from agents.registry.metadata_schema import RegistrationGate, RegistrationError
from agents.registry.promotion_state_machine import (
    PromotionStateMachine, PromotionRequest, ModelStage, PromotionGateError,
)
from agents.packaging.dockerfile_agent import (
    DockerfileAgent, PackagingRequest, DockerfileAgentError,
)
from configs.a2a_registry.registry import A2ARegistry
from configs.settings import settings
from configs.telemetry import get_tracer
from mcp_servers.mlflow_server import log_agent_decision

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)

# High-risk transitions that require human approval before proceeding
HIGH_RISK_TRANSITIONS: set[str] = {
    "training_to_deployment",
    "canary_to_full_rollout",
    "initiate_rollback",
}

_ZERO_SHA256 = "0" * 64


def _resolve_git_provenance() -> tuple[Optional[str], str]:
    """
    Resolve the current git commit SHA. Returns (commit_or_None, repo_url).
    A missing/unavailable git commit is surfaced as missing provenance by the
    registration node — it is never faked with a placeholder SHA.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        commit = proc.stdout.strip()
        if len(commit) != 40:
            commit = None
    except Exception as exc:
        logger.warning("git rev-parse failed — no git commit available: %s", exc)
        commit = None
    return commit, settings.git_repo_url


def _resolve_framework_version(framework: Optional[str]) -> str:
    """Best-effort installed package version for the mlops.framework.version tag."""
    import importlib.metadata as _md

    pkg_map = {
        "xgboost": "xgboost", "pytorch": "torch",
        "sklearn": "scikit-learn", "huggingface": "transformers",
    }
    pkg = pkg_map.get(framework or "", framework or "")
    try:
        return _md.version(pkg)
    except Exception:
        return "unknown"


class OrchestratorAgent:
    """
    Central coordinator for the multi-agent MLOps pipeline.

    Builds a LangGraph StateGraph where each node delegates to a
    specialist agent.  Checkpoints via Neon PostgreSQL so interrupted
    workflows can be resumed automatically.
    """

    def __init__(self, db_conn_string: str = settings.database_sync_url) -> None:
        self.registry = A2ARegistry()
        self._planner_agent = PlannerAgent()
        self._data_agent = DataAgent()
        self._training_agent = TrainingAgent()
        self._evaluation_agent = EvaluationAgent()
        self._deployment_agent = DeploymentAgent()
        self._monitoring_agent = MonitoringAgent()
        self._governance_agent = GovernanceAgent()
        self._security_agent = SecurityAgent()

        self._pool: Optional[asyncpg.Pool] = None
        self._connected = False
        self._db_conn_string = db_conn_string
        # Kept open for the orchestrator's lifetime — AsyncPostgresSaver is
        # an async context manager, so its connection can't be entered
        # inside this synchronous __init__; it's opened in _ensure_connected.
        self._checkpointer_cm: Optional[Any] = None

        # Uncompiled builder, so _ensure_connected can recompile once a real
        # checkpointer is available. Compiled stateless for now as a safe
        # fallback for any caller that never calls _ensure_connected.
        self._graph_builder = self._build_graph()
        self._graph = self._graph_builder.compile()

    # Shared pool + checkpointer wiring
    async def _ensure_connected(self) -> None:
        """
        Lazily create the single shared Neon pool (wired into every agent
        that needs it — previously this never happened at all, so every
        agent's self._pool stayed None for the life of the process,
        silently disabling lineage writes, the LLM Gateway's semantic
        cache, and every Phase 3/4 registry/packaging call this module
        makes directly) AND the real async Postgres checkpointer.

        The checkpointer MUST be langgraph's AsyncPostgresSaver, not the
        sync PostgresSaver — PostgresSaver.aget_tuple() raises
        NotImplementedError, which crashes the very first ainvoke() call on
        a graph compiled with it. This was verified against a real
        Postgres instance while building Phase 5: the sync saver + ainvoke
        combination the orchestrator originally used does not work at all.

        interrupt_before=["human_approval"] is what actually makes the
        human-approval gate pause execution — without it, ainvoke() runs
        straight through the node in one call and there is nothing to
        "resume" later (also verified against a real checkpointer).
        """
        if self._connected:
            return

        if settings.database_url:
            try:
                self._pool = await asyncpg.create_pool(
                    settings.asyncpg_url,
                    min_size=1,
                    max_size=settings.database_pool_size,
                    command_timeout=30,
                    statement_cache_size=0,   # required for Neon pgBouncer
                )
                logger.info("Orchestrator: shared Neon pool ready")
            except Exception as exc:
                logger.warning("Orchestrator: could not create Neon pool — %s", exc)
                self._pool = None
        else:
            logger.warning("DATABASE_URL not set — registry/lineage/governance writes disabled")

        if self._db_conn_string:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                self._checkpointer_cm = AsyncPostgresSaver.from_conn_string(self._db_conn_string)
                checkpointer = await self._checkpointer_cm.__aenter__()
                await checkpointer.setup()
                self._graph = self._graph_builder.compile(
                    checkpointer=checkpointer,
                    interrupt_before=["plan_approval", "human_approval"],
                )
                logger.info("Orchestrator: async Postgres checkpointer ready — human_approval can pause/resume")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Checkpointer unavailable — running stateless. "
                    "human_approval will NOT actually pause in this mode: %s", exc,
                )
        else:
            logger.warning(
                "No DATABASE_SYNC_URL configured — human_approval cannot "
                "actually pause the graph in this mode (see run_workflow's docstring)."
            )

        for agent in (
            self._planner_agent, self._data_agent, self._training_agent,
            self._evaluation_agent, self._security_agent, self._governance_agent,
            self._monitoring_agent,
        ):
            connect = getattr(agent, "connect", None)
            if connect is not None:
                try:
                    await connect(self._pool)
                except Exception as exc:
                    logger.warning("%s.connect() failed: %s", type(agent).__name__, exc)

        self._connected = True

    # Graph construction
    def _build_graph(self) -> StateGraph:
        g = StateGraph(dict)

        # Nodes
        g.add_node("plan", self._plan_node)
        g.add_node("plan_approval", self._plan_approval_node)
        g.add_node("security_gate", self._security_gate_node)
        g.add_node("data", self._data_node)
        g.add_node("training", self._training_node)
        g.add_node("security_scan", self._security_scan_node)
        g.add_node("evaluation", self._evaluation_node)
        g.add_node("registration", self._registration_node)
        g.add_node("human_approval", self._human_approval_node)
        g.add_node("approval_stamp", self._approval_stamp_node)
        g.add_node("packaging", self._packaging_node)
        g.add_node("promotion", self._promotion_node)
        g.add_node("deployment", self._deployment_node)
        g.add_node("monitoring", self._monitoring_node)
        g.add_node("governance_audit", self._governance_audit_node)
        g.add_node("error_handler", self._error_handler_node)

        # Entry point
        g.set_entry_point("plan")

        # Edges
        g.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {"ok": "security_gate", "needs_approval": "plan_approval", "error": "error_handler"},
        )
        g.add_conditional_edges(
            "plan_approval",
            self._route_after_plan_approval,
            {"approved": "security_gate", "rejected": "error_handler"},
        )
        g.add_conditional_edges(
            "security_gate",
            self._route_after_security,
            {"pass": "data", "fail": "error_handler"},
        )
        g.add_conditional_edges(
            "data",
            self._route_after_data,
            {"ok": "training", "error": "error_handler"},
        )
        g.add_conditional_edges(
            "training",
            self._route_after_training,
            {"ok": "security_scan", "error": "error_handler"},
        )
        g.add_conditional_edges(
            "security_scan",
            self._route_after_security_scan,
            {"ok": "evaluation", "error": "error_handler"},
        )
        g.add_conditional_edges(
            "evaluation",
            self._route_after_evaluation,
            {
                "approved": "registration",
                "rejected": "training",   # re-train with RL suggestions
                "error": "error_handler",
            },
        )
        g.add_conditional_edges(
            "registration",
            self._route_after_registration,
            {"ok": "human_approval", "error": "error_handler"},
        )
        g.add_conditional_edges(
            "human_approval",
            self._route_after_approval,
            {"approved": "approval_stamp", "rejected": "error_handler"},
        )
        g.add_edge("approval_stamp", "packaging")
        g.add_conditional_edges(
            "packaging",
            self._route_after_packaging,
            {"ok": "promotion", "error": "error_handler"},
        )
        g.add_conditional_edges(
            "promotion",
            self._route_after_promotion,
            {"ok": "deployment", "error": "error_handler"},
        )
        g.add_conditional_edges(
            "deployment",
            self._route_after_deployment,
            {"ok": "monitoring", "error": "error_handler"},
        )
        g.add_edge("monitoring", "governance_audit")
        g.add_edge("governance_audit", END)
        g.add_edge("error_handler", END)

        return g

    # Nodes
    @mlflow.trace(name="orchestrator.plan")
    def _plan_node(self, state: dict) -> dict:
        """
        Real LLM-backed planning via PlannerAgent (agents/planner_agent.py) —
        previously this node built a static inline dict and PlannerAgent,
        despite being a complete Phase 2 ReAct implementation with real
        MLflow history lookup, RL recommendations, and its own HITL gate,
        was never actually called from the graph at all.
        """
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run(run_name=f"workflow-{state.get('workflow_id', 'unknown')}"):
            mlflow.log_param("workflow_id", state.get("workflow_id"))
            mlflow.log_param("dataset_uri", state.get("dataset_uri"))
            mlflow.set_tag("stage", "planning")

        result: AgentTaskResult = self._planner_agent.run(state)
        output = result.output or {}

        updated = {
            **state,
            "agent_decisions": state.get("agent_decisions", []) + [result.model_dump()],
        }

        if result.status == "pending_approval":
            return {
                **updated,
                "current_stage": "plan_pending_approval",
                "plan_db_id": output.get("plan_db_id"),
                "plan_summary": output.get("plan_summary"),
            }
        if result.status != "success":
            updated["errors"] = state.get("errors", []) + [result.error or "Planning failed"]
            return updated

        return {
            **updated,
            "current_stage": "data",
            "execution_plan": output.get("execution_plan", {}),
            "rl_recommendations": output.get("rl_recommendations") or state.get("rl_recommendations"),
        }

    def _plan_approval_node(self, state: dict) -> dict:
        """
        Pause for human sign-off on a high-impact execution plan
        (ExecutionPlan.human_approval_required). Mirrors _human_approval_node's
        interrupt_before-driven pause; see that node's docstring for how the
        pause/resume mechanism actually works — it was previously broken for
        BOTH gates (no interrupt_before was ever configured at all).
        """
        decision = state.get("plan_approval_decision")
        return {
            **state,
            "awaiting_approval": False,
            "agent_decisions": state.get("agent_decisions", []) + [{
                "node": "plan_approval",
                "action": f"plan_approval_{decision or 'unknown'}",
                "approved_by": state.get("plan_approved_by"),
            }],
        }

    @mlflow.trace(name="orchestrator.security_gate")
    def _security_gate_node(self, state: dict) -> dict:
        """Pre-flight input-validation check before any data is touched."""
        result: AgentTaskResult = self._security_agent.pre_flight_check(state)
        return {
            **state,
            "agent_decisions": state.get("agent_decisions", []) + [result.model_dump()],
            "_security_passed": result.status == "success",
        }

    @mlflow.trace(name="orchestrator.data")
    def _data_node(self, state: dict) -> dict:
        result: AgentTaskResult = self._data_agent.run(state)
        output = result.output or {}
        updated = {
            **state,
            "metrics": {**state.get("metrics", {}), **output},
            "agent_decisions": state.get("agent_decisions", []) + [result.model_dump()],
            "current_stage": "data",
            # Surfaced at top level — downstream nodes read these directly,
            # not through the metrics dict.
            "lineage_id": output.get("lineage_id", state.get("lineage_id")),
            "feast_feature_view": output.get("feast_feature_view", state.get("feast_feature_view")),
            "dataset_hash": output.get("content_hash", state.get("dataset_hash")),
            "dataset_row_count": output.get("row_count", state.get("dataset_row_count")),
        }
        if result.status in ("failed", "partial"):
            updated["errors"] = state.get("errors", []) + [result.error or "Data stage failed"]
        return updated

    @mlflow.trace(name="orchestrator.training")
    def _training_node(self, state: dict) -> dict:
        result: AgentTaskResult = self._training_agent.run(state)
        output = result.output or {}
        best_run_id = output.get("best_run_id", state.get("best_run_id"))
        model_uri = f"runs:/{best_run_id}/model" if best_run_id else state.get("model_uri")
        updated = {
            **state,
            "model_uri": model_uri,
            "best_run_id": best_run_id,
            "best_framework": output.get("best_framework", state.get("best_framework")),
            "training_script_path": state.get("training_script_path", ""),
            "metrics": {**state.get("metrics", {}), **output.get("best_metrics", {})},
            "agent_decisions": state.get("agent_decisions", []) + [result.model_dump()],
            "current_stage": "training",
        }
        if result.status in ("failed", "partial"):
            updated["errors"] = state.get("errors", []) + [result.error or "Training failed"]
        return updated

    @mlflow.trace(name="orchestrator.security_scan")
    def _security_scan_node(self, state: dict) -> dict:
        """
        Post-training Trivy/Semgrep/secrets/PII scan (SecurityAgent.run),
        never previously wired into the graph — governance's OPA input used
        to always see an empty security_scan dict as a result.
        """
        result: AgentTaskResult = self._security_agent.run(state)
        output = result.output or {}
        updated = {
            **state,
            "security_result": output.get("security_result", output),
            "agent_decisions": state.get("agent_decisions", []) + [result.model_dump()],
            "current_stage": "security_scan",
        }
        if result.status in ("failed", "partial"):
            updated["errors"] = state.get("errors", []) + [result.error or "Security scan failed"]
        return updated

    @mlflow.trace(name="orchestrator.evaluation")
    def _evaluation_node(self, state: dict) -> dict:
        result: AgentTaskResult = self._evaluation_agent.run(state)
        output = result.output or {}
        eval_report = output.get("eval_report", {})
        updated = {
            **state,
            "metrics": {**state.get("metrics", {}), **eval_report},
            "eval_report": eval_report,
            "holdout_hash": output.get("holdout_hash", state.get("holdout_hash")),
            "agent_decisions": state.get("agent_decisions", []) + [result.model_dump()],
            "current_stage": "evaluation",
            "_eval_approved": result.status == "success" and bool(output.get("overall_passed", False)),
        }
        return updated

    @mlflow.trace(name="orchestrator.registration")
    def _registration_node(self, state: dict) -> dict:
        """Synchronous LangGraph entry point for Staging registration."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._registration_async(state))

    async def _registration_async(self, state: dict) -> dict:
        """
        None → Staging via RegistrationGate — the only place a model may be
        registered. See agents/registry/metadata_schema.py.
        """
        task_id: str = state.get("workflow_id", "unknown")
        model_name = state.get("model_name") or settings.mlflow_registered_model_name
        best_run_id: str = state.get("best_run_id", "")
        eval_report: dict = state.get("eval_report", {})
        security_result: dict = state.get("security_result", {})

        if not best_run_id:
            return {**state, "errors": state.get("errors", []) + [
                "Registration blocked: no best_run_id in state (Training/Security stage incomplete)"
            ]}

        git_commit, git_repo = _resolve_git_provenance()

        missing_provenance: list[str] = []
        if not state.get("dataset_hash"):
            missing_provenance.append("dataset_hash (Data Agent lineage tracking failed or was skipped)")
        if not state.get("holdout_hash"):
            missing_provenance.append("holdout_hash (Evaluation Agent)")
        if not git_commit:
            missing_provenance.append("git_commit (git rev-parse failed — is this a git checkout?)")
        if missing_provenance:
            return {**state, "errors": state.get("errors", []) + [
                f"Registration blocked — missing provenance: {missing_provenance}"
            ]}

        raw_tags = {
            "mlops.dataset.uri":            state.get("dataset_uri") or "",
            "mlops.dataset.hash":            state["dataset_hash"],
            "mlops.dataset.row_count":       str(state.get("dataset_row_count") or 1),
            "mlops.framework":               state.get("best_framework") or "xgboost",
            "mlops.framework.version":       _resolve_framework_version(state.get("best_framework")),
            "mlops.python.version":          platform.python_version(),
            "mlops.eval.accuracy":           str(eval_report.get("accuracy", 0.0)),
            "mlops.eval.f1":                 str(eval_report.get("f1", 0.0)),
            "mlops.eval.auc":                str(eval_report.get("auc", 0.0)),
            "mlops.eval.holdout_hash":       state["holdout_hash"],
            "mlops.eval.bias_passed":        "true" if eval_report.get("bias_passed") else "false",
            "mlops.security.trivy_scan":     "passed" if security_result.get("trivy_passed") else "failed",
            "mlops.security.cve_critical":   str(security_result.get("critical_cves", 0)),
            "mlops.git.commit":              git_commit,
            "mlops.git.repo":                git_repo,
        }

        try:
            gate = RegistrationGate(pool=self._pool)
            version, _tags = await gate.register(
                run_id=best_run_id, model_name=model_name,
                raw_tags=raw_tags, workflow_id=task_id,
            )
        except RegistrationError as exc:
            logger.error("Registration blocked for workflow %s: %s", task_id, exc)
            return {**state, "errors": state.get("errors", []) + [f"Registration blocked: {exc}"]}
        except Exception as exc:
            logger.exception("Registration failed for workflow %s", task_id)
            return {**state, "errors": state.get("errors", []) + [f"Registration failed: {exc}"]}

        return {
            **state,
            "model_name": model_name,
            "model_version": version,
            "current_stage": "registration",
            "agent_decisions": state.get("agent_decisions", []) + [{
                "node": "orchestrator.registration",
                "action": "registered_to_staging",
                "model_name": model_name,
                "model_version": version,
            }],
        }

    def _human_approval_node(self, state: dict) -> dict:
        """
        This node only executes AFTER the graph has been resumed —
        interrupt_before=["human_approval"] (set at compile time) is what
        actually pauses execution before this node runs; ainvoke() returns
        to the caller at that point without this function ever being called.
        run_workflow() detects that paused state via aget_state() and sets
        awaiting_approval/approval_context there, since this node hasn't run
        yet to do it itself. By the time this node DOES run (resume_workflow
        having supplied approval_decision/approved_by), the human decision
        already exists in state — this just clears the waiting flag and
        logs the outcome.
        """
        decision = state.get("approval_decision")
        return {
            **state,
            "awaiting_approval": False,
            "agent_decisions": state.get("agent_decisions", []) + [{
                "node": "human_approval",
                "action": f"approval_{decision or 'unknown'}",
                "approved_by": state.get("approved_by"),
            }],
        }

    @mlflow.trace(name="orchestrator.approval_stamp")
    def _approval_stamp_node(self, state: dict) -> dict:
        """
        Stamp mlops.approved_by / mlops.approved_at on the Staging model
        version once a human has approved it — PromotionStateMachine's HITL
        gate reads these tags when Production promotion runs next.
        """
        model_name = state.get("model_name")
        model_version = state.get("model_version")
        approved_by = state.get("approved_by") or "unknown"

        if model_name and model_version:
            try:
                client = mlflow.tracking.MlflowClient()
                now_iso = datetime.now(tz=timezone.utc).isoformat()
                client.set_model_version_tag(model_name, model_version, "mlops.approved_by", approved_by)
                client.set_model_version_tag(model_name, model_version, "mlops.approved_at", now_iso)
            except Exception as exc:
                logger.warning("Failed to stamp approval tags on %s v%s: %s", model_name, model_version, exc)

        return {
            **state,
            "current_stage": "approval_stamp",
            "agent_decisions": state.get("agent_decisions", []) + [{
                "node": "orchestrator.approval_stamp", "approved_by": approved_by,
            }],
        }

    @mlflow.trace(name="orchestrator.packaging")
    def _packaging_node(self, state: dict) -> dict:
        """Synchronous LangGraph entry point for OCI packaging (Phase 4)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._packaging_async(state))

    async def _packaging_async(self, state: dict) -> dict:
        """
        Build, Trivy-scan, SBOM, and Cosign-sign the OCI image for the
        Staging model version via DockerfileAgent. Runs before the
        Production promotion gate so its OCI check has a real image, scan,
        and (optionally) signature to verify.
        """
        task_id: str = state.get("workflow_id", "unknown")
        model_name = state.get("model_name")
        model_version = state.get("model_version")

        if not model_name or not model_version:
            return {**state, "errors": state.get("errors", []) + [
                "Packaging blocked: no registered model_name/model_version in state"
            ]}

        try:
            agent = DockerfileAgent(pool=self._pool)
            result = await agent.run(PackagingRequest(
                model_name=model_name, model_version=model_version, workflow_id=task_id,
            ))
        except DockerfileAgentError as exc:
            logger.error("Packaging blocked for %s v%s: %s", model_name, model_version, exc)
            return {**state, "errors": state.get("errors", []) + [f"Packaging blocked: {exc}"]}
        except Exception as exc:
            logger.exception("Packaging failed for workflow %s", task_id)
            return {**state, "errors": state.get("errors", []) + [f"Packaging failed: {exc}"]}

        return {
            **state,
            "oci_image_uri": result.image_uri,
            "oci_image_digest": result.image_digest,
            "current_stage": "packaging",
            "agent_decisions": state.get("agent_decisions", []) + [{
                "node": "orchestrator.packaging",
                "image_uri": result.image_uri,
                "cosign_signed": result.cosign_signed,
            }],
        }

    @mlflow.trace(name="orchestrator.promotion")
    def _promotion_node(self, state: dict) -> dict:
        """Synchronous LangGraph entry point for Production promotion."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._promotion_async(state))

    async def _promotion_async(self, state: dict) -> dict:
        """
        Staging → Production via PromotionStateMachine — runs all five gates
        (evaluation, HITL, security, governance/OPA, OCI) and is the ONLY
        place a model may reach Production. See
        agents/registry/promotion_state_machine.py.
        """
        task_id: str = state.get("workflow_id", "unknown")
        model_name = state.get("model_name")
        model_version = state.get("model_version")

        if not model_name or not model_version:
            return {**state, "errors": state.get("errors", []) + [
                "Promotion blocked: no registered model_name/model_version in state"
            ]}

        try:
            sm = PromotionStateMachine(pool=self._pool)
            result = await sm.promote(PromotionRequest(
                model_name=model_name,
                model_version=model_version,
                target_stage=ModelStage.PRODUCTION,
                triggered_by=state.get("approved_by") or "system",
                trigger_type="human" if state.get("approved_by") else "automatic",
                workflow_id=task_id,
            ))
        except PromotionGateError as exc:
            logger.error("Promotion blocked for %s v%s: gate=%s reason=%s",
                         model_name, model_version, exc.gate.value, exc.reason)
            return {**state, "errors": state.get("errors", []) + [
                f"Promotion blocked: gate={exc.gate.value} reason={exc.reason}"
            ]}
        except Exception as exc:
            logger.exception("Promotion failed for workflow %s", task_id)
            return {**state, "errors": state.get("errors", []) + [f"Promotion failed: {exc}"]}

        return {
            **state,
            "current_stage": "promotion",
            "agent_decisions": state.get("agent_decisions", []) + [{
                "node": "orchestrator.promotion", "gates_passed": result.gates_passed,
            }],
        }

    @mlflow.trace(name="orchestrator.deployment")
    def _deployment_node(self, state: dict) -> dict:
        result: AgentTaskResult = self._deployment_agent.run(state)
        updated = {
            **state,
            "agent_decisions": state.get("agent_decisions", []) + [result.model_dump()],
            "current_stage": "deployment",
        }
        if result.status in ("failed", "partial"):
            updated["errors"] = state.get("errors", []) + [result.error or "Deployment failed"]
        return updated

    @mlflow.trace(name="orchestrator.monitoring")
    def _monitoring_node(self, state: dict) -> dict:
        result: AgentTaskResult = self._monitoring_agent.run(state)
        return {
            **state,
            "agent_decisions": state.get("agent_decisions", []) + [result.model_dump()],
            "current_stage": "monitoring",
        }

    @mlflow.trace(name="orchestrator.governance_audit")
    def _governance_audit_node(self, state: dict) -> dict:
        result: AgentTaskResult = self._governance_agent.audit(state)
        return {
            **state,
            "agent_decisions": state.get("agent_decisions", []) + [result.model_dump()],
            "current_stage": "done",
        }

    def _error_handler_node(self, state: dict) -> dict:
        errors = state.get("errors", [])
        logger.error("Workflow %s failed: %s", state.get("workflow_id"), errors)
        return {**state, "current_stage": "error"}

    # Routing functions
    def _route_after_plan(self, state: dict) -> str:
        if state.get("errors"):
            return "error"
        return "needs_approval" if state.get("current_stage") == "plan_pending_approval" else "ok"

    def _route_after_plan_approval(self, state: dict) -> str:
        return "approved" if state.get("plan_approval_decision") == "approved" else "rejected"

    def _route_after_security(self, state: dict) -> str:
        return "pass" if state.get("_security_passed") else "fail"

    def _route_after_data(self, state: dict) -> str:
        return "error" if state.get("errors") else "ok"

    def _route_after_training(self, state: dict) -> str:
        return "error" if state.get("errors") else "ok"

    def _route_after_security_scan(self, state: dict) -> str:
        return "error" if state.get("errors") else "ok"

    def _route_after_evaluation(self, state: dict) -> str:
        if state.get("errors"):
            return "error"
        return "approved" if state.get("_eval_approved") else "rejected"

    def _route_after_registration(self, state: dict) -> str:
        return "error" if state.get("errors") else "ok"

    def _route_after_approval(self, state: dict) -> str:
        return "approved" if state.get("approval_decision") == "approved" else "rejected"

    def _route_after_packaging(self, state: dict) -> str:
        return "error" if state.get("errors") else "ok"

    def _route_after_promotion(self, state: dict) -> str:
        return "error" if state.get("errors") else "ok"

    def _route_after_deployment(self, state: dict) -> str:
        return "error" if state.get("errors") else "ok"

    # Public API
    async def run_workflow(
        self,
        dataset_uri: str,
        model_uri: str = "",
        model_name: str | None = None,
        workflow_id: str | None = None,
        source: str = "user",
        trigger_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Launch a full MLOps pipeline workflow.

        source/trigger_type distinguish a manually-launched workflow from
        one process_pending_triggers() launched off a MonitoringAgent
        drift/accuracy alert (Phase 5 closed-loop retraining, plan §5.3).
        Both paths run through the identical graph, including the human-
        approval gate before any deployment — nothing about auto-triggered
        retraining skips that gate.

        Returns the final state dict with metrics, model_uri, and audit trail.
        """
        await self._ensure_connected()

        wf_id = workflow_id or str(uuid.uuid4())

        with _tracer.start_as_current_span("orchestrator.run_workflow") as span:
            span.set_attribute("workflow.id", wf_id)
            span.set_attribute("workflow.source", source)
            if trigger_type:
                span.set_attribute("workflow.trigger_type", trigger_type)

            initial_state = make_initial_state(
                workflow_id=wf_id,
                dataset_uri=dataset_uri,
                model_uri=model_uri,
            )
            initial_state["model_name"] = model_name or settings.mlflow_registered_model_name
            config = {"configurable": {"thread_id": wf_id}}

            await self._upsert_workflow_row(wf_id, initial_state, source, trigger_type)

            logger.info("Starting workflow %s (source=%s trigger_type=%s)", wf_id, source, trigger_type)
            final_state: dict = await self._graph.ainvoke(initial_state, config=config)
            final_state = await self._reflect_pending_approval(config, final_state)
            logger.info(
                "Workflow %s finished at stage=%s errors=%s awaiting_approval=%s",
                wf_id,
                final_state.get("current_stage"),
                final_state.get("errors"),
                final_state.get("awaiting_approval"),
            )
            span.set_attribute("workflow.final_stage", final_state.get("current_stage") or "")
            span.set_attribute("workflow.awaiting_approval", bool(final_state.get("awaiting_approval")))
            await self._upsert_workflow_row(wf_id, final_state, source, trigger_type)
            return final_state

    async def resume_workflow(
        self, workflow_id: str, approved: bool, approved_by: str = "unknown",
    ) -> dict[str, Any]:
        """
        Resume a workflow paused at either interrupt point
        (interrupt_before=["plan_approval", "human_approval"], set at
        compile time). Which gate is pending is read from the graph's own
        state — plan_approval and human_approval use distinct decision
        fields (plan_approval_decision vs approval_decision) so resolving
        one gate can never be misread as auto-approving the other.

        ainvoke() with a partial dict against an interrupted thread_id does
        NOT resume it on this langgraph version — it starts a new
        super-step from that dict as fresh input rather than continuing the
        paused one (verified against a real checkpointer while building
        Phase 5). The working pattern is update_state() then ainvoke(None,
        ...), which is what's used below.
        """
        await self._ensure_connected()
        config = {"configurable": {"thread_id": workflow_id}}

        snapshot = await self._graph.aget_state(config)
        pending_gate = "plan_approval" if snapshot.next and "plan_approval" in snapshot.next else "human_approval"

        if pending_gate == "plan_approval":
            patch = {
                "plan_approval_decision": "approved" if approved else "rejected",
                "plan_approved_by": approved_by,
            }
        else:
            patch = {
                "approval_decision": "approved" if approved else "rejected",
                "approved_by": approved_by,
            }

        await self._graph.aupdate_state(config, patch)
        final_state = await self._graph.ainvoke(None, config=config)
        final_state = await self._reflect_pending_approval(config, final_state)
        await self._upsert_workflow_row(workflow_id, final_state, None, None)
        return final_state

    async def _reflect_pending_approval(self, config: dict, state: dict) -> dict:
        """
        ainvoke() returns as soon as the graph hits interrupt_before — the
        approval node itself never ran, so state still shows
        awaiting_approval=False from make_initial_state()/the previous
        resume. Check whether the graph is actually paused right before
        plan_approval or human_approval and, if so, set the fields the
        API/UI/DB need to reflect that a human decision is pending.
        """
        try:
            snapshot = await self._graph.aget_state(config)
        except Exception as exc:
            logger.warning("aget_state failed (no checkpointer configured?): %s", exc)
            return state

        if snapshot.next and "plan_approval" in snapshot.next:
            return {
                **state,
                "awaiting_approval": True,
                "approval_context": {
                    "transition": "plan_review",
                    "plan_db_id": state.get("plan_db_id"),
                    "plan_summary": state.get("plan_summary"),
                },
            }
        if snapshot.next and "human_approval" in snapshot.next:
            return {
                **state,
                "awaiting_approval": True,
                "approval_context": {
                    "transition": "staging_to_production",
                    "model_name": state.get("model_name"),
                    "model_version": state.get("model_version"),
                    "model_uri": state.get("model_uri"),
                    "eval_metrics": state.get("metrics"),
                },
            }
        return state

    async def _upsert_workflow_row(
        self, workflow_id: str, state: dict, source: Optional[str], trigger_type: Optional[str],
    ) -> None:
        """
        Persist workflow state to the `workflows` table (designed in
        migration 0001, but previously never written to — the orchestrator
        kept everything in an in-process dict and the LangGraph
        checkpointer's opaque internal tables). This is what lets
        workflow_triggers.launched_workflow_id and any SQL-level "list
        workflows" query mean something.
        """
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                if source is not None:
                    await conn.execute(
                        """
                        INSERT INTO workflows (
                            id, status, current_stage, dataset_uri, model_uri,
                            metrics, errors, awaiting_approval, source, trigger_type
                        ) VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb,$8,$9,$10)
                        ON CONFLICT (id) DO UPDATE SET
                            status=EXCLUDED.status, current_stage=EXCLUDED.current_stage,
                            model_uri=EXCLUDED.model_uri, metrics=EXCLUDED.metrics,
                            errors=EXCLUDED.errors, awaiting_approval=EXCLUDED.awaiting_approval,
                            updated_at=NOW()
                        """,
                        workflow_id,
                        "error" if state.get("errors") else (state.get("current_stage") or "running"),
                        state.get("current_stage"),
                        state.get("dataset_uri"),
                        state.get("model_uri"),
                        json.dumps(state.get("metrics", {}), default=str),
                        json.dumps(state.get("errors", [])),
                        bool(state.get("awaiting_approval", False)),
                        source,
                        trigger_type,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE workflows SET
                            status=$2, current_stage=$3, model_uri=$4,
                            metrics=$5::jsonb, errors=$6::jsonb,
                            awaiting_approval=$7, updated_at=NOW()
                        WHERE id=$1
                        """,
                        workflow_id,
                        "error" if state.get("errors") else (state.get("current_stage") or "running"),
                        state.get("current_stage"),
                        state.get("model_uri"),
                        json.dumps(state.get("metrics", {}), default=str),
                        json.dumps(state.get("errors", [])),
                        bool(state.get("awaiting_approval", False)),
                    )
        except Exception as exc:
            logger.warning("Failed to persist workflows row for %s (non-fatal): %s", workflow_id, exc)

        # Phase 6 §6.3 real-time stream — mcp_servers/api_server.py's
        # /ws/events endpoint relays this straight to connected UI clients.
        await publish_workflow_status(
            workflow_id,
            status="error" if state.get("errors") else (state.get("current_stage") or "running"),
            current_stage=state.get("current_stage"),
            awaiting_approval=bool(state.get("awaiting_approval", False)),
            metrics=state.get("metrics", {}),
            errors=state.get("errors", []),
        )

    async def list_workflows(self, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """
        Real backing for the Phase 6 Command Center's workflow stream — the
        `workflows` table _upsert_workflow_row writes to on every state
        change. Previously nothing read this table at all: the V1 UI's
        in-memory `_workflows` dict in api_server.py was the only source,
        which meant a page reload or a second browser tab saw nothing.

        Real offset-based pagination (not a client-side slice over a
        capped fetch) — total_count comes from a real COUNT(*), so the UI
        can show "page N of M" and disable Next past the real last page
        instead of guessing from a fixed batch size.
        """
        await self._ensure_connected()
        if self._pool is None:
            return {"workflows": [], "total_count": 0, "running_count": 0, "awaiting_approval_count": 0}
        async with self._pool.acquire() as conn:
            total_count = await conn.fetchval("SELECT COUNT(*) FROM workflows")
            # Real global aggregates — the paginated `workflows` list below
            # is only the current page, so the Command Center's summary
            # stat cards can't derive "how many are running right now"
            # from it once pagination is in play; these two counts cover
            # the whole table regardless of which page is being viewed.
            agg = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status = 'running') AS running_count,
                    COUNT(*) FILTER (WHERE awaiting_approval) AS awaiting_approval_count
                FROM workflows
                """
            )
            rows = await conn.fetch(
                """
                SELECT id, status, current_stage, dataset_uri, model_uri,
                       metrics, errors, awaiting_approval, source, trigger_type,
                       created_at, updated_at
                FROM workflows ORDER BY updated_at DESC LIMIT $1 OFFSET $2
                """,
                limit, offset,
            )
        workflows = [
            {
                "workflow_id": r["id"],
                "status": r["status"],
                "current_stage": r["current_stage"],
                "dataset_uri": r["dataset_uri"],
                "model_uri": r["model_uri"],
                "metrics": json.loads(r["metrics"]) if isinstance(r["metrics"], str) else (r["metrics"] or {}),
                "errors": json.loads(r["errors"]) if isinstance(r["errors"], str) else (r["errors"] or []),
                "awaiting_approval": r["awaiting_approval"],
                "source": r["source"],
                "trigger_type": r["trigger_type"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
            }
            for r in rows
        ]
        return {
            "workflows": workflows,
            "total_count": total_count or 0,
            "running_count": (agg["running_count"] if agg else 0) or 0,
            "awaiting_approval_count": (agg["awaiting_approval_count"] if agg else 0) or 0,
        }

    # Closed-loop retraining (Phase 5 §5.3)
    async def process_pending_triggers(self, limit: int = 5) -> list[dict[str, Any]]:
        """
        Poll workflow_triggers for 'pending' rows written by MonitoringAgent
        (or a manual trigger) and launch a real workflow for each — the
        durable, pollable equivalent of the plan's "internal event queue".
        Meant to be called periodically by scripts/process_triggers.py (a
        K8s CronJob in production).

        Each launched workflow runs through the exact same graph a manual
        run does, so the human-approval gate before deployment still applies
        even for auto-triggered retraining.
        """
        await self._ensure_connected()
        if self._pool is None:
            logger.warning("process_pending_triggers: no DB pool — nothing to do")
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                UPDATE workflow_triggers SET status='processing', processed_at=NOW()
                WHERE id IN (
                    SELECT id FROM workflow_triggers
                    WHERE status='pending' ORDER BY created_at ASC LIMIT $1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id, trigger_type, model_name, model_version, dataset_uri, reason
                """,
                limit,
            )

        results: list[dict[str, Any]] = []
        for row in rows:
            trigger_id = row["id"]
            wf_id = str(uuid.uuid4())
            try:
                logger.warning(
                    "Processing retraining trigger id=%s model=%s reason=%s",
                    trigger_id, row["model_name"], row["reason"],
                )
                final_state = await self.run_workflow(
                    dataset_uri=row["dataset_uri"] or "",
                    model_name=row["model_name"],
                    workflow_id=wf_id,
                    source="monitoring",
                    trigger_type=row["trigger_type"],
                )
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE workflow_triggers
                        SET status='completed', launched_workflow_id=$2
                        WHERE id=$1
                        """,
                        trigger_id, wf_id,
                    )
                results.append({"trigger_id": trigger_id, "workflow_id": wf_id, "status": "completed"})
            except Exception as exc:
                logger.exception("Trigger %s failed to launch workflow", trigger_id)
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE workflow_triggers
                        SET status='failed', error_message=$2, launched_workflow_id=$3
                        WHERE id=$1
                        """,
                        trigger_id, str(exc)[:1000], wf_id,
                    )
                results.append({"trigger_id": trigger_id, "workflow_id": wf_id, "status": "failed", "error": str(exc)})

        return results
