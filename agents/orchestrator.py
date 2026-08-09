"""
Orchestrator Agent

The root node of the LangGraph workflow.

Responsibilities
- Discover specialist agents via the A2A registry
- Decompose incoming requests into a phased execution plan
- Route control between Data → Training → Evaluation → Deployment → Monitoring
- Handle human-in-the-loop gates before high-risk transitions
- Surface errors and initiate rollback / retry sequences

Pattern: Orchestrator-Worker (see agentic-systems.md § Multi-Agent Coordination)
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import mlflow
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage, AIMessage

from agents import AgentState, AgentTaskResult, make_initial_state
from agents.data_agent import DataAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.deployment_agent import DeploymentAgent
from agents.monitoring_agent import MonitoringAgent
from agents.governance_agent import GovernanceAgent
from agents.security_agent import SecurityAgent
from configs.a2a_registry.registry import A2ARegistry
from configs.settings import settings
from mcp_servers.mlflow_server import log_agent_decision

logger = logging.getLogger(__name__)
          
# High-risk transitions that require human approval before proceeding                                                                            
HIGH_RISK_TRANSITIONS: set[str] = {
    "training_to_deployment",
    "canary_to_full_rollout",
    "initiate_rollback",
}


class OrchestratorAgent:
    """
    Central coordinator for the multi-agent MLOps pipeline.

    Builds a LangGraph StateGraph where each node delegates to a
    specialist agent.  Checkpoints via Neon PostgreSQL so interrupted
    workflows can be resumed automatically.
    """

    def __init__(self, db_conn_string: str = settings.database_url) -> None:
        self.registry = A2ARegistry()
        self._data_agent = DataAgent()
        self._training_agent = TrainingAgent()
        self._evaluation_agent = EvaluationAgent()
        self._deployment_agent = DeploymentAgent()
        self._monitoring_agent = MonitoringAgent()
        self._governance_agent = GovernanceAgent()
        self._security_agent = SecurityAgent()

        self._graph = self._build_graph()
        # Persist checkpoints in Neon so the workflow survives VM restarts
        if db_conn_string:
            try:
                checkpointer = PostgresSaver.from_conn_string(db_conn_string)
                self._graph = self._graph.compile(checkpointer=checkpointer)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Checkpointer unavailable, running stateless: %s", exc)
                self._graph = self._graph.compile()
        else:
            self._graph = self._graph.compile()

    # Graph construction                                                     
    def _build_graph(self) -> StateGraph:
        g = StateGraph(dict)

        # Nodes
        g.add_node("plan", self._plan_node)
        g.add_node("security_gate", self._security_gate_node)
        g.add_node("data", self._data_node)
        g.add_node("training", self._training_node)
        g.add_node("evaluation", self._evaluation_node)
        g.add_node("human_approval", self._human_approval_node)
        g.add_node("deployment", self._deployment_node)
        g.add_node("monitoring", self._monitoring_node)
        g.add_node("governance_audit", self._governance_audit_node)
        g.add_node("error_handler", self._error_handler_node)

        # Entry point
        g.set_entry_point("plan")

        # Edges
        g.add_edge("plan", "security_gate")
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
            {"ok": "evaluation", "error": "error_handler"},
        )
        g.add_conditional_edges(
            "evaluation",
            self._route_after_evaluation,
            {
                "approved": "human_approval",
                "rejected": "training",   # re-train with RL suggestions
                "error": "error_handler",
            },
        )
        g.add_conditional_edges(
            "human_approval",
            self._route_after_approval,
            {"approved": "deployment", "rejected": "error_handler"},
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
        Generate the execution plan and log it as an MLflow run.
        Consults the RL recommendations if available.
        """
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run(run_name=f"workflow-{state.get('workflow_id', 'unknown')}"):
            mlflow.log_param("workflow_id", state.get("workflow_id"))
            mlflow.log_param("dataset_uri", state.get("dataset_uri"))
            mlflow.set_tag("stage", "planning")

        plan = {
            "phases": ["data", "training", "evaluation", "deployment", "monitoring"],
            "rl_adjustments": state.get("rl_recommendations") or {},
        }
        decision = {
            "node": "orchestrator.plan",
            "action": "created_execution_plan",
            "plan": plan,
        }
        log_agent_decision(decision)
        return {**state, "current_stage": "data", "agent_decisions": state.get("agent_decisions", []) + [decision]}

    @mlflow.trace(name="orchestrator.security_gate")
    def _security_gate_node(self, state: dict) -> dict:
        """Pre-flight security check before any data is touched."""
        result: AgentTaskResult = self._security_agent.pre_flight_check(state)
        return {
            **state,
            "agent_decisions": state.get("agent_decisions", []) + [result.model_dump()],
            "_security_passed": result.status == "success",
        }

    @mlflow.trace(name="orchestrator.data")
    def _data_node(self, state: dict) -> dict:
        result: AgentTaskResult = self._data_agent.run(state)
        updated = {
            **state,
            "metrics": {**state.get("metrics", {}), **(result.output or {})},
            "agent_decisions": state.get("agent_decisions", []) + [result.model_dump()],
            "current_stage": "data",
        }
        if result.status in ("failed", "partial"):
            updated["errors"] = state.get("errors", []) + [result.error or "Data stage failed"]
        return updated

    @mlflow.trace(name="orchestrator.training")
    def _training_node(self, state: dict) -> dict:
        result: AgentTaskResult = self._training_agent.run(state)
        updated = {
            **state,
            "model_uri": (result.output or {}).get("model_uri", state.get("model_uri")),
            "metrics": {**state.get("metrics", {}), **(result.output or {}).get("metrics", {})},
            "agent_decisions": state.get("agent_decisions", []) + [result.model_dump()],
            "current_stage": "training",
        }
        if result.status in ("failed", "partial"):
            updated["errors"] = state.get("errors", []) + [result.error or "Training failed"]
        return updated

    @mlflow.trace(name="orchestrator.evaluation")
    def _evaluation_node(self, state: dict) -> dict:
        result: AgentTaskResult = self._evaluation_agent.run(state)
        updated = {
            **state,
            "metrics": {**state.get("metrics", {}), **(result.output or {}).get("eval_metrics", {})},
            "agent_decisions": state.get("agent_decisions", []) + [result.model_dump()],
            "current_stage": "evaluation",
            "_eval_approved": result.status == "success" and (result.output or {}).get("approved", False),
        }
        return updated

    def _human_approval_node(self, state: dict) -> dict:
        """
        Pause execution and wait for human sign-off on production deployment.
        In production this suspends the LangGraph thread; the UI resumes it
        by calling /api/workflows/{id}/approve.
        """
        return {
            **state,
            "awaiting_approval": True,
            "approval_context": {
                "transition": "training_to_deployment",
                "model_uri": state.get("model_uri"),
                "eval_metrics": state.get("metrics"),
            },
            "agent_decisions": state.get("agent_decisions", []) + [{
                "node": "human_approval",
                "action": "awaiting_human_approval",
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
    def _route_after_security(self, state: dict) -> str:
        return "pass" if state.get("_security_passed") else "fail"

    def _route_after_data(self, state: dict) -> str:
        return "error" if state.get("errors") else "ok"

    def _route_after_training(self, state: dict) -> str:
        return "error" if state.get("errors") else "ok"

    def _route_after_evaluation(self, state: dict) -> str:
        if state.get("errors"):
            return "error"
        return "approved" if state.get("_eval_approved") else "rejected"

    def _route_after_approval(self, state: dict) -> str:
        # awaiting_approval=False means approval was granted (resumed by API)
        return "rejected" if state.get("awaiting_approval") else "approved"

    def _route_after_deployment(self, state: dict) -> str:
        return "error" if state.get("errors") else "ok"

    # Public API                                                              
    async def run_workflow(
        self,
        dataset_uri: str,
        model_uri: str = "",
        workflow_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Launch a full MLOps pipeline workflow.

        Returns the final state dict with metrics, model_uri, and audit trail.
        """
        wf_id = workflow_id or str(uuid.uuid4())
        initial_state = make_initial_state(
            workflow_id=wf_id,
            dataset_uri=dataset_uri,
            model_uri=model_uri,
        )
        config = {"configurable": {"thread_id": wf_id}}
        logger.info("Starting workflow %s", wf_id)
        final_state: dict = await self._graph.ainvoke(initial_state, config=config)
        logger.info(
            "Workflow %s finished at stage=%s errors=%s",
            wf_id,
            final_state.get("current_stage"),
            final_state.get("errors"),
        )
        return final_state

    async def resume_workflow(self, workflow_id: str, approved: bool) -> dict[str, Any]:
        """Resume a workflow that is paused at the human_approval gate."""
        config = {"configurable": {"thread_id": workflow_id}}
        patch = {"awaiting_approval": not approved}
        return await self._graph.ainvoke(patch, config=config)
