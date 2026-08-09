"""
agents package – shared state schema and base classes used by every agent.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional
import operator

from langgraph.graph import StateGraph, END  # noqa: F401  (re-exported for convenience)
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Shared state that flows through the entire LangGraph workflow
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(dict):
    """
    Top-level workflow state shared across all LangGraph nodes.

    Fields use Annotated[list, operator.add] so that parallel sub-graphs
    can append to them safely without overwriting each other.
    """
    workflow_id: str
    current_stage: Literal["idle", "data", "training", "evaluation", "deployment", "monitoring", "done", "error"]
    model_uri: Optional[str]
    dataset_uri: Optional[str]
    metrics: dict[str, Any]
    errors: Annotated[list[str], operator.add]
    trace_id: str                            # MLflow trace ID for observability
    agent_decisions: Annotated[list[dict], operator.add]   # audit trail
    rl_recommendations: Optional[dict]       # suggestions from RL agent
    awaiting_approval: bool                  # Human-in-the-loop gate
    approval_context: Optional[dict]
    iteration_count: int
    max_iterations: int


# ─────────────────────────────────────────────────────────────────────────────
# Typed inter-agent message  (A2A protocol payload)
# ─────────────────────────────────────────────────────────────────────────────

class AgentMessage(BaseModel):
    sender: str
    recipient: str
    task_id: str
    payload: dict[str, Any]
    message_type: Literal["task_request", "task_result", "error", "clarification"]
    priority: int = 0


class AgentTaskResult(BaseModel):
    task_id: str
    status: Literal["success", "partial", "failed", "needs_clarification"]
    output: Optional[dict] = None
    error: Optional[str] = None
    confidence: float = 1.0
    next_actions: list[str] = []


def make_initial_state(
    workflow_id: str,
    dataset_uri: str = "",
    model_uri: str = "",
    trace_id: str = "",
) -> dict:
    """Factory to create a clean initial AgentState dict."""
    return {
        "workflow_id": workflow_id,
        "current_stage": "idle",
        "model_uri": model_uri or None,
        "dataset_uri": dataset_uri or None,
        "metrics": {},
        "errors": [],
        "trace_id": trace_id,
        "agent_decisions": [],
        "rl_recommendations": None,
        "awaiting_approval": False,
        "approval_context": None,
        "iteration_count": 0,
        "max_iterations": 20,
    }
