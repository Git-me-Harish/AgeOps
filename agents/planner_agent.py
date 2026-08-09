"""
Planner Agent
             
Responsibilities
- Analyse the incoming request and historical MLflow runs
- Integrate RL agent recommendations into a concrete execution plan
- Generate a priority-ordered task graph for the Orchestrator
- Surface optimisation suggestions based on past workflow patterns

This is a LangGraph conditional-edge node: it runs before Data Agent
and its output shapes how the Orchestrator routes subsequent stages.
"""
from __future__ import annotations

import logging
from typing import Any

import mlflow

from agents import AgentTaskResult
from configs.settings import settings
from rl_agent.rl_optimizer import predict_adjustments

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Generates execution plans and integrates RL-driven hyperparameter
    and workflow recommendations before the pipeline kicks off.
    """

    def __init__(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # Main entry point                                                         
    @mlflow.trace(name="planner_agent.run")
    def run(self, state: dict) -> AgentTaskResult:
        task_id = state.get("workflow_id", "unknown")

        try:
            with mlflow.start_run(run_name=f"plan-{task_id}", nested=True):
                mlflow.set_tag("agent", "planner_agent")

                # Step 1: pull historical performance data
                historical_summary = self._analyse_history()
                mlflow.log_metrics({
                    "historical_runs_analysed": historical_summary.get("run_count", 0),
                    "avg_historical_accuracy":  historical_summary.get("avg_accuracy", 0.0),
                })

                # Step 2: ask RL optimizer for hyperparameter tweaks
                rl_recs = self._get_rl_recommendations(state)
                mlflow.log_param("rl_recommendations", str(rl_recs))

                # Step 3: build the execution plan
                plan = self._build_plan(state, historical_summary, rl_recs)
                mlflow.log_param("execution_plan_phases", str(plan["phases"]))

                return AgentTaskResult(
                    task_id=task_id,
                    status="success",
                    output={
                        "execution_plan": plan,
                        "rl_recommendations": rl_recs,
                        "historical_summary": historical_summary,
                    },
                    next_actions=plan["phases"],
                )

        except Exception as exc:
            logger.exception("PlannerAgent failed for workflow %s", task_id)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))

    # Historical analysis                                                     
    @mlflow.trace(name="planner_agent.analyse_history")
    def _analyse_history(self) -> dict[str, Any]:
        """
        Query the last 20 MLflow runs to surface patterns
        (e.g., consistently low drift → can skip heavy drift checks).
        """
        try:
            df = mlflow.search_runs(
                experiment_names=[settings.mlflow_experiment_name],
                max_results=20,
                order_by=["start_time DESC"],
            )
            if df.empty:
                return {"run_count": 0, "avg_accuracy": 0.0, "avg_drift": 0.0}

            return {
                "run_count":    len(df),
                "avg_accuracy": float(df.get("metrics.accuracy", df.iloc[:, 0]).fillna(0).mean()),
                "avg_drift":    float(df.get("metrics.drift_score", df.iloc[:, 0]).fillna(0).mean()),
            }
        except Exception as exc:
            logger.warning("Historical analysis failed: %s", exc)
            return {"run_count": 0, "avg_accuracy": 0.0, "avg_drift": 0.0}

    # RL recommendations                                                      
    def _get_rl_recommendations(self, state: dict) -> dict[str, Any]:
        """
        Load the trained RL model (if available) and return
        recommended hyperparameter adjustments.
        """
        try:
            return predict_adjustments(state)
        except Exception as exc:
            logger.debug("RL recommendations unavailable: %s", exc)
            return {}

    # Plan construction                                                       
    def _build_plan(
        self,
        state: dict,
        history: dict[str, Any],
        rl_recs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Combine request context, historical patterns, and RL
        recommendations into a prioritised execution plan.
        """
        phases = ["data", "training", "evaluation", "deployment", "monitoring"]

        # If historical drift is consistently low, we can lighten the drift check
        skip_heavy_drift = history.get("avg_drift", 1.0) < 0.05

        # If historical accuracy is high and RL suggests no change, fast-track eval
        fast_track_eval = (
            history.get("avg_accuracy", 0.0) > 0.90
            and not rl_recs  # no changes suggested
        )

        optimisations: list[str] = []
        if skip_heavy_drift:
            optimisations.append("light_drift_check")
        if fast_track_eval:
            optimisations.append("fast_track_evaluation")

        return {
            "phases": phases,
            "optimisations": optimisations,
            "rl_adjustments": rl_recs,
            "priority": "normal",
            "estimated_duration_minutes": self._estimate_duration(rl_recs),
        }

    def _estimate_duration(self, rl_recs: dict) -> int:
        """Rough estimate of total pipeline duration in minutes."""
        base = 30  # baseline minutes
        if rl_recs.get("n_estimators", 100) > 150:
            base += 10   # more trees → longer training
        return base
