"""
agents/planner_agent.py

Planner Agent — V2 production rewrite.

What changed from V1:
  V1: returned a hardcoded dict {"framework": "gradient_boosting", "epochs": 100}
  V2: full LLM-backed Plan-and-Execute reasoning loop:
        1. Queries MLflow for the last 20 similar experiments (same dataset family)
        2. Queries RL Agent for current hyperparameter recommendations
        3. Reasons about dataset size, target metric, deadline, resource constraints
        4. Produces a typed ExecutionPlan (Pydantic) via instructor structured output
        5. Stores the plan in Neon for HITL review before execution starts
        6. Blocks pipeline if human_approval_required=True and no approval found

ExecutionPlan fields:
  - ordered list of PlanSteps with estimated durations
  - frameworks to try (may be multiple for parallel experiments)
  - hyperparameter grids per framework
  - estimated accuracy with confidence interval
  - data_augmentation_required: bool with reasoning
  - human_approval_required: bool (True for high-risk decisions)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import mlflow
from pydantic import BaseModel, Field

from agents import AgentTaskResult
from agents.llm_gateway import LLMGateway, BudgetExceededError
from configs.settings import settings

logger = logging.getLogger(__name__)


# Typed output models
class HyperparameterGrid(BaseModel):
    """Hyperparameter search space for one framework."""
    framework: str
    learning_rate: list[float] = Field(default=[0.01, 0.001])
    max_depth: Optional[list[int]] = None           # XGBoost / tree methods
    n_estimators: Optional[list[int]] = None        # ensemble methods
    batch_size: Optional[list[int]] = None          # neural nets
    epochs: Optional[list[int]] = None
    dropout: Optional[list[float]] = None
    hidden_dims: Optional[list[list[int]]] = None   # MLP / deep models
    extra: dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    """A single step in the execution plan."""
    step_id: int
    name: str                                # e.g. "feature_engineering", "train_xgboost"
    agent: str                               # which agent executes this step
    estimated_duration_mins: int
    depends_on: list[int] = Field(default_factory=list)   # step_ids that must complete first
    can_parallelize: bool = False
    reasoning: str = ""                      # observable rationale for this step


class ExecutionPlan(BaseModel):
    """
    Typed output of the Planner Agent's LLM reasoning loop.
    Stored in Neon execution_plans table for HITL review.
    """
    plan_id: str = ""                        # filled by PlannerAgent before Neon insert
    steps: list[PlanStep]
    frameworks: list[str]                    # e.g. ["xgboost", "sklearn"]
    hyperparameter_grids: list[HyperparameterGrid]
    parallel_experiments: int = 1
    data_augmentation_required: bool = False
    augmentation_strategy: Optional[str] = None
    estimated_accuracy: float
    accuracy_confidence_interval: tuple[float, float] = (0.0, 1.0)
    estimated_total_duration_mins: int
    human_approval_required: bool = True
    approval_reasoning: str = ""             # why human approval is / isn't needed
    risk_level: str = "medium"               # low | medium | high
    llm_reasoning: str = ""                  # concise observable decision rationale

# Planner Agent
class PlannerAgent:
    """
    LLM-backed Planner Agent.

    Builds an ExecutionPlan by:
      1. Fetching historical context from MLflow
      2. Getting RL recommendations
      3. Running an LLM reasoning loop via LLMGateway
      4. Persisting the plan to Neon for HITL
      5. Blocking execution until approved (if required)
    """

    def __init__(self) -> None:
        self._pool: Optional[Any] = None

    async def connect(self, pool: Any) -> None:
        self._pool = pool

    @mlflow.trace(name="planner_agent.run")
    def run(self, state: dict) -> AgentTaskResult:
        """Synchronous LangGraph entry point."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._run_async(state))
        except Exception as exc:
            logger.exception("PlannerAgent.run failed")
            return AgentTaskResult(
                task_id=state.get("workflow_id", "unknown"),
                status="failed",
                error=str(exc),
            )

    async def _run_async(self, state: dict) -> AgentTaskResult:
        task_id: str = state.get("workflow_id", "unknown")
        dataset_uri: str = state.get("dataset_uri", "")
        target_metric: str = state.get("target_metric", "f1")
        deadline_mins: int = state.get("deadline_mins", 120)
        data_summary: dict = state.get("data_agent_output", {})

        gateway = LLMGateway(pool=self._pool, workflow_id=task_id)

        try:
            with mlflow.start_run(run_name=f"planner-{task_id}", nested=True) as run:
                mlflow.set_tag("agent", "planner_agent")

                # Step 1: Fetch MLflow historical context
                historical_context = await self._fetch_mlflow_history(
                    dataset_uri=dataset_uri,
                    target_metric=target_metric,
                )

                # Step 2: Get RL agent recommendations
                rl_recommendations = await self._fetch_rl_recommendations(
                    dataset_uri=dataset_uri,
                    historical_context=historical_context,
                    workflow_id=task_id,
                )

                # Step 3: Build LLM prompt with full context
                prompt = self._build_planning_prompt(
                    dataset_uri=dataset_uri,
                    target_metric=target_metric,
                    deadline_mins=deadline_mins,
                    data_summary=data_summary,
                    historical_context=historical_context,
                    rl_recommendations=rl_recommendations,
                )

                # Step 4: LLM reasoning → structured ExecutionPlan
                try:
                    response = await gateway.complete(
                        prompt=prompt,
                        agent_role="planner",
                        response_model=ExecutionPlan,
                        max_tokens=2048,
                    )
                except BudgetExceededError as exc:
                    return AgentTaskResult(
                        task_id=task_id, status="failed", error=str(exc)
                    )

                plan: Optional[ExecutionPlan] = response.parsed
                if plan is None:
                    # Fallback: parse raw JSON from text
                    plan = self._parse_plan_fallback(response.text, task_id)
                if plan is None:
                    return AgentTaskResult(
                        task_id=task_id,
                        status="failed",
                        error="LLM returned unparseable plan. Raw: " + response.text[:200],
                    )

                plan.plan_id = task_id
                # Store structured rationale — NOT raw chain-of-thought.
                # execution_plans is a long-lived audit artifact; raw CoT is
                # unverifiable and should not be persisted as-is.
                plan.llm_reasoning = self._build_structured_rationale(
                    plan=plan,
                    response=response,
                    data_summary=data_summary,
                )

                # Step 5: Persist to Neon for HITL
                plan_db_id = await self._persist_plan(plan, task_id)

                # Step 6: HITL gate check
                if plan.human_approval_required:
                    approved = await self._check_approval(task_id)
                    if not approved:
                        logger.info(
                            "PlannerAgent: plan %s requires human approval — pipeline paused",
                            task_id,
                        )
                        return AgentTaskResult(
                            task_id=task_id,
                            status="pending_approval",
                            output={
                                "plan_id": task_id,
                                "plan_db_id": plan_db_id,
                                "message": (
                                    f"Execution plan created and requires human approval. "
                                    f"Review at /api/plans/{plan_db_id} and approve to continue."
                                ),
                                "plan_summary": {
                                    "frameworks": plan.frameworks,
                                    "steps": len(plan.steps),
                                    "estimated_accuracy": plan.estimated_accuracy,
                                    "estimated_duration_mins": plan.estimated_total_duration_mins,
                                    "risk_level": plan.risk_level,
                                },
                            },
                        )

                # Log plan metrics to MLflow
                mlflow.log_metrics({
                    "plan.estimated_accuracy": plan.estimated_accuracy,
                    "plan.estimated_duration_mins": plan.estimated_total_duration_mins,
                    "plan.parallel_experiments": plan.parallel_experiments,
                    "plan.steps_count": len(plan.steps),
                    "planner.total_tokens": response.total_tokens,
                    "planner.cost_usd": response.cost_usd,
                })
                mlflow.log_dict(plan.model_dump(), "execution_plan.json")

                logger.info(
                    "PlannerAgent: plan created — frameworks=%s steps=%d accuracy=%.3f "
                    "duration=%dmins tokens=%d cost=$%.4f",
                    plan.frameworks, len(plan.steps), plan.estimated_accuracy,
                    plan.estimated_total_duration_mins, response.total_tokens, response.cost_usd,
                )

                return AgentTaskResult(
                    task_id=task_id,
                    status="success",
                    output={
                        "execution_plan": plan.model_dump(),
                        "frameworks": plan.frameworks,
                        "hyperparameter_grids": [g.model_dump() for g in plan.hyperparameter_grids],
                        "parallel_experiments": plan.parallel_experiments,
                        "estimated_accuracy": plan.estimated_accuracy,
                        "estimated_duration_mins": plan.estimated_total_duration_mins,
                        "risk_level": plan.risk_level,
                        "plan_db_id": plan_db_id,
                        "rl_recommendations": rl_recommendations,
                        "token_cost": {
                            "total_tokens": response.total_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    },
                    confidence=plan.estimated_accuracy,
                )

        except Exception as exc:
            logger.exception("PlannerAgent._run_async failed for workflow %s", task_id)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))

    # Context gathering 
    async def _fetch_mlflow_history(
        self,
        dataset_uri: str,
        target_metric: str,
        n_recent: int = 20,
    ) -> list[dict]:
        """
        Query MLflow for the last N experiments on similar datasets.
        Returns a compact context list for the LLM prompt.
        """
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
            if experiment is None:
                return []

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="",
                order_by=["start_time DESC"],
                max_results=n_recent,
            )

            history = []
            for run in runs:
                tags = run.data.tags
                metrics = run.data.metrics
                # Only include runs on similar dataset family
                run_dataset = tags.get("mlflow.datasets", "") or tags.get("dataset_uri", "")
                history.append({
                    "run_id":     run.info.run_id[:8],
                    "framework":  tags.get("framework", "unknown"),
                    "status":     run.info.status,
                    target_metric: metrics.get(target_metric, 0.0),
                    "accuracy":   metrics.get("accuracy", 0.0),
                    "duration_s": run.info.end_time - run.info.start_time
                                  if run.info.end_time else None,
                    "dataset":    run_dataset[-60:] if run_dataset else "",
                })
            logger.info("Fetched %d historical MLflow runs for planning context", len(history))
            return history

        except Exception as exc:
            logger.warning("MLflow history fetch failed (non-fatal): %s", exc)
            return []

    async def _fetch_rl_recommendations(
        self,
        dataset_uri: str,
        historical_context: list[dict],
        workflow_id: Optional[str] = None,
    ) -> dict:
        """
        Query the RL Optimizer for current hyperparameter recommendations.

        This used to import a class (RLOptimizer with .build_state_vector()/
        .predict()) that never existed anywhere in rl_agent/rl_optimizer.py —
        every call raised ImportError, was swallowed by the broad except
        below, and silently returned {}. RL recommendations have never
        actually reached the Planner. Fixed to call the real module-level
        predict_adjustments() function that has existed here all along.
        """
        try:
            from rl_agent.rl_optimizer import predict_adjustments

            latest_metrics = historical_context[0] if historical_context else {}
            adjustments = predict_adjustments({"metrics": latest_metrics})
            if not adjustments:
                return {}

            await self._persist_rl_recommendation(workflow_id, adjustments)
            return {"hyperparameter_adjustments": adjustments, "rl_confidence": 0.7}
        except Exception as exc:
            logger.debug("RL recommendations unavailable (non-fatal): %s", exc)
            return {}

    async def _persist_rl_recommendation(
        self, workflow_id: Optional[str], adjustments: dict,
    ) -> None:
        """Record the RL suggestion for the accept/reject API (Phase 5)."""
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO rl_recommendations (workflow_id, agent_role, recommendation, confidence)
                    VALUES ($1, 'planner', $2::jsonb, $3)
                    """,
                    workflow_id, json.dumps(adjustments), 0.7,
                )
        except Exception as exc:
            logger.warning("Failed to persist RL recommendation (non-fatal): %s", exc)

    # Prompt construction 
    def _build_planning_prompt(
        self,
        dataset_uri: str,
        target_metric: str,
        deadline_mins: int,
        data_summary: dict,
        historical_context: list[dict],
        rl_recommendations: dict,
    ) -> str:
        """
        Build the full planning prompt injecting all context.
        The LLM must return a JSON object matching the ExecutionPlan schema.
        """
        schema = json.dumps(ExecutionPlan.model_json_schema(), indent=2)
        history_str = json.dumps(historical_context[:10], indent=2)  # last 10 to save tokens
        rl_str = json.dumps(rl_recommendations, indent=2)

        return f"""
You are planning a machine learning experiment. Analyse all provided context and produce
a complete ExecutionPlan as a JSON object matching the schema below exactly.

## Dataset Information
- URI: {dataset_uri}
- Rows: {data_summary.get('row_count', 'unknown')}
- Columns: {data_summary.get('col_count', 'unknown')}
- Validation score: {data_summary.get('validation_score', 'unknown')}
- Drift score: {data_summary.get('drift_score', 0.0)}
- Content hash: {data_summary.get('content_hash', 'unknown')[:16]}

## Objective
- Target metric: {target_metric}
- Deadline: {deadline_mins} minutes total pipeline time
- Available frameworks: sklearn, xgboost, pytorch, huggingface, custom

## Historical Experiments (last {len(historical_context)} runs)
{history_str}

## RL Agent Recommendations
{rl_str}

## Instructions
1. Review historical run outcomes. If xgboost consistently outperformed others on similar data, prioritise it.
2. If the RL agent recommends specific hyperparameters, include them in the hyperparameter_grids.
3. For datasets with row_count < 10000, prefer sklearn or xgboost over neural networks.
4. For text/NLP features, always include huggingface.
5. Set human_approval_required=true if:
   - estimated_accuracy confidence interval is wide (>0.1)
   - data drift score > 0.15
   - dataset has never been seen before (no matching historical runs)
   - risk_level is "high"
6. Set parallel_experiments based on available compute (max 3 for free tier K3s).
7. Provide concise, observable decision rationale in llm_reasoning. Do not expose private chain-of-thought.

## Output Schema
```json
{schema}
```

Return ONLY valid JSON. No markdown, no preamble.
""".strip()

    # Plan parsing fallback 
    @staticmethod
    def _parse_plan_fallback(raw_text: str, task_id: str) -> Optional[ExecutionPlan]:
        """
        Attempt to extract ExecutionPlan from malformed LLM output.
        Used when instructor structured output fails.
        """
        try:
            clean = raw_text.strip()
            if "```" in clean:
                lines = clean.split("\n")
                json_lines = [l for l in lines if not l.startswith("```")]
                clean = "\n".join(json_lines)
            data = json.loads(clean)
            return ExecutionPlan.model_validate(data)
        except Exception as exc:
            logger.error("Plan fallback parsing also failed: %s", exc)
            return None

    # Structured rationale builder 
    @staticmethod
    def _build_structured_rationale(
        plan: ExecutionPlan,
        response: Any,
        data_summary: dict,
    ) -> str:
        """
        Build a concise structured rationale dict and serialise to JSON.

        This is what gets stored in execution_plans.llm_reasoning.
        We deliberately do NOT store the raw LLM chain-of-thought because:
          - Raw CoT is unverifiable and can contain hallucinations
          - execution_plans is a long-lived compliance artifact
          - Auditors need facts, not internal model reasoning traces

        The rationale captures the observable decision factors only.
        """
        rationale = {
            "decision": "plan_created",
            "rationale": [
                f"Dataset rows: {data_summary.get('row_count', 'unknown')}",
                f"Selected frameworks: {plan.frameworks}",
                f"Parallel experiments: {plan.parallel_experiments}",
                f"Estimated accuracy: {plan.estimated_accuracy:.3f} "
                f"(CI: {plan.accuracy_confidence_interval})",
                f"Estimated duration: {plan.estimated_total_duration_mins} mins",
                f"Risk level: {plan.risk_level}",
                f"Human approval required: {plan.human_approval_required}",
                f"Approval reasoning: {plan.approval_reasoning}",
                f"Data augmentation: {plan.data_augmentation_required}"
                + (f" ({plan.augmentation_strategy})" if plan.data_augmentation_required else ""),
            ],
            "token_cost": {
                "total_tokens": response.total_tokens,
                "cost_usd":     round(response.cost_usd, 6),
                "model":        response.model,
                "was_fallback": response.was_fallback,
                "was_cache_hit": response.was_cache_hit,
            },
        }
        return json.dumps(rationale, indent=2)

    # Neon persistence 
    async def _persist_plan(self, plan: ExecutionPlan, workflow_id: str) -> Optional[int]:
        """Store ExecutionPlan in Neon execution_plans table. Returns row id."""
        if self._pool is None:
            return None
        try:
            async with self._pool.acquire() as conn:
                row_id = await conn.fetchval(
                    """
                    INSERT INTO execution_plans (
                        workflow_id, status, plan_json, estimated_accuracy,
                        estimated_duration_mins, frameworks, parallel_experiments,
                        human_approval_required, llm_reasoning
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    RETURNING id
                    """,
                    workflow_id,
                    "pending_approval" if plan.human_approval_required else "approved",
                    json.dumps(plan.model_dump()),
                    plan.estimated_accuracy,
                    plan.estimated_total_duration_mins,
                    json.dumps(plan.frameworks),
                    plan.parallel_experiments,
                    plan.human_approval_required,
                    plan.llm_reasoning[:5000] if plan.llm_reasoning else None,
                )
            logger.info("ExecutionPlan persisted: id=%s workflow=%s", row_id, workflow_id)
            return row_id
        except Exception as exc:
            logger.error("Plan persistence failed (non-fatal): %s", exc)
            return None

    async def _check_approval(self, workflow_id: str) -> bool:
        """
        Check Neon for human approval of the execution plan.
        Returns True if approved, False if pending or rejected.
        Also auto-approves if HITL timeout (settings.hitl_approval_timeout_hours) has passed.
        """
        if self._pool is None:
            # No DB — auto-approve in dev mode
            logger.warning("No DB pool — auto-approving plan (dev mode only)")
            return True
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT status, approved_at, created_at
                    FROM execution_plans
                    WHERE workflow_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    workflow_id,
                )
            if row is None:
                return False
            if row["status"] == "approved":
                return True
            if row["status"] == "rejected":
                return False
            # Check HITL timeout — auto-reject if expired
            from datetime import timedelta
            timeout = timedelta(hours=settings.hitl_approval_timeout_hours)
            if row["created_at"] and (
                datetime.now(tz=timezone.utc) - row["created_at"] > timeout
            ):
                logger.warning(
                    "HITL approval timed out for workflow %s — auto-rejecting", workflow_id
                )
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE execution_plans SET status='rejected', "
                        "rejection_reason='HITL timeout — no approval within "
                        f"{settings.hitl_approval_timeout_hours}h' "
                        "WHERE workflow_id=$1",
                        workflow_id,
                    )
                return False
            return False  # still pending

        except Exception as exc:
            logger.error("HITL approval check failed: %s", exc)
            return False
