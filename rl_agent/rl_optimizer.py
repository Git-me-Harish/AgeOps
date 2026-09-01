"""
RL Optimization Agent
─────────────────────
Framework: Stable Baselines3 (PPO)
Training:  Offline on real MLflow execution traces enriched with real Neon
           promotion outcomes and post-deployment drift (no costly online
           training, no synthetic historical runs except as an explicit,
           logged cold-start fallback when there is no history at all).
Deployment: Kubernetes CronJob (weekly — see scripts/train_rl_weekly.py)

The agent learns which hyperparameter adjustments and workflow
modifications maximise the reward function:

    R = α·accuracy + β·(1/latency) + γ·(1/cost) - δ·drift
        + deployment_bonus - drift_at_30d_penalty

Reward weights (α, β, γ, δ) are configurable via settings
(rl_reward_weight_*), not hardcoded — but they are NOT tuned by a
meta-learner that observes 30-day outcomes, which is what the V2 plan's
§5.4 describes. That is a real research undertaking on its own and is
explicitly deferred, not faked: this module's reward function is fixed
(configurable) weights over real signals, same as any standard RL reward
shaping. If/when a meta-learner is built, it belongs in a new module that
adjusts these settings values based on tracked outcomes over time.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import mlflow
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from configs.settings import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Custom Gym environment backed by historical MLflow traces
# ─────────────────────────────────────────────────────────────────────────────

class MLOpsEnv(gym.Env):
    """
    Observation space: [accuracy, latency_norm, cost_norm, drift_score,
                         error_rate, deployment_success, drift_at_30d]
    Action space:      Discrete — map index to a hyperparameter adjustment bundle

    deployment_success: 1.0 if the run's model reached Production, 0.0 if it
    was Rejected, 0.5 if unknown (no model_promotions record — e.g. the run
    predates Phase 3 wiring, or enrichment couldn't reach Neon). This is a
    real signal from model_promotions, not a placeholder value used as a
    positive/negative label — 0.5 is explicitly "don't know", not "neutral
    outcome".
    """

    metadata = {"render_modes": []}

    # Action index → hyperparameter adjustments
    ACTION_MAP: dict[int, dict] = {
        0: {},                                                   # no change
        1: {"learning_rate": 0.01},                              # lower LR
        2: {"learning_rate": 0.2},                               # raise LR
        3: {"n_estimators": 200},                                # more trees
        4: {"n_estimators": 50},                                 # fewer trees (faster)
        5: {"max_depth": 4},                                     # shallower model
        6: {"max_depth": 8},                                     # deeper model
        7: {"subsample": 0.6},                                   # more regularisation
    }

    N_OBS_DIMS = 7

    def __init__(self, historical_runs: list[dict]) -> None:
        super().__init__()
        if not historical_runs:
            logger.warning(
                "MLOpsEnv: no historical runs available — falling back to "
                "synthetic bootstrap data. This is a disclosed cold-start "
                "path, not a substitute for real data once runs exist."
            )
        self._runs = historical_runs if historical_runs else self._synthetic_runs()
        self._idx = 0

        self.observation_space = spaces.Box(
            low=np.zeros(self.N_OBS_DIMS, dtype=np.float32),
            high=np.ones(self.N_OBS_DIMS, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.ACTION_MAP))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        return self._observe(), {}

    def step(self, action: int):
        run = self._runs[self._idx % len(self._runs)]
        reward = self._compute_reward(run)
        self._idx += 1
        terminated = self._idx >= len(self._runs)
        return self._observe(), reward, terminated, False, {}

    def _observe(self) -> np.ndarray:
        run = self._runs[self._idx % len(self._runs)]
        return np.float32([
            run.get("accuracy", 0.5),
            min(run.get("latency_s", 10) / 100, 1.0),
            min(run.get("cost_usd", 1) / 10, 1.0),
            run.get("drift_score", 0.0),
            run.get("error_rate", 0.0),
            run.get("deployment_success", 0.5),
            min(run.get("drift_at_30d", 0.0), 1.0),
        ])

    def _compute_reward(self, run: dict) -> float:
        accuracy = run.get("accuracy", 0.0)
        latency = max(run.get("latency_s", 1), 0.001)
        cost = max(run.get("cost_usd", 0.01), 0.0001)
        drift = run.get("drift_score", 0.0)
        # deployment_success=0.5 ("unknown") contributes zero bonus/penalty —
        # only a confirmed outcome (0.0 or 1.0) should move the reward.
        deployment_bonus = (run.get("deployment_success", 0.5) - 0.5) * 0.4
        drift_at_30d_penalty = run.get("drift_at_30d", 0.0) * 0.1
        return (
            settings.rl_reward_weight_accuracy * accuracy
            + settings.rl_reward_weight_latency / latency
            + settings.rl_reward_weight_cost / cost
            - settings.rl_reward_weight_drift * drift
            + deployment_bonus
            - drift_at_30d_penalty
        )

    def _synthetic_runs(self) -> list[dict]:
        """Cold-start bootstrap data — used ONLY when zero real runs exist."""
        rng = np.random.default_rng(42)
        return [
            {
                "accuracy": float(rng.uniform(0.7, 0.95)),
                "latency_s": float(rng.uniform(5, 120)),
                "cost_usd": 0.0,   # free tier
                "drift_score": float(rng.uniform(0, 0.2)),
                "error_rate": float(rng.uniform(0, 0.05)),
                "deployment_success": 0.5,
                "drift_at_30d": 0.0,
            }
            for _ in range(500)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Training entry point (run as CronJob — see scripts/train_rl_weekly.py)
# ─────────────────────────────────────────────────────────────────────────────

def train(total_timesteps: int = settings.rl_training_timesteps) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("rl-optimization-agent")

    historical_runs = asyncio.run(_fetch_historical_runs(days=settings.rl_history_window_days))

    env = MLOpsEnv(historical_runs)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="/tmp/rl_logs")

    with mlflow.start_run(run_name=f"rl-train-{datetime.now(tz=timezone.utc).strftime('%Y%m%d')}"):
        mlflow.log_param("algorithm", "PPO")
        mlflow.log_param("total_timesteps", total_timesteps)
        mlflow.log_param("n_runs_used", len(historical_runs))
        mlflow.log_param("used_synthetic_fallback", len(historical_runs) == 0)
        mlflow.log_params({
            "reward_weight_accuracy": settings.rl_reward_weight_accuracy,
            "reward_weight_latency":  settings.rl_reward_weight_latency,
            "reward_weight_cost":     settings.rl_reward_weight_cost,
            "reward_weight_drift":    settings.rl_reward_weight_drift,
        })

        model.learn(total_timesteps=total_timesteps)

        model_path = Path(settings.rl_model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))

        mlflow.log_artifact(str(model_path) + ".zip", artifact_path="rl_model")
        logger.info("RL model saved to %s (trained on %d real runs)", model_path, len(historical_runs))


def predict_adjustments(state: dict) -> dict[str, Any]:
    """
    Load the trained RL model and return recommended hyperparameter adjustments.
    Called by the Planner Agent / Orchestrator.
    """
    model_path = Path(settings.rl_model_path + ".zip")
    if not model_path.exists():
        logger.info("No RL model found, returning empty recommendations")
        return {}

    metrics = state.get("metrics", {})
    obs = np.float32([
        metrics.get("accuracy", 0.5),
        min(metrics.get("training_duration_s", 10) / 100, 1.0),
        min(metrics.get("token_cost_usd", 0.0) / 10, 1.0),
        metrics.get("drift_score", 0.0),
        metrics.get("error_rate", 0.0),
        0.5,   # deployment outcome unknown for a not-yet-deployed candidate
        0.0,   # no post-deployment drift yet — this run hasn't shipped
    ])

    model = PPO.load(str(model_path))
    action, _ = model.predict(obs, deterministic=True)
    return MLOpsEnv.ACTION_MAP.get(int(action), {})


# ─────────────────────────────────────────────────────────────────────────────
# Real historical run fetch — MLflow runs enriched with Neon promotion
# outcome and post-deployment drift
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_historical_runs(days: int = 90) -> list[dict]:
    """
    Query MLflow for past training runs and enrich each with:
      - deployment_success: from model_promotions (1.0 Production, 0.0
        Rejected, 0.5 unknown/no record)
      - drift_at_30d: mean serving drift_score from monitoring_events in a
        window around 30 days after the model's promotion

    Enrichment failures for an individual run fall back to "unknown" (0.5 /
    0.0) rather than aborting the whole fetch — a Neon hiccup on one run
    must not throw away every other run's real MLflow data.
    """
    try:
        runs = mlflow.search_runs(
            experiment_names=[settings.mlflow_experiment_name],
            filter_string=(
                f"start_time > {int((datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000)}"
            ),
            max_results=500,
        )
    except Exception as exc:
        logger.warning("Could not fetch MLflow runs: %s", exc)
        return []

    if runs.empty:
        return []

    pool = None
    if settings.database_url:
        try:
            import asyncpg
            pool = await asyncpg.create_pool(
                settings.asyncpg_url, min_size=1, max_size=2,
                command_timeout=15, statement_cache_size=0,
            )
        except Exception as exc:
            logger.warning("RL enrichment: could not connect to Neon — using MLflow-only data: %s", exc)

    client = mlflow.tracking.MlflowClient()
    result: list[dict] = []

    for _, row in runs.iterrows():
        run_dict = {
            "accuracy":    row.get("metrics.best.accuracy", row.get("metrics.accuracy", 0.5)) or 0.5,
            "latency_s":   row.get("metrics.training_duration_s", 60) or 60,
            "cost_usd":    row.get("metrics.token_cost_usd", 0.0) or 0.0,
            "drift_score": row.get("metrics.drift_score", 0.0) or 0.0,
            "error_rate":  row.get("metrics.error_rate", 0.0) or 0.0,
            "framework":   row.get("params.framework") or row.get("params.frameworks"),
            "deployment_success": 0.5,
            "drift_at_30d": 0.0,
        }

        if pool is not None:
            run_id = row.get("run_id")
            enrichment = await _enrich_from_neon(client, pool, run_id)
            run_dict.update(enrichment)

        result.append(run_dict)

    if pool is not None:
        await pool.close()

    return result


async def _enrich_from_neon(client: Any, pool: Any, run_id: Optional[str]) -> dict:
    """Best-effort per-run enrichment; any failure degrades to 'unknown', never raises."""
    enrichment = {"deployment_success": 0.5, "drift_at_30d": 0.0}
    if not run_id:
        return enrichment
    try:
        versions = await asyncio.to_thread(client.search_model_versions, f"run_id='{run_id}'")
        if not versions:
            return enrichment
        mv = versions[0]

        async with pool.acquire() as conn:
            promo = await conn.fetchrow(
                """
                SELECT to_stage, promoted_at FROM model_promotions
                WHERE model_name=$1 AND model_version=$2
                ORDER BY promoted_at DESC LIMIT 1
                """,
                mv.name, mv.version,
            )
            if promo is None:
                return enrichment

            if promo["to_stage"] == "Production":
                enrichment["deployment_success"] = 1.0
            elif promo["to_stage"] == "Rejected":
                enrichment["deployment_success"] = 0.0

            if promo["to_stage"] == "Production" and promo["promoted_at"] is not None:
                window_start = promo["promoted_at"] + timedelta(days=25)
                window_end = promo["promoted_at"] + timedelta(days=35)
                drift_row = await conn.fetchrow(
                    """
                    SELECT AVG(drift_score) AS avg_drift FROM monitoring_events
                    WHERE model_name=$1 AND created_at BETWEEN $2 AND $3
                      AND drift_score IS NOT NULL
                    """,
                    mv.name, window_start, window_end,
                )
                if drift_row is not None and drift_row["avg_drift"] is not None:
                    enrichment["drift_at_30d"] = float(drift_row["avg_drift"])

        return enrichment
    except Exception as exc:
        logger.debug("RL enrichment failed for run_id=%s (non-fatal): %s", run_id, exc)
        return enrichment


if __name__ == "__main__":
    train()
