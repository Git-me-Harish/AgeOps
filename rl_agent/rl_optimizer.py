"""
RL Optimization Agent
─────────────────────
Framework: Stable Baselines3 (PPO)
Training:  Offline on MLflow execution traces (no costly online training)
Deployment: Kubernetes CronJob (once/day)

The agent learns which hyperparameter adjustments and workflow
modifications maximise the reward function:

    R = α·accuracy + β·(1/latency) + γ·(1/cost) – δ·drift
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import gymnasium as gym
import mlflow
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from configs.settings import settings

logger = logging.getLogger(__name__)

# Reward weights (tunable via MLflow params)
ALPHA = 0.4   # accuracy
BETA = 0.2    # latency (inverse)
GAMMA = 0.2   # cost (inverse)
DELTA = 0.2   # drift (penalty)


# ─────────────────────────────────────────────────────────────────────────────
# Custom Gym environment backed by historical MLflow traces
# ─────────────────────────────────────────────────────────────────────────────

class MLOpsEnv(gym.Env):
    """
    Observation space: [accuracy, latency_norm, cost_norm, drift_score, error_rate]
    Action space:      Discrete — map index to a hyperparameter adjustment bundle
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

    def __init__(self, historical_runs: list[dict]) -> None:
        super().__init__()
        self._runs = historical_runs if historical_runs else self._synthetic_runs()
        self._idx = 0

        self.observation_space = spaces.Box(
            low=np.float32([0, 0, 0, 0, 0]),
            high=np.float32([1, 1, 1, 1, 1]),
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
            min(run.get("latency_s", 10) / 100, 1.0),   # normalise to [0,1]
            min(run.get("cost_usd", 1) / 10, 1.0),
            run.get("drift_score", 0.0),
            run.get("error_rate", 0.0),
        ])

    def _compute_reward(self, run: dict) -> float:
        accuracy = run.get("accuracy", 0.0)
        latency = max(run.get("latency_s", 1), 0.001)
        cost = max(run.get("cost_usd", 0.01), 0.0001)
        drift = run.get("drift_score", 0.0)
        return ALPHA * accuracy + BETA / latency + GAMMA / cost - DELTA * drift

    def _synthetic_runs(self) -> list[dict]:
        """Generate synthetic historical traces for development / tests."""
        rng = np.random.default_rng(42)
        return [
            {
                "accuracy": float(rng.uniform(0.7, 0.95)),
                "latency_s": float(rng.uniform(5, 120)),
                "cost_usd": 0.0,   # free tier
                "drift_score": float(rng.uniform(0, 0.2)),
                "error_rate": float(rng.uniform(0, 0.05)),
            }
            for _ in range(500)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Training entry point (run as CronJob)
# ─────────────────────────────────────────────────────────────────────────────

def train(total_timesteps: int = settings.rl_training_timesteps) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("rl-optimization-agent")

    # Fetch last 30 days of MLflow runs as training data
    historical_runs = _fetch_historical_runs(days=30)

    env = MLOpsEnv(historical_runs)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="/tmp/rl_logs")

    with mlflow.start_run(run_name=f"rl-train-{datetime.utcnow().strftime('%Y%m%d')}"):
        mlflow.log_param("algorithm", "PPO")
        mlflow.log_param("total_timesteps", total_timesteps)
        mlflow.log_param("n_runs_used", len(historical_runs))

        model.learn(total_timesteps=total_timesteps)

        model_path = Path(settings.rl_model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))

        mlflow.log_artifact(str(model_path) + ".zip", artifact_path="rl_model")
        logger.info("RL model saved to %s", model_path)


def predict_adjustments(state: dict) -> dict[str, Any]:
    """
    Load the trained RL model and return recommended hyperparameter adjustments.
    Called by the Planner Agent / Orchestrator.
    """
    model_path = Path(settings.rl_model_path + ".zip")
    if not model_path.exists():
        logger.info("No RL model found, returning empty recommendations")
        return {}

    obs = np.float32([
        state.get("metrics", {}).get("accuracy", 0.5),
        0.1,   # latency norm (stub)
        0.0,   # cost norm (free tier)
        state.get("metrics", {}).get("drift_score", 0.0),
        0.0,   # error rate (stub)
    ])

    model = PPO.load(str(model_path))
    action, _ = model.predict(obs, deterministic=True)
    return MLOpsEnv.ACTION_MAP.get(int(action), {})


def _fetch_historical_runs(days: int = 30) -> list[dict]:
    """Query MLflow for past experiment runs and convert to RL episode data."""
    try:
        runs = mlflow.search_runs(
            experiment_names=[settings.mlflow_experiment_name],
            filter_string=f"start_time > {int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)}",
        )
        result = []
        for _, row in runs.iterrows():
            result.append({
                "accuracy": row.get("metrics.accuracy", 0.5),
                "latency_s": row.get("metrics.training_duration_s", 60),
                "cost_usd": 0.0,
                "drift_score": row.get("metrics.drift_score", 0.0),
                "error_rate": row.get("metrics.error_rate", 0.0),
            })
        return result or []
    except Exception as exc:
        logger.warning("Could not fetch MLflow runs: %s", exc)
        return []


if __name__ == "__main__":
    train()
