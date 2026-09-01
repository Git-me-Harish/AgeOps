#!/usr/bin/env python3
"""
Weekly RL Optimizer Retrain
════════════════════════════
Retrains the PPO hyperparameter-adjustment policy on real MLflow run history
enriched with real Neon promotion outcomes and post-deployment drift — see
rl_agent/rl_optimizer.py. Plan §5.4 calls for weekly retraining (not daily —
needs enough new data to be worth it).

Run via Kubernetes CronJob (weekly) or directly:
  python scripts/train_rl_weekly.py
"""
from __future__ import annotations

import logging
import sys

sys.path.insert(0, ".")

from rl_agent.rl_optimizer import train  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_rl_weekly")


if __name__ == "__main__":
    logger.info("Starting weekly RL Optimizer retrain")
    train()
    logger.info("RL Optimizer retrain complete")
