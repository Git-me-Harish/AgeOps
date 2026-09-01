#!/usr/bin/env python3
"""
Production Monitoring Loop
═══════════════════════════
Long-running process: every settings.monitoring_interval_seconds, ticks
MonitoringAgent.tick() for every model currently in the MLflow Registry's
Production stage. Real Evidently drift check, real Prometheus queries, real
ground-truth accuracy where available — see agents/monitoring_agent.py.

This is the standalone entry point for plan §5.2's "every 60 seconds" loop,
independent of any single workflow (agents/monitoring_agent.py's run() is a
separate, per-workflow LangGraph node used right after a fresh deployment).

Run via a long-running Kubernetes Deployment (not a CronJob — this process
sleeps and loops itself) or directly:
  python scripts/monitor_loop.py
"""
from __future__ import annotations

import asyncio
import logging
import sys

sys.path.insert(0, ".")

import asyncpg
import mlflow

from agents.monitoring_agent import MonitoringAgent  # noqa: E402
from configs.settings import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("monitor_loop")


async def _list_production_models() -> list[tuple[str, str]]:
    """Return [(model_name, model_version), ...] currently in Production."""
    client = mlflow.tracking.MlflowClient()
    try:
        registered = await asyncio.to_thread(list, client.search_registered_models())
    except Exception as exc:
        logger.error("Could not list registered models: %s", exc)
        return []

    result: list[tuple[str, str]] = []
    for rm in registered:
        try:
            versions = await asyncio.to_thread(client.get_latest_versions, rm.name, ["Production"])
            for v in versions:
                result.append((rm.name, v.version))
        except Exception as exc:
            logger.warning("Could not fetch Production version for %s: %s", rm.name, exc)
    return result


async def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    agent = MonitoringAgent()

    pool = None
    if settings.database_url:
        try:
            pool = await asyncpg.create_pool(
                settings.asyncpg_url, min_size=1, max_size=settings.database_pool_size,
                command_timeout=30, statement_cache_size=0,
            )
            await agent.connect(pool)
        except Exception as exc:
            logger.warning("Monitor loop: could not connect to Neon — accuracy checks will report unavailable: %s", exc)

    logger.info(
        "Monitor loop starting: interval=%ds", settings.monitoring_interval_seconds,
    )
    try:
        while True:
            models = await _list_production_models()
            if not models:
                logger.info("No Production models found — nothing to monitor this tick.")
            for model_name, model_version in models:
                try:
                    event = await agent.tick(model_name, model_version)
                    logger.info(
                        "Tick: model=%s v=%s alert_level=%s alerts=%s",
                        model_name, model_version, event.get("alert_level"), event.get("alerts"),
                    )
                except Exception:
                    logger.exception("Monitoring tick failed for %s v%s", model_name, model_version)
            await asyncio.sleep(settings.monitoring_interval_seconds)
    finally:
        if pool is not None:
            await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
