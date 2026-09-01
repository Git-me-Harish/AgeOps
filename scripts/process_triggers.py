#!/usr/bin/env python3
"""
Process Pending Retraining Triggers
════════════════════════════════════
Polls workflow_triggers for 'pending' rows — written by MonitoringAgent on a
CRITICAL drift/accuracy breach (agents/monitoring_agent.py), or manually via
POST /monitoring/v1/trigger_retraining — and launches a real workflow for
each through OrchestratorAgent.process_pending_triggers().

This is the consumer side of Phase 5's closed retraining loop (plan §5.3),
implemented as a durable table poll rather than ephemeral pub/sub so a
trigger survives a process restart and is auditable.

Run via Kubernetes CronJob (every few minutes) or directly:
  python scripts/process_triggers.py
"""
from __future__ import annotations

import asyncio
import logging
import sys

sys.path.insert(0, ".")

from agents.orchestrator import OrchestratorAgent  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("process_triggers")


async def main() -> None:
    orchestrator = OrchestratorAgent()
    results = await orchestrator.process_pending_triggers(limit=5)
    if not results:
        logger.info("No pending retraining triggers.")
        return
    for r in results:
        logger.info(
            "Trigger %s -> workflow %s: %s",
            r["trigger_id"], r["workflow_id"], r["status"],
        )


if __name__ == "__main__":
    asyncio.run(main())
