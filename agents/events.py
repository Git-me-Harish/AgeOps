"""
agents/events.py

Phase 6 real-time event bus. Thin wrapper around Redis pub/sub used by
agents to announce state changes (workflow status, alerts, deployment
progress); mcp_servers/api_server.py's /ws/events WebSocket endpoint
subscribes to these same channels and relays them to connected browser
clients so the UI never has to poll.

Channel naming (multi-agent-mlops-v2-plan.md §6.3):
    workflow.{workflow_id}.status_changed
    alert.{severity}.{source}
    model.{model_name}.deployment_status_changed

Publishing is fire-and-forget and never fatal to the caller: a Redis outage
must not break a workflow run, so every failure here is logged and
swallowed — the same degrade-gracefully pattern already used by
agents/llm_gateway.py's semantic cache for the same Redis instance.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from configs.settings import settings

logger = logging.getLogger(__name__)

_redis_client: Optional[Any] = None


async def _get_client() -> Optional[Any]:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )
        await client.ping()
        _redis_client = client
    except Exception as exc:
        logger.warning("Redis unavailable — event publish disabled (non-fatal): %s", exc)
        return None
    return _redis_client


async def publish_event(channel: str, payload: dict[str, Any]) -> None:
    """Publish a JSON event to a Redis pub/sub channel. Never raises."""
    client = await _get_client()
    if client is None:
        return
    body = {**payload, "channel": channel, "emitted_at": datetime.now(timezone.utc).isoformat()}
    try:
        await client.publish(channel, json.dumps(body, default=str))
    except Exception as exc:
        logger.warning("Event publish to %s failed (non-fatal): %s", channel, exc)


async def publish_workflow_status(workflow_id: str, **fields: Any) -> None:
    await publish_event(f"workflow.{workflow_id}.status_changed", {"workflow_id": workflow_id, **fields})


async def publish_alert(severity: str, source: str, **fields: Any) -> None:
    await publish_event(f"alert.{severity}.{source}", {"severity": severity, "source": source, **fields})


async def publish_deployment_status(model_name: str, **fields: Any) -> None:
    await publish_event(
        f"model.{model_name}.deployment_status_changed", {"model_name": model_name, **fields}
    )


_sync_redis_client: Optional[Any] = None


def publish_deployment_status_sync(model_name: str, **fields: Any) -> None:
    """
    Synchronous counterpart of publish_deployment_status, for callers that
    run the canary rollout loop on a plain thread with time.sleep() rather
    than an asyncio event loop (agents/deployment_agent.py._deploy_canary).
    Same fire-and-forget, non-fatal contract.
    """
    global _sync_redis_client
    try:
        if _sync_redis_client is None:
            import redis as sync_redis

            _sync_redis_client = sync_redis.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                decode_responses=True,
            )
            _sync_redis_client.ping()
        body = {
            **fields,
            "model_name": model_name,
            "channel": f"model.{model_name}.deployment_status_changed",
            "emitted_at": datetime.now(timezone.utc).isoformat(),
        }
        _sync_redis_client.publish(f"model.{model_name}.deployment_status_changed", json.dumps(body, default=str))
    except Exception as exc:
        logger.warning("Sync event publish for %s failed (non-fatal): %s", model_name, exc)
