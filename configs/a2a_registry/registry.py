"""
A2A (Agent-to-Agent) Registry
─────────────────────────────
Implements dynamic agent discovery via Agent Cards.
Each agent registers its capabilities; the Orchestrator
queries this registry at runtime to select the right agent.

Agent Cards are stored in agent_cards.json and can be
patched via the UI → Agent Configuration Panel.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_CARDS_PATH = Path(__file__).parent / "agent_cards.json"


class AgentCard(BaseModel):
    """Schema for a registered agent's capability declaration."""
    agent_id: str
    name: str
    version: str
    description: str
    capabilities: list[str]
    mcp_tools: list[str] = Field(default_factory=list)
    health_endpoint: Optional[str] = None
    max_concurrent_tasks: int = 5
    timeout_seconds: int = 120
    requires_human_approval_for: list[str] = Field(default_factory=list)


class A2ARegistry:
    """
    In-memory agent registry backed by a JSON file.
    In production, replace with a Redis-backed or
    database-backed registry for HA.
    """

    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}
        self._load()

    def _load(self) -> None:
        if _CARDS_PATH.exists():
            try:
                data = json.loads(_CARDS_PATH.read_text())
                for card_data in data.get("agents", []):
                    card = AgentCard(**card_data)
                    self._cards[card.agent_id] = card
                logger.info("Loaded %d agent cards from %s", len(self._cards), _CARDS_PATH)
            except Exception as exc:
                logger.warning("Failed to load agent cards: %s", exc)

    def register(self, card: AgentCard) -> None:
        self._cards[card.agent_id] = card
        self._persist()

    def discover(self, capability: str) -> list[AgentCard]:
        """Return all agents that have a given capability."""
        return [c for c in self._cards.values() if capability in c.capabilities]

    def get(self, agent_id: str) -> Optional[AgentCard]:
        return self._cards.get(agent_id)

    def list_all(self) -> list[AgentCard]:
        return list(self._cards.values())

    def _persist(self) -> None:
        _CARDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CARDS_PATH.write_text(
            json.dumps({"agents": [c.model_dump() for c in self._cards.values()]}, indent=2)
        )
