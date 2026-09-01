"""
serving/health.py

Liveness and readiness state for the MLOps inference server.

Liveness  (/health/live):
  Returns 200 immediately on any request — the process is alive if it can respond.
  Kubernetes uses this to decide whether to restart the container.

Readiness (/health/ready):
  Returns 200 only when:
    1. model_loaded is True — MLflow pyfunc model loaded without error
    2. warmup_done  is True — at least one prediction completed successfully
  Returns 503 (Service Unavailable) otherwise — Kubernetes stops routing traffic.

Design:
  HealthStatus is a module-level singleton — no class instantiation needed.
  The inference server sets flags directly:
    health.model_loaded = True
    health.warmup_done  = True
  The /health/ready handler reads these flags synchronously.

  Thread safety: reads and writes to bool fields are atomic in CPython (GIL).
  No locking required for this use case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class _HealthStatus:
    """Module-level singleton tracking inference server readiness."""

    model_loaded:  bool = False
    warmup_done:   bool = False
    model_name:    str  = "unknown"
    model_version: str  = "unknown"
    framework:     str  = "unknown"
    started_at:    Optional[datetime] = None
    ready_at:      Optional[datetime] = None

    @property
    def is_live(self) -> bool:
        """Liveness: always True once the process is handling requests."""
        return True

    @property
    def is_ready(self) -> bool:
        """Readiness: True only after model load + warmup."""
        return self.model_loaded and self.warmup_done

    def mark_model_loaded(
        self,
        model_name:    str,
        model_version: str,
        framework:     str,
    ) -> None:
        """Call once MLflow pyfunc model is in memory."""
        self.model_loaded  = True
        self.model_name    = model_name
        self.model_version = model_version
        self.framework     = framework
        self.started_at    = datetime.now(tz=timezone.utc)

    def mark_warmup_done(self) -> None:
        """Call once the warmup prediction completes successfully."""
        self.warmup_done = True
        self.ready_at    = datetime.now(tz=timezone.utc)

    def to_dict(self) -> dict:
        return {
            "model_loaded":  self.model_loaded,
            "warmup_done":   self.warmup_done,
            "model_name":    self.model_name,
            "model_version": self.model_version,
            "framework":     self.framework,
            "is_ready":      self.is_ready,
            "started_at":    self.started_at.isoformat() if self.started_at else None,
            "ready_at":      self.ready_at.isoformat() if self.ready_at else None,
        }


# Module-level singleton — import and use directly
health = _HealthStatus()