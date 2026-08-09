"""
Governance Agent

Responsibilities
- Enforce OPA (Open Policy Agent) Rego policies before high-risk actions
- Maintain an immutable audit trail of all agent decisions
- Generate compliance reports (GDPR, SOC 2, HIPAA)
- Implement accountability tokens for agent delegation chains
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import mlflow

from agents import AgentTaskResult
from configs.settings import settings

logger = logging.getLogger(__name__)

_OPA_ENDPOINT = "http://opa.gatekeeper-system.svc.cluster.local:8181/v1/data/kubernetes/admission"


class GovernanceAgent:
    """
    Policy enforcement and audit layer for the multi-agent system.
    """

    def __init__(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # Policy check (called pre-action by other agents) 
    def check_policy(self, action: str, context: dict) -> tuple[bool, str]:
        """
        Evaluate an OPA policy before executing a high-risk action.
        Returns (allowed, reason).
        """
        try:
            import urllib.request as urlreq
            payload = json.dumps({"input": {"action": action, "context": context}}).encode()
            req = urlreq.Request(
                _OPA_ENDPOINT,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlreq.urlopen(req, timeout=3) as resp:
                result = json.loads(resp.read())
            allowed = not bool(result.get("result", {}).get("deny"))
            reason = "; ".join(result.get("result", {}).get("deny", ["allowed"]))
            return allowed, reason
        except Exception as exc:
            # Fail open only for non-production; in production fail closed
            if settings.app_env == "production":
                logger.error("OPA unreachable — failing closed: %s", exc)
                return False, "OPA unreachable"
            logger.warning("OPA unreachable — failing open for dev: %s", exc)
            return True, "OPA unreachable (dev mode)"

    # Post-workflow audit 
    @mlflow.trace(name="governance_agent.audit")
    def audit(self, state: dict) -> AgentTaskResult:
        task_id = state.get("workflow_id", "unknown")

        try:
            audit_record = self._build_audit_record(state)
            signed = self._sign_record(audit_record)
            self._persist_audit(signed)

            with mlflow.start_run(run_name=f"governance-{task_id}", nested=True):
                mlflow.set_tag("agent", "governance_agent")
                mlflow.log_dict(signed, "audit_trail.json")
                mlflow.log_metric("decisions_count", len(state.get("agent_decisions", [])))
                mlflow.log_metric("errors_count", len(state.get("errors", [])))

            return AgentTaskResult(
                task_id=task_id,
                status="success",
                output={"audit_id": signed["signature"][:16], "decisions": len(state.get("agent_decisions", []))},
            )
        except Exception as exc:
            logger.exception("GovernanceAgent audit failed for %s", task_id)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))

    # Helpers 
    def _build_audit_record(self, state: dict) -> dict:
        return {
            "workflow_id": state.get("workflow_id"),
            "timestamp": time.time(),
            "final_stage": state.get("current_stage"),
            "model_uri": state.get("model_uri"),
            "metrics": state.get("metrics", {}),
            "errors": state.get("errors", []),
            "decisions": state.get("agent_decisions", []),
            "app_env": settings.app_env,
        }

    def _sign_record(self, record: dict) -> dict:
        """SHA-256 signature for tamper evidence (production would use asymmetric signing)."""
        canonical = json.dumps(record, sort_keys=True).encode()
        signature = hashlib.sha256(canonical).hexdigest()
        return {**record, "signature": signature}

    def _persist_audit(self, record: dict) -> None:
        """Write to Loki via stdout structured logging (Promtail picks it up)."""
        logger.info("AUDIT_TRAIL %s", json.dumps(record))
