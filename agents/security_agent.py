"""
Security Agent

Responsibilities
- Pre-flight security check before any pipeline executes
- Prompt injection detection and sanitization
- PII leakage detection in agent outputs
- Runtime secrets detection in logs / artifacts
- Integration hook for Trivy scan results via CI/CD

This agent is a "guardrail node" in LangGraph terminology.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import mlflow

from agents import AgentTaskResult
from configs.settings import settings

logger = logging.getLogger(__name__)

# Detection patterns 
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"bypass\s+security", re.IGNORECASE),
    re.compile(r"execute\s+arbitrary\s+code", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
]

_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("phone", re.compile(r"\b\+?1?\s*[-.]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
]

_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("generic_secret", re.compile(r'(?i)(password|secret|token|api_key)\s*=\s*["\']?[^\s"\']{8,}')),
]


class SecurityAgent:
    """
    Stateless guardrail node — called at the start of every workflow
    and as a validator on agent outputs.
    """

    def __init__(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # Pre-flight check (called by Orchestrator at pipeline start) 
    @mlflow.trace(name="security_agent.pre_flight_check")
    def pre_flight_check(self, state: dict) -> AgentTaskResult:
        task_id = state.get("workflow_id", "unknown")
        issues: list[str] = []

        # Check dataset_uri for obvious path traversal
        dataset_uri = state.get("dataset_uri", "")
        if ".." in dataset_uri or dataset_uri.startswith("/etc"):
            issues.append(f"Suspicious dataset_uri: {dataset_uri}")

        # Sanitize any string fields for injection patterns
        for field in ("dataset_uri", "model_uri"):
            value = state.get(field, "") or ""
            injections = self.detect_prompt_injection(value)
            if injections:
                issues.extend([f"Injection in {field}: {p}" for p in injections])

        with mlflow.start_run(run_name=f"security-preflight-{task_id}", nested=True):
            mlflow.set_tag("agent", "security_agent")
            mlflow.log_metric("pre_flight_issues", len(issues))
            if issues:
                mlflow.set_tag("security_issues", "; ".join(issues))

        if issues:
            return AgentTaskResult(
                task_id=task_id,
                status="failed",
                error=f"Pre-flight security check failed: {'; '.join(issues)}",
            )
        return AgentTaskResult(task_id=task_id, status="success")

    # Prompt injection detection 
    def detect_prompt_injection(self, text: str) -> list[str]:
        """Return list of matched injection pattern names."""
        return [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]

    def sanitize_prompt(self, prompt: str) -> str:
        """Replace injection patterns with a safe placeholder."""
        for pattern in _INJECTION_PATTERNS:
            prompt = pattern.sub("[REMOVED]", prompt)
        return prompt

    # Output validation (PII / secrets) 
    def validate_agent_output(self, output: str) -> tuple[bool, list[str]]:
        """
        Check agent output for PII leakage and secrets.
        Returns (safe, list_of_violation_types).
        """
        violations: list[str] = []

        for name, pattern in _PII_PATTERNS:
            if pattern.search(output):
                violations.append(f"PII:{name}")
                logger.warning("PII leak detected in agent output: %s", name)

        for name, pattern in _SECRET_PATTERNS:
            if pattern.search(output):
                violations.append(f"SECRET:{name}")
                logger.error("Secret leak detected in agent output: %s", name)

        return len(violations) == 0, violations

    # Trivy scan result ingestion 
    def ingest_trivy_results(self, sarif_path: str, task_id: str) -> AgentTaskResult:
        """
        Parse Trivy SARIF output and fail the pipeline if CRITICAL CVEs exist.
        Called from CI/CD via the MCP security server.
        """
        import json

        try:
            with open(sarif_path) as f:
                sarif = json.load(f)

            critical_count = 0
            for run in sarif.get("runs", []):
                for result in run.get("results", []):
                    level = result.get("level", "")
                    if level in ("error",):   # SARIF error = CRITICAL/HIGH
                        critical_count += 1

            with mlflow.start_run(run_name=f"trivy-{task_id}", nested=True):
                mlflow.log_metric("critical_cves", critical_count)

            if critical_count > 0:
                return AgentTaskResult(
                    task_id=task_id,
                    status="failed",
                    error=f"Trivy found {critical_count} CRITICAL/HIGH CVEs — pipeline blocked",
                )
            return AgentTaskResult(task_id=task_id, status="success", output={"critical_cves": 0})

        except Exception as exc:
            logger.exception("Trivy result ingestion failed")
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))
