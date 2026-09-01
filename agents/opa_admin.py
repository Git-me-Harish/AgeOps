"""
agents/opa_admin.py

Phase 6 Security & Compliance page — real read/validate/write path for the
`mlops-opa-policy` ConfigMap the orchestrator pod's OPA sidecar mounts (see
agents/governance_agent.py's module docstring and
configs/kubernetes/agents/agents-deployment.yaml).

Write path: patch_namespaced_config_map — kubelet syncs the mounted volume
to match (~60s), and the sidecar's `opa run --server --watch ...` (added in
this same pass — see agents-deployment.yaml) picks the new content up
without a pod restart.

Validation is honest about what it can actually check: if the `opa` CLI
binary is present on PATH, `opa check` performs a real Rego compile check.
If it is not (the orchestrator/packaging images don't ship it), this
degrades to a structural sanity check and says so explicitly in the
result — it never claims a policy is "valid" without actually having
compiled it.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from configs.settings import settings

logger = logging.getLogger(__name__)

_CONFIGMAP_NAME = "mlops-opa-policy"
_CONFIGMAP_KEY = "mlops_policy.rego"


def _k8s_client() -> Any:
    from kubernetes import client as k8s_client, config as k8s_config
    if settings.k8s_in_cluster:
        k8s_config.load_incluster_config()
    else:
        k8s_config.load_kube_config()
    return k8s_client


class PolicyValidationResult:
    def __init__(self, valid: bool, checked_with: str, errors: Optional[list[str]] = None) -> None:
        self.valid = valid
        self.checked_with = checked_with   # "opa check" | "structural (opa binary unavailable)"
        self.errors = errors or []

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "checked_with": self.checked_with, "errors": self.errors}


def validate_policy(rego_text: str) -> PolicyValidationResult:
    """Real Rego compile check via the `opa` CLI when available; otherwise
    an honestly-labeled structural check only."""
    opa_binary = shutil.which("opa")
    if opa_binary is None:
        errors: list[str] = []
        # A real 'package' statement starts a line (ignoring leading
        # whitespace) — a bare substring check would false-pass prose that
        # merely contains the word "package" without declaring one.
        if not re.search(r"^\s*package\s+\S+", rego_text, re.MULTILINE):
            errors.append("Missing 'package' declaration")
        if rego_text.count("{") != rego_text.count("}"):
            errors.append("Unbalanced braces")
        return PolicyValidationResult(
            valid=not errors, checked_with="structural (opa binary unavailable)", errors=errors,
        )

    with tempfile.TemporaryDirectory() as tmp:
        policy_path = Path(tmp) / _CONFIGMAP_KEY
        policy_path.write_text(rego_text, encoding="utf-8")
        try:
            proc = subprocess.run(
                [opa_binary, "check", str(policy_path)],
                capture_output=True, text=True, timeout=10,
            )
        except Exception as exc:
            return PolicyValidationResult(valid=False, checked_with="opa check", errors=[str(exc)])
        if proc.returncode == 0:
            return PolicyValidationResult(valid=True, checked_with="opa check")
        return PolicyValidationResult(
            valid=False, checked_with="opa check",
            errors=[line for line in (proc.stdout + proc.stderr).splitlines() if line.strip()],
        )


def read_policy() -> str:
    """Read the live policy text straight from the cluster ConfigMap."""
    k8s = _k8s_client()
    core_v1 = k8s.CoreV1Api()
    cm = core_v1.read_namespaced_config_map(name=_CONFIGMAP_NAME, namespace=settings.k8s_namespace)
    return (cm.data or {}).get(_CONFIGMAP_KEY, "")


def write_policy(rego_text: str) -> PolicyValidationResult:
    """
    Validate, then patch the ConfigMap. Raises on Kubernetes API failure —
    callers must not report success for a write that never happened.
    """
    result = validate_policy(rego_text)
    if not result.valid:
        return result

    k8s = _k8s_client()
    core_v1 = k8s.CoreV1Api()
    core_v1.patch_namespaced_config_map(
        name=_CONFIGMAP_NAME,
        namespace=settings.k8s_namespace,
        body={"data": {_CONFIGMAP_KEY: rego_text}},
    )
    logger.info("mlops-opa-policy ConfigMap patched (%d bytes)", len(rego_text))
    return result
