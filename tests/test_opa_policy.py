"""
tests/test_opa_policy.py
─────────────────────────
Tests for GovernanceAgent policy helpers and the OPA Rego policy itself.

Two test surfaces:
  A) GovernanceAgent pure-Python helpers — no OPA, no LLM, no DB required.
  B) OPA Rego policy evaluation via `opa eval` subprocess.
     These tests are SKIPPED automatically when the opa binary is absent
     (CI without OPA installed, local dev on Windows, etc.)

Run all (requires opa binary for group B):
    pytest tests/test_opa_policy.py -v

Run only group A (always available):
    pytest tests/test_opa_policy.py -v -k "not opa_binary"
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.governance_agent import GovernanceAgent, PolicyDecision, PolicyInput

# Helpers
OPA_POLICY_PATH = Path("configs/opa_policies/mlops_policy.rego")
OPA_AVAILABLE = shutil.which("opa") is not None

opa_binary = pytest.mark.skipif(
    not OPA_AVAILABLE,
    reason="opa binary not found — install OPA CLI to run these tests",
)


def make_agent() -> GovernanceAgent:
    agent = GovernanceAgent()
    return agent


def full_model_tags() -> dict[str, str]:
    """A complete set of required MLflow model tags."""
    return {
        "mlops.dataset.uri":      "s3://bucket/data/train.parquet",
        "mlops.dataset.hash":     "abc123def456abc123def456abc123de",
        "mlops.dataset.row_count": "125000",
        "mlops.framework":        "xgboost",
        "mlops.eval.accuracy":    "0.87",
        "mlops.eval.f1":          "0.83",
        "mlops.eval.bias_passed": "True",
        "mlops.security.trivy_scan": "passed",
    }


def passing_eval_metrics() -> dict[str, Any]:
    return {
        "accuracy": 0.87,
        "f1": 0.83,
        "auc": 0.91,
        "bias_passed": True,
        "overall_passed": True,
    }


def passing_security() -> dict[str, Any]:
    return {"trivy_passed": True, "critical_cves": 0,
            "high_cves": 0, "semgrep_passed": True, "secrets_detected": False}


# A) Pure-Python GovernanceAgent helper tests
class TestValidateRequiredTags:
    def test_all_tags_present_passes(self) -> None:
        agent = make_agent()
        result = agent._validate_required_tags(full_model_tags())
        assert result["passed"] is True
        assert result["missing"] == []

    def test_missing_single_tag_fails(self) -> None:
        agent = make_agent()
        tags = full_model_tags()
        del tags["mlops.dataset.hash"]
        result = agent._validate_required_tags(tags)
        assert result["passed"] is False
        assert "mlops.dataset.hash" in result["missing"]

    def test_missing_multiple_tags_all_reported(self) -> None:
        agent = make_agent()
        tags = full_model_tags()
        del tags["mlops.eval.f1"]
        del tags["mlops.security.trivy_scan"]
        result = agent._validate_required_tags(tags)
        assert result["passed"] is False
        assert "mlops.eval.f1" in result["missing"]
        assert "mlops.security.trivy_scan" in result["missing"]

    def test_empty_tag_value_counts_as_missing(self) -> None:
        """A tag with an empty string value must count as missing."""
        agent = make_agent()
        tags = full_model_tags()
        tags["mlops.dataset.hash"] = ""
        result = agent._validate_required_tags(tags)
        assert result["passed"] is False
        assert "mlops.dataset.hash" in result["missing"]

    def test_empty_tags_dict_reports_all_missing(self) -> None:
        agent = make_agent()
        result = agent._validate_required_tags({})
        assert result["passed"] is False
        assert len(result["missing"]) == len(GovernanceAgent.REQUIRED_MODEL_TAGS)


class TestFailClosedDecision:
    def test_always_denies(self) -> None:
        decision = GovernanceAgent._fail_closed_decision(
            input_hash="abc", elapsed_ms=0.5, reason="OPA timeout"
        )
        assert decision.allowed is False

    def test_deny_reason_included(self) -> None:
        decision = GovernanceAgent._fail_closed_decision(
            input_hash="abc", elapsed_ms=1.0, reason="OPA sidecar unreachable"
        )
        assert len(decision.deny_reasons) == 1
        assert "OPA sidecar unreachable" in decision.deny_reasons[0]

    def test_returns_policy_decision_type(self) -> None:
        decision = GovernanceAgent._fail_closed_decision("hash", 0.0, "reason")
        assert isinstance(decision, PolicyDecision)

    def test_was_sidecar_true(self) -> None:
        """Fail-closed decisions are attributed to the sidecar path."""
        decision = GovernanceAgent._fail_closed_decision("hash", 0.0, "reason")
        assert decision.was_sidecar is True


class TestCheckPolicyDevMode:
    """
    Tests for GovernanceAgent.check_policy() in dev mode (OPA unreachable).
    In dev (is_production=False), an unreachable OPA must fail-OPEN.
    """

    @pytest.mark.asyncio
    async def test_opa_unreachable_fails_open_in_dev(self) -> None:
        agent = make_agent()
        agent._session = AsyncMock()
        agent._session.post = MagicMock(side_effect=ConnectionError("OPA down"))

        policy_input = PolicyInput(
            workflow_id="wf-dev-001",
            agent_role="governance",
            action="promote_to_staging",
            model_tags=full_model_tags(),
            eval_metrics=passing_eval_metrics(),
            security_scan=passing_security(),
        )

        with patch("configs.settings.settings.app_env", "development"):
            decision = await agent.check_policy(policy_input)

        assert decision.allowed is True
        assert decision.policy_name == "dev_bypass"

    @pytest.mark.asyncio
    async def test_opa_timeout_fails_closed_in_production(self) -> None:
        """In production, OPA timeout must fail-CLOSED."""
        import asyncio as _asyncio

        agent = make_agent()
        agent._session = AsyncMock()
        agent._session.post = MagicMock(
            side_effect=_asyncio.TimeoutError()
        )

        policy_input = PolicyInput(
            workflow_id="wf-prod-001",
            agent_role="governance",
            action="promote_to_production",
            model_tags=full_model_tags(),
            eval_metrics=passing_eval_metrics(),
            security_scan=passing_security(),
        )

        with patch("agents.governance_agent.settings") as mock_settings:
            mock_settings.opa_endpoint = "http://127.0.0.1:8181/v1/data"
            mock_settings.opa_policy_base_path = "kubernetes/admission"
            mock_settings.opa_timeout_seconds = 1.0
            mock_settings.is_production = True

            decision = await agent.check_policy(policy_input)

        assert decision.allowed is False
        assert any("timeout" in r.lower() for r in decision.deny_reasons)

    @pytest.mark.asyncio
    async def test_opa_success_response_allow_true(self) -> None:
        """When OPA returns allow=true, decision.allowed must be True."""
        agent = make_agent()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"result": {"allow": True, "deny_reasons": []}}
        )
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        agent._session = MagicMock()
        agent._session.post = MagicMock(return_value=mock_ctx)

        policy_input = PolicyInput(
            workflow_id="wf-001",
            agent_role="governance",
            action="promote_to_staging",
            model_tags=full_model_tags(),
            eval_metrics=passing_eval_metrics(),
            security_scan=passing_security(),
        )
        decision = await agent.check_policy(policy_input)
        assert decision.allowed is True
        assert decision.deny_reasons == []

    @pytest.mark.asyncio
    async def test_opa_success_response_allow_false(self) -> None:
        """When OPA returns allow=false with reasons, they must be propagated."""
        agent = make_agent()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "result": {
                "allow": False,
                "deny_reasons": ["F1 score 0.50 below threshold 0.65"]
            }
        })
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        agent._session = MagicMock()
        agent._session.post = MagicMock(return_value=mock_ctx)

        policy_input = PolicyInput(
            workflow_id="wf-001",
            agent_role="governance",
            action="promote_to_staging",
            model_tags=full_model_tags(),
            eval_metrics=passing_eval_metrics(),
            security_scan=passing_security(),
        )
        decision = await agent.check_policy(policy_input)
        assert decision.allowed is False
        assert len(decision.deny_reasons) == 1
        assert "F1" in decision.deny_reasons[0]


def opa_eval(input_data: dict, query: str = "data.mlops.allow") -> Any:
    result = subprocess.run(
        [
            "opa",
            "eval",
            "--data",
            str(OPA_POLICY_PATH),
            "--stdin-input",
            "--format",
            "raw",
            query,
        ],
        input=json.dumps(input_data).encode(),
        capture_output=True,
        timeout=10,
    )

    if result.returncode != 0:
        raise RuntimeError(f"opa eval failed: {result.stderr.decode()}")

    output = result.stdout.decode()
    print(f"\nOPA QUERY: {query}")
    print(f"OPA OUTPUT:\n{output}")

    return json.loads(output)


def make_opa_input(
    action: str = "promote_to_staging",
    agent_role: str = "governance",
    model_tags: Optional[dict] = None,
    eval_metrics: Optional[dict] = None,
    security_scan: Optional[dict] = None,
) -> dict:
    return {
        "workflow_id": "wf-test",
        "agent_role": agent_role,
        "action": action,
        "model_tags": model_tags or full_model_tags(),
        "eval_metrics": eval_metrics or passing_eval_metrics(),
        "security_scan": security_scan or passing_security(),
        "lineage_id": 1,
        "lineage_hash": "abc123",
        "requester": "system",
    }


@opa_binary
class TestOpaRegoPolicy:
    def test_allow_staging_with_all_passing(self) -> None:
        result = opa_eval(make_opa_input(action="promote_to_staging"))
        assert result is True

    def test_deny_staging_low_f1(self) -> None:
        """F1 < 0.60 must deny staging promotion."""
        metrics = passing_eval_metrics()
        metrics["f1"] = 0.55
        result = opa_eval(make_opa_input(action="promote_to_staging", eval_metrics=metrics))
        assert result is False

    def test_deny_staging_failed_trivy(self) -> None:
        """Failed Trivy scan must deny promotion."""
        security = passing_security()
        security["trivy_passed"] = False
        security["critical_cves"] = 2
        result = opa_eval(make_opa_input(action="promote_to_staging", security_scan=security))
        assert result is False

    def test_deny_staging_secrets_detected(self) -> None:
        """Secrets in dataset must deny promotion."""
        security = passing_security()
        security["secrets_detected"] = True
        result = opa_eval(make_opa_input(action="promote_to_staging", security_scan=security))
        assert result is False

    def test_deny_staging_missing_required_tag(self) -> None:
        """Missing required model tag must deny promotion."""
        tags = full_model_tags()
        del tags["mlops.dataset.hash"]
        result = opa_eval(make_opa_input(action="promote_to_staging", model_tags=tags))
        assert result is False

    def test_deny_staging_bias_not_passed(self) -> None:
        metrics = passing_eval_metrics()
        metrics["bias_passed"] = 0.0   # False
        result = opa_eval(make_opa_input(action="promote_to_staging", eval_metrics=metrics))
        assert result is False

    def test_deny_production_without_human_approval(self) -> None:
        """Production promotion without approved_by tag must be denied."""
        tags = full_model_tags()
        # No 'mlops.approved_by' tag set
        result = opa_eval(make_opa_input(action="promote_to_production", model_tags=tags))
        assert result is False

    def test_allow_production_with_human_approval(self) -> None:
        """Production promotion with approved_by tag must pass all other gates."""
        tags = full_model_tags()
        tags["mlops.approved_by"] = "harish-engineer"
        result = opa_eval(make_opa_input(action="promote_to_production", model_tags=tags))
        assert result is True

    def test_deny_wrong_agent_role_for_action(self) -> None:
        """training role must not be able to promote to staging."""
        result = opa_eval(make_opa_input(
            action="promote_to_staging",
            agent_role="training",
        ))
        assert result is False

    def test_deny_unknown_agent_role(self) -> None:
        result = opa_eval(make_opa_input(agent_role="unknown_bot"))
        assert result is False

    def test_allow_run_training_job_from_planner(self) -> None:
        result = opa_eval(make_opa_input(
            action="run_training_job",
            agent_role="planner",
        ))
        assert result is True

    def test_deny_run_training_job_with_secrets(self) -> None:
        security = passing_security()
        security["secrets_detected"] = True
        result = opa_eval(make_opa_input(
            action="run_training_job",
            agent_role="planner",
            security_scan=security,
        ))
        assert result is False

    def test_deny_reason_list_populated_on_denial(self) -> None:
        """deny_reasons must be a non-empty list when the policy denies."""
        metrics = passing_eval_metrics()
        metrics["f1"] = 0.40
        metrics["accuracy"] = 0.50
        deny_reasons = opa_eval(
            make_opa_input(action="promote_to_staging", eval_metrics=metrics),
            query="data.mlops.deny_reasons",
        )
        assert isinstance(deny_reasons, list)
        assert len(deny_reasons) >= 1

    def test_staging_threshold_lower_than_production(self) -> None:
        staging_input = make_opa_input(
            action="promote_to_staging",
            eval_metrics={
                "accuracy": 0.67,
                "f1": 0.62,
                "auc": 0.90,
                "bias_passed": True,
                "overall_passed": True,
            },
        )

        staging_result = opa_eval(
            staging_input,
            query="data.mlops.deny_reasons",
        )

        print(f"\nSTAGING DENY REASONS:\n{staging_result}")

        assert staging_result == []