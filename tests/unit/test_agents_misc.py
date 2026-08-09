# tests/unit/test_agents_misc.py
"""
Unit tests for Evaluation Agent, RL Optimizer, Governance Agent,
and Training Agent in-process fallback.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from agents.evaluation_agent import EvaluationAgent, THRESHOLDS
from agents.governance_agent import GovernanceAgent
from agents.training_agent import TrainingAgent


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Agent
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def eval_agent() -> EvaluationAgent:
    return EvaluationAgent()


class TestEvaluationAgent:
    def test_threshold_check_all_pass(self, eval_agent):
        metrics = {"accuracy": 0.90, "f1_score": 0.85, "max_drift_score": 0.10}
        passed, failures = eval_agent._check_thresholds(metrics)
        assert passed is True
        assert failures == []

    def test_accuracy_below_threshold_fails(self, eval_agent):
        metrics = {"accuracy": 0.70, "f1_score": 0.80, "max_drift_score": 0.05}
        passed, failures = eval_agent._check_thresholds(metrics)
        assert passed is False
        assert any("accuracy" in f for f in failures)

    def test_drift_above_threshold_fails(self, eval_agent):
        metrics = {"accuracy": 0.85, "f1_score": 0.80, "max_drift_score": 0.20}
        passed, failures = eval_agent._check_thresholds(metrics)
        assert passed is False
        assert any("max_drift_score" in f for f in failures)

    def test_missing_metric_does_not_fail(self, eval_agent):
        # If a metric isn't in state, it should be skipped gracefully
        passed, failures = eval_agent._check_thresholds({"accuracy": 0.85})
        assert passed is True

    def test_bias_check_returns_bool(self, eval_agent):
        result = eval_agent._check_bias("runs:/fake/model")
        assert isinstance(result, bool)

    def test_run_no_model_uri_fails(self, eval_agent, sample_state):
        state = {**sample_state, "model_uri": ""}
        result = eval_agent.run(state)
        assert result.status == "failed"
        assert result.error is not None

    def test_full_run_with_mock_evaluate(self, eval_agent, sample_state):
        mock_metrics = {"accuracy": 0.88, "f1_score": 0.86, "roc_auc": 0.91}
        with patch.object(eval_agent, "_run_mlflow_evaluate", return_value=mock_metrics):
            result = eval_agent.run(sample_state)
        assert result.status == "success"
        assert result.output["approved"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Governance Agent
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def gov_agent() -> GovernanceAgent:
    return GovernanceAgent()


class TestGovernanceAgent:
    def test_build_audit_record_has_required_keys(self, gov_agent, sample_state):
        record = gov_agent._build_audit_record(sample_state)
        for key in ("workflow_id", "timestamp", "final_stage", "model_uri", "metrics", "errors", "decisions"):
            assert key in record

    def test_sign_record_adds_signature(self, gov_agent, sample_state):
        record = gov_agent._build_audit_record(sample_state)
        signed = gov_agent._sign_record(record)
        assert "signature" in signed
        assert len(signed["signature"]) == 64  # SHA-256 hex

    def test_sign_record_is_deterministic(self, gov_agent, sample_state):
        record = gov_agent._build_audit_record(sample_state)
        sig1 = gov_agent._sign_record(record)["signature"]
        sig2 = gov_agent._sign_record(record)["signature"]
        assert sig1 == sig2

    def test_sign_record_changes_on_mutation(self, gov_agent, sample_state):
        record = gov_agent._build_audit_record(sample_state)
        sig1 = gov_agent._sign_record(record)["signature"]
        record["errors"] = ["tampered"]
        sig2 = gov_agent._sign_record(record)["signature"]
        assert sig1 != sig2

    def test_audit_run_returns_success(self, gov_agent, sample_state):
        result = gov_agent.audit(sample_state)
        assert result.status == "success"
        assert "audit_id" in (result.output or {})

    def test_policy_check_dev_open_on_opa_unreachable(self, gov_agent):
        with patch("agents.governance_agent.settings") as mock_settings:
            mock_settings.app_env = "development"
            allowed, reason = gov_agent.check_policy("deploy", {"model": "test"})
        assert allowed is True


# ═══════════════════════════════════════════════════════════════════════════════
# Training Agent — in-process fallback
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def train_agent() -> TrainingAgent:
    with patch("agents.training_agent.k8s_config"):
        agent = TrainingAgent()
        agent._k8s_available = False
        return agent


class TestTrainingAgent:
    def test_merge_hyperparams_defaults(self, train_agent):
        params = train_agent._merge_hyperparams({})
        assert "learning_rate" in params
        assert "n_estimators" in params

    def test_rl_recommendations_override_defaults(self, train_agent):
        params = train_agent._merge_hyperparams({"learning_rate": 0.001, "n_estimators": 50})
        assert params["learning_rate"] == 0.001
        assert params["n_estimators"] == 50

    def test_in_process_training_returns_metrics(self, train_agent):
        import mlflow
        with mlflow.start_run():
            run_id = mlflow.active_run().info.run_id
            model_uri, metrics = train_agent._run_in_process(
                "test-wf", "s3://test/data.csv", {"n_estimators": 10, "max_depth": 3}, run_id
            )
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.5
        assert "runs:/" in model_uri

    def test_full_run_in_process(self, train_agent, sample_state):
        result = train_agent.run(sample_state)
        assert result.status == "success"
        assert result.output["metrics"]["accuracy"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# RL Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

class TestRLOptimizer:
    def test_mlops_env_reset(self):
        from rl_agent.rl_optimizer import MLOpsEnv
        env = MLOpsEnv([])
        obs, _ = env.reset()
        assert obs.shape == (5,)
        assert all(0.0 <= v <= 1.0 for v in obs)

    def test_action_map_coverage(self):
        from rl_agent.rl_optimizer import MLOpsEnv
        env = MLOpsEnv([])
        # All action indices should be in the map
        for i in range(env.action_space.n):
            assert i in MLOpsEnv.ACTION_MAP

    def test_predict_no_model_returns_empty(self):
        from rl_agent.rl_optimizer import predict_adjustments
        with patch("rl_agent.rl_optimizer.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = predict_adjustments({"metrics": {}})
        assert result == {}

    def test_reward_is_positive_for_good_run(self):
        from rl_agent.rl_optimizer import MLOpsEnv
        env = MLOpsEnv([])
        good_run = {"accuracy": 0.95, "latency_s": 10, "cost_usd": 0.01, "drift_score": 0.0, "error_rate": 0.0}
        reward = env._compute_reward(good_run)
        assert reward > 0
