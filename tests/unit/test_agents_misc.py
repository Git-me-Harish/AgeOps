# tests/unit/test_agents_misc.py
"""
Unit tests for Evaluation Agent, Governance Agent, Training Agent
(dev in-process fallback), and RL Optimizer.

This file previously targeted an older agent API (EvaluationAgent.
_check_thresholds/_run_mlflow_evaluate, GovernanceAgent._build_audit_record/
_sign_record/a 2-arg sync check_policy, TrainingAgent._merge_hyperparams/
_run_in_process) that was superseded by the V2 rewrites and no longer exists
anywhere in the codebase — every test in TestEvaluationAgent/
TestGovernanceAgent/TestTrainingAgent failed or errored. Rewritten against
the real, current methods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.evaluation_agent import EvaluationAgent, THRESHOLDS
from agents.governance_agent import GovernanceAgent
from agents.training_agent import TrainingAgent, TrainingJobConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Agent
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def eval_agent() -> EvaluationAgent:
    return EvaluationAgent()


class TestEvaluationAgent:
    """
    The old threshold gate lived in a standalone _check_thresholds() method;
    the real V2 agent inlines the same THRESHOLDS-dict comparison directly
    in _run_async. These tests exercise the real standalone pieces that DO
    exist instead: _compute_standard_metrics, _check_calibration, and
    _bias_check — all real sklearn/fairness math, not mocked.
    """

    def test_thresholds_constant_has_expected_keys(self, eval_agent):
        # THRESHOLDS is what the real inline gate in _run_async compares
        # std_metrics against — pin its shape so a rename doesn't silently
        # disable the gate.
        assert set(THRESHOLDS.keys()) == {"accuracy", "f1", "auc"}
        assert all(0.0 <= v <= 1.0 for v in THRESHOLDS.values())

    def test_compute_standard_metrics_perfect_predictions(self, eval_agent):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.85])
        metrics = eval_agent._compute_standard_metrics(y_true, y_pred, y_proba)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["auc"] == 1.0

    def test_compute_standard_metrics_no_proba_skips_auc(self, eval_agent):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        metrics = eval_agent._compute_standard_metrics(y_true, y_pred, None)
        assert "accuracy" in metrics
        assert "auc" not in metrics

    def test_calibration_well_calibrated_passes(self, eval_agent):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])
        score, passed = eval_agent._check_calibration(y_true, y_proba)
        assert passed is True
        assert score < 0.15

    def test_calibration_poorly_calibrated_fails(self, eval_agent):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.95, 0.9, 0.99, 0.05, 0.1, 0.02])  # confidently wrong
        score, passed = eval_agent._check_calibration(y_true, y_proba)
        assert passed is False
        assert score >= 0.15

    def test_calibration_no_proba_passes_by_default(self, eval_agent):
        score, passed = eval_agent._check_calibration(np.array([0, 1]), None)
        assert passed is True
        assert score == 0.0

    def test_bias_check_no_protected_attributes_returns_empty(self, eval_agent):
        df = pd.DataFrame({"x": [1, 2, 3, 4]})
        reports = eval_agent._bias_check(df, np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]))
        assert reports == []

    def test_bias_check_flags_demographic_parity_gap(self, eval_agent):
        # Group A always predicted positive, group B always predicted negative
        # — as large a demographic-parity gap as is possible to construct.
        df = pd.DataFrame({"category": ["A"] * 10 + ["B"] * 10})
        y_true = np.array([1] * 20)
        y_pred = np.array([1] * 10 + [0] * 10)
        reports = eval_agent._bias_check(df, y_true, y_pred)
        assert len(reports) == 1
        assert reports[0].attribute == "category"
        assert reports[0].demographic_parity_diff > 0.5
        assert reports[0].passed is False

    def test_run_no_model_uri_fails(self, eval_agent, sample_state):
        state = {**sample_state, "best_run_id": ""}
        result = eval_agent.run(state)
        assert result.status == "failed"
        assert result.error is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Governance Agent
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def gov_agent() -> GovernanceAgent:
    return GovernanceAgent()


class TestGovernanceAgent:
    """
    check_policy() itself (real OPA-backed, fail-closed-in-production logic)
    already has 15 real `opa eval` subprocess tests in tests/test_opa_policy.py
    — not duplicated here. These cover audit() (the Phase 5 post-workflow
    compliance node, previously untested) and _tag_mlflow_run's non-fatal
    error handling, both against the real current implementation.
    """

    def test_audit_returns_success_with_signature(self, gov_agent, sample_state):
        gov_agent._pool = None  # no DB — audit() must still succeed
        result = gov_agent.audit(sample_state)
        assert result.status == "success"
        assert "audit_signature" in result.output
        assert len(result.output["audit_signature"]) == 64  # SHA-256 hex
        assert "summary" in result.output

    def test_audit_signature_is_deterministic_for_same_summary(self, gov_agent, sample_state):
        gov_agent._pool = None
        sig1 = gov_agent.audit(sample_state).output["audit_signature"]
        sig2 = gov_agent.audit(sample_state).output["audit_signature"]
        assert sig1 == sig2

    def test_audit_signature_changes_when_errors_present(self, gov_agent, sample_state):
        gov_agent._pool = None
        clean_result = gov_agent.audit(sample_state)
        errored_state = {**sample_state, "errors": ["something broke"]}
        errored_result = gov_agent.audit(errored_state)
        assert clean_result.output["audit_signature"] != errored_result.output["audit_signature"]

    def test_audit_reflects_deployed_status(self, gov_agent, sample_state):
        gov_agent._pool = None
        deployed_state = {**sample_state, "current_stage": "done", "errors": []}
        result = gov_agent.audit(deployed_state)
        assert result.output["summary"]["deployed"] is True

        failed_state = {**sample_state, "current_stage": "error", "errors": ["boom"]}
        result2 = gov_agent.audit(failed_state)
        assert result2.output["summary"]["deployed"] is False

    async def test_audit_persists_to_pool_when_available(self, gov_agent, sample_state):
        conn = AsyncMock()
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        gov_agent._pool = pool

        await gov_agent._audit_async(sample_state)

        conn.execute.assert_awaited_once()
        args = conn.execute.call_args.args
        assert "agent_audit_trail" in args[0]

    def test_tag_mlflow_run_failure_is_non_fatal(self, gov_agent):
        # _tag_mlflow_run must never raise even if MLflow itself is broken —
        # governance persistence failing should not take down the pipeline.
        import asyncio
        from agents.governance_agent import PolicyDecision

        decision = PolicyDecision(
            allowed=True, deny_reasons=[], policy_name="mlops_policy",
            evaluation_time_ms=1.0, was_sidecar=True, policy_input_hash="a" * 64,
        )
        with patch("mlflow.tracking.MlflowClient", side_effect=RuntimeError("mlflow down")):
            asyncio.run(gov_agent._tag_mlflow_run("fake-run-id", decision, "promote_to_staging"))
        # No exception propagated — that's the assertion.


# ═══════════════════════════════════════════════════════════════════════════════
# Training Agent — dev in-process fallback (real sklearn training, mocked
# only at the ConnectorFactory I/O boundary — same pattern test_data_agent.py
# uses for DataAgent)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def train_agent() -> TrainingAgent:
    # TrainingAgent() itself never touches the kubernetes package — the
    # k8s client is only initialised later via the async connect() ->
    # _init_k8s() path, so _k8s_client is already None here, which is
    # exactly what selects the dev in-process path under test.
    agent = TrainingAgent()
    assert agent._k8s_client is None
    return agent


def _fake_classification_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    label = (x0 + x1 > 0).astype(int)  # learnable real signal, not noise
    return pd.DataFrame({"x0": x0, "x1": x1, "label": label})


class TestTrainingAgentDevMode:
    async def test_simulate_job_dev_trains_on_real_data_and_reports_metrics(self, train_agent):
        """
        Direct regression test for the Phase 1-4 fix that removed
        sklearn.datasets.make_classification() from this path: the model
        must be fit on whatever ConnectorFactory hands back, not synthetic
        noise generated internally.
        """
        import mlflow

        mock_result = MagicMock()
        mock_result.dataframe = _fake_classification_df()
        mock_connector = MagicMock()
        mock_connector.connect = AsyncMock()
        mock_connector.read = AsyncMock(return_value=mock_result)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
        cfg = TrainingJobConfig(
            job_name="test-job", framework="sklearn", runner_image="n/a",
            dataset_uri="s3://bucket/data.csv", feast_feature_view="v1",
            mlflow_run_id=run_id, mlflow_tracking_uri="sqlite:///test.db",
            hyperparams={"learning_rate": 0.05, "n_estimators": 20},
        )

        with patch("agents.connectors.ConnectorFactory.from_uri", return_value=mock_connector):
            status, error = await train_agent._simulate_job_dev(cfg)

        assert status == "succeeded"
        assert error is None

        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run_id).data
        assert run_data.metrics["accuracy"] > 0.5  # learnable signal, should beat chance
        assert run_data.params["learning_rate"] == "0.05"

    async def test_simulate_job_dev_fails_loudly_without_dataset_uri(self, train_agent):
        cfg = TrainingJobConfig(
            job_name="test-job", framework="sklearn", runner_image="n/a",
            dataset_uri="", feast_feature_view="v1",
            mlflow_run_id="fake", mlflow_tracking_uri="sqlite:///test.db",
            hyperparams={},
        )
        status, error = await train_agent._simulate_job_dev(cfg)
        assert status == "failed"
        assert "dataset_uri" in error

    async def test_simulate_job_dev_fails_loudly_on_no_numeric_columns(self, train_agent):
        mock_result = MagicMock()
        mock_result.dataframe = pd.DataFrame({"label": ["a", "b", "a", "b"]})
        mock_connector = MagicMock()
        mock_connector.connect = AsyncMock()
        mock_connector.read = AsyncMock(return_value=mock_result)

        cfg = TrainingJobConfig(
            job_name="test-job", framework="sklearn", runner_image="n/a",
            dataset_uri="s3://bucket/data.csv", feast_feature_view="v1",
            mlflow_run_id="fake", mlflow_tracking_uri="sqlite:///test.db",
            hyperparams={},
        )
        with patch("agents.connectors.ConnectorFactory.from_uri", return_value=mock_connector):
            status, error = await train_agent._simulate_job_dev(cfg)
        assert status == "failed"
        assert "numeric" in error.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# RL Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

class TestRLOptimizer:
    def test_mlops_env_reset(self):
        from rl_agent.rl_optimizer import MLOpsEnv
        env = MLOpsEnv([])
        obs, _ = env.reset()
        # 7 dims as of Phase 5: + deployment_success, + drift_at_30d
        assert obs.shape == (MLOpsEnv.N_OBS_DIMS,)
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
