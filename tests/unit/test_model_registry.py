"""
tests/unit/test_model_registry.py

Phase 3 unit tests — Model Registry as Source of Truth.

Coverage:
  A) TagValidator         — tag parsing, type coercion, validation error shapes
  B) ModelRegistryTags    — Pydantic model field constraints and serialisation
  C) RegistrationGate     — tag enforcement, MLflow registration flow, DB writes
  D) LineageGraphBuilder  — node insertion, edge insertion, semantic record helpers
  E) LineageGraphQuerier  — DAG retrieval and BFS topological sort
  F) PromotionStateMachine— legal/illegal transitions, gate outcomes, DB persistence
  G) ModelComparator      — version loading, metric diffs, recommendation, persistence

What is NOT tested here (requires live external services):
  - Real asyncpg connections
  - Real MLflow tracking server
  - Real OPA sidecar

What IS tested (zero external dependencies):
  - All pure logic (validation, gate checks, diff computation)
  - All DB-interaction code via AsyncMock pool patterns
  - All MLflow-interaction code via MagicMock client
  - All OPA-interaction code via mocked aiohttp sessions
  - Error propagation and exception types

Run:
    pytest tests/unit/test_model_registry.py -v
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.registry.metadata_schema import (
    MANDATORY_TAGS_STAGING,
    MANDATORY_TAGS_PRODUCTION,
    ModelRegistryTags,
    RegistrationError,
    TagValidator,
    RegistrationGate,
)
from agents.registry.lineage_graph import (
    EdgeRelationship,
    LineageGraphBuilder,
    LineageGraphQuerier,
    LineageNodeType,
)
from agents.registry.promotion_state_machine import (
    ALLOWED_TRANSITIONS,
    EvalThresholds,
    Gate,
    GATES_FOR_PRODUCTION,
    ModelStage,
    PromotionGateError,
    PromotionRequest,
    PromotionStateMachine,
)
from agents.registry.model_comparator import (
    MetricDiff,
    ModelComparator,
    VersionComparison,
    _HIGHER_IS_BETTER,
    _LOWER_IS_BETTER,
)

# Shared test fixtures
@pytest.fixture
def valid_staging_tags() -> dict[str, str]:
    """Minimal valid tag dict that passes staging validation."""
    return {
        "mlops.dataset.uri":           "s3://mlops-bucket/datasets/train_v3.parquet",
        "mlops.dataset.hash":          "a" * 64,
        "mlops.dataset.row_count":     "125000",
        "mlops.framework":             "xgboost",
        "mlops.framework.version":     "2.1.0",
        "mlops.python.version":        "3.11.6",
        "mlops.eval.accuracy":         "0.923",
        "mlops.eval.f1":               "0.901",
        "mlops.eval.auc":              "0.948",
        "mlops.eval.holdout_hash":     "b" * 64,
        "mlops.eval.bias_passed":      "true",
        "mlops.security.trivy_scan":   "passed",
        "mlops.security.cve_critical": "0",
        "mlops.git.commit":            "c" * 40,
        "mlops.git.repo":              "https://github.com/org/repo",
    }


@pytest.fixture
def valid_production_tags(valid_staging_tags) -> dict[str, str]:
    """Full 18-tag dict that passes production validation."""
    return {
        **valid_staging_tags,
        "mlops.approved_by":          "harish-engineer",
        "mlops.approved_at":          "2025-08-09T12:00:00+00:00",
        "mlops.governance.audit_id":  "d" * 64,
    }


def _make_pool(fetchval=None, fetchrow=None, fetch=None, execute=None) -> MagicMock:
    """
    Build a mock asyncpg pool whose conn.fetchval / fetchrow / fetch / execute
    return the provided values.
    """
    conn = AsyncMock()
    conn.fetchval  = AsyncMock(return_value=fetchval)
    conn.fetchrow  = AsyncMock(return_value=fetchrow)
    conn.fetch     = AsyncMock(return_value=fetch or [])
    conn.execute   = AsyncMock(return_value=execute or "INSERT 0 1")

    pool = MagicMock()
    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__  = AsyncMock(return_value=None)
    return pool

# A) TagValidator
class TestTagValidatorStagingValid:
    def test_passes_all_15_tags_present(self, valid_staging_tags):
        result = TagValidator.validate_for_staging(valid_staging_tags)
        assert isinstance(result, ModelRegistryTags)

    def test_framework_normalised_to_lowercase(self, valid_staging_tags):
        valid_staging_tags["mlops.framework"] = "XGBoost"
        result = TagValidator.validate_for_staging(valid_staging_tags)
        assert result.framework == "xgboost"

    def test_row_count_coerced_to_int(self, valid_staging_tags):
        valid_staging_tags["mlops.dataset.row_count"] = "99999"
        result = TagValidator.validate_for_staging(valid_staging_tags)
        assert result.dataset_row_count == 99999

    def test_accuracy_coerced_to_float(self, valid_staging_tags):
        result = TagValidator.validate_for_staging(valid_staging_tags)
        assert isinstance(result.eval_accuracy, float)
        assert abs(result.eval_accuracy - 0.923) < 1e-9

    def test_bias_passed_true_string(self, valid_staging_tags):
        valid_staging_tags["mlops.eval.bias_passed"] = "True"
        result = TagValidator.validate_for_staging(valid_staging_tags)
        assert result.eval_bias_passed is True

    def test_bias_passed_false_string(self, valid_staging_tags):
        valid_staging_tags["mlops.eval.bias_passed"] = "false"
        result = TagValidator.validate_for_staging(valid_staging_tags)
        assert result.eval_bias_passed is False

    def test_trivy_scan_normalised_to_lowercase(self, valid_staging_tags):
        valid_staging_tags["mlops.security.trivy_scan"] = "Passed"
        result = TagValidator.validate_for_staging(valid_staging_tags)
        assert result.security_trivy_scan == "passed"


class TestTagValidatorStagingErrors:
    def test_raises_on_single_missing_tag(self, valid_staging_tags):
        del valid_staging_tags["mlops.dataset.hash"]
        with pytest.raises(RegistrationError) as exc_info:
            TagValidator.validate_for_staging(valid_staging_tags)
        assert "mlops.dataset.hash" in exc_info.value.missing_tags

    def test_raises_on_multiple_missing_tags(self, valid_staging_tags):
        del valid_staging_tags["mlops.eval.f1"]
        del valid_staging_tags["mlops.security.trivy_scan"]
        with pytest.raises(RegistrationError) as exc_info:
            TagValidator.validate_for_staging(valid_staging_tags)
        assert len(exc_info.value.missing_tags) == 2

    def test_raises_on_all_tags_missing(self):
        with pytest.raises(RegistrationError) as exc_info:
            TagValidator.validate_for_staging({})
        assert len(exc_info.value.missing_tags) == len(MANDATORY_TAGS_STAGING)

    def test_raises_on_non_numeric_row_count(self, valid_staging_tags):
        valid_staging_tags["mlops.dataset.row_count"] = "not_a_number"
        with pytest.raises(RegistrationError) as exc_info:
            TagValidator.validate_for_staging(valid_staging_tags)
        assert "mlops.dataset.row_count" in exc_info.value.invalid_tags

    def test_raises_on_accuracy_above_1(self, valid_staging_tags):
        valid_staging_tags["mlops.eval.accuracy"] = "1.5"
        with pytest.raises(RegistrationError):
            TagValidator.validate_for_staging(valid_staging_tags)

    def test_raises_on_invalid_framework(self, valid_staging_tags):
        valid_staging_tags["mlops.framework"] = "tensorflow"   # not in the allowed set
        with pytest.raises(RegistrationError):
            TagValidator.validate_for_staging(valid_staging_tags)

    def test_raises_on_short_dataset_hash(self, valid_staging_tags):
        valid_staging_tags["mlops.dataset.hash"] = "abc123"   # only 6 chars, needs 64
        with pytest.raises(RegistrationError):
            TagValidator.validate_for_staging(valid_staging_tags)

    def test_raises_on_non_hex_git_commit(self, valid_staging_tags):
        valid_staging_tags["mlops.git.commit"] = "z" * 40   # 'z' is not hex
        with pytest.raises(RegistrationError):
            TagValidator.validate_for_staging(valid_staging_tags)

    def test_raises_on_short_git_commit(self, valid_staging_tags):
        valid_staging_tags["mlops.git.commit"] = "abc"   # too short
        with pytest.raises(RegistrationError):
            TagValidator.validate_for_staging(valid_staging_tags)

    def test_error_to_dict_has_expected_keys(self, valid_staging_tags):
        del valid_staging_tags["mlops.dataset.hash"]
        try:
            TagValidator.validate_for_staging(valid_staging_tags)
        except RegistrationError as exc:
            d = exc.to_dict()
            assert "error" in d
            assert "missing_tags" in d
            assert "invalid_tags" in d


class TestTagValidatorProductionErrors:
    def test_raises_when_approved_by_missing(self, valid_production_tags):
        del valid_production_tags["mlops.approved_by"]
        with pytest.raises(RegistrationError) as exc_info:
            TagValidator.validate_for_production(valid_production_tags)
        assert "mlops.approved_by" in exc_info.value.missing_tags

    def test_raises_when_approved_at_missing(self, valid_production_tags):
        del valid_production_tags["mlops.approved_at"]
        with pytest.raises(RegistrationError) as exc_info:
            TagValidator.validate_for_production(valid_production_tags)
        assert "mlops.approved_at" in exc_info.value.missing_tags

    def test_passes_with_all_18_tags(self, valid_production_tags):
        result = TagValidator.validate_for_production(valid_production_tags)
        assert result.approved_by == "harish-engineer"
        assert result.approved_at is not None

# B) ModelRegistryTags
class TestModelRegistryTagsModel:
    def test_to_mlflow_tags_returns_strings(self, valid_staging_tags):
        tags = TagValidator.validate_for_staging(valid_staging_tags)
        mlflow_dict = tags.to_mlflow_tags()
        for k, v in mlflow_dict.items():
            assert isinstance(v, str), f"Expected string for {k}, got {type(v)}"

    def test_to_mlflow_tags_bias_is_lowercase_true_false(self, valid_staging_tags):
        valid_staging_tags["mlops.eval.bias_passed"] = "true"
        tags = TagValidator.validate_for_staging(valid_staging_tags)
        assert tags.to_mlflow_tags()["mlops.eval.bias_passed"] in ("true", "false")

    def test_to_mlflow_tags_excludes_none_optional_fields(self, valid_staging_tags):
        tags = TagValidator.validate_for_staging(valid_staging_tags)
        mlflow_dict = tags.to_mlflow_tags()
        # approved_by / approved_at / governance_audit_id should be absent
        assert "mlops.approved_by" not in mlflow_dict
        assert "mlops.approved_at" not in mlflow_dict
        assert "mlops.governance.audit_id" not in mlflow_dict

    def test_to_mlflow_tags_includes_optional_when_set(self, valid_production_tags):
        tags = TagValidator.validate_for_production(valid_production_tags)
        mlflow_dict = tags.to_mlflow_tags()
        assert "mlops.approved_by" in mlflow_dict
        assert mlflow_dict["mlops.approved_by"] == "harish-engineer"

    def test_to_db_dict_has_all_columns(self, valid_staging_tags):
        tags = TagValidator.validate_for_staging(valid_staging_tags)
        db = tags.to_db_dict()
        expected_keys = {
            "dataset_uri", "dataset_hash", "dataset_row_count",
            "framework", "framework_version", "python_version",
            "eval_accuracy", "eval_f1", "eval_auc",
            "eval_holdout_hash", "eval_bias_passed",
            "security_trivy_scan", "security_cve_critical",
            "git_commit", "git_repo",
            "approved_by", "approved_at", "governance_audit_id",
        }
        assert expected_keys == set(db.keys())

    def test_model_is_frozen(self, valid_staging_tags):
        """ModelRegistryTags must be immutable after construction."""
        tags = TagValidator.validate_for_staging(valid_staging_tags)
        with pytest.raises(Exception):   # ValidationError or AttributeError
            tags.eval_accuracy = 0.0    # type: ignore[misc]

# C) RegistrationGate
class TestRegistrationGate:
    def _make_gate(self, pool, mlflow_client) -> RegistrationGate:
        return RegistrationGate(pool=pool, mlflow_client=mlflow_client)

    def _make_mlflow_client(self, version: str = "3") -> MagicMock:
        """MLflow client that returns a mock RegisteredModel version."""
        client = MagicMock()
        client.get_registered_model = MagicMock(return_value=MagicMock())
        mock_version = MagicMock()
        mock_version.version = version
        client.create_registered_model = MagicMock()
        with patch("mlflow.register_model", return_value=mock_version):
            pass   # Patch is applied in each test
        return client, mock_version

    async def test_register_returns_version_and_typed_tags(self, valid_staging_tags):
        pool = _make_pool(fetchval=None, execute="INSERT 0 1")
        client = MagicMock()
        client.get_registered_model = MagicMock(return_value=MagicMock())
        mock_version = MagicMock()
        mock_version.version = "5"

        with patch("mlflow.register_model", return_value=mock_version):
            gate = RegistrationGate(pool=pool, mlflow_client=client)
            version, tags = await gate.register(
                run_id="abc123run",
                model_name="fraud-detector",
                raw_tags=valid_staging_tags,
                workflow_id="wf-001",
            )

        assert version == "5"
        assert isinstance(tags, ModelRegistryTags)
        assert tags.framework == "xgboost"

    async def test_register_raises_on_missing_tags(self):
        pool = _make_pool()
        client = MagicMock()
        gate = RegistrationGate(pool=pool, mlflow_client=client)

        with pytest.raises(RegistrationError) as exc_info:
            await gate.register(
                run_id="abc123",
                model_name="fraud-detector",
                raw_tags={"mlops.framework": "xgboost"},   # most tags missing
            )
        assert len(exc_info.value.missing_tags) > 0

    async def test_register_applies_all_tags_to_mlflow_version(self, valid_staging_tags):
        pool = _make_pool(execute="INSERT 0 1")
        client = MagicMock()
        client.get_registered_model = MagicMock(return_value=MagicMock())
        mock_version = MagicMock()
        mock_version.version = "2"

        with patch("mlflow.register_model", return_value=mock_version):
            gate = RegistrationGate(pool=pool, mlflow_client=client)
            await gate.register("run123", "my-model", valid_staging_tags)

        # set_model_version_tag must be called once per tag
        assert client.set_model_version_tag.call_count == len(valid_staging_tags)

    async def test_register_writes_audit_trail_entry(self, valid_staging_tags):
        """An agent_audit_trail INSERT must happen on every successful registration."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        conn.fetchval = AsyncMock(return_value=None)

        pool = MagicMock()
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__  = AsyncMock(return_value=None)

        client = MagicMock()
        client.get_registered_model = MagicMock(return_value=MagicMock())
        mock_ver = MagicMock()
        mock_ver.version = "1"

        with patch("mlflow.register_model", return_value=mock_ver):
            gate = RegistrationGate(pool=pool, mlflow_client=client)
            await gate.register("run999", "my-model", valid_staging_tags)

        # At minimum 2 DB calls: model_registry_tags INSERT + audit_trail INSERT
        assert conn.execute.call_count >= 2

    async def test_mlflow_exception_propagates(self, valid_staging_tags):
        pool = _make_pool()
        client = MagicMock()
        client.get_registered_model = MagicMock(return_value=MagicMock())

        import mlflow.exceptions
        with patch("mlflow.register_model", side_effect=mlflow.exceptions.MlflowException("boom")):
            gate = RegistrationGate(pool=pool, mlflow_client=client)
            with pytest.raises(mlflow.exceptions.MlflowException):
                await gate.register("run000", "my-model", valid_staging_tags)

# D) LineageGraphBuilder
class TestLineageGraphBuilder:
    def _builder(self, pool=None) -> LineageGraphBuilder:
        return LineageGraphBuilder(
            pool=pool or _make_pool(fetchval=42),
            model_name="fraud-detector",
            model_version="3",
        )

    async def test_add_node_returns_db_id(self):
        pool = _make_pool(fetchval=7)
        builder = self._builder(pool)
        node_id = await builder.add_node(
            node_type=LineageNodeType.DATASET,
            external_id="101",
            display_label="Training dataset",
            metadata={"rows": 10000},
        )
        assert node_id == 7

    async def test_add_node_fetches_existing_id_on_conflict(self):
        """ON CONFLICT DO NOTHING returns None; builder must SELECT for existing id."""
        conn = AsyncMock()
        # First fetchval: RETURNING id returns None (conflict)
        # Second fetchval: SELECT id returns 99
        conn.fetchval = AsyncMock(side_effect=[None, 99])
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__  = AsyncMock(return_value=None)

        builder = self._builder(pool)
        node_id = await builder.add_node(
            node_type=LineageNodeType.TRAINING_RUN,
            external_id="run-abc",
            display_label="Training run",
        )
        assert node_id == 99

    async def test_add_edge_executes_insert(self):
        pool = _make_pool()
        builder = self._builder(pool)
        # Must not raise
        await builder.add_edge(1, 2, EdgeRelationship.PRODUCED_FEATURES)

        conn = pool.acquire.return_value.__aenter__.return_value
        conn.execute.assert_called_once()

    async def test_record_data_ingestion_returns_node_id(self):
        pool = _make_pool(fetchval=10)
        builder = self._builder(pool)
        node_id = await builder.record_data_ingestion(
            lineage_id=5,
            content_hash="a" * 64,
            source_uri="s3://bucket/data.parquet",
            row_count=50000,
        )
        assert node_id == 10

    async def test_record_feature_engineering_adds_edge(self):
        """record_feature_engineering must call add_edge after inserting the node."""
        pool = _make_pool(fetchval=20)
        builder = self._builder(pool)

        with patch.object(builder, "add_edge", new_callable=AsyncMock) as mock_edge:
            await builder.record_feature_engineering(
                feast_event_id=3,
                feature_view="fraud_features_v2",
                rows_written=40000,
                parent_dataset_node_id=10,
            )
        mock_edge.assert_called_once_with(
            10, 20, EdgeRelationship.PRODUCED_FEATURES
        )

    async def test_record_training_run_adds_consumed_by_edge(self):
        pool = _make_pool(fetchval=30)
        builder = self._builder(pool)

        with patch.object(builder, "add_edge", new_callable=AsyncMock) as mock_edge:
            await builder.record_training_run(
                run_id="mlflow-run-abc123",
                framework="pytorch",
                hyperparams={"lr": 1e-4, "epochs": 10},
                parent_feat_node_id=20,
            )
        mock_edge.assert_called_once_with(
            20, 30, EdgeRelationship.CONSUMED_BY
        )

    async def test_record_model_artifact_display_label(self):
        pool = _make_pool(fetchval=40)
        builder = self._builder(pool)

        with patch.object(builder, "add_node", new_callable=AsyncMock, return_value=40) as mock_node, \
             patch.object(builder, "add_edge", new_callable=AsyncMock):
            await builder.record_model_artifact("runs:/run123/model", parent_run_node_id=30)

        call_kwargs = mock_node.call_args.kwargs
        assert "Model Artifact" in call_kwargs.get("display_label", "")

    async def test_record_model_version_external_id_format(self):
        pool = _make_pool(fetchval=50)
        builder = self._builder(pool)

        with patch.object(builder, "add_node", new_callable=AsyncMock, return_value=50) as mock_node, \
             patch.object(builder, "add_edge", new_callable=AsyncMock):
            await builder.record_model_version("3", parent_eval_node_id=45)

        call_kwargs = mock_node.call_args.kwargs
        assert call_kwargs["external_id"] == "fraud-detector/3"

# E) LineageGraphQuerier
def _make_node_row(nid: int, node_type: str, external_id: str, label: str) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, key: {
        "id": nid, "node_type": node_type, "external_id": external_id,
        "display_label": label, "metadata": {}, "created_at": None,
    }[key]
    return row


def _make_edge_row(eid: int, parent: int, child: int, rel: str) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, key: {
        "id": eid, "parent_node_id": parent, "child_node_id": child, "relationship": rel,
    }[key]
    return row


class TestLineageGraphQuerier:
    def _querier(self, pool) -> LineageGraphQuerier:
        return LineageGraphQuerier(pool=pool)

    async def test_returns_empty_when_no_nodes(self):
        pool = _make_pool(fetch=[])
        querier = self._querier(pool)
        result = await querier.get_full_lineage("model", "1")
        assert result["nodes"] == []
        assert result["edges"] == []

    async def test_returns_model_name_and_version_in_response(self):
        pool = _make_pool(fetch=[])
        querier = self._querier(pool)
        result = await querier.get_full_lineage("fraud-detector", "5")
        assert result["model_name"] == "fraud-detector"
        assert result["model_version"] == "5"

    async def test_get_lineage_path_returns_empty_for_no_nodes(self):
        pool = _make_pool(fetch=[])
        querier = self._querier(pool)
        # get_lineage_path calls get_full_lineage internally
        with patch.object(querier, "get_full_lineage", new_callable=AsyncMock,
                          return_value={"nodes": [], "edges": []}):
            path = await querier.get_lineage_path("model", "1")
        assert path == []

    async def test_get_lineage_path_topological_order(self):
        """
        Given: dataset(1) → training_run(2) → model_version(3)
        Expected path order: [dataset, training_run, model_version]
        """
        nodes = [
            {"id": 1, "node_type": "dataset", "external_id": "5",
             "display_label": "Dataset", "metadata": {}, "created_at": None},
            {"id": 2, "node_type": "training_run", "external_id": "run-abc",
             "display_label": "Training", "metadata": {}, "created_at": None},
            {"id": 3, "node_type": "model_version", "external_id": "model/1",
             "display_label": "Version", "metadata": {}, "created_at": None},
        ]
        edges = [
            {"id": 1, "parent_node_id": 1, "child_node_id": 2, "relationship": "consumed_by"},
            {"id": 2, "parent_node_id": 2, "child_node_id": 3, "relationship": "produced"},
        ]
        pool = _make_pool()
        querier = self._querier(pool)

        with patch.object(querier, "get_full_lineage", new_callable=AsyncMock,
                          return_value={"model_name": "m", "model_version": "1",
                                        "nodes": nodes, "edges": edges}):
            path = await querier.get_lineage_path("m", "1")

        assert [n["id"] for n in path] == [1, 2, 3]

    async def test_find_by_run_id_returns_none_when_not_found(self):
        pool = _make_pool(fetchrow=None)
        querier = self._querier(pool)
        result = await querier.find_by_run_id("run-does-not-exist")
        assert result is None

# F) PromotionStateMachine
def _make_mlflow_client_with_stage(stage: str, version: str = "3") -> MagicMock:
    client = MagicMock()
    mv = MagicMock()
    mv.current_stage = stage
    mv.tags = {}
    client.get_model_version = MagicMock(return_value=mv)
    client.get_latest_versions = MagicMock(return_value=[])
    client.transition_model_version_stage = MagicMock()
    return client


class TestAllowedTransitions:
    def test_none_can_go_to_staging(self):
        assert ModelStage.STAGING in ALLOWED_TRANSITIONS[ModelStage.NONE]

    def test_staging_can_go_to_production_or_rejected(self):
        allowed = ALLOWED_TRANSITIONS[ModelStage.STAGING]
        assert ModelStage.PRODUCTION in allowed
        assert ModelStage.REJECTED in allowed

    def test_production_can_only_go_to_archived(self):
        allowed = ALLOWED_TRANSITIONS[ModelStage.PRODUCTION]
        assert allowed == frozenset({ModelStage.ARCHIVED})

    def test_archived_is_terminal(self):
        assert len(ALLOWED_TRANSITIONS[ModelStage.ARCHIVED]) == 0

    def test_rejected_is_terminal(self):
        assert len(ALLOWED_TRANSITIONS[ModelStage.REJECTED]) == 0

    def test_gates_for_production_has_5_entries(self):
        assert len(GATES_FOR_PRODUCTION) == 5
        assert Gate.EVALUATION in GATES_FOR_PRODUCTION
        assert Gate.HITL       in GATES_FOR_PRODUCTION
        assert Gate.SECURITY   in GATES_FOR_PRODUCTION
        assert Gate.GOVERNANCE in GATES_FOR_PRODUCTION
        assert Gate.OCI        in GATES_FOR_PRODUCTION


class TestEvalThresholds:
    def test_defaults_are_reasonable(self):
        t = EvalThresholds()
        assert 0.0 < t.min_accuracy < 1.0
        assert 0.0 < t.min_f1 < 1.0
        assert 0.0 < t.min_auc < 1.0

    def test_custom_thresholds_accepted(self):
        t = EvalThresholds(min_accuracy=0.70, min_f1=0.65, min_auc=0.70)
        assert t.min_accuracy == 0.70


class TestEvaluationGate:
    def _sm(self) -> PromotionStateMachine:
        pool = _make_pool(fetchval=99)
        client = _make_mlflow_client_with_stage("Staging")
        return PromotionStateMachine(
            pool=pool,
            mlflow_client=client,
            thresholds=EvalThresholds(min_accuracy=0.80, min_f1=0.75, min_auc=0.75),
        )

    def _make_tags(self, accuracy=0.92, f1=0.90, auc=0.95, bias=True) -> ModelRegistryTags:
        return MagicMock(
            eval_accuracy=accuracy, eval_f1=f1, eval_auc=auc,
            eval_bias_passed=bias,
            security_trivy_scan="passed", security_cve_critical=0,
            approved_by="engineer", approved_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

    async def test_passes_when_all_metrics_above_threshold(self):
        sm = self._sm()
        passed, reason = await sm._check_evaluation_gate(self._make_tags())
        assert passed is True

    async def test_fails_when_accuracy_below_threshold(self):
        sm = self._sm()
        passed, reason = await sm._check_evaluation_gate(self._make_tags(accuracy=0.70))
        assert passed is False
        assert "accuracy" in reason

    async def test_fails_when_f1_below_threshold(self):
        sm = self._sm()
        passed, reason = await sm._check_evaluation_gate(self._make_tags(f1=0.50))
        assert passed is False
        assert "f1" in reason

    async def test_fails_when_auc_below_threshold(self):
        sm = self._sm()
        passed, reason = await sm._check_evaluation_gate(self._make_tags(auc=0.60))
        assert passed is False
        assert "auc" in reason

    async def test_fails_when_bias_not_passed(self):
        sm = self._sm()
        passed, reason = await sm._check_evaluation_gate(self._make_tags(bias=False))
        assert passed is False
        assert "bias" in reason


_UNSET = object()   # sentinel — lets test helpers distinguish "use default" from "pass None"


class TestHITLGate:
    def _sm(self) -> PromotionStateMachine:
        return PromotionStateMachine(pool=_make_pool(), mlflow_client=MagicMock())

    def _make_tags(self, approved_by="engineer", approved_at=_UNSET) -> MagicMock:
        # Use _UNSET as the default so callers can explicitly pass approved_at=None
        # without having it silently replaced by the datetime fallback.
        if approved_at is _UNSET:
            approved_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        return MagicMock(approved_by=approved_by, approved_at=approved_at)

    async def test_passes_with_valid_approval(self):
        sm = self._sm()
        passed, _ = await sm._check_hitl_gate(self._make_tags())
        assert passed is True

    async def test_fails_when_approved_by_is_none(self):
        sm = self._sm()
        passed, reason = await sm._check_hitl_gate(self._make_tags(approved_by=None))
        assert passed is False
        assert "approved_by" in reason

    async def test_fails_when_approved_by_is_empty(self):
        sm = self._sm()
        passed, reason = await sm._check_hitl_gate(self._make_tags(approved_by="   "))
        assert passed is False

    async def test_fails_when_approved_at_is_none(self):
        sm = self._sm()
        passed, reason = await sm._check_hitl_gate(self._make_tags(approved_at=None))
        assert passed is False
        assert "approved_at" in reason

    async def test_fails_when_approved_at_is_in_the_future(self):
        sm = self._sm()
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        passed, reason = await sm._check_hitl_gate(self._make_tags(approved_at=future))
        assert passed is False
        assert "future" in reason


class TestSecurityGate:
    def _sm(self) -> PromotionStateMachine:
        return PromotionStateMachine(pool=_make_pool(), mlflow_client=MagicMock())

    async def test_passes_on_clean_scan(self):
        sm = self._sm()
        tags = MagicMock(security_trivy_scan="passed", security_cve_critical=0)
        passed, _ = await sm._check_security_gate(tags)
        assert passed is True

    async def test_fails_when_trivy_scan_failed(self):
        sm = self._sm()
        tags = MagicMock(security_trivy_scan="failed", security_cve_critical=0)
        passed, reason = await sm._check_security_gate(tags)
        assert passed is False
        assert "passed" in reason.lower() or "trivy" in reason.lower()

    async def test_fails_when_critical_cves_nonzero(self):
        sm = self._sm()
        tags = MagicMock(security_trivy_scan="passed", security_cve_critical=3)
        passed, reason = await sm._check_security_gate(tags)
        assert passed is False
        assert "3" in reason

    async def test_fails_when_both_trivy_failed_and_cves_present(self):
        sm = self._sm()
        tags = MagicMock(security_trivy_scan="failed", security_cve_critical=5)
        passed, _ = await sm._check_security_gate(tags)
        assert passed is False


class TestGovernanceGate:
    def _sm(self, opa_url: str = "http://127.0.0.1:8181") -> PromotionStateMachine:
        return PromotionStateMachine(
            pool=_make_pool(),
            mlflow_client=MagicMock(),
            opa_url=opa_url,
        )

    def _tags(self) -> MagicMock:
        return MagicMock(
            framework="xgboost", eval_accuracy=0.92, eval_f1=0.90,
            eval_bias_passed=True, security_trivy_scan="passed",
            security_cve_critical=0, approved_by="engineer",
        )

    async def test_passes_when_opa_returns_true(self):
        sm = self._sm()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"result": True})

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__  = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_ctx)

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cls.return_value.__aexit__  = AsyncMock(return_value=None)
            passed, reason = await sm._check_governance_gate("model", "3", self._tags(), "wf-1")

        assert passed is True

    async def test_fails_when_opa_returns_false(self):
        sm = self._sm()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"result": False})

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__  = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_ctx)

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cls.return_value.__aexit__  = AsyncMock(return_value=None)
            passed, reason = await sm._check_governance_gate("model", "3", self._tags(), None)

        assert passed is False
        assert "denied" in reason.lower()

    async def test_fails_closed_when_opa_unreachable(self):
        """OPA unreachable must BLOCK the transition (fail-closed per ADR-002)."""
        sm = self._sm()

        with patch("aiohttp.ClientSession") as mock_session_cls:
            import aiohttp as _aiohttp
            mock_session = MagicMock()
            mock_session.post = MagicMock(
                side_effect=_aiohttp.ClientConnectorError(
                    connection_key=MagicMock(), os_error=OSError("refused")
                )
            )
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cls.return_value.__aexit__  = AsyncMock(return_value=None)

            passed, reason = await sm._check_governance_gate("model", "3", self._tags(), None)

        assert passed is False
        assert "fail-closed" in reason.lower() or "unreachable" in reason.lower()


class TestOCIGateStub:
    async def test_oci_gate_blocks_when_no_image_built(self):
        """
        Phase 4 is implemented — the OCI gate is real now (see
        TestRealOCIGate for full coverage). A model with no oci_images row
        (packaging never ran) must be BLOCKED, not waved through — this
        replaces the old Phase 3 stub-always-passes test.
        """
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__  = AsyncMock(return_value=None)
        sm = PromotionStateMachine(pool=pool, mlflow_client=MagicMock())
        passed, reason = await sm._check_oci_gate("model", "3")
        assert passed is False
        assert "No OCI image" in reason


class TestPromotionSignature:
    def test_signature_is_64_char_hex(self):
        sig = PromotionStateMachine._sign_promotion(
            "fraud-detector", "3", "Staging", "Production", "2025-08-09T12:00:00+00:00"
        )
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_same_inputs_produce_same_signature(self):
        args = ("model", "1", "None", "Staging", "2025-01-01T00:00:00+00:00")
        assert PromotionStateMachine._sign_promotion(*args) == \
               PromotionStateMachine._sign_promotion(*args)

    def test_different_stages_produce_different_signatures(self):
        sig1 = PromotionStateMachine._sign_promotion("m", "1", "None", "Staging", "t")
        sig2 = PromotionStateMachine._sign_promotion("m", "1", "None", "Production", "t")
        assert sig1 != sig2


class TestIllegalTransition:
    async def test_archived_to_staging_raises(self):
        pool = _make_pool(fetchval=1)
        client = _make_mlflow_client_with_stage("Archived")
        sm = PromotionStateMachine(pool=pool, mlflow_client=client)

        req = PromotionRequest(
            model_name="model", model_version="1",
            target_stage=ModelStage.STAGING,
            triggered_by="agent", trigger_type="automatic",
        )
        with pytest.raises(ValueError, match="Illegal transition"):
            await sm.promote(req)

    async def test_rejected_to_production_raises(self):
        pool = _make_pool(fetchval=1)
        client = _make_mlflow_client_with_stage("Rejected")
        # MLflow doesn't know "Rejected" — return "Archived" as the stored stage
        sm = PromotionStateMachine(pool=pool, mlflow_client=client)

        with patch.object(sm, "_get_current_stage",
                          new_callable=AsyncMock, return_value=ModelStage.REJECTED):
            req = PromotionRequest(
                model_name="model", model_version="1",
                target_stage=ModelStage.PRODUCTION,
                triggered_by="agent", trigger_type="automatic",
            )
            with pytest.raises(ValueError, match="Illegal transition"):
                await sm.promote(req)

    async def test_none_to_production_raises(self):
        pool = _make_pool(fetchval=1)
        client = _make_mlflow_client_with_stage("None")
        sm = PromotionStateMachine(pool=pool, mlflow_client=client)

        req = PromotionRequest(
            model_name="model", model_version="1",
            target_stage=ModelStage.PRODUCTION,
            triggered_by="agent", trigger_type="automatic",
        )
        with pytest.raises(ValueError, match="Illegal transition"):
            await sm.promote(req)

# G) ModelComparator
def _make_version_row(
    accuracy=0.90, f1=0.88, auc=0.93,
    framework="xgboost", cve=0,
) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, key: {
        "framework":             framework,
        "framework_version":     "2.1.0",
        "python_version":        "3.11.6",
        "dataset_hash":          "a" * 64,
        "dataset_row_count":     100000,
        "eval_accuracy":         accuracy,
        "eval_f1":               f1,
        "eval_auc":              auc,
        "eval_bias_passed":      True,
        "security_trivy_scan":   "passed",
        "security_cve_critical": cve,
        "git_commit":            "c" * 40,
        "git_repo":              "https://github.com/org/repo",
    }[key]
    return row


class TestMetricDiff:
    def test_positive_delta_is_improvement_for_accuracy(self):
        diff = MetricDiff(
            metric="eval_accuracy",
            baseline_version="1", challenger_version="2",
            baseline_value=0.85, challenger_value=0.92,
            delta=0.07, delta_pct=8.24, is_improvement=True,
        )
        assert diff.is_improvement is True

    def test_negative_delta_is_improvement_for_cve_critical(self):
        diff = MetricDiff(
            metric="security_cve_critical",
            baseline_version="1", challenger_version="2",
            baseline_value=3.0, challenger_value=0.0,
            delta=-3.0, delta_pct=-100.0, is_improvement=True,
        )
        assert diff.is_improvement is True

    def test_to_dict_rounds_delta_to_6_places(self):
        diff = MetricDiff(
            metric="eval_f1",
            baseline_version="1", challenger_version="2",
            baseline_value=0.8333333333, challenger_value=0.9111111111,
            delta=0.07777777780000001, delta_pct=9.33, is_improvement=True,
        )
        d = diff.to_dict()
        assert len(str(d["delta"]).rstrip("0").split(".")[-1]) <= 6

    def test_delta_pct_is_none_when_baseline_is_zero(self):
        diff = MetricDiff(
            metric="security_cve_critical",
            baseline_version="1", challenger_version="2",
            baseline_value=0.0, challenger_value=2.0,
            delta=2.0, delta_pct=None, is_improvement=False,
        )
        assert diff.to_dict()["delta_pct"] is None


class TestModelComparatorComputeDiffs:
    def _make_vc(self, version, accuracy, f1, auc, cve=0) -> VersionComparison:
        return VersionComparison(
            model_name="m", version=version,
            framework="xgboost", framework_version="2.0", python_version="3.11",
            dataset_hash="a" * 64, dataset_row_count=100000,
            eval_accuracy=accuracy, eval_f1=f1, eval_auc=auc,
            eval_bias_passed=True,
            security_trivy_scan="passed", security_cve_critical=cve,
            git_commit="c" * 40, git_repo="https://github.com/org/repo",
            current_stage="Staging",
        )

    def test_produces_diff_for_every_numeric_field(self):
        baseline    = self._make_vc("1", 0.85, 0.80, 0.88)
        challenger  = self._make_vc("2", 0.92, 0.89, 0.94)
        diffs = ModelComparator._compute_diffs(baseline, challenger)
        diff_metrics = {d.metric for d in diffs}
        assert "eval_accuracy" in diff_metrics
        assert "eval_f1"       in diff_metrics
        assert "eval_auc"      in diff_metrics
        assert "security_cve_critical" in diff_metrics

    def test_challenger_improvement_marked_correctly_for_f1(self):
        baseline   = self._make_vc("1", 0.85, 0.80, 0.88)
        challenger = self._make_vc("2", 0.92, 0.89, 0.94)
        diffs = ModelComparator._compute_diffs(baseline, challenger)
        f1_diff = next(d for d in diffs if d.metric == "eval_f1")
        assert f1_diff.is_improvement is True
        assert abs(f1_diff.delta - 0.09) < 1e-6

    def test_regression_marked_correctly_for_f1(self):
        baseline   = self._make_vc("1", 0.92, 0.89, 0.94)
        challenger = self._make_vc("2", 0.85, 0.80, 0.88)
        diffs = ModelComparator._compute_diffs(baseline, challenger)
        f1_diff = next(d for d in diffs if d.metric == "eval_f1")
        assert f1_diff.is_improvement is False

    def test_cve_reduction_is_improvement(self):
        baseline   = self._make_vc("1", 0.90, 0.88, 0.93, cve=5)
        challenger = self._make_vc("2", 0.91, 0.89, 0.94, cve=0)
        diffs = ModelComparator._compute_diffs(baseline, challenger)
        cve_diff = next(d for d in diffs if d.metric == "security_cve_critical")
        assert cve_diff.is_improvement is True


class TestModelComparatorCompare:
    def _make_comparator(self, row1, row2) -> tuple[ModelComparator, MagicMock]:
        conn = AsyncMock()
        # fetchrow returns row1 for first call, row2 for second call
        conn.fetchrow = AsyncMock(side_effect=[row1, row2])
        conn.execute  = AsyncMock(return_value="INSERT 0 1")
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__  = AsyncMock(return_value=None)

        client = MagicMock()
        mv = MagicMock()
        mv.current_stage = "Staging"
        mv.run_id = "run-abc"
        mv.tags = {}
        client.get_model_version = MagicMock(return_value=mv)

        return ModelComparator(pool=pool, mlflow_client=client), pool

    async def test_compare_returns_model_comparison(self):
        r1 = _make_version_row(f1=0.88)
        r2 = _make_version_row(f1=0.92)
        comparator, _ = self._make_comparator(r1, r2)

        result = await comparator.compare("fraud-detector", ["1", "2"])
        assert result.model_name == "fraud-detector"
        assert len(result.versions) == 2

    async def test_recommended_version_has_highest_f1(self):
        r1 = _make_version_row(f1=0.80)
        r2 = _make_version_row(f1=0.92)
        comparator, _ = self._make_comparator(r1, r2)

        result = await comparator.compare("model", ["1", "2"])
        assert result.recommended_version == "2"

    async def test_when_baseline_has_higher_f1_it_is_recommended(self):
        r1 = _make_version_row(f1=0.95)
        r2 = _make_version_row(f1=0.82)
        comparator, _ = self._make_comparator(r1, r2)

        result = await comparator.compare("model", ["1", "2"])
        assert result.recommended_version == "1"

    async def test_metric_diffs_reference_correct_versions(self):
        r1 = _make_version_row(f1=0.80)
        r2 = _make_version_row(f1=0.90)
        comparator, _ = self._make_comparator(r1, r2)

        result = await comparator.compare("model", ["1", "2"])
        f1_diff = next(d for d in result.metric_diffs if d.metric == "eval_f1")
        assert f1_diff.baseline_version == "1"
        assert f1_diff.challenger_version == "2"

    async def test_raises_with_fewer_than_2_versions(self):
        comparator, _ = self._make_comparator(
            _make_version_row(), _make_version_row()
        )
        with pytest.raises(ValueError, match="At least 2"):
            await comparator.compare("model", ["1"])

    async def test_raises_with_more_than_4_versions(self):
        # Only 2 rows configured, but the ValueError fires before any DB call
        comparator, _ = self._make_comparator(
            _make_version_row(), _make_version_row()
        )
        with pytest.raises(ValueError, match="Maximum 4"):
            await comparator.compare("model", ["1", "2", "3", "4", "5"])

    async def test_comparison_id_is_uuid_format(self):
        r1 = _make_version_row(f1=0.88)
        r2 = _make_version_row(f1=0.90)
        comparator, _ = self._make_comparator(r1, r2)

        result = await comparator.compare("model", ["1", "2"])
        import uuid
        # Must not raise — a valid UUID can be parsed
        uuid.UUID(result.comparison_id)

    async def test_to_dict_is_json_serialisable(self):
        r1 = _make_version_row(f1=0.88)
        r2 = _make_version_row(f1=0.91)
        comparator, _ = self._make_comparator(r1, r2)

        result = await comparator.compare("model", ["1", "2"])
        # Must not raise
        json.dumps(result.to_dict())

    async def test_fallback_to_mlflow_tags_when_no_db_row(self):
        """If model_registry_tags has no row, comparator must fall back to MLflow tags."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)   # no Neon row
        conn.execute  = AsyncMock(return_value="INSERT 0 1")
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__  = AsyncMock(return_value=None)

        client = MagicMock()
        mv = MagicMock()
        mv.current_stage = "Staging"
        mv.run_id = "run-abc"
        mv.tags = {
            "mlops.framework":             "sklearn",
            "mlops.framework.version":     "1.5.0",
            "mlops.python.version":        "3.11.0",
            "mlops.dataset.hash":          "a" * 64,
            "mlops.dataset.row_count":     "5000",
            "mlops.eval.accuracy":         "0.80",
            "mlops.eval.f1":               "0.78",
            "mlops.eval.auc":              "0.82",
            "mlops.eval.bias_passed":      "true",
            "mlops.security.trivy_scan":   "passed",
            "mlops.security.cve_critical": "0",
            "mlops.git.commit":            "c" * 40,
            "mlops.git.repo":              "https://github.com/org/repo",
        }
        client.get_model_version = MagicMock(return_value=mv)

        comparator = ModelComparator(pool=pool, mlflow_client=client)
        result = await comparator.compare("model", ["1", "2"])
        # Both versions fall back to MLflow tags — framework should be "sklearn"
        assert all(v.framework == "sklearn" for v in result.versions)

    async def test_raises_lookup_error_when_no_row_and_no_mlflow_tags(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        conn.execute  = AsyncMock(return_value="INSERT 0 1")
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__  = AsyncMock(return_value=None)

        client = MagicMock()
        mv = MagicMock()
        mv.current_stage = "None"
        mv.run_id = None
        mv.tags = {}   # no tags at all
        client.get_model_version = MagicMock(return_value=mv)

        comparator = ModelComparator(pool=pool, mlflow_client=client)
        with pytest.raises(LookupError):
            await comparator.compare("model", ["1", "2"])