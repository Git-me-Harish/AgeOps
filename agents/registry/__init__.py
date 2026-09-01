"""
agents/registry — Phase 3: Model Registry as Source of Truth.

This package owns everything related to model lifecycle management:
  metadata_schema        — 18 mandatory tag definitions, validation, and enforcement
  lineage_graph          — DAG builder and querier for full model provenance
  promotion_state_machine— enforced stage transition rules with multi-gate approval
  model_comparator       — side-by-side metric comparison for N model versions

Design contracts:
  - Every public class method is async (asyncpg pool pattern).
  - All MLflow mutations go through RegistrationGate or PromotionStateMachine —
    never call mlflow.register_model() directly from agent code.
  - Every state transition is persisted to model_promotions (immutable audit log)
    before the MLflow stage is changed.
  - Tag validation is fail-hard: RegistrationError stops the pipeline immediately.
"""
from __future__ import annotations

from agents.registry.metadata_schema import (
    MANDATORY_TAGS_STAGING,
    MANDATORY_TAGS_PRODUCTION,
    ModelRegistryTags,
    RegistrationError,
    TagValidator,
    RegistrationGate,
)
from agents.registry.lineage_graph import (
    LineageNodeType,
    LineageNode,
    LineageEdge,
    LineageGraphBuilder,
    LineageGraphQuerier,
)
from agents.registry.promotion_state_machine import (
    ModelStage,
    Gate,
    PromotionRequest,
    PromotionResult,
    PromotionGateError,
    PromotionStateMachine,
    ALLOWED_TRANSITIONS,
    GATES_FOR_PRODUCTION,
)
from agents.registry.model_comparator import (
    MetricDiff,
    VersionComparison,
    ModelComparison,
    ModelComparator,
)

__all__ = [
    # metadata_schema
    "MANDATORY_TAGS_STAGING",
    "MANDATORY_TAGS_PRODUCTION",
    "ModelRegistryTags",
    "RegistrationError",
    "TagValidator",
    "RegistrationGate",
    # lineage_graph
    "LineageNodeType",
    "LineageNode",
    "LineageEdge",
    "LineageGraphBuilder",
    "LineageGraphQuerier",
    # promotion_state_machine
    "ModelStage",
    "Gate",
    "PromotionRequest",
    "PromotionResult",
    "PromotionGateError",
    "PromotionStateMachine",
    "ALLOWED_TRANSITIONS",
    "GATES_FOR_PRODUCTION",
    # model_comparator
    "MetricDiff",
    "VersionComparison",
    "ModelComparison",
    "ModelComparator",
]