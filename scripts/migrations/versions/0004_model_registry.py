"""
Alembic migration 0004 — Model Registry as Source of Truth (Phase 3).

New tables:
  model_registry_tags  — 18 mandatory tags per (model_name, version), enforced at
                         registration; tags 1–15 required at Staging, all 18 at Production.
  model_lineage_nodes  — typed DAG nodes (dataset → feat_eng → training_run →
                         model_artifact → eval_run → model_version → [oci_image →
                         infer_service → live_traffic] in Phase 4)
  model_lineage_edges  — directed edges between lineage nodes (parent → child)
  model_promotions     — immutable state machine transition log; HMAC-signed; append-only
  model_comparisons    — saved comparison sessions from the UI comparison view

Run:
  alembic upgrade head

Rollback:
  alembic downgrade 0003_llm_gateway
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004_model_registry"
down_revision = "0003_llm_gateway"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # model_registry_tags 
    # One row per (model_name, model_version).
    # Columns map 1:1 to the 18 mandatory tag keys from metadata_schema.py.
    #
    # Split between staging and production requirements:
    #  Tags 1–15 (dataset, framework, eval, security, git)  → required at Staging
    #  Tags 16–18 (approved_by, approved_at, audit_id)     → required at Production
    #
    # approved_by / approved_at / governance_audit_id are nullable at Staging;
    # application-level enforcement in PromotionStateMachine makes them mandatory
    # before any Staging → Production transition proceeds.
    #
    # raw_tags stores the full MLflow tag dict as JSONB for forward compatibility
    # (future tags can be added without a schema migration).
    op.create_table(
        "model_registry_tags",
        sa.Column("id",                   sa.BigInteger,  primary_key=True, autoincrement=True),
        sa.Column("model_name",           sa.String(128), nullable=False),
        sa.Column("model_version",        sa.String(32),  nullable=False),
        # data provenance 
        sa.Column("dataset_uri",          sa.Text,        nullable=False),
        sa.Column("dataset_hash",         sa.String(64),  nullable=False),      # SHA-256 hex
        sa.Column("dataset_row_count",    sa.BigInteger,  nullable=False),
        # framework 
        sa.Column("framework",            sa.String(32),  nullable=False),      # xgboost|pytorch|sklearn|huggingface|custom
        sa.Column("framework_version",    sa.String(32),  nullable=False),
        sa.Column("python_version",       sa.String(16),  nullable=False),      # e.g. 3.11.6
        # evaluation metrics 
        sa.Column("eval_accuracy",        sa.Float,       nullable=False),
        sa.Column("eval_f1",              sa.Float,       nullable=False),
        sa.Column("eval_auc",             sa.Float,       nullable=False),
        sa.Column("eval_holdout_hash",    sa.String(64),  nullable=False),      # SHA-256 of holdout set
        sa.Column("eval_bias_passed",     sa.Boolean,     nullable=False),
        # security 
        sa.Column("security_trivy_scan",  sa.String(16),  nullable=False),      # passed | failed
        sa.Column("security_cve_critical",sa.Integer,     nullable=False),      # count of CRITICAL CVEs
        # git provenance 
        sa.Column("git_commit",           sa.String(40),  nullable=False),      # full 40-char SHA
        sa.Column("git_repo",             sa.Text,        nullable=False),
        # HITL approval (nullable until Staging → Production) 
        sa.Column("approved_by",          sa.String(120), nullable=True),       # GitHub username
        sa.Column("approved_at",          sa.TIMESTAMP(timezone=True), nullable=True),
        # governance 
        sa.Column("governance_audit_id",  sa.String(64),  nullable=True),       # SHA-256 audit trail sig
        # raw snapshot 
        sa.Column("raw_tags",             postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at",           sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at",           sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index(
        "ix_mrt_name_version", "model_registry_tags",
        ["model_name", "model_version"], unique=True,
    )
    op.create_index("ix_mrt_framework",  "model_registry_tags", ["framework"])
    op.create_index("ix_mrt_created_at", "model_registry_tags", ["created_at"])

    # model_lineage_nodes 
    # One row per DAG node.  All nodes for a given (model_name, model_version)
    # form that version's lineage chain.
    #
    # node_type enum values (ordered by DAG position):
    #  dataset       → external_id = data_lineage.id as TEXT
    #  feat_eng      → external_id = feature_store_events.id as TEXT
    #  training_run  → external_id = MLflow run ID
    #  model_artifact→ external_id = MLflow artifact URI
    #  eval_run      → external_id = MLflow run ID (eval run)
    #  model_version → external_id = "<model_name>/<version>"
    #  oci_image     → external_id = "<registry>/<image>@sha256:<digest>" (Phase 4)
    #  infer_service → external_id = KServe CRD name (Phase 4)
    #  live_traffic  → external_id = Prometheus job label (Phase 4+)
    op.create_table(
        "model_lineage_nodes",
        sa.Column("id",            sa.BigInteger,  primary_key=True, autoincrement=True),
        sa.Column("model_name",    sa.String(128), nullable=False),
        sa.Column("model_version", sa.String(32),  nullable=False),
        sa.Column("node_type",     sa.String(32),  nullable=False),   # see node_type enum above
        sa.Column("external_id",   sa.Text,        nullable=False),   # natural ID from the source system
        sa.Column("display_label", sa.Text,        nullable=True),    # human-readable label for the UI DAG
        sa.Column("metadata",      postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at",    sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_mln_model_version", "model_lineage_nodes", ["model_name", "model_version"])
    op.create_index("ix_mln_node_type",     "model_lineage_nodes", ["node_type"])
    op.create_index("ix_mln_external_id",   "model_lineage_nodes", ["external_id"])

    # model_lineage_edges 
    # Directed edges: parent_node_id → child_node_id.
    # relationship labels follow the DAG spec from Phase 3.2:
    #  produced_features  dataset      → feat_eng
    #  consumed_by        feat_eng     → training_run
    #  produced           training_run → model_artifact
    #  registered_as      model_artifact→ model_version
    #  evaluated_by       model_version → eval_run
    #  packaged_as        model_version → oci_image       (Phase 4)
    #  deployed_as        oci_image    → infer_service    (Phase 4)
    #  receives_traffic   infer_service→ live_traffic     (Phase 4+)
    op.create_table(
        "model_lineage_edges",
        sa.Column("id",             sa.BigInteger,  primary_key=True, autoincrement=True),
        sa.Column("parent_node_id", sa.BigInteger,  sa.ForeignKey("model_lineage_nodes.id"), nullable=False),
        sa.Column("child_node_id",  sa.BigInteger,  sa.ForeignKey("model_lineage_nodes.id"), nullable=False),
        sa.Column("relationship",   sa.String(64),  nullable=False),
        sa.Column("created_at",     sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_mle_parent",       "model_lineage_edges", ["parent_node_id"])
    op.create_index("ix_mle_child",        "model_lineage_edges", ["child_node_id"])
    op.create_index("ix_mle_relationship", "model_lineage_edges", ["relationship"])
    # Prevent duplicate edges — same semantic edge must not be inserted twice
    op.create_index(
        "ix_mle_unique",
        "model_lineage_edges",
        ["parent_node_id", "child_node_id", "relationship"],
        unique=True,
    )

    # model_promotions 
    # Immutable promotion event log.  One row per state machine transition.
    # Never updated, never deleted — this is the ground-truth audit trail.
    #
    # gates_passed / gates_failed: JSONB arrays of gate name strings.
    # rejection_reason: free-text, non-null when to_stage = 'Rejected'.
    # signature: HMAC-SHA256 of (model_name, version, from_stage, to_stage, promoted_at)
    #           for tamper detection.
    op.create_table(
        "model_promotions",
        sa.Column("id",               sa.BigInteger,    primary_key=True, autoincrement=True),
        sa.Column("model_name",       sa.String(128),   nullable=False),
        sa.Column("model_version",    sa.String(32),    nullable=False),
        sa.Column("from_stage",       sa.String(32),    nullable=False),         # None|Staging|Production|Archived
        sa.Column("to_stage",         sa.String(32),    nullable=False),         # Staging|Production|Archived|Rejected
        sa.Column("triggered_by",     sa.String(64),    nullable=False),         # agent name or GitHub username
        sa.Column("trigger_type",     sa.String(16),    nullable=False),         # automatic | human
        sa.Column("gates_passed",     postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("gates_failed",     postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("rejection_reason", sa.Text,          nullable=True),
        sa.Column("workflow_id",      sa.String(36),    sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("signature",        sa.String(64),    nullable=True),          # HMAC-SHA256 tamper evidence
        sa.Column("promoted_at",      sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_mp_model_version", "model_promotions", ["model_name", "model_version"])
    op.create_index("ix_mp_to_stage",      "model_promotions", ["to_stage"])
    op.create_index("ix_mp_promoted_at",   "model_promotions", ["promoted_at"])
    op.create_index("ix_mp_workflow_id",   "model_promotions", ["workflow_id"])

    # model_comparisons 
    # Saved comparison sessions initiated from the UI's model comparison view.
    # model_versions: JSONB array of {model_name, version} objects.
    # comparison_result: JSONB comparison matrix (per-version, per-metric).
    # decision: optional free-text note written by the engineer after reviewing.
    op.create_table(
        "model_comparisons",
        sa.Column("id",                 sa.BigInteger,    primary_key=True, autoincrement=True),
        sa.Column("comparison_id",      sa.String(36),    nullable=False, unique=True),  # UUID
        sa.Column("model_versions",     postgresql.JSONB, nullable=False),               # [{name,version},...]
        sa.Column("comparison_result",  postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("initiated_by",       sa.String(120),   nullable=True),                # GitHub username or agent
        sa.Column("decision",           sa.Text,          nullable=True),                # "Use v3 for production"
        sa.Column("workflow_id",        sa.String(36),    sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("created_at",         sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_mc_comparison_id", "model_comparisons", ["comparison_id"])
    op.create_index("ix_mc_created_at",    "model_comparisons", ["created_at"])


def downgrade() -> None:
    op.drop_table("model_comparisons")
    op.drop_table("model_promotions")
    # Drop edges before nodes (FK constraint)
    op.drop_table("model_lineage_edges")
    op.drop_table("model_lineage_nodes")
    op.drop_table("model_registry_tags")