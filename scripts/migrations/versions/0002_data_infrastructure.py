"""
Alembic migration 0002 — Data Infrastructure tables for V2 Phase 1.

New tables:
  data_lineage         — immutable ingestion provenance (content hash, source, schema version)
  schema_registry      — versioned JSON Schema / Avro schema definitions per dataset type
  validation_results   — Great Expectations suite run records (blocking gate outcomes)
  drift_reports        — Evidently drift run records + per-feature scores + R2 report path
  feature_store_events — Feast materialization audit events

Run:
  alembic upgrade head

Rollback:
  alembic downgrade 0001_initial
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# Alembic revision identifiers
revision = "0002_data_infrastructure"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # data_lineage 
    # Every dataset ingested by the Data Agent gets one row here.
    # content_hash is the SHA-256 of the raw bytes — immutable tamper evidence.
    # governance_signature is HMAC-SHA256 of the full row, written by GovernanceAgent.
    op.create_table(
        "data_lineage",
        sa.Column("id",                  sa.BigInteger,  primary_key=True, autoincrement=True),
        sa.Column("workflow_id",         sa.String(36),  sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("source_uri",          sa.Text,        nullable=False),
        sa.Column("source_type",         sa.String(32),  nullable=False),   # r2 | postgres | file | kafka | rest
        sa.Column("content_hash",        sa.String(64),  nullable=False),   # SHA-256 hex
        sa.Column("row_count",           sa.BigInteger,  nullable=True),
        sa.Column("col_count",           sa.Integer,     nullable=True),
        sa.Column("schema_version",      sa.String(32),  nullable=True),    # FK into schema_registry
        sa.Column("schema_registry_id",  sa.BigInteger,  nullable=True),    # FK resolved at ingest time
        sa.Column("byte_size",           sa.BigInteger,  nullable=True),
        sa.Column("format",              sa.String(16),  nullable=True),    # csv | parquet | json | avro
        sa.Column("mlflow_dataset_id",   sa.String(64),  nullable=True),   # mlflow.log_input() ID
        sa.Column("mlflow_run_id",       sa.String(64),  nullable=True),
        sa.Column("governance_signature", sa.String(64), nullable=True),   # tamper evidence
        sa.Column("ingested_at",         sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_lineage_workflow",       "data_lineage", ["workflow_id"])
    op.create_index("ix_lineage_content_hash",   "data_lineage", ["content_hash"])
    op.create_index("ix_lineage_ingested_at",    "data_lineage", ["ingested_at"])
    # Uniqueness: same content at same source_uri is idempotent within a workflow
    op.create_index(
        "ix_lineage_dedup",
        "data_lineage",
        ["workflow_id", "content_hash"],
        unique=True,
    )

    # schema_registry 
    # One row per (dataset_id, version) pair.
    # schema_def stores a JSON Schema or Avro schema as JSONB.
    # deprecated_at is set when a schema version is superseded.
    op.create_table(
        "schema_registry",
        sa.Column("id",               sa.BigInteger,  primary_key=True, autoincrement=True),
        sa.Column("dataset_id",       sa.String(128), nullable=False),  # logical dataset name
        sa.Column("version",          sa.String(32),  nullable=False),  # semver e.g. "1.0.0"
        sa.Column("schema_format",    sa.String(16),  nullable=False),  # jsonschema | avro
        sa.Column("schema_def",       postgresql.JSONB, nullable=False),
        sa.Column("column_count",     sa.Integer,     nullable=True),
        sa.Column("description",      sa.Text,        nullable=True),
        sa.Column("author",           sa.String(120), nullable=True),
        sa.Column("deprecated_at",    sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("created_at",       sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_schema_dataset_version", "schema_registry", ["dataset_id", "version"], unique=True)
    op.create_index("ix_schema_dataset",         "schema_registry", ["dataset_id"])

    # validation_results 
    # One row per Great Expectations suite run.
    # failures stores the list of failing expectation dicts as JSONB.
    # A pipeline is blocked (blocking=True) when passed=False and threshold < pass_threshold.
    op.create_table(
        "validation_results",
        sa.Column("id",              sa.BigInteger,   primary_key=True, autoincrement=True),
        sa.Column("workflow_id",     sa.String(36),   sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("lineage_id",      sa.BigInteger,   sa.ForeignKey("data_lineage.id"), nullable=True),
        sa.Column("dataset_uri",     sa.Text,         nullable=False),
        sa.Column("suite_name",      sa.String(128),  nullable=False),
        sa.Column("passed",          sa.Boolean,      nullable=False),
        sa.Column("pass_rate",       sa.Float,        nullable=False),  # 0.0–1.0
        sa.Column("total_expectations", sa.Integer,   nullable=True),
        sa.Column("passed_expectations", sa.Integer,  nullable=True),
        sa.Column("failures",        postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("blocking",        sa.Boolean,      nullable=False, server_default="false"),
        sa.Column("github_issue_url", sa.Text,        nullable=True),  # auto-created on blocking failure
        sa.Column("run_duration_ms", sa.Integer,      nullable=True),
        sa.Column("created_at",      sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_validation_workflow",    "validation_results", ["workflow_id"])
    op.create_index("ix_validation_passed",      "validation_results", ["passed"])
    op.create_index("ix_validation_created_at",  "validation_results", ["created_at"])

    # drift_reports 
    # One row per Evidently DataDriftPreset run.
    # per_feature_scores is JSONB: {feature_name: drift_score, ...}
    # r2_report_path is the R2 object key where the HTML report is stored.
    op.create_table(
        "drift_reports",
        sa.Column("id",                 sa.BigInteger,   primary_key=True, autoincrement=True),
        sa.Column("workflow_id",        sa.String(36),   sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("current_lineage_id", sa.BigInteger,   sa.ForeignKey("data_lineage.id"), nullable=True),
        sa.Column("reference_uri",      sa.Text,         nullable=True),
        sa.Column("current_uri",        sa.Text,         nullable=False),
        sa.Column("drift_score",        sa.Float,        nullable=False),  # share_of_drifted_columns
        sa.Column("per_feature_scores", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("drifted_features",   postgresql.JSONB, nullable=False, server_default="[]"),  # list of names
        sa.Column("r2_report_path",     sa.Text,         nullable=True),  # R2 object key of HTML
        sa.Column("alert_level",        sa.String(16),   nullable=True),  # none | warning | critical
        sa.Column("triggered_retraining", sa.Boolean,    nullable=False, server_default="false"),
        sa.Column("run_duration_ms",    sa.Integer,      nullable=True),
        sa.Column("created_at",         sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_drift_workflow",     "drift_reports", ["workflow_id"])
    op.create_index("ix_drift_created_at",   "drift_reports", ["created_at"])
    op.create_index("ix_drift_alert_level",  "drift_reports", ["alert_level"])

    # feature_store_events 
    # Audit log for every Feast materialization call.
    op.create_table(
        "feature_store_events",
        sa.Column("id",              sa.BigInteger,  primary_key=True, autoincrement=True),
        sa.Column("workflow_id",     sa.String(36),  sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("lineage_id",      sa.BigInteger,  sa.ForeignKey("data_lineage.id"), nullable=True),
        sa.Column("event_type",      sa.String(32),  nullable=False),  # materialize | online_read | offline_read
        sa.Column("feature_view",    sa.String(128), nullable=True),
        sa.Column("rows_written",    sa.BigInteger,  nullable=True),
        sa.Column("status",          sa.String(16),  nullable=False),  # success | failed
        sa.Column("error_message",   sa.Text,        nullable=True),
        sa.Column("duration_ms",     sa.Integer,     nullable=True),
        sa.Column("created_at",      sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_feast_events_workflow",     "feature_store_events", ["workflow_id"])
    op.create_index("ix_feast_events_created_at",   "feature_store_events", ["created_at"])


def downgrade() -> None:
    op.drop_table("feature_store_events")
    op.drop_table("drift_reports")
    op.drop_table("validation_results")
    op.drop_table("schema_registry")
    op.drop_table("data_lineage")