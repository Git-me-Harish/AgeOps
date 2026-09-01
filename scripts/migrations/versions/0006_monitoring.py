"""
Alembic migration 0006 — Production Observability & Closed-Loop Retraining (Phase 5).

New tables:
  serving_drift_reports — live inference-traffic drift checks (distinct from
                           drift_reports, which is ingestion-time only)
  ground_truth_labels   — delayed labels for production accuracy tracking
  monitoring_events      — per-tick health snapshot (drift/accuracy/latency)
  workflow_triggers      — the retraining queue MonitoringAgent writes to and
                            the Orchestrator polls (the "internal event queue"
                            from the plan, implemented as a durable table
                            instead of an ephemeral pub/sub channel)
  rl_recommendations     — RL Optimizer suggestions with accept/reject state

workflows gets two new columns (source, trigger_type) so a workflow launched
by the monitoring loop is distinguishable from a user-initiated one — this is
also the migration that makes the orchestrator actually write to `workflows`
at all; it previously kept everything in an in-process dict and the LangGraph
checkpointer's opaque tables, so this row-per-workflow table (designed in
migration 0001) had zero writers.

Run:
  alembic upgrade 0006_monitoring

Rollback:
  alembic downgrade 0005_oci_packaging
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006_monitoring"
down_revision = "0005_oci_packaging"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # workflows gains source/trigger tracking
    op.add_column("workflows", sa.Column("source", sa.String(16), nullable=False, server_default="user"))
    op.add_column("workflows", sa.Column("trigger_type", sa.String(32), nullable=True))

    # serving_drift_reports
    # One row per MonitoringAgent tick that ran a real drift check against
    # sampled live inference inputs vs. the model's training dataset.
    op.create_table(
        "serving_drift_reports",
        sa.Column("id",                    sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("model_name",            sa.String(128), nullable=False),
        sa.Column("model_version",         sa.String(32),  nullable=False),
        sa.Column("reference_dataset_uri", sa.Text,        nullable=True),
        sa.Column("sample_window_start",   sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("sample_window_end",     sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("sample_count",          sa.Integer,     nullable=False, server_default="0"),
        sa.Column("drift_score",           sa.Float,       nullable=True),   # NULL = check could not run
        sa.Column("per_feature_scores",    postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("drifted_features",      postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("r2_report_path",        sa.Text,        nullable=True),
        sa.Column("alert_level",           sa.String(16),  nullable=False),  # none|warning|critical|error
        sa.Column("check_error",           sa.Text,        nullable=True),   # populated iff alert_level='error'
        sa.Column("created_at",            sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_servedrift_model_version", "serving_drift_reports", ["model_name", "model_version"])
    op.create_index("ix_servedrift_created_at",    "serving_drift_reports", ["created_at"])
    op.create_index("ix_servedrift_alert_level",   "serving_drift_reports", ["alert_level"])

    # ground_truth_labels
    # Delayed ground-truth labels joined back to a specific prediction by
    # request_id (see serving/inference_server.py's PredictResponse.request_id).
    # Populated by whatever labelling process the deployment has (a human
    # review queue, a downstream outcome event, etc.) — this migration only
    # creates the landing table; no labelling pipeline is implemented here.
    op.create_table(
        "ground_truth_labels",
        sa.Column("id",              sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("request_id",      sa.String(64), nullable=False),
        sa.Column("model_name",      sa.String(128), nullable=False),
        sa.Column("model_version",   sa.String(32),  nullable=False),
        sa.Column("predicted_value", sa.Text,        nullable=True),
        sa.Column("true_value",      sa.Text,        nullable=False),
        sa.Column("correct",         sa.Boolean,      nullable=True),  # NULL until predicted_value is also known
        sa.Column("labeled_at",      sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_gtl_request_id",    "ground_truth_labels", ["request_id"], unique=True)
    op.create_index("ix_gtl_model_version", "ground_truth_labels", ["model_name", "model_version"])
    op.create_index("ix_gtl_labeled_at",    "ground_truth_labels", ["labeled_at"])

    # monitoring_events
    # One row per MonitoringAgent tick — the full health snapshot, regardless
    # of whether any threshold was breached. This is what Phase 5's
    # "historical drift reports" / "monitoring event history" UI reads from.
    op.create_table(
        "monitoring_events",
        sa.Column("id",              sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("model_name",      sa.String(128), nullable=False),
        sa.Column("model_version",   sa.String(32),  nullable=True),
        sa.Column("drift_score",     sa.Float,       nullable=True),
        sa.Column("accuracy",        sa.Float,       nullable=True),   # NULL when no ground truth available yet
        sa.Column("baseline_accuracy", sa.Float,     nullable=True),
        sa.Column("p99_latency_ms",  sa.Float,       nullable=True),
        sa.Column("error_rate",      sa.Float,       nullable=True),
        sa.Column("alert_level",     sa.String(16),  nullable=False),  # none|warning|critical|error
        sa.Column("alerts",          postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("created_at",      sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_monevt_model_version", "monitoring_events", ["model_name", "model_version"])
    op.create_index("ix_monevt_created_at",    "monitoring_events", ["created_at"])
    op.create_index("ix_monevt_alert_level",   "monitoring_events", ["alert_level"])

    # workflow_triggers
    # The retraining queue. MonitoringAgent INSERTs a 'pending' row when a
    # CRITICAL threshold is breached; OrchestratorAgent.process_pending_triggers()
    # (run by scripts/process_triggers.py, e.g. a K8s CronJob every few
    # minutes) claims pending rows, launches a real workflow through the same
    # graph a manual run uses (including the human-approval gate before any
    # deployment), and records the outcome here.
    op.create_table(
        "workflow_triggers",
        sa.Column("id",                  sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("trigger_type",        sa.String(32), nullable=False),  # drift_detected|accuracy_degraded|manual
        sa.Column("source",              sa.String(32), nullable=False, server_default="monitoring"),
        sa.Column("model_name",          sa.String(128), nullable=False),
        sa.Column("model_version",       sa.String(32),  nullable=True),
        sa.Column("dataset_uri",         sa.Text,        nullable=True),
        sa.Column("reason",              sa.Text,        nullable=False),
        sa.Column("status",              sa.String(16),  nullable=False, server_default="pending"),  # pending|processing|completed|failed
        sa.Column("launched_workflow_id", sa.String(36), sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("error_message",       sa.Text,        nullable=True),
        sa.Column("created_at",          sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("processed_at",        sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_index("ix_wftrig_status",      "workflow_triggers", ["status"])
    op.create_index("ix_wftrig_model",       "workflow_triggers", ["model_name", "model_version"])
    op.create_index("ix_wftrig_created_at",  "workflow_triggers", ["created_at"])

    # rl_recommendations
    # RL Optimizer suggestions surfaced to a human via the API (UI accept/
    # reject panel is Phase 6 — this table and its endpoints are Phase 5's
    # scope). agent_role identifies which agent's decision the suggestion
    # applies to, matching the plan's "consumed by Planner/Training/
    # Deployment/Monitoring" language.
    op.create_table(
        "rl_recommendations",
        sa.Column("id",             sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("workflow_id",    sa.String(36), sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("agent_role",     sa.String(32), nullable=False),  # planner|training|deployment|monitoring
        sa.Column("recommendation", postgresql.JSONB, nullable=False),
        sa.Column("confidence",     sa.Float,       nullable=True),
        sa.Column("rationale",      sa.Text,        nullable=True),
        sa.Column("status",         sa.String(16),  nullable=False, server_default="pending"),  # pending|accepted|rejected
        sa.Column("reviewed_by",    sa.String(120), nullable=True),
        sa.Column("reviewed_at",    sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("created_at",     sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_rlrec_status",      "rl_recommendations", ["status"])
    op.create_index("ix_rlrec_agent_role",  "rl_recommendations", ["agent_role"])
    op.create_index("ix_rlrec_created_at",  "rl_recommendations", ["created_at"])


def downgrade() -> None:
    op.drop_table("rl_recommendations")
    op.drop_table("workflow_triggers")
    op.drop_table("monitoring_events")
    op.drop_table("ground_truth_labels")
    op.drop_table("serving_drift_reports")
    op.drop_column("workflows", "trigger_type")
    op.drop_column("workflows", "source")
