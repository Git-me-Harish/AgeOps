"""
Alembic initial migration — creates all production tables.

Tables
  workflows         — persistent workflow state (checkpointer fallback)
  agent_audit_trail — immutable append-only audit log
  budget_snapshots  — daily resource usage snapshots
  a2a_registry      — persisted agent cards (DB-backed registry)

Run:
  alembic upgrade head
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Alembic revision identifiers
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # workflows 
    op.create_table(
        "workflows",
        sa.Column("id",            sa.String(36),    primary_key=True),
        sa.Column("status",        sa.String(20),    nullable=False, server_default="running"),
        sa.Column("current_stage", sa.String(30),    nullable=True),
        sa.Column("dataset_uri",   sa.Text,          nullable=True),
        sa.Column("model_uri",     sa.Text,          nullable=True),
        sa.Column("metrics",       sa.JSON,          nullable=False, server_default="{}"),
        sa.Column("errors",        sa.JSON,          nullable=False, server_default="[]"),
        sa.Column("trace_id",      sa.String(64),    nullable=True),
        sa.Column("created_at",    sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at",    sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("awaiting_approval", sa.Boolean,  nullable=False, server_default="false"),
        sa.Column("approval_context",  sa.JSON,     nullable=True),
    )
    op.create_index("ix_workflows_status",     "workflows", ["status"])
    op.create_index("ix_workflows_created_at", "workflows", ["created_at"])

    # agent_audit_trail 
    op.create_table(
        "agent_audit_trail",
        sa.Column("id",          sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("workflow_id", sa.String(36), sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("agent_id",    sa.String(64), nullable=False),
        sa.Column("action",      sa.String(120), nullable=False),
        sa.Column("payload",     sa.JSON,        nullable=False, server_default="{}"),
        sa.Column("signature",   sa.String(64),  nullable=True),  # SHA-256 tamper evidence
        sa.Column("ts",          sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_audit_workflow",  "agent_audit_trail", ["workflow_id"])
    op.create_index("ix_audit_agent",     "agent_audit_trail", ["agent_id"])
    op.create_index("ix_audit_ts",        "agent_audit_trail", ["ts"])

    # budget_snapshots 
    op.create_table(
        "budget_snapshots",
        sa.Column("id",               sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("snapshot_date",    sa.Date,       nullable=False),
        sa.Column("r2_storage_gb",    sa.Float,      nullable=True),
        sa.Column("neon_storage_mb",  sa.Float,      nullable=True),
        sa.Column("alerts_fired",     sa.Integer,    nullable=False, server_default="0"),
        sa.Column("raw_payload",      sa.JSON,       nullable=True),
        sa.Column("created_at",       sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_budget_date", "budget_snapshots", ["snapshot_date"], unique=True)

    # a2a_registry (DB-backed agent cards) 
    op.create_table(
        "a2a_registry",
        sa.Column("agent_id",         sa.String(64),  primary_key=True),
        sa.Column("name",             sa.String(120), nullable=False),
        sa.Column("version",          sa.String(20),  nullable=False),
        sa.Column("description",      sa.Text,        nullable=True),
        sa.Column("capabilities",     sa.JSON,        nullable=False, server_default="[]"),
        sa.Column("mcp_tools",        sa.JSON,        nullable=False, server_default="[]"),
        sa.Column("health_endpoint",  sa.String(255), nullable=True),
        sa.Column("max_concurrent",   sa.Integer,     nullable=False, server_default="5"),
        sa.Column("timeout_seconds",  sa.Integer,     nullable=False, server_default="120"),
        sa.Column("hitl_actions",     sa.JSON,        nullable=False, server_default="[]"),
        sa.Column("last_heartbeat",   sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("registered_at",    sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )


def downgrade() -> None:
    op.drop_table("a2a_registry")
    op.drop_table("budget_snapshots")
    op.drop_table("agent_audit_trail")
    op.drop_table("workflows")
