"""
Alembic migration 0003 — LLM Gateway & Agent Intelligence tables for V2 Phase 2.

New tables:
  prompt_templates   — versioned LLM prompt templates stored in Neon
  token_usage        — per-agent, per-workflow token cost tracking
  llm_cache          — Redis-backed semantic cache metadata + pgvector embeddings
  execution_plans    — planner output for HITL review before execution
  policy_decisions   — every OPA governance decision, immutable audit log

Run:
  alembic upgrade head

Rollback:
  alembic downgrade 0002_data_infrastructure
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003_llm_gateway"
down_revision = "0002_data_infrastructure"
branch_labels = None
depends_on = None


def upgrade() -> None:

    # prompt_templates
    # Versioned prompt templates per agent role.
    # Agents retrieve by (role, version) — never hardcode prompts in agent code.
    # content is the raw Jinja2 template string.
    # variables is a JSONB array of variable names the template expects.
    op.create_table(
        "prompt_templates",
        sa.Column("id",           sa.BigInteger,           primary_key=True, autoincrement=True),
        sa.Column("role",         sa.String(64),           nullable=False),   # planner | training | evaluation | governance | security
        sa.Column("version",      sa.String(32),           nullable=False),   # semver
        sa.Column("content",      sa.Text,                 nullable=False),   # Jinja2 template
        sa.Column("variables",    postgresql.JSONB,        nullable=False, server_default="[]"),
        sa.Column("model_hint",   sa.String(64),           nullable=True),    # preferred model for this template
        sa.Column("max_tokens",   sa.Integer,              nullable=True),
        sa.Column("temperature",  sa.Float,                nullable=True),
        sa.Column("deprecated_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("created_at",   sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_prompt_role_version", "prompt_templates", ["role", "version"], unique=True)
    op.create_index("ix_prompt_role",         "prompt_templates", ["role"])

    # token_usage
    # One row per LLM Gateway call. Used for cost tracking and budget enforcement.
    # prompt_tokens + completion_tokens = total_tokens.
    # cost_usd is computed from the model's published pricing at call time.
    op.create_table(
        "token_usage",
        sa.Column("id",                 sa.BigInteger,  primary_key=True, autoincrement=True),
        sa.Column("workflow_id",        sa.String(36),  sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("agent_role",         sa.String(64),  nullable=False),
        sa.Column("model",              sa.String(64),  nullable=False),   # actual model used (may differ from primary if fallback)
        sa.Column("was_fallback",       sa.Boolean,     nullable=False, server_default="false"),  # True if Ollama fallback used
        sa.Column("was_cache_hit",      sa.Boolean,     nullable=False, server_default="false"),  # True if semantic cache served
        sa.Column("prompt_tokens",      sa.Integer,     nullable=False),
        sa.Column("completion_tokens",  sa.Integer,     nullable=False),
        sa.Column("total_tokens",       sa.Integer,     nullable=False),
        sa.Column("cost_usd",           sa.Float,       nullable=True),    # computed at call time
        sa.Column("latency_ms",         sa.Integer,     nullable=True),
        sa.Column("template_id",        sa.BigInteger,  sa.ForeignKey("prompt_templates.id"), nullable=True),
        sa.Column("created_at",         sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_token_usage_workflow",    "token_usage", ["workflow_id"])
    op.create_index("ix_token_usage_agent_role",  "token_usage", ["agent_role"])
    op.create_index("ix_token_usage_created_at",  "token_usage", ["created_at"])

    # llm_cache
    # Semantic cache metadata. The actual cache hit detection uses Redis for
    # speed, but we persist metadata here for:
    #   - Cache analytics (hit rate per agent, most common queries)
    #   - Debugging (what was returned for a given query)
    #   - Eviction policy (prune entries older than N days via cron)
    # embedding column stores a pgvector vector for similarity search.
    # Requires: CREATE EXTENSION IF NOT EXISTS vector; on Neon first.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "llm_cache",
        sa.Column("id",              sa.BigInteger,  primary_key=True, autoincrement=True),
        sa.Column("query_hash",      sa.String(64),  nullable=False, unique=True),  # SHA-256 of canonical query
        sa.Column("agent_role",      sa.String(64),  nullable=False),
        sa.Column("model",           sa.String(64),  nullable=False),
        sa.Column("response_text",   sa.Text,        nullable=False),
        sa.Column("prompt_tokens",   sa.Integer,     nullable=True),
        sa.Column("completion_tokens", sa.Integer,   nullable=True),
        sa.Column("hit_count",       sa.Integer,     nullable=False, server_default="0"),
        sa.Column("last_hit_at",     sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("expires_at",      sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("created_at",      sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    # pgvector index for cosine similarity search — added via raw SQL
    # because SQLAlchemy Column type for vector requires sqlalchemy-pgvector
    op.execute(
        "ALTER TABLE llm_cache ADD COLUMN IF NOT EXISTS embedding vector(1536)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_llm_cache_embedding "
        "ON llm_cache USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
    op.create_index("ix_llm_cache_query_hash",   "llm_cache", ["query_hash"])
    op.create_index("ix_llm_cache_agent_role",   "llm_cache", ["agent_role"])
    op.create_index("ix_llm_cache_expires_at",   "llm_cache", ["expires_at"])

    # Execution_plans
    # Planner Agent output stored for HITL review before execution starts.
    # status: draft → pending_approval → approved → executing → completed | rejected
    # plan_json: full ExecutionPlan as JSONB — typed Pydantic model serialised.
    op.create_table(
        "execution_plans",
        sa.Column("id",                      sa.BigInteger,    primary_key=True, autoincrement=True),
        sa.Column("workflow_id",             sa.String(36),    sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("status",                  sa.String(32),    nullable=False, server_default="pending_approval"),
        sa.Column("plan_json",               postgresql.JSONB, nullable=False),
        sa.Column("estimated_accuracy",      sa.Float,         nullable=True),
        sa.Column("estimated_duration_mins", sa.Integer,       nullable=True),
        sa.Column("frameworks",              postgresql.JSONB, nullable=False, server_default="[]"),   # list of framework strings
        sa.Column("parallel_experiments",    sa.Integer,       nullable=False, server_default="1"),
        sa.Column("human_approval_required", sa.Boolean,       nullable=False, server_default="true"),
        sa.Column("approved_by",             sa.String(120),   nullable=True),
        sa.Column("approved_at",             sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("rejection_reason",        sa.Text,          nullable=True),
        sa.Column("llm_reasoning",           sa.Text,          nullable=True),  # raw LLM chain-of-thought
        sa.Column("created_at",              sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_execution_plans_workflow", "execution_plans", ["workflow_id"])
    op.create_index("ix_execution_plans_status",   "execution_plans", ["status"])
    op.create_index("ix_execution_plans_created",  "execution_plans", ["created_at"])

    # policy_decisions
    # Immutable audit log of every OPA governance evaluation.
    # One row per policy check — never updated, never deleted.
    # policy_input_hash: SHA-256 of the serialised OPA input (tamper evidence).
    # rego_policy_version: the version of the Rego file that evaluated this.
    op.create_table(
        "policy_decisions",
        sa.Column("id",                   sa.BigInteger,    primary_key=True, autoincrement=True),
        sa.Column("workflow_id",          sa.String(36),    sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("policy_name",          sa.String(128),   nullable=False),
        sa.Column("policy_package",       sa.String(128),   nullable=True),   # OPA package path
        sa.Column("rego_policy_version",  sa.String(32),    nullable=True),
        sa.Column("decision",             sa.Boolean,       nullable=False),   # True=allow, False=deny
        sa.Column("deny_reasons",         postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("policy_input_hash",    sa.String(64),    nullable=True),   # SHA-256 of input
        sa.Column("evaluation_time_ms",   sa.Float,         nullable=True),
        sa.Column("agent_role",           sa.String(64),    nullable=True),
        sa.Column("was_sidecar",          sa.Boolean,       nullable=False, server_default="true"),  # False = network call (dev)
        sa.Column("created_at",           sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_policy_decisions_workflow",   "policy_decisions", ["workflow_id"])
    op.create_index("ix_policy_decisions_decision",   "policy_decisions", ["decision"])
    op.create_index("ix_policy_decisions_created_at", "policy_decisions", ["created_at"])
    op.create_index("ix_policy_decisions_policy_name","policy_decisions", ["policy_name"])


def downgrade() -> None:
    op.drop_table("policy_decisions")
    op.drop_table("execution_plans")
    op.execute("DROP INDEX IF EXISTS ix_llm_cache_embedding")
    op.drop_table("llm_cache")
    op.drop_table("token_usage")
    op.drop_table("prompt_templates")