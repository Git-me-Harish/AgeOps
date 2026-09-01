"""
Alembic migration 0007 — Real session storage + RBAC (Phase 6 hardening).

New tables:
  users, accounts, sessions, verification_token
    — the exact schema @auth/pg-adapter's installed source
      (ui/node_modules/@auth/pg-adapter/index.js) queries against, copied
      column-for-column (including the camelCase quoted identifiers the
      adapter's raw SQL expects) rather than guessed from documentation —
      this is what makes NextAuth sessions real Postgres rows instead of
      opaque, non-revocable JWTs. Deleting a row from `sessions` now force-
      logs-out that browser on its next request.
  user_roles
    — one row per authenticated user, real RBAC enforced server-side by
      the Next.js BFF gateway (app/api/gateway/.../route.ts) and by
      middleware.ts. Roles: viewer < operator < approver < admin (see
      migration docstring below for the exact matrix). Not derived from
      GitHub org/team membership — deliberately simple and
      self-contained, admin-managed via the Admin > Users page.

Run:
  alembic upgrade 0007_auth_rbac

Rollback:
  alembic downgrade 0006_monitoring
"""
from __future__ import annotations

from alembic import op

revision = "0007_auth_rbac"
down_revision = "0006_monitoring"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── @auth/pg-adapter schema (exact match to its raw SQL queries) ───
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS verification_token (
            identifier TEXT NOT NULL,
            expires TIMESTAMPTZ NOT NULL,
            token TEXT NOT NULL,
            PRIMARY KEY (identifier, token)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            email VARCHAR(255) UNIQUE,
            "emailVerified" TIMESTAMPTZ,
            image TEXT
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS accounts (
            id SERIAL PRIMARY KEY,
            "userId" INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            type VARCHAR(255) NOT NULL,
            provider VARCHAR(255) NOT NULL,
            "providerAccountId" VARCHAR(255) NOT NULL,
            refresh_token TEXT,
            access_token TEXT,
            expires_at BIGINT,
            id_token TEXT,
            scope TEXT,
            session_state TEXT,
            token_type TEXT,
            UNIQUE (provider, "providerAccountId")
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            "userId" INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires TIMESTAMPTZ NOT NULL,
            "sessionToken" VARCHAR(255) NOT NULL UNIQUE
        )
        """
    )
    op.execute('CREATE INDEX IF NOT EXISTS ix_sessions_user_id ON sessions ("userId")')
    op.execute('CREATE INDEX IF NOT EXISTS ix_accounts_user_id ON accounts ("userId")')

    # ── RBAC ─────────────────────────────────────────────────────────
    # Role matrix (enforced server-side, never client-side-only):
    #   viewer   — read-only everywhere (any authenticated session)
    #   operator — + start workflows, trigger retraining, decide RL recs
    #   approver — + approve/reject workflows, promote models, rollback
    #   admin    — + edit OPA policy, manage user roles
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS user_roles (
            user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            role VARCHAR(16) NOT NULL DEFAULT 'viewer'
                CHECK (role IN ('viewer', 'operator', 'approver', 'admin')),
            granted_by INTEGER REFERENCES users(id),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS user_roles")
    op.execute("DROP TABLE IF EXISTS sessions")
    op.execute("DROP TABLE IF EXISTS accounts")
    op.execute("DROP TABLE IF EXISTS users")
    op.execute("DROP TABLE IF EXISTS verification_token")
