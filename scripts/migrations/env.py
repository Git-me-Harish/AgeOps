from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from configs.settings import settings

# Alembic config object
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# No ORM Base in use — raw DDL migrations only.
# Set to Base.metadata if you add SQLAlchemy models later.
target_metadata = None

# Inject the sync URL so alembic.ini doesn't need to hardcode credentials
sync_url = settings.database_sync_url

if not sync_url or sync_url == "postgresql://user:pass@host/db?sslmode=require":
    raise RuntimeError(
        "DATABASE_SYNC_URL is not set or still has the placeholder value.\n"
        "Set it in your .env to: postgresql://user:pass@host/db?sslmode=require\n"
        "This is the psycopg2 (sync) URL — same Neon host as DATABASE_URL "
        "but without the +asyncpg prefix and without channel_binding."
    )

config.set_main_option("sqlalchemy.url", sync_url)


def run_migrations_offline() -> None:
    """
    Run migrations without a live DB connection.
    Produces a SQL script you can inspect or apply manually.
    Usage: alembic upgrade head --sql
    """
    context.configure(
        url=sync_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations against a live Neon PostgreSQL connection.
    Uses NullPool so connections are not kept alive after the migration completes —
    important for serverless / short-lived processes (CI, K8s job).
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,   # don't reuse connections in a migration CLI run
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Compare server defaults so Alembic detects DEFAULT changes
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()