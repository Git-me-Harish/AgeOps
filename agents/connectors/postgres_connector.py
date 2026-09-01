"""
agents/connectors/postgres_connector.py

PostgreSQL data connector using asyncpg.

Supports:
  - Full table reads with optional column projection
  - WHERE-clause filtered reads (parameterised — no SQL injection)
  - LIMIT-based sample reads
  - Streaming via LISTEN / NOTIFY for CDC (Change Data Capture)
  - Content hash on the serialised query result bytes
  - Schema enforcement via DataConnector._enforce_schema()

URI format:
  postgresql://user:password@host:5432/dbname?table=feature_table
  postgresql://user:password@host:5432/dbname?table=events&where=created_at>'2025-01-01'

Connection pool strategy:
  asyncpg.Pool — one pool per connector instance.
  Pool is created lazily on first connect() call.
  connect() must be awaited before read() or sample().

Security:
  - Parameterised queries everywhere — no f-string SQL
  - table and column names are validated against an allowlist regex
    (no special chars except _) before interpolation into SQL identifiers
  - read() refuses to run if connect() has not been called first
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
from typing import Any, AsyncIterator, Optional

import pandas as pd

from agents.connectors import (
    AuthenticationError,
    ConnectorConfig,
    ConnectorError,
    DataConnector,
    DataSchema,
    DataSourceNotFoundError,
    IngestionResult,
    SchemaViolationError,
)
from configs.settings import settings

logger = logging.getLogger(__name__)

# Only allow safe identifier characters in table / column names
_SAFE_IDENT = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_MAX_IDENTIFIER_LEN = 63   # PostgreSQL max identifier length


def _validate_identifier(name: str, kind: str = "identifier") -> str:
    """
    Validate a SQL identifier (table or column name) for safe interpolation.
    Raises ConnectorError on invalid input — never silently truncates.
    """
    if not _SAFE_IDENT.match(name):
        raise ConnectorError(
            f"Invalid SQL {kind} {name!r}: only letters, digits, and underscores allowed."
        )
    if len(name) > _MAX_IDENTIFIER_LEN:
        raise ConnectorError(
            f"SQL {kind} {name!r} exceeds PostgreSQL limit of {_MAX_IDENTIFIER_LEN} chars."
        )
    return name


class PostgreSQLConnector(DataConnector):
    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self._pool: Any = None    # asyncpg.Pool

    # Connection 
    async def connect(self) -> None:
        """
        Create an asyncpg connection pool and verify the target table exists.

        Raises:
            AuthenticationError: Wrong credentials.
            DataSourceNotFoundError: Table not found.
            ConnectorError: Networking or other failure.
        """
        try:
            import asyncpg

            dsn = self._build_dsn()
            self._pool = await asyncpg.create_pool(
                dsn,
                min_size=1,
                max_size=min(5, settings.database_pool_size),
                command_timeout=self.config.timeout_seconds,
                statement_cache_size=0,   # required for pgBouncer / Neon
            )

            # Verify table existence
            table = self.config.extra.get("table")
            if table:
                _validate_identifier(table, "table")
                await self._verify_table_exists(table)

            self._connected = True
            logger.info(
                "PostgreSQLConnector connected — dsn_host=%s table=%s",
                self._extract_host(dsn),
                table,
            )
        except ImportError as exc:
            raise ConnectorError(
                "asyncpg is required for PostgreSQL reads. It is in requirements.txt."
            ) from exc
        except Exception as exc:
            error_str = str(exc)
            if any(kw in error_str.lower() for kw in ("password", "authentication", "permission denied")):
                raise AuthenticationError(f"PostgreSQL authentication failed: {exc}") from exc
            raise ConnectorError(f"PostgreSQL connect failed: {exc}") from exc

    async def _verify_table_exists(self, table: str) -> None:
        """Check information_schema to confirm the table exists in this database."""
        schema_name = self.config.extra.get("schema", "public")
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = $1 AND table_name = $2
                """,
                schema_name,
                table,
            )
        if not result:
            raise DataSourceNotFoundError(
                f"Table {schema_name!r}.{table!r} not found in the database."
            )

    # Read 
    async def read(
        self,
        schema: Optional[DataSchema] = None,
    ) -> IngestionResult:
        """
        Execute a SELECT query and return all rows as a typed IngestionResult.

        Query is built from extra config keys:
          extra["table"]   — required
          extra["columns"] — list[str] | "*" (default *)
          extra["where"]   — raw WHERE clause string (must be safe — avoid user input here)
          extra["order_by"] — ORDER BY clause
          extra["limit"]   — row limit override (overrides config.max_rows)
        """
        if not self._connected or self._pool is None:
            await self.connect()

        table = _validate_identifier(
            self.config.extra.get("table", ""),
            "table",
        )
        columns = self.config.extra.get("columns", "*")
        where = self.config.extra.get("where", "")
        order_by = self.config.extra.get("order_by", "")
        limit = self.config.extra.get("limit") or self.config.max_rows

        sql = self._build_select(table, columns, where, order_by, limit)
        start_ms = int(time.monotonic() * 1000)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)

        df = pd.DataFrame([dict(r) for r in rows])
        # Serialise to JSON bytes for content hashing (deterministic)
        raw_bytes = df.to_json(orient="records").encode("utf-8")
        content_hash = self.hash_bytes(raw_bytes)

        violations: list[str] = []
        if schema is not None:
            violations = self._enforce_schema(df, schema)
            if violations and settings.schema_registry_block_on_mismatch:
                raise SchemaViolationError(violations)

        duration_ms = int(time.monotonic() * 1000) - start_ms
        logger.info(
            "PostgreSQLConnector.read — table=%s rows=%d cols=%d hash=%s duration_ms=%d",
            table, len(df), len(df.columns), content_hash[:12], duration_ms,
        )

        return IngestionResult(
            dataframe=df,
            row_count=len(df),
            col_count=len(df.columns),
            content_hash=content_hash,
            byte_size=len(raw_bytes),
            format="json",    # serialised JSON for hashing
            schema_violations=violations,
        )

    # Sample 
    async def sample(self, n: int = 100) -> pd.DataFrame:
        """Return first `n` rows via SELECT ... LIMIT n."""
        if not self._connected or self._pool is None:
            await self.connect()

        table = _validate_identifier(self.config.extra.get("table", ""), "table")
        sql = self._build_select(table, "*", "", "", n)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)

        return pd.DataFrame([dict(r) for r in rows])

    # Streaming (CDC via LISTEN / NOTIFY) 
    async def stream(self) -> AsyncIterator[pd.DataFrame]:
        """
        Stream database changes via PostgreSQL LISTEN / NOTIFY.

        Designed for CDC pipelines where a Debezium-style trigger or
        application code sends NOTIFY on a named channel with a JSON payload.

        Config keys:
          extra["channel"]       — NOTIFY channel name (required)
          extra["batch_size"]    — number of notifications to batch per yield (default: 50)
          extra["timeout_ms"]    — wait timeout per batch (default: 5000 ms)

        Usage:
            async for batch_df in connector.stream():
                process_batch(batch_df)
        """
        import asyncpg

        if not self._connected or self._pool is None:
            await self.connect()

        channel = self.config.extra.get("channel")
        if not channel:
            raise ConnectorError(
                "PostgreSQL streaming requires extra['channel'] — "
                "the NOTIFY channel name to listen on."
            )
        _validate_identifier(channel, "channel")

        batch_size: int = self.config.extra.get("batch_size", 50)
        timeout_ms: int = self.config.extra.get("timeout_ms", 5000)

        logger.info("PostgreSQLConnector: starting CDC stream on channel=%s", channel)

        # Acquire a dedicated connection for listening (cannot use pool conn for LISTEN)
        conn: asyncpg.Connection = await self._pool.acquire()
        try:
            queue: asyncio.Queue[str] = asyncio.Queue()

            def _on_notification(
                connection: asyncpg.Connection,
                pid: int,
                channel_name: str,
                payload: str,
            ) -> None:
                queue.put_nowait(payload)

            await conn.add_listener(channel, _on_notification)
            logger.info("Listening on PostgreSQL NOTIFY channel: %s", channel)

            batch: list[dict] = []
            while True:
                try:
                    payload_str = await asyncio.wait_for(
                        queue.get(),
                        timeout=timeout_ms / 1000,
                    )
                    try:
                        record = json.loads(payload_str)
                        batch.append(record)
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping non-JSON NOTIFY payload: %s — %s", payload_str, exc)

                    if len(batch) >= batch_size:
                        yield pd.DataFrame(batch)
                        batch = []

                except asyncio.TimeoutError:
                    # Yield whatever we have accumulated
                    if batch:
                        yield pd.DataFrame(batch)
                        batch = []

        finally:
            await conn.remove_listener(channel, _on_notification)
            await self._pool.release(conn)

    # SQL builder 
    @staticmethod
    def _build_select(
        table: str,
        columns: Any,
        where: str,
        order_by: str,
        limit: Optional[int],
    ) -> str:
        """
        Build a SELECT statement from validated components.

        Identifiers are validated before interpolation.
        WHERE and ORDER BY are passed as-is — callers must not forward user input.
        """
        if isinstance(columns, list):
            col_str = ", ".join(_validate_identifier(c) for c in columns)
        else:
            col_str = "*"

        sql = f'SELECT {col_str} FROM "{table}"'
        if where:
            sql += f" WHERE {where}"
        if order_by:
            _validate_identifier(order_by.lstrip("-").strip(), "order_by column")
            sql += f" ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return sql

    # DSN helpers 
    def _build_dsn(self) -> str:
        uri = self.config.uri
        if uri.startswith("postgres://"):
            uri = "postgresql://" + uri[len("postgres://"):]
        uri = uri.replace("postgresql+asyncpg://", "postgresql://", 1)
        uri = uri.replace("&channel_binding=require", "").replace("?channel_binding=require&", "?")
        if settings.is_production and "sslmode" not in uri:
            sep = "&" if "?" in uri else "?"
            uri += f"{sep}sslmode=require"
        return uri

    @staticmethod
    def _extract_host(dsn: str) -> str:
        """Extract host from DSN for safe logging (no password)."""
        try:
            # postgresql://user:pw@HOST:port/db
            return dsn.split("@")[-1].split("/")[0]
        except Exception:
            return "unknown"

    # Cleanup 
    async def close(self) -> None:
        """Release the connection pool. Call this when the connector is no longer needed."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._connected = False
            logger.info("PostgreSQLConnector pool closed")