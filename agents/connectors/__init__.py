"""
agents/connectors/__init__.py

DataConnector abstraction layer — the single seam between the Data Agent
and every external data source.

Design decisions:
  - ABC interface ensures every connector is substitutable (LSP)
  - ConnectorFactory resolves URI → concrete class (Open/Closed)
  - DataSchema wraps column-level type constraints for pre-read enforcement
  - AsyncIterator[pd.DataFrame] for streaming sources (Kafka)
  - No silent fallbacks in production — raises ConnectorError explicitly

Supported URI schemes:
  s3://   → R2Connector  (Cloudflare R2 via boto3 S3-compat)
  r2://   → R2Connector  (alias)
  postgresql:// / postgres:// → PostgreSQLConnector
  file:// → FileConnector (CSV, Parquet, JSON, Avro)
"""
from __future__ import annotations

import hashlib
import io
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# Domain types
class ColumnType(str, Enum):
    """Supported column types for schema enforcement."""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    BINARY = "binary"


class ColumnSchema(BaseModel):
    """Schema for a single column."""
    name: str
    dtype: ColumnType
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_length: Optional[int] = None     # for STRING columns
    allowed_values: Optional[list[Any]] = None  # categorical constraint


class DataSchema(BaseModel):
    """
    Versioned schema for a dataset.

    Passed to DataConnector.read() to enforce types at read time.
    Registered in SchemaRegistry before ingestion proceeds.
    """
    dataset_id: str                    # logical dataset name (e.g. "fraud_features_v2")
    version: str = "1.0.0"            # semver
    columns: list[ColumnSchema] = Field(default_factory=list)
    allow_extra_columns: bool = True   # False = strict — extra cols raise SchemaError

    @field_validator("version")
    @classmethod
    def semver_format(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"version must be semver (X.Y.Z), got: {v!r}")
        return v

    @property
    def required_columns(self) -> list[str]:
        return [c.name for c in self.columns if not c.nullable]


class ConnectorType(str, Enum):
    """Supported data source types."""
    R2 = "r2"
    S3 = "s3"
    POSTGRESQL = "postgresql"
    FILE = "file"
    KAFKA = "kafka"


class ConnectorConfig(BaseModel):
    """
    Connector configuration injected at construction time.
    Credentials should come from settings / Kubernetes secrets — never hardcoded.
    """
    source_type: ConnectorType
    uri: str
    # Auth overrides — if None, connector falls back to settings.*
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    region: Optional[str] = None
    # Read options
    max_rows: Optional[int] = None        # None = no limit
    timeout_seconds: int = 60
    extra: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """Typed result returned by DataConnector.read()."""
    dataframe: Any                        # pd.DataFrame — not type-hinted to avoid Pydantic parse
    row_count: int
    col_count: int
    content_hash: str                     # SHA-256 of raw bytes before parsing
    byte_size: int
    format: str                           # csv | parquet | json | avro
    schema_violations: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# Exceptions
class ConnectorError(RuntimeError):
    """Base for all connector failures — always caught explicitly, never broadly."""

class AuthenticationError(ConnectorError):
    """Raised when credential validation fails."""

class SchemaViolationError(ConnectorError):
    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(f"Schema violations: {violations}")

class DataSourceNotFoundError(ConnectorError):
    """Raised when the target object / table / topic does not exist."""


# Abstract base class for all connectors
class DataConnector(ABC):
    def __init__(self, config: ConnectorConfig) -> None:
        self.config = config
        self._connected: bool = False

    @abstractmethod
    async def connect(self) -> None:
        """
        Validate credentials and test connectivity.
        Must set self._connected = True on success.
        Raises AuthenticationError or ConnectorError on failure.
        """

    @abstractmethod
    async def read(
        self,
        schema: Optional[DataSchema] = None,
    ) -> IngestionResult:
        """
        Read the full dataset and return a typed IngestionResult.

        Args:
            schema: If provided, enforce column types and constraints after read.
                    If schema_registry_block_on_mismatch=True, raises SchemaViolationError.

        Returns:
            IngestionResult with dataframe, content_hash, and violation list.
        """

    @abstractmethod
    async def sample(self, n: int = 100) -> pd.DataFrame:
        """
        Return the first `n` rows as a lightweight preview.
        Used by the Planner Agent for fast dataset characterisation.
        """

    async def stream(self) -> AsyncIterator[pd.DataFrame]:
        """
        Stream data as an async generator of DataFrame batches.
        Only implemented by streaming connectors (Kafka).
        Default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming. "
            "Use KafkaConnector for real-time sources."
        )
        # Satisfy AsyncIterator protocol — never reached
        yield pd.DataFrame()  # noqa: unreachable

    # Schema enforcement (shared across all connectors) 
    def _enforce_schema(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
    ) -> list[str]:
        """
        Validate a DataFrame against a DataSchema.

        Returns a list of violation messages (empty = clean).
        Does NOT raise — caller decides whether to block or warn.
        """
        violations: list[str] = []
        col_set = set(df.columns)

        # Missing required columns
        for col_name in schema.required_columns:
            if col_name not in col_set:
                violations.append(f"Required column missing: {col_name!r}")

        # Type and constraint checks per defined column
        dtype_map: dict[ColumnType, list] = {
            ColumnType.INT: ["int8", "int16", "int32", "int64", "Int8", "Int16", "Int32", "Int64"],
            ColumnType.FLOAT: ["float16", "float32", "float64", "Float32", "Float64"],
            ColumnType.BOOLEAN: ["bool"],
            ColumnType.DATETIME: ["datetime64[ns]", "datetime64[ns, UTC]"],
            ColumnType.STRING: ["object", "string"],
        }

        for col_schema in schema.columns:
            col = col_schema.name
            if col not in col_set:
                continue  # already caught above if required

            series = df[col]

            # Nullable check
            null_count = series.isna().sum()
            if not col_schema.nullable and null_count > 0:
                violations.append(f"Column {col!r}: {null_count} null values (nullable=False)")

            # Range checks for numeric columns
            if col_schema.min_value is not None:
                actual_min = series.dropna().min() if len(series.dropna()) > 0 else None
                if actual_min is not None and actual_min < col_schema.min_value:
                    violations.append(
                        f"Column {col!r}: min value {actual_min} < schema min {col_schema.min_value}"
                    )
            if col_schema.max_value is not None:
                actual_max = series.dropna().max() if len(series.dropna()) > 0 else None
                if actual_max is not None and actual_max > col_schema.max_value:
                    violations.append(
                        f"Column {col!r}: max value {actual_max} > schema max {col_schema.max_value}"
                    )

            # Categorical check
            if col_schema.allowed_values is not None:
                invalid_mask = ~series.isin(col_schema.allowed_values) & series.notna()
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    sample = series[invalid_mask].unique()[:5].tolist()
                    violations.append(
                        f"Column {col!r}: {invalid_count} values not in allowed set "
                        f"(sample: {sample})"
                    )

        # Extra columns check (strict mode)
        if not schema.allow_extra_columns:
            expected = {c.name for c in schema.columns}
            extras = col_set - expected
            if extras:
                violations.append(f"Unexpected columns in strict schema: {sorted(extras)}")

        return violations

    # Content hashing (shared) 
    @staticmethod
    def hash_bytes(raw: bytes) -> str:
        """Compute SHA-256 hex digest of raw bytes — used for lineage tamper evidence."""
        return hashlib.sha256(raw).hexdigest()


def _split_postgres_connector_params(uri: str) -> tuple[str, dict[str, Any]]:
    """
    Split a postgresql:// URI into (connection-only DSN, connector extras).

    `table` and `where` are PostgreSQLConnector's own read-configuration
    knobs (see its module docstring), not real PostgreSQL connection
    parameters — asyncpg rejects unknown query params outright. Everything
    else in the query string (sslmode, etc.) is a genuine libpq parameter
    and is left on the DSN untouched.
    """
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

    parts = urlsplit(uri)
    query_pairs = parse_qsl(parts.query, keep_blank_values=True)

    connector_keys = {"table", "where"}
    extra = {k: v for k, v in query_pairs if k in connector_keys}
    remaining_query = urlencode([(k, v) for k, v in query_pairs if k not in connector_keys])

    base_uri = urlunsplit((parts.scheme, parts.netloc, parts.path, remaining_query, parts.fragment))
    return base_uri, extra


# Factory
class ConnectorFactory:
    """
    Resolves a URI string to the appropriate DataConnector implementation.

    Usage:
        connector = ConnectorFactory.from_uri("s3://my-bucket/data/train.parquet")
        connector = ConnectorFactory.from_uri("postgresql://user:pw@host/db?table=features")
        connector = ConnectorFactory.from_uri("file:///tmp/uploads/data.csv")
    """

    @staticmethod
    def from_uri(uri: str, **kwargs: Any) -> DataConnector:
        """
        Instantiate the correct connector from a URI.

        Extra kwargs are forwarded to ConnectorConfig.extra — use for
        bucket name overrides, table names, Kafka topic names, etc.
        """
        # Import here to avoid circular imports between connector modules
        from agents.connectors.r2_connector import R2Connector
        from agents.connectors.postgres_connector import PostgreSQLConnector
        from agents.connectors.file_connector import FileConnector

        if uri.startswith(("s3://", "r2://")):
            config = ConnectorConfig(
                source_type=ConnectorType.R2,
                uri=uri,
                **kwargs,
            )
            return R2Connector(config)

        if uri.startswith(("postgresql://", "postgres://")):
            # The documented URI form (postgresql://...?table=foo&where=...)
            # was never actually parsed anywhere: PostgreSQLConnector._build_dsn()
            # forwarded the whole URI — table/where included — straight to
            # asyncpg.create_pool() as the DSN, which asyncpg then tried to
            # interpret as real Postgres connection/session parameters,
            # raising "unrecognized configuration parameter". Every caller
            # that relied on the query-string form (rather than passing
            # table=... as a from_uri kwarg directly) has always failed at
            # connect time — found by exercising it against a real Postgres
            # instance for the first time, not by inspection.
            base_uri, connector_extra = _split_postgres_connector_params(uri)
            merged_extra = {**connector_extra, **kwargs.pop("extra", {})}
            config = ConnectorConfig(
                source_type=ConnectorType.POSTGRESQL,
                uri=base_uri,
                extra=merged_extra,
                **kwargs,
            )
            return PostgreSQLConnector(config)

        if uri.startswith("file://") or uri.startswith("/"):
            config = ConnectorConfig(
                source_type=ConnectorType.FILE,
                uri=uri,
                **kwargs,
            )
            return FileConnector(config)

        raise ConnectorError(
            f"Unsupported URI scheme — cannot resolve connector for: {uri!r}. "
            "Supported: s3://, r2://, postgresql://, file://"
        )

    @staticmethod
    def from_config(config: ConnectorConfig) -> DataConnector:
        """Instantiate from a pre-built ConnectorConfig."""
        return ConnectorFactory.from_uri(config.uri)


__all__ = [
    "DataConnector",
    "ConnectorFactory",
    "ConnectorConfig",
    "ConnectorType",
    "DataSchema",
    "ColumnSchema",
    "ColumnType",
    "IngestionResult",
    "ConnectorError",
    "AuthenticationError",
    "SchemaViolationError",
    "DataSourceNotFoundError",
]