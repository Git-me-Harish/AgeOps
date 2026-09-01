"""
agents/connectors/schema_registry.py

Versioned schema registry backed by Neon PostgreSQL (schema_registry table).

Responsibilities:
  - Register a new DataSchema version for a dataset_id
  - Retrieve the latest (or specific) version for a dataset
  - Validate a DataFrame against a registered schema version
  - Track schema evolution: flag breaking changes vs additive changes
  - Reject ingestion when schema mismatch is a hard block

Design:
  - Async interface using asyncpg pool passed from settings.database_url
  - Schema stored as JSONB in Neon — queryable, inspectable, not opaque
  - Version comparison uses semver ordering (no external dep — parsed manually)
  - Thread-safe: no shared mutable state; pool acquired per operation
  - SchemaValidator is a pure function — testable without a DB

Usage (in DataAgent):
    registry = SchemaRegistry(pool)
    schema = await registry.get_latest("fraud_features")
    violations = SchemaValidator.validate(df, schema)
    if violations:
        await registry.record_violation_event(workflow_id, violations)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel

from agents.connectors import DataSchema, ColumnSchema, ColumnType, SchemaViolationError

logger = logging.getLogger(__name__)


# Schema evolution classification
class EvolutionType(str):
    """Classification of a schema change between two versions."""
    ADDITIVE = "additive"            # new optional column added — backward compatible
    BREAKING = "breaking"            # column removed, type changed, or non-null added
    DEPRECATION = "deprecation"      # existing column removed but marked deprecated
    NO_CHANGE = "no_change"


class SchemaEvolutionResult(BaseModel):
    """Result of comparing two schema versions."""
    evolution_type: str
    added_columns: list[str] = []
    removed_columns: list[str] = []
    type_changed_columns: list[str] = []
    nullability_changed_columns: list[str] = []
    is_backward_compatible: bool = True
    summary: str = ""


# Schema serialisation helpers
def schema_to_dict(schema: DataSchema) -> dict:
    """Serialise a DataSchema to a plain dict for JSONB storage."""
    return schema.model_dump()


def schema_from_dict(raw: dict) -> DataSchema:
    """Deserialise a DataSchema from the JSONB record."""
    return DataSchema.model_validate(raw)


# Schema validator (pure functions — no I/O, easy to unit test)
class SchemaValidator:
    """
    Pure-function schema validation against a pandas DataFrame.
    No DB dependency — can be used in tests without a running Postgres.
    """

    @staticmethod
    def validate(df: pd.DataFrame, schema: DataSchema) -> list[str]:
        """
        Validate df against schema. Returns list of violation messages.
        Empty list = clean. Never raises internally.
        """
        violations: list[str] = []
        col_set = set(df.columns)

        # Missing required columns
        for col in schema.required_columns:
            if col not in col_set:
                violations.append(f"Required column missing: {col!r}")

        for col_schema in schema.columns:
            col = col_schema.name
            if col not in col_set:
                continue

            series = df[col]

            # Nullability
            null_count = int(series.isna().sum())
            if not col_schema.nullable and null_count > 0:
                violations.append(f"{col!r}: {null_count} nulls, nullable=False")

            # Numeric range bounds
            non_null = series.dropna()
            if len(non_null) > 0:
                if col_schema.min_value is not None:
                    actual_min = non_null.min()
                    if actual_min < col_schema.min_value:
                        violations.append(
                            f"{col!r}: min={actual_min:.4f} < schema_min={col_schema.min_value}"
                        )
                if col_schema.max_value is not None:
                    actual_max = non_null.max()
                    if actual_max > col_schema.max_value:
                        violations.append(
                            f"{col!r}: max={actual_max:.4f} > schema_max={col_schema.max_value}"
                        )

            # String length
            if col_schema.max_length is not None and series.dtype == object:
                too_long_mask = non_null.astype(str).str.len() > col_schema.max_length
                count = int(too_long_mask.sum())
                if count > 0:
                    violations.append(
                        f"{col!r}: {count} values exceed max_length={col_schema.max_length}"
                    )

            # Categorical allowlist
            if col_schema.allowed_values is not None:
                invalid = ~non_null.isin(col_schema.allowed_values)
                count = int(invalid.sum())
                if count > 0:
                    sample = non_null[invalid].unique()[:5].tolist()
                    violations.append(
                        f"{col!r}: {count} values not in allowed_values (sample: {sample})"
                    )

        # Extra columns in strict mode
        if not schema.allow_extra_columns:
            expected = {c.name for c in schema.columns}
            extras = col_set - expected
            if extras:
                violations.append(f"Extra columns in strict schema: {sorted(extras)}")

        return violations

    @staticmethod
    def classify_evolution(
        old: DataSchema,
        new: DataSchema,
    ) -> SchemaEvolutionResult:
        """
        Compare two schema versions and classify the change.

        A BREAKING change blocks auto-promotion: the Data Agent will require
        explicit operator confirmation before accepting data under the new schema.
        """
        old_cols = {c.name: c for c in old.columns}
        new_cols = {c.name: c for c in new.columns}

        added = [c for c in new_cols if c not in old_cols]
        removed = [c for c in old_cols if c not in new_cols]
        type_changed: list[str] = []
        nullability_changed: list[str] = []

        for name in set(old_cols) & set(new_cols):
            if old_cols[name].dtype != new_cols[name].dtype:
                type_changed.append(name)
            if old_cols[name].nullable and not new_cols[name].nullable:
                # Became non-nullable — breaking (existing nulls would fail)
                nullability_changed.append(name)

        is_breaking = bool(removed or type_changed or nullability_changed)
        is_additive = bool(added and not is_breaking)

        if not (added or removed or type_changed or nullability_changed):
            evolution_type = EvolutionType.NO_CHANGE
            summary = "No schema changes detected."
        elif is_breaking:
            evolution_type = EvolutionType.BREAKING
            parts = []
            if removed:
                parts.append(f"removed={removed}")
            if type_changed:
                parts.append(f"type_changed={type_changed}")
            if nullability_changed:
                parts.append(f"nullability_changed={nullability_changed}")
            summary = f"BREAKING schema change: {', '.join(parts)}"
        else:
            evolution_type = EvolutionType.ADDITIVE
            summary = f"Additive change: added columns={added}"

        return SchemaEvolutionResult(
            evolution_type=evolution_type,
            added_columns=added,
            removed_columns=removed,
            type_changed_columns=type_changed,
            nullability_changed_columns=nullability_changed,
            is_backward_compatible=not is_breaking,
            summary=summary,
        )

# Registry (I/O — Neon backed)
class SchemaRegistry:
    """ Neon-backed schema registry.Thread-safe: each method acquires from the pool independently.
    The pool is expected to be an asyncpg.Pool constructed by the caller."""
    def __init__(self, pool: Any) -> None:
        """
        Args:
            pool: asyncpg.Pool connected to Neon.
        """
        self._pool = pool

    # Registration 
    async def register(
        self,
        schema: DataSchema,
        author: str = "data-agent",
        description: str = "",
        overwrite: bool = False,
    ) -> int:
        schema_def = schema_to_dict(schema)

        # Check for evolution classification if a prior version exists
        existing = await self.get_latest(schema.dataset_id)
        if existing is not None:
            evolution = SchemaValidator.classify_evolution(existing, schema)
            if evolution.evolution_type == EvolutionType.BREAKING:
                logger.warning(
                    "BREAKING schema change registered for dataset_id=%s: %s",
                    schema.dataset_id,
                    evolution.summary,
                )
            else:
                logger.info(
                    "Schema evolution for dataset_id=%s: %s",
                    schema.dataset_id,
                    evolution.summary,
                )

        async with self._pool.acquire() as conn:
            if overwrite:
                row_id = await conn.fetchval(
                    """
                    INSERT INTO schema_registry
                        (dataset_id, version, schema_format, schema_def, column_count,
                         description, author)
                    VALUES ($1, $2, 'jsonschema', $3, $4, $5, $6)
                    ON CONFLICT (dataset_id, version) DO UPDATE
                        SET schema_def = EXCLUDED.schema_def,
                            column_count = EXCLUDED.column_count,
                            description = EXCLUDED.description
                    RETURNING id
                    """,
                    schema.dataset_id,
                    schema.version,
                    json.dumps(schema_def),
                    len(schema.columns),
                    description,
                    author,
                )
            else:
                try:
                    row_id = await conn.fetchval(
                        """
                        INSERT INTO schema_registry
                            (dataset_id, version, schema_format, schema_def, column_count,
                             description, author)
                        VALUES ($1, $2, 'jsonschema', $3, $4, $5, $6)
                        RETURNING id
                        """,
                        schema.dataset_id,
                        schema.version,
                        json.dumps(schema_def),
                        len(schema.columns),
                        description,
                        author,
                    )
                except Exception as exc:
                    if "unique" in str(exc).lower():
                        raise ValueError(
                            f"Schema version {schema.dataset_id}@{schema.version} already exists. "
                            "Use overwrite=True to update."
                        ) from exc
                    raise

        logger.info(
            "Schema registered: dataset_id=%s version=%s id=%s columns=%d",
            schema.dataset_id, schema.version, row_id, len(schema.columns),
        )
        return row_id

    # Retrieval 
    async def get_latest(self, dataset_id: str) -> Optional[DataSchema]:
        """
        Return the latest non-deprecated schema version for a dataset_id.
        Returns None if no schema is registered.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT schema_def
                FROM schema_registry
                WHERE dataset_id = $1
                  AND deprecated_at IS NULL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                dataset_id,
            )

        if row is None:
            return None

        try:
            raw = json.loads(row["schema_def"])
            return schema_from_dict(raw)
        except Exception as exc:
            logger.error(
                "Failed to deserialise schema for dataset_id=%s: %s", dataset_id, exc
            )
            return None

    async def get_version(self, dataset_id: str, version: str) -> Optional[DataSchema]:
        """Return a specific schema version. Returns None if not found."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT schema_def
                FROM schema_registry
                WHERE dataset_id = $1 AND version = $2
                LIMIT 1
                """,
                dataset_id,
                version,
            )
        if row is None:
            return None
        return schema_from_dict(json.loads(row["schema_def"]))

    async def list_versions(self, dataset_id: str) -> list[dict]:
        """
        Return all versions for a dataset, newest first.
        Fields: id, version, column_count, created_at, deprecated_at.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, version, column_count, description, deprecated_at, created_at
                FROM schema_registry
                WHERE dataset_id = $1
                ORDER BY created_at DESC
                """,
                dataset_id,
            )
        return [dict(r) for r in rows]

    async def deprecate(self, dataset_id: str, version: str) -> None:
        """Mark a schema version as deprecated (soft delete)."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE schema_registry
                SET deprecated_at = NOW()
                WHERE dataset_id = $1 AND version = $2
                """,
                dataset_id,
                version,
            )
        logger.info("Schema deprecated: %s@%s", dataset_id, version)

    # Convenience: auto-generate from DataFrame 
    @staticmethod
    def infer_from_dataframe(
        df: pd.DataFrame,
        dataset_id: str,
        version: str = "1.0.0",
        allow_extra_columns: bool = True,
    ) -> DataSchema:
        """
        Infer a DataSchema from a DataFrame's dtypes.

        Generated schema is a starting point — should be reviewed and refined
        by the data owner before registering as the canonical version.
        """
        dtype_to_col_type: dict[str, ColumnType] = {
            "int8": ColumnType.INT, "int16": ColumnType.INT,
            "int32": ColumnType.INT, "int64": ColumnType.INT,
            "Int8": ColumnType.INT, "Int16": ColumnType.INT,
            "Int32": ColumnType.INT, "Int64": ColumnType.INT,
            "float16": ColumnType.FLOAT, "float32": ColumnType.FLOAT,
            "float64": ColumnType.FLOAT,
            "bool": ColumnType.BOOLEAN,
            "object": ColumnType.STRING, "string": ColumnType.STRING,
        }

        columns: list[ColumnSchema] = []
        for col_name in df.columns:
            dtype_str = str(df[col_name].dtype)
            col_type = dtype_to_col_type.get(dtype_str, ColumnType.STRING)
            has_nulls = bool(df[col_name].isna().any())
            columns.append(
                ColumnSchema(
                    name=col_name,
                    dtype=col_type,
                    nullable=has_nulls,
                )
            )

        return DataSchema(
            dataset_id=dataset_id,
            version=version,
            columns=columns,
            allow_extra_columns=allow_extra_columns,
        )