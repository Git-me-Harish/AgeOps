"""
agents/connectors/file_connector.py
                                     
Local file connector for CSV, Parquet, JSON, and Avro uploads.

Intended for:
  1. File uploads via the UI's "drag-and-drop dataset" feature —
     the API writes the upload to /tmp, then reads it via this connector.
  2. Local development without R2 configured.
  3. CI/CD test fixtures.

URI formats:
  file:///tmp/uploads/data.csv
  file:///tmp/uploads/features.parquet
  /absolute/path/to/data.json     ← leading slash also accepted

Signed upload URL flow (for UI uploads → R2):
  The API server calls generate_r2_upload_url() to get a presigned PUT URL.
  The UI uploads the file directly to R2.
  The API then converts the R2 URI and passes it to R2Connector.
  FileConnector is only used when the file is already on the local filesystem.
"""
from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from agents.connectors import (
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

# Allowed extensions — security boundary to prevent path traversal to system files
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".csv", ".parquet", ".json", ".jsonl", ".avro"}
)

# Directories from which reads are permitted (prevent escaping temp dir)
_ALLOWED_PREFIXES: tuple[str, ...] = (
    "/tmp/",
    "/var/tmp/",
    "/home/",
    "/mnt/",
)


class FileConnector(DataConnector):
    """
    Local filesystem data connector.

    Security hardening:
      - Resolves the path with Path.resolve() to eliminate ../ traversal.
      - Validates the resolved path is under an allowed prefix.
      - Only reads files with allowed extensions.
    """

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self._resolved_path: Optional[Path] = None

    # Connection                                                             
    async def connect(self) -> None:
        """
        Validate the file path: resolve, check prefix allowlist, verify existence.

        Raises:
            ConnectorError: Invalid path, disallowed directory, or wrong extension.
            DataSourceNotFoundError: File does not exist.
        """
        raw_path = self._normalize_uri(self.config.uri)
        resolved = Path(raw_path).resolve()

        # Prevent path traversal — resolved path must be under an allowed prefix
        if not any(str(resolved).startswith(p) for p in _ALLOWED_PREFIXES):
            raise ConnectorError(
                f"File path {resolved} is outside allowed directories: {_ALLOWED_PREFIXES}. "
                "File uploads must be placed under /tmp or /mnt."
            )

        # Extension allowlist
        if resolved.suffix.lower() not in _ALLOWED_EXTENSIONS:
            raise ConnectorError(
                f"Unsupported file extension {resolved.suffix!r}. "
                f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
            )

        if not resolved.exists():
            raise DataSourceNotFoundError(f"File not found: {resolved}")

        if not resolved.is_file():
            raise ConnectorError(f"Path is not a regular file: {resolved}")

        self._resolved_path = resolved
        self._connected = True
        logger.info("FileConnector connected: %s (%.1f KB)", resolved, resolved.stat().st_size / 1024)

    # Read                                                                   
    async def read(
        self,
        schema: Optional[DataSchema] = None,
    ) -> IngestionResult:
        """
        Read the local file and return a typed IngestionResult.

        Content hash is computed on the raw file bytes before parsing —
        this is the value stored in data_lineage for tamper evidence.
        """
        if not self._connected or self._resolved_path is None:
            await self.connect()

        assert self._resolved_path is not None
        start_ms = int(time.monotonic() * 1000)

        raw_bytes = self._resolved_path.read_bytes()
        content_hash = self.hash_bytes(raw_bytes)
        fmt = self._resolved_path.suffix.lower().lstrip(".")
        if fmt == "jsonl":
            fmt = "jsonlines"

        df = self._parse(raw_bytes, fmt)

        violations: list[str] = []
        if schema is not None:
            violations = self._enforce_schema(df, schema)
            if violations and settings.schema_registry_block_on_mismatch:
                raise SchemaViolationError(violations)

        duration_ms = int(time.monotonic() * 1000) - start_ms
        logger.info(
            "FileConnector.read — path=%s rows=%d cols=%d hash=%s duration_ms=%d",
            self._resolved_path, len(df), len(df.columns), content_hash[:12], duration_ms,
        )

        return IngestionResult(
            dataframe=df,
            row_count=len(df),
            col_count=len(df.columns),
            content_hash=content_hash,
            byte_size=len(raw_bytes),
            format=fmt,
            schema_violations=violations,
        )

    # Sample                                                                 
    async def sample(self, n: int = 100) -> pd.DataFrame:
        """Return first `n` rows. Parquet uses row-group-aware reading."""
        if not self._connected or self._resolved_path is None:
            await self.connect()

        assert self._resolved_path is not None
        ext = self._resolved_path.suffix.lower()

        if ext == ".parquet":
            return self._sample_parquet(n)
        if ext == ".csv":
            return pd.read_csv(self._resolved_path, nrows=n)
        if ext in (".json", ".jsonl"):
            return pd.read_json(self._resolved_path, lines=(ext == ".jsonl")).head(n)

        # Fallback for Avro — read all then head
        result = await self.read()
        return result.dataframe.head(n)

    def _sample_parquet(self, n: int) -> pd.DataFrame:
        """Read first row group from Parquet via pyarrow."""
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(self._resolved_path)
        # Iterate first batch only
        batch = next(pf.iter_batches(batch_size=max(n, 1000)))
        return batch.to_pandas().head(n)

    # Parsing                                                                
    @staticmethod
    def _parse(raw: bytes, fmt: str) -> pd.DataFrame:
        """Parse raw bytes into a DataFrame based on format string."""
        buf = io.BytesIO(raw)

        if fmt == "csv":
            return pd.read_csv(buf)
        if fmt == "parquet":
            return pd.read_parquet(buf)
        if fmt == "json":
            return pd.read_json(buf)
        if fmt == "jsonlines":
            return pd.read_json(buf, lines=True)
        if fmt == "avro":
            try:
                import fastavro
                records = list(fastavro.reader(buf))
                return pd.DataFrame(records)
            except ImportError:
                raise ConnectorError(
                    "fastavro is required for Avro parsing. pip install fastavro."
                )
        raise ConnectorError(f"Unsupported format: {fmt!r}")

    # URI normalisation                                                      
    @staticmethod
    def _normalize_uri(uri: str) -> str:
        """Strip file:// prefix, return an absolute path string."""
        if uri.startswith("file://"):
            return uri[len("file://"):]
        return uri   # already a bare absolute path

# Presigned upload URL helper (used by API server for UI uploads)
def generate_r2_upload_url(
    filename: str,
    content_type: str = "application/octet-stream",
    expires_in: int = 900,   # 15 minutes
) -> dict[str, str]:
    """
    Generate a presigned PUT URL for direct UI → R2 uploads.

    Returns:
        {"upload_url": str, "r2_uri": str, "key": str}

    The UI POSTs the file directly to `upload_url`.
    After upload, the API converts `r2_uri` → R2Connector.
    This keeps the API server out of the data path for large files.
    """
    import boto3
    from botocore.config import Config

    if not settings.r2_configured:
        raise ConnectorError(
            "R2 credentials not configured — cannot generate presigned upload URL."
        )

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name=settings.r2_region,
        config=Config(signature_version="s3v4"),
    )

    # Place uploads under a dedicated prefix to avoid collision with model artifacts
    key = f"{settings.r2_datasets_prefix}/uploads/{filename}"
    upload_url = s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.r2_bucket_name,
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=expires_in,
    )
    r2_uri = f"s3://{settings.r2_bucket_name}/{key}"

    logger.info("Generated presigned upload URL for %s → R2 key: %s", filename, key)
    return {
        "upload_url": upload_url,
        "r2_uri": r2_uri,
        "key": key,
    }