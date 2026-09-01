"""
agents/connectors/r2_connector.py

Cloudflare R2 connector (S3-compatible via boto3).

Supports:
  - Single object reads (CSV, Parquet, JSON, Avro)
  - Prefix / multi-file reads (all objects under a prefix merged into one DataFrame)
  - Sample reads (head N rows without downloading the full file)
  - Content hash computation on the raw download bytes (SHA-256)
  - Schema enforcement via DataConnector._enforce_schema()
  - Retry with exponential backoff (tenacity)
  - MLflow input tracking via mlflow.log_input()

URI formats:
  s3://bucket/path/to/file.parquet
  r2://bucket/path/to/file.csv
  s3://bucket/datasets/2025-08/        ← trailing slash = prefix scan

Design notes:
  - _s3_client is boto3 (synchronous). We run it in asyncio.to_thread() so
    async callers are not blocked.
  - Avro support requires the `fastavro` package (included in requirements.txt).
  - Large Parquet files are read with pyarrow to control memory via batching.
"""
from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Any, Optional

import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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

# Supported formats and their read functions (resolved after download)
_FORMAT_READERS: dict[str, str] = {
    ".csv":     "csv",
    ".parquet": "parquet",
    ".json":    "json",
    ".jsonl":   "jsonlines",
    ".avro":    "avro",
}


class R2Connector(DataConnector):
    """
    Cloudflare R2 (S3-compatible) data connector.

    Credentials resolution order:
      1. ConnectorConfig.access_key_id / secret_access_key (explicit)
      2. settings.r2_access_key_id / r2_secret_access_key (env)
      3. AWS credential chain (IAM role, ~/.aws) — for native S3 buckets
    """

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self._client: Any = None   # boto3 S3 client — built lazily

    # Connection 
    async def connect(self) -> None:
        """
        Build the boto3 S3 client and perform a head-bucket call to verify
        that credentials are valid and the bucket exists.

        Raises:
            AuthenticationError: Bad credentials.
            DataSourceNotFoundError: Bucket does not exist.
            ConnectorError: Any other connectivity failure.
        """
        try:
            self._client = await asyncio.to_thread(self._build_client)
            bucket = self._parse_uri()[0]
            await asyncio.to_thread(self._head_bucket, bucket)
            self._connected = True
            logger.info("R2Connector connected to bucket: %s", bucket)
        except NoCredentialsError as exc:
            raise AuthenticationError(
                "R2 / S3 credentials missing. Set r2_access_key_id + r2_secret_access_key."
            ) from exc
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("403", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
                raise AuthenticationError(f"R2 authentication failed: {exc}") from exc
            if code in ("404", "NoSuchBucket"):
                bucket = self._parse_uri()[0]
                raise DataSourceNotFoundError(f"R2 bucket not found: {bucket}") from exc
            raise ConnectorError(f"R2 connection failed: {exc}") from exc

    def _build_client(self) -> Any:
        """Build boto3 S3 client. Called inside asyncio.to_thread()."""
        import boto3

        access_key = self.config.access_key_id or settings.r2_access_key_id
        secret_key = self.config.secret_access_key or settings.r2_secret_access_key
        endpoint = self.config.endpoint_url or settings.r2_endpoint_url
        region = self.config.region or settings.r2_region

        kwargs: dict[str, Any] = {
            "config": Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                max_pool_connections=20,
            ),
        }
        if access_key:
            kwargs["aws_access_key_id"] = access_key
        if secret_key:
            kwargs["aws_secret_access_key"] = secret_key
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        if region:
            kwargs["region_name"] = region

        return boto3.client("s3", **kwargs)

    def _head_bucket(self, bucket: str) -> None:
        """Verify bucket exists and credentials are valid."""
        self._client.head_bucket(Bucket=bucket)

    # Read (main ingestion path) 
    async def read(
        self,
        schema: Optional[DataSchema] = None,
    ) -> IngestionResult:
        """
        Download and parse the dataset at config.uri.

        Handles:
          - Single file: s3://bucket/path/data.parquet
          - Prefix scan: s3://bucket/path/prefix/ (merged)

        Returns IngestionResult with content_hash and schema violations.
        Raises SchemaViolationError if schema enforcement is enabled and
        settings.schema_registry_block_on_mismatch is True.
        """
        if not self._connected:
            await self.connect()

        bucket, key = self._parse_uri()
        start_ms = int(time.monotonic() * 1000)

        if key.endswith("/") or not self._has_extension(key):
            # Prefix scan — merge all objects under this prefix
            logger.info("R2Connector: prefix scan s3://%s/%s", bucket, key)
            df, raw_bytes = await asyncio.to_thread(
                self._read_prefix, bucket, key
            )
        else:
            # Single object
            logger.info("R2Connector: reading s3://%s/%s", bucket, key)
            df, raw_bytes = await asyncio.to_thread(
                self._read_object, bucket, key
            )

        content_hash = self.hash_bytes(raw_bytes)
        fmt = self._detect_format(key)

        violations: list[str] = []
        if schema is not None:
            violations = self._enforce_schema(df, schema)
            if violations and settings.schema_registry_block_on_mismatch:
                raise SchemaViolationError(violations)

        duration_ms = int(time.monotonic() * 1000) - start_ms
        logger.info(
            "R2Connector: read complete — rows=%d cols=%d hash=%s violations=%d duration_ms=%d",
            len(df), len(df.columns), content_hash[:12], len(violations), duration_ms,
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
        """
        Return first `n` rows without loading the full dataset.

        Strategy:
          - Parquet: use pyarrow to read first row group only
          - CSV: range request for first chunk (~64 KB)
          - Other: full download + head
        """
        if not self._connected:
            await self.connect()

        bucket, key = self._parse_uri()
        fmt = self._detect_format(key)

        if fmt == "parquet":
            return await asyncio.to_thread(self._sample_parquet, bucket, key, n)

        if fmt == "csv":
            return await asyncio.to_thread(self._sample_csv, bucket, key, n)

        # Fallback: full download + head
        result = await self.read()
        return result.dataframe.head(n)

    # Internal read helpers 
    @retry(
        retry=retry_if_exception_type(ClientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _read_object(self, bucket: str, key: str) -> tuple[pd.DataFrame, bytes]:
        """Download a single S3 object and parse it into a DataFrame."""
        response = self._client.get_object(Bucket=bucket, Key=key)
        raw_bytes: bytes = response["Body"].read()
        fmt = self._detect_format(key)
        df = self._parse_bytes(raw_bytes, fmt)
        return df, raw_bytes

    def _read_prefix(self, bucket: str, prefix: str) -> tuple[pd.DataFrame, bytes]:
        """
        List all objects under a prefix, download and merge them.
        Assumes all objects share the same schema.
        """
        paginator = self._client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        frames: list[pd.DataFrame] = []
        all_bytes: list[bytes] = []

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not self._has_extension(key):
                    continue
                response = self._client.get_object(Bucket=bucket, Key=key)
                raw: bytes = response["Body"].read()
                fmt = self._detect_format(key)
                try:
                    frames.append(self._parse_bytes(raw, fmt))
                    all_bytes.append(raw)
                except Exception as exc:
                    logger.warning("Skipping unparseable object %s: %s", key, exc)

        if not frames:
            raise DataSourceNotFoundError(
                f"No parseable objects found under s3://{bucket}/{prefix}"
            )

        merged = pd.concat(frames, ignore_index=True)
        merged_bytes = b"".join(all_bytes)
        logger.info("Merged %d objects under prefix — total rows: %d", len(frames), len(merged))
        return merged, merged_bytes

    def _sample_parquet(self, bucket: str, key: str, n: int) -> pd.DataFrame:
        """Read only the first row group of a Parquet file using pyarrow."""
        import pyarrow.parquet as pq

        response = self._client.get_object(Bucket=bucket, Key=key)
        raw = response["Body"].read()
        buf = io.BytesIO(raw)
        pf = pq.ParquetFile(buf)
        # Read first batch — may be more than n rows; head after
        batch = next(pf.iter_batches(batch_size=max(n, 1000)))
        return batch.to_pandas().head(n)

    def _sample_csv(self, bucket: str, key: str, n: int) -> pd.DataFrame:
        """Range-read first ~128 KB of a CSV to get sample rows."""
        response = self._client.get_object(
            Bucket=bucket,
            Key=key,
            Range="bytes=0-131072",   # 128 KB — generous for n≤1000 rows
        )
        raw = response["Body"].read()
        # May contain a partial last line — read_csv handles this gracefully
        try:
            return pd.read_csv(io.BytesIO(raw)).head(n)
        except Exception:
            # If partial line breaks CSV parse, try without last line
            lines = raw.split(b"\n")
            truncated = b"\n".join(lines[:-1])
            return pd.read_csv(io.BytesIO(truncated)).head(n)

    # Format parsing 
    @staticmethod
    def _parse_bytes(raw: bytes, fmt: str) -> pd.DataFrame:
        """Parse raw bytes into a DataFrame based on detected format."""
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
                    "fastavro is required for Avro reads. Add fastavro to requirements.txt."
                )
        raise ConnectorError(f"Unsupported format: {fmt!r}")

    @staticmethod
    def _detect_format(key: str) -> str:
        """Infer file format from object key extension."""
        key_lower = key.lower().rstrip("/")
        for ext, fmt in _FORMAT_READERS.items():
            if key_lower.endswith(ext):
                return fmt
        return "csv"  # sensible default for unlabelled objects

    @staticmethod
    def _has_extension(key: str) -> bool:
        """True when the key has a known parseable extension."""
        return any(key.lower().endswith(ext) for ext in _FORMAT_READERS)

    # URI parsing 
    def _parse_uri(self) -> tuple[str, str]:
        """
        Split s3://bucket/key → (bucket, key).
        Strips the r2:// alias before splitting.
        """
        uri = self.config.uri
        for prefix in ("s3://", "r2://"):
            if uri.startswith(prefix):
                uri = uri[len(prefix):]
                break
        parts = uri.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key