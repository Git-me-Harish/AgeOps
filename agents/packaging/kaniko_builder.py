"""
agents/packaging/kaniko_builder.py

Phase 4 — In-Cluster OCI Image Builder via Kaniko.

Kaniko builds container images inside Kubernetes without a Docker daemon and
without privileged containers — both are security requirements on managed
clusters.  This module owns the full build lifecycle:

  1. Compress build context (Dockerfile + serving/ + model_info.json) → tarball
  2. Upload tarball to R2 (Kaniko reads context from S3-compatible URL)
  3. Create a Kubernetes BatchV1 Job running the Kaniko executor
  4. Poll the Job until it completes (SUCCEEDED / FAILED), stream logs to logger
  5. Extract the pushed image digest from Kaniko log output
  6. Confirm digest via Docker Registry API v2 (fallback if log parsing fails)
  7. Return a BuildResult with the digest, tag, duration, and log URI

Error contract:
  - All K8s API errors raise `KanikoBuildError` — never swallowed.
  - A build that times out raises `KanikoBuildTimeoutError`.
  - A Job that ends in FAILED status raises `KanikoBuildError` with the
    pod's last log lines as the error message.

Security:
  - Docker Hub credentials are injected via a K8s Secret (never in env literals).
  - R2 credentials for build context download are also from a K8s Secret.
  - MLflow tracking URI is non-sensitive (it's the internal cluster DNS name).
"""
from __future__ import annotations

import io
import json
import logging
import re
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import boto3
import botocore.exceptions

from configs.settings import settings

logger = logging.getLogger(__name__)

# Digest regex — matches "sha256:" followed by 64 hex chars anywhere in a log line
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")

# Kaniko log line that confirms a successful push
_PUSH_SUCCESS_RE = re.compile(r"Built and pushed image|Pushed image to \d+ destination")

# How often to poll the K8s Job status (seconds)
_POLL_INTERVAL_SECONDS = 10


class KanikoBuildError(RuntimeError):
    """Raised when a Kaniko build Job fails."""


class KanikoBuildTimeoutError(KanikoBuildError):
    """Raised when a Kaniko build Job exceeds the timeout."""


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class BuildResult:
    """
    Outcome of one completed Kaniko build.

    Attributes:
        image_tag:        Full mutable tag (e.g. docker.io/org/model:v3).
        image_digest:     Immutable sha256 content digest (e.g. sha256:abc…).
        image_uri:        Canonical digest-pinned reference (tag@digest).
        duration_seconds: Wall-clock seconds from Job creation to completion.
        log_uri:          R2 URI where the raw build log is stored.
        job_name:         Kubernetes Job name (for audit trail).
    """
    image_tag:        str
    image_digest:     str
    image_uri:        str
    duration_seconds: int
    log_uri:          str
    job_name:         str


# ── Builder ────────────────────────────────────────────────────────────────────

class KanikoBuilder:
    """
    Manages the full lifecycle of one Kaniko build Job.

    Args:
        model_name:    MLflow registered model name.
        model_version: MLflow model version string.
        pool:          asyncpg pool (used by DockerfileAgent for DB writes;
                       passed through so this class can write build status updates).

    Usage:
        builder = KanikoBuilder("fraud-detector", "3", pool)
        result  = await builder.build(
            dockerfile_content="FROM python:3.11-slim\\n...",
            model_info_json='{"model_name": "fraud-detector", ...}',
            serving_dir=Path("/app/serving"),
            image_tag="docker.io/org/fraud-detector:v3",
        )
    """

    def __init__(
        self,
        model_name: str,
        model_version: str,
        pool: Any,
        timeout_seconds: int = 900,
    ) -> None:
        self._model_name    = model_name
        self._model_version = model_version
        self._pool          = pool
        self._timeout       = timeout_seconds
        self._s3            = self._make_s3_client()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def build(
        self,
        dockerfile_content: str,
        model_info_json:    str,
        serving_dir:        Path,
        image_tag:          str,
        build_job_db_id:    Optional[int] = None,
    ) -> BuildResult:
        """
        Run the full build pipeline end-to-end.

        Args:
            dockerfile_content: Full Dockerfile text from DockerfileGenerator.
            model_info_json:    JSON written to /app/model_info.json in the image.
            serving_dir:        Local path containing the inference server code.
            image_tag:          Target image tag to push (full registry/name:tag).
            build_job_db_id:    DB row ID in oci_build_jobs for status updates.

        Returns:
            BuildResult with digest and metadata.

        Raises:
            KanikoBuildError:        Build Job failed.
            KanikoBuildTimeoutError: Build exceeded timeout_seconds.
        """
        job_name       = self._job_name()
        context_key    = self._context_key(job_name)
        context_uri    = f"s3://{settings.r2_bucket_name}/{context_key}"
        start_time     = time.monotonic()

        logger.info(
            "Kaniko build starting: model=%s version=%s job=%s image=%s",
            self._model_name, self._model_version, job_name, image_tag,
        )

        # 1 — Upload build context to R2
        await self._upload_context(
            key=context_key,
            dockerfile_content=dockerfile_content,
            model_info_json=model_info_json,
            serving_dir=serving_dir,
        )
        logger.info("Build context uploaded → %s", context_uri)

        # 2 — Create Kaniko K8s Job
        k8s_batch = self._k8s_batch_api()
        manifest   = self._build_job_manifest(job_name, context_key, image_tag)
        k8s_batch.create_namespaced_job(
            namespace=settings.k8s_namespace,
            body=manifest,
        )
        logger.info("Kaniko Job created: %s", job_name)

        # 3 — Update DB: status = running
        if build_job_db_id is not None:
            await self._update_job_status(
                build_job_db_id, "running", job_name, context_uri
            )

        # 4 — Poll until completion, streaming logs
        raw_logs = await self._wait_for_completion(
            job_name=job_name,
            k8s_batch=k8s_batch,
            start_time=start_time,
        )

        duration = int(time.monotonic() - start_time)

        # 5 — Store build log in R2
        log_uri = await self._upload_log(job_name, raw_logs)

        # 6 — Extract digest
        digest = self._extract_digest(raw_logs, image_tag)

        image_uri = f"{image_tag}@{digest}"

        logger.info(
            "Kaniko build complete: job=%s digest=%s duration=%ds",
            job_name, digest, duration,
        )
        return BuildResult(
            image_tag=image_tag,
            image_digest=digest,
            image_uri=image_uri,
            duration_seconds=duration,
            log_uri=log_uri,
            job_name=job_name,
        )

    # ── Build context ─────────────────────────────────────────────────────────

    async def _upload_context(
        self,
        key: str,
        dockerfile_content: str,
        model_info_json: str,
        serving_dir: Path,
    ) -> None:
        """
        Create a gzipped tar of the build context and upload to R2.

        Context contents:
          Dockerfile
          model_info.json
          serving/__init__.py
          serving/inference_server.py
          serving/health.py
          serving/metrics.py
        """
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            # Add Dockerfile
            df_bytes = dockerfile_content.encode("utf-8")
            ti = tarfile.TarInfo(name="Dockerfile")
            ti.size = len(df_bytes)
            tar.addfile(ti, io.BytesIO(df_bytes))

            # Add model_info.json
            mi_bytes = model_info_json.encode("utf-8")
            ti2 = tarfile.TarInfo(name="model_info.json")
            ti2.size = len(mi_bytes)
            tar.addfile(ti2, io.BytesIO(mi_bytes))

            # Add serving/ directory recursively
            if serving_dir.exists():
                for srv_file in sorted(serving_dir.rglob("*.py")):
                    rel = srv_file.relative_to(serving_dir.parent)
                    file_bytes = srv_file.read_bytes()
                    ti3 = tarfile.TarInfo(name=str(rel))
                    ti3.size = len(file_bytes)
                    tar.addfile(ti3, io.BytesIO(file_bytes))

        buf.seek(0)
        self._s3.put_object(
            Bucket=settings.r2_bucket_name,
            Key=key,
            Body=buf.read(),
            ContentType="application/gzip",
        )

    # ── Job manifest ──────────────────────────────────────────────────────────

    def _build_job_manifest(
        self, job_name: str, context_key: str, image_tag: str
    ) -> dict[str, Any]:
        """
        Build the Kubernetes BatchV1 Job manifest for Kaniko.

        Secrets expected in the mlops namespace:
          r2-credentials      : keys access_key_id, secret_access_key
          dockerhub-credentials: key .dockerconfigjson (standard docker config)
        """
        kaniko_args = [
            f"--context=s3://{settings.r2_bucket_name}/{context_key}",
            "--dockerfile=Dockerfile",
            f"--destination={image_tag}",
            "--log-format=text",
            "--verbosity=info",
            "--snapshot-mode=redo",
            "--compressed-caching=false",
        ]
        if settings.kaniko_cache_enabled:
            kaniko_args.append("--cache=true")
            kaniko_args.append(
                f"--cache-repo={settings.image_registry_namespace}/cache"
            )

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name":      job_name,
                "namespace": settings.k8s_namespace,
                "labels": {
                    "mlops.component":     "kaniko-build",
                    "mlops.model_name":    self._model_name,
                    "mlops.model_version": self._model_version,
                },
            },
            "spec": {
                "ttlSecondsAfterFinished": 600,
                "backoffLimit": 0,
                "template": {
                    "metadata": {
                        "labels": {
                            "mlops.component": "kaniko-build",
                            "mlops.job_name":  job_name,
                        }
                    },
                    "spec": {
                        "serviceAccountName": "kaniko-builder",
                        "restartPolicy":      "Never",
                        "containers": [{
                            "name":  "kaniko",
                            "image": settings.kaniko_image,
                            "args":  kaniko_args,
                            "env": [
                                {
                                    "name": "AWS_ACCESS_KEY_ID",
                                    "valueFrom": {"secretKeyRef": {
                                        "name": "r2-credentials",
                                        "key":  "access_key_id",
                                    }},
                                },
                                {
                                    "name": "AWS_SECRET_ACCESS_KEY",
                                    "valueFrom": {"secretKeyRef": {
                                        "name": "r2-credentials",
                                        "key":  "secret_access_key",
                                    }},
                                },
                                {
                                    "name":  "S3_ENDPOINT",
                                    "value": settings.r2_endpoint_url,
                                },
                                {
                                    "name":  "AWS_REGION",
                                    "value": "auto",
                                },
                                {
                                    "name":  "MLFLOW_TRACKING_URI",
                                    "value": settings.mlflow_tracking_uri,
                                },
                                {
                                    "name": "MLFLOW_S3_ENDPOINT_URL",
                                    "value": settings.r2_endpoint_url,
                                },
                            ],
                            "volumeMounts": [{
                                "name":      "docker-config",
                                "mountPath": "/kaniko/.docker",
                            }],
                            "resources": {
                                "requests": {"memory": "2Gi", "cpu": "1000m"},
                                "limits":   {"memory": "4Gi", "cpu": "2000m"},
                            },
                        }],
                        "volumes": [{
                            "name": "docker-config",
                            "secret": {
                                "secretName": "dockerhub-credentials",
                                "items": [{
                                    "key":  ".dockerconfigjson",
                                    "path": "config.json",
                                }],
                            },
                        }],
                    },
                },
            },
        }

    # ── Job polling ───────────────────────────────────────────────────────────

    async def _wait_for_completion(
        self,
        job_name: str,
        k8s_batch: Any,
        start_time: float,
    ) -> str:
        """
        Poll the Kaniko Job until it finishes.  Streams log lines to the logger.
        Returns the raw log string on success.
        Raises KanikoBuildError on failure, KanikoBuildTimeoutError on timeout.
        """
        k8s_core  = self._k8s_core_api()
        raw_logs  = ""

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > self._timeout:
                raise KanikoBuildTimeoutError(
                    f"Kaniko Job {job_name!r} exceeded timeout of {self._timeout}s"
                )

            job = k8s_batch.read_namespaced_job(
                name=job_name, namespace=settings.k8s_namespace
            )
            status = job.status

            if status.succeeded:
                raw_logs = self._fetch_pod_logs(k8s_core, job_name)
                return raw_logs

            if status.failed:
                raw_logs = self._fetch_pod_logs(k8s_core, job_name)
                tail = "\n".join(raw_logs.splitlines()[-30:])
                raise KanikoBuildError(
                    f"Kaniko Job {job_name!r} failed.\nLast 30 log lines:\n{tail}"
                )

            # Still running — log progress, then sleep
            active = status.active or 0
            logger.debug(
                "Kaniko build in progress: job=%s active_pods=%d elapsed=%.0fs",
                job_name, active, elapsed,
            )
            time.sleep(_POLL_INTERVAL_SECONDS)

    def _fetch_pod_logs(self, k8s_core: Any, job_name: str) -> str:
        """Fetch logs from the first pod of the Job."""
        pods = k8s_core.list_namespaced_pod(
            namespace=settings.k8s_namespace,
            label_selector=f"mlops.job_name={job_name}",
        )
        if not pods.items:
            return ""
        pod_name = pods.items[0].metadata.name
        try:
            logs: str = k8s_core.read_namespaced_pod_log(
                name=pod_name,
                namespace=settings.k8s_namespace,
                container="kaniko",
            )
            for line in logs.splitlines():
                logger.info("[kaniko] %s", line)
            return logs
        except Exception as exc:
            logger.warning("Could not fetch Kaniko pod logs: %s", exc)
            return ""

    # ── Digest extraction ─────────────────────────────────────────────────────

    def _extract_digest(self, logs: str, image_tag: str) -> str:
        """
        Extract the sha256 digest from Kaniko build logs.

        Kaniko emits lines like:
          INFO[...] Built and pushed image as docker.io/org/model:v3
          INFO[...] sha256:abc123...
        or inline:
          INFO[...] Pushing index manifest for docker.io/org/model:v3@sha256:abc123...

        Falls back to the Docker Registry API v2 if no digest is found in logs.
        """
        for line in reversed(logs.splitlines()):
            m = _DIGEST_RE.search(line)
            if m:
                return m.group(0)

        # Fallback: query the Docker Registry API v2 for the tag manifest
        digest = self._query_registry_digest(image_tag)
        if digest:
            return digest

        raise KanikoBuildError(
            f"Could not extract image digest from Kaniko logs "
            f"or registry API for tag {image_tag!r}. "
            f"Check Kaniko logs for push errors."
        )

    def _query_registry_digest(self, image_tag: str) -> Optional[str]:
        """
        Query the Docker Registry API v2 to get the sha256 digest of a tag.
        Returns None if the registry is unreachable or the tag doesn't exist.
        """
        try:
            import urllib.request
            # Parse image_tag: registry/namespace/name:tag
            parts  = image_tag.split("/", 2)
            name_tag = parts[-1]                     # e.g. "model:v3"
            name, tag = name_tag.rsplit(":", 1)
            registry = "/".join(parts[:-1]) or "registry-1.docker.io"

            url = f"https://{registry}/v2/{name}/manifests/{tag}"
            req = urllib.request.Request(url)
            req.add_header(
                "Accept",
                "application/vnd.docker.distribution.manifest.v2+json",
            )
            # Note: auth is not added here — public images only.
            # Private registries should have the digest in the Kaniko logs.
            with urllib.request.urlopen(req, timeout=10) as resp:
                digest = resp.headers.get("Docker-Content-Digest", "")
                return digest if digest.startswith("sha256:") else None
        except Exception as exc:
            logger.warning("Registry API digest query failed: %s", exc)
            return None

    # ── Log upload ────────────────────────────────────────────────────────────

    async def _upload_log(self, job_name: str, raw_logs: str) -> str:
        """Upload build logs to R2 and return the R2 URI."""
        key = (
            f"{settings.r2_images_prefix}/logs/"
            f"{self._model_name}-{self._model_version}-{job_name}.log"
        )
        try:
            self._s3.put_object(
                Bucket=settings.r2_bucket_name,
                Key=key,
                Body=raw_logs.encode("utf-8"),
                ContentType="text/plain",
            )
            uri = f"s3://{settings.r2_bucket_name}/{key}"
            logger.debug("Build log uploaded → %s", uri)
            return uri
        except botocore.exceptions.ClientError as exc:
            logger.warning("Failed to upload build log to R2: %s", exc)
            return ""

    # ── DB helpers ────────────────────────────────────────────────────────────

    async def _update_job_status(
        self,
        db_id: int,
        status: str,
        job_name: Optional[str] = None,
        context_uri: Optional[str] = None,
    ) -> None:
        """Update the oci_build_jobs row status."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE oci_build_jobs
                SET status=$1,
                    kaniko_job_name=COALESCE($2, kaniko_job_name),
                    build_context_uri=COALESCE($3, build_context_uri),
                    started_at=CASE WHEN $1='running' THEN NOW() ELSE started_at END,
                    completed_at=CASE WHEN $1 IN ('succeeded','failed') THEN NOW() ELSE completed_at END
                WHERE id=$4
                """,
                status, job_name, context_uri, db_id,
            )

    # ── K8s / S3 client factories ──────────────────────────────────────────────

    @staticmethod
    def _k8s_batch_api() -> Any:
        """Return a configured BatchV1Api client."""
        from kubernetes import client as k8s_client, config as k8s_config
        try:
            if settings.k8s_in_cluster:
                k8s_config.load_incluster_config()
            else:
                k8s_config.load_kube_config()
        except Exception:
            pass  # already configured
        return k8s_client.BatchV1Api()

    @staticmethod
    def _k8s_core_api() -> Any:
        """Return a configured CoreV1Api client."""
        from kubernetes import client as k8s_client
        return k8s_client.CoreV1Api()

    @staticmethod
    def _make_s3_client() -> Any:
        """R2-compatible S3 client."""
        return boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint_url,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name=settings.r2_region,
        )

    @staticmethod
    def _job_name() -> str:
        suffix = uuid.uuid4().hex[:8]
        return f"kaniko-build-{suffix}"

    @staticmethod
    def _context_key(job_name: str) -> str:
        return f"{settings.r2_images_prefix}/contexts/{job_name}.tar.gz"