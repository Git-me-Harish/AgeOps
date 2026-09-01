"""
agents/packaging/trivy_scanner.py

Phase 4 — Trivy Container Image CVE Scanner.

Runs Aqua Security's Trivy against a pushed OCI image to detect CVEs before
the image is allowed to reach the Staging → Production promotion gate.

Execution model:
  In-cluster (k8s_in_cluster=True):
    Creates a Kubernetes BatchV1 Job running aquasec/trivy.
    Trivy scans the remote image (pulls from registry), outputs JSON to stdout.
    Job logs are fetched and parsed.

  Local dev (k8s_in_cluster=False):
    Runs `trivy image --format json ...` as a subprocess.
    Requires `trivy` binary on PATH.  Not required in production.

Gate logic (mirrors settings.trivy_critical_cve_limit):
  - CRITICAL CVEs > limit (default 0) → scan fails → image blocked
  - HIGH CVEs > settings.trivy_high_cve_limit (default 5) → scan fails
  - MEDIUM / LOW → informational only, scan passes

Trivy JSON schema (SchemaVersion 2, subset parsed here):
  {
    "SchemaVersion": 2,
    "Results": [
      {
        "Target": "<layer or package>",
        "Type": "debian",
        "Vulnerabilities": [
          {
            "VulnerabilityID": "CVE-2023-...",
            "Severity": "CRITICAL",
            "PkgName": "openssl",
            "InstalledVersion": "1.1.1",
            "FixedVersion": "1.1.2",
            "Title": "...",
            "Description": "..."
          }
        ]
      }
    ]
  }

The full raw report JSON is stored in trivy_scan_results.raw_report (JSONB)
for per-CVE SQL queries without re-scanning.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel

from configs.settings import settings

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 8
_TRIVY_IMAGE = "aquasec/trivy:latest"


# ── CVE finding model ─────────────────────────────────────────────────────────

class CVEFinding(BaseModel):
    """Single CVE found in a Trivy scan."""
    vulnerability_id:  str
    severity:          str           # CRITICAL | HIGH | MEDIUM | LOW | UNKNOWN
    package:           str
    installed_version: str
    fixed_version:     Optional[str] = None
    title:             str = ""
    description:       str = ""


class TrivyScanResult(BaseModel):
    """Aggregated result from a Trivy container image scan."""
    image_uri:       str
    image_digest:    str
    critical_count:  int
    high_count:      int
    medium_count:    int
    low_count:       int
    total_count:     int
    passed:          bool          # True iff critical/high counts are within limits
    rejection_reason: Optional[str] = None
    findings:        list[CVEFinding] = field(default_factory=list)
    raw_report:      dict[str, Any] = field(default_factory=dict)
    scan_duration_ms: int = 0
    trivy_version:   Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


# ── Scanner ────────────────────────────────────────────────────────────────────

class TrivyScanner:
    """
    Scans an OCI image for CVEs using Trivy.

    Args:
        model_name:    For logging and DB record annotation.
        model_version: For logging and DB record annotation.
        pool:          asyncpg pool — used to persist scan results.
        timeout_seconds: Maximum time to wait for the Trivy Job (default 600s).
    """

    def __init__(
        self,
        model_name:    str,
        model_version: str,
        pool:          Any,
        timeout_seconds: int = 600,
    ) -> None:
        self._model_name    = model_name
        self._model_version = model_version
        self._pool          = pool
        self._timeout       = timeout_seconds

    async def scan(
        self,
        image_uri:    str,
        image_digest: str,
        workflow_id:  Optional[str] = None,
    ) -> TrivyScanResult:
        """
        Run a CVE scan against the given image.

        Args:
            image_uri:    Full digest-pinned image reference (tag@sha256:...).
            image_digest: Standalone sha256 digest string.
            workflow_id:  Optional FK to workflows table.

        Returns:
            TrivyScanResult with pass/fail status and structured findings.

        Raises:
            RuntimeError: if the scan Job cannot be created or the binary is missing.
        """
        start_ms = int(time.monotonic() * 1000)

        if settings.k8s_in_cluster:
            raw_json = await self._scan_via_k8s_job(image_uri)
        else:
            raw_json = self._scan_via_subprocess(image_uri)

        result = self._parse_trivy_json(raw_json, image_uri, image_digest)
        result.scan_duration_ms = int(time.monotonic() * 1000) - start_ms

        # Persist to trivy_scan_results
        await self._persist(result, workflow_id)

        logger.info(
            "Trivy scan complete: image=%s critical=%d high=%d passed=%s",
            image_uri,
            result.critical_count,
            result.high_count,
            result.passed,
        )
        return result

    # ── K8s Job path ──────────────────────────────────────────────────────────

    async def _scan_via_k8s_job(self, image_uri: str) -> dict[str, Any]:
        """Create a Trivy Job, wait for it to finish, return parsed JSON."""
        job_name = f"trivy-scan-{uuid.uuid4().hex[:8]}"
        manifest = self._build_trivy_job_manifest(job_name, image_uri)

        k8s_batch = self._k8s_batch_api()
        k8s_batch.create_namespaced_job(
            namespace=settings.k8s_namespace,
            body=manifest,
        )
        logger.info("Trivy Job created: %s for image %s", job_name, image_uri)

        raw_json = await self._wait_for_trivy_job(job_name, k8s_batch)
        return raw_json

    def _build_trivy_job_manifest(
        self, job_name: str, image_uri: str
    ) -> dict[str, Any]:
        """Build the K8s Job manifest that runs trivy image --format json."""
        return {
            "apiVersion": "batch/v1",
            "kind":       "Job",
            "metadata": {
                "name":      job_name,
                "namespace": settings.k8s_namespace,
                "labels": {
                    "mlops.component":     "trivy-scan",
                    "mlops.model_name":    self._model_name,
                    "mlops.model_version": self._model_version,
                },
            },
            "spec": {
                "ttlSecondsAfterFinished": 300,
                "backoffLimit": 1,
                "template": {
                    "spec": {
                        "restartPolicy": "OnFailure",
                        "containers": [{
                            "name":  "trivy",
                            "image": _TRIVY_IMAGE,
                            "command": ["trivy"],
                            "args": [
                                "image",
                                "--format",   "json",
                                "--exit-code", "0",    # don't fail on CVEs — we gate
                                "--no-progress",
                                "--timeout",  "10m",
                                image_uri,
                            ],
                            "env": [
                                {
                                    "name": "TRIVY_USERNAME",
                                    "valueFrom": {"secretKeyRef": {
                                        "name": "dockerhub-credentials",
                                        "key":  "username",
                                    }},
                                },
                                {
                                    "name": "TRIVY_PASSWORD",
                                    "valueFrom": {"secretKeyRef": {
                                        "name": "dockerhub-credentials",
                                        "key":  "password",
                                    }},
                                },
                            ],
                            "resources": {
                                "requests": {"memory": "512Mi", "cpu": "250m"},
                                "limits":   {"memory": "1Gi",   "cpu": "500m"},
                            },
                        }],
                    }
                },
            },
        }

    async def _wait_for_trivy_job(
        self, job_name: str, k8s_batch: Any
    ) -> dict[str, Any]:
        """Poll the Trivy Job until complete, then fetch and parse its log output."""
        k8s_core   = self._k8s_core_api()
        start_time = time.monotonic()

        while True:
            if time.monotonic() - start_time > self._timeout:
                raise RuntimeError(
                    f"Trivy Job {job_name!r} timed out after {self._timeout}s"
                )

            job = k8s_batch.read_namespaced_job(
                name=job_name, namespace=settings.k8s_namespace
            )
            if job.status.succeeded:
                break
            if job.status.failed:
                logs = self._fetch_pod_log(k8s_core, job_name)
                raise RuntimeError(
                    f"Trivy Job {job_name!r} failed. Last output:\n"
                    + "\n".join(logs.splitlines()[-20:])
                )
            time.sleep(_POLL_INTERVAL_SECONDS)

        log_text = self._fetch_pod_log(k8s_core, job_name)
        return self._extract_json_from_log(log_text)

    def _fetch_pod_log(self, k8s_core: Any, job_name: str) -> str:
        """Fetch stdout log from the first pod of the given Job."""
        pods = k8s_core.list_namespaced_pod(
            namespace=settings.k8s_namespace,
            label_selector=f"job-name={job_name}",
        )
        if not pods.items:
            raise RuntimeError(f"No pods found for Job {job_name!r}")
        pod_name = pods.items[0].metadata.name
        return k8s_core.read_namespaced_pod_log(
            name=pod_name,
            namespace=settings.k8s_namespace,
            container="trivy",
        )

    # ── Local subprocess path ─────────────────────────────────────────────────

    def _scan_via_subprocess(self, image_uri: str) -> dict[str, Any]:
        """
        Run `trivy image --format json` locally (dev mode only).
        Requires trivy binary on PATH.
        """
        logger.info("Running Trivy locally (dev mode) against %s", image_uri)
        cmd = [
            "trivy", "image",
            "--format",    "json",
            "--exit-code", "0",
            "--no-progress",
            "--timeout",   "10m",
            image_uri,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,  # exit-code 0 always — we check CVE counts ourselves
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Trivy binary not found on PATH. "
                "Install trivy or set k8s_in_cluster=true to use the K8s Job path."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Trivy subprocess timed out scanning {image_uri!r}"
            ) from exc

        if proc.returncode not in (0, 1):
            raise RuntimeError(
                f"Trivy subprocess exited with code {proc.returncode}.\n"
                f"stderr: {proc.stderr[:500]}"
            )

        return self._extract_json_from_log(proc.stdout)

    # ── JSON parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json_from_log(log_text: str) -> dict[str, Any]:
        """
        Extract the Trivy JSON report from log output.
        Trivy writes pure JSON to stdout; this handles any prefix noise.
        """
        # Find the first '{' that starts the JSON object
        start = log_text.find("{")
        if start == -1:
            raise RuntimeError(
                "Trivy output contains no JSON object. "
                f"Raw output (first 500 chars): {log_text[:500]!r}"
            )
        try:
            return json.loads(log_text[start:])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse Trivy JSON output: {exc}. "
                f"Raw output snippet: {log_text[start:start+200]!r}"
            ) from exc

    @staticmethod
    def _parse_trivy_json(
        report: dict[str, Any],
        image_uri: str,
        image_digest: str,
    ) -> TrivyScanResult:
        """
        Parse the Trivy JSON report into a typed TrivyScanResult.

        Supports SchemaVersion 2 (current Trivy output format).
        Handles missing 'Vulnerabilities' key gracefully (clean scan).
        """
        findings:  list[CVEFinding] = []
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

        for result_block in report.get("Results", []):
            for vuln in result_block.get("Vulnerabilities") or []:
                severity = (vuln.get("Severity") or "UNKNOWN").upper()
                if severity in counts:
                    counts[severity] += 1

                findings.append(CVEFinding(
                    vulnerability_id=  vuln.get("VulnerabilityID",  "UNKNOWN"),
                    severity=          severity,
                    package=           vuln.get("PkgName",           "unknown"),
                    installed_version= vuln.get("InstalledVersion",  "unknown"),
                    fixed_version=     vuln.get("FixedVersion"),
                    title=             vuln.get("Title",             ""),
                    description=       (vuln.get("Description") or "")[:256],
                ))

        total = sum(counts.values())

        # Apply gate thresholds from settings
        rejection_reason: Optional[str] = None
        if counts["CRITICAL"] > settings.trivy_critical_cve_limit:
            rejection_reason = (
                f"{counts['CRITICAL']} CRITICAL CVE(s) found — "
                f"limit is {settings.trivy_critical_cve_limit}"
            )
        elif counts["HIGH"] > settings.trivy_high_cve_limit:
            rejection_reason = (
                f"{counts['HIGH']} HIGH CVE(s) found — "
                f"limit is {settings.trivy_high_cve_limit}"
            )

        passed = rejection_reason is None

        return TrivyScanResult(
            image_uri=        image_uri,
            image_digest=     image_digest,
            critical_count=   counts["CRITICAL"],
            high_count=       counts["HIGH"],
            medium_count=     counts["MEDIUM"],
            low_count=        counts["LOW"],
            total_count=      total,
            passed=           passed,
            rejection_reason= rejection_reason,
            findings=         findings,
            raw_report=       report,
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    async def _persist(
        self, result: TrivyScanResult, workflow_id: Optional[str]
    ) -> int:
        """INSERT scan result into trivy_scan_results. Returns the row ID."""
        async with self._pool.acquire() as conn:
            row_id: int = await conn.fetchval(
                """
                INSERT INTO trivy_scan_results (
                    image_digest, image_tag, model_name, model_version,
                    critical_count, high_count, medium_count, low_count, total_count,
                    passed, raw_report, scan_duration_ms, workflow_id
                ) VALUES (
                    $1, $2, $3, $4,
                    $5, $6, $7, $8, $9,
                    $10, $11::jsonb, $12, $13
                )
                RETURNING id
                """,
                result.image_digest,
                result.image_uri,
                self._model_name,
                self._model_version,
                result.critical_count,
                result.high_count,
                result.medium_count,
                result.low_count,
                result.total_count,
                result.passed,
                json.dumps(result.raw_report),
                result.scan_duration_ms,
                workflow_id,
            )
        return row_id

    # ── K8s client factories ──────────────────────────────────────────────────

    @staticmethod
    def _k8s_batch_api() -> Any:
        from kubernetes import client as k8s_client, config as k8s_config
        try:
            if settings.k8s_in_cluster:
                k8s_config.load_incluster_config()
            else:
                k8s_config.load_kube_config()
        except Exception:
            pass
        return k8s_client.BatchV1Api()

    @staticmethod
    def _k8s_core_api() -> Any:
        from kubernetes import client as k8s_client
        return k8s_client.CoreV1Api()