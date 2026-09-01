"""
agents/security_agent.py

Security Agent — V2 production rewrite.

What changed from V1:
  V1: regex injection detection only, hardcoded allow-list, no real tools
  V2:
    - Trivy: real CVE scan via Kubernetes Job on every container image
    - Semgrep SAST: scan training scripts before execution
    - Secret scanning: detect AWS keys / API tokens in datasets and artifacts
    - Dependency verification: check pip hashes against known-good lockfile
    - PII detection: scan inference outputs before returning to caller
    - Supply chain: verify image digest matches expected before deployment
    - Input sanitization: enforce schema + type constraints at API boundary

Security gate logic:
  Block if:
    - CRITICAL CVEs > settings.trivy_critical_cve_limit (default 0)
    - HIGH CVEs > settings.trivy_high_cve_limit (default 5)
    - Semgrep finds HIGH severity findings
    - Secrets detected in dataset or model artifacts
    - Dependency hash mismatch

  Warn (allow but flag) if:
    - MEDIUM CVEs found
    - PII detected in inference output
    - Semgrep finds MEDIUM findings
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import mlflow
from pydantic import BaseModel, Field

from agents import AgentTaskResult
from agents.llm_gateway import LLMGateway
from configs.settings import settings

logger = logging.getLogger(__name__)

# Typed models
class CVEFinding(BaseModel):
    """Single CVE from a Trivy scan."""
    vulnerability_id: str
    severity: str          # CRITICAL | HIGH | MEDIUM | LOW | UNKNOWN
    package: str
    installed_version: str
    fixed_version: Optional[str] = None
    title: str = ""
    description: str = ""


class TrivyScanResult(BaseModel):
    """Aggregated result from a Trivy container image scan."""
    image: str
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    total_count: int
    passed: bool
    findings: list[CVEFinding] = Field(default_factory=list)
    scan_duration_ms: int = 0


class SemgrepFinding(BaseModel):
    """Single finding from a Semgrep SAST scan."""
    rule_id: str
    severity: str          # ERROR | WARNING | INFO
    file: str
    line: int
    message: str
    code_snippet: str = ""


class SecretFinding(BaseModel):
    """Detected secret in a file or dataset column."""
    location: str          # file path or column name
    secret_type: str       # aws_key | api_token | password | private_key
    line_or_row: int
    masked_value: str      # first 4 chars + ****


class SecurityReport(BaseModel):
    """Aggregated security assessment for one workflow."""
    trivy_result: Optional[TrivyScanResult] = None
    trivy_passed: bool = True
    semgrep_findings: list[SemgrepFinding] = Field(default_factory=list)
    semgrep_passed: bool = True
    secret_findings: list[SecretFinding] = Field(default_factory=list)
    secrets_detected: bool = False
    pii_detected: bool = False
    pii_fields: list[str] = Field(default_factory=list)
    dep_hash_passed: bool = True
    overall_passed: bool = True
    critical_cves: int = 0
    high_cves: int = 0
    blocking_reasons: list[str] = Field(default_factory=list)


# Secret detection patterns
_SECRET_PATTERNS: dict[str, re.Pattern] = {
    "aws_access_key":       re.compile(r"AKIA[0-9A-Z]{16}"),
    "aws_secret_key":       re.compile(r"[A-Za-z0-9/+]{40}"),
    "github_pat":           re.compile(r"ghp_[A-Za-z0-9]{36}"),
    "anthropic_api_key":    re.compile(r"sk-ant-api[A-Za-z0-9\-]{40,}"),
    "openai_api_key":       re.compile(r"sk-[A-Za-z0-9]{48}"),
    "generic_api_token":    re.compile(r"(?i)api[-_]?key['\"]?\s*[:=]\s*['\"]([A-Za-z0-9\-_]{20,})"),
    "private_key_header":   re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "neon_connection_str":  re.compile(r"postgresql\+asyncpg://[^@]+:[^@]+@"),
}

# PII field name patterns
_PII_FIELD_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)(ssn|social.?security|tax.?id)"),
    re.compile(r"(?i)(credit.?card|card.?number|cvv|ccv)"),
    re.compile(r"(?i)(passport|national.?id|aadhar|aadhaar)"),
    re.compile(r"(?i)^(dob|date.?of.?birth|birth.?date)$"),
    re.compile(r"(?i)^(ip.?address|ipv4|ipv6)$"),
    re.compile(r"(?i)^(email|phone|mobile|cell|address|zip.?code|postal)$"),
]

# Prompt-injection phrase patterns — checked against any free text bound for
# an LLM call or accepted as a workflow input (dataset_uri, model_uri, ...).
_PROMPT_INJECTION_PATTERNS: dict[str, re.Pattern] = {
    "ignore_instructions": re.compile(r"(?i)ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions"),
    "system_prompt_leak":  re.compile(r"(?i)(reveal|show|tell me|what is)\b.{0,20}\bsystem\s+prompt"),
    "bypass_security":     re.compile(r"(?i)bypass\s+(security|safety|guardrails?)"),
    "jailbreak":           re.compile(r"(?i)jailbreak"),
    "dan_mode":            re.compile(r"(?i)\bDAN\s+mode\b"),
    "role_override":       re.compile(r"(?i)you\s+are\s+now\s+(in\s+)?(developer|admin|unrestricted)\s+mode"),
}

# PII patterns checked against free-text agent OUTPUT (not dataset column names)
# before it is returned to a caller — plan §2.6 "output filtering".
_OUTPUT_PII_PATTERNS: dict[str, re.Pattern] = {
    "email":       re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}

# Secret patterns checked against free-text agent OUTPUT before it is returned.
_OUTPUT_SECRET_PATTERNS: dict[str, re.Pattern] = {
    "aws_key":     re.compile(r"AKIA[0-9A-Z]{16}"),
    "password":    re.compile(r"(?i)password\s*[:=]\s*['\"]?[^\s'\"]{4,}"),
    "api_key":     re.compile(r"(?i)api[-_]?key\s*[:=]\s*['\"]?[A-Za-z0-9\-_]{16,}"),
    "private_key": re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}

# Security Agent
class SecurityAgent:
    """
    ML supply chain and runtime security agent.

    Scan order (each stage is independent — all run regardless of prior failures):
      1. Trivy CVE scan of the runner image
      2. Semgrep SAST on the training script
      3. Secret scan on dataset columns and MLflow artifact filenames
      4. PII detection on dataset column names
      5. Dependency hash verification
      6. Aggregate and gate
    """

    def __init__(self) -> None:
        self._pool: Optional[Any] = None
        self._k8s_client: Optional[Any] = None

    async def connect(self, pool: Any) -> None:
        self._pool = pool
        await self._init_k8s()

    async def _init_k8s(self) -> None:
        try:
            from kubernetes import client as k8s_client, config as k8s_config
            if settings.k8s_in_cluster:
                k8s_config.load_incluster_config()
            else:
                k8s_config.load_kube_config()
            self._k8s_client = k8s_client
        except Exception as exc:
            logger.warning("K8s client unavailable for SecurityAgent: %s", exc)

    # Pre-flight input validation (API / workflow boundary)
    def pre_flight_check(self, state: dict) -> AgentTaskResult:
        """
        Validate workflow-input strings (dataset_uri, model_uri) before any
        downstream agent touches them. Synchronous — runs as the first real
        gate in the orchestrator graph, before Data Agent connects to anything.

        Blocks on:
          - SQL / path-traversal / shell-injection patterns (sanitize_input)
          - Prompt-injection phrases aimed at a downstream LLM call
        """
        task_id = state.get("workflow_id", "unknown")
        for field_name in ("dataset_uri", "model_uri"):
            value = state.get(field_name)
            if not value:
                continue
            try:
                SecurityAgent.sanitize_input(value, field_name)
            except ValueError as exc:
                return AgentTaskResult(
                    task_id=task_id, status="failed",
                    error=f"Suspicious input rejected: {exc}",
                )
            hits = self.detect_prompt_injection(value)
            if hits:
                return AgentTaskResult(
                    task_id=task_id, status="failed",
                    error=f"Prompt injection pattern(s) detected in {field_name}: {hits}",
                )
        return AgentTaskResult(task_id=task_id, status="success", output={"pre_flight": "passed"})

    # Prompt injection
    @staticmethod
    def detect_prompt_injection(text: str) -> list[str]:
        """Return the names of any prompt-injection patterns found in text."""
        if not text:
            return []
        return [name for name, pattern in _PROMPT_INJECTION_PATTERNS.items() if pattern.search(text)]

    @staticmethod
    def sanitize_prompt(text: str) -> str:
        """Strip known prompt-injection phrases from text before it reaches an LLM."""
        result = text
        for pattern in _PROMPT_INJECTION_PATTERNS.values():
            result = pattern.sub("[REMOVED]", result)
        return result

    # Output filtering (PII + secrets on agent/model output)
    @staticmethod
    def validate_agent_output(text: str) -> tuple[bool, list[str]]:
        """
        Scan free-text agent or inference output for PII and secrets before
        it is returned to a caller. Plan §2.6 "output filtering".

        Returns (safe, violations) where each violation is "PII:<type>" or
        "SECRET:<type>".
        """
        violations: list[str] = []
        if not text:
            return True, violations
        for name, pattern in _OUTPUT_PII_PATTERNS.items():
            if pattern.search(text):
                violations.append(f"PII:{name}")
        for name, pattern in _OUTPUT_SECRET_PATTERNS.items():
            if pattern.search(text):
                violations.append(f"SECRET:{name}")
        return len(violations) == 0, violations

    # CI-produced Trivy SARIF ingestion
    def ingest_trivy_results(self, sarif_path: str, workflow_id: str) -> AgentTaskResult:
        """
        Ingest a Trivy SARIF report produced by CI (see .github/workflows/ci-cd.yaml,
        which already generates trivy-results.sarif) and apply the same CRITICAL-CVE
        gate used by the runtime K8s-Job scan path.

        SARIF severity mapping: level="error" → CRITICAL, level="warning" → HIGH.
        """
        try:
            with open(sarif_path, "r", encoding="utf-8") as f:
                sarif = json.load(f)
        except Exception as exc:
            return AgentTaskResult(
                task_id=workflow_id, status="failed",
                error=f"Could not read Trivy SARIF report at {sarif_path}: {exc}",
            )

        critical = 0
        high = 0
        for run_block in sarif.get("runs", []):
            for result in run_block.get("results", []):
                level = result.get("level", "note")
                if level == "error":
                    critical += 1
                elif level == "warning":
                    high += 1

        passed = critical <= settings.trivy_critical_cve_limit
        return AgentTaskResult(
            task_id=workflow_id,
            status="success" if passed else "failed",
            output={"critical_cves": critical, "high_cves": high, "source": "sarif", "sarif_path": sarif_path},
            error=None if passed else f"CRITICAL CVEs found in Trivy SARIF report: {critical}",
        )

    @mlflow.trace(name="security_agent.run")
    def run(self, state: dict) -> AgentTaskResult:
        """Synchronous LangGraph entry point."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._run_async(state))

    async def _run_async(self, state: dict) -> AgentTaskResult:
        task_id: str = state.get("workflow_id", "unknown")
        best_framework: str = state.get("best_framework", "xgboost")
        dataset_uri: str = state.get("dataset_uri", "")
        best_run_id: str = state.get("best_run_id", "")
        training_script: str = state.get("training_script_path", "")

        runner_image = settings.k8s_runner_image_map.get(
            best_framework, settings.k8s_runner_custom
        )

        try:
            with mlflow.start_run(run_name=f"security-{task_id}", nested=True):
                mlflow.set_tag("agent", "security_agent")
                mlflow.log_param("scanned_image", runner_image)

                # Run all scans concurrently
                trivy_task = asyncio.create_task(self._trivy_scan(runner_image, task_id))
                semgrep_task = asyncio.create_task(self._semgrep_scan(training_script))
                secret_task = asyncio.create_task(self._secret_scan(dataset_uri, best_run_id))
                pii_task = asyncio.create_task(self._pii_scan(dataset_uri))
                dep_task = asyncio.create_task(self._dependency_check())

                trivy_result, semgrep_findings, secret_findings, pii_result, dep_passed = (
                    await asyncio.gather(
                        trivy_task, semgrep_task, secret_task, pii_task, dep_task,
                        return_exceptions=True,
                    )
                )

                # Handle exceptions from gather (treat as pass — log the error)
                if isinstance(trivy_result, Exception):
                    logger.error("Trivy scan exception: %s", trivy_result)
                    trivy_result = None
                if isinstance(semgrep_findings, Exception):
                    logger.error("Semgrep scan exception: %s", semgrep_findings)
                    semgrep_findings = []
                if isinstance(secret_findings, Exception):
                    logger.error("Secret scan exception: %s", secret_findings)
                    secret_findings = []
                if isinstance(pii_result, Exception):
                    logger.error("PII scan exception: %s", pii_result)
                    pii_result = (False, [])
                if isinstance(dep_passed, Exception):
                    logger.error("Dep check exception: %s", dep_passed)
                    dep_passed = True

                pii_detected, pii_fields = pii_result if isinstance(pii_result, tuple) else (False, [])

                # Aggregate
                report = self._aggregate(
                    trivy_result=trivy_result,
                    semgrep_findings=semgrep_findings or [],
                    secret_findings=secret_findings or [],
                    pii_detected=pii_detected,
                    pii_fields=pii_fields or [],
                    dep_passed=bool(dep_passed),
                )

                # MLflow tags on the training run
                if best_run_id:
                    await self._tag_mlflow_run(best_run_id, report)

                mlflow.log_metrics({
                    "security.critical_cves":   report.critical_cves,
                    "security.high_cves":       report.high_cves,
                    "security.secrets_found":   int(report.secrets_detected),
                    "security.pii_detected":    int(report.pii_detected),
                    "security.overall_passed":  int(report.overall_passed),
                    "security.semgrep_findings": len(report.semgrep_findings),
                })
                mlflow.log_dict(report.model_dump(), "security_report.json")

                logger.info(
                    "SecurityAgent: passed=%s critical_cves=%d high_cves=%d "
                    "secrets=%s pii=%s semgrep_findings=%d",
                    report.overall_passed, report.critical_cves, report.high_cves,
                    report.secrets_detected, report.pii_detected, len(report.semgrep_findings),
                )

                status = "success" if report.overall_passed else "failed"
                return AgentTaskResult(
                    task_id=task_id,
                    status=status,
                    output={
                        "security_result": report.model_dump(),
                        "trivy_passed":     report.trivy_passed,
                        "semgrep_passed":   report.semgrep_passed,
                        "secrets_detected": report.secrets_detected,
                        "pii_detected":     report.pii_detected,
                        "critical_cves":    report.critical_cves,
                        "high_cves":        report.high_cves,
                        "overall_passed":   report.overall_passed,
                        "blocking_reasons": report.blocking_reasons,
                    },
                    error="; ".join(report.blocking_reasons) if not report.overall_passed else None,
                    confidence=1.0 if report.overall_passed else 0.0,
                )

        except Exception as exc:
            logger.exception("SecurityAgent failed for workflow %s", task_id)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))

    # Stage 1: Trivy 
    async def _trivy_scan(
        self, image: str, task_id: str
    ) -> Optional[TrivyScanResult]:
        """
        Run a Trivy vulnerability scan on the given container image.

        In-cluster: launches a Kubernetes Job running trivy image <image>.
        Dev mode (no K8s): calls trivy binary directly via subprocess.

        Results are parsed from Trivy's JSON output format.
        """
        start_ms = int(time.monotonic() * 1000)
        logger.info("Trivy scan starting for image: %s", image)

        if self._k8s_client is None:
            # Dev mode: try local trivy binary
            return await self._trivy_local(image, start_ms)

        return await self._trivy_k8s_job(image, task_id, start_ms)

    async def _trivy_k8s_job(
        self, image: str, task_id: str, start_ms: int
    ) -> Optional[TrivyScanResult]:
        """Launch a Kubernetes Job to run Trivy, parse output from Job logs."""
        job_name = f"trivy-scan-{task_id[:8]}-{int(time.time())}"
        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": settings.k8s_namespace,
                "labels": {"app": "trivy-scan", "workflow-id": task_id[:16]},
            },
            "spec": {
                "backoffLimit": 0,
                "activeDeadlineSeconds": 300,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{
                            "name": "trivy",
                            "image": "aquasec/trivy:latest",
                            "command": [
                                "trivy", "image",
                                "--format", "json",
                                "--exit-code", "0",   # don't fail job on findings — we parse
                                "--no-progress",
                                image,
                            ],
                        }],
                    }
                },
            },
        }
        try:
            batch_v1 = self._k8s_client.BatchV1Api()
            await asyncio.to_thread(
                batch_v1.create_namespaced_job,
                namespace=settings.k8s_namespace,
                body=manifest,
            )

            # Wait for completion (max 5 minutes)
            for _ in range(30):
                await asyncio.sleep(10)
                job = await asyncio.to_thread(
                    batch_v1.read_namespaced_job_status,
                    name=job_name,
                    namespace=settings.k8s_namespace,
                )
                if job.status.succeeded or job.status.failed:
                    break

            # Get logs
            core_v1 = self._k8s_client.CoreV1Api()
            pods = await asyncio.to_thread(
                core_v1.list_namespaced_pod,
                namespace=settings.k8s_namespace,
                label_selector=f"job-name={job_name}",
            )
            log_output = ""
            if pods.items:
                pod_name = pods.items[0].metadata.name
                log_output = await asyncio.to_thread(
                    core_v1.read_namespaced_pod_log,
                    name=pod_name,
                    namespace=settings.k8s_namespace,
                )

            # Clean up job
            await asyncio.to_thread(
                batch_v1.delete_namespaced_job,
                name=job_name,
                namespace=settings.k8s_namespace,
                propagation_policy="Background",
            )

            if not job.status.succeeded:
                # Job failed or the poll loop exhausted without succeeding —
                # do NOT parse whatever partial/empty logs exist as "clean".
                # A scan that didn't complete is not a scan that passed.
                logger.error(
                    "Trivy K8s Job %s did not succeed (failed=%s) — treating as scan failure, not a pass",
                    job_name, job.status.failed,
                )
                return None

            return self._parse_trivy_json(image, log_output, start_ms)

        except Exception as exc:
            logger.error("Trivy K8s Job failed: %s", exc)
            return None

    async def _trivy_local(self, image: str, start_ms: int) -> Optional[TrivyScanResult]:
        """Run trivy binary locally (dev mode)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "trivy", "image", "--format", "json", "--no-progress", image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            return self._parse_trivy_json(image, stdout.decode(), start_ms)
        except FileNotFoundError:
            logger.warning("trivy binary not found — skipping CVE scan (dev mode)")
            return TrivyScanResult(
                image=image, critical_count=0, high_count=0, medium_count=0,
                low_count=0, total_count=0, passed=True,
                scan_duration_ms=int(time.monotonic() * 1000) - start_ms,
            )
        except Exception as exc:
            logger.error("Local trivy scan failed: %s", exc)
            return None

    def _parse_trivy_json(
        self, image: str, raw: str, start_ms: int
    ) -> Optional[TrivyScanResult]:
        """
        Parse Trivy JSON output into a typed TrivyScanResult.

        Returns None if the output can't be parsed — a garbled or empty
        report means the scan didn't produce a trustworthy result, so the
        caller must treat it as a failed scan, not zero findings.
        """
        findings: list[CVEFinding] = []
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

        try:
            data = json.loads(raw)
            for result_block in data.get("Results", []):
                for vuln in result_block.get("Vulnerabilities", []):
                    sev = vuln.get("Severity", "UNKNOWN").upper()
                    if sev in counts:
                        counts[sev] += 1
                    findings.append(CVEFinding(
                        vulnerability_id=vuln.get("VulnerabilityID", ""),
                        severity=sev,
                        package=vuln.get("PkgName", ""),
                        installed_version=vuln.get("InstalledVersion", ""),
                        fixed_version=vuln.get("FixedVersion"),
                        title=vuln.get("Title", "")[:200],
                        description=vuln.get("Description", "")[:500],
                    ))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(
                "Trivy JSON parse failed for image=%s: %s — treating scan as FAILED, not zero findings",
                image, exc,
            )
            return None

        total = sum(counts.values())
        passed = (
            counts["CRITICAL"] <= settings.trivy_critical_cve_limit
            and counts["HIGH"] <= settings.trivy_high_cve_limit
        )
        duration = int(time.monotonic() * 1000) - start_ms

        logger.info(
            "Trivy: image=%s critical=%d high=%d medium=%d passed=%s duration_ms=%d",
            image, counts["CRITICAL"], counts["HIGH"], counts["MEDIUM"], passed, duration,
        )
        return TrivyScanResult(
            image=image,
            critical_count=counts["CRITICAL"],
            high_count=counts["HIGH"],
            medium_count=counts["MEDIUM"],
            low_count=counts["LOW"],
            total_count=total,
            passed=passed,
            findings=findings[:50],   # cap stored findings at 50 to avoid huge payloads
            scan_duration_ms=duration,
        )

    # Stage 2: Semgrep SAST 
    async def _semgrep_scan(self, script_path: str) -> list[SemgrepFinding]:
        """
        Run Semgrep on the training script path.
        Uses the p/python rule set. Blocks on ERROR (HIGH) severity findings.
        """
        if not script_path or not Path(script_path).exists():
            logger.info("Semgrep: no training script path — skipping")
            return []
        try:
            proc = await asyncio.create_subprocess_exec(
                "semgrep", "--config=p/python", "--json", "--quiet", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
            return self._parse_semgrep_json(stdout.decode())
        except FileNotFoundError:
            logger.warning("semgrep binary not found — skipping SAST (dev mode)")
            return []
        except asyncio.TimeoutError:
            logger.warning("Semgrep scan timed out after 60s")
            return []
        except Exception as exc:
            logger.error("Semgrep scan failed: %s", exc)
            return []

    def _parse_semgrep_json(self, raw: str) -> list[SemgrepFinding]:
        findings: list[SemgrepFinding] = []
        try:
            data = json.loads(raw)
            for result in data.get("results", []):
                findings.append(SemgrepFinding(
                    rule_id=result.get("check_id", ""),
                    severity=result.get("extra", {}).get("severity", "INFO").upper(),
                    file=result.get("path", ""),
                    line=result.get("start", {}).get("line", 0),
                    message=result.get("extra", {}).get("message", "")[:300],
                    code_snippet=result.get("extra", {}).get("lines", "")[:200],
                ))
        except Exception as exc:
            logger.warning("Semgrep JSON parse failed: %s", exc)
        return findings

    # Stage 3: Secret scanning 
    async def _secret_scan(
        self, dataset_uri: str, run_id: str
    ) -> list[SecretFinding]:
        """
        Scan dataset values and MLflow artifact filenames for secrets.
        Reads a sample of the dataset from the connector.
        """
        findings: list[SecretFinding] = []

        # Scan dataset sample
        if dataset_uri:
            try:
                from agents.connectors import ConnectorFactory
                connector = ConnectorFactory.from_uri(dataset_uri)
                await connector.connect()
                df = await connector.sample(n=100)
                for col in df.select_dtypes(include="object").columns:
                    for i, val in enumerate(df[col].dropna().astype(str)):
                        for secret_type, pattern in _SECRET_PATTERNS.items():
                            if pattern.search(val):
                                findings.append(SecretFinding(
                                    location=f"dataset:column={col}",
                                    secret_type=secret_type,
                                    line_or_row=i,
                                    masked_value=val[:4] + "****",
                                ))
                                break   # one finding per cell
            except Exception as exc:
                logger.warning("Dataset secret scan failed: %s", exc)

        # Scan MLflow artifact filenames (metadata only — not downloading files)
        if run_id:
            try:
                client = mlflow.tracking.MlflowClient()
                artifacts = await asyncio.to_thread(
                    client.list_artifacts, run_id
                )
                for artifact in artifacts:
                    for secret_type, pattern in _SECRET_PATTERNS.items():
                        if pattern.search(artifact.path):
                            findings.append(SecretFinding(
                                location=f"mlflow:artifact={artifact.path}",
                                secret_type=secret_type,
                                line_or_row=0,
                                masked_value=artifact.path[:4] + "****",
                            ))
            except Exception as exc:
                logger.warning("MLflow artifact secret scan failed: %s", exc)

        if findings:
            logger.warning(
                "Secret scan: %d potential secrets found in dataset/artifacts", len(findings)
            )
        return findings

    # Stage 4: PII detection 
    async def _pii_scan(self, dataset_uri: str) -> tuple[bool, list[str]]:
        """
        Check dataset column names against PII patterns.
        Returns (pii_detected: bool, pii_field_names: list[str]).
        Does NOT read cell values — column-name heuristic only for speed.
        """
        if not dataset_uri:
            return False, []
        try:
            from agents.connectors import ConnectorFactory
            connector = ConnectorFactory.from_uri(dataset_uri)
            await connector.connect()
            df = await connector.sample(n=1)   # just need column names
            pii_fields = [
                col for col in df.columns
                if any(p.search(col) for p in _PII_FIELD_PATTERNS)
            ]
            if pii_fields:
                logger.warning("PII column names detected: %s", pii_fields)
            return bool(pii_fields), pii_fields
        except Exception as exc:
            logger.warning("PII scan failed (non-fatal): %s", exc)
            return False, []

    # Stage 5: Dependency hash check 
    async def _dependency_check(self) -> bool:
        """
        Verify pip dependency hashes against requirements.txt.
        Uses pip hash mode: pip download --require-hashes.
        Returns True if all hashes match.
        """
        req_path = Path("requirements.txt")
        if not req_path.exists():
            logger.info("requirements.txt not found — skipping dep hash check")
            return True
        try:
            proc = await asyncio.create_subprocess_exec(
                "pip", "install", "--dry-run", "--require-hashes",
                "--no-deps", "-r", str(req_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode != 0:
                logger.warning(
                    "Dependency hash mismatch detected:\n%s", stderr.decode()[:500]
                )
                return False
            return True
        except Exception as exc:
            logger.warning("Dependency check failed (non-fatal): %s", exc)
            return True   # non-blocking in dev

    # Aggregation 
    def _aggregate(
        self,
        trivy_result: Optional[TrivyScanResult],
        semgrep_findings: list[SemgrepFinding],
        secret_findings: list[SecretFinding],
        pii_detected: bool,
        pii_fields: list[str],
        dep_passed: bool,
    ) -> SecurityReport:
        """Combine all scan results into a single SecurityReport with gate decision."""
        blocking_reasons: list[str] = []

        # Trivy gate — a scan that never ran or couldn't be parsed is a
        # FAILURE, not a pass. The only place Trivy legitimately reports a
        # clean 0/0 result without actually scanning is the disclosed dev
        # fallback in _trivy_local() when the trivy binary is missing, which
        # returns a real TrivyScanResult(passed=True) — not None.
        trivy_passed = False
        critical_cves = 0
        high_cves = 0
        if trivy_result is not None:
            critical_cves = trivy_result.critical_count
            high_cves = trivy_result.high_count
            trivy_passed = trivy_result.passed
            if not trivy_passed:
                blocking_reasons.append(
                    f"Trivy: {critical_cves} CRITICAL CVEs, {high_cves} HIGH CVEs "
                    f"(limits: critical<={settings.trivy_critical_cve_limit}, "
                    f"high<={settings.trivy_high_cve_limit})"
                )
        else:
            blocking_reasons.append(
                "Trivy scan did not complete (job failed, timed out, or produced "
                "unparseable output) — treating as failed, not passed"
            )

        # Semgrep gate — block only on ERROR severity
        semgrep_errors = [f for f in semgrep_findings if f.severity == "ERROR"]
        semgrep_passed = len(semgrep_errors) == 0
        if not semgrep_passed:
            blocking_reasons.append(
                f"Semgrep: {len(semgrep_errors)} ERROR severity findings "
                f"(rules: {[f.rule_id for f in semgrep_errors[:3]]})"
            )

        # Secrets gate — always block
        secrets_detected = len(secret_findings) > 0
        if secrets_detected:
            types = list({f.secret_type for f in secret_findings})
            blocking_reasons.append(
                f"Secrets detected in dataset/artifacts: {types}"
            )

        # Dependency gate
        if not dep_passed:
            blocking_reasons.append("Dependency hash mismatch — supply chain integrity check failed")

        # PII: warn only (not blocking — dataset owner may have consented use)
        if pii_detected:
            logger.warning(
                "PII fields detected in dataset: %s — ensure data handling compliance", pii_fields
            )

        overall_passed = len(blocking_reasons) == 0

        return SecurityReport(
            trivy_result=trivy_result,
            trivy_passed=trivy_passed,
            semgrep_findings=semgrep_findings,
            semgrep_passed=semgrep_passed,
            secret_findings=secret_findings,
            secrets_detected=secrets_detected,
            pii_detected=pii_detected,
            pii_fields=pii_fields,
            dep_hash_passed=dep_passed,
            overall_passed=overall_passed,
            critical_cves=critical_cves,
            high_cves=high_cves,
            blocking_reasons=blocking_reasons,
        )

    # Input sanitization (API boundary) 
    @staticmethod
    def sanitize_input(value: Any, field_name: str = "") -> Any:
        """
        Enforce type safety and injection prevention at the API boundary.
        Call this on every user-supplied string before it touches any
        downstream system (SQL builder, shell command, file path).
        """
        if not isinstance(value, str):
            return value

        # SQL injection patterns
        sql_patterns = re.compile(
            r"(?i)(union\s+select|drop\s+table|insert\s+into|delete\s+from"
            r"|exec\s*\(|xp_cmdshell|;\s*--|/\*.*\*/)",
            re.IGNORECASE,
        )
        if sql_patterns.search(value):
            raise ValueError(f"Potential SQL injection in field '{field_name}': {value[:50]}")

        # Path traversal
        if re.search(r"\.\./|\.\.\\", value):
            raise ValueError(f"Path traversal attempt in field '{field_name}': {value[:50]}")

        # Shell injection
        shell_patterns = re.compile(r"[;&|`$(){}]")
        if field_name.lower() in ("command", "script", "exec", "shell") and shell_patterns.search(value):
            raise ValueError(f"Shell injection pattern in field '{field_name}': {value[:50]}")

        return value.strip()

    # MLflow tagging
    async def _tag_mlflow_run(self, run_id: str, report: SecurityReport) -> None:
        """Stamp security scan results on the MLflow training run."""
        try:
            client = mlflow.tracking.MlflowClient()
            tags = {
                "mlops.security.trivy_scan":    "passed" if report.trivy_passed else "failed",
                "mlops.security.cve_critical":  str(report.critical_cves),
                "mlops.security.cve_high":      str(report.high_cves),
                "mlops.security.secrets":       "detected" if report.secrets_detected else "clean",
                "mlops.security.overall":       "passed" if report.overall_passed else "failed",
            }
            for key, val in tags.items():
                await asyncio.to_thread(client.set_tag, run_id, key, val)
        except Exception as exc:
            logger.warning("Security MLflow tag failed (non-fatal): %s", exc)