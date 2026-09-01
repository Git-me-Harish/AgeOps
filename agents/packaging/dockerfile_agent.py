"""
agents/packaging/dockerfile_agent.py

Phase 4 — DockerfileAgent: End-to-End OCI Packaging Pipeline Orchestrator.

This is the main Phase 4 agent.  It is triggered automatically after the
PromotionStateMachine transitions a model to Staging and the Evaluation gate
passes.  It runs the complete OCI packaging pipeline:

  Step 1  Read Phase 3 mandatory tags from model_registry_tags (Neon)
  Step 2  Resolve image tag from model name + version + registry config
  Step 3  Generate Dockerfile (DockerfileGenerator) — framework-specific
  Step 4  Create oci_build_jobs row (status=pending)
  Step 5  Upload build context + trigger Kaniko Job (KanikoBuilder)
  Step 6  Run Trivy CVE scan (TrivyScanner) → gate on CRITICAL CVEs
  Step 7  Generate SPDX SBOM (SBOMGenerator) → upload to R2
  Step 8  Sign image with Cosign (CosignSigner) → Rekor transparency log
  Step 9  Persist oci_images row with digest + labels
  Step 10 Update Phase 3 lineage graph: record_oci_image() on LineageGraphBuilder
  Step 11 Patch MLflow model version tags with OCI image URI and digest
  Step 12 Update oci_build_jobs row (status=succeeded)
  Step 13 Wire Phase 3 OCI gate: gate now passes for this (model_name, version)

Failure handling:
  - Trivy gate failure (CRITICAL CVEs): sets oci_build_jobs.status=failed,
    raises DockerfileAgentError — pipeline halts, model stays in Staging.
  - Any step raising an exception: oci_build_jobs.status=failed, error_message
    recorded, exception re-raised to the orchestrator.
  - All steps are logged with structured fields — no silent swallowing.

MLflow tags written after successful build:
  mlops.oci.image_uri    = <registry>/model:v3@sha256:...
  mlops.oci.image_digest = sha256:...
  mlops.oci.sbom_uri     = s3://bucket/security/sbom/...
  mlops.oci.cosign_signed = true | false
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import mlflow
from pydantic import BaseModel

from agents.packaging.dockerfile_generator import DockerfileConfig, DockerfileGenerator
from agents.packaging.kaniko_builder import KanikoBuilder, KanikoBuildError
from agents.packaging.trivy_scanner import TrivyScanner, TrivyScanResult
from agents.packaging.sbom_generator import SBOMGenerator
from agents.packaging.cosign_signer import CosignSigner
from agents.registry.lineage_graph import LineageGraphBuilder, LineageGraphQuerier
from agents.registry.metadata_schema import ModelRegistryTags, TagValidator
from configs.settings import settings

logger = logging.getLogger(__name__)

# MLflow tag keys written after a successful OCI build
OCI_TAG_IMAGE_URI     = "mlops.oci.image_uri"
OCI_TAG_IMAGE_DIGEST  = "mlops.oci.image_digest"
OCI_TAG_SBOM_URI      = "mlops.oci.sbom_uri"
OCI_TAG_COSIGN_SIGNED = "mlops.oci.cosign_signed"


class DockerfileAgentError(RuntimeError):
    """Raised when the OCI packaging pipeline fails at any step."""


# ── Input / output models ──────────────────────────────────────────────────────

class PackagingRequest(BaseModel):
    """Input to DockerfileAgent.run()."""
    model_name:    str
    model_version: str
    workflow_id:   Optional[str] = None
    serving_dir:   str = "serving"   # path relative to repo root


class PackagingResult(BaseModel):
    """Returned by DockerfileAgent.run() on success."""
    model_name:    str
    model_version: str
    image_tag:     str
    image_digest:  str
    image_uri:     str          # tag@digest
    sbom_uri:      str
    cosign_signed: bool
    build_job_db_id: int
    oci_image_db_id: int
    duration_seconds: int


# ── Agent ─────────────────────────────────────────────────────────────────────

class DockerfileAgent:
    """
    Orchestrates the full OCI packaging pipeline.

    Args:
        pool:          asyncpg.Pool connected to Neon.
        mlflow_client: mlflow.MlflowClient (injected for testability).
    """

    def __init__(
        self,
        pool:          Any,
        mlflow_client: Optional[Any] = None,
    ) -> None:
        self._pool   = pool
        self._client = mlflow_client or mlflow.MlflowClient()

    @mlflow.trace(name="dockerfile_agent.run")
    async def run(self, request: PackagingRequest) -> PackagingResult:
        """
        Execute the full packaging pipeline.

        Raises:
            DockerfileAgentError: on any pipeline failure (CVE gate, build, etc.).
        """
        import time
        start = time.monotonic()

        model_name    = request.model_name
        model_version = request.model_version
        workflow_id   = request.workflow_id

        logger.info(
            "DockerfileAgent starting: model=%s version=%s workflow=%s",
            model_name, model_version, workflow_id,
        )

        # ── Step 1: Load Phase 3 tags from Neon ──────────────────────────────
        tags = await self._load_tags(model_name, model_version)

        # ── Step 2: Resolve image tag ─────────────────────────────────────────
        image_tag = self._resolve_image_tag(model_name, model_version)

        # ── Step 3: Generate Dockerfile ───────────────────────────────────────
        config = DockerfileConfig(
            model_name=          model_name,
            model_version=       model_version,
            framework=           tags.framework,
            tags=                tags,
            mlflow_run_id=       self._get_run_id(model_name, model_version),
            mlflow_tracking_uri= settings.mlflow_tracking_uri,
            image_tag=           image_tag,
        )
        generator = DockerfileGenerator()
        generated = generator.generate(config)
        logger.info(
            "Dockerfile generated: framework=%s base_image=%s",
            generated.framework, generated.base_image,
        )

        # ── Step 4: Create oci_build_jobs row ─────────────────────────────────
        build_job_id = await self._create_build_job_row(
            model_name, model_version, image_tag, workflow_id
        )

        # ── Steps 5–12: Wrapped so any failure updates the job row ────────────
        try:
            result = await self._execute_pipeline(
                model_name=      model_name,
                model_version=   model_version,
                image_tag=       image_tag,
                generated=       generated,
                tags=            tags,
                build_job_id=    build_job_id,
                workflow_id=     workflow_id,
                serving_dir=     Path(request.serving_dir),
                start_time=      start,
            )
        except Exception as exc:
            await self._fail_build_job(build_job_id, str(exc))
            raise DockerfileAgentError(
                f"Packaging pipeline failed for {model_name} v{model_version}: {exc}"
            ) from exc

        return result

    # ── Pipeline execution ────────────────────────────────────────────────────

    async def _execute_pipeline(
        self,
        model_name:    str,
        model_version: str,
        image_tag:     str,
        generated:     Any,
        tags:          ModelRegistryTags,
        build_job_id:  int,
        workflow_id:   Optional[str],
        serving_dir:   Path,
        start_time:    float,
    ) -> PackagingResult:
        import time

        # ── Step 5: Kaniko build ───────────────────────────────────────────────
        builder = KanikoBuilder(
            model_name=    model_name,
            model_version= model_version,
            pool=          self._pool,
        )
        build_result = await builder.build(
            dockerfile_content= generated.dockerfile_content,
            model_info_json=    generated.model_info_json,
            serving_dir=        serving_dir,
            image_tag=          image_tag,
            build_job_db_id=    build_job_id,
        )
        logger.info(
            "Kaniko build succeeded: digest=%s job=%s",
            build_result.image_digest, build_result.job_name,
        )

        # ── Step 6: Trivy CVE scan + gate ─────────────────────────────────────
        scanner = TrivyScanner(model_name, model_version, self._pool)
        scan_result = await scanner.scan(
            image_uri=    build_result.image_uri,
            image_digest= build_result.image_digest,
            workflow_id=  workflow_id,
        )
        if not scan_result.passed:
            raise DockerfileAgentError(
                f"Trivy CVE gate BLOCKED: {scan_result.rejection_reason}"
            )
        logger.info(
            "Trivy gate passed: critical=%d high=%d medium=%d",
            scan_result.critical_count,
            scan_result.high_count,
            scan_result.medium_count,
        )

        # ── Step 7: SBOM generation ───────────────────────────────────────────
        sbom_gen = SBOMGenerator(model_name, model_version, self._pool)
        sbom_record = await sbom_gen.generate(
            image_uri=    build_result.image_uri,
            image_digest= build_result.image_digest,
            workflow_id=  workflow_id,
        )
        logger.info("SBOM generated: packages=%d uri=%s", sbom_record.package_count, sbom_record.r2_uri)

        # ── Step 8: Cosign signing ─────────────────────────────────────────────
        signer = CosignSigner(model_name, model_version, self._pool)
        attest = await signer.sign(
            image_uri=    build_result.image_uri,
            image_digest= build_result.image_digest,
            workflow_id=  workflow_id,
        )

        # ── Step 9: Persist oci_images row ────────────────────────────────────
        oci_image_id = await self._persist_oci_image(
            model_name=    model_name,
            model_version= model_version,
            build_result=  build_result,
            generated=     generated,
            build_job_id=  build_job_id,
            workflow_id=   workflow_id,
        )

        # ── Step 10: Update Phase 3 lineage graph ─────────────────────────────
        await self._update_lineage(
            model_name=    model_name,
            model_version= model_version,
            image_uri=     build_result.image_uri,
            image_digest=  build_result.image_digest,
        )

        # ── Step 11: Patch MLflow tags ─────────────────────────────────────────
        self._patch_mlflow_tags(
            model_name=    model_name,
            model_version= model_version,
            image_uri=     build_result.image_uri,
            image_digest=  build_result.image_digest,
            sbom_uri=      sbom_record.r2_uri,
            cosign_signed= attest.signed,
        )

        # ── Step 12: Mark build job succeeded ────────────────────────────────
        duration = int(time.monotonic() - start_time)
        await self._succeed_build_job(
            build_job_id=     build_job_id,
            log_uri=          build_result.log_uri,
            duration_seconds= duration,
        )

        logger.info(
            "DockerfileAgent complete: model=%s version=%s image=%s duration=%ds",
            model_name, model_version, build_result.image_uri, duration,
        )

        return PackagingResult(
            model_name=       model_name,
            model_version=    model_version,
            image_tag=        build_result.image_tag,
            image_digest=     build_result.image_digest,
            image_uri=        build_result.image_uri,
            sbom_uri=         sbom_record.r2_uri,
            cosign_signed=    attest.signed,
            build_job_db_id=  build_job_id,
            oci_image_db_id=  oci_image_id,
            duration_seconds= duration,
        )

    # ── DB helpers ────────────────────────────────────────────────────────────

    async def _load_tags(
        self, model_name: str, model_version: str
    ) -> ModelRegistryTags:
        """Load and re-validate Phase 3 tags from model_registry_tags."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    dataset_uri, dataset_hash, dataset_row_count,
                    framework, framework_version, python_version,
                    eval_accuracy, eval_f1, eval_auc,
                    eval_holdout_hash, eval_bias_passed,
                    security_trivy_scan, security_cve_critical,
                    git_commit, git_repo,
                    approved_by, approved_at, governance_audit_id
                FROM model_registry_tags
                WHERE model_name=$1 AND model_version=$2
                """,
                model_name, model_version,
            )
        if row is None:
            raise DockerfileAgentError(
                f"No model_registry_tags row for {model_name} v{model_version}. "
                "Run RegistrationGate.register() first (Phase 3)."
            )
        # Re-validate through TagValidator to get a typed ModelRegistryTags
        raw = {
            "mlops.dataset.uri":           row["dataset_uri"],
            "mlops.dataset.hash":          row["dataset_hash"],
            "mlops.dataset.row_count":     str(row["dataset_row_count"]),
            "mlops.framework":             row["framework"],
            "mlops.framework.version":     row["framework_version"],
            "mlops.python.version":        row["python_version"],
            "mlops.eval.accuracy":         str(row["eval_accuracy"]),
            "mlops.eval.f1":               str(row["eval_f1"]),
            "mlops.eval.auc":              str(row["eval_auc"]),
            "mlops.eval.holdout_hash":     row["eval_holdout_hash"],
            "mlops.eval.bias_passed":      "true" if row["eval_bias_passed"] else "false",
            "mlops.security.trivy_scan":   row["security_trivy_scan"],
            "mlops.security.cve_critical": str(row["security_cve_critical"]),
            "mlops.git.commit":            row["git_commit"],
            "mlops.git.repo":              row["git_repo"],
        }
        if row["approved_by"]:
            raw["mlops.approved_by"] = row["approved_by"]
        if row["approved_at"]:
            raw["mlops.approved_at"] = row["approved_at"].isoformat()
        if row["governance_audit_id"]:
            raw["mlops.governance.audit_id"] = row["governance_audit_id"]

        return TagValidator.validate_for_staging(raw)

    async def _create_build_job_row(
        self,
        model_name:    str,
        model_version: str,
        image_tag:     str,
        workflow_id:   Optional[str],
    ) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                """
                INSERT INTO oci_build_jobs
                    (model_name, model_version, image_tag, status, workflow_id)
                VALUES ($1, $2, $3, 'pending', $4)
                RETURNING id
                """,
                model_name, model_version, image_tag, workflow_id,
            )

    async def _persist_oci_image(
        self,
        model_name:    str,
        model_version: str,
        build_result:  Any,
        generated:     Any,
        build_job_id:  int,
        workflow_id:   Optional[str],
    ) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                """
                INSERT INTO oci_images (
                    model_name, model_version, image_tag, image_digest, image_uri,
                    registry_host, base_image, labels, build_job_id, workflow_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10)
                ON CONFLICT (image_digest) DO UPDATE
                    SET image_uri=EXCLUDED.image_uri
                RETURNING id
                """,
                model_name,
                model_version,
                build_result.image_tag,
                build_result.image_digest,
                build_result.image_uri,
                settings.image_registry_host,
                generated.base_image,
                json.dumps(generated.oci_labels),
                build_job_id,
                workflow_id,
            )

    async def _update_lineage(
        self,
        model_name:    str,
        model_version: str,
        image_uri:     str,
        image_digest:  str,
    ) -> None:
        """
        Locate the model_version node in the lineage graph and attach the OCI
        image node to it using LineageGraphBuilder.record_oci_image().
        """
        try:
            querier  = LineageGraphQuerier(self._pool)
            graph    = await querier.get_full_lineage(model_name, model_version)
            mv_nodes = [
                n for n in graph["nodes"]
                if n["node_type"] == "model_version"
            ]
            if not mv_nodes:
                logger.warning(
                    "No model_version lineage node found for %s v%s — "
                    "OCI image node will be orphaned.",
                    model_name, model_version,
                )
                return

            parent_node_id = mv_nodes[-1]["id"]
            builder = LineageGraphBuilder(self._pool, model_name, model_version)
            await builder.record_oci_image(
                image_uri=              image_uri,
                digest=                 image_digest,
                parent_version_node_id= parent_node_id,
            )
            logger.info(
                "Lineage updated: model_version node %d → OCI image %s",
                parent_node_id, image_digest,
            )
        except Exception as exc:
            # Lineage update failure is non-fatal — log it but don't block the pipeline
            logger.warning(
                "Lineage update failed for %s v%s (non-fatal): %s",
                model_name, model_version, exc,
            )

    async def _fail_build_job(self, db_id: int, error_message: str) -> None:
        """Mark oci_build_jobs row as failed with error message."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE oci_build_jobs
                SET status='failed', error_message=$1, completed_at=NOW()
                WHERE id=$2
                """,
                error_message[:1000],
                db_id,
            )

    async def _succeed_build_job(
        self, build_job_id: int, log_uri: str, duration_seconds: int
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE oci_build_jobs
                SET status='succeeded', log_uri=$1,
                    duration_seconds=$2, completed_at=NOW()
                WHERE id=$3
                """,
                log_uri, duration_seconds, build_job_id,
            )

    # ── MLflow helpers ────────────────────────────────────────────────────────

    def _get_run_id(self, model_name: str, model_version: str) -> str:
        """Fetch the MLflow run_id associated with this model version."""
        try:
            mv = self._client.get_model_version(model_name, model_version)
            return mv.run_id or ""
        except Exception as exc:
            logger.warning(
                "Could not fetch run_id for %s v%s: %s", model_name, model_version, exc
            )
            return ""

    def _patch_mlflow_tags(
        self,
        model_name:    str,
        model_version: str,
        image_uri:     str,
        image_digest:  str,
        sbom_uri:      str,
        cosign_signed: bool,
    ) -> None:
        """Write OCI build results back as MLflow model version tags."""
        tag_map = {
            OCI_TAG_IMAGE_URI:     image_uri,
            OCI_TAG_IMAGE_DIGEST:  image_digest,
            OCI_TAG_SBOM_URI:      sbom_uri,
            OCI_TAG_COSIGN_SIGNED: "true" if cosign_signed else "false",
        }
        for key, value in tag_map.items():
            try:
                self._client.set_model_version_tag(model_name, model_version, key, value)
            except Exception as exc:
                # Tag write failure is non-fatal — digest is already in Neon
                logger.warning(
                    "Failed to write MLflow tag %s for %s v%s: %s",
                    key, model_name, model_version, exc,
                )

    # ── Tag resolution ─────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_image_tag(model_name: str, model_version: str) -> str:
        """
        Build the full image tag from settings.

        Format: <registry_host>/<namespace>/<safe_model_name>:v<version>

        Model name is sanitised: spaces → hyphens, lowercase, no special chars.
        """
        safe_name = model_name.lower().replace(" ", "-").replace("_", "-")
        return (
            f"{settings.image_registry_host}/"
            f"{settings.image_registry_namespace}/"
            f"{safe_name}:v{model_version}"
        )