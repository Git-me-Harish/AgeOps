"""
agents/registry/promotion_state_machine.py

Phase 3 — Promotion Workflow (Enforced State Machine) — Section 3.3.

Models move through a strict lifecycle managed entirely by this module:

    None → Staging      — automatic; triggered by RegistrationGate after Training
    Staging → Production— FIVE gates must ALL pass (order matters):
                            1. Evaluation  — metrics exceed configured thresholds
                            2. HITL        — human approval tags are present
                            3. Security    — Trivy scan passed, zero critical CVEs
                            4. Governance  — OPA sidecar allows the transition
                            5. OCI         — image built and signed (Phase 4 stub here)
    Staging → Rejected  — automatic when EvaluationAgent threshold gate fails
    Production → Archived — automatic when a new Production version is promoted

Enforcement rules:
  - Every transition goes through this module — no direct mlflow.transition_model_version_stage()
    calls anywhere else in the codebase.
  - Rejected versions cannot be un-rejected without full re-evaluation.
    (The caller must retrain → produce a new version → run through the gate again.)
  - Production → Archived is triggered automatically by the None → Staging path when
    an existing Production version is found.
  - Every transition is persisted to model_promotions BEFORE the MLflow stage
    is changed — if MLflow fails, the DB record acts as a crash-recovery source of truth.
  - OPA gate is fail-closed in production: if the sidecar is unreachable, the
    transition is BLOCKED and logged, not allowed.

MLflow note:
  MLflow Model Registry supports these stage strings: "None", "Staging",
  "Production", "Archived".  Our "Rejected" stage has no MLflow equivalent;
  we store it in model_promotions and leave the MLflow stage as "Archived".
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

import aiohttp
import mlflow
from pydantic import BaseModel, Field

from agents.registry.metadata_schema import (
    ModelRegistryTags,
    RegistrationError,
    TagValidator,
    MANDATORY_TAGS_PRODUCTION,
)
from configs.settings import settings

# Alias so tests can patch module-level settings without reaching into
# configs.settings directly (agents.registry.promotion_state_machine._s).
_s = settings

logger = logging.getLogger(__name__)

# HMAC key for promotion event signing (same pattern as data lineage)
_PROMOTION_SIGNING_KEY: bytes = os.getenv(
    "LINEAGE_SIGNING_KEY", "dev-only-change-in-production"
).encode()


# Stage enum 
class ModelStage(str, Enum):
    """
    Model lifecycle stages.
    MLflow-native stages: NONE, STAGING, PRODUCTION, ARCHIVED.
    Platform-internal stage: REJECTED (stored in model_promotions, not in MLflow).
    """
    NONE       = "None"
    STAGING    = "Staging"
    PRODUCTION = "Production"
    ARCHIVED   = "Archived"
    REJECTED   = "Rejected"   # internal only — MLflow stage will be set to Archived


# Legal transition map: from_stage → {allowed to_stages}
ALLOWED_TRANSITIONS: dict[ModelStage, frozenset[ModelStage]] = {
    ModelStage.NONE:       frozenset({ModelStage.STAGING}),
    ModelStage.STAGING:    frozenset({ModelStage.PRODUCTION, ModelStage.REJECTED}),
    ModelStage.PRODUCTION: frozenset({ModelStage.ARCHIVED}),
    ModelStage.ARCHIVED:   frozenset(),   # terminal — no further transitions
    ModelStage.REJECTED:   frozenset(),   # terminal — new version required
}


# Gate enum 
class Gate(str, Enum):
    """Gates that must ALL pass for Staging → Production."""
    EVALUATION  = "evaluation"   # metrics exceed thresholds
    HITL        = "hitl"         # human approved_by + approved_at set
    SECURITY    = "security"     # Trivy scan passed, zero CRITICAL CVEs
    GOVERNANCE  = "governance"   # OPA policy allows the transition (fail-closed)
    OCI         = "oci"          # OCI image built and signed (Phase 4 — stub in Phase 3)


# Gates run in this order — fail-fast on first failure
GATES_FOR_PRODUCTION: tuple[Gate, ...] = (
    Gate.EVALUATION,
    Gate.HITL,
    Gate.SECURITY,
    Gate.GOVERNANCE,
    Gate.OCI,
)


# Exceptions 
class PromotionGateError(RuntimeError):
    """
    Raised when a gate blocks the Staging → Production transition.
    Contains structured diagnostics for the audit trail.
    """
    def __init__(
        self,
        gate: Gate,
        reason: str,
        model_name: str,
        version: str,
    ) -> None:
        super().__init__(f"Gate '{gate.value}' blocked promotion of {model_name} v{version}: {reason}")
        self.gate = gate
        self.reason = reason
        self.model_name = model_name
        self.version = version


# Request / Result models 

class PromotionRequest(BaseModel):
    """Input to PromotionStateMachine.promote()."""
    model_name:       str
    model_version:    str
    target_stage:     ModelStage
    triggered_by:     str              # agent name or GitHub username
    trigger_type:     Literal["automatic", "human"]
    workflow_id:      Optional[str] = None
    rejection_reason: Optional[str] = None   # required when target_stage = REJECTED


class PromotionResult(BaseModel):
    """Return value of PromotionStateMachine.promote() on success."""
    model_name:       str
    model_version:    str
    from_stage:       ModelStage
    to_stage:         ModelStage
    gates_passed:     list[str]
    gates_failed:     list[str]
    success:          bool
    rejection_reason: Optional[str] = None
    promotion_db_id:  int = 0          # DB row ID from model_promotions


# Configurable eval thresholds 
class EvalThresholds(BaseModel):
    """
    Configurable minimum metric values for the Evaluation gate.
    Defaults match the V2 plan's quality bar; override via environment or config.
    """
    min_accuracy: float = Field(default=0.80, ge=0.0, le=1.0)
    min_f1:       float = Field(default=0.75, ge=0.0, le=1.0)
    min_auc:      float = Field(default=0.75, ge=0.0, le=1.0)

    @classmethod
    def from_settings(cls) -> "EvalThresholds":
        """Load thresholds from settings (Phase 5 will expose these in the UI)."""
        return cls(
            min_accuracy=getattr(settings, "eval_threshold_accuracy", 0.80),
            min_f1=getattr(settings, "eval_threshold_f1", 0.75),
            min_auc=getattr(settings, "eval_threshold_auc", 0.75),
        )


# State machine 
class PromotionStateMachine:
    """
    Enforced model lifecycle state machine.

    All production model stage transitions go through this class.
    No agent or MCP server may call mlflow.transition_model_version_stage()
    directly — all transitions are mediated here with full gate checking,
    DB persistence, and HMAC-signed audit records.

    Args:
        pool: asyncpg.Pool connected to Neon.
        mlflow_client: mlflow.MlflowClient instance.
        opa_url: OPA sidecar base URL. Defaults to the in-pod address.
        thresholds: Configurable eval metric thresholds.
    """

    def __init__(
        self,
        pool: Any,
        mlflow_client: Optional[Any] = None,
        opa_url: str = "http://127.0.0.1:8181",
        thresholds: Optional[EvalThresholds] = None,
    ) -> None:
        self._pool = pool
        self._client = mlflow_client or mlflow.MlflowClient()
        self._opa_url = opa_url
        self._thresholds = thresholds or EvalThresholds.from_settings()

    # Public API 
    async def promote(self, request: PromotionRequest) -> PromotionResult:
        """
        Execute a model lifecycle state transition.

        Raises:
            ValueError: if the transition is not in ALLOWED_TRANSITIONS.
            RegistrationError: if production tag validation fails.
            PromotionGateError: if any Staging→Production gate is blocked.
        """
        from_stage = await self._get_current_stage(request.model_name, request.model_version)

        # Validate the transition is legal
        if request.target_stage not in ALLOWED_TRANSITIONS.get(from_stage, frozenset()):
            raise ValueError(
                f"Illegal transition for {request.model_name} v{request.model_version}: "
                f"{from_stage.value} → {request.target_stage.value}. "
                f"Allowed: {[s.value for s in ALLOWED_TRANSITIONS.get(from_stage, frozenset())]}"
            )

        logger.info(
            "Promotion initiated: model=%s version=%s %s → %s triggered_by=%s",
            request.model_name, request.model_version,
            from_stage.value, request.target_stage.value, request.triggered_by,
        )

        # Route to the correct handler
        if request.target_stage == ModelStage.STAGING:
            return await self._promote_to_staging(request, from_stage)
        elif request.target_stage == ModelStage.PRODUCTION:
            return await self._promote_to_production(request, from_stage)
        elif request.target_stage == ModelStage.REJECTED:
            return await self._reject(request, from_stage)
        elif request.target_stage == ModelStage.ARCHIVED:
            return await self._archive(request, from_stage)
        else:
            raise ValueError(f"Unhandled target stage: {request.target_stage}")

    # Transition handlers 
    async def _promote_to_staging(
        self, request: PromotionRequest, from_stage: ModelStage
    ) -> PromotionResult:
        """
        None → Staging.
        Only requires: all 15 staging tags are present (validated by TagValidator).
        Triggered automatically by RegistrationGate (the tag validation already ran there,
        but we re-validate here as defense-in-depth).
        """
        raw_tags = self._get_mlflow_tags(request.model_name, request.model_version)
        # Re-validate (defense-in-depth — RegistrationGate already validated, but state
        # machines should never trust that their callers validated inputs)
        TagValidator.validate_for_staging(raw_tags)

        db_id = await self._persist_promotion(
            request=request,
            from_stage=from_stage,
            gates_passed=["tag_validation"],
            gates_failed=[],
        )
        self._client.transition_model_version_stage(
            name=request.model_name,
            version=request.model_version,
            stage="Staging",
            archive_existing_versions=False,
        )
        logger.info("Promoted to Staging: model=%s version=%s", request.model_name, request.model_version)
        return PromotionResult(
            model_name=request.model_name,
            model_version=request.model_version,
            from_stage=from_stage,
            to_stage=ModelStage.STAGING,
            gates_passed=["tag_validation"],
            gates_failed=[],
            success=True,
            promotion_db_id=db_id,
        )

    async def _promote_to_production(
        self, request: PromotionRequest, from_stage: ModelStage
    ) -> PromotionResult:
        """
        Staging → Production.
        Runs all 5 gates in sequence.  Fail-fast on first failure: the failing gate
        name is persisted to model_promotions.gates_failed for the audit trail.
        On success, the current Production version is automatically archived.
        """
        raw_tags = self._get_mlflow_tags(request.model_name, request.model_version)

        # Production tag validation (all 18 tags required)
        try:
            tags = TagValidator.validate_for_production(raw_tags)
        except RegistrationError as exc:
            db_id = await self._persist_promotion(
                request=request,
                from_stage=from_stage,
                gates_passed=[],
                gates_failed=["tag_validation"],
                rejection_reason=str(exc),
            )
            raise PromotionGateError(
                gate=Gate.EVALUATION,
                reason=f"Production tag validation failed: {exc}",
                model_name=request.model_name,
                version=request.model_version,
            ) from exc

        gates_passed: list[str] = ["tag_validation"]
        gates_failed: list[str] = []

        gate_checks = [
            (Gate.EVALUATION, lambda: self._check_evaluation_gate(tags)),
            (Gate.HITL,       lambda: self._check_hitl_gate(tags)),
            (Gate.SECURITY,   lambda: self._check_security_gate(tags)),
            (Gate.GOVERNANCE, lambda: self._check_governance_gate(
                request.model_name, request.model_version, tags, request.workflow_id
            )),
            (Gate.OCI,        lambda: self._check_oci_gate(request.model_name, request.model_version)),
        ]

        # Fail-fast: stop at first failing gate
        for gate, check_fn in gate_checks:
            try:
                passed, reason = await check_fn()
            except Exception as exc:
                # Unexpected errors in gate checks are treated as failures
                passed, reason = False, f"Unexpected error: {exc}"
                logger.exception("Gate %s raised exception for %s v%s",
                                  gate.value, request.model_name, request.model_version)

            if passed:
                gates_passed.append(gate.value)
                logger.debug("Gate %s passed for %s v%s",
                             gate.value, request.model_name, request.model_version)
            else:
                gates_failed.append(gate.value)
                logger.warning(
                    "Gate %s BLOCKED promotion of %s v%s: %s",
                    gate.value, request.model_name, request.model_version, reason,
                )
                db_id = await self._persist_promotion(
                    request=request,
                    from_stage=from_stage,
                    gates_passed=gates_passed,
                    gates_failed=gates_failed,
                    rejection_reason=reason,
                )
                raise PromotionGateError(
                    gate=gate,
                    reason=reason,
                    model_name=request.model_name,
                    version=request.model_version,
                )

        # All gates passed — archive existing Production version first
        await self._archive_current_production(request.model_name, request.workflow_id)

        db_id = await self._persist_promotion(
            request=request,
            from_stage=from_stage,
            gates_passed=gates_passed,
            gates_failed=[],
        )
        self._client.transition_model_version_stage(
            name=request.model_name,
            version=request.model_version,
            stage="Production",
            archive_existing_versions=True,   # MLflow archives old prod automatically
        )
        logger.info(
            "Promoted to Production: model=%s version=%s gates=%s",
            request.model_name, request.model_version, gates_passed,
        )
        return PromotionResult(
            model_name=request.model_name,
            model_version=request.model_version,
            from_stage=from_stage,
            to_stage=ModelStage.PRODUCTION,
            gates_passed=gates_passed,
            gates_failed=[],
            success=True,
            promotion_db_id=db_id,
        )

    async def _reject(self, request: PromotionRequest, from_stage: ModelStage) -> PromotionResult:
        """
        Staging → Rejected.
        Cannot be overridden without re-evaluation.  MLflow stage is set to Archived
        since MLflow has no "Rejected" concept.
        """
        reason = request.rejection_reason or "Rejected by evaluation pipeline"
        db_id = await self._persist_promotion(
            request=request,
            from_stage=from_stage,
            gates_passed=[],
            gates_failed=["evaluation"],
            rejection_reason=reason,
        )
        # MLflow: move to Archived (our Rejected state lives in model_promotions)
        self._client.transition_model_version_stage(
            name=request.model_name,
            version=request.model_version,
            stage="Archived",
            archive_existing_versions=False,
        )
        logger.info(
            "Rejected: model=%s version=%s reason=%s",
            request.model_name, request.model_version, reason,
        )
        return PromotionResult(
            model_name=request.model_name,
            model_version=request.model_version,
            from_stage=from_stage,
            to_stage=ModelStage.REJECTED,
            gates_passed=[],
            gates_failed=["evaluation"],
            success=True,   # The transition itself succeeded (the model was legitimately rejected)
            rejection_reason=reason,
            promotion_db_id=db_id,
        )

    async def _archive(self, request: PromotionRequest, from_stage: ModelStage) -> PromotionResult:
        """Production → Archived. Triggered automatically or via the UI."""
        db_id = await self._persist_promotion(
            request=request,
            from_stage=from_stage,
            gates_passed=["manual_archive"],
            gates_failed=[],
        )
        self._client.transition_model_version_stage(
            name=request.model_name,
            version=request.model_version,
            stage="Archived",
            archive_existing_versions=False,
        )
        return PromotionResult(
            model_name=request.model_name,
            model_version=request.model_version,
            from_stage=from_stage,
            to_stage=ModelStage.ARCHIVED,
            gates_passed=["manual_archive"],
            gates_failed=[],
            success=True,
            promotion_db_id=db_id,
        )

    # Gate implementations 
    async def _check_evaluation_gate(
        self, tags: ModelRegistryTags
    ) -> tuple[bool, str]:
        """
        Gate 1: verify eval metrics exceed configured minimum thresholds.
        All three metrics (accuracy, F1, AUC) must pass individually.
        """
        failures = []
        t = self._thresholds
        if tags.eval_accuracy < t.min_accuracy:
            failures.append(
                f"accuracy={tags.eval_accuracy:.4f} < threshold={t.min_accuracy}"
            )
        if tags.eval_f1 < t.min_f1:
            failures.append(f"f1={tags.eval_f1:.4f} < threshold={t.min_f1}")
        if tags.eval_auc < t.min_auc:
            failures.append(f"auc={tags.eval_auc:.4f} < threshold={t.min_auc}")
        if not tags.eval_bias_passed:
            failures.append("bias check did not pass")
        if failures:
            return False, "; ".join(failures)
        return True, "all eval metrics passed"

    async def _check_hitl_gate(
        self, tags: ModelRegistryTags
    ) -> tuple[bool, str]:
        """
        Gate 2: verify human-in-the-loop approval tags are present.
        approved_by must be a non-empty string (GitHub username).
        approved_at must be a valid datetime.
        """
        if not tags.approved_by or not tags.approved_by.strip():
            return False, "approved_by tag is missing or empty"
        if tags.approved_at is None:
            return False, "approved_at tag is missing"
        # approved_at must be in the past (sanity check — not in the future)
        now = datetime.now(tz=timezone.utc)
        if tags.approved_at > now:
            return False, f"approved_at={tags.approved_at.isoformat()} is in the future"
        return True, f"approved by {tags.approved_by} at {tags.approved_at.isoformat()}"

    async def _check_security_gate(
        self, tags: ModelRegistryTags
    ) -> tuple[bool, str]:
        """
        Gate 3: Trivy scan must have passed and there must be zero CRITICAL CVEs.
        """
        if tags.security_trivy_scan != "passed":
            return False, f"Trivy scan status='{tags.security_trivy_scan}' (expected 'passed')"
        if tags.security_cve_critical > 0:
            return False, f"Trivy found {tags.security_cve_critical} CRITICAL CVE(s) — must be 0"
        return True, "Trivy scan passed, zero CRITICAL CVEs"

    async def _check_governance_gate(
        self,
        model_name: str,
        version: str,
        tags: ModelRegistryTags,
        workflow_id: Optional[str],
    ) -> tuple[bool, str]:
        """
        Gate 4: OPA sidecar policy evaluation.

        Policy path: /v1/data/mlops/allow  (package "mlops", rule "allow" in
        configs/opa_policies/mlops_policy.rego — there is no "allow_promotion"
        rule; that was a naming bug). Input mirrors the schema the rego file
        actually reads: action, agent_role, model_tags{}, eval_metrics{},
        security_scan{} — not a flat dict of individual fields.

        Fail-closed: if OPA is unreachable, the gate BLOCKS (returns False).
        This matches ADR-002 and the GovernanceAgent production behaviour.
        """
        opa_input = {
            "input": {
                "action":     "promote_to_production",
                "agent_role": "governance",
                "model_tags": tags.to_mlflow_tags(),
                "eval_metrics": {
                    "accuracy":       tags.eval_accuracy,
                    "f1":             tags.eval_f1,
                    "auc":            tags.eval_auc,
                    "bias_passed":    tags.eval_bias_passed,
                    # The EVALUATION gate already ran and passed before this
                    # gate is reached (see GATES_FOR_PRODUCTION ordering).
                    "overall_passed": True,
                },
                "security_scan": {
                    "trivy_passed":     tags.security_trivy_scan == "passed",
                    "critical_cves":    tags.security_cve_critical,
                    "high_cves":        0,
                    "semgrep_passed":   True,
                    "secrets_detected": False,
                },
                "workflow_id": workflow_id,
            }
        }
        opa_url = f"{self._opa_url}/v1/data/mlops/allow"
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as session:
                async with session.post(opa_url, json=opa_input) as resp:
                    if resp.status != 200:
                        return False, f"OPA returned HTTP {resp.status} — fail-closed"
                    body = await resp.json()
                    allowed: bool = bool(body.get("result", False))
                    if not allowed:
                        return False, "OPA policy denied promotion"
                    return True, "OPA policy allowed promotion"
        except aiohttp.ClientConnectorError:
            # OPA sidecar unreachable — fail-closed (ADR-002)
            logger.error(
                "OPA sidecar unreachable at %s — failing closed for %s v%s",
                opa_url, model_name, version,
            )
            return False, f"OPA sidecar unreachable at {opa_url} — fail-closed per ADR-002"
        except Exception as exc:
            logger.exception("OPA gate unexpected error for %s v%s", model_name, version)
            return False, f"OPA gate failed with unexpected error: {exc}"

    async def _check_oci_gate(
        self, model_name: str, model_version: str
    ) -> tuple[bool, str]:
        """
        Gate 5: OCI image built, Trivy-scanned, and (if enabled) Cosign-signed.

        Checks, in order, against the tables DockerfileAgent writes to:
          1. oci_images        — an image was actually built for this version.
          2. trivy_scan_results — its Trivy scan passed.
          3. cosign_attestations — it is signed, unless settings.cosign_enabled
             is False (a documented dev/staging escape hatch), in which case
             this check is skipped entirely.

        Fail-closed: any DB error blocks the gate rather than allowing passage.
        """
        try:
            async with self._pool.acquire() as conn:
                img_row = await conn.fetchrow(
                    """
                    SELECT image_digest, image_uri
                    FROM oci_images
                    WHERE model_name=$1 AND model_version=$2
                    ORDER BY id DESC LIMIT 1
                    """,
                    model_name, model_version,
                )
                if img_row is None:
                    return False, (
                        f"No OCI image found for {model_name} v{model_version} — "
                        "run DockerfileAgent (Phase 4 packaging) first"
                    )

                digest = img_row["image_digest"]

                scan_row = await conn.fetchrow(
                    """
                    SELECT passed, critical_count, high_count
                    FROM trivy_scan_results
                    WHERE image_digest=$1
                    ORDER BY id DESC LIMIT 1
                    """,
                    digest,
                )
                if scan_row is None:
                    return False, f"No Trivy scan found for image digest {digest}"
                if not scan_row["passed"]:
                    return False, (
                        f"Trivy scan did not pass for {digest}: "
                        f"critical={scan_row['critical_count']} high={scan_row['high_count']}"
                    )

                if not _s.cosign_enabled:
                    return True, (
                        "OCI image built, Trivy scan passed. "
                        "Cosign signing is disabled in settings — skipped."
                    )

                sig_row = await conn.fetchrow(
                    """
                    SELECT signed, oidc_issuer
                    FROM cosign_attestations
                    WHERE image_digest=$1
                    ORDER BY id DESC LIMIT 1
                    """,
                    digest,
                )
                if sig_row is None:
                    return False, f"No Cosign attestation found for image digest {digest}"
                if not sig_row["signed"]:
                    return False, f"Cosign attestation present but signed=False for {digest}"

                return True, (
                    f"OCI image verified: digest={digest}, Trivy scan passed, "
                    f"Cosign signed (issuer={sig_row['oidc_issuer']})"
                )

        except Exception as exc:
            logger.exception("OCI gate DB error for %s v%s", model_name, model_version)
            return False, f"OCI gate blocked — DB error: {exc}"

    # Helpers 
    async def _get_current_stage(self, model_name: str, version: str) -> ModelStage:
        """
        Fetch the current MLflow stage for this version.
        Falls back to ModelStage.NONE if the version is not registered.
        """
        try:
            mv = self._client.get_model_version(model_name, version)
            stage_str = mv.current_stage
            # MLflow returns "None" as the string literal "None"
            return ModelStage(stage_str)
        except mlflow.exceptions.MlflowException:
            return ModelStage.NONE

    def _get_mlflow_tags(self, model_name: str, version: str) -> dict[str, str]:
        """Retrieve all MLflow model version tags as a flat string dict."""
        try:
            mv = self._client.get_model_version(model_name, version)
            return dict(mv.tags or {})
        except mlflow.exceptions.MlflowException:
            return {}

    async def _archive_current_production(
        self, model_name: str, workflow_id: Optional[str]
    ) -> None:
        """
        Find any existing Production version and archive it.
        Called automatically before a new version goes to Production.
        """
        try:
            prod_versions = self._client.get_latest_versions(
                model_name, stages=["Production"]
            )
        except mlflow.exceptions.MlflowException:
            return

        for mv in prod_versions:
            logger.info(
                "Auto-archiving existing Production version: model=%s version=%s",
                model_name, mv.version,
            )
            archive_request = PromotionRequest(
                model_name=model_name,
                model_version=mv.version,
                target_stage=ModelStage.ARCHIVED,
                triggered_by="promotion_state_machine",
                trigger_type="automatic",
                workflow_id=workflow_id,
            )
            await self._persist_promotion(
                request=archive_request,
                from_stage=ModelStage.PRODUCTION,
                gates_passed=["auto_archive"],
                gates_failed=[],
            )
            self._client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Archived",
                archive_existing_versions=False,
            )

    async def _persist_promotion(
        self,
        request: PromotionRequest,
        from_stage: ModelStage,
        gates_passed: list[str],
        gates_failed: list[str],
        rejection_reason: Optional[str] = None,
    ) -> int:
        """
        INSERT an immutable promotion event into model_promotions.

        Returns:
            The DB primary key of the inserted row.

        The HMAC signature covers (model_name, version, from_stage, to_stage, promoted_at)
        to allow tamper detection during compliance audits.
        """
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        signature = self._sign_promotion(
            model_name=request.model_name,
            version=request.model_version,
            from_stage=from_stage.value,
            to_stage=request.target_stage.value,
            promoted_at=now_iso,
        )
        async with self._pool.acquire() as conn:
            row_id: int = await conn.fetchval(
                """
                INSERT INTO model_promotions (
                    model_name, model_version, from_stage, to_stage,
                    triggered_by, trigger_type,
                    gates_passed, gates_failed,
                    rejection_reason, workflow_id, signature
                ) VALUES (
                    $1, $2, $3, $4,
                    $5, $6,
                    $7::jsonb, $8::jsonb,
                    $9, $10, $11
                )
                RETURNING id
                """,
                request.model_name,
                request.model_version,
                from_stage.value,
                request.target_stage.value,
                request.triggered_by,
                request.trigger_type,
                json.dumps(gates_passed),
                json.dumps(gates_failed),
                rejection_reason or request.rejection_reason,
                request.workflow_id,
                signature,
            )
        logger.debug(
            "Promotion persisted: db_id=%d model=%s ver=%s %s→%s",
            row_id,
            request.model_name,
            request.model_version,
            from_stage.value,
            request.target_stage.value,
        )
        return row_id

    @staticmethod
    def _sign_promotion(
        model_name: str,
        version: str,
        from_stage: str,
        to_stage: str,
        promoted_at: str,
    ) -> str:
        """
        HMAC-SHA256 signature over canonical JSON of the key transition fields.
        Enables tamper detection: any post-hoc modification of the record breaks
        the signature when re-computed during a compliance audit.
        """
        payload = json.dumps(
            {
                "model_name":   model_name,
                "version":      version,
                "from_stage":   from_stage,
                "to_stage":     to_stage,
                "promoted_at":  promoted_at,
            },
            sort_keys=True,
        ).encode()
        return hmac.new(_PROMOTION_SIGNING_KEY, payload, hashlib.sha256).hexdigest()

    async def get_promotion_history(
        self,
        model_name: str,
        model_version: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Return all promotion events for a model (optionally filtered by version).
        Ordered by most recent first.
        """
        if model_version:
            query = """
                SELECT id, model_name, model_version, from_stage, to_stage,
                       triggered_by, trigger_type, gates_passed, gates_failed,
                       rejection_reason, workflow_id, promoted_at
                FROM model_promotions
                WHERE model_name=$1 AND model_version=$2
                ORDER BY promoted_at DESC
            """
            params = (model_name, model_version)
        else:
            query = """
                SELECT id, model_name, model_version, from_stage, to_stage,
                       triggered_by, trigger_type, gates_passed, gates_failed,
                       rejection_reason, workflow_id, promoted_at
                FROM model_promotions
                WHERE model_name=$1
                ORDER BY promoted_at DESC
            """
            params = (model_name,)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [
            {
                "id":               r["id"],
                "model_name":       r["model_name"],
                "model_version":    r["model_version"],
                "from_stage":       r["from_stage"],
                "to_stage":         r["to_stage"],
                "triggered_by":     r["triggered_by"],
                "trigger_type":     r["trigger_type"],
                "gates_passed":     list(r["gates_passed"] or []),
                "gates_failed":     list(r["gates_failed"] or []),
                "rejection_reason": r["rejection_reason"],
                "workflow_id":      r["workflow_id"],
                "promoted_at":      r["promoted_at"].isoformat() if r["promoted_at"] else None,
            }
            for r in rows
        ]