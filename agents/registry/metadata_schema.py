"""
agents/registry/metadata_schema.py

Phase 3 — Structured Model Metadata Schema.

This module owns the 18 mandatory tag definitions from the V2 plan (Section 3.1)
and enforces them at two checkpoints:

  Checkpoint A — None → Staging (15 tags, enforced by RegistrationGate):
    All dataset, framework, evaluation, security, and git tags must be present
    and valid before MLflow model registration is allowed.

  Checkpoint B — Staging → Production (all 18, enforced by PromotionStateMachine):
    The HITL approval tags (approved_by, approved_at) and the governance audit ID
    must be set before any Staging → Production transition proceeds.

Design decisions:
  - Validation is fail-hard: RegistrationError is raised immediately on the first
    missing or invalid tag. The pipeline does not proceed with partial metadata.
  - Tags are stored in MLflow as strings (MLflow tag API stores strings only);
    this module handles the parsing and type coercion.
  - Tag keys follow the dotted namespace convention: mlops.<section>.<field>
  - The Pydantic model (ModelRegistryTags) is the canonical typed representation;
    raw MLflow tag dicts are always converted through TagValidator before use.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import mlflow
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Tag key constants 
# These are the literal strings used as MLflow tag keys.

_T_DATASET_URI          = "mlops.dataset.uri"
_T_DATASET_HASH         = "mlops.dataset.hash"
_T_DATASET_ROW_COUNT    = "mlops.dataset.row_count"
_T_FRAMEWORK            = "mlops.framework"
_T_FRAMEWORK_VERSION    = "mlops.framework.version"
_T_PYTHON_VERSION       = "mlops.python.version"
_T_EVAL_ACCURACY        = "mlops.eval.accuracy"
_T_EVAL_F1              = "mlops.eval.f1"
_T_EVAL_AUC             = "mlops.eval.auc"
_T_EVAL_HOLDOUT_HASH    = "mlops.eval.holdout_hash"
_T_EVAL_BIAS_PASSED     = "mlops.eval.bias_passed"
_T_SECURITY_TRIVY_SCAN  = "mlops.security.trivy_scan"
_T_SECURITY_CVE_CRITICAL= "mlops.security.cve_critical"
_T_GIT_COMMIT           = "mlops.git.commit"
_T_GIT_REPO             = "mlops.git.repo"
_T_APPROVED_BY          = "mlops.approved_by"
_T_APPROVED_AT          = "mlops.approved_at"
_T_GOVERNANCE_AUDIT_ID  = "mlops.governance.audit_id"

# Tags required at Staging registration (gates 1–15)
MANDATORY_TAGS_STAGING: frozenset[str] = frozenset({
    _T_DATASET_URI,
    _T_DATASET_HASH,
    _T_DATASET_ROW_COUNT,
    _T_FRAMEWORK,
    _T_FRAMEWORK_VERSION,
    _T_PYTHON_VERSION,
    _T_EVAL_ACCURACY,
    _T_EVAL_F1,
    _T_EVAL_AUC,
    _T_EVAL_HOLDOUT_HASH,
    _T_EVAL_BIAS_PASSED,
    _T_SECURITY_TRIVY_SCAN,
    _T_SECURITY_CVE_CRITICAL,
    _T_GIT_COMMIT,
    _T_GIT_REPO,
})

# Tags required at Production promotion (all 18)
MANDATORY_TAGS_PRODUCTION: frozenset[str] = MANDATORY_TAGS_STAGING | frozenset({
    _T_APPROVED_BY,
    _T_APPROVED_AT,
    _T_GOVERNANCE_AUDIT_ID,
})

# Valid framework identifiers
VALID_FRAMEWORKS: frozenset[str] = frozenset({
    "xgboost", "pytorch", "sklearn", "huggingface", "custom",
})


# Exceptions 
class RegistrationError(ValueError):
    """
    Raised when mandatory tags are missing or invalid.
    Callers should treat this as a hard pipeline block — not a warning.

    Attributes:
        missing_tags: Tag keys that were absent from the input.
        invalid_tags: {tag_key: reason} for tags that were present but invalid.
    """

    def __init__(
        self,
        message: str,
        missing_tags: Optional[list[str]] = None,
        invalid_tags: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.missing_tags: list[str] = missing_tags or []
        self.invalid_tags: dict[str, str] = invalid_tags or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": str(self),
            "missing_tags": self.missing_tags,
            "invalid_tags": self.invalid_tags,
        }


# Typed tag model 
class ModelRegistryTags(BaseModel):
    """
    Typed representation of the 18 mandatory model registry tags.

    All fields map 1:1 to the tag keys in MANDATORY_TAGS_STAGING /
    MANDATORY_TAGS_PRODUCTION.  Use TagValidator.from_mlflow_tags() to
    construct from a raw MLflow tag dict (where all values are strings).
    """

    model_config = {"frozen": True}   # immutable after construction

    # data provenance 
    dataset_uri: str = Field(..., min_length=1)
    dataset_hash: str = Field(..., min_length=64, max_length=64)   # SHA-256 hex
    dataset_row_count: int = Field(..., ge=1)

    # framework 
    framework: Literal["xgboost", "pytorch", "sklearn", "huggingface", "custom"]
    framework_version: str = Field(..., min_length=1)
    python_version: str = Field(..., min_length=1)   # e.g. "3.11.6"

    # evaluation 
    eval_accuracy: float = Field(..., ge=0.0, le=1.0)
    eval_f1: float = Field(..., ge=0.0, le=1.0)
    eval_auc: float = Field(..., ge=0.0, le=1.0)
    eval_holdout_hash: str = Field(..., min_length=64, max_length=64)
    eval_bias_passed: bool

    # security 
    security_trivy_scan: Literal["passed", "failed"]
    security_cve_critical: int = Field(..., ge=0)

    # git provenance 
    git_commit: str = Field(..., min_length=40, max_length=40)   # full 40-char SHA
    git_repo: str = Field(..., min_length=1)

    # HITL approval (nullable until Staging → Production) 
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    # governance 
    governance_audit_id: Optional[str] = None   # SHA-256 audit trail signature

    @field_validator("dataset_hash", "eval_holdout_hash", mode="before")
    @classmethod
    def _validate_sha256_hex(cls, v: str) -> str:
        """Ensure the value is a 64-character lowercase hex string."""
        if not isinstance(v, str):
            raise ValueError("expected a string")
        v = v.lower().strip()
        if len(v) != 64:
            raise ValueError(f"SHA-256 hash must be 64 hex chars, got {len(v)}")
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError("SHA-256 hash must contain only hex characters")
        return v

    @field_validator("git_commit", mode="before")
    @classmethod
    def _validate_git_sha(cls, v: str) -> str:
        """Ensure the value is a 40-character lowercase hex string."""
        if not isinstance(v, str):
            raise ValueError("expected a string")
        v = v.lower().strip()
        if len(v) != 40:
            raise ValueError(f"Git SHA must be 40 hex chars, got {len(v)}")
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError("Git SHA must contain only hex characters")
        return v

    def to_mlflow_tags(self) -> dict[str, str]:
        """
        Serialize to the flat string dict expected by the MLflow tag API.
        All values are coerced to strings as MLflow requires.
        """
        tags: dict[str, str] = {
            _T_DATASET_URI:          self.dataset_uri,
            _T_DATASET_HASH:         self.dataset_hash,
            _T_DATASET_ROW_COUNT:    str(self.dataset_row_count),
            _T_FRAMEWORK:            self.framework,
            _T_FRAMEWORK_VERSION:    self.framework_version,
            _T_PYTHON_VERSION:       self.python_version,
            _T_EVAL_ACCURACY:        str(self.eval_accuracy),
            _T_EVAL_F1:              str(self.eval_f1),
            _T_EVAL_AUC:             str(self.eval_auc),
            _T_EVAL_HOLDOUT_HASH:    self.eval_holdout_hash,
            _T_EVAL_BIAS_PASSED:     "true" if self.eval_bias_passed else "false",
            _T_SECURITY_TRIVY_SCAN:  self.security_trivy_scan,
            _T_SECURITY_CVE_CRITICAL:str(self.security_cve_critical),
            _T_GIT_COMMIT:           self.git_commit,
            _T_GIT_REPO:             self.git_repo,
        }
        if self.approved_by is not None:
            tags[_T_APPROVED_BY] = self.approved_by
        if self.approved_at is not None:
            tags[_T_APPROVED_AT] = self.approved_at.isoformat()
        if self.governance_audit_id is not None:
            tags[_T_GOVERNANCE_AUDIT_ID] = self.governance_audit_id
        return tags

    def to_db_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict for the model_registry_tags table INSERT."""
        return {
            "dataset_uri":           self.dataset_uri,
            "dataset_hash":          self.dataset_hash,
            "dataset_row_count":     self.dataset_row_count,
            "framework":             self.framework,
            "framework_version":     self.framework_version,
            "python_version":        self.python_version,
            "eval_accuracy":         self.eval_accuracy,
            "eval_f1":               self.eval_f1,
            "eval_auc":              self.eval_auc,
            "eval_holdout_hash":     self.eval_holdout_hash,
            "eval_bias_passed":      self.eval_bias_passed,
            "security_trivy_scan":   self.security_trivy_scan,
            "security_cve_critical": self.security_cve_critical,
            "git_commit":            self.git_commit,
            "git_repo":              self.git_repo,
            "approved_by":           self.approved_by,
            "approved_at":           self.approved_at,
            "governance_audit_id":   self.governance_audit_id,
        }


# Tag validator 
class TagValidator:
    """
    Parses and validates raw MLflow tag dicts into typed ModelRegistryTags.

    MLflow stores all tags as strings; this class handles coercion and raises
    RegistrationError with precise diagnostics on any validation failure.

    Usage:
        tags = TagValidator.validate_for_staging(raw_mlflow_tags)
        # → ModelRegistryTags or raises RegistrationError
    """

    @staticmethod
    def validate_for_staging(raw: dict[str, str]) -> ModelRegistryTags:
        """
        Validate that all 15 staging-required tags are present and well-formed.

        Raises:
            RegistrationError: if any mandatory tag is missing or invalid.
        """
        return TagValidator._validate(raw, MANDATORY_TAGS_STAGING, "staging")

    @staticmethod
    def validate_for_production(raw: dict[str, str]) -> ModelRegistryTags:
        """
        Validate all 18 production-required tags (staging set + HITL + governance).

        Raises:
            RegistrationError: if any mandatory tag is missing or invalid.
        """
        return TagValidator._validate(raw, MANDATORY_TAGS_PRODUCTION, "production")

    @staticmethod
    def _validate(
        raw: dict[str, str],
        required: frozenset[str],
        checkpoint: str,
    ) -> ModelRegistryTags:
        # Step 1: check for missing keys
        missing = sorted(required - set(raw.keys()))
        if missing:
            raise RegistrationError(
                f"Registration blocked at {checkpoint}: "
                f"{len(missing)} mandatory tag(s) missing: {missing}",
                missing_tags=missing,
            )

        # Step 2: coerce string values to typed Python objects.
        #
        # _safe() takes both:
        #  field_name — the Pydantic model field name (used as the key in `parsed`)
        #  tag_key    — the full MLflow tag key, e.g. "mlops.dataset.row_count"
        #               (used as the key in `invalid` so callers see the exact tag
        #                they need to fix, not an opaque internal field name)
        invalid: dict[str, str] = {}
        parsed: dict[str, Any] = {}

        def _safe(field_name: str, tag_key: str, coerce, value: str) -> None:
            try:
                parsed[field_name] = coerce(value)
            except (ValueError, TypeError) as exc:
                invalid[tag_key] = str(exc)

        _safe("dataset_uri",          _T_DATASET_URI,           str,   raw[_T_DATASET_URI])
        _safe("dataset_hash",         _T_DATASET_HASH,          str,   raw[_T_DATASET_HASH])
        _safe("dataset_row_count",    _T_DATASET_ROW_COUNT,     int,   raw[_T_DATASET_ROW_COUNT])
        _safe("framework",            _T_FRAMEWORK,             str,   raw[_T_FRAMEWORK].lower().strip())
        _safe("framework_version",    _T_FRAMEWORK_VERSION,     str,   raw[_T_FRAMEWORK_VERSION])
        _safe("python_version",       _T_PYTHON_VERSION,        str,   raw[_T_PYTHON_VERSION])
        _safe("eval_accuracy",        _T_EVAL_ACCURACY,         float, raw[_T_EVAL_ACCURACY])
        _safe("eval_f1",              _T_EVAL_F1,               float, raw[_T_EVAL_F1])
        _safe("eval_auc",             _T_EVAL_AUC,              float, raw[_T_EVAL_AUC])
        _safe("eval_holdout_hash",    _T_EVAL_HOLDOUT_HASH,     str,   raw[_T_EVAL_HOLDOUT_HASH])
        _safe("eval_bias_passed",     _T_EVAL_BIAS_PASSED,      lambda v: v.lower() == "true", raw[_T_EVAL_BIAS_PASSED])
        _safe("security_trivy_scan",  _T_SECURITY_TRIVY_SCAN,   str,   raw[_T_SECURITY_TRIVY_SCAN].lower().strip())
        _safe("security_cve_critical",_T_SECURITY_CVE_CRITICAL, int,   raw[_T_SECURITY_CVE_CRITICAL])
        _safe("git_commit",           _T_GIT_COMMIT,            str,   raw[_T_GIT_COMMIT])
        _safe("git_repo",             _T_GIT_REPO,              str,   raw[_T_GIT_REPO])

        # Production-only tags
        if _T_APPROVED_BY in raw:
            _safe("approved_by", _T_APPROVED_BY, str, raw[_T_APPROVED_BY])
        if _T_APPROVED_AT in raw:
            def _parse_dt(v: str) -> datetime:
                return datetime.fromisoformat(v)
            _safe("approved_at", _T_APPROVED_AT, _parse_dt, raw[_T_APPROVED_AT])
        if _T_GOVERNANCE_AUDIT_ID in raw:
            _safe("governance_audit_id", _T_GOVERNANCE_AUDIT_ID, str, raw[_T_GOVERNANCE_AUDIT_ID])

        if invalid:
            raise RegistrationError(
                f"Registration blocked at {checkpoint}: "
                f"{len(invalid)} tag(s) failed type validation: {list(invalid)}",
                invalid_tags=invalid,
            )

        # Step 3: Pydantic validation (field constraints, Literal checks, etc.)
        try:
            return ModelRegistryTags(**parsed)
        except Exception as exc:
            raise RegistrationError(
                f"Registration blocked at {checkpoint}: Pydantic validation failed: {exc}",
                invalid_tags={"__pydantic__": str(exc)},
            ) from exc


# Registration gate 
class RegistrationGate:
    """
    Enforces tag completeness before MLflow model registration is allowed.

    Called by TrainingAgent after EvaluationAgent approval, before calling
    mlflow.register_model().  This is the mandatory checkpoint between
    model training and the MLflow Model Registry.

    Flow:
        1. Validate all 15 staging-required tags (fail-hard on error)
        2. Register model in MLflow Registry → receive version string
        3. Apply all tags to the MLflow model version via the client API
        4. Persist tags to model_registry_tags table (Neon)
        5. Log registration event to agent_audit_trail (tamper-evident)

    Args:
        pool: asyncpg.Pool connected to Neon.
        mlflow_client: mlflow.MlflowClient instance (injected for testability).
    """

    def __init__(self, pool: Any, mlflow_client: Optional[Any] = None) -> None:
        self._pool = pool
        self._client = mlflow_client or mlflow.MlflowClient()

    async def register(
        self,
        run_id: str,
        model_name: str,
        raw_tags: dict[str, str],
        artifact_path: str = "model",
        workflow_id: Optional[str] = None,
    ) -> tuple[str, ModelRegistryTags]:
        """
        Validate tags, register the model, and persist all metadata.

        Returns:
            (version_string, ModelRegistryTags) on success.

        Raises:
            RegistrationError: if tags are missing or invalid (pipeline blocked).
            mlflow.MlflowException: if MLflow registration fails.
        """
        # Gate: all 15 staging tags must be present and valid 
        tags = TagValidator.validate_for_staging(raw_tags)
        logger.info(
            "Tag validation passed for run=%s model=%s framework=%s accuracy=%.4f",
            run_id, model_name, tags.framework, tags.eval_accuracy,
        )

        # Register in MLflow Registry 
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mv = self._client.create_registered_model(model_name) if not self._model_exists(model_name) else None  # noqa
        registered = mlflow.register_model(model_uri, model_name)
        version = registered.version
        logger.info("MLflow registration complete: model=%s version=%s", model_name, version)

        # Apply tags to MLflow model version 
        mlflow_tag_dict = tags.to_mlflow_tags()
        for key, value in mlflow_tag_dict.items():
            self._client.set_model_version_tag(model_name, version, key, value)

        # Persist to Neon 
        await self._persist_tags(model_name, version, tags, raw_tags)

        # Audit trail entry 
        await self._log_audit(
            workflow_id=workflow_id,
            agent_id="registration_gate",
            action="model_registered",
            payload={
                "model_name":  model_name,
                "version":     version,
                "run_id":      run_id,
                "framework":   tags.framework,
                "eval_f1":     tags.eval_f1,
                "eval_accuracy": tags.eval_accuracy,
            },
        )

        return version, tags

    def _model_exists(self, model_name: str) -> bool:
        """Check if the registered model name already exists in MLflow."""
        try:
            self._client.get_registered_model(model_name)
            return True
        except mlflow.exceptions.MlflowException:
            return False

    async def _persist_tags(
        self,
        model_name: str,
        version: str,
        tags: ModelRegistryTags,
        raw_tags: dict[str, str],
    ) -> None:
        """
        INSERT into model_registry_tags.
        ON CONFLICT (model_name, model_version) DO UPDATE — idempotent if called twice.
        """
        db = tags.to_db_dict()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_registry_tags (
                    model_name, model_version,
                    dataset_uri, dataset_hash, dataset_row_count,
                    framework, framework_version, python_version,
                    eval_accuracy, eval_f1, eval_auc,
                    eval_holdout_hash, eval_bias_passed,
                    security_trivy_scan, security_cve_critical,
                    git_commit, git_repo,
                    approved_by, approved_at, governance_audit_id,
                    raw_tags
                ) VALUES (
                    $1,  $2,
                    $3,  $4,  $5,
                    $6,  $7,  $8,
                    $9,  $10, $11,
                    $12, $13,
                    $14, $15,
                    $16, $17,
                    $18, $19, $20,
                    $21
                )
                ON CONFLICT (model_name, model_version) DO UPDATE
                    SET raw_tags   = EXCLUDED.raw_tags,
                        updated_at = NOW()
                """,
                model_name, version,
                db["dataset_uri"], db["dataset_hash"], db["dataset_row_count"],
                db["framework"], db["framework_version"], db["python_version"],
                db["eval_accuracy"], db["eval_f1"], db["eval_auc"],
                db["eval_holdout_hash"], db["eval_bias_passed"],
                db["security_trivy_scan"], db["security_cve_critical"],
                db["git_commit"], db["git_repo"],
                db["approved_by"], db["approved_at"], db["governance_audit_id"],
                json.dumps(raw_tags),
            )

    async def _log_audit(
        self,
        workflow_id: Optional[str],
        agent_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        """Write a tamper-evident audit entry to agent_audit_trail."""
        sig = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_audit_trail (workflow_id, agent_id, action, payload, signature)
                VALUES ($1, $2, $3, $4, $5)
                """,
                workflow_id, agent_id, action, json.dumps(payload), sig,
            )