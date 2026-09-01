"""
agents/registry/model_comparator.py

Phase 3 — Model Comparison View (Section 3.4).

Fetches metadata for 2–4 model versions from MLflow Registry + Neon and produces
a structured comparison matrix.  The comparison result is persisted to the
model_comparisons table so the UI can render it without re-querying MLflow.

Comparison surface (mirrors the plan's Section 3.4 spec):
  ├ Metric comparison       accuracy, F1, AUC (from model_registry_tags)
  ├ Training data comparison dataset_hash, dataset_row_count
  ├ Framework info          framework, framework_version, python_version
  ├ Security posture        trivy_scan status, critical CVE count
  ├ Git provenance          commit SHA, repo URL
  └ Recommendation          version with highest F1 (primary ranking metric)

Design decisions:
  - Baseline is always versions[0] — all diffs are computed relative to it.
  - MetricDiff.is_improvement uses "higher is better" for accuracy/F1/AUC and
    "lower is better" for cve_critical.  Other fields are informational only.
  - ModelComparator.compare() is idempotent per (model_name, versions) tuple —
    calling it twice for the same inputs produces a new DB row each time
    (comparison sessions are ephemeral, not cached).
  - The DB row stores a JSON snapshot so the UI never re-queries MLflow for
    historical comparisons.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import mlflow

logger = logging.getLogger(__name__)


# Typed result models 
@dataclass(frozen=True)
class MetricDiff:
    """
    Pairwise delta for a single metric between a baseline and challenger version.

    Attributes:
        metric:              Metric name (e.g. "eval_accuracy").
        baseline_version:   Version string used as the reference point.
        challenger_version: Version string being compared against the baseline.
        baseline_value:     Metric value for the baseline version.
        challenger_value:   Metric value for the challenger version.
        delta:              challenger_value − baseline_value.
        delta_pct:          (delta / |baseline_value|) × 100, or None if baseline is 0.
        is_improvement:     True if the delta moves in the "better" direction.
    """
    metric:             str
    baseline_version:   str
    challenger_version: str
    baseline_value:     float
    challenger_value:   float
    delta:              float
    delta_pct:          Optional[float]
    is_improvement:     bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric":             self.metric,
            "baseline_version":   self.baseline_version,
            "challenger_version": self.challenger_version,
            "baseline_value":     self.baseline_value,
            "challenger_value":   self.challenger_value,
            "delta":              round(self.delta, 6),
            "delta_pct":          round(self.delta_pct, 2) if self.delta_pct is not None else None,
            "is_improvement":     self.is_improvement,
        }


@dataclass
class VersionComparison:
    """
    Metadata snapshot for one model version in a comparison session.
    Sourced from model_registry_tags (Neon) with MLflow run metrics as fallback.
    """
    model_name:         str
    version:            str
    framework:          str
    framework_version:  str
    python_version:     str
    dataset_hash:       str
    dataset_row_count:  int
    eval_accuracy:      float
    eval_f1:            float
    eval_auc:           float
    eval_bias_passed:   bool
    security_trivy_scan:str
    security_cve_critical: int
    git_commit:         str
    git_repo:           str
    current_stage:      str           # MLflow stage string
    training_run_id:    Optional[str] = None
    extra_metrics:      dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name":          self.model_name,
            "version":             self.version,
            "framework":           self.framework,
            "framework_version":   self.framework_version,
            "python_version":      self.python_version,
            "dataset_hash":        self.dataset_hash,
            "dataset_row_count":   self.dataset_row_count,
            "eval_accuracy":       self.eval_accuracy,
            "eval_f1":             self.eval_f1,
            "eval_auc":            self.eval_auc,
            "eval_bias_passed":    self.eval_bias_passed,
            "security_trivy_scan": self.security_trivy_scan,
            "security_cve_critical": self.security_cve_critical,
            "git_commit":          self.git_commit,
            "git_repo":            self.git_repo,
            "current_stage":       self.current_stage,
            "training_run_id":     self.training_run_id,
            "extra_metrics":       self.extra_metrics,
        }


@dataclass
class ModelComparison:
    """
    Full comparison result for a set of model versions.

    Attributes:
        comparison_id:       UUID generated at comparison time (DB row key).
        model_name:          MLflow registered model name.
        versions:            Ordered list of VersionComparison objects.
        metric_diffs:        Pairwise diffs (challenger vs. versions[0] baseline).
        recommended_version: Version with highest F1 across all compared versions.
        decision:            Optional free-text note set after human review.
    """
    comparison_id:       str
    model_name:          str
    versions:            list[VersionComparison]
    metric_diffs:        list[MetricDiff]
    recommended_version: str
    decision:            Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id":       self.comparison_id,
            "model_name":          self.model_name,
            "versions":            [v.to_dict() for v in self.versions],
            "metric_diffs":        [d.to_dict() for d in self.metric_diffs],
            "recommended_version": self.recommended_version,
            "decision":            self.decision,
        }


# Comparator 
# Metrics for which "higher is better" (delta > 0 = improvement)
_HIGHER_IS_BETTER: frozenset[str] = frozenset({
    "eval_accuracy", "eval_f1", "eval_auc",
})
# Metrics for which "lower is better" (delta < 0 = improvement)
_LOWER_IS_BETTER: frozenset[str] = frozenset({
    "security_cve_critical",
})

# Numeric fields extracted for pairwise diff computation
_NUMERIC_METRIC_FIELDS: tuple[str, ...] = (
    "eval_accuracy",
    "eval_f1",
    "eval_auc",
    "security_cve_critical",
    "dataset_row_count",
)


class ModelComparator:
    """
    Loads metadata for N model versions and produces a structured comparison.

    Args:
        pool:          asyncpg.Pool connected to Neon.
        mlflow_client: mlflow.MlflowClient instance (injected for testability).
    """

    def __init__(self, pool: Any, mlflow_client: Optional[Any] = None) -> None:
        self._pool = pool
        self._client = mlflow_client or mlflow.MlflowClient()

    async def compare(
        self,
        model_name: str,
        versions: list[str],
        initiated_by: str = "system",
        workflow_id: Optional[str] = None,
    ) -> ModelComparison:
        """
        Build a ModelComparison for the given list of version strings.

        The first version in the list is treated as the baseline — all MetricDiff
        objects express delta relative to it.

        Args:
            model_name:   MLflow registered model name.
            versions:     2–4 version strings to compare.  First = baseline.
            initiated_by: GitHub username or agent name that initiated the comparison.
            workflow_id:  Optional FK to the triggering workflow.

        Returns:
            A ModelComparison with the full comparison matrix and recommendation.

        Raises:
            ValueError: if fewer than 2 versions are provided.
            LookupError: if a version has no metadata in model_registry_tags.
        """
        if len(versions) < 2:
            raise ValueError(
                f"At least 2 versions required for comparison, got {len(versions)}"
            )
        if len(versions) > 4:
            raise ValueError(
                f"Maximum 4 versions per comparison, got {len(versions)}"
            )

        # Load metadata for each version
        loaded: list[VersionComparison] = []
        for v in versions:
            vc = await self._load_version(model_name, v)
            loaded.append(vc)

        # Compute pairwise diffs (each challenger vs. baseline = loaded[0])
        baseline = loaded[0]
        diffs: list[MetricDiff] = []
        for challenger in loaded[1:]:
            diffs.extend(self._compute_diffs(baseline, challenger))

        # Recommendation: version with highest F1
        recommended = max(loaded, key=lambda v: v.eval_f1).version

        comparison_id = str(uuid.uuid4())
        comparison = ModelComparison(
            comparison_id=comparison_id,
            model_name=model_name,
            versions=loaded,
            metric_diffs=diffs,
            recommended_version=recommended,
        )

        # Persist to model_comparisons table
        await self._persist(
            comparison=comparison,
            initiated_by=initiated_by,
            workflow_id=workflow_id,
        )

        logger.info(
            "Comparison complete: model=%s versions=%s recommended=%s comparison_id=%s",
            model_name, versions, recommended, comparison_id,
        )
        return comparison

    # Internal helpers 
    async def _load_version(self, model_name: str, version: str) -> VersionComparison:
        """
        Load metadata for one version.

        Primary source: model_registry_tags (Neon) — strongly typed and always
        present for versions that went through RegistrationGate.

        Fallback: MLflow model version tags — used for versions registered outside
        this system (legacy or manually created).
        """
        # Try Neon first (typed, authoritative)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    dataset_hash, dataset_row_count,
                    framework, framework_version, python_version,
                    eval_accuracy, eval_f1, eval_auc, eval_bias_passed,
                    security_trivy_scan, security_cve_critical,
                    git_commit, git_repo
                FROM model_registry_tags
                WHERE model_name=$1 AND model_version=$2
                """,
                model_name, version,
            )

        # Fetch current MLflow stage (always from MLflow — it's the authoritative source)
        try:
            mv = self._client.get_model_version(model_name, version)
            current_stage = mv.current_stage
            training_run_id = mv.run_id
            mlflow_tags = dict(mv.tags or {})
        except Exception:
            current_stage = "None"
            training_run_id = None
            mlflow_tags = {}

        if row is not None:
            # Happy path: full typed data from Neon
            return VersionComparison(
                model_name=model_name,
                version=version,
                framework=row["framework"],
                framework_version=row["framework_version"],
                python_version=row["python_version"],
                dataset_hash=row["dataset_hash"],
                dataset_row_count=row["dataset_row_count"],
                eval_accuracy=float(row["eval_accuracy"]),
                eval_f1=float(row["eval_f1"]),
                eval_auc=float(row["eval_auc"]),
                eval_bias_passed=bool(row["eval_bias_passed"]),
                security_trivy_scan=row["security_trivy_scan"],
                security_cve_critical=int(row["security_cve_critical"]),
                git_commit=row["git_commit"],
                git_repo=row["git_repo"],
                current_stage=current_stage,
                training_run_id=training_run_id,
            )

        # Fallback: reconstruct from MLflow tags (best-effort, may be partial)
        logger.warning(
            "No model_registry_tags row for %s v%s — falling back to MLflow tags",
            model_name, version,
        )
        if not mlflow_tags:
            raise LookupError(
                f"No metadata found for {model_name} v{version} — "
                "not in model_registry_tags and no MLflow tags available"
            )

        def _float_tag(key: str, default: float = 0.0) -> float:
            return float(mlflow_tags.get(key, default))

        return VersionComparison(
            model_name=model_name,
            version=version,
            framework=mlflow_tags.get("mlops.framework", "unknown"),
            framework_version=mlflow_tags.get("mlops.framework.version", "unknown"),
            python_version=mlflow_tags.get("mlops.python.version", "unknown"),
            dataset_hash=mlflow_tags.get("mlops.dataset.hash", ""),
            dataset_row_count=int(mlflow_tags.get("mlops.dataset.row_count", 0)),
            eval_accuracy=_float_tag("mlops.eval.accuracy"),
            eval_f1=_float_tag("mlops.eval.f1"),
            eval_auc=_float_tag("mlops.eval.auc"),
            eval_bias_passed=mlflow_tags.get("mlops.eval.bias_passed", "false").lower() == "true",
            security_trivy_scan=mlflow_tags.get("mlops.security.trivy_scan", "unknown"),
            security_cve_critical=int(mlflow_tags.get("mlops.security.cve_critical", 0)),
            git_commit=mlflow_tags.get("mlops.git.commit", ""),
            git_repo=mlflow_tags.get("mlops.git.repo", ""),
            current_stage=current_stage,
            training_run_id=training_run_id,
        )

    @staticmethod
    def _compute_diffs(
        baseline: VersionComparison,
        challenger: VersionComparison,
    ) -> list[MetricDiff]:
        """
        Compute MetricDiff for every numeric field between baseline and challenger.
        """
        diffs: list[MetricDiff] = []
        baseline_dict = baseline.to_dict()
        challenger_dict = challenger.to_dict()

        for metric in _NUMERIC_METRIC_FIELDS:
            b_val = float(baseline_dict.get(metric, 0.0))
            c_val = float(challenger_dict.get(metric, 0.0))
            delta = c_val - b_val
            delta_pct = (delta / abs(b_val) * 100.0) if b_val != 0.0 else None

            if metric in _HIGHER_IS_BETTER:
                is_improvement = delta > 0
            elif metric in _LOWER_IS_BETTER:
                is_improvement = delta < 0
            else:
                # Informational fields (dataset_row_count) — no polarity
                is_improvement = False

            diffs.append(MetricDiff(
                metric=metric,
                baseline_version=baseline.version,
                challenger_version=challenger.version,
                baseline_value=b_val,
                challenger_value=c_val,
                delta=round(delta, 6),
                delta_pct=delta_pct,
                is_improvement=is_improvement,
            ))

        return diffs

    async def _persist(
        self,
        comparison: ModelComparison,
        initiated_by: str,
        workflow_id: Optional[str],
    ) -> None:
        """
        INSERT a comparison session into model_comparisons.
        The full comparison_result dict is stored as JSONB for the UI to query.
        """
        model_versions_json = json.dumps([
            {"model_name": v.model_name, "version": v.version}
            for v in comparison.versions
        ])
        result_json = json.dumps(comparison.to_dict())

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_comparisons
                    (comparison_id, model_versions, comparison_result, initiated_by, workflow_id)
                VALUES ($1, $2::jsonb, $3::jsonb, $4, $5)
                """,
                comparison.comparison_id,
                model_versions_json,
                result_json,
                initiated_by,
                workflow_id,
            )

    async def get_comparison(self, comparison_id: str) -> Optional[dict[str, Any]]:
        """
        Retrieve a saved comparison session from Neon by its UUID.
        Returns None if not found.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT comparison_id, model_versions, comparison_result,
                       initiated_by, decision, created_at
                FROM model_comparisons
                WHERE comparison_id=$1
                """,
                comparison_id,
            )
        if row is None:
            return None
        return {
            "comparison_id":     row["comparison_id"],
            "model_versions":    list(row["model_versions"] or []),
            "comparison_result": dict(row["comparison_result"] or {}),
            "initiated_by":      row["initiated_by"],
            "decision":          row["decision"],
            "created_at":        row["created_at"].isoformat() if row["created_at"] else None,
        }

    async def set_decision(self, comparison_id: str, decision: str) -> bool:
        """
        Write a human decision note to an existing comparison session.
        Called from the UI after the engineer reviews the comparison matrix.

        Returns:
            True if the row was found and updated; False if not found.
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE model_comparisons SET decision=$1 WHERE comparison_id=$2",
                decision, comparison_id,
            )
        # asyncpg returns "UPDATE N" — N=1 means found and updated
        return result.endswith("1")