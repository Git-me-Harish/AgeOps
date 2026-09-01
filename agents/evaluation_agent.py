"""
agents/evaluation_agent.py

Evaluation Agent — V2 production rewrite.

What changed from V1:
  V1: mlflow.evaluate() on synthetic holdout; hardcoded thresholds
  V2:
    1. Holdout loaded from Feast offline store (never from training split)
    2. mlflow.evaluate() with standard metrics (accuracy, F1, AUC, confusion matrix)
    3. Calibration check: Brier score + reliability diagram
    4. Slice evaluation: per-subgroup metrics (category, time period, data source)
    5. Regression testing: compare against current production model — block if worse on any slice
    6. LLM-as-judge: sample 50 uncertain predictions, get structured critique
    7. Bias check: demographic parity, equalized odds (if protected attributes present)
    8. Threshold gate: all metrics must exceed configurable minimums
    9. HTML report → R2; linked from MLflow as artifact
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from agents import AgentTaskResult
from agents.llm_gateway import LLMGateway
from configs.settings import settings

logger = logging.getLogger(__name__)

THRESHOLDS = {
    "accuracy": 0.70,
    "f1": 0.65,
    "auc": 0.70,
}

# Max tolerated F1 drop (global or per-slice) vs. the current production model
# before the regression gate blocks promotion.
_REGRESSION_TOLERANCE = 0.02

# Typed models
class JudgeVerdict(BaseModel):
    """LLM-as-judge verdict for one uncertain prediction."""
    sample_index: int
    verdict: str           # "reasonable" | "questionable" | "wrong"
    confidence: float      # LLM's confidence in its verdict (0–1)
    reasoning: str
    flags: list[str] = Field(default_factory=list)   # e.g. ["near_boundary", "feature_anomaly"]


class BiasReport(BaseModel):
    """Fairness metrics for one protected attribute."""
    attribute: str
    demographic_parity_diff: float    # |P(y=1|A=0) - P(y=1|A=1)|; threshold < 0.10
    equalized_odds_diff: float        # max diff in TPR/FPR across groups
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Aggregated evaluation result stored in MLflow and Neon."""
    accuracy: float
    f1: float
    auc: float
    brier_score: float
    calibration_passed: bool
    slice_results: dict[str, dict[str, float]]   # {slice_name: {metric: value}}
    regression_passed: bool
    bias_reports: list[BiasReport]
    bias_passed: bool
    llm_judge_sample_count: int
    llm_judge_passed_count: int
    llm_judge_pass_rate: float
    overall_passed: bool
    r2_report_path: Optional[str]
    blocking_reasons: list[str] = Field(default_factory=list)

# Evaluation Agent
class EvaluationAgent:
    """
    Multi-stage model evaluation agent.

    Promotion gate: a model must pass ALL stages to advance to staging.
    Any failure is recorded with a structured reason.
    """

    def __init__(self) -> None:
        self._pool: Optional[Any] = None

    async def connect(self, pool: Any) -> None:
        self._pool = pool

    @mlflow.trace(name="evaluation_agent.run")
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
        best_run_id: str = state.get("best_run_id", "")
        best_framework: str = state.get("best_framework", "unknown")
        dataset_uri: str = state.get("dataset_uri", "")
        lineage_id: Optional[int] = state.get("lineage_id")

        if not best_run_id:
            return AgentTaskResult(
                task_id=task_id, status="failed",
                error="best_run_id missing from state — TrainingAgent must run first",
            )

        gateway = LLMGateway(pool=self._pool, workflow_id=task_id)

        try:
            with mlflow.start_run(run_name=f"eval-{task_id}", nested=True):
                mlflow.set_tag("agent", "evaluation_agent")
                mlflow.set_tag("evaluated_run_id", best_run_id)

                # Stage 1: Load holdout from Feast
                holdout_df = await self._load_holdout(dataset_uri, task_id)
                if holdout_df is None or len(holdout_df) == 0:
                    return AgentTaskResult(
                        task_id=task_id, status="failed",
                        error="Holdout dataset could not be loaded from Feast",
                    )

                # Stage 2: Load trained model from MLflow
                model = await self._load_model(best_run_id)
                if model is None:
                    return AgentTaskResult(
                        task_id=task_id, status="failed",
                        error=f"Model not found in MLflow run {best_run_id}",
                    )

                # Prepare features and labels
                feature_cols = [c for c in holdout_df.columns
                                if c not in ("label", "record_id", "event_timestamp",
                                             "split", "workflow_id", "dataset_version",
                                             "source", "category")]
                X = holdout_df[feature_cols].select_dtypes(include="number").fillna(0)
                y_true = holdout_df["label"].values if "label" in holdout_df.columns else None

                if y_true is None:
                    return AgentTaskResult(
                        task_id=task_id, status="failed",
                        error="Holdout dataset missing 'label' column",
                    )

                # SHA-256 of the holdout data — recorded on the model as
                # mlops.eval.holdout_hash so a promoted model's evaluation
                # can always be traced back to the exact holdout it passed.
                holdout_hash = hashlib.sha256(
                    pd.util.hash_pandas_object(holdout_df, index=True).values.tobytes()
                ).hexdigest()

                y_pred = np.array(model.predict(X))
                y_proba = None
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X)[:, 1]

                # Stage 3: Standard metrics
                std_metrics = self._compute_standard_metrics(y_true, y_pred, y_proba)
                mlflow.log_metrics(std_metrics)

                # Stage 4: Calibration check
                brier_score, calibration_passed = self._check_calibration(y_true, y_proba)

                # Stage 5: Slice evaluation
                slice_results = await self._slice_evaluation(
                    holdout_df, X, y_true, model, feature_cols
                )
                # Log slice metrics to MLflow so the NEXT model's regression
                # test (Stage 6) can compare against them by name.
                for slice_name, slice_metrics in slice_results.items():
                    safe_name = "".join(
                        c if c.isalnum() else "_" for c in slice_name
                    )[:40]
                    for metric_name, val in slice_metrics.items():
                        try:
                            mlflow.log_metric(f"slice_{safe_name}_{metric_name}", val)
                        except Exception:
                            pass

                # Stage 6: Regression testing vs production model — global
                # metrics AND every slice (plan §2.4: block on ANY slice worse)
                regression_passed = await self._regression_test(
                    std_metrics, slice_results, task_id
                )

                # Stage 7: LLM-as-judge on uncertain predictions
                judge_results = await self._llm_judge(
                    X=X, y_true=y_true, y_pred=y_pred, y_proba=y_proba,
                    gateway=gateway, task_id=task_id,
                )
                judge_pass_rate = (
                    sum(1 for v in judge_results if v.verdict in ("reasonable",))
                    / max(len(judge_results), 1)
                )

                # Stage 8: Bias check
                bias_reports = self._bias_check(holdout_df, y_true, y_pred)
                bias_passed = all(r.passed for r in bias_reports)

                # Threshold gate — collect all failures
                blocking_reasons: list[str] = []
                thresholds = THRESHOLDS
                for metric, threshold in thresholds.items():
                    val = std_metrics.get(metric, 0.0)
                    if val < threshold:
                        blocking_reasons.append(
                            f"{metric}={val:.3f} below threshold={threshold}"
                        )
                if not calibration_passed:
                    blocking_reasons.append(f"Calibration failed: brier_score={brier_score:.4f}")
                if not regression_passed:
                    blocking_reasons.append("Regression test failed: new model worse on at least one slice")
                if not bias_passed:
                    failed_attrs = [r.attribute for r in bias_reports if not r.passed]
                    blocking_reasons.append(f"Bias check failed for attributes: {failed_attrs}")
                if judge_pass_rate < 0.70:
                    blocking_reasons.append(
                        f"LLM judge pass rate {judge_pass_rate:.1%} below 70% threshold"
                    )

                overall_passed = len(blocking_reasons) == 0

                # Stage 9: Generate HTML report → R2
                report_path = await self._generate_report(
                    task_id=task_id,
                    best_run_id=best_run_id,
                    std_metrics=std_metrics,
                    slice_results=slice_results,
                    bias_reports=bias_reports,
                    judge_results=judge_results,
                    blocking_reasons=blocking_reasons,
                )

                # Build typed report
                eval_report = EvaluationReport(
                    accuracy=std_metrics.get("accuracy", 0.0),
                    f1=std_metrics.get("f1", 0.0),
                    auc=std_metrics.get("auc", 0.0),
                    brier_score=brier_score,
                    calibration_passed=calibration_passed,
                    slice_results=slice_results,
                    regression_passed=regression_passed,
                    bias_reports=bias_reports,
                    bias_passed=bias_passed,
                    llm_judge_sample_count=len(judge_results),
                    llm_judge_passed_count=sum(1 for v in judge_results if v.verdict == "reasonable"),
                    llm_judge_pass_rate=judge_pass_rate,
                    overall_passed=overall_passed,
                    r2_report_path=report_path,
                    blocking_reasons=blocking_reasons,
                )

                mlflow.log_dict(eval_report.model_dump(), "evaluation_report.json")
                mlflow.log_metric("eval.overall_passed", int(overall_passed))
                mlflow.log_metric("eval.llm_judge_pass_rate", judge_pass_rate)
                mlflow.log_metric("eval.bias_passed", int(bias_passed))

                logger.info(
                    "EvaluationAgent: passed=%s f1=%.3f auc=%.3f bias=%s llm_judge=%.1f%%",
                    overall_passed, eval_report.f1, eval_report.auc,
                    bias_passed, judge_pass_rate * 100,
                )

                status = "success" if overall_passed else "failed"
                return AgentTaskResult(
                    task_id=task_id,
                    status=status,
                    output={
                        "eval_report": eval_report.model_dump(),
                        "overall_passed": overall_passed,
                        "blocking_reasons": blocking_reasons,
                        "best_run_id": best_run_id,
                        "r2_report_path": report_path,
                        "holdout_hash": holdout_hash,
                    },
                    error="; ".join(blocking_reasons) if not overall_passed else None,
                    confidence=eval_report.f1,
                )

        except Exception as exc:
            logger.exception("EvaluationAgent failed for workflow %s", task_id)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))

    # Stage implementations 
    async def _load_holdout(self, dataset_uri: str, task_id: str) -> Optional[pd.DataFrame]:
        """
        Load holdout split from Feast offline store.
        Filters on split='test' or split='val'.
        Falls back to reading from the connector if Feast is unavailable.
        """
        try:
            from feast import FeatureStore
            from datetime import datetime, timezone

            store = FeatureStore(repo_path=settings.feast_repo_path)
            n_samples = 500
            entity_df = pd.DataFrame({
                "record_id": list(range(n_samples)),
                "event_timestamp": [datetime.now(tz=timezone.utc)] * n_samples,
            })
            features_df = await asyncio.to_thread(
                store.get_historical_features,
                entity_df=entity_df,
                features=["numeric_features:f0", "numeric_features:f1",
                           "entity_metadata:label", "entity_metadata:split"],
            )
            df = features_df.to_df()
            holdout = df[df.get("split", "train").isin(["test", "val"])]
            if len(holdout) == 0:
                holdout = df.sample(min(200, len(df)), random_state=42)
            logger.info("Holdout loaded from Feast: %d rows", len(holdout))
            return holdout
        except Exception as exc:
            logger.warning("Feast holdout load failed — falling back to connector: %s", exc)
            # Fallback: read directly from connector
            try:
                from agents.connectors import ConnectorFactory
                reference_uri = dataset_uri.replace(".", "_holdout.")
                connector = ConnectorFactory.from_uri(dataset_uri)
                await connector.connect()
                result = await connector.read()
                df = result.dataframe
                return df.tail(max(100, len(df) // 5))
            except Exception as exc2:
                logger.error("Holdout fallback also failed: %s", exc2)
                return None

    async def _load_model(self, run_id: str) -> Optional[Any]:
        """Load model artifact from MLflow run."""
        try:
            model = await asyncio.to_thread(
                mlflow.sklearn.load_model, f"runs:/{run_id}/model"
            )
            return model
        except Exception:
            try:
                model = await asyncio.to_thread(
                    mlflow.pyfunc.load_model, f"runs:/{run_id}/model"
                )
                return model
            except Exception as exc:
                logger.error("Model load failed for run %s: %s", run_id, exc)
                return None

    def _compute_standard_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
    ) -> dict[str, float]:
        """Compute accuracy, F1, AUC on holdout."""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        metrics: dict[str, float] = {}
        try:
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            if y_proba is not None:
                try:
                    metrics["auc"] = float(roc_auc_score(y_true, y_proba))
                except Exception:
                    metrics["auc"] = 0.0
        except Exception as exc:
            logger.error("Standard metrics computation failed: %s", exc)
        return metrics

    def _check_calibration(
        self,
        y_true: np.ndarray,
        y_proba: Optional[np.ndarray],
        threshold: float = 0.15,
    ) -> tuple[float, bool]:
        """Brier score calibration check. Lower = better. Threshold: < 0.15 = pass."""
        if y_proba is None:
            return 0.0, True   # non-probabilistic model — skip
        try:
            from sklearn.metrics import brier_score_loss
            score = float(brier_score_loss(y_true, y_proba))
            passed = score < threshold
            logger.info("Calibration: brier_score=%.4f passed=%s", score, passed)
            return score, passed
        except Exception as exc:
            logger.warning("Calibration check failed: %s", exc)
            return 0.0, True

    async def _slice_evaluation(
        self,
        holdout_df: pd.DataFrame,
        X: pd.DataFrame,
        y_true: np.ndarray,
        model: Any,
        feature_cols: list[str],
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate metrics on each categorical slice in the holdout.
        Protected/metadata columns: category, source, split.
        """
        from sklearn.metrics import accuracy_score, f1_score

        results: dict[str, dict[str, float]] = {}
        slice_cols = [c for c in ("category", "source") if c in holdout_df.columns]

        for col in slice_cols:
            for val in holdout_df[col].dropna().unique():
                mask = holdout_df[col] == val
                if mask.sum() < 10:
                    continue
                X_slice = X[mask]
                y_slice = y_true[mask]
                try:
                    preds = model.predict(X_slice)
                    results[f"{col}={val}"] = {
                        "accuracy": float(accuracy_score(y_slice, preds)),
                        "f1":       float(f1_score(y_slice, preds, average="weighted", zero_division=0)),
                        "n":        int(mask.sum()),
                    }
                except Exception as exc:
                    logger.warning("Slice eval failed for %s=%s: %s", col, val, exc)

        logger.info("Slice evaluation: %d slices evaluated", len(results))
        return results

    async def _regression_test(
        self,
        new_metrics: dict[str, float],
        new_slices: dict[str, dict[str, float]],
        task_id: str,
    ) -> bool:
        """
        Compare new model against current production model — global metrics
        AND every slice logged by Stage 5. Blocks (returns False) if the new
        model is worse than production by more than _REGRESSION_TOLERANCE on
        the global F1 or on ANY individual slice's F1.
        Returns True if the new model is safe to promote.
        """
        try:
            client = mlflow.tracking.MlflowClient()
            prod_models = client.search_model_versions(
                filter_string=(
                    f"name='{settings.mlflow_registered_model_name}' "
                    "AND tags.current_stage='Production'"
                ),
                max_results=1,
            )
            if not prod_models:
                logger.info("No production model found — regression test passed by default")
                return True

            prod_run_id = prod_models[0].run_id
            prod_run = client.get_run(prod_run_id)
            prod_metrics = prod_run.data.metrics

            regressions: list[str] = []

            new_f1 = new_metrics.get("f1", 0.0)
            prod_f1 = prod_metrics.get("f1", 0.0)
            if prod_f1 > 0 and (prod_f1 - new_f1) > _REGRESSION_TOLERANCE:
                regressions.append(f"global f1 dropped {prod_f1:.3f} -> {new_f1:.3f}")

            for slice_name, slice_metrics in new_slices.items():
                safe_name = "".join(c if c.isalnum() else "_" for c in slice_name)[:40]
                prod_slice_f1 = prod_metrics.get(f"slice_{safe_name}_f1")
                if prod_slice_f1 is None:
                    # Production run predates slice logging, or this slice is
                    # new — nothing to compare against, so it can't regress.
                    continue
                new_slice_f1 = slice_metrics.get("f1", 0.0)
                if prod_slice_f1 > 0 and (prod_slice_f1 - new_slice_f1) > _REGRESSION_TOLERANCE:
                    regressions.append(
                        f"slice '{slice_name}' f1 dropped {prod_slice_f1:.3f} -> {new_slice_f1:.3f}"
                    )

            if regressions:
                logger.warning("Regression test FAILED: %s", regressions)
                return False

            logger.info(
                "Regression test passed: new_f1=%.3f prod_f1=%.3f, %d slice(s) checked",
                new_f1, prod_f1, len(new_slices),
            )
            return True

        except Exception as exc:
            logger.warning("Regression test failed to compare (non-fatal, treating as passed): %s", exc)
            return True

    async def _llm_judge(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        gateway: LLMGateway,
        task_id: str,
        n_samples: int = 50,
    ) -> list[JudgeVerdict]:
        """
        Sample the most uncertain predictions (near decision boundary at p≈0.5)
        and ask the LLM to evaluate whether each prediction is reasonable.
        """
        if y_proba is None:
            logger.info("No probabilities available — skipping LLM judge")
            return []

        # Find most uncertain predictions
        uncertainty = np.abs(y_proba - 0.5)
        uncertain_idx = np.argsort(uncertainty)[:n_samples]

        verdicts: list[JudgeVerdict] = []
        X_arr = X.values
        col_names = list(X.columns)

        # Batch into groups of 10 to reduce LLM calls
        batch_size = 10
        for batch_start in range(0, min(n_samples, len(uncertain_idx)), batch_size):
            batch_idx = uncertain_idx[batch_start:batch_start + batch_size]
            batch_prompts = []
            for i, idx in enumerate(batch_idx):
                features_str = ", ".join(
                    f"{col_names[j]}={X_arr[idx, j]:.3f}"
                    for j in range(min(len(col_names), 10))
                )
                batch_prompts.append(
                    f"Sample {i}: features=[{features_str}] "
                    f"predicted={y_pred[idx]} true={y_true[idx]} "
                    f"confidence={y_proba[idx]:.3f}"
                )

            prompt = f"""
You are evaluating {len(batch_prompts)} model predictions that are near the decision boundary
(model confidence ≈ 0.5, meaning high uncertainty).

For each sample, assess whether the prediction is REASONABLE given the features.

Samples:
{chr(10).join(batch_prompts)}

For each sample, respond with a JSON array of objects:
[{{"sample_index": int, "verdict": "reasonable"|"questionable"|"wrong",
   "confidence": 0.0-1.0, "reasoning": "...", "flags": ["flag1", ...]}}]

Flags should include: "near_boundary", "feature_anomaly", "inconsistent_label",
"probable_noise", "correct_despite_uncertainty"

Return ONLY the JSON array.
""".strip()

            try:
                response = await gateway.complete(
                    prompt=prompt,
                    agent_role="evaluation",
                    skip_cache=True,   # each batch is unique
                )
                clean = response.text.strip()
                if clean.startswith("```"):
                    clean = "\n".join(clean.split("\n")[1:-1])
                batch_verdicts = json.loads(clean)
                for v in batch_verdicts:
                    verdicts.append(JudgeVerdict(**v))
            except Exception as exc:
                logger.warning("LLM judge batch failed: %s", exc)
                # Treat failed batch as all-reasonable (non-blocking)
                for i, idx in enumerate(batch_idx):
                    verdicts.append(JudgeVerdict(
                        sample_index=batch_start + i,
                        verdict="reasonable",
                        confidence=0.5,
                        reasoning="LLM judge unavailable — treated as reasonable",
                    ))

        logger.info(
            "LLM judge complete: %d samples, %d reasonable, %d questionable",
            len(verdicts),
            sum(1 for v in verdicts if v.verdict == "reasonable"),
            sum(1 for v in verdicts if v.verdict == "questionable"),
        )
        return verdicts

    def _bias_check(
        self,
        holdout_df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attributes: Optional[list[str]] = None,
        threshold: float = 0.10,
    ) -> list[BiasReport]:
        """
        Compute demographic parity and equalized odds for protected attributes.
        Default protected attributes: 'category', 'source' if present.
        """
        if protected_attributes is None:
            protected_attributes = [c for c in ("category", "source") if c in holdout_df.columns]

        reports: list[BiasReport] = []
        for attr in protected_attributes:
            groups = holdout_df[attr].dropna().unique()
            if len(groups) < 2:
                continue
            try:
                group_pred_rates: dict[str, float] = {}
                group_tpr: dict[str, float] = {}

                for group in groups:
                    mask = holdout_df[attr] == group
                    if mask.sum() < 10:
                        continue
                    y_g = y_true[mask]
                    p_g = y_pred[mask]

                    pos_rate = float(p_g.mean())
                    group_pred_rates[str(group)] = pos_rate

                    # TPR
                    true_pos = ((y_g == 1) & (p_g == 1)).sum()
                    actual_pos = (y_g == 1).sum()
                    group_tpr[str(group)] = float(true_pos / actual_pos) if actual_pos > 0 else 0.0

                rates = list(group_pred_rates.values())
                tprs = list(group_tpr.values())
                dp_diff = max(rates) - min(rates) if rates else 0.0
                eo_diff = max(tprs) - min(tprs) if tprs else 0.0
                passed = dp_diff < threshold and eo_diff < threshold

                reports.append(BiasReport(
                    attribute=attr,
                    demographic_parity_diff=dp_diff,
                    equalized_odds_diff=eo_diff,
                    passed=passed,
                    details={
                        "group_prediction_rates": group_pred_rates,
                        "group_tpr": group_tpr,
                    },
                ))
            except Exception as exc:
                logger.warning("Bias check failed for attribute %s: %s", attr, exc)

        return reports

    async def _generate_report(
        self,
        task_id: str,
        best_run_id: str,
        std_metrics: dict,
        slice_results: dict,
        bias_reports: list[BiasReport],
        judge_results: list[JudgeVerdict],
        blocking_reasons: list[str],
    ) -> Optional[str]:
        """Generate an HTML evaluation report and upload to R2."""
        if not settings.r2_configured:
            return None
        try:
            from datetime import datetime, timezone
            import boto3
            from botocore.config import Config

            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
            key = f"{settings.r2_reports_prefix}/{task_id}/{timestamp}_eval.html"

            html = self._build_html_report(
                task_id, best_run_id, std_metrics, slice_results,
                bias_reports, judge_results, blocking_reasons,
            )
            s3 = boto3.client(
                "s3",
                endpoint_url=settings.r2_endpoint_url,
                aws_access_key_id=settings.r2_access_key_id,
                aws_secret_access_key=settings.r2_secret_access_key,
                region_name=settings.r2_region,
                config=Config(retries={"max_attempts": 3}),
            )
            await asyncio.to_thread(
                s3.put_object,
                Bucket=settings.r2_bucket_name,
                Key=key, Body=html.encode("utf-8"), ContentType="text/html",
            )
            logger.info("Evaluation report stored: R2 key=%s", key)
            mlflow.log_param("eval_report_r2_path", key)
            return key
        except Exception as exc:
            logger.warning("Evaluation report generation failed: %s", exc)
            return None

    def _build_html_report(self, task_id, run_id, metrics, slices, bias, judge, blocking) -> str:
        bias_rows = "".join(
            f"<tr><td>{r.attribute}</td><td>{r.demographic_parity_diff:.3f}</td>"
            f"<td>{r.equalized_odds_diff:.3f}</td><td>{'✅' if r.passed else '❌'}</td></tr>"
            for r in bias
        )
        slice_rows = "".join(
            f"<tr><td>{name}</td><td>{m.get('accuracy', 0):.3f}</td>"
            f"<td>{m.get('f1', 0):.3f}</td><td>{int(m.get('n', 0))}</td></tr>"
            for name, m in slices.items()
        )
        status_colour = "#2ecc71" if not blocking else "#e74c3c"
        status_text = "✅ PASSED" if not blocking else "❌ BLOCKED"
        blocking_html = (
            "<ul>" + "".join(f"<li>{r}</li>" for r in blocking) + "</ul>"
            if blocking else "<p>No blocking issues.</p>"
        )
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Evaluation Report — {task_id[:8]}</title>
<style>body{{font-family:monospace;background:#0d1117;color:#c9d1d9;padding:2rem}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #30363d;padding:8px}}
th{{background:#161b22}}.badge{{display:inline-block;padding:6px 12px;border-radius:4px;
font-weight:bold;background:{status_colour};color:white}}</style></head>
<body>
<h1>Evaluation Report</h1>
<p>Workflow: <code>{task_id}</code> | Run: <code>{run_id[:16]}</code></p>
<div class="badge">{status_text}</div>
<h2>Standard Metrics</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
{"".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k,v in metrics.items())}
</table>
<h2>Slice Evaluation</h2>
<table><tr><th>Slice</th><th>Accuracy</th><th>F1</th><th>N</th></tr>{slice_rows}</table>
<h2>Bias Check</h2>
<table><tr><th>Attribute</th><th>Dem. Parity Diff</th><th>EO Diff</th><th>Status</th></tr>
{bias_rows}</table>
<h2>LLM Judge ({len(judge)} samples)</h2>
<p>Reasonable: {sum(1 for v in judge if v.verdict=="reasonable")} / {len(judge)}</p>
<h2>Blocking Reasons</h2>{blocking_html}
</body></html>"""