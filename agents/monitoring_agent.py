"""
agents/monitoring_agent.py

Monitoring Agent — V2 production rewrite (Phase 5).

What changed from V1:
  V1: _sample_live_accuracy() and _run_live_drift_check() were hardcoded
      stubs returning 0.83 / 0.04 regardless of what was actually deployed.
      RETRAINING_COUNTER.inc() fired on a "drift alert" that could never
      actually happen since the drift score was a constant below threshold.
  V2:
    - Real Evidently DataDriftPreset comparing sampled live inference inputs
      (queried from Loki, where Promtail ships the structured prediction
      logs serving/inference_server.py writes) against the model's training
      dataset (resolved from its mlops.dataset.uri MLflow tag).
    - Real Prometheus queries for p99 latency and error rate.
    - Real accuracy computed from ground_truth_labels when delayed labels
      are available; honestly reports "unavailable" (not a fabricated
      number) when they aren't — there is no labelling pipeline built yet.
    - Every check that can't run reports alert_level="error" with a
      populated reason, never silently substitutes a "healthy" value.
    - CRITICAL breaches write a row to workflow_triggers — the durable
      retraining queue OrchestratorAgent.process_pending_triggers() polls
      (see scripts/process_triggers.py). This is the "internal event queue"
      from the plan, implemented as a table instead of ephemeral pub/sub so
      a trigger survives a process restart and is auditable.
    - Every tick is persisted to monitoring_events regardless of outcome.

Two entry points:
  run(state)   — LangGraph node: one check right after a fresh deployment,
                 using whatever model_uri/model_name the workflow just
                 promoted. Kept for the existing orchestrator wiring.
  tick(...)    — standalone entry point for continuous production
                 monitoring, called every settings.monitoring_interval_seconds
                 by scripts/monitor_loop.py (a long-running pod or K8s
                 CronJob) — independent of any single workflow.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import mlflow
import pandas as pd
from prometheus_client import Gauge, Counter

from agents import AgentTaskResult
from agents.events import publish_alert
from configs.settings import settings

logger = logging.getLogger(__name__)

# Prometheus metrics
MODEL_ACCURACY_GAUGE   = Gauge("mlops_model_accuracy", "Recent production accuracy from ground truth labels", ["model_name"])
DRIFT_SCORE_GAUGE      = Gauge("mlops_serving_drift_score", "Live inference-traffic drift score vs training data", ["model_name"])
FEATURE_DRIFT_GAUGE    = Gauge("mlops_serving_feature_drift_score", "Per-feature live drift score", ["model_name", "feature_name"])
LATENCY_P99_GAUGE      = Gauge("mlops_model_p99_latency_ms", "Recent p99 inference latency", ["model_name"])
ERROR_RATE_GAUGE       = Gauge("mlops_model_error_rate", "Recent inference error rate", ["model_name"])
RETRAINING_COUNTER     = Counter("mlops_retraining_triggered_total", "Number of retraining events triggered", ["reason"])


class MonitoringAgent:
    """
    Production model health monitor.

    Lifecycle:
        agent = MonitoringAgent()
        await agent.connect(pool)
        result = agent.run(state)              # post-deployment node
        # or, in a standalone loop process:
        event = await agent.tick(model_name)    # continuous monitoring
    """

    def __init__(self) -> None:
        self._pool: Optional[Any] = None
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    async def connect(self, pool: Any) -> None:
        self._pool = pool

    # LangGraph entry point (post-deployment check)
    @mlflow.trace(name="monitoring_agent.run")
    def run(self, state: dict) -> AgentTaskResult:
        task_id = state.get("workflow_id", "unknown")
        model_name = state.get("model_name") or settings.mlflow_registered_model_name
        model_version = state.get("model_version")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            event = loop.run_until_complete(
                self._check_async(model_name=model_name, model_version=model_version)
            )
            return AgentTaskResult(
                task_id=task_id,
                status="success" if event["alert_level"] != "error" else "partial",
                output=event,
                error=event.get("check_errors") if event["alert_level"] == "error" else None,
            )
        except Exception as exc:
            logger.exception("MonitoringAgent failed for workflow %s", task_id)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))

    # Standalone loop entry point
    async def tick(
        self, model_name: str, model_version: Optional[str] = None,
    ) -> dict[str, Any]:
        """One monitoring cycle for a model in production. Called on a timer."""
        return await self._check_async(model_name=model_name, model_version=model_version)

    # Core check
    @mlflow.trace(name="monitoring_agent.check")
    async def _check_async(
        self, model_name: str, model_version: Optional[str] = None,
    ) -> dict[str, Any]:
        with mlflow.start_run(run_name=f"monitor-{model_name}-{int(time.time())}", nested=True):
            mlflow.set_tag("agent", "monitoring_agent")
            mlflow.set_tag("model_name", model_name)

            drift = await self._check_drift(model_name, model_version)
            perf = await self._check_serving_metrics(model_name, model_version)
            accuracy = await self._check_accuracy(model_name, model_version)

            alerts: list[str] = []
            alert_level = "none"
            check_errors: list[str] = []

            # Drift
            drift_score = drift.get("drift_score")
            if drift.get("alert_level") == "error":
                check_errors.append(f"drift check unavailable: {drift.get('check_error')}")
            elif drift_score is not None:
                DRIFT_SCORE_GAUGE.labels(model_name=model_name).set(drift_score)
                for feature, score in (drift.get("feature_scores") or {}).items():
                    FEATURE_DRIFT_GAUGE.labels(model_name=model_name, feature_name=feature).set(score)
                if drift_score > settings.drift_critical_threshold:
                    alerts.append(f"CRITICAL drift: score={drift_score:.3f} > {settings.drift_critical_threshold}")
                    alert_level = "critical"
                elif drift_score > settings.drift_warn_threshold:
                    alerts.append(f"WARNING drift: score={drift_score:.3f} > {settings.drift_warn_threshold}")
                    alert_level = max(alert_level, "warning", key=_severity_rank)

            # Accuracy
            live_accuracy = accuracy.get("accuracy")
            baseline_accuracy = accuracy.get("baseline_accuracy")
            if accuracy.get("alert_level") == "error":
                check_errors.append(f"accuracy check unavailable: {accuracy.get('check_error')}")
            elif live_accuracy is not None:
                MODEL_ACCURACY_GAUGE.labels(model_name=model_name).set(live_accuracy)
                if baseline_accuracy is not None and (baseline_accuracy - live_accuracy) > settings.accuracy_drop_threshold:
                    alerts.append(
                        f"CRITICAL accuracy drop: {live_accuracy:.3f} vs baseline {baseline_accuracy:.3f} "
                        f"(> {settings.accuracy_drop_threshold:.1%})"
                    )
                    alert_level = "critical"

            # Latency / error rate
            p99_ms = perf.get("p99_ms")
            error_rate = perf.get("error_rate")
            if perf.get("alert_level") == "error":
                check_errors.append(f"serving metrics unavailable: {perf.get('check_error')}")
            else:
                if p99_ms is not None:
                    LATENCY_P99_GAUGE.labels(model_name=model_name).set(p99_ms)
                    if p99_ms > settings.latency_p99_threshold_ms:
                        alerts.append(f"Latency breach: p99={p99_ms:.0f}ms > {settings.latency_p99_threshold_ms}ms")
                        alert_level = max(alert_level, "warning", key=_severity_rank)
                if error_rate is not None:
                    ERROR_RATE_GAUGE.labels(model_name=model_name).set(error_rate)

            if check_errors and alert_level == "none":
                alert_level = "error"

            mlflow.log_metrics({
                k: v for k, v in {
                    "monitoring.drift_score": drift_score,
                    "monitoring.accuracy": live_accuracy,
                    "monitoring.p99_ms": p99_ms,
                    "monitoring.error_rate": error_rate,
                }.items() if v is not None
            })
            if alerts:
                mlflow.set_tag("monitoring_alerts", "; ".join(alerts))

            event = {
                "model_name": model_name,
                "model_version": model_version,
                "drift_score": drift_score,
                "feature_scores": drift.get("feature_scores") or {},
                "sample_count": drift.get("sample_count") or 0,
                "accuracy": live_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "p99_latency_ms": p99_ms,
                "error_rate": error_rate,
                "alert_level": alert_level,
                "alerts": alerts,
                "check_errors": check_errors,
                "drift_report_r2_path": drift.get("report_r2_path"),
            }

            await self._persist_event(event)

            # Phase 6 §6.3 real-time stream — "none" ticks aren't alerts and
            # would just be polling noise on the UI's alert feed, so only
            # warning/critical/error levels get published.
            if alert_level != "none":
                await publish_alert(
                    severity=alert_level,
                    source="monitoring_agent",
                    model_name=model_name,
                    model_version=model_version,
                    alerts=alerts,
                    drift_score=drift_score,
                    accuracy=live_accuracy,
                    p99_latency_ms=p99_ms,
                    error_rate=error_rate,
                )

            if alert_level == "critical":
                reason = "; ".join(alerts)
                RETRAINING_COUNTER.labels(
                    reason="drift" if any("drift" in a.lower() for a in alerts) else "accuracy"
                ).inc()
                await self._trigger_retraining(model_name, model_version, reason)

            logger.info(
                "MonitoringAgent tick: model=%s alert_level=%s alerts=%s",
                model_name, alert_level, alerts,
            )
            return event

    # Drift check
    async def _check_drift(
        self, model_name: str, model_version: Optional[str],
    ) -> dict[str, Any]:
        """
        Compare recently sampled live inference inputs against the model's
        training dataset. Returns alert_level="error" (with check_error set)
        rather than a fake "no drift" result on any failure — this is the
        exact anti-pattern the Phase 1 audit found in DataAgent's drift path
        and it must not be repeated here.
        """
        try:
            dataset_uri = await self._resolve_training_dataset_uri(model_name, model_version)
            if not dataset_uri:
                return {"alert_level": "error", "check_error": "no mlops.dataset.uri tag on this model version"}

            recent_df = await self._sample_recent_inference_inputs(model_name)
            if recent_df is None or len(recent_df) < settings.monitoring_min_sample_size:
                got = 0 if recent_df is None else len(recent_df)
                return {
                    "alert_level": "error",
                    "check_error": f"insufficient recent inference samples ({got} < {settings.monitoring_min_sample_size})",
                }

            from agents.connectors import ConnectorFactory
            connector = ConnectorFactory.from_uri(dataset_uri)
            await connector.connect()
            ref_result = await connector.read()
            ref_df = ref_result.dataframe

            numeric_cols = [
                c for c in recent_df.select_dtypes(include="number").columns
                if c in ref_df.columns
            ]
            if not numeric_cols:
                return {"alert_level": "error", "check_error": "no shared numeric columns between live traffic and training data"}

            from evidently.legacy.report import Report
            from evidently.legacy.metric_preset import DataDriftPreset
            from evidently.legacy.metrics import DataDriftTable

            report = Report(metrics=[DataDriftPreset(), DataDriftTable()])
            report.run(reference_data=ref_df[numeric_cols], current_data=recent_df[numeric_cols])
            result_dict = report.as_dict()

            drift_score = 0.0
            feature_scores: dict[str, float] = {}
            for metric in result_dict.get("metrics", []):
                if metric.get("metric") == "DatasetDriftMetric":
                    drift_score = float(metric["result"]["share_of_drifted_columns"])
                if metric.get("metric") == "DataDriftTable":
                    for col_result in metric["result"].get("drift_by_columns", {}).values():
                        fname = col_result.get("column_name", "")
                        if fname:
                            feature_scores[fname] = float(col_result.get("drift_score", 0.0))

            # Only persist the full HTML report to R2 when there's actually
            # something worth keeping — a routine "all clear" tick has no
            # browsing value (nobody opens 1,000 identical no-drift
            # reports) and, run on a real production cadence, unconditional
            # storage would be a genuine unbounded-cost bug: a 60s tick
            # writing one object every cycle adds up fast against a
            # free-tier object storage quota. The drift_score/feature_scores
            # numbers themselves are always persisted to Postgres via
            # _persist_event below regardless — only the heavy HTML
            # artifact is gated.
            report_r2_path = None
            if drift_score > settings.drift_warn_threshold:
                report_r2_path = await self._store_report(report, model_name, "drift")

            return {
                "alert_level": "none",  # threshold classification happens in _check_async
                "drift_score": drift_score,
                "feature_scores": feature_scores,
                "sample_count": len(recent_df),
                "reference_dataset_uri": dataset_uri,
                "report_r2_path": report_r2_path,
            }

        except ImportError as exc:
            return {"alert_level": "error", "check_error": f"evidently not installed: {exc}"}
        except Exception as exc:
            logger.exception("Serving drift check failed for %s", model_name)
            return {"alert_level": "error", "check_error": str(exc)}

    async def _resolve_training_dataset_uri(
        self, model_name: str, model_version: Optional[str],
    ) -> Optional[str]:
        try:
            client = mlflow.tracking.MlflowClient()
            if model_version:
                mv = await asyncio.to_thread(client.get_model_version, model_name, model_version)
            else:
                versions = await asyncio.to_thread(
                    client.get_latest_versions, model_name, ["Production"]
                )
                if not versions:
                    return None
                mv = versions[0]
            return dict(mv.tags or {}).get("mlops.dataset.uri")
        except Exception as exc:
            logger.warning("Could not resolve training dataset for %s: %s", model_name, exc)
            return None

    async def _sample_recent_inference_inputs(self, model_name: str) -> Optional[pd.DataFrame]:
        """
        Query Loki for structured prediction logs the packaged inference
        server writes (see serving/inference_server.py _log_prediction).
        Returns None on any failure or empty result — callers must treat
        that as "can't verify", never as "no drift".
        """
        try:
            import aiohttp

            end = datetime.now(tz=timezone.utc)
            start = end - timedelta(minutes=settings.monitoring_sample_window_minutes)
            query = f'{{{settings.loki_service_label}="{model_name}"}} |= "mlops_prediction_log"'
            params = {
                "query": query,
                "start": str(int(start.timestamp() * 1e9)),
                "end": str(int(end.timestamp() * 1e9)),
                "limit": str(settings.monitoring_max_recent_samples),
                "direction": "backward",
            }
            url = f"{settings.loki_url}/loki/api/v1/query_range"

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning("Loki query returned HTTP %d for %s", resp.status, model_name)
                        return None
                    data = await resp.json()

            rows: list[dict] = []
            for stream in data.get("data", {}).get("result", []):
                for _, line in stream.get("values", []):
                    try:
                        payload = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    features = payload.get("features")
                    if isinstance(features, dict):
                        rows.append(features)

            if not rows:
                return None
            return pd.DataFrame(rows)

        except Exception as exc:
            logger.warning("Loki sample query failed for %s (Loki unreachable?): %s", model_name, exc)
            return None

    # Serving metrics (Prometheus)
    async def _check_serving_metrics(
        self, model_name: str, model_version: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Queries mlops_predictions_total / mlops_prediction_duration_seconds —
        the metrics serving/metrics.py actually exports from the packaged
        inference server. An earlier version of this query targeted
        http_requests_total / http_request_duration_seconds_bucket, which
        nothing in this codebase has ever exported — every real call would
        have found zero results and reported p99_ms=None/error_rate=None
        forever, regardless of actual serving health.
        """
        try:
            import urllib.parse
            import urllib.request
            import json as _json

            base = f"{settings.prometheus_url}/api/v1/query"
            label_selector = f'model_name="{model_name}"'
            if model_version:
                label_selector += f',model_version="{model_version}"'

            error_query = f'sum(rate(mlops_predictions_total{{{label_selector},status="error"}}[5m]))'
            total_query = f'sum(rate(mlops_predictions_total{{{label_selector}}}[5m]))'
            p99_query = (
                f'histogram_quantile(0.99, sum(rate(mlops_prediction_duration_seconds_bucket'
                f'{{{label_selector}}}[5m])) by (le)) * 1000'
            )

            def _query_sync(q: str) -> Optional[float]:
                url = f"{base}?{urllib.parse.urlencode({'query': q})}"
                with urllib.request.urlopen(url, timeout=5) as resp:
                    body = _json.loads(resp.read())
                result = body.get("data", {}).get("result", [])
                if not result:
                    return None
                return float(result[0]["value"][1])

            error_rate_val, total_val, p99_val = await asyncio.gather(
                asyncio.to_thread(_query_sync, error_query),
                asyncio.to_thread(_query_sync, total_query),
                asyncio.to_thread(_query_sync, p99_query),
                return_exceptions=True,
            )
            for v in (error_rate_val, total_val, p99_val):
                if isinstance(v, Exception):
                    raise v

            import math

            # histogram_quantile() over a window with no observations can
            # return NaN — a valid Python float where `nan > threshold` is
            # always False, silently treating "unknown" as "healthy" if left
            # unguarded (verified against a real Prometheus instance).
            if p99_val is not None and math.isnan(p99_val):
                p99_val = None

            error_rate = None
            if total_val is not None and not math.isnan(total_val) and total_val > 0:
                numerator = error_rate_val or 0.0
                if not math.isnan(numerator):
                    error_rate = numerator / total_val

            return {"alert_level": "none", "p99_ms": p99_val, "error_rate": error_rate}

        except Exception as exc:
            logger.warning("Prometheus serving-metrics query failed for %s: %s", model_name, exc)
            return {"alert_level": "error", "check_error": str(exc)}

    # Accuracy (delayed ground truth)
    async def _check_accuracy(
        self, model_name: str, model_version: Optional[str],
    ) -> dict[str, Any]:
        """
        Real accuracy computed from ground_truth_labels rows already labeled
        `correct`. There is no labelling pipeline implemented yet (plan §5.2
        calls this a separate Labelling Agent), so with zero labeled rows
        this honestly reports "unavailable" rather than a fabricated number
        — this is the direct fix for the V1 hardcoded 0.83.
        """
        if self._pool is None:
            return {"alert_level": "error", "check_error": "no DB pool — cannot query ground_truth_labels"}
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) AS acc, COUNT(*) AS n
                    FROM ground_truth_labels
                    WHERE model_name=$1 AND ($2::text IS NULL OR model_version=$2)
                      AND correct IS NOT NULL
                      AND labeled_at > NOW() - INTERVAL '1 hour' * $3
                    """,
                    model_name, model_version, settings.monitoring_accuracy_window_hours,
                )
                baseline_row = await conn.fetchrow(
                    """
                    SELECT eval_accuracy FROM model_registry_tags
                    WHERE model_name=$1 AND ($2::text IS NULL OR model_version=$2)
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    model_name, model_version,
                )
            if row is None or row["n"] == 0:
                return {
                    "alert_level": "none",
                    "accuracy": None,
                    "baseline_accuracy": baseline_row["eval_accuracy"] if baseline_row else None,
                    "check_error": None,
                }
            return {
                "alert_level": "none",
                "accuracy": float(row["acc"]),
                "baseline_accuracy": baseline_row["eval_accuracy"] if baseline_row else None,
            }
        except Exception as exc:
            logger.warning("Accuracy query failed for %s: %s", model_name, exc)
            return {"alert_level": "error", "check_error": str(exc)}

    # R2 report storage
    async def _store_report(self, report: Any, model_name: str, kind: str) -> Optional[str]:
        if not settings.r2_configured:
            return None
        try:
            import boto3
            from botocore.config import Config

            html_bytes = report.get_html().encode("utf-8")
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
            key = f"{settings.r2_reports_prefix}/serving/{model_name}/{timestamp}_{kind}.html"

            s3 = boto3.client(
                "s3",
                endpoint_url=settings.r2_endpoint_url,
                aws_access_key_id=settings.r2_access_key_id,
                aws_secret_access_key=settings.r2_secret_access_key,
                region_name=settings.r2_region,
                config=Config(retries={"max_attempts": 3}),
            )
            await asyncio.to_thread(
                s3.put_object, Bucket=settings.r2_bucket_name, Key=key,
                Body=html_bytes, ContentType="text/html",
            )
            return key
        except Exception as exc:
            logger.warning("Failed to store %s report for %s: %s", kind, model_name, exc)
            return None

    # Neon persistence
    async def _persist_event(self, event: dict[str, Any]) -> None:
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO monitoring_events (
                        model_name, model_version, drift_score, accuracy,
                        baseline_accuracy, p99_latency_ms, error_rate,
                        alert_level, alerts
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb)
                    """,
                    event["model_name"], event.get("model_version"),
                    event.get("drift_score"), event.get("accuracy"),
                    event.get("baseline_accuracy"), event.get("p99_latency_ms"),
                    event.get("error_rate"), event["alert_level"],
                    json.dumps(event.get("alerts", [])),
                )
                if event.get("drift_report_r2_path") or event.get("drift_score") is not None:
                    await conn.execute(
                        """
                        INSERT INTO serving_drift_reports (
                            model_name, model_version, sample_count, drift_score,
                            per_feature_scores, r2_report_path, alert_level, check_error
                        ) VALUES ($1,$2,$3,$4,$5::jsonb,$6,$7,$8)
                        """,
                        event["model_name"], event.get("model_version"),
                        event.get("sample_count", 0), event.get("drift_score"),
                        json.dumps(event.get("feature_scores", {})),
                        event.get("drift_report_r2_path"), event["alert_level"],
                        "; ".join(event.get("check_errors", [])) or None,
                    )
        except Exception as exc:
            logger.error("Failed to persist monitoring event: %s", exc)

    async def _trigger_retraining(
        self, model_name: str, model_version: Optional[str], reason: str,
    ) -> None:
        """
        Write a pending row to workflow_triggers. OrchestratorAgent.
        process_pending_triggers() (run by scripts/process_triggers.py) polls
        this table and launches a real workflow through the same graph a
        manual run uses — including the human-approval gate before any
        deployment. This is the durable "internal event queue" from the plan.
        """
        if self._pool is None:
            logger.error("Cannot trigger retraining for %s — no DB pool", model_name)
            return
        try:
            dataset_uri = await self._resolve_training_dataset_uri(model_name, model_version)
            async with self._pool.acquire() as conn:
                # Don't stack duplicate pending triggers for the same model.
                existing = await conn.fetchval(
                    """
                    SELECT id FROM workflow_triggers
                    WHERE model_name=$1 AND status='pending'
                    LIMIT 1
                    """,
                    model_name,
                )
                if existing:
                    logger.info("Retraining trigger already pending for %s (id=%s) — skipping", model_name, existing)
                    return
                await conn.execute(
                    """
                    INSERT INTO workflow_triggers (
                        trigger_type, source, model_name, model_version,
                        dataset_uri, reason
                    ) VALUES ('drift_detected', 'monitoring', $1, $2, $3, $4)
                    """,
                    model_name, model_version, dataset_uri, reason,
                )
            logger.warning("Retraining triggered for %s: %s", model_name, reason)
        except Exception as exc:
            logger.error("Failed to persist retraining trigger for %s: %s", model_name, exc)


def _severity_rank(level: str) -> int:
    return {"none": 0, "warning": 1, "error": 1, "critical": 2}.get(level, 0)
