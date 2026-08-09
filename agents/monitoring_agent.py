"""
Monitoring Agent

Responsibilities
- Continuously poll Prometheus for model performance metrics
- Run Evidently drift checks against live inference traffic
- Trigger retraining alerts when drift or degradation is detected
- Expose custom Prometheus metrics for Grafana dashboards
"""
from __future__ import annotations

import logging
import time
from typing import Any

import mlflow
from prometheus_client import Gauge, Counter, start_http_server

from agents import AgentTaskResult
from configs.settings import settings

logger = logging.getLogger(__name__)

# Prometheus metrics 
MODEL_ACCURACY_GAUGE = Gauge("mlops_model_accuracy", "Current production model accuracy", ["model_name"])
DRIFT_SCORE_GAUGE = Gauge("mlops_drift_score", "Data/concept drift score", ["model_name"])
RETRAINING_COUNTER = Counter("mlops_retraining_triggered_total", "Number of retraining events triggered")

_DEGRADATION_THRESHOLD = 0.05   # accuracy drop that triggers alert
_DRIFT_ALERT_THRESHOLD = 0.15


class MonitoringAgent:
    """
    Periodic background agent that tracks production model health.
    """

    def __init__(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self._baseline_accuracy: float | None = None

    # Main entry point 
    @mlflow.trace(name="monitoring_agent.run")
    def run(self, state: dict) -> AgentTaskResult:
        task_id = state.get("workflow_id", "unknown")
        model_uri = state.get("model_uri", "")

        try:
            with mlflow.start_run(run_name=f"monitor-{task_id}", nested=True):
                mlflow.set_tag("agent", "monitoring_agent")

                # Sample live performance
                live_accuracy = self._sample_live_accuracy(model_uri)
                drift_score = self._run_live_drift_check(model_uri)

                # Update Prometheus gauges
                model_name = model_uri.split("/")[-1] if model_uri else "unknown"
                MODEL_ACCURACY_GAUGE.labels(model_name=model_name).set(live_accuracy)
                DRIFT_SCORE_GAUGE.labels(model_name=model_name).set(drift_score)
                mlflow.log_metric("live_accuracy", live_accuracy)
                mlflow.log_metric("live_drift_score", drift_score)

                alerts = []
                if drift_score > _DRIFT_ALERT_THRESHOLD:
                    alerts.append(f"Drift alert: {drift_score:.3f} > {_DRIFT_ALERT_THRESHOLD}")
                    RETRAINING_COUNTER.inc()
                if self._baseline_accuracy and live_accuracy < self._baseline_accuracy - _DEGRADATION_THRESHOLD:
                    alerts.append(
                        f"Accuracy degradation: {live_accuracy:.3f} (was {self._baseline_accuracy:.3f})"
                    )

                if alerts:
                    mlflow.set_tag("monitoring_alerts", "; ".join(alerts))

                return AgentTaskResult(
                    task_id=task_id,
                    status="success",
                    output={
                        "live_accuracy": live_accuracy,
                        "drift_score": drift_score,
                        "alerts": alerts,
                    },
                )

        except Exception as exc:
            logger.exception("MonitoringAgent failed for workflow %s", task_id)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))

    def _sample_live_accuracy(self, model_uri: str) -> float:
        """Query Prometheus or MLflow for recent accuracy. Stub for dev."""
        return 0.83

    def _run_live_drift_check(self, model_uri: str) -> float:
        """Compare recent inference inputs against training distribution. Stub."""
        return 0.04
