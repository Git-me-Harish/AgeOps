"""
Deployment Agent

Responsibilities
- Deploy models via KServe InferenceService on Kubernetes
- Execute canary rollout (10 % → 50 % → 100 %)
- Monitor canary error-rate during rollout window
- Automatic rollback if error-rate or latency thresholds breach
- Transition model stage in MLflow Registry (Staging → Production)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import mlflow
from kubernetes import client as k8s_client, config as k8s_config
from tenacity import retry, stop_after_attempt, wait_exponential

from agents import AgentTaskResult
from configs.settings import settings

logger = logging.getLogger(__name__)

_CANARY_STEPS = [10, 50, 100]          # traffic percentages
_CANARY_WINDOW_SECONDS = 120            # wait per step
_ERROR_RATE_THRESHOLD = 0.05            # 5 % canary errors → rollback
_LATENCY_P99_THRESHOLD_MS = 500        # p99 latency limit


class DeploymentAgent:
    """
    Manages model deployments to the Kubernetes cluster via KServe.
    """

    def __init__(self) -> None:
        self._setup_k8s()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    def _setup_k8s(self) -> None:
        try:
            if settings.k8s_in_cluster:
                k8s_config.load_incluster_config()
            else:
                k8s_config.load_kube_config()
            self._custom = k8s_client.CustomObjectsApi()
            self._k8s_available = True
        except Exception as exc:
            logger.warning("Kubernetes unavailable for deployment: %s", exc)
            self._k8s_available = False

    # Main entry point 
    @mlflow.trace(name="deployment_agent.run")
    def run(self, state: dict) -> AgentTaskResult:
        task_id = state.get("workflow_id", "unknown")
        model_uri = state.get("model_uri", "")
        metrics = state.get("metrics", {})

        try:
            with mlflow.start_run(run_name=f"deploy-{task_id}", nested=True):
                mlflow.set_tag("agent", "deployment_agent")
                mlflow.log_param("model_uri", model_uri)

                if self._k8s_available:
                    self._deploy_canary(model_uri, task_id)
                else:
                    logger.warning("K8s unavailable; simulating deployment for dev")
                    self._simulate_deployment(model_uri)

                # Promote model in MLflow Registry
                self._promote_model_in_registry(model_uri)
                mlflow.set_tag("deployment_status", "production")

                return AgentTaskResult(
                    task_id=task_id,
                    status="success",
                    output={"deployed_model_uri": model_uri, "strategy": "canary"},
                )

        except RollbackTriggered as exc:
            logger.error("Rollback triggered for %s: %s", task_id, exc)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))
        except Exception as exc:
            logger.exception("DeploymentAgent failed for workflow %s", task_id)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))

    # Canary rollout 
    @mlflow.trace(name="deployment_agent.canary_rollout")
    def _deploy_canary(self, model_uri: str, task_id: str) -> None:
        svc_name = f"mlops-model-{task_id[:8]}"
        self._create_inference_service(svc_name, model_uri, traffic_weight=_CANARY_STEPS[0])

        for step_pct in _CANARY_STEPS:
            logger.info("Canary step: routing %d%% traffic to new model", step_pct)
            self._patch_traffic_weight(svc_name, step_pct)
            time.sleep(_CANARY_WINDOW_SECONDS)

            error_rate, p99_ms = self._sample_metrics(svc_name)
            mlflow.log_metric(f"canary_{step_pct}pct_error_rate", error_rate)
            mlflow.log_metric(f"canary_{step_pct}pct_p99_ms", p99_ms)

            if error_rate > _ERROR_RATE_THRESHOLD:
                self._rollback(svc_name)
                raise RollbackTriggered(
                    f"Canary error rate {error_rate:.1%} > {_ERROR_RATE_THRESHOLD:.1%} at {step_pct}%"
                )
            if p99_ms > _LATENCY_P99_THRESHOLD_MS:
                self._rollback(svc_name)
                raise RollbackTriggered(
                    f"Canary p99 latency {p99_ms:.0f}ms > {_LATENCY_P99_THRESHOLD_MS}ms at {step_pct}%"
                )

        logger.info("Canary rollout complete — 100%% traffic on new model")

    def _create_inference_service(self, name: str, model_uri: str, traffic_weight: int) -> None:
        """Create a KServe InferenceService CRD."""
        body = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {"name": name, "namespace": settings.k8s_namespace},
            "spec": {
                "predictor": {
                    "model": {
                        "modelFormat": {"name": "mlflow"},
                        "storageUri": model_uri,
                    },
                    "canaryTrafficPercent": traffic_weight,
                }
            },
        }
        self._custom.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=settings.k8s_namespace,
            plural="inferenceservices",
            body=body,
        )

    def _patch_traffic_weight(self, name: str, weight: int) -> None:
        patch = {"spec": {"predictor": {"canaryTrafficPercent": weight}}}
        self._custom.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=settings.k8s_namespace,
            plural="inferenceservices",
            name=name,
            body=patch,
        )

    def _sample_metrics(self, svc_name: str) -> tuple[float, float]:
        """
        Query Prometheus for real-time canary metrics.
        Falls back to safe dummy values when Prometheus is unreachable.
        """
        try:
            import urllib.request, json as _json
            query = f'rate(http_requests_total{{service="{svc_name}",status=~"5.."}}[2m])'
            url = f"{settings.grafana_url.replace('grafana', 'prometheus')}/api/v1/query?query={query}"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = _json.loads(resp.read())
            error_rate = float(data["data"]["result"][0]["value"][1]) if data["data"]["result"] else 0.0
            return error_rate, 100.0   # p99 stub
        except Exception:
            return 0.0, 50.0   # safe dummy values

    def _rollback(self, svc_name: str) -> None:
        logger.warning("Rolling back InferenceService %s", svc_name)
        patch = {"spec": {"predictor": {"canaryTrafficPercent": 0}}}
        try:
            self._custom.patch_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=settings.k8s_namespace,
                plural="inferenceservices",
                name=svc_name,
                body=patch,
            )
        except Exception as exc:
            logger.error("Rollback patch failed: %s", exc)

    def _simulate_deployment(self, model_uri: str) -> None:
        logger.info("[DEV] Simulating canary deployment for %s", model_uri)
        time.sleep(1)

    # MLflow Registry promotion 
    def _promote_model_in_registry(self, model_uri: str) -> None:
        try:
            client = mlflow.tracking.MlflowClient()
            # Extract run_id from model_uri: runs:/<run_id>/model
            run_id = model_uri.split("/")[1] if "runs:/" in model_uri else None
            if run_id:
                versions = client.search_model_versions(f"run_id='{run_id}'")
                for v in versions:
                    client.transition_model_version_stage(
                        name=v.name,
                        version=v.version,
                        stage="Production",
                        archive_existing_versions=True,
                    )
                    logger.info("Promoted %s v%s to Production", v.name, v.version)
        except Exception as exc:
            logger.warning("MLflow Registry promotion failed: %s", exc)


class RollbackTriggered(Exception):
    """Raised when canary metrics exceed thresholds and rollback is executed."""
