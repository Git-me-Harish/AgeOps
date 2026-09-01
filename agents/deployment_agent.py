"""
Deployment Agent

Responsibilities
- Deploy models via KServe InferenceService on Kubernetes, using the signed,
  Trivy-scanned OCI image built by DockerfileAgent (Phase 4) when one is
  available in state — falling back to KServe's native MLflow runtime only
  when no packaged image exists (dev mode, or packaging was skipped).
- Execute canary rollout (10 % → 50 % → 100 %)
- Monitor canary error-rate during rollout window
- Automatic rollback if error-rate or latency thresholds breach, OR if
  canary metrics can't be sampled at all (fail-closed, not fail-open)

Registry promotion (Staging → Production) is NOT this agent's job — that
happens earlier in the orchestrator graph via PromotionStateMachine, before
packaging/deployment ever run. By the time DeploymentAgent runs, the model
is already Production in the MLflow Registry.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import mlflow
from kubernetes import client as k8s_client, config as k8s_config
from tenacity import retry, stop_after_attempt, wait_exponential

from agents import AgentTaskResult
from agents.events import publish_deployment_status_sync
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
        oci_image_uri = state.get("oci_image_uri", "")
        model_name = state.get("model_name", "")
        model_version = state.get("model_version", "")

        try:
            with mlflow.start_run(run_name=f"deploy-{task_id}", nested=True):
                mlflow.set_tag("agent", "deployment_agent")
                mlflow.log_param("model_uri", model_uri)
                mlflow.log_param("oci_image_uri", oci_image_uri or "none")

                if not oci_image_uri:
                    logger.warning(
                        "No OCI image in state — falling back to KServe's native "
                        "MLflow runtime against model_uri. This model was not "
                        "packaged through Phase 4 (Trivy/SBOM/Cosign) before deploy."
                    )

                if self._k8s_available:
                    self._deploy_canary(model_uri, oci_image_uri, task_id, model_name, model_version)
                else:
                    logger.warning("K8s unavailable; simulating deployment for dev")
                    self._simulate_deployment(model_uri)

                mlflow.set_tag("deployment_status", "production")

                return AgentTaskResult(
                    task_id=task_id,
                    status="success",
                    output={
                        "deployed_model_uri": model_uri,
                        "deployed_image_uri": oci_image_uri or None,
                        "strategy": "canary",
                    },
                )

        except RollbackTriggered as exc:
            logger.error("Rollback triggered for %s: %s", task_id, exc)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))
        except Exception as exc:
            logger.exception("DeploymentAgent failed for workflow %s", task_id)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))

    # Canary rollout
    @mlflow.trace(name="deployment_agent.canary_rollout")
    def _deploy_canary(
        self, model_uri: str, oci_image_uri: str, task_id: str,
        model_name: str = "", model_version: str = "",
    ) -> None:
        svc_name = f"mlops-model-{task_id[:8]}"
        self._create_inference_service(
            svc_name, model_uri, oci_image_uri, traffic_weight=_CANARY_STEPS[0]
        )

        for step_pct in _CANARY_STEPS:
            logger.info("Canary step: routing %d%% traffic to new model", step_pct)
            self._patch_traffic_weight(svc_name, step_pct)
            publish_deployment_status_sync(
                model_name, model_version=model_version, service=svc_name,
                phase="canary_step", traffic_pct=step_pct, status="in_progress",
            )
            time.sleep(_CANARY_WINDOW_SECONDS)

            sampled = self._sample_metrics(svc_name, model_name, model_version)
            if sampled is None:
                # Metrics could not be verified — a canary step that can't be
                # observed is not a canary step that passed. Roll back rather
                # than silently advancing traffic on faith.
                reason = (
                    f"Canary metrics unavailable at {step_pct}% (Prometheus "
                    "unreachable or query failed) — rolling back, not assuming pass"
                )
                self._rollback(svc_name)
                publish_deployment_status_sync(
                    model_name, model_version=model_version, service=svc_name,
                    phase="rollback", traffic_pct=step_pct, status="rolled_back", reason=reason,
                )
                raise RollbackTriggered(reason)

            error_rate, p99_ms = sampled
            mlflow.log_metric(f"canary_{step_pct}pct_error_rate", error_rate)
            mlflow.log_metric(f"canary_{step_pct}pct_p99_ms", p99_ms)

            if error_rate > _ERROR_RATE_THRESHOLD:
                reason = f"Canary error rate {error_rate:.1%} > {_ERROR_RATE_THRESHOLD:.1%} at {step_pct}%"
                self._rollback(svc_name)
                publish_deployment_status_sync(
                    model_name, model_version=model_version, service=svc_name,
                    phase="rollback", traffic_pct=step_pct, status="rolled_back", reason=reason,
                    error_rate=error_rate,
                )
                raise RollbackTriggered(reason)
            if p99_ms > _LATENCY_P99_THRESHOLD_MS:
                reason = f"Canary p99 latency {p99_ms:.0f}ms > {_LATENCY_P99_THRESHOLD_MS}ms at {step_pct}%"
                self._rollback(svc_name)
                publish_deployment_status_sync(
                    model_name, model_version=model_version, service=svc_name,
                    phase="rollback", traffic_pct=step_pct, status="rolled_back", reason=reason,
                    p99_ms=p99_ms,
                )
                raise RollbackTriggered(reason)

        logger.info("Canary rollout complete — 100%% traffic on new model")
        publish_deployment_status_sync(
            model_name, model_version=model_version, service=svc_name,
            phase="canary_complete", traffic_pct=100, status="deployed",
        )

    def _create_inference_service(
        self, name: str, model_uri: str, oci_image_uri: str, traffic_weight: int
    ) -> None:
        """
        Create a KServe InferenceService CRD.

        When an OCI image is available (the normal Phase 4 path), KServe runs
        it directly as a custom container — the exact image that was Trivy-
        scanned, SBOM'd, and Cosign-signed, exposing /predict per
        serving/inference_server.py on port 8080. This is "the image IS the
        deployment unit" from the plan's ADR-001.

        Falls back to KServe's native MLflow runtime (storageUri) only when
        no OCI image exists.
        """
        if oci_image_uri:
            predictor: dict[str, Any] = {
                "containers": [{
                    "name": "kserve-container",
                    "image": oci_image_uri,
                    "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                }],
                "canaryTrafficPercent": traffic_weight,
            }
        else:
            predictor = {
                "model": {
                    "modelFormat": {"name": "mlflow"},
                    "storageUri": model_uri,
                },
                "canaryTrafficPercent": traffic_weight,
            }

        body = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {"name": name, "namespace": settings.k8s_namespace},
            "spec": {"predictor": predictor},
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

    def _sample_metrics(
        self, svc_name: str, model_name: str = "", model_version: str = "",
    ) -> Optional[tuple[float, float]]:
        """
        Query Prometheus for real-time canary error rate and p99 latency.

        Queries mlops_predictions_total / mlops_prediction_duration_seconds —
        the metrics serving/metrics.py actually exports from the packaged
        inference server (labeled model_name/model_version/status). Earlier
        versions of this query targeted http_requests_total /
        http_request_duration_seconds_bucket, which nothing in this codebase
        has ever exported; every canary check would have found zero results
        and returned None, meaning every real canary rollout would roll back
        at the first step regardless of how the model was actually
        performing. Filtering on model_version (not the KServe service name)
        is what actually isolates canary (new version) traffic from stable
        (old version) traffic — KServe routes canaryTrafficPercent of
        model_name's traffic to pods running the new model_version.

        Returns None when Prometheus is unreachable, the query fails, or
        model_name/model_version aren't known — the caller must treat that
        as "can't verify this canary step," not as a passing result. A
        Prometheus outage must not be able to wave a bad model into 100%
        traffic.
        """
        if not model_name or not model_version:
            logger.error(
                "Canary metrics query skipped for %s — no model_name/model_version in state", svc_name,
            )
            return None
        try:
            import math
            import urllib.parse
            import urllib.request
            import json as _json

            base = f"{settings.prometheus_url}/api/v1/query"
            label_selector = f'model_name="{model_name}",model_version="{model_version}"'

            def _query(promql: str) -> list:
                url = f"{base}?{urllib.parse.urlencode({'query': promql})}"
                with urllib.request.urlopen(url, timeout=5) as resp:
                    body = _json.loads(resp.read())
                if body.get("status") != "success":
                    raise RuntimeError(f"Prometheus query failed: {body}")
                return body["data"]["result"]

            error_query = (
                f'sum(rate(mlops_predictions_total{{{label_selector},status="error"}}[2m])) / '
                f'sum(rate(mlops_predictions_total{{{label_selector}}}[2m]))'
            )
            error_result = _query(error_query)
            if not error_result:
                # No traffic in this window at all — can't confirm the model
                # is healthy, so don't assume 0% errors.
                return None
            error_rate = float(error_result[0]["value"][1])

            p99_query = (
                f'histogram_quantile(0.99, sum(rate(mlops_prediction_duration_seconds_bucket'
                f'{{{label_selector}}}[2m])) by (le)) * 1000'
            )
            p99_result = _query(p99_query)
            if not p99_result:
                # Error rate came back but latency didn't — still can't verify
                # the latency gate, so this counts as an unavailable sample.
                return None
            p99_ms = float(p99_result[0]["value"][1])

            # PromQL division of two zero-rate series (no traffic at all in
            # the window, or a histogram with no observations) returns NaN,
            # which is a perfectly valid Python float — `nan > threshold` is
            # always False, so an unguarded comparison would silently treat
            # "we don't know" as "it passed." Verified against a real
            # Prometheus: a static (non-incrementing) counter produces
            # exactly this NaN response.
            if math.isnan(error_rate) or math.isnan(p99_ms):
                return None

            return error_rate, p99_ms
        except Exception as exc:
            logger.error("Canary metrics query failed for %s (model=%s v%s): %s", svc_name, model_name, model_version, exc)
            return None

    def manual_rollback(self, svc_name: str, reason: str, model_name: str = "", model_version: str = "") -> None:
        """
        Operator-initiated rollback (Phase 6 Live Serving page's Rollback
        button) — distinct from the automatic in-loop rollback _deploy_canary
        triggers on a metrics breach, but reuses the exact same
        _rollback() K8s patch so there is only one rollback code path.
        Requires a reason (enforced by the API layer, not here) so every
        rollback is auditable regardless of who/what triggered it.
        """
        if not self._k8s_available:
            raise RuntimeError("Kubernetes client unavailable — cannot patch InferenceService")
        self._rollback(svc_name)
        publish_deployment_status_sync(
            model_name or svc_name, model_version=model_version, service=svc_name,
            phase="manual_rollback", traffic_pct=0, status="rolled_back", reason=reason,
        )

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


class RollbackTriggered(Exception):
    """Raised when canary metrics exceed thresholds and rollback is executed."""
