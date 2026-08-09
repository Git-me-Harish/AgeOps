"""
Training Agent

Responsibilities
- Launch Kubernetes training Jobs (or run locally for dev)
- Enable MLflow autologging for all supported frameworks
- Track hyperparameters, metrics, and model artifacts
- Register the best model in the MLflow Model Registry
- Honour RL agent recommendations for hyperparameter adjustments
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

import mlflow
import mlflow.sklearn
from kubernetes import client as k8s_client, config as k8s_config
from tenacity import retry, stop_after_attempt, wait_exponential

from agents import AgentTaskResult
from configs.settings import settings

logger = logging.getLogger(__name__)

_TRAINING_JOB_TIMEOUT = 3600  # 1 hour max per training job


class TrainingAgent:
    """
    Executes training jobs via Kubernetes Jobs and tracks everything in MLflow.
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
            self._batch_v1 = k8s_client.BatchV1Api()
            self._k8s_available = True
        except Exception as exc:
            logger.warning("Kubernetes unavailable, training will run in-process: %s", exc)
            self._k8s_available = False

    # Main entry point 
    @mlflow.trace(name="training_agent.run")
    def run(self, state: dict) -> AgentTaskResult:
        task_id = state.get("workflow_id", "unknown")
        dataset_uri = state.get("dataset_uri", "")
        rl_recs = state.get("rl_recommendations") or {}

        try:
            with mlflow.start_run(run_name=f"train-{task_id}", nested=True) as run:
                mlflow.set_tag("agent", "training_agent")
                mlflow.log_param("dataset_uri", dataset_uri)
                mlflow.log_param("rl_adjustments", str(rl_recs))

                hyperparams = self._merge_hyperparams(rl_recs)
                mlflow.log_params(hyperparams)

                if self._k8s_available:
                    model_uri, metrics = self._run_k8s_job(task_id, dataset_uri, hyperparams, run.info.run_id)
                else:
                    model_uri, metrics = self._run_in_process(task_id, dataset_uri, hyperparams, run.info.run_id)

                # Register in MLflow Model Registry as "Staging"
                registered = mlflow.register_model(model_uri=model_uri, name="mlops-model")
                mlflow.set_tag("model_version", registered.version)
                mlflow.log_metrics(metrics)

                return AgentTaskResult(
                    task_id=task_id,
                    status="success",
                    output={"model_uri": model_uri, "metrics": metrics, "run_id": run.info.run_id},
                    confidence=metrics.get("accuracy", 0.0),
                )

        except Exception as exc:
            logger.exception("TrainingAgent failed for workflow %s", task_id)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))

    # Hyperparameter resolution 
    def _merge_hyperparams(self, rl_recs: dict) -> dict[str, Any]:
        """
        Start with sensible defaults and overlay RL agent suggestions.
        """
        defaults: dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
        }
        # RL agent may suggest batch_size, learning_rate adjustments etc.
        for k, v in rl_recs.items():
            if k in defaults:
                logger.info("RL agent overriding %s: %s → %s", k, defaults[k], v)
                defaults[k] = v
        return defaults

    # Kubernetes-backed training 
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=5, max=30))
    @mlflow.trace(name="training_agent.k8s_job")
    def _run_k8s_job(
        self,
        task_id: str,
        dataset_uri: str,
        hyperparams: dict,
        run_id: str,
    ) -> tuple[str, dict]:
        job_name = f"train-{task_id[:8]}-{uuid.uuid4().hex[:6]}"
        job_manifest = self._build_job_manifest(job_name, dataset_uri, hyperparams, run_id)

        logger.info("Dispatching Kubernetes training Job: %s", job_name)
        self._batch_v1.create_namespaced_job(
            namespace=settings.k8s_namespace,
            body=job_manifest,
        )

        # Poll until complete
        model_uri, metrics = self._wait_for_job(job_name, run_id)
        return model_uri, metrics

    def _build_job_manifest(
        self, job_name: str, dataset_uri: str, hyperparams: dict, run_id: str
    ) -> dict:
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "namespace": settings.k8s_namespace},
            "spec": {
                "backoffLimit": 2,
                "ttlSecondsAfterFinished": 3600,
                "template": {
                    "spec": {
                        "restartPolicy": "OnFailure",
                        "serviceAccountName": "training-agent",
                        "securityContext": {"runAsNonRoot": True, "runAsUser": 1000},
                        "containers": [{
                            "name": "trainer",
                            "image": "your-dockerhub-user/mlops-trainer:latest",
                            "imagePullPolicy": "Always",
                            "resources": {
                                "requests": {"memory": "1Gi", "cpu": "500m"},
                                "limits": {"memory": "2Gi", "cpu": "1000m"},
                            },
                            "env": [
                                {"name": "DATASET_URI", "value": dataset_uri},
                                {"name": "MLFLOW_RUN_ID", "value": run_id},
                                {"name": "MLFLOW_TRACKING_URI", "value": settings.mlflow_tracking_uri},
                                {"name": "HYPERPARAMS", "value": str(hyperparams)},
                                {"name": "AWS_ENDPOINT_URL", "value": settings.r2_endpoint_url},
                                {
                                    "name": "AWS_ACCESS_KEY_ID",
                                    "valueFrom": {"secretKeyRef": {"name": "r2-credentials", "key": "access-key"}},
                                },
                                {
                                    "name": "AWS_SECRET_ACCESS_KEY",
                                    "valueFrom": {"secretKeyRef": {"name": "r2-credentials", "key": "secret-key"}},
                                },
                            ],
                        }],
                    }
                },
            },
        }

    def _wait_for_job(self, job_name: str, run_id: str) -> tuple[str, dict]:
        """Poll Kubernetes Job until succeeded or timeout."""
        deadline = time.time() + _TRAINING_JOB_TIMEOUT
        while time.time() < deadline:
            job = self._batch_v1.read_namespaced_job(job_name, settings.k8s_namespace)
            if job.status.succeeded:
                break
            if job.status.failed and job.status.failed >= 3:
                raise RuntimeError(f"Kubernetes training job {job_name} failed after 3 attempts")
            time.sleep(15)
        else:
            raise TimeoutError(f"Training job {job_name} exceeded {_TRAINING_JOB_TIMEOUT}s")

        # The training container logs the model_uri and metrics to MLflow
        run = mlflow.get_run(run_id)
        model_uri = f"runs:/{run_id}/model"
        metrics = run.data.metrics
        return model_uri, metrics

    # In-process fallback training (dev / CI) 
    @mlflow.trace(name="training_agent.in_process")
    def _run_in_process(
        self,
        task_id: str,
        dataset_uri: str,
        hyperparams: dict,
        run_id: str,
    ) -> tuple[str, dict]:
        """
        Lightweight in-process training using scikit-learn + MLflow autolog.
        Used when Kubernetes is not available (local dev, CI).
        """
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score

        mlflow.sklearn.autolog(log_models=True)

        # Synthetic data when R2 not available in tests
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = GradientBoostingClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", 6),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            subsample=hyperparams.get("subsample", 0.8),
            random_state=hyperparams.get("random_state", 42),
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        }
        mlflow.log_metrics(metrics)
        model_uri = f"runs:/{run_id}/model"
        return model_uri, metrics
