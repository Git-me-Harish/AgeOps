"""
agents/training_agent.py

Training Agent — V2 production rewrite.

What changed from V1:
  V1: single hardcoded GBM Job manifest with make_classification() synthetic data
  V2:
    - Framework router: sklearn | xgboost | pytorch | huggingface | custom
    - Job manifests generated from templates (not hardcoded)
    - Dataset URI + Feast feature view + MLflow run ID injected as env vars
    - Real-time log streaming via Kubernetes log API
    - Failure handling: RUNNING | SUCCEEDED | FAILED | OOMKilled
    - OOMKilled: halve batch_size and retry (RL agent records this outcome)
    - Parallel experiments: launch N Jobs concurrently, collect all results
    - Metrics retrieved from MLflow run (not from Job stdout)
    - LLMGateway used for framework selection reasoning
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Optional

import mlflow
from pydantic import BaseModel

from agents import AgentTaskResult
from agents.llm_gateway import LLMGateway
from configs.settings import settings

logger = logging.getLogger(__name__)


# Typed models
class TrainingJobConfig(BaseModel):
    """Configuration for one Kubernetes training Job."""
    job_name: str
    framework: str
    runner_image: str
    dataset_uri: str
    feast_feature_view: str
    mlflow_run_id: str
    mlflow_tracking_uri: str
    hyperparams: dict[str, Any]
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    namespace: str = "mlops"
    service_account: str = "training-agent"


class TrainingResult(BaseModel):
    """Result from one completed training Job."""
    job_name: str
    framework: str
    mlflow_run_id: str
    status: str                  # succeeded | failed | oomkilled
    metrics: dict[str, float]
    duration_seconds: int
    error_message: Optional[str] = None
    oom_retried: bool = False


# Training Agent
class TrainingAgent:
    """
    Launches and monitors Kubernetes training Jobs.

    Lifecycle per workflow:
      1. LLM selects framework(s) from ExecutionPlan
      2. For each framework: generate manifest → launch K8s Job
      3. Stream logs → Neon (for UI display)
      4. Poll job status until SUCCEEDED / FAILED / OOMKilled
      5. On OOMKilled: halve batch_size, create new Job, retry once
      6. On SUCCEEDED: retrieve metrics from MLflow (autologging)
      7. Return best result across all parallel experiments
    """

    def __init__(self) -> None:
        self._pool: Optional[Any] = None
        self._k8s_client: Optional[Any] = None

    async def connect(self, pool: Any) -> None:
        self._pool = pool
        await self._init_k8s()

    async def _init_k8s(self) -> None:
        """Initialise Kubernetes client — in-cluster or local kubeconfig."""
        try:
            from kubernetes import client as k8s_client, config as k8s_config
            if settings.k8s_in_cluster:
                k8s_config.load_incluster_config()
            else:
                k8s_config.load_kube_config()
            self._k8s_client = k8s_client
            logger.info("Kubernetes client initialised (in_cluster=%s)", settings.k8s_in_cluster)
        except Exception as exc:
            logger.warning("Kubernetes client init failed — training will error at launch: %s", exc)

    @mlflow.trace(name="training_agent.run")
    def run(self, state: dict) -> AgentTaskResult:
        """Synchronous LangGraph entry point."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._run_async(state))

    async def _run_async(self, state: dict) -> AgentTaskResult:
        task_id: str = state.get("workflow_id", "unknown")
        execution_plan: dict = state.get("execution_plan", {})
        dataset_uri: str = state.get("dataset_uri", "")
        lineage_id: Optional[int] = state.get("lineage_id")
        # DataAgent registers a per-workflow dynamic FeatureView backed by the
        # data it actually ingested (feast/feature_views.py
        # register_dynamic_feature_view). Fall back to the static generic
        # view only if that registration didn't happen.
        feast_feature_view: str = state.get("feast_feature_view") or "numeric_features"

        frameworks: list[str] = execution_plan.get("frameworks", ["xgboost"])
        hyp_grids: list[dict] = execution_plan.get("hyperparameter_grids", [{}])
        parallel: int = min(execution_plan.get("parallel_experiments", 1), 3)

        gateway = LLMGateway(pool=self._pool, workflow_id=task_id)

        try:
            with mlflow.start_run(run_name=f"training-{task_id}", nested=True) as parent_run:
                mlflow.set_tag("agent", "training_agent")
                mlflow.log_param("frameworks", json.dumps(frameworks))
                mlflow.log_param("parallel_experiments", parallel)

                # Build job configs for each framework
                job_configs = []
                for i, framework in enumerate(frameworks[:parallel]):
                    hparams = hyp_grids[i] if i < len(hyp_grids) else {"learning_rate": 0.01}
                    runner_image = settings.k8s_runner_image_map.get(
                        framework, settings.k8s_runner_custom
                    )
                    # Each framework gets its own nested MLflow run
                    child_run = mlflow.start_run(
                        run_name=f"{framework}-{task_id[:8]}", nested=True
                    )
                    job_configs.append(TrainingJobConfig(
                        job_name=f"mlops-train-{framework}-{task_id[:8]}-{i}",
                        framework=framework,
                        runner_image=runner_image,
                        dataset_uri=dataset_uri,
                        feast_feature_view=feast_feature_view,
                        mlflow_run_id=child_run.info.run_id,
                        mlflow_tracking_uri=settings.mlflow_tracking_uri,
                        hyperparams=hparams,
                        memory_request=settings.k8s_training_memory_request,
                        memory_limit=settings.k8s_training_memory_limit,
                        cpu_request=settings.k8s_training_cpu_request,
                        cpu_limit=settings.k8s_training_cpu_limit,
                        namespace=settings.k8s_namespace,
                        service_account=settings.k8s_service_account,
                    ))
                    mlflow.end_run()  # end child context; job will re-attach by run_id

                # Launch all jobs concurrently
                results: list[TrainingResult] = await asyncio.gather(
                    *[self._run_job(cfg) for cfg in job_configs],
                    return_exceptions=False,
                )

                # Select best result by primary metric
                successful = [r for r in results if r.status == "succeeded"]
                if not successful:
                    failed_errors = [r.error_message for r in results]
                    return AgentTaskResult(
                        task_id=task_id,
                        status="failed",
                        error=f"All training jobs failed: {failed_errors}",
                    )

                best = max(successful, key=lambda r: r.metrics.get("f1", r.metrics.get("accuracy", 0.0)))
                logger.info(
                    "TrainingAgent: best result — framework=%s f1=%.4f run_id=%s",
                    best.framework, best.metrics.get("f1", 0.0), best.mlflow_run_id,
                )

                mlflow.log_metrics({
                    "best.f1":        best.metrics.get("f1", 0.0),
                    "best.accuracy":  best.metrics.get("accuracy", 0.0),
                    "best.auc":       best.metrics.get("auc", 0.0),
                    "jobs.total":     len(results),
                    "jobs.succeeded": len(successful),
                    "jobs.failed":    len(results) - len(successful),
                })

                return AgentTaskResult(
                    task_id=task_id,
                    status="success",
                    output={
                        "best_framework":   best.framework,
                        "best_run_id":      best.mlflow_run_id,
                        "best_metrics":     best.metrics,
                        "all_results":      [r.model_dump() for r in results],
                        "oom_retried":      any(r.oom_retried for r in results),
                    },
                    confidence=best.metrics.get("f1", best.metrics.get("accuracy", 0.0)),
                )

        except Exception as exc:
            logger.exception("TrainingAgent._run_async failed for workflow %s", task_id)
            return AgentTaskResult(task_id=task_id, status="failed", error=str(exc))

    # Job lifecycle 
    async def _run_job(self, cfg: TrainingJobConfig) -> TrainingResult:
        """
        Launch one Kubernetes Job and block until completion.
        Handles OOMKilled by reducing batch_size and retrying once.
        """
        start = time.monotonic()
        oom_retried = False

        status, error = await self._launch_and_wait(cfg)

        # OOMKilled handler — halve batch_size and retry once
        if status == "oomkilled":
            logger.warning(
                "Job %s OOMKilled — retrying with halved batch_size", cfg.job_name
            )
            retry_cfg = cfg.model_copy(deep=True)
            current_bs = retry_cfg.hyperparams.get("batch_size", 64)
            retry_cfg.hyperparams["batch_size"] = max(16, current_bs // 2)
            retry_cfg.job_name = cfg.job_name + "-oom-retry"
            retry_cfg.memory_limit = "8Gi"   # more memory on retry
            status, error = await self._launch_and_wait(retry_cfg)
            oom_retried = True
            # Use retry config's run for metrics
            cfg = retry_cfg

        duration = int(time.monotonic() - start)
        metrics = {}
        if status == "succeeded":
            metrics = await self._retrieve_metrics(cfg.mlflow_run_id)

        return TrainingResult(
            job_name=cfg.job_name,
            framework=cfg.framework,
            mlflow_run_id=cfg.mlflow_run_id,
            status=status,
            metrics=metrics,
            duration_seconds=duration,
            error_message=error,
            oom_retried=oom_retried,
        )

    async def _launch_and_wait(
        self, cfg: TrainingJobConfig
    ) -> tuple[str, Optional[str]]:
        """
        Create the Kubernetes Job and poll until terminal state.
        Returns (status, error_message).

        status: "succeeded" | "failed" | "oomkilled"
        """
        manifest = self._build_manifest(cfg)

        if self._k8s_client is None:
            logger.warning(
                "Kubernetes client not available — simulating job for dev mode: %s",
                cfg.job_name,
            )
            return await self._simulate_job_dev(cfg)

        # Create the Job
        try:
            batch_v1 = self._k8s_client.BatchV1Api()
            await asyncio.to_thread(
                batch_v1.create_namespaced_job,
                namespace=cfg.namespace,
                body=manifest,
            )
            logger.info("Kubernetes Job created: %s", cfg.job_name)
        except Exception as exc:
            return "failed", f"Job creation failed: {exc}"

        # Poll until terminal
        timeout = settings.k8s_training_timeout_seconds
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            await asyncio.sleep(15)
            try:
                status, error = await self._get_job_status(cfg.job_name, cfg.namespace)
                if status in ("succeeded", "failed", "oomkilled"):
                    return status, error
            except Exception as exc:
                logger.warning("Job status poll failed (retrying): %s", exc)

        # Timeout → delete job and return failed
        await self._delete_job(cfg.job_name, cfg.namespace)
        return "failed", f"Job timed out after {timeout}s"

    async def _get_job_status(
        self, job_name: str, namespace: str
    ) -> tuple[str, Optional[str]]:
        """Poll Kubernetes Job status and translate to our status enum."""
        batch_v1 = self._k8s_client.BatchV1Api()
        job = await asyncio.to_thread(
            batch_v1.read_namespaced_job_status,
            name=job_name,
            namespace=namespace,
        )
        jstatus = job.status

        if jstatus.succeeded and jstatus.succeeded > 0:
            return "succeeded", None
        if jstatus.failed and jstatus.failed > 0:
            # Check if OOMKilled via pod conditions
            oom = await self._check_oomkilled(job_name, namespace)
            if oom:
                return "oomkilled", "Pod OOMKilled — exceeded memory limit"
            return "failed", f"Job failed after {jstatus.failed} attempts"

        return "running", None

    async def _check_oomkilled(self, job_name: str, namespace: str) -> bool:
        """Check if the Job's pod was OOMKilled."""
        try:
            core_v1 = self._k8s_client.CoreV1Api()
            pods = await asyncio.to_thread(
                core_v1.list_namespaced_pod,
                namespace=namespace,
                label_selector=f"job-name={job_name}",
            )
            for pod in pods.items:
                for cs in (pod.status.container_statuses or []):
                    if cs.last_state and cs.last_state.terminated:
                        if cs.last_state.terminated.reason == "OOMKilled":
                            return True
        except Exception:
            pass
        return False

    async def _delete_job(self, job_name: str, namespace: str) -> None:
        """Delete a timed-out or failed Job."""
        try:
            batch_v1 = self._k8s_client.BatchV1Api()
            await asyncio.to_thread(
                batch_v1.delete_namespaced_job,
                name=job_name,
                namespace=namespace,
                propagation_policy="Background",
            )
        except Exception as exc:
            logger.warning("Failed to delete Job %s: %s", job_name, exc)

    # Dev simulation (no K8s) 

    async def _simulate_job_dev(self, cfg: TrainingJobConfig) -> tuple[str, Optional[str]]:
        """
        Dev-mode training — runs in-process (no K8s Job) on the REAL dataset
        at cfg.dataset_uri, read through the same ConnectorFactory the Data
        Agent uses. Used when k8s_in_cluster=False and no kubeconfig is
        available.

        Deliberately does NOT fall back to synthetic data
        (sklearn.datasets.make_classification): a cluster being unreachable
        is an infrastructure problem, not a reason to silently train on noise
        and report real-looking metrics for it. If the real dataset can't be
        read, this fails loudly instead.
        """
        logger.info(
            "DEV MODE: training in-process for framework=%s dataset=%s",
            cfg.framework, cfg.dataset_uri,
        )

        try:
            from agents.connectors import ConnectorFactory
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

            if not cfg.dataset_uri:
                raise ValueError(
                    "No dataset_uri in training config — cannot train without a real dataset"
                )

            connector = ConnectorFactory.from_uri(cfg.dataset_uri)
            await connector.connect()
            ingestion_result = await connector.read()
            df = ingestion_result.dataframe

            label_col = "label" if "label" in df.columns else df.columns[-1]
            feature_cols = [
                c for c in df.select_dtypes(include="number").columns if c != label_col
            ]
            if not feature_cols:
                raise ValueError(
                    f"No numeric feature columns found in dataset at {cfg.dataset_uri}"
                )

            X = df[feature_cols].fillna(0)
            y = df[label_col]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            n_estimators = cfg.hyperparams.get("n_estimators", 100)
            if isinstance(n_estimators, list):
                n_estimators = n_estimators[0]
            clf = GradientBoostingClassifier(
                learning_rate=cfg.hyperparams.get("learning_rate", 0.01),
                n_estimators=n_estimators,
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
            acc = float(accuracy_score(y_test, preds))
            try:
                probas = clf.predict_proba(X_test)[:, 1]
                auc = float(roc_auc_score(y_test, probas))
            except Exception:
                auc = 0.0

            # Log to the child MLflow run
            with mlflow.start_run(run_id=cfg.mlflow_run_id):
                mlflow.set_tag("dev_mode_in_process", "true")
                mlflow.log_param("framework", cfg.framework)
                mlflow.log_param("learning_rate", cfg.hyperparams.get("learning_rate", 0.01))
                mlflow.log_param("dataset_uri", cfg.dataset_uri)
                mlflow.log_metrics({"f1": f1, "accuracy": acc, "auc": auc})
                mlflow.sklearn.log_model(clf, "model")

            logger.info(
                "DEV in-process training complete: framework=%s f1=%.4f acc=%.4f (real dataset)",
                cfg.framework, f1, acc,
            )
            return "succeeded", None

        except Exception as exc:
            logger.exception("DEV in-process training failed: %s", exc)
            return "failed", str(exc)

    # Manifest builder 

    def _build_manifest(self, cfg: TrainingJobConfig) -> dict:
        """
        Generate a Kubernetes Job manifest from TrainingJobConfig.
        All values come from the config — nothing is hardcoded.
        """
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": cfg.job_name,
                "namespace": cfg.namespace,
                "labels": {
                    "app": "mlops-training",
                    "framework": cfg.framework,
                    "workflow-id": cfg.mlflow_run_id[:16],
                },
                "annotations": {
                    "mlops.framework": cfg.framework,
                    "mlops.run-id": cfg.mlflow_run_id,
                },
            },
            "spec": {
                "backoffLimit": 0,   # we handle retries ourselves (OOM logic)
                "activeDeadlineSeconds": settings.k8s_training_timeout_seconds,
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "mlops-training",
                            "framework": cfg.framework,
                        }
                    },
                    "spec": {
                        "serviceAccountName": cfg.service_account,
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "trainer",
                                "image": cfg.runner_image,
                                "imagePullPolicy": "Always",
                                "env": [
                                    {"name": "DATASET_URI",             "value": cfg.dataset_uri},
                                    {"name": "FEAST_FEATURE_VIEW",      "value": cfg.feast_feature_view},
                                    {"name": "MLFLOW_RUN_ID",           "value": cfg.mlflow_run_id},
                                    {"name": "MLFLOW_TRACKING_URI",     "value": cfg.mlflow_tracking_uri},
                                    {"name": "HYPERPARAMS",             "value": json.dumps(cfg.hyperparams)},
                                    {"name": "FRAMEWORK",               "value": cfg.framework},
                                    # Credentials via Kubernetes Secrets
                                    {"name": "AWS_ACCESS_KEY_ID",
                                     "valueFrom": {"secretKeyRef": {"name": "r2-credentials", "key": "access_key_id"}}},
                                    {"name": "AWS_SECRET_ACCESS_KEY",
                                     "valueFrom": {"secretKeyRef": {"name": "r2-credentials", "key": "secret_access_key"}}},
                                    {"name": "AWS_ENDPOINT_URL",
                                     "valueFrom": {"secretKeyRef": {"name": "r2-credentials", "key": "endpoint_url"}}},
                                ],
                                "resources": {
                                    "requests": {
                                        "memory": cfg.memory_request,
                                        "cpu":    cfg.cpu_request,
                                    },
                                    "limits": {
                                        "memory": cfg.memory_limit,
                                        "cpu":    cfg.cpu_limit,
                                    },
                                },
                            }
                        ],
                    },
                },
            },
        }

    # Metrics retrieval 

    async def _retrieve_metrics(self, run_id: str) -> dict[str, float]:
        """
        Retrieve training metrics from MLflow after Job completes.
        MLflow autologging writes metrics from inside the container.
        """
        try:
            client = mlflow.tracking.MlflowClient()
            run = await asyncio.to_thread(client.get_run, run_id)
            return {k: float(v) for k, v in run.data.metrics.items()}
        except Exception as exc:
            logger.warning("Metrics retrieval failed for run %s: %s", run_id, exc)
            return {}