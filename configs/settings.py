"""
configs/settings.py

Project-wide settings loaded from environment variables.

V2 additions:
  - Redis (Feast online store + LLM semantic cache)
  - Schema Registry toggle + drift/validation thresholds (configurable)
  - R2 report path for Evidently HTML artifacts
  - Ollama fallback for LLM Gateway (Phase 2)
  - pgvector / agent memory settings (Phase 2)
  - Kaniko / OCI image registry (Phase 4)
  - OTEL + Tempo (Phase 5)
  - Cosign / SBOM (Phase 4)

Usage:
    from configs.settings import settings
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application 
    app_name: str = "multi-agent-mlops"
    app_env: str = "development"   # development | test | staging | production
    log_level: str = "INFO"
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # MLflow 
    mlflow_tracking_uri: str = "http://mlflow-tracking:5000"
    mlflow_experiment_name: str = "multi-agent-mlops"
    mlflow_artifact_root: str = "s3://mlflow-artifacts/mlflow"

    # Cloudflare R2 (S3-compatible object storage) 
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_endpoint_url: str = ""       
    r2_bucket_name: str = "mlflow-artifacts"
    r2_region: str = "auto"
    # Sub-paths inside the bucket — change per environment
    r2_datasets_prefix: str = "datasets"
    r2_reports_prefix: str = "reports/evidently"
    r2_sbom_prefix: str = "security/sbom"
    r2_images_prefix: str = "images"   # Kaniko build context tarballs

    # Neon PostgreSQL 
    database_url: str = ""         # postgresql+asyncpg://user:pass@host/db?sslmode=require
    database_sync_url: str = ""    # postgresql://user:pass@host/db?sslmode=require (Alembic)
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Redis (Feast online store + Phase 2 LLM semantic cache) 
    redis_url: str = "redis://redis.mlops.svc.cluster.local:6379"
    redis_max_connections: int = 20
    # Semantic cache settings (Phase 2)
    semantic_cache_ttl_seconds: int = 3600
    semantic_cache_similarity_threshold: float = 0.92

    # Kubernetes 
    k8s_namespace: str = "mlops"
    k8s_in_cluster: bool = True              # False when running locally
    k8s_service_account: str = "training-agent"
    # Framework runner images — overridable per environment
    k8s_runner_sklearn: str = "your-dockerhub-user/mlops-runner-sklearn:latest"
    k8s_runner_xgboost: str = "your-dockerhub-user/mlops-runner-xgboost:latest"
    k8s_runner_pytorch: str = "your-dockerhub-user/mlops-runner-pytorch:latest"
    k8s_runner_huggingface: str = "your-dockerhub-user/mlops-runner-huggingface:latest"
    k8s_runner_custom: str = "your-dockerhub-user/mlops-runner-custom:latest"
    k8s_training_timeout_seconds: int = 3600
    k8s_training_memory_request: str = "1Gi"
    k8s_training_memory_limit: str = "4Gi"
    k8s_training_cpu_request: str = "500m"
    k8s_training_cpu_limit: str = "2000m"

    # LLM Provider (primary) 
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"       # cost-efficient primary
    llm_temperature: float = 0.0         # deterministic agents
    llm_max_tokens: int = 4096
    llm_token_budget_per_agent: int = 50_000   # hard limit per agent per workflow

    # Ollama fallback LLM (Phase 2 — local ARM VM) 
    ollama_base_url: str = "http://ollama.mlops.svc.cluster.local:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_seconds: int = 120
    # Which agent roles are allowed to fall back to Ollama
    ollama_allowed_roles: list[str] = Field(
        default=["planner", "monitoring", "governance"],
    )

    # Data validation thresholds 
    # Validation: Great Expectations suite pass rate
    validation_pass_threshold: float = 0.85   # block pipeline below this
    # Drift: Evidently share_of_drifted_columns
    drift_warn_threshold: float = 0.15        # WARNING alert
    drift_critical_threshold: float = 0.30    # CRITICAL + trigger retraining
    # Schema registry
    schema_registry_enabled: bool = True
    schema_registry_block_on_mismatch: bool = True  # hard block vs warn only

    # Feast Feature Store 
    feast_repo_path: str = "/feast"
    feast_offline_store_type: str = "file"    # "file" (local dev) | "r2" (production)
    feast_redis_connection_string: str = "redis://redis.mlops.svc.cluster.local:6379"

    # Monitoring 
    prometheus_url: str = "http://prometheus:9090"
    prometheus_port: int = 9090
    grafana_url: str = "http://grafana:3000"
    loki_url: str = "http://loki:3100"
    # Monitoring agent loop
    monitoring_interval_seconds: int = 60
    latency_p99_threshold_ms: float = 500.0
    accuracy_drop_threshold: float = 0.05     # trigger retraining if drops > 5%
    monitoring_sample_window_minutes: int = 15   # Loki query window per tick
    monitoring_accuracy_window_hours: int = 24   # ground-truth label lookback window
    monitoring_min_sample_size: int = 30         # below this, drift check is inconclusive
    loki_service_label: str = "app"              # Promtail label carrying the KServe service name
    monitoring_max_recent_samples: int = 2000    # cap on rows pulled from Loki per tick

    # OCI Image Registry (Phase 4) 
    image_registry_host: str = "docker.io"    # docker.io | ghcr.io
    image_registry_namespace: str = "your-dockerhub-user"
    image_registry_username: str = ""
    image_registry_password: str = ""
    kaniko_image: str = "gcr.io/kaniko-project/executor:latest"
    kaniko_cache_enabled: bool = True
    # CVE gate: block images with this many CRITICAL findings
    trivy_critical_cve_limit: int = 0
    trivy_high_cve_limit: int = 5
    cosign_enabled: bool = True

    # OPA Policy Engine (Phase 2) 
    # Sidecar mode: OPA runs on localhost (same pod)
    opa_endpoint: str = "http://127.0.0.1:8181/v1/data"
    opa_policy_base_path: str = "kubernetes/admission"
    opa_timeout_seconds: float = 1.0    # < 1ms expected for sidecar

    # GitHub OAuth (UI auth)
    github_client_id: str = ""
    github_client_secret: str = ""
    oauth2_cookie_secret: str = ""
    nextauth_secret: str = ""
    nextauth_url: str = "http://localhost:3000"

    # CORS — all 5 FastAPI servers previously hardcoded allow_origins=["*"]
    # directly in code. Harmless in the sense that these servers are meant
    # to be reached only through the Next.js BFF (never directly by a
    # browser), but that assumption has no real enforcement without the
    # matching NetworkPolicy (configs/kubernetes/network-policies.yaml) —
    # settings-driven so it can be tightened per environment without a
    # code change.
    cors_allowed_origins: list[str] = Field(default=["http://localhost:3000"])

    # Internal MCP server URLs (Phase 6 UI gateway) — same in-cluster DNS
    # pattern as mlflow_tracking_uri; overridable for local dev where each
    # server is run on localhost via uvicorn.
    orchestrator_api_url: str = "http://orchestrator-agent.mlops.svc.cluster.local:8000"
    registry_server_url: str = "http://registry-server.mlops.svc.cluster.local:8002"
    packaging_server_url: str = "http://packaging-server.mlops.svc.cluster.local:8003"
    monitoring_server_url: str = "http://monitoring-server.mlops.svc.cluster.local:8004"

    # Model registry (Phase 3)
    mlflow_registered_model_name: str = "mlops-model"
    git_repo_url: str = "https://github.com/your-org/multi-agent-mlops"

    # Agent Safety Limits 
    agent_max_iterations: int = 20
    agent_max_tool_calls: int = 50
    agent_task_timeout_seconds: int = 120
    agent_max_concurrent: int = 10

    # Budget Thresholds (free-tier guards) 
    r2_storage_alert_gb: float = 8.0       # alert at 80% of 10 GB free tier
    neon_storage_alert_mb: float = 400.0   # alert at 80% of 0.5 GB free tier
    neon_vector_max_age_days: int = 90     # prune embeddings older than this

    # Cloudflare API (budget monitoring) 
    cloudflare_api_token: str = ""
    cloudflare_account_id: str = ""

    # Neon API 
    neon_api_key: str = ""
    neon_project_id: str = ""

    # RL Agent 
    rl_model_path: str = "/models/rl_optimizer"
    rl_training_timesteps: int = 10_000
    rl_history_window_days: int = 90
    # Reward weights — configurable, not hardcoded magic numbers, but not
    # self-tuned by a meta-learner either (explicitly deferred; see
    # rl_agent/rl_optimizer.py module docstring).
    rl_reward_weight_accuracy: float = 0.4
    rl_reward_weight_latency: float = 0.2
    rl_reward_weight_cost: float = 0.2
    rl_reward_weight_drift: float = 0.2
    rl_retrain_interval_days: int = 7     # retrain weekly on real MLflow data
    rl_lookback_days: int = 90            # historical run window for training

    # OTEL Tracing 
    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"
    otel_service_name: str = "multi-agent-mlops"
    otel_enabled: bool = True

    # WebSocket / Real-time (Phase 6) 
    ws_heartbeat_interval_seconds: int = 30
    ws_max_connections_per_client: int = 3

    # HITL Gate 
    hitl_approval_timeout_hours: int = 24   # auto-reject if not reviewed

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        """Enforce valid environment names."""
        allowed = {"development", "test", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"app_env must be one of {allowed}, got: {v!r}")
        return v
    
    @property
    def asyncpg_url(self) -> str:
        url = self.database_url
        url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
        url = url.replace("&channel_binding=require", "").replace("?channel_binding=require&", "?")
        return url

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def r2_configured(self) -> bool:
        """True when R2 credentials are present — controls real vs local-dev paths."""
        return bool(self.r2_endpoint_url and self.r2_access_key_id)

    @property
    def k8s_runner_image_map(self) -> dict[str, str]:
        """Mapping from framework name to Kubernetes runner image."""
        return {
            "sklearn": self.k8s_runner_sklearn,
            "xgboost": self.k8s_runner_xgboost,
            "pytorch": self.k8s_runner_pytorch,
            "huggingface": self.k8s_runner_huggingface,
            "custom": self.k8s_runner_custom,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
