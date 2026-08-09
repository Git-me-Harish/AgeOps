"""
Project-wide settings loaded from environment variables.
Usage:
    from configs.settings import settings
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

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
    app_env: str = "development"          # development | staging | production
    log_level: str = "INFO"
    debug: bool = False

    # MLflow 
    mlflow_tracking_uri: str = "http://mlflow-tracking:5000"
    mlflow_experiment_name: str = "multi-agent-mlops"
    mlflow_artifact_root: str = "s3://mlflow-artifacts/mlflow"

    # Cloudflare R2  (S3-compatible) 
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_endpoint_url: str = ""          # https://<ACCOUNT_ID>.r2.cloudflarestorage.com
    r2_bucket_name: str = "mlflow-artifacts"
    r2_region: str = "auto"

    # Neon PostgreSQL 
    database_url: str = ""             # postgresql+asyncpg://user:pass@host/db?sslmode=require
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Kubernetes 
    k8s_namespace: str = "mlops"
    k8s_in_cluster: bool = True        # False when running locally

    # LLM Provider 
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"     # cost-efficient default
    llm_temperature: float = 0.0       # deterministic agents
    llm_max_tokens: int = 4096

    # GitHub OAuth 
    github_client_id: str = ""
    github_client_secret: str = ""
    oauth2_cookie_secret: str = ""

    # Agent Safety Limits 
    agent_max_iterations: int = 20
    agent_max_tool_calls: int = 50
    agent_task_timeout_seconds: int = 120
    agent_max_concurrent: int = 10

    # Monitoring 
    prometheus_port: int = 9090
    grafana_url: str = "http://grafana:3000"
    loki_url: str = "http://loki:3100"

    # Budget Thresholds 
    r2_storage_alert_gb: float = 8.0   # alert at 80 % of 10 GB free tier
    neon_storage_alert_mb: float = 400.0  # alert at 80 % of 0.5 GB free tier

    # Cloudflare (for budget monitoring) 
    cloudflare_api_token: str = ""
    cloudflare_account_id: str = ""

    # Neon API 
    neon_api_key: str = ""
    neon_project_id: str = ""

    # RL Agent 
    rl_model_path: str = "/models/rl_optimizer"
    rl_training_timesteps: int = 10_000

    # Feature Store (Feast) 
    feast_repo_path: str = "/feast"

    # OTEL Tracing 
    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
