# tests/conftest.py
"""
Shared pytest fixtures used across unit, integration, and security tests.
"""
from __future__ import annotations

import os
import pytest
import mlflow

# ── Force test environment before any app code is imported ───────────────────
os.environ.setdefault("APP_ENV",             "test")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")
os.environ.setdefault("DATABASE_URL",        "postgresql+asyncpg://mlops:mlops_dev@localhost:5432/mlops_test")
os.environ.setdefault("K8S_IN_CLUSTER",      "false")
os.environ.setdefault("R2_ENDPOINT_URL",     "https://fake.r2.endpoint")
os.environ.setdefault("R2_ACCESS_KEY_ID",    "test-key")
os.environ.setdefault("R2_SECRET_ACCESS_KEY","test-secret")
os.environ.setdefault("R2_BUCKET_NAME",      "test-bucket")


@pytest.fixture(autouse=True)
def reset_mlflow(tmp_path):
    """Each test gets its own SQLite MLflow backend."""
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("test-experiment")
    yield
    mlflow.end_run()


@pytest.fixture
def sample_state():
    """Minimal valid AgentState dict for testing individual agents."""
    return {
        "workflow_id":       "test-wf-0001",
        "current_stage":     "idle",
        "model_uri":         "runs:/fake_run_id/model",
        "dataset_uri":       "s3://test-bucket/data/sample.csv",
        "metrics":           {},
        "errors":            [],
        "trace_id":          "trace-abc",
        "agent_decisions":   [],
        "rl_recommendations": None,
        "awaiting_approval":  False,
        "approval_context":  None,
        "iteration_count":   0,
        "max_iterations":    20,
    }


@pytest.fixture
def sample_dataframe():
    """Small pandas DataFrame for data-agent tests."""
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((100, 5)), columns=["a", "b", "c", "d", "e"])
