"""
serving/metrics.py

Prometheus metrics for the MLOps inference server.

Exported metrics:
  mlops_predictions_total          counter   — total prediction requests by status
  mlops_prediction_duration_seconds histogram — end-to-end request latency
  mlops_model_load_duration_seconds gauge     — how long model loading took
  mlops_active_requests            gauge      — in-flight requests right now
  mlops_input_size_bytes           histogram  — serialised input payload size
  mlops_output_size_bytes          histogram  — serialised output payload size

Labels:
  model_name     — from MODEL_NAME env var
  model_version  — from MODEL_VERSION env var
  status         — "success" | "error"

Scraped by Prometheus at GET /metrics (standard prometheus_client text format).

Usage in inference_server.py:
    from serving.metrics import metrics
    with metrics.track_prediction():
        result = model.predict(df)
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Generator

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# ── Metric definitions ────────────────────────────────────────────────────────

# Use a dedicated registry so tests can create isolated instances
DEFAULT_REGISTRY = CollectorRegistry()

_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)
_SIZE_BUCKETS = (
    128, 512, 1024, 4096, 16384, 65536, 262144, 1048576,  # bytes
)


@dataclass
class _PrometheusMetrics:
    """
    Container for all Prometheus metric instruments.

    Instantiated once per process (module-level singleton `metrics`).
    Tests that need isolation can call PrometheusMetrics(registry=...) directly.
    """
    registry: CollectorRegistry = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.registry is None:
            self.registry = DEFAULT_REGISTRY

        lbl = ["model_name", "model_version"]

        self.predictions_total = Counter(
            "mlops_predictions_total",
            "Total prediction requests handled by the inference server.",
            lbl + ["status"],
            registry=self.registry,
        )
        self.prediction_duration = Histogram(
            "mlops_prediction_duration_seconds",
            "End-to-end prediction latency in seconds.",
            lbl,
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.model_load_duration = Gauge(
            "mlops_model_load_duration_seconds",
            "Time taken to load the MLflow pyfunc model on startup.",
            lbl,
            registry=self.registry,
        )
        self.active_requests = Gauge(
            "mlops_active_requests",
            "Number of prediction requests currently in flight.",
            lbl,
            registry=self.registry,
        )
        self.input_size_bytes = Histogram(
            "mlops_input_size_bytes",
            "Size of serialised prediction input payload in bytes.",
            lbl,
            buckets=_SIZE_BUCKETS,
            registry=self.registry,
        )
        self.output_size_bytes = Histogram(
            "mlops_output_size_bytes",
            "Size of serialised prediction output payload in bytes.",
            lbl,
            buckets=_SIZE_BUCKETS,
            registry=self.registry,
        )

        # Label values set at startup from environment / model_info.json
        self._model_name:    str = "unknown"
        self._model_version: str = "unknown"

    def configure(self, model_name: str, model_version: str) -> None:
        """Call once at startup after reading model_info.json."""
        self._model_name    = model_name
        self._model_version = model_version

    @contextlib.contextmanager
    def track_prediction(
        self, input_bytes: int = 0
    ) -> Generator[None, None, None]:
        """
        Context manager that tracks one prediction request end-to-end.

        Records:
          - active_requests gauge (increment on enter, decrement on exit)
          - input payload size
          - prediction duration histogram
          - predictions_total counter (success or error)

        Usage:
            with metrics.track_prediction(input_bytes=len(raw_body)):
                result = model.predict(df)
        """
        lbl = [self._model_name, self._model_version]
        self.active_requests.labels(*lbl).inc()
        if input_bytes > 0:
            self.input_size_bytes.labels(*lbl).observe(input_bytes)

        t0     = time.perf_counter()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.perf_counter() - t0
            self.prediction_duration.labels(*lbl).observe(duration)
            self.predictions_total.labels(*lbl, status).inc()
            self.active_requests.labels(*lbl).dec()

    def render(self) -> tuple[bytes, str]:
        """Return (body_bytes, content_type) for the /metrics endpoint."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


# Module-level singleton — import and use directly in inference_server.py
metrics = _PrometheusMetrics()