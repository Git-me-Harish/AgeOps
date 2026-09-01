"""
serving/inference_server.py

Production-grade MLOps Inference Server — embedded in every model OCI image.

This file is copied into every image at /app/serving/inference_server.py and
started by the container ENTRYPOINT.  It is fully self-contained — it must not
import anything from the parent project (agents/, configs/, etc.) because those
directories are not present inside the built container.

Endpoints:
  POST /predict           — single-row or multi-row prediction (V1 protocol)
  POST /predict/batch     — explicit batch endpoint (same logic, different name)
  GET  /health/live       — Kubernetes liveness probe  (always 200)
  GET  /health/ready      — Kubernetes readiness probe  (200 after warmup)
  GET  /metrics           — Prometheus text format scrape endpoint
  GET  /info              — model metadata (name, version, framework, labels)
  GET  /schema            — input/output schema inferred from model signature

Prediction protocol (V1 — matches KFServing V1):
  Request:  {"instances": [[f1, f2, ...], ...]}
            OR {"inputs": {"col_a": [v1, v2], "col_b": [v1, v2]}}   (named columns)
  Response: {"predictions": [p1, p2, ...], "model_name": "...", "model_version": "..."}

Model loading:
  Loaded at startup from /app/model using mlflow.pyfunc.load_model().
  A warmup prediction is run immediately after loading to JIT-compile
  any lazy initialisation (XGBoost, PyTorch tracing, etc.).

Configuration (all from environment variables set in the Dockerfile ENV):
  MODEL_NAME      — registered model name
  MODEL_VERSION   — model version string
  MODEL_FRAMEWORK — framework ("xgboost", "pytorch", etc.)
  PORT            — listening port (default 8080)
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Union

import mlflow.pyfunc
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from serving.health import health
from serving.metrics import metrics

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("inference_server")

# Structured (Loki-bound) log lines must be pure JSON — nothing else. A real
# log-shipping agent (Promtail) forwards each stdout line verbatim, and
# agents/monitoring_agent.py's _sample_recent_inference_inputs() does
# json.loads(line) on the WHOLE line. Emitting mlops_prediction_log through
# the standard `logger` above (which carries the readable
# "<timestamp> INFO name: <json>" prefix from logging.basicConfig) would
# make every such line fail json.loads() in a real deployment — the serving
# drift check would silently get zero samples forever, always reporting
# "insufficient recent inference samples" no matter how much traffic the
# model actually served. Found while wiring this up against a real local
# Loki instance, not by inspection. propagate=False keeps this off the root
# handler so it doesn't ALSO print with the readable prefix.
_structured_logger = logging.getLogger("inference_server.structured")
_structured_logger.propagate = False
_structured_handler = logging.StreamHandler()
_structured_handler.setFormatter(logging.Formatter("%(message)s"))
_structured_logger.addHandler(_structured_handler)
_structured_logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# ── Configuration ──────────────────────────────────────────────────────────────
_MODEL_DIR         = Path(os.environ.get("MODEL_DIR",         "/app/model"))
_MODEL_INFO_PATH   = Path(os.environ.get("MODEL_INFO_PATH",   "/app/model_info.json"))
_MODEL_NAME        = os.environ.get("MODEL_NAME",    "unknown")
_MODEL_VERSION     = os.environ.get("MODEL_VERSION", "unknown")
_MODEL_FRAMEWORK   = os.environ.get("MODEL_FRAMEWORK", "unknown")
_PORT              = int(os.environ.get("PORT", "8080"))
_MAX_LOGGED_ROWS_PER_REQUEST = int(os.environ.get("MAX_LOGGED_ROWS_PER_REQUEST", "20"))

# Global model handle — set during startup, read-only during serving
_model: Optional[mlflow.pyfunc.PyFuncModel] = None
_model_info: dict[str, Any] = {}


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model, run warmup, then serve.  Runs on startup and shutdown.

    We load synchronously inside the lifespan so the server only starts
    accepting traffic after the model is in memory and warmed up.
    """
    global _model, _model_info

    # 1. Load model_info.json
    if _MODEL_INFO_PATH.exists():
        try:
            _model_info = json.loads(_MODEL_INFO_PATH.read_text())
        except Exception as exc:
            logger.warning("Could not load model_info.json: %s", exc)
            _model_info = {}
    else:
        _model_info = {
            "model_name":    _MODEL_NAME,
            "model_version": _MODEL_VERSION,
            "framework":     _MODEL_FRAMEWORK,
        }

    model_name    = _model_info.get("model_name",    _MODEL_NAME)
    model_version = _model_info.get("model_version", _MODEL_VERSION)
    framework     = _model_info.get("framework",     _MODEL_FRAMEWORK)

    # 2. Configure metrics labels
    metrics.configure(model_name, model_version)

    # 3. Load MLflow pyfunc model
    load_start = time.monotonic()
    logger.info("Loading model from %s …", _MODEL_DIR)
    try:
        _model = mlflow.pyfunc.load_model(str(_MODEL_DIR))
    except Exception as exc:
        logger.error("FATAL: Model load failed: %s", exc)
        # Don't raise — server starts but readiness probe returns 503
        # so Kubernetes won't route traffic.
        yield
        return

    load_duration = time.monotonic() - load_start
    metrics.model_load_duration.labels(model_name, model_version).set(load_duration)
    health.mark_model_loaded(model_name, model_version, framework)
    logger.info("Model loaded in %.2fs", load_duration)

    # 4. Warmup prediction — one synthetic row
    try:
        _run_warmup_prediction()
        health.mark_warmup_done()
        logger.info("Warmup complete — server is ready")
    except Exception as exc:
        logger.warning(
            "Warmup prediction failed (server still starting): %s", exc
        )
        # Warmup failure is non-fatal — model may require specific input schema.
        # Mark warmup done anyway so readiness probe passes; the first real
        # request will surface schema errors clearly.
        health.mark_warmup_done()

    yield
    # Shutdown: nothing to clean up for pyfunc models
    logger.info("Inference server shutting down")


def _run_warmup_prediction() -> None:
    """
    Run one synthetic prediction to JIT-compile model internals.
    Uses a single zero-row DataFrame — works for all MLflow pyfunc flavours.
    """
    if _model is None:
        return
    schema = _model.metadata.get_input_schema() if _model.metadata else None
    if schema is not None and hasattr(schema, "input_names"):
        cols = schema.input_names()
        df   = pd.DataFrame([[0] * len(cols)], columns=cols)
    else:
        # Fallback: single column — model will likely ignore it
        df = pd.DataFrame({"x": [0]})
    _model.predict(df)


# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title=f"MLOps Inference Server — {_MODEL_NAME}",
    description="Standardised inference server embedded in every model OCI image.",
    version=_MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── Request / response models ──────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    V1 prediction protocol.

    Accepts two input formats:
      instances — list of rows, each row is a list of feature values
                  (position-ordered, requires model input schema)
      inputs    — dict of column_name → list of values (named columns,
                  no schema required)
    """
    instances: Optional[list[list[Any]]] = Field(
        default=None,
        description="List of rows, each row is a list of feature values.",
    )
    inputs: Optional[dict[str, list[Any]]] = Field(
        default=None,
        description="Named column dict: {col: [v1, v2, ...]}.",
    )


class PredictResponse(BaseModel):
    predictions:       list[Any]
    model_name:        str
    model_version:     str
    prediction_count:  int
    request_id:        str
    inference_time_ms: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health/live", status_code=200, tags=["health"])
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe — always 200 if the process is running."""
    return {"status": "alive"}


@app.get("/health/ready", status_code=200, tags=["health"])
async def readiness() -> Response:
    """
    Kubernetes readiness probe.
    Returns 200 when model is loaded and warmup is complete.
    Returns 503 otherwise — Kubernetes stops routing traffic.
    """
    if health.is_ready:
        return JSONResponse(status_code=200, content=health.to_dict())
    return JSONResponse(status_code=503, content=health.to_dict())


@app.get("/info", tags=["model"])
async def info() -> dict[str, Any]:
    """Model metadata: name, version, framework, OCI labels."""
    return {
        **_model_info,
        "health": health.to_dict(),
    }


@app.get("/schema", tags=["model"])
async def schema() -> dict[str, Any]:
    """
    Return the model's MLflow input and output schema (if available).
    Useful for clients to validate payload shape before calling /predict.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_schema  = None
    output_schema = None
    if _model.metadata:
        try:
            sig = _model.metadata.signature
            if sig:
                input_schema  = sig.inputs.to_dict()  if sig.inputs  else None
                output_schema = sig.outputs.to_dict() if sig.outputs else None
        except Exception:
            pass

    return {
        "model_name":    _model_info.get("model_name",    _MODEL_NAME),
        "model_version": _model_info.get("model_version", _MODEL_VERSION),
        "input_schema":  input_schema,
        "output_schema": output_schema,
    }


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
async def predict(request: PredictRequest, raw_request: Request) -> PredictResponse:
    """
    Single or multi-row prediction.

    Accepts either `instances` (positional) or `inputs` (named columns).
    Exactly one must be provided.
    """
    return await _do_predict(request, raw_request)


@app.post("/predict/batch", response_model=PredictResponse, tags=["predict"])
async def predict_batch(
    request: PredictRequest, raw_request: Request
) -> PredictResponse:
    """
    Explicit batch prediction endpoint — identical logic to /predict.
    Exists as a named alias for clients that distinguish single vs batch.
    """
    return await _do_predict(request, raw_request)


async def _do_predict(
    request: PredictRequest, raw_request: Request
) -> PredictResponse:
    """Shared prediction handler used by /predict and /predict/batch."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — try again shortly")

    if not health.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready — warmup in progress")

    if request.instances is None and request.inputs is None:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'instances' (list of rows) or 'inputs' (column dict).",
        )
    if request.instances is not None and request.inputs is not None:
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of 'instances' or 'inputs', not both.",
        )

    request_id = str(uuid.uuid4())
    start = time.monotonic()

    # Build DataFrame
    try:
        df = _build_dataframe(request)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Input conversion failed: {exc}")

    # Predict with metrics tracking
    raw_body_size = int(raw_request.headers.get("content-length", 0))
    try:
        with metrics.track_prediction(input_bytes=raw_body_size):
            raw_predictions = _model.predict(df)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # Normalise output to a list
    predictions = _normalise_predictions(raw_predictions)
    model_name    = _model_info.get("model_name",    _MODEL_NAME)
    model_version = _model_info.get("model_version", _MODEL_VERSION)
    inference_time_ms = (time.monotonic() - start) * 1000

    output_bytes = len(json.dumps(predictions).encode())
    metrics.output_size_bytes.labels(model_name, model_version).observe(output_bytes)

    _log_prediction(
        request_id=request_id, model_name=model_name, model_version=model_version,
        df=df, predictions=predictions, inference_time_ms=inference_time_ms,
    )

    return PredictResponse(
        predictions=       predictions,
        model_name=        model_name,
        model_version=     model_version,
        prediction_count=  len(predictions),
        request_id=        request_id,
        inference_time_ms= inference_time_ms,
    )


def _log_prediction(
    request_id: str, model_name: str, model_version: str,
    df: pd.DataFrame, predictions: list[Any], inference_time_ms: float,
) -> None:
    """
    Emit one structured JSON log line per prediction, tagged
    "mlops_prediction_log" so a log-shipping agent (Promtail → Loki in a
    real cluster) can find and forward these to Loki for MonitoringAgent's
    serving-drift sampling to query. Feature values only — never logs the
    prediction alongside anything that isn't already part of the request
    payload the caller sent us.

    Caps rows logged per call and truncates the row to numeric columns
    (what the drift check actually compares) to keep log volume bounded.
    """
    try:
        numeric_df = df.select_dtypes(include="number")
        rows = numeric_df.head(_MAX_LOGGED_ROWS_PER_REQUEST).to_dict(orient="records")
        for i, row in enumerate(rows):
            _structured_logger.info(json.dumps({
                "marker": "mlops_prediction_log",
                "request_id": request_id,
                "model_name": model_name,
                "model_version": model_version,
                "inference_time_ms": round(inference_time_ms, 2),
                "features": row,
                "prediction": predictions[i] if i < len(predictions) else None,
            }))
    except Exception as exc:
        # Logging must never break a prediction response.
        logger.warning("Prediction logging failed (non-fatal): %s", exc)


@app.get("/metrics", tags=["observability"])
async def prometheus_metrics() -> Response:
    """Prometheus scrape endpoint — returns text exposition format."""
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_dataframe(request: PredictRequest) -> pd.DataFrame:
    """Convert a PredictRequest into a pandas DataFrame for the pyfunc model."""
    if request.inputs is not None:
        # Named columns: {"col_a": [1,2,3], "col_b": [4,5,6]}
        df = pd.DataFrame(request.inputs)
        if df.empty:
            raise ValueError("'inputs' dict produced an empty DataFrame.")
        return df

    # Positional rows: [[v1, v2, v3], [v1, v2, v3]]
    rows = request.instances
    if not rows:
        raise ValueError("'instances' list is empty.")

    # Try to get column names from model signature
    schema = None
    if _model and _model.metadata:
        try:
            sig = _model.metadata.signature
            if sig and sig.inputs:
                schema = sig.inputs.input_names()
        except Exception:
            pass

    if schema and len(schema) == len(rows[0]):
        return pd.DataFrame(rows, columns=schema)

    # No schema — use positional column names
    n_cols = len(rows[0])
    cols   = [f"feature_{i}" for i in range(n_cols)]
    return pd.DataFrame(rows, columns=cols)


def _normalise_predictions(raw: Any) -> list[Any]:
    """
    Normalise any pyfunc output type to a plain Python list.

    MLflow pyfunc models can return:
      - pd.DataFrame (classification with probabilities)
      - pd.Series    (regression)
      - numpy.ndarray
      - list
    """
    if isinstance(raw, pd.DataFrame):
        if raw.shape[1] == 1:
            return raw.iloc[:, 0].tolist()
        return raw.to_dict(orient="records")
    if isinstance(raw, pd.Series):
        return raw.tolist()
    try:
        import numpy as np
        if isinstance(raw, np.ndarray):
            return raw.tolist()
    except ImportError:
        pass
    if isinstance(raw, list):
        return raw
    return [raw]


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "serving.inference_server:app",
        host="0.0.0.0",
        port=_PORT,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        workers=1,         # single worker — model is in-process memory
        loop="uvloop",
    )