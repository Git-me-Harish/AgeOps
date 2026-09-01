# tests/unit/test_inference_server_logging.py
"""
Regression test for a real bug found while wiring a local Loki instance up
against the packaged inference server: mlops_prediction_log lines were
emitted through the standard `logger` (carrying logging.basicConfig's
"<timestamp> INFO name: <message>" prefix), so in a real deployment
Promtail would ship a line that is NOT valid JSON, and
agents/monitoring_agent.py._sample_recent_inference_inputs()'s
json.loads(line) would fail on every single one — the serving-drift check
would silently get zero samples forever. Fixed with a dedicated
prefix-free structured logger (serving/inference_server.py's
_structured_logger, propagate=False).
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from serving.inference_server import _log_prediction, _structured_logger


def test_structured_log_line_is_pure_json(caplog):
    # _structured_logger has propagate=False (by design — see module
    # docstring), so pytest's caplog handler (attached to the root logger)
    # never sees its records unless we attach it directly here too.
    caplog.set_level(logging.INFO, logger="inference_server.structured")
    _structured_logger.addHandler(caplog.handler)
    try:
        df = pd.DataFrame([{"f0": 0.1, "f1": 0.2}])
        _log_prediction("req-1", "demo-model", "3", df, [1], 12.5)
    finally:
        _structured_logger.removeHandler(caplog.handler)

    structured_records = [r for r in caplog.records if r.name == "inference_server.structured"]
    assert len(structured_records) == 1
    line = structured_records[0].getMessage()

    # The critical assertion: the emitted message is standalone valid JSON —
    # no timestamp/level/logger-name prefix in front of it.
    payload = json.loads(line)
    assert payload["marker"] == "mlops_prediction_log"
    assert payload["model_name"] == "demo-model"
    assert payload["model_version"] == "3"
    assert payload["prediction"] == 1


def test_structured_logger_does_not_propagate_to_root(caplog):
    """propagate=False must hold, or the JSON line would ALSO print through
    the readable root formatter, defeating the fix for any handler that
    reads the root logger's own output."""
    assert _structured_logger.propagate is False
