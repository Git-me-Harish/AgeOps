"""
configs/telemetry.py

Phase 5 §5.1 — Application-layer observability: OpenTelemetry traces to
Tempo. opentelemetry-api/-sdk/-exporter-otlp-proto-grpc have been pinned in
requirements.txt since Phase 4, but nothing in the codebase ever configured
a TracerProvider or exported a span — this module is that wiring.

Usage:
    from configs.telemetry import get_tracer
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("orchestrator.run_workflow"):
        ...

init_tracing() is idempotent and safe to call from every process entry
point (orchestrator, MCP servers, scripts) — the first call configures the
global TracerProvider; later calls are no-ops.

Without a reachable OTLP collector (settings.otel_exporter_otlp_endpoint),
spans are still created and can be inspected in-process (e.g. via an
InMemorySpanExporter in tests) but are never exported anywhere — there is
no fallback to a fake "it worked" state; export failures are the
BatchSpanProcessor's problem to retry/drop, not something this module papers
over.
"""
from __future__ import annotations

import logging
from typing import Optional

from configs.settings import settings

logger = logging.getLogger(__name__)

_initialized = False


def init_tracing(service_name: Optional[str] = None) -> None:
    """Configure the global OTel TracerProvider once per process."""
    global _initialized
    if _initialized:
        return
    if not settings.otel_enabled:
        logger.info("OTel tracing disabled via settings.otel_enabled=False")
        _initialized = True
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        resource = Resource.create({"service.name": service_name or settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info(
            "OTel tracing initialised: service=%s endpoint=%s",
            service_name or settings.otel_service_name, settings.otel_exporter_otlp_endpoint,
        )
    except Exception as exc:
        # A missing/unreachable collector must not crash the process that's
        # trying to trace itself — spans just won't export anywhere.
        logger.warning("OTel tracing init failed (spans will not export): %s", exc)

    _initialized = True


def get_tracer(name: str):
    """Return a tracer, initialising the global provider on first use."""
    init_tracing()
    from opentelemetry import trace
    return trace.get_tracer(name)
