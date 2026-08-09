# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: dependency builder
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel cache
RUN pip install --upgrade pip wheel

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: production image
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim

LABEL org.opencontainers.image.title="mlops-orchestrator"
LABEL org.opencontainers.image.description="Multi-Agent MLOps Orchestrator + REST API"
LABEL org.opencontainers.image.source="https://github.com/Git-me-Harish/multi-agent-mlops"

# Security: non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd  --uid 1000 --gid appgroup --no-create-home appuser

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-built wheels from builder stage
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && \
    rm -rf /wheels

# Copy source (honour .dockerignore)
COPY agents/         ./agents/
COPY mcp_servers/    ./mcp_servers/
COPY configs/        ./configs/
COPY rl_agent/       ./rl_agent/
COPY scripts/        ./scripts/
COPY feast/          ./feast/
COPY policies/       ./policies/

# Set ownership
RUN chown -R appuser:appgroup /app

USER appuser

# Expose REST API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -sf http://localhost:8000/api/health || exit 1

# Default: start the REST API server (agents run as background tasks within it)
CMD ["uvicorn", "mcp_servers.api_server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
