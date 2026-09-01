"""
mcp_servers/packaging_server.py

Phase 4 — OCI Packaging MCP Server.

Exposes Phase 4 packaging operations as MCP-compatible tools for agents.
Follows the exact same FastAPI + rate-limit + tool-discovery pattern as
registry_server.py (port 8002).  This server runs on port 8003.

Registered tools (GET /tools):
  build_model_image     — full pipeline: Dockerfile → Kaniko → Trivy → SBOM → Cosign
  get_build_status      — poll an oci_build_jobs row by ID
  get_trivy_scan        — retrieve latest Trivy scan results for an image digest
  get_sbom              — retrieve SBOM record (R2 URI + package count) for an image
  get_oci_image         — retrieve oci_images row for a (model_name, version) pair
  list_build_jobs       — list build jobs for a model (latest first)

Run alongside registry_server and mlflow_server:
  uvicorn mcp_servers.packaging_server:app --host 0.0.0.0 --port 8003
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import asyncpg
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agents.packaging.dockerfile_agent import DockerfileAgent, DockerfileAgentError, PackagingRequest
from configs.settings import settings

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url.replace("+asyncpg", ""),
            min_size=2,
            max_size=settings.database_pool_size,
            command_timeout=30,
        )
    return _pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    try:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url.replace("+asyncpg", ""),
            min_size=2,
            max_size=settings.database_pool_size,
            command_timeout=30,
        )
        logger.info("Packaging MCP server: DB pool initialised")
    except Exception as exc:
        logger.warning("Packaging MCP server: DB pool init failed (%s) — will retry", exc)
        _pool = None
    yield
    if _pool:
        await _pool.close()


app = FastAPI(
    title="Packaging MCP Server",
    description="MCP tools for Phase 4 OCI model packaging operations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiter ───────────────────────────────────────────────────────────────
_request_counts: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT  = 60
_RATE_WINDOW = 60


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    agent_id = request.headers.get("X-Agent-Id", "anonymous")
    now = time.time()
    calls = [t for t in _request_counts[agent_id] if t > now - _RATE_WINDOW]
    if len(calls) >= _RATE_LIMIT:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    _request_counts[agent_id] = calls + [now]
    return await call_next(request)


# ── Tool discovery ─────────────────────────────────────────────────────────────

@app.get("/tools")
async def list_tools():
    return {
        "tools": [
            {
                "name":        "build_model_image",
                "endpoint":    "/packaging/v1/build",
                "version":     "1",
                "description": (
                    "Run the full OCI packaging pipeline for a Staging model: "
                    "generate Dockerfile, run Kaniko build, Trivy CVE scan, "
                    "SBOM generation, and Cosign signing. Blocks until complete."
                ),
            },
            {
                "name":        "get_build_status",
                "endpoint":    "/packaging/v1/build/{build_job_id}",
                "version":     "1",
                "description": "Poll an oci_build_jobs row by ID. Returns status, duration, and log URI.",
            },
            {
                "name":        "get_trivy_scan",
                "endpoint":    "/packaging/v1/scan/{image_digest}",
                "version":     "1",
                "description": "Get the latest Trivy CVE scan result for an image digest.",
            },
            {
                "name":        "get_sbom",
                "endpoint":    "/packaging/v1/sbom/{model_name}/{model_version}",
                "version":     "1",
                "description": "Get SBOM record (R2 URI and package count) for a model version.",
            },
            {
                "name":        "get_oci_image",
                "endpoint":    "/packaging/v1/image/{model_name}/{model_version}",
                "version":     "1",
                "description": "Get the oci_images row for a (model_name, version) pair.",
            },
            {
                "name":        "list_build_jobs",
                "endpoint":    "/packaging/v1/jobs/{model_name}",
                "version":     "1",
                "description": "List oci_build_jobs for a model (latest first, max 20).",
            },
        ]
    }


# ── Request schemas ────────────────────────────────────────────────────────────

class BuildRequest(BaseModel):
    model_name:    str
    model_version: str
    workflow_id:   Optional[str] = None
    serving_dir:   str = Field(default="serving", description="Path to serving/ relative to repo root.")


class BuildResponse(BaseModel):
    model_name:       str
    model_version:    str
    image_tag:        str
    image_digest:     str
    image_uri:        str
    sbom_uri:         str
    cosign_signed:    bool
    build_job_db_id:  int
    oci_image_db_id:  int
    duration_seconds: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/packaging/v1/build", response_model=BuildResponse)
async def build_model_image(request: BuildRequest) -> BuildResponse:
    """
    Trigger the full OCI packaging pipeline for a model.
    This is a synchronous long-running call (Kaniko build can take several minutes).
    Clients should set HTTP timeout to at least 1200s.
    """
    try:
        pool = await _get_pool()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")

    agent = DockerfileAgent(pool=pool)
    try:
        result = await agent.run(
            PackagingRequest(
                model_name=    request.model_name,
                model_version= request.model_version,
                workflow_id=   request.workflow_id,
                serving_dir=   request.serving_dir,
            )
        )
    except DockerfileAgentError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error in build pipeline")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    return BuildResponse(**result.model_dump())


@app.get("/packaging/v1/build/{build_job_id}")
async def get_build_status(build_job_id: int) -> dict[str, Any]:
    """Return the status row for one oci_build_jobs entry."""
    try:
        pool = await _get_pool()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, model_name, model_version, image_tag, status,
                   kaniko_job_name, log_uri, error_message,
                   duration_seconds, started_at, completed_at, created_at
            FROM oci_build_jobs WHERE id=$1
            """,
            build_job_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Build job {build_job_id} not found")

    return {
        "id":               row["id"],
        "model_name":       row["model_name"],
        "model_version":    row["model_version"],
        "image_tag":        row["image_tag"],
        "status":           row["status"],
        "kaniko_job_name":  row["kaniko_job_name"],
        "log_uri":          row["log_uri"],
        "error_message":    row["error_message"],
        "duration_seconds": row["duration_seconds"],
        "started_at":       row["started_at"].isoformat() if row["started_at"] else None,
        "completed_at":     row["completed_at"].isoformat() if row["completed_at"] else None,
        "created_at":       row["created_at"].isoformat() if row["created_at"] else None,
    }


@app.get("/packaging/v1/scan/{image_digest:path}")
async def get_trivy_scan(image_digest: str) -> dict[str, Any]:
    """Return the latest Trivy scan result for an image digest."""
    try:
        pool = await _get_pool()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT image_digest, image_tag, model_name, model_version,
                   critical_count, high_count, medium_count, low_count,
                   total_count, passed, scan_duration_ms, scanned_at
            FROM trivy_scan_results
            WHERE image_digest=$1
            ORDER BY scanned_at DESC LIMIT 1
            """,
            image_digest,
        )
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No Trivy scan found for digest {image_digest!r}",
        )
    return dict(row)


@app.get("/packaging/v1/sbom/{model_name}/{model_version}")
async def get_sbom(model_name: str, model_version: str) -> dict[str, Any]:
    """Return the SBOM record for the latest image of a model version."""
    try:
        pool = await _get_pool()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT image_digest, model_name, model_version,
                   format, r2_uri, package_count, syft_version, generated_at
            FROM sbom_records
            WHERE model_name=$1 AND model_version=$2
            ORDER BY generated_at DESC LIMIT 1
            """,
            model_name, model_version,
        )
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No SBOM found for {model_name} v{model_version}",
        )
    return {
        "image_digest":  row["image_digest"],
        "model_name":    row["model_name"],
        "model_version": row["model_version"],
        "format":        row["format"],
        "r2_uri":        row["r2_uri"],
        "package_count": row["package_count"],
        "syft_version":  row["syft_version"],
        "generated_at":  row["generated_at"].isoformat() if row["generated_at"] else None,
    }


@app.get("/packaging/v1/image/{model_name}/{model_version}")
async def get_oci_image(model_name: str, model_version: str) -> dict[str, Any]:
    """Return the oci_images row for a (model_name, version) pair."""
    try:
        pool = await _get_pool()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, model_name, model_version, image_tag,
                   image_digest, image_uri, registry_host,
                   base_image, labels, pushed_at
            FROM oci_images
            WHERE model_name=$1 AND model_version=$2
            ORDER BY pushed_at DESC LIMIT 1
            """,
            model_name, model_version,
        )
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No OCI image found for {model_name} v{model_version}",
        )
    return {
        "id":            row["id"],
        "model_name":    row["model_name"],
        "model_version": row["model_version"],
        "image_tag":     row["image_tag"],
        "image_digest":  row["image_digest"],
        "image_uri":     row["image_uri"],
        "registry_host": row["registry_host"],
        "base_image":    row["base_image"],
        "labels":        dict(row["labels"] or {}),
        "pushed_at":     row["pushed_at"].isoformat() if row["pushed_at"] else None,
    }


@app.get("/packaging/v1/jobs/{model_name}")
async def list_build_jobs(
    model_name: str,
    model_version: Optional[str] = None,
    limit: int = 20,
) -> dict[str, Any]:
    """List oci_build_jobs for a model (latest first)."""
    try:
        pool = await _get_pool()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")

    limit = min(max(limit, 1), 100)
    async with pool.acquire() as conn:
        if model_version:
            rows = await conn.fetch(
                """
                SELECT id, model_name, model_version, image_tag, status,
                       duration_seconds, error_message, created_at
                FROM oci_build_jobs
                WHERE model_name=$1 AND model_version=$2
                ORDER BY created_at DESC LIMIT $3
                """,
                model_name, model_version, limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, model_name, model_version, image_tag, status,
                       duration_seconds, error_message, created_at
                FROM oci_build_jobs
                WHERE model_name=$1
                ORDER BY created_at DESC LIMIT $2
                """,
                model_name, limit,
            )

    return {
        "model_name": model_name,
        "jobs": [
            {
                "id":               r["id"],
                "model_version":    r["model_version"],
                "image_tag":        r["image_tag"],
                "status":           r["status"],
                "duration_seconds": r["duration_seconds"],
                "error_message":    r["error_message"],
                "created_at":       r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ],
    }


@app.get("/health")
async def health():
    return {
        "status":    "ok" if _pool else "degraded",
        "db_pool":   "connected" if _pool else "not initialised",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)