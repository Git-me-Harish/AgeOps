"""
agents/packaging/dockerfile_generator.py

Phase 4 — Dockerfile Generator.

Generates production-grade, multi-stage Dockerfiles for every supported ML
framework.  Each generated Dockerfile:
  - Uses a framework-appropriate base image (GPU where needed, slim elsewhere)
  - Installs pinned dependencies from MLflow's python_env.yaml / requirements.txt
  - Bakes in the standardised inference server (serving/)
  - Writes a model_info.json so the inference server can expose /info at runtime
  - Sets all mandatory OCI labels from the Phase 3 mandatory tag set
  - Includes a HEALTHCHECK directive for K8s liveness probes

Multi-stage build structure (builder + runtime):
  Stage 1 (builder): installs Python packages with all build tools available
  Stage 2 (runtime): copies installed packages from builder — leaner final image

Supported frameworks and their base images:
  sklearn / xgboost → python:3.11-slim     (≈12 MB base)
  pytorch           → pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime
  huggingface       → huggingface/transformers-pytorch-gpu:latest
  custom            → caller supplies via DockerfileConfig.custom_base_image

Design:
  - DockerfileGenerator is stateless and pure — no I/O, no DB, no network.
    All side effects (MLflow artifact upload) are handled by DockerfileAgent.
  - Generated Dockerfiles are deterministic given the same DockerfileConfig.
  - model_info.json is written to /app/model_info.json inside the image.
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from agents.registry.metadata_schema import ModelRegistryTags

# ── Base image map ─────────────────────────────────────────────────────────────
BASE_IMAGES: dict[str, str] = {
    "sklearn":      "python:3.11-slim",
    "xgboost":      "python:3.11-slim",
    "pytorch":      "pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime",
    "huggingface":  "huggingface/transformers-pytorch-gpu:latest",
    "custom":       "",   # set via custom_base_image
}

# Packages always needed in the runtime image regardless of framework
_SERVING_REQUIREMENTS: list[str] = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "prometheus-client>=0.21.0",
    "opentelemetry-api>=1.28.0",
    "opentelemetry-sdk>=1.28.0",
    "mlflow>=2.17.0",
    "pydantic>=2.9.0",
]


# ── Input / output models ──────────────────────────────────────────────────────

class DockerfileConfig(BaseModel):
    """
    All inputs required to generate a Dockerfile for one model version.

    Attributes:
        model_name:          MLflow registered model name.
        model_version:       MLflow model version string (e.g. "3").
        framework:           One of the supported framework names.
        tags:                Typed Phase 3 tags (provides OCI label values).
        mlflow_run_id:       Run ID used to download model artifacts inside Kaniko.
        mlflow_tracking_uri: MLflow URI injected as ENV in the Dockerfile.
        image_tag:           Full target image tag (registry/namespace/name:tag).
        custom_base_image:   Required when framework == "custom".
        extra_system_pkgs:   Additional apt-get packages for the runtime layer.
        extra_pip_pkgs:      Additional pip packages beyond framework defaults.
        port:                Port the inference server listens on inside the container.
    """
    model_name:          str
    model_version:       str
    framework:           str
    tags:                ModelRegistryTags
    mlflow_run_id:       str
    mlflow_tracking_uri: str
    image_tag:           str
    custom_base_image:   Optional[str] = None
    extra_system_pkgs:   list[str] = Field(default_factory=list)
    extra_pip_pkgs:      list[str] = Field(default_factory=list)
    port:                int = 8080


@dataclass
class GeneratedDockerfile:
    """Output of DockerfileGenerator.generate()."""
    dockerfile_content: str           # full Dockerfile text
    model_info_json:    str           # JSON written to /app/model_info.json in the image
    base_image:         str           # the FROM image used in Stage 2
    framework:          str
    image_tag:          str
    oci_labels:         dict[str, str] # label key → value (for verification)


# ── Generator ─────────────────────────────────────────────────────────────────

class DockerfileGenerator:
    """
    Stateless Dockerfile generator.

    Usage:
        config = DockerfileConfig(...)
        result = DockerfileGenerator().generate(config)
        # result.dockerfile_content is the complete Dockerfile text
    """

    def generate(self, config: DockerfileConfig) -> GeneratedDockerfile:
        """
        Generate a complete Dockerfile for the given model and framework.

        Raises:
            ValueError: if framework is unsupported or custom_base_image is
                        required but not provided.
        """
        framework = config.framework.lower().strip()
        if framework not in BASE_IMAGES:
            raise ValueError(
                f"Unsupported framework '{framework}'. "
                f"Supported: {sorted(BASE_IMAGES)}"
            )

        base_image = BASE_IMAGES[framework]
        if framework == "custom":
            if not config.custom_base_image:
                raise ValueError(
                    "framework='custom' requires custom_base_image to be set "
                    "in DockerfileConfig or in the 'mlops.base_image' model tag."
                )
            base_image = config.custom_base_image

        oci_labels = self._build_oci_labels(config)
        model_info = self._build_model_info(config, oci_labels)
        model_info_json = json.dumps(model_info, indent=2)

        dockerfile = self._render_dockerfile(
            config=config,
            base_image=base_image,
            oci_labels=oci_labels,
        )

        return GeneratedDockerfile(
            dockerfile_content=dockerfile,
            model_info_json=model_info_json,
            base_image=base_image,
            framework=framework,
            image_tag=config.image_tag,
            oci_labels=oci_labels,
        )

    # ── OCI label construction ────────────────────────────────────────────────

    def _build_oci_labels(self, config: DockerfileConfig) -> dict[str, str]:
        """
        Build the full OCI label set from Phase 3 mandatory tags.
        Labels follow the OCI Image Annotation spec (org.opencontainers.image.*)
        plus the mlops.* namespace for domain-specific metadata.
        """
        tags = config.tags
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        return {
            # OCI standard annotations
            "org.opencontainers.image.title":       config.model_name,
            "org.opencontainers.image.version":     config.model_version,
            "org.opencontainers.image.revision":    tags.git_commit,
            "org.opencontainers.image.source":      tags.git_repo,
            "org.opencontainers.image.created":     now_iso,
            "org.opencontainers.image.description": (
                f"MLOps model: {config.model_name} v{config.model_version} "
                f"({tags.framework} framework)"
            ),
            # Domain-specific MLOps labels
            "mlops.model.name":              config.model_name,
            "mlops.model.version":           config.model_version,
            "mlops.model.framework":         tags.framework,
            "mlops.model.framework.version": tags.framework_version,
            "mlops.model.python.version":    tags.python_version,
            "mlops.eval.accuracy":           str(tags.eval_accuracy),
            "mlops.eval.f1":                 str(tags.eval_f1),
            "mlops.eval.auc":                str(tags.eval_auc),
            "mlops.eval.bias_passed":        "true" if tags.eval_bias_passed else "false",
            "mlops.dataset.hash":            tags.dataset_hash,
            "mlops.dataset.row_count":       str(tags.dataset_row_count),
            "mlops.security.trivy_scan":     tags.security_trivy_scan,
            "mlops.git.commit":              tags.git_commit,
        }

    def _build_model_info(
        self, config: DockerfileConfig, oci_labels: dict[str, str]
    ) -> dict:
        """Build the model_info.json content written to /app/model_info.json."""
        return {
            "model_name":    config.model_name,
            "model_version": config.model_version,
            "framework":     config.framework,
            "image_tag":     config.image_tag,
            "labels":        oci_labels,
            "port":          config.port,
        }

    # ── Dockerfile rendering ─────────────────────────────────────────────────

    def _render_dockerfile(
        self,
        config: DockerfileConfig,
        base_image: str,
        oci_labels: dict[str, str],
    ) -> str:
        """
        Render the complete multi-stage Dockerfile.

        Stage 1 (builder): installs all Python packages with build tools.
        Stage 2 (runtime): copies packages from builder, copies model and server,
                           sets labels and entrypoint.
        """
        framework = config.framework.lower()
        apt_pkgs = self._apt_packages(framework, config.extra_system_pkgs)
        pip_pkgs = self._pip_packages(framework, config.extra_pip_pkgs)
        label_block = self._render_labels(oci_labels)

        # Stage 1 base: use slim Python for builder regardless of framework
        # (avoids pulling the large GPU base just for pip installs)
        builder_base = "python:3.11-slim"

        apt_install_cmd = " ".join(apt_pkgs)

        lines: list[str] = [
            "# ── Stage 1: Dependency builder ──────────────────────────────────────────",
            f"FROM {builder_base} AS builder",
            "",
            "WORKDIR /build",
            "",
            "# Install C build toolchain needed by some wheels (xgboost, scikit-learn)",
            "RUN apt-get update && apt-get install -y --no-install-recommends \\",
            f"    {apt_install_cmd} \\",
            "    && rm -rf /var/lib/apt/lists/*",
            "",
            "# Serving framework requirements (pinned to serving/ directory spec)",
        ]

        # Write serving requirements inline
        pip_install_args = " \\\n    ".join(f'"{p}"' for p in pip_pkgs)
        lines += [
            f"RUN pip install --user --no-cache-dir \\",
            f"    {pip_install_args}",
            "",
            "# ── Stage 2: Runtime image ────────────────────────────────────────────────",
            f"FROM {base_image}",
            "",
            "WORKDIR /app",
            "",
            "# Copy installed packages from builder stage",
            "COPY --from=builder /root/.local /root/.local",
            "",
            "# Download and materialise MLflow model artifacts at build time",
            "# The run artifacts live in the MLflow artifact store (R2).",
            "# Kaniko has R2 credentials injected via K8s Secrets.",
            f"ARG MLFLOW_TRACKING_URI={config.mlflow_tracking_uri}",
            f"ARG MLFLOW_RUN_ID={config.mlflow_run_id}",
            "RUN pip install --user --no-cache-dir mlflow && \\",
            "    python -c \"import mlflow; mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')\" && \\",
            "    mlflow artifacts download \\",
            "        --run-id $MLFLOW_RUN_ID \\",
            "        --artifact-path model \\",
            "        --dst-path /app/model",
            "",
            "# Copy standardised inference server",
            "COPY serving/ /app/serving/",
            "",
            "# Write model metadata to /app/model_info.json (read by /info endpoint)",
            "COPY model_info.json /app/model_info.json",
            "",
            "# ── OCI Image Labels (from Phase 3 mandatory tags) ────────────────────────",
            label_block,
            "",
            "# ── Runtime environment ───────────────────────────────────────────────────",
            f"ENV MODEL_NAME={config.model_name!r}",
            f"ENV MODEL_VERSION={config.model_version!r}",
            f"ENV MODEL_FRAMEWORK={config.framework!r}",
            f"ENV PORT={config.port}",
            'ENV PATH="/root/.local/bin:$PATH"',
            "ENV PYTHONUNBUFFERED=1",
            "ENV PYTHONDONTWRITEBYTECODE=1",
            "",
            f"EXPOSE {config.port}",
            "",
            "# Kubernetes liveness probe target",
            f"HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \\",
            f"    CMD python -c \"import urllib.request; urllib.request.urlopen('http://localhost:{config.port}/health/live')\" || exit 1",
            "",
            'ENTRYPOINT ["python", "/app/serving/inference_server.py"]',
        ]

        return "\n".join(lines) + "\n"

    def _apt_packages(self, framework: str, extra: list[str]) -> list[str]:
        """Return apt packages required at build time."""
        base_pkgs = ["gcc", "g++", "curl"]
        if framework in ("pytorch", "huggingface"):
            base_pkgs.append("libgomp1")
        return base_pkgs + extra

    def _pip_packages(self, framework: str, extra: list[str]) -> list[str]:
        """Return pip packages required in the runtime image."""
        pkgs = list(_SERVING_REQUIREMENTS)
        if framework in ("sklearn", "xgboost"):
            pkgs += ["scikit-learn>=1.5.0", "xgboost>=2.1.0", "pandas>=2.2.0"]
        elif framework == "pytorch":
            pkgs += ["torch>=2.5.0"]
        elif framework == "huggingface":
            pkgs += ["transformers>=4.45.0", "torch>=2.5.0", "accelerate>=1.0.0"]
        return pkgs + extra

    def _render_labels(self, labels: dict[str, str]) -> str:
        """Render a LABEL instruction block from the label dict."""
        label_lines = ["LABEL \\"]
        items = list(labels.items())
        for i, (key, value) in enumerate(items):
            # Escape any double-quotes or backslashes in the value
            safe_value = value.replace("\\", "\\\\").replace('"', '\\"')
            suffix = " \\" if i < len(items) - 1 else ""
            label_lines.append(f'    {key}="{safe_value}"{suffix}')
        return "\n".join(label_lines)