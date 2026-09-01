"""
agents/packaging — Phase 4: Model Packaging as OCI Images.

Public surface exposed by this package:

  DockerfileGenerator   — Generates framework-specific multi-stage Dockerfiles
  DockerfileConfig      — Input configuration for Dockerfile generation
  GeneratedDockerfile   — Output of DockerfileGenerator.generate()

  KanikoBuilder         — Submits and monitors Kaniko K8s build Jobs
  BuildResult           — Output of KanikoBuilder.build()
  KanikoBuildError      — Raised on build failure
  KanikoBuildTimeoutError — Raised when build exceeds timeout

  TrivyScanner          — Runs Trivy CVE scans via K8s Job or subprocess
  TrivyScanResult       — Parsed CVE scan result with pass/fail gate
  CVEFinding            — Individual CVE record

  SBOMGenerator         — Generates SPDX-JSON SBOMs via Syft
  SBOMRecord            — SBOM metadata with R2 URI

  CosignSigner          — Keyless Cosign signing with Rekor attestation
  SignatureAttestation  — Signing outcome record
  CosignSigningError    — Raised when cosign subprocess fails

  DockerfileAgent       — End-to-end pipeline orchestrator (Steps 1–13)
  PackagingRequest      — Input to DockerfileAgent.run()
  PackagingResult       — Output of DockerfileAgent.run()
  DockerfileAgentError  — Raised on any pipeline step failure
"""
from __future__ import annotations

from agents.packaging.dockerfile_generator import (
    DockerfileConfig,
    DockerfileGenerator,
    GeneratedDockerfile,
    BASE_IMAGES,
)
from agents.packaging.kaniko_builder import (
    BuildResult,
    KanikoBuilder,
    KanikoBuildError,
    KanikoBuildTimeoutError,
)
from agents.packaging.trivy_scanner import (
    CVEFinding,
    TrivyScanResult,
    TrivyScanner,
)
from agents.packaging.sbom_generator import (
    SBOMGenerator,
    SBOMRecord,
)
from agents.packaging.cosign_signer import (
    CosignSigningError,
    CosignSigner,
    SignatureAttestation,
)
from agents.packaging.dockerfile_agent import (
    DockerfileAgent,
    DockerfileAgentError,
    PackagingRequest,
    PackagingResult,
    OCI_TAG_IMAGE_URI,
    OCI_TAG_IMAGE_DIGEST,
    OCI_TAG_SBOM_URI,
    OCI_TAG_COSIGN_SIGNED,
)

__all__ = [
    # dockerfile_generator
    "DockerfileConfig",
    "DockerfileGenerator",
    "GeneratedDockerfile",
    "BASE_IMAGES",
    # kaniko_builder
    "BuildResult",
    "KanikoBuilder",
    "KanikoBuildError",
    "KanikoBuildTimeoutError",
    # trivy_scanner
    "CVEFinding",
    "TrivyScanResult",
    "TrivyScanner",
    # sbom_generator
    "SBOMGenerator",
    "SBOMRecord",
    # cosign_signer
    "CosignSigningError",
    "CosignSigner",
    "SignatureAttestation",
    # dockerfile_agent
    "DockerfileAgent",
    "DockerfileAgentError",
    "PackagingRequest",
    "PackagingResult",
    "OCI_TAG_IMAGE_URI",
    "OCI_TAG_IMAGE_DIGEST",
    "OCI_TAG_SBOM_URI",
    "OCI_TAG_COSIGN_SIGNED",
]