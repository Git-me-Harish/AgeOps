"""
Alembic migration 0005 — OCI Packaging (Phase 4).

New tables:
  oci_build_jobs       — Kaniko build attempt tracking (one row per build job)
  oci_images           — Immutable record of every successfully pushed OCI image
  trivy_scan_results   — CVE scan output per image digest
  sbom_records         — SBOM metadata and R2 storage pointer per image
  cosign_attestations  — Keyless Cosign signing records per image digest

Relationships:
  oci_build_jobs  → (model_name, model_version)  → model_registry_tags
  oci_images      → oci_build_jobs.id            → FK
  trivy_scan_results → oci_images.image_digest   → natural key
  sbom_records       → oci_images.image_digest   → natural key
  cosign_attestations→ oci_images.image_digest   → natural key

Design decisions:
  - oci_images.image_digest is the canonical reference (sha256 digest), not the tag.
    Tags are mutable; digests are cryptographically bound to the content.
  - trivy_scan_results.raw_report is the full Trivy JSON stored as JSONB so
    individual CVEs can be queried without re-scanning.
  - Every table has a FK to workflows.id (nullable) for workflow tracing.

Run:
  alembic upgrade 0005_oci_packaging

Rollback:
  alembic downgrade 0004_model_registry
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005_oci_packaging"
down_revision = "0004_model_registry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # oci_build_jobs
    # One row per Kaniko build attempt.  Status lifecycle:
    #   pending → running → succeeded | failed
    # build_context_uri: R2 URI of the uploaded build context tarball
    # kaniko_job_name:   K8s Job name in the mlops namespace
    # log_uri:           R2 URI of the captured build logs (written after completion)
    op.create_table(
        "oci_build_jobs",
        sa.Column("id",                 sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("model_name",         sa.String(128), nullable=False),
        sa.Column("model_version",      sa.String(32),  nullable=False),
        sa.Column("image_tag",          sa.Text,        nullable=False),   # full tag e.g. docker.io/org/model:v3
        sa.Column("status",             sa.String(16),  nullable=False, server_default="pending"),  # pending|running|succeeded|failed
        sa.Column("kaniko_job_name",    sa.String(128), nullable=True),
        sa.Column("build_context_uri",  sa.Text,        nullable=True),    # R2 URI of context tarball
        sa.Column("log_uri",            sa.Text,        nullable=True),    # R2 URI of captured logs
        sa.Column("error_message",      sa.Text,        nullable=True),
        sa.Column("duration_seconds",   sa.Integer,     nullable=True),
        sa.Column("workflow_id",        sa.String(36),  sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("started_at",         sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("completed_at",       sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("created_at",         sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_ocibj_model_version", "oci_build_jobs", ["model_name", "model_version"])
    op.create_index("ix_ocibj_status",        "oci_build_jobs", ["status"])
    op.create_index("ix_ocibj_workflow_id",   "oci_build_jobs", ["workflow_id"])

    # oci_images
    # Immutable record of every successfully built and pushed OCI image.
    # image_digest: the sha256 content-addressable digest — canonical reference.
    # image_tag:    the mutable human-readable tag — may be reused across versions.
    # image_uri:    full digest reference e.g. docker.io/org/model@sha256:abc
    # labels:       all OCI labels baked into the image, stored as JSONB snapshot.
    # build_job_id: FK to the oci_build_jobs row that produced this image.
    op.create_table(
        "oci_images",
        sa.Column("id",                 sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("model_name",         sa.String(128), nullable=False),
        sa.Column("model_version",      sa.String(32),  nullable=False),
        sa.Column("image_tag",          sa.Text,        nullable=False),
        sa.Column("image_digest",       sa.String(71),  nullable=False),   # "sha256:" + 64 hex
        sa.Column("image_uri",          sa.Text,        nullable=False),   # tag@digest canonical form
        sa.Column("registry_host",      sa.String(128), nullable=False),   # e.g. docker.io
        sa.Column("image_size_bytes",   sa.BigInteger,  nullable=True),
        sa.Column("base_image",         sa.Text,        nullable=True),    # FROM <base>
        sa.Column("labels",             postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("build_job_id",       sa.BigInteger, sa.ForeignKey("oci_build_jobs.id"), nullable=True),
        sa.Column("workflow_id",        sa.String(36), sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("pushed_at",          sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_ociimg_model_version", "oci_images", ["model_name", "model_version"])
    op.create_index("ix_ociimg_digest",        "oci_images", ["image_digest"], unique=True)
    op.create_index("ix_ociimg_pushed_at",     "oci_images", ["pushed_at"])

    # trivy_scan_results
    # One row per Trivy scan (tied to an image digest, not a job).
    # Scanning the same digest twice inserts two rows — scan history is preserved.
    # raw_report: full Trivy JSON output as JSONB (enables per-CVE SQL queries).
    # passed: True iff critical_count == 0 (configurable threshold in settings).
    op.create_table(
        "trivy_scan_results",
        sa.Column("id",               sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("image_digest",     sa.String(71), nullable=False),
        sa.Column("image_tag",        sa.Text,       nullable=True),
        sa.Column("model_name",       sa.String(128), nullable=False),
        sa.Column("model_version",    sa.String(32),  nullable=False),
        sa.Column("critical_count",   sa.Integer, nullable=False, server_default="0"),
        sa.Column("high_count",       sa.Integer, nullable=False, server_default="0"),
        sa.Column("medium_count",     sa.Integer, nullable=False, server_default="0"),
        sa.Column("low_count",        sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_count",      sa.Integer, nullable=False, server_default="0"),
        sa.Column("passed",           sa.Boolean, nullable=False, server_default="false"),
        sa.Column("raw_report",       postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("scan_duration_ms", sa.Integer, nullable=True),
        sa.Column("trivy_version",    sa.String(32), nullable=True),
        sa.Column("workflow_id",      sa.String(36), sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("scanned_at",       sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_trivy_digest",         "trivy_scan_results", ["image_digest"])
    op.create_index("ix_trivy_model_version",  "trivy_scan_results", ["model_name", "model_version"])
    op.create_index("ix_trivy_passed",         "trivy_scan_results", ["passed"])

    # sbom_records
    # SBOM generated by Syft per image digest.  Stored in SPDX JSON format.
    # r2_uri: R2 path where the SPDX JSON file is stored.
    # package_count: total number of packages inventoried (for quick overview).
    op.create_table(
        "sbom_records",
        sa.Column("id",             sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("image_digest",   sa.String(71), nullable=False),
        sa.Column("model_name",     sa.String(128), nullable=False),
        sa.Column("model_version",  sa.String(32),  nullable=False),
        sa.Column("format",         sa.String(32),  nullable=False, server_default="spdx-json"),
        sa.Column("r2_uri",         sa.Text,        nullable=False),   # s3://bucket/security/sbom/...
        sa.Column("package_count",  sa.Integer,     nullable=True),
        sa.Column("syft_version",   sa.String(32),  nullable=True),
        sa.Column("workflow_id",    sa.String(36),  sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("generated_at",   sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_sbom_digest",         "sbom_records", ["image_digest"])
    op.create_index("ix_sbom_model_version",  "sbom_records", ["model_name", "model_version"])

    # cosign_attestations
    # Keyless Cosign signing record per image digest.
    # signature_bundle: the full Cosign bundle JSON (transparency log entry + sig).
    # transparency_log_url: Rekor entry URL for independent verification.
    # In dev mode (cosign_enabled=False): signed=False, entry is recorded but empty.
    op.create_table(
        "cosign_attestations",
        sa.Column("id",                    sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("image_digest",          sa.String(71), nullable=False),
        sa.Column("model_name",            sa.String(128), nullable=False),
        sa.Column("model_version",         sa.String(32),  nullable=False),
        sa.Column("signed",                sa.Boolean, nullable=False, server_default="false"),
        sa.Column("oidc_issuer",           sa.Text, nullable=True),   # e.g. https://token.actions.githubusercontent.com
        sa.Column("signing_identity",      sa.Text, nullable=True),   # GitHub Actions workflow ref
        sa.Column("transparency_log_url",  sa.Text, nullable=True),   # Rekor entry URL
        sa.Column("signature_bundle",      postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("workflow_id",           sa.String(36), sa.ForeignKey("workflows.id"), nullable=True),
        sa.Column("signed_at",             sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_cosign_digest",         "cosign_attestations", ["image_digest"])
    op.create_index("ix_cosign_model_version",  "cosign_attestations", ["model_name", "model_version"])
    op.create_index("ix_cosign_signed",         "cosign_attestations", ["signed"])


def downgrade() -> None:
    op.drop_table("cosign_attestations")
    op.drop_table("sbom_records")
    op.drop_table("trivy_scan_results")
    op.drop_table("oci_images")
    op.drop_table("oci_build_jobs")