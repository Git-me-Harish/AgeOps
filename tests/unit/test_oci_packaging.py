"""
tests/unit/test_oci_packaging.py

Phase 4 unit tests — OCI Model Packaging.

Coverage sections:
  A) DockerfileGenerator  — per-framework Dockerfiles, OCI labels, error cases
  B) KanikoBuilder        — job manifest structure, digest extraction from logs
  C) TrivyScanner         — real JSON parsing, CVE counting, gate logic
  D) SBOMGenerator        — SPDX JSON parsing, package count, R2 upload
  E) CosignSigner         — output parsing (Rekor URL, identity), signed/unsigned paths
  F) DockerfileAgent      — full pipeline with all collaborators mocked
  G) Inference server     — FastAPI TestClient covering all endpoints
  H) Real OCI gate        — PromotionStateMachine._check_oci_gate with DB mocks

External system boundaries mocked (same pattern as all other tests):
  - kubernetes.client.*    — K8s API calls
  - subprocess.run         — trivy / cosign / syft binaries
  - boto3.client           — R2 / S3 uploads
  - asyncpg pool           — DB reads / writes

No silent exceptions:  every test asserts an explicit outcome.
No try/except/pass:    all exception tests use pytest.raises.

Run:
  pytest tests/unit/test_oci_packaging.py -v --override-ini="addopts="
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_pool(fetchval=None, fetchrow=None, fetch=None, execute="INSERT 0 1") -> MagicMock:
    conn = AsyncMock()
    conn.fetchval  = AsyncMock(return_value=fetchval)
    conn.fetchrow  = AsyncMock(return_value=fetchrow)
    conn.fetch     = AsyncMock(return_value=fetch or [])
    conn.execute   = AsyncMock(return_value=execute)
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__  = AsyncMock(return_value=None)
    return pool


def _valid_tags_dict() -> dict[str, str]:
    return {
        "mlops.dataset.uri":           "s3://bucket/data.parquet",
        "mlops.dataset.hash":          "a" * 64,
        "mlops.dataset.row_count":     "100000",
        "mlops.framework":             "xgboost",
        "mlops.framework.version":     "2.1.0",
        "mlops.python.version":        "3.11.6",
        "mlops.eval.accuracy":         "0.923",
        "mlops.eval.f1":               "0.901",
        "mlops.eval.auc":              "0.948",
        "mlops.eval.holdout_hash":     "b" * 64,
        "mlops.eval.bias_passed":      "true",
        "mlops.security.trivy_scan":   "passed",
        "mlops.security.cve_critical": "0",
        "mlops.git.commit":            "c" * 40,
        "mlops.git.repo":              "https://github.com/org/repo",
    }


def _valid_typed_tags():
    from agents.registry.metadata_schema import TagValidator
    return TagValidator.validate_for_staging(_valid_tags_dict())


# ─────────────────────────────────────────────────────────────────────────────
# A) DockerfileGenerator
# ─────────────────────────────────────────────────────────────────────────────

class TestDockerfileGeneratorFrameworks:

    def _config(self, framework: str, **kwargs):
        from agents.packaging.dockerfile_generator import DockerfileConfig
        return DockerfileConfig(
            model_name=          "fraud-detector",
            model_version=       "3",
            framework=           framework,
            tags=                _valid_typed_tags(),
            mlflow_run_id=       "abc123run",
            mlflow_tracking_uri= "http://mlflow:5000",
            image_tag=           f"docker.io/org/fraud-detector:v3",
            **kwargs,
        )

    def test_sklearn_generates_slim_base_image(self):
        from agents.packaging.dockerfile_generator import DockerfileGenerator
        result = DockerfileGenerator().generate(self._config("sklearn"))
        assert "python:3.11-slim" in result.base_image
        assert "python:3.11-slim" in result.dockerfile_content

    def test_xgboost_generates_slim_base_image(self):
        from agents.packaging.dockerfile_generator import DockerfileGenerator
        result = DockerfileGenerator().generate(self._config("xgboost"))
        assert "python:3.11-slim" in result.base_image

    def test_pytorch_generates_gpu_base_image(self):
        from agents.packaging.dockerfile_generator import DockerfileGenerator
        result = DockerfileGenerator().generate(self._config("pytorch"))
        assert "pytorch" in result.base_image.lower()

    def test_huggingface_generates_hf_base_image(self):
        from agents.packaging.dockerfile_generator import DockerfileGenerator
        result = DockerfileGenerator().generate(self._config("huggingface"))
        assert "huggingface" in result.base_image.lower()

    def test_custom_framework_uses_custom_base_image(self):
        from agents.packaging.dockerfile_generator import DockerfileGenerator
        result = DockerfileGenerator().generate(
            self._config("custom", custom_base_image="my-registry.io/base:3.11")
        )
        assert result.base_image == "my-registry.io/base:3.11"

    def test_custom_framework_without_base_image_raises(self):
        from agents.packaging.dockerfile_generator import DockerfileGenerator
        with pytest.raises(ValueError, match="custom_base_image"):
            DockerfileGenerator().generate(self._config("custom"))

    def test_unsupported_framework_raises(self):
        from agents.packaging.dockerfile_generator import DockerfileGenerator
        with pytest.raises(ValueError, match="Unsupported framework"):
            DockerfileGenerator().generate(self._config("tensorflow"))

    def test_framework_normalised_to_lowercase(self):
        from agents.packaging.dockerfile_generator import DockerfileGenerator
        result = DockerfileGenerator().generate(self._config("XGBoost"))
        assert result.framework == "xgboost"


class TestDockerfileContent:

    def _generate(self, framework="xgboost"):
        from agents.packaging.dockerfile_generator import DockerfileGenerator, DockerfileConfig
        cfg = DockerfileConfig(
            model_name="fraud-detector", model_version="3",
            framework=framework, tags=_valid_typed_tags(),
            mlflow_run_id="run123", mlflow_tracking_uri="http://mlflow:5000",
            image_tag="docker.io/org/fraud-detector:v3",
        )
        return DockerfileGenerator().generate(cfg)

    def test_dockerfile_has_two_from_stages(self):
        result = self._generate()
        from_count = result.dockerfile_content.count("\nFROM ")
        assert from_count >= 2, "Expected at least 2 FROM stages (builder + runtime)"

    def test_dockerfile_has_expose(self):
        result = self._generate()
        assert "EXPOSE 8080" in result.dockerfile_content

    def test_dockerfile_has_healthcheck(self):
        result = self._generate()
        assert "HEALTHCHECK" in result.dockerfile_content

    def test_dockerfile_has_entrypoint(self):
        result = self._generate()
        assert "ENTRYPOINT" in result.dockerfile_content
        assert "inference_server" in result.dockerfile_content

    def test_dockerfile_copies_serving_directory(self):
        result = self._generate()
        assert "COPY serving/" in result.dockerfile_content

    def test_dockerfile_copies_model_info_json(self):
        result = self._generate()
        assert "model_info.json" in result.dockerfile_content

    def test_model_info_json_is_valid_json(self):
        result = self._generate()
        parsed = json.loads(result.model_info_json)
        assert parsed["model_name"] == "fraud-detector"
        assert parsed["model_version"] == "3"

    def test_model_name_in_env_directives(self):
        result = self._generate()
        assert "MODEL_NAME" in result.dockerfile_content

    def test_model_version_in_env_directives(self):
        result = self._generate()
        assert "MODEL_VERSION" in result.dockerfile_content


class TestOCILabels:

    def _labels(self):
        from agents.packaging.dockerfile_generator import DockerfileGenerator, DockerfileConfig
        cfg = DockerfileConfig(
            model_name="fraud-detector", model_version="3",
            framework="sklearn", tags=_valid_typed_tags(),
            mlflow_run_id="r", mlflow_tracking_uri="http://m:5000",
            image_tag="docker.io/org/m:v3",
        )
        return DockerfileGenerator().generate(cfg).oci_labels

    def test_oci_standard_title_label_is_model_name(self):
        assert self._labels()["org.opencontainers.image.title"] == "fraud-detector"

    def test_oci_version_label_is_model_version(self):
        assert self._labels()["org.opencontainers.image.version"] == "3"

    def test_oci_revision_label_is_git_commit(self):
        assert self._labels()["org.opencontainers.image.revision"] == "c" * 40

    def test_mlops_framework_label_is_set(self):
        assert self._labels()["mlops.model.framework"] == "xgboost"

    def test_mlops_eval_accuracy_label_is_set(self):
        assert self._labels()["mlops.eval.accuracy"] == "0.923"

    def test_mlops_git_commit_label_is_set(self):
        assert "mlops.git.commit" in self._labels()

    def test_all_labels_are_strings(self):
        for k, v in self._labels().items():
            assert isinstance(v, str), f"Label {k!r} is not a string: {type(v)}"

    def test_label_block_in_dockerfile(self):
        from agents.packaging.dockerfile_generator import DockerfileGenerator, DockerfileConfig
        cfg = DockerfileConfig(
            model_name="fraud-detector", model_version="3",
            framework="sklearn", tags=_valid_typed_tags(),
            mlflow_run_id="r", mlflow_tracking_uri="http://m:5000",
            image_tag="docker.io/org/m:v3",
        )
        result = DockerfileGenerator().generate(cfg)
        assert "LABEL \\" in result.dockerfile_content
        assert "org.opencontainers.image.title" in result.dockerfile_content


# ─────────────────────────────────────────────────────────────────────────────
# B) KanikoBuilder — manifest structure + digest extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestKanikoJobManifest:

    def _builder(self) -> Any:
        from agents.packaging.kaniko_builder import KanikoBuilder
        return KanikoBuilder("fraud-detector", "3", _make_pool())

    def test_manifest_kind_is_job(self):
        b = self._builder()
        m = b._build_job_manifest("kaniko-abc", "contexts/ctx.tar.gz", "docker.io/org/m:v3")
        assert m["kind"] == "Job"
        assert m["apiVersion"] == "batch/v1"

    def test_manifest_image_tag_in_destination_arg(self):
        b = self._builder()
        m = b._build_job_manifest("kaniko-abc", "contexts/ctx.tar.gz", "docker.io/org/m:v3")
        args = m["spec"]["template"]["spec"]["containers"][0]["args"]
        assert any("docker.io/org/m:v3" in a for a in args)

    def test_manifest_context_uri_in_args(self):
        b = self._builder()
        m = b._build_job_manifest("kaniko-abc", "contexts/ctx.tar.gz", "docker.io/org/m:v3")
        args = m["spec"]["template"]["spec"]["containers"][0]["args"]
        assert any("contexts/ctx.tar.gz" in a for a in args)

    def test_manifest_restart_policy_is_never(self):
        b = self._builder()
        m = b._build_job_manifest("kaniko-abc", "ctx", "img:v1")
        assert m["spec"]["template"]["spec"]["restartPolicy"] == "Never"

    def test_manifest_ttl_after_finished_is_set(self):
        b = self._builder()
        m = b._build_job_manifest("j", "ctx", "img")
        assert m["spec"]["ttlSecondsAfterFinished"] > 0

    def test_manifest_backoff_limit_is_zero(self):
        """No retries — build failures should surface immediately."""
        b = self._builder()
        m = b._build_job_manifest("j", "ctx", "img")
        assert m["spec"]["backoffLimit"] == 0

    def test_manifest_r2_credentials_from_secret(self):
        b = self._builder()
        m = b._build_job_manifest("j", "ctx", "img")
        env = m["spec"]["template"]["spec"]["containers"][0]["env"]
        ak_env = next(e for e in env if e["name"] == "AWS_ACCESS_KEY_ID")
        assert "secretKeyRef" in ak_env["valueFrom"]
        assert ak_env["valueFrom"]["secretKeyRef"]["name"] == "r2-credentials"

    def test_manifest_model_name_label(self):
        b = self._builder()
        m = b._build_job_manifest("j", "ctx", "img")
        labels = m["metadata"]["labels"]
        assert labels["mlops.model_name"] == "fraud-detector"

    def test_manifest_resource_limits_set(self):
        b = self._builder()
        m = b._build_job_manifest("j", "ctx", "img")
        resources = m["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert "limits" in resources
        assert "requests" in resources


class TestKanikoDigestExtraction:

    def _builder(self):
        from agents.packaging.kaniko_builder import KanikoBuilder
        return KanikoBuilder("m", "1", _make_pool())

    def test_extracts_digest_from_log_line(self):
        b = self._builder()
        logs = (
            "INFO[0043] Built and pushed image as docker.io/org/m:v1\n"
            "INFO[0044] sha256:" + "a" * 64 + "\n"
        )
        digest = b._extract_digest(logs, "docker.io/org/m:v1")
        assert digest == "sha256:" + "a" * 64

    def test_extracts_digest_inline_in_log_line(self):
        b = self._builder()
        digest_hex = "f" * 64
        logs = f"INFO Built and pushed image docker.io/org/m:v1@sha256:{digest_hex}\n"
        digest = b._extract_digest(logs, "docker.io/org/m:v1")
        assert digest == f"sha256:{digest_hex}"

    def test_uses_last_sha256_match_in_logs(self):
        """If multiple sha256 appear (base layers etc.), last one is the final image."""
        b = self._builder()
        first  = "1" * 64
        second = "2" * 64
        logs = f"sha256:{first}\nsome other line\nsha256:{second}\n"
        digest = b._extract_digest(logs, "docker.io/org/m:v1")
        assert digest == f"sha256:{second}"

    def test_raises_when_no_digest_in_logs_and_registry_unreachable(self):
        from agents.packaging.kaniko_builder import KanikoBuildError
        b = self._builder()
        with patch.object(b, "_query_registry_digest", return_value=None):
            with pytest.raises(KanikoBuildError, match="digest"):
                b._extract_digest("no sha256 here", "docker.io/org/m:v1")

    def test_falls_back_to_registry_api_when_logs_empty(self):
        b = self._builder()
        expected = "sha256:" + "b" * 64
        with patch.object(b, "_query_registry_digest", return_value=expected):
            digest = b._extract_digest("no digest in logs", "docker.io/org/m:v1")
        assert digest == expected

    def test_job_name_is_unique(self):
        from agents.packaging.kaniko_builder import KanikoBuilder
        names = {KanikoBuilder._job_name() for _ in range(50)}
        assert len(names) == 50, "Job names must be unique"


# ─────────────────────────────────────────────────────────────────────────────
# C) TrivyScanner — real JSON parsing + CVE gate logic
# ─────────────────────────────────────────────────────────────────────────────

def _trivy_report(critical=0, high=0, medium=0, low=0) -> dict:
    """Build a minimal but structurally correct Trivy JSON report."""
    vulns = []
    for _ in range(critical):
        vulns.append({
            "VulnerabilityID": "CVE-2024-1111",
            "Severity": "CRITICAL",
            "PkgName": "openssl",
            "InstalledVersion": "1.1.1",
            "FixedVersion": "1.1.2",
            "Title": "Critical vuln",
            "Description": "desc",
        })
    for _ in range(high):
        vulns.append({
            "VulnerabilityID": "CVE-2024-2222",
            "Severity": "HIGH",
            "PkgName": "libxml2",
            "InstalledVersion": "2.9.9",
            "FixedVersion": "2.9.10",
            "Title": "High vuln",
            "Description": "desc",
        })
    for _ in range(medium):
        vulns.append({
            "VulnerabilityID": "CVE-2024-3333",
            "Severity": "MEDIUM",
            "PkgName": "curl",
            "InstalledVersion": "7.0",
            "FixedVersion": None,
            "Title": "Med vuln",
            "Description": "desc",
        })
    for _ in range(low):
        vulns.append({
            "VulnerabilityID": "CVE-2024-4444",
            "Severity": "LOW",
            "PkgName": "zlib",
            "InstalledVersion": "1.2.11",
            "FixedVersion": "1.2.12",
            "Title": "Low vuln",
            "Description": "desc",
        })
    return {
        "SchemaVersion": 2,
        "Results": [
            {
                "Target": "python:3.11-slim",
                "Type": "debian",
                "Vulnerabilities": vulns,
            }
        ],
    }


class TestTrivyJSONParsing:

    def _parse(self, report: dict):
        from agents.packaging.trivy_scanner import TrivyScanner
        return TrivyScanner._parse_trivy_json(report, "docker.io/org/m:v3", "sha256:" + "a" * 64)

    def test_clean_image_passes(self):
        result = self._parse(_trivy_report())
        assert result.passed is True
        assert result.critical_count == 0
        assert result.total_count == 0

    def test_critical_cve_fails_gate(self):
        result = self._parse(_trivy_report(critical=1))
        assert result.passed is False
        assert result.critical_count == 1
        assert "CRITICAL" in (result.rejection_reason or "")

    def test_high_cve_under_limit_passes(self):
        """3 HIGH CVEs is under the default limit of 5."""
        result = self._parse(_trivy_report(high=3))
        assert result.passed is True

    def test_high_cve_over_limit_fails(self):
        """6 HIGH CVEs is over the default limit of 5."""
        result = self._parse(_trivy_report(high=6))
        assert result.passed is False
        assert "HIGH" in (result.rejection_reason or "")

    def test_medium_and_low_only_passes(self):
        result = self._parse(_trivy_report(medium=10, low=20))
        assert result.passed is True
        assert result.medium_count == 10
        assert result.low_count == 20

    def test_total_count_is_sum_of_all_severities(self):
        result = self._parse(_trivy_report(critical=1, high=2, medium=3, low=4))
        assert result.total_count == 10

    def test_findings_list_matches_vulnerability_count(self):
        result = self._parse(_trivy_report(critical=2, high=3))
        assert len(result.findings) == 5

    def test_finding_fields_are_populated(self):
        result = self._parse(_trivy_report(critical=1))
        f = result.findings[0]
        assert f.vulnerability_id == "CVE-2024-1111"
        assert f.severity == "CRITICAL"
        assert f.package == "openssl"

    def test_missing_vulnerabilities_key_treated_as_clean(self):
        """Results block without Vulnerabilities key is a clean target."""
        report = {"SchemaVersion": 2, "Results": [{"Target": "scratch", "Type": "oci"}]}
        result = self._parse(report)
        assert result.passed is True
        assert result.total_count == 0

    def test_empty_results_array_is_clean(self):
        result = self._parse({"SchemaVersion": 2, "Results": []})
        assert result.passed is True

    def test_raw_report_stored_on_result(self):
        report = _trivy_report(critical=1)
        result = self._parse(report)
        assert result.raw_report == report

    def test_image_uri_and_digest_stored(self):
        result = self._parse(_trivy_report())
        assert result.image_uri    == "docker.io/org/m:v3"
        assert result.image_digest == "sha256:" + "a" * 64


class TestTrivyJSONExtraction:

    def test_extracts_json_from_pure_stdout(self):
        from agents.packaging.trivy_scanner import TrivyScanner
        report = _trivy_report()
        text   = json.dumps(report)
        result = TrivyScanner._extract_json_from_log(text)
        assert result["SchemaVersion"] == 2

    def test_extracts_json_when_prefix_noise_present(self):
        from agents.packaging.trivy_scanner import TrivyScanner
        report = _trivy_report()
        text   = "some prefix noise\n" + json.dumps(report)
        result = TrivyScanner._extract_json_from_log(text)
        assert result["SchemaVersion"] == 2

    def test_raises_when_no_json_in_output(self):
        from agents.packaging.trivy_scanner import TrivyScanner
        with pytest.raises(RuntimeError, match="no JSON"):
            TrivyScanner._extract_json_from_log("no json here at all")

    def test_raises_on_malformed_json(self):
        from agents.packaging.trivy_scanner import TrivyScanner
        with pytest.raises(RuntimeError, match="parse"):
            TrivyScanner._extract_json_from_log("{broken json")


# ─────────────────────────────────────────────────────────────────────────────
# D) SBOMGenerator — SPDX JSON parsing + package count + R2 upload
# ─────────────────────────────────────────────────────────────────────────────

def _spdx_report(n_packages: int = 5, syft_version: str = "1.19.0") -> dict:
    """Minimal structurally-valid SPDX-2.3 JSON from Syft."""
    packages = [
        {
            "SPDXID":           f"SPDXRef-Package-{i}",
            "name":             f"package-{i}",
            "versionInfo":      f"1.0.{i}",
            "filesAnalyzed":    False,
        }
        for i in range(n_packages)
    ]
    return {
        "spdxVersion": "SPDX-2.3",
        "name":        "docker.io/org/m:v3",
        "SPDXID":      "SPDXRef-DOCUMENT",
        "packages":    packages,
        "creationInfo": {
            "created":  "2025-01-01T00:00:00Z",
            "creators": [f"Tool: syft-{syft_version}", "Organization: Anchore"],
        },
    }


class TestSBOMParsing:

    def test_package_count_matches_packages_array(self):
        from agents.packaging.sbom_generator import SBOMGenerator
        report = _spdx_report(n_packages=7)
        count  = SBOMGenerator._count_packages(report)
        assert count == 7

    def test_empty_packages_returns_zero(self):
        from agents.packaging.sbom_generator import SBOMGenerator
        assert SBOMGenerator._count_packages({"packages": []}) == 0

    def test_missing_packages_key_returns_zero(self):
        from agents.packaging.sbom_generator import SBOMGenerator
        assert SBOMGenerator._count_packages({}) == 0

    def test_syft_version_extracted_correctly(self):
        from agents.packaging.sbom_generator import SBOMGenerator
        report  = _spdx_report(syft_version="1.19.0")
        version = SBOMGenerator._extract_syft_version(report)
        assert version == "1.19.0"

    def test_syft_version_returns_none_when_missing(self):
        from agents.packaging.sbom_generator import SBOMGenerator
        report = {"creationInfo": {"creators": ["Organization: Anchore"]}}
        assert SBOMGenerator._extract_syft_version(report) is None

    def test_parse_spdx_json_from_pure_stdout(self):
        from agents.packaging.sbom_generator import SBOMGenerator
        report = _spdx_report()
        text   = json.dumps(report)
        parsed = SBOMGenerator._parse_spdx_json(text)
        assert parsed["spdxVersion"] == "SPDX-2.3"

    def test_parse_raises_on_no_json(self):
        from agents.packaging.sbom_generator import SBOMGenerator
        with pytest.raises(RuntimeError, match="no JSON"):
            SBOMGenerator._parse_spdx_json("no json at all")

    def test_parse_raises_on_malformed_json(self):
        from agents.packaging.sbom_generator import SBOMGenerator
        with pytest.raises(RuntimeError, match="parse"):
            SBOMGenerator._parse_spdx_json("{invalid")


class TestSBOMUpload:

    async def test_generate_calls_s3_put_object(self):
        from agents.packaging.sbom_generator import SBOMGenerator

        pool = _make_pool(execute="INSERT 0 1")
        gen  = SBOMGenerator("fraud-detector", "3", pool)

        mock_s3 = MagicMock()
        mock_s3.put_object = MagicMock()
        gen._s3 = mock_s3

        report = _spdx_report(n_packages=10)
        with patch.object(gen, "_generate_via_subprocess", return_value=report):
            with patch("agents.packaging.sbom_generator.settings") as mock_settings:
                mock_settings.k8s_in_cluster      = False
                mock_settings.r2_bucket_name       = "test-bucket"
                mock_settings.r2_sbom_prefix       = "security/sbom"
                record = await gen.generate("docker.io/org/m:v3@sha256:" + "a" * 64, "sha256:" + "a" * 64)

        mock_s3.put_object.assert_called_once()
        assert record.package_count == 10
        assert record.r2_uri.startswith("s3://")

    async def test_generate_raises_when_s3_put_fails(self):
        from agents.packaging.sbom_generator import SBOMGenerator
        import botocore.exceptions

        pool = _make_pool()
        gen  = SBOMGenerator("m", "1", pool)
        gen._s3 = MagicMock()
        gen._s3.put_object = MagicMock(
            side_effect=botocore.exceptions.ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "Denied"}},
                "PutObject",
            )
        )
        report = _spdx_report()
        with patch.object(gen, "_generate_via_subprocess", return_value=report):
            with patch("agents.packaging.sbom_generator.settings") as ms:
                ms.k8s_in_cluster = False
                ms.r2_bucket_name = "bucket"
                ms.r2_sbom_prefix = "sbom"
                with pytest.raises(RuntimeError, match="R2"):
                    await gen.generate("docker.io/org/m:v1@sha256:" + "a" * 64, "sha256:" + "a" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# E) CosignSigner — output parsing + signed/unsigned paths
# ─────────────────────────────────────────────────────────────────────────────

class TestCosignOutputParsing:

    def test_extracts_rekor_url_from_standard_output(self):
        from agents.packaging.cosign_signer import CosignSigner
        output = (
            "Pushing signature to: docker.io/org/m\n"
            "tlog entry created with index: 12345678\n"
            "https://rekor.sigstore.dev/api/v1/log/entries?logIndex=12345678\n"
        )
        url = CosignSigner._extract_rekor_url(output)
        assert url is not None
        assert "rekor.sigstore.dev" in url

    def test_returns_none_when_no_rekor_url(self):
        from agents.packaging.cosign_signer import CosignSigner
        assert CosignSigner._extract_rekor_url("no url here") is None

    def test_extracts_issuer_from_cosign_output(self):
        from agents.packaging.cosign_signer import CosignSigner
        output = (
            "Signing with identity: https://github.com/org/repo/.github/workflows/build.yml@refs/heads/main\n"
            "with issuer: https://token.actions.githubusercontent.com\n"
        )
        issuer, identity = CosignSigner._extract_identity(output)
        assert issuer is not None
        assert "token.actions" in issuer

    def test_returns_none_for_both_when_output_empty(self):
        from agents.packaging.cosign_signer import CosignSigner
        issuer, identity = CosignSigner._extract_identity("")
        assert issuer is None
        assert identity is None


class TestCosignSignerPaths:

    async def test_skips_signing_when_disabled(self):
        from agents.packaging.cosign_signer import CosignSigner
        pool   = _make_pool()
        signer = CosignSigner("m", "1", pool)

        with patch("agents.packaging.cosign_signer.settings") as ms:
            ms.cosign_enabled = False
            attest = await signer.sign(
                image_uri=    "docker.io/org/m:v3@sha256:" + "a" * 64,
                image_digest= "sha256:" + "a" * 64,
            )

        assert attest.signed is False
        assert attest.skip_reason is not None
        assert "cosign_enabled" in attest.skip_reason

    async def test_skipped_signing_still_persists_db_row(self):
        from agents.packaging.cosign_signer import CosignSigner
        pool = _make_pool()
        conn = pool.acquire.return_value.__aenter__.return_value

        signer = CosignSigner("m", "1", pool)
        with patch("agents.packaging.cosign_signer.settings") as ms:
            ms.cosign_enabled = False
            await signer.sign("docker.io/org/m:v3@sha256:" + "a" * 64, "sha256:" + "a" * 64)

        conn.execute.assert_called_once()

    async def test_raises_cosign_signing_error_when_binary_missing(self):
        from agents.packaging.cosign_signer import CosignSigner, CosignSigningError
        import subprocess
        pool   = _make_pool()
        signer = CosignSigner("m", "1", pool)

        with patch("agents.packaging.cosign_signer.settings") as ms:
            ms.cosign_enabled = True
            with patch("subprocess.run", side_effect=FileNotFoundError("cosign not found")):
                with pytest.raises(CosignSigningError, match="cosign binary not found"):
                    await signer.sign("img@sha256:" + "a" * 64, "sha256:" + "a" * 64)

    async def test_raises_cosign_signing_error_on_nonzero_exit(self):
        from agents.packaging.cosign_signer import CosignSigner, CosignSigningError
        import subprocess
        pool   = _make_pool()
        signer = CosignSigner("m", "1", pool)

        err = subprocess.CalledProcessError(1, ["cosign"], output="", stderr="OIDC token expired")
        with patch("agents.packaging.cosign_signer.settings") as ms:
            ms.cosign_enabled = True
            with patch("subprocess.run", side_effect=err):
                with pytest.raises(CosignSigningError, match="exited 1"):
                    await signer.sign("img@sha256:" + "a" * 64, "sha256:" + "a" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# F) DockerfileAgent — full pipeline with all collaborators mocked
# ─────────────────────────────────────────────────────────────────────────────

def _make_tag_row() -> MagicMock:
    """DB row mock for model_registry_tags."""
    row = MagicMock()
    row.__getitem__ = lambda self, k: {
        "dataset_uri":           "s3://b/d",
        "dataset_hash":          "a" * 64,
        "dataset_row_count":     100000,
        "framework":             "xgboost",
        "framework_version":     "2.1.0",
        "python_version":        "3.11.6",
        "eval_accuracy":         0.923,
        "eval_f1":               0.901,
        "eval_auc":              0.948,
        "eval_holdout_hash":     "b" * 64,
        "eval_bias_passed":      True,
        "security_trivy_scan":   "passed",
        "security_cve_critical": 0,
        "git_commit":            "c" * 40,
        "git_repo":              "https://github.com/org/repo",
        "approved_by":           None,
        "approved_at":           None,
        "governance_audit_id":   None,
    }[k]
    return row


class TestDockerfileAgentPipeline:

    def _make_agent(self, pool) -> Any:
        from agents.packaging.dockerfile_agent import DockerfileAgent
        client = MagicMock()
        mv = MagicMock()
        mv.run_id = "run-abc"
        client.get_model_version = MagicMock(return_value=mv)
        client.set_model_version_tag = MagicMock()
        return DockerfileAgent(pool=pool, mlflow_client=client)

    def _pool_with_tags(self, build_job_id=1, oci_image_id=2) -> MagicMock:
        conn = AsyncMock()
        # fetchrow returns tag row on first call, None (no lineage) thereafter
        conn.fetchrow  = AsyncMock(side_effect=[_make_tag_row(), None])
        conn.fetchval  = AsyncMock(side_effect=[build_job_id, oci_image_id])
        conn.execute   = AsyncMock(return_value="UPDATE 1")
        conn.fetch     = AsyncMock(return_value=[])
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__  = AsyncMock(return_value=None)
        return pool

    def _mock_build_result(self):
        from agents.packaging.kaniko_builder import BuildResult
        return BuildResult(
            image_tag=        "docker.io/org/m:v3",
            image_digest=     "sha256:" + "a" * 64,
            image_uri=        "docker.io/org/m:v3@sha256:" + "a" * 64,
            duration_seconds= 180,
            log_uri=          "s3://bucket/logs/build.log",
            job_name=         "kaniko-abc12345",
        )

    def _mock_scan_result(self, passed=True):
        from agents.packaging.trivy_scanner import TrivyScanResult
        return TrivyScanResult(
            image_uri=        "docker.io/org/m:v3@sha256:" + "a" * 64,
            image_digest=     "sha256:" + "a" * 64,
            critical_count=   0,
            high_count=       0,
            medium_count=     0,
            low_count=        0,
            total_count=      0,
            passed=           passed,
            rejection_reason= None if passed else "1 CRITICAL CVE found",
        )

    def _mock_sbom_record(self):
        from agents.packaging.sbom_generator import SBOMRecord
        return SBOMRecord(
            image_digest=  "sha256:" + "a" * 64,
            model_name=    "fraud-detector",
            model_version= "3",
            format=        "spdx-json",
            r2_uri=        "s3://bucket/sbom/m.json",
            package_count= 42,
            syft_version=  "1.19.0",
            raw_sbom=      {},
        )

    def _mock_attestation(self, signed=True):
        from agents.packaging.cosign_signer import SignatureAttestation
        return SignatureAttestation(
            image_digest= "sha256:" + "a" * 64,
            signed=       signed,
        )

    async def test_successful_pipeline_returns_packaging_result(self):
        from agents.packaging.dockerfile_agent import PackagingRequest
        pool  = self._pool_with_tags()
        agent = self._make_agent(pool)

        with patch("agents.packaging.dockerfile_agent.KanikoBuilder") as MockKaniko, \
             patch("agents.packaging.dockerfile_agent.TrivyScanner")  as MockTrivy,  \
             patch("agents.packaging.dockerfile_agent.SBOMGenerator")  as MockSBOM,   \
             patch("agents.packaging.dockerfile_agent.CosignSigner")   as MockCosign, \
             patch("agents.packaging.dockerfile_agent.LineageGraphQuerier") as MockQ,  \
             patch("agents.packaging.dockerfile_agent.settings") as ms:

            ms.mlflow_tracking_uri       = "http://mlflow:5000"
            ms.image_registry_host       = "docker.io"
            ms.image_registry_namespace  = "org"
            ms.kaniko_cache_enabled      = False
            ms.k8s_namespace             = "mlops"

            MockKaniko.return_value.build   = AsyncMock(return_value=self._mock_build_result())
            MockTrivy.return_value.scan     = AsyncMock(return_value=self._mock_scan_result(passed=True))
            MockSBOM.return_value.generate  = AsyncMock(return_value=self._mock_sbom_record())
            MockCosign.return_value.sign    = AsyncMock(return_value=self._mock_attestation(signed=True))

            mock_querier = AsyncMock()
            mock_querier.get_full_lineage = AsyncMock(return_value={"nodes": [], "edges": []})
            MockQ.return_value = mock_querier

            result = await agent.run(
                PackagingRequest(model_name="fraud-detector", model_version="3")
            )

        assert result.model_name    == "fraud-detector"
        assert result.model_version == "3"
        assert result.image_digest  == "sha256:" + "a" * 64
        assert result.cosign_signed is True

    async def test_trivy_gate_failure_raises_dockerfile_agent_error(self):
        from agents.packaging.dockerfile_agent import PackagingRequest, DockerfileAgentError
        pool  = self._pool_with_tags()
        agent = self._make_agent(pool)

        with patch("agents.packaging.dockerfile_agent.KanikoBuilder") as MockKaniko, \
             patch("agents.packaging.dockerfile_agent.TrivyScanner")  as MockTrivy,  \
             patch("agents.packaging.dockerfile_agent.settings") as ms:

            ms.mlflow_tracking_uri       = "http://mlflow:5000"
            ms.image_registry_host       = "docker.io"
            ms.image_registry_namespace  = "org"
            ms.kaniko_cache_enabled      = False
            ms.k8s_namespace             = "mlops"

            MockKaniko.return_value.build = AsyncMock(return_value=self._mock_build_result())
            MockTrivy.return_value.scan   = AsyncMock(return_value=self._mock_scan_result(passed=False))

            with pytest.raises(DockerfileAgentError, match="Trivy CVE gate BLOCKED"):
                await agent.run(
                    PackagingRequest(model_name="fraud-detector", model_version="3")
                )

    async def test_missing_registry_tags_raises_dockerfile_agent_error(self):
        from agents.packaging.dockerfile_agent import PackagingRequest, DockerfileAgentError
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)  # no tags row
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__  = AsyncMock(return_value=None)

        agent = self._make_agent(pool)
        with patch("agents.packaging.dockerfile_agent.settings") as ms:
            ms.mlflow_tracking_uri = "http://mlflow:5000"
            with pytest.raises(DockerfileAgentError, match="No model_registry_tags"):
                await agent.run(
                    PackagingRequest(model_name="missing-model", model_version="99")
                )

    async def test_mlflow_tags_patched_after_successful_build(self):
        from agents.packaging.dockerfile_agent import PackagingRequest
        pool  = self._pool_with_tags()
        agent = self._make_agent(pool)

        with patch("agents.packaging.dockerfile_agent.KanikoBuilder") as MK, \
             patch("agents.packaging.dockerfile_agent.TrivyScanner")  as MT, \
             patch("agents.packaging.dockerfile_agent.SBOMGenerator")  as MS, \
             patch("agents.packaging.dockerfile_agent.CosignSigner")   as MC, \
             patch("agents.packaging.dockerfile_agent.LineageGraphQuerier") as MQ, \
             patch("agents.packaging.dockerfile_agent.settings") as ms:

            ms.mlflow_tracking_uri = "http://mlflow:5000"
            ms.image_registry_host = "docker.io"
            ms.image_registry_namespace = "org"
            ms.kaniko_cache_enabled = False
            ms.k8s_namespace = "mlops"

            MK.return_value.build  = AsyncMock(return_value=self._mock_build_result())
            MT.return_value.scan   = AsyncMock(return_value=self._mock_scan_result())
            MS.return_value.generate = AsyncMock(return_value=self._mock_sbom_record())
            MC.return_value.sign   = AsyncMock(return_value=self._mock_attestation())
            MQ.return_value.get_full_lineage = AsyncMock(return_value={"nodes": [], "edges": []})

            await agent.run(PackagingRequest(model_name="fraud-detector", model_version="3"))

        # set_model_version_tag must be called for each OCI tag key
        assert agent._client.set_model_version_tag.call_count == 4


# ─────────────────────────────────────────────────────────────────────────────
# G) Inference server — FastAPI TestClient
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceServerEndpoints:
    """
    Tests run against the FastAPI app WITHOUT loading a real MLflow model.
    We patch the global _model to a MagicMock and set health flags directly.
    """

    @pytest.fixture(autouse=True)
    def reset_health_and_model(self):
        """Reset singleton state before each test."""
        from serving import health as h_mod, inference_server as srv
        h_mod.health.model_loaded  = False
        h_mod.health.warmup_done   = False
        h_mod.health.model_name    = "test-model"
        h_mod.health.model_version = "1"
        h_mod.health.framework     = "sklearn"
        srv._model      = None
        srv._model_info = {
            "model_name":    "test-model",
            "model_version": "1",
            "framework":     "sklearn",
            "port":          8080,
        }
        yield

    def _client(self):
        from fastapi.testclient import TestClient
        from serving.inference_server import app
        return TestClient(app, raise_server_exceptions=False)

    def _ready_client(self):
        """Client with model already loaded and warmed up."""
        import pandas as pd
        from serving import health as h_mod, inference_server as srv
        from fastapi.testclient import TestClient

        mock_model = MagicMock()
        mock_model.metadata = None
        mock_model.predict   = MagicMock(return_value=pd.Series([0.9, 0.1]))

        srv._model = mock_model
        h_mod.health.mark_model_loaded("test-model", "1", "sklearn")
        h_mod.health.mark_warmup_done()

        from serving.inference_server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_liveness_always_200(self):
        resp = self._client().get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_readiness_503_when_model_not_loaded(self):
        resp = self._client().get("/health/ready")
        assert resp.status_code == 503

    def test_readiness_200_when_model_loaded_and_warmed(self):
        resp = self._ready_client().get("/health/ready")
        assert resp.status_code == 200

    def test_info_returns_model_name(self):
        resp = self._ready_client().get("/info")
        assert resp.status_code == 200
        assert resp.json()["model_name"] == "test-model"

    def test_schema_returns_404_or_200_depending_on_model(self):
        """When model has no signature, schema returns None fields."""
        client = self._ready_client()
        resp   = client.get("/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "input_schema" in data

    def test_predict_503_when_model_not_loaded(self):
        resp = self._client().post(
            "/predict", json={"instances": [[1.0, 2.0, 3.0]]}
        )
        assert resp.status_code == 503

    def test_predict_returns_predictions(self):
        client = self._ready_client()
        resp   = client.post("/predict", json={"instances": [[1.0, 2.0, 3.0]]})
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert data["model_name"] == "test-model"

    def test_predict_with_named_inputs(self):
        client = self._ready_client()
        resp   = client.post(
            "/predict",
            json={"inputs": {"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]}},
        )
        assert resp.status_code == 200

    def test_predict_422_when_both_instances_and_inputs_provided(self):
        client = self._ready_client()
        resp   = client.post(
            "/predict",
            json={"instances": [[1.0]], "inputs": {"a": [1.0]}},
        )
        assert resp.status_code == 422

    def test_predict_422_when_neither_instances_nor_inputs_provided(self):
        client = self._ready_client()
        resp   = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_predict_batch_mirrors_predict(self):
        client = self._ready_client()
        resp   = client.post("/predict/batch", json={"instances": [[1.0, 2.0]]})
        assert resp.status_code == 200

    def test_metrics_endpoint_returns_prometheus_format(self):
        client = self._ready_client()
        resp   = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]

    def test_prediction_count_in_response(self):
        import pandas as pd
        from serving import inference_server as srv
        # _ready_client() assigns a fresh mock_model to srv._model — the
        # predict() override must be applied AFTER that call, not before,
        # or it gets discarded when _ready_client() replaces srv._model.
        client = self._ready_client()
        srv._model.predict = MagicMock(return_value=pd.Series([0.8, 0.6, 0.4]))
        resp   = client.post("/predict", json={"instances": [[1], [2], [3]]})
        assert resp.json()["prediction_count"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# H) Real OCI gate in PromotionStateMachine
# ─────────────────────────────────────────────────────────────────────────────

def _make_img_row(digest: str = None) -> MagicMock:
    digest = digest or ("sha256:" + "a" * 64)
    row = MagicMock()
    row.__getitem__ = lambda self, k: {
        "image_digest": digest,
        "image_uri":    f"docker.io/org/m:v3@{digest}",
    }[k]
    return row


def _make_scan_row(passed: bool, critical: int = 0, high: int = 0) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, k: {
        "passed":         passed,
        "critical_count": critical,
        "high_count":     high,
    }[k]
    return row


def _make_sig_row(signed: bool) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, k: {
        "signed":      signed,
        "oidc_issuer": "https://token.actions.githubusercontent.com",
    }[k]
    return row


class TestRealOCIGate:

    def _sm(self, conn_side_effects: list) -> Any:
        from agents.registry.promotion_state_machine import PromotionStateMachine
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=conn_side_effects)
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__  = AsyncMock(return_value=None)
        return PromotionStateMachine(pool=pool, mlflow_client=MagicMock())

    async def test_passes_when_image_exists_scan_passed_cosign_signed(self):
        digest  = "sha256:" + "a" * 64
        sm = self._sm([_make_img_row(digest), _make_scan_row(True), _make_sig_row(True)])
        with patch("agents.registry.promotion_state_machine._s") as ms:
            ms.cosign_enabled = True
            passed, reason = await sm._check_oci_gate("model", "3")
        assert passed is True
        assert "verified" in reason.lower()

    async def test_fails_when_no_oci_image_row(self):
        sm = self._sm([None])  # no row in oci_images
        with patch("agents.registry.promotion_state_machine._s") as ms:
            ms.cosign_enabled = True
            passed, reason = await sm._check_oci_gate("model", "3")
        assert passed is False
        assert "No OCI image" in reason

    async def test_fails_when_trivy_scan_not_passed(self):
        digest = "sha256:" + "a" * 64
        sm = self._sm([_make_img_row(digest), _make_scan_row(False, critical=2)])
        with patch("agents.registry.promotion_state_machine._s") as ms:
            ms.cosign_enabled = True
            passed, reason = await sm._check_oci_gate("model", "3")
        assert passed is False
        assert "Trivy" in reason

    async def test_fails_when_no_trivy_scan_row(self):
        digest = "sha256:" + "a" * 64
        sm = self._sm([_make_img_row(digest), None])  # no scan row
        with patch("agents.registry.promotion_state_machine._s") as ms:
            ms.cosign_enabled = True
            passed, reason = await sm._check_oci_gate("model", "3")
        assert passed is False
        assert "No Trivy scan" in reason

    async def test_fails_when_cosign_not_signed(self):
        digest = "sha256:" + "a" * 64
        sm = self._sm([_make_img_row(digest), _make_scan_row(True), _make_sig_row(False)])
        with patch("agents.registry.promotion_state_machine._s") as ms:
            ms.cosign_enabled = True
            passed, reason = await sm._check_oci_gate("model", "3")
        assert passed is False
        assert "signed=False" in reason

    async def test_fails_when_cosign_row_missing_and_enabled(self):
        digest = "sha256:" + "a" * 64
        sm = self._sm([_make_img_row(digest), _make_scan_row(True), None])
        with patch("agents.registry.promotion_state_machine._s") as ms:
            ms.cosign_enabled = True
            passed, reason = await sm._check_oci_gate("model", "3")
        assert passed is False
        assert "No Cosign attestation" in reason

    async def test_passes_when_cosign_disabled_and_unsigned(self):
        """When cosign_enabled=False, skip the signature check entirely."""
        digest = "sha256:" + "a" * 64
        # Only 2 DB calls needed (no cosign lookup)
        sm = self._sm([_make_img_row(digest), _make_scan_row(True)])
        with patch("agents.registry.promotion_state_machine._s") as ms:
            ms.cosign_enabled = False
            passed, reason = await sm._check_oci_gate("model", "3")
        assert passed is True
        assert "disabled" in reason.lower() or "skipped" in reason.lower()

    async def test_fails_closed_on_db_error_for_oci_images(self):
        """DB exception must block the gate, not allow passage."""
        from agents.registry.promotion_state_machine import PromotionStateMachine
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=Exception("connection reset by peer"))
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__  = AsyncMock(return_value=None)
        sm = PromotionStateMachine(pool=pool, mlflow_client=MagicMock())

        with patch("agents.registry.promotion_state_machine._s") as ms:
            ms.cosign_enabled = True
            passed, reason = await sm._check_oci_gate("model", "3")
        assert passed is False
        assert "DB error" in reason