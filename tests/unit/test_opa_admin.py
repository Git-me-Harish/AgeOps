# tests/unit/test_opa_admin.py
"""
Unit tests for agents/opa_admin.py (Phase 6 Security & Compliance page's
real OPA ConfigMap read/validate/write path).

Mocks only the Kubernetes client boundary (same convention already used for
agents/training_agent.py's kubernetes usage in this suite) — the validation
logic itself runs for real.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents import opa_admin


class TestValidatePolicyStructural:
    """opa binary unavailable -> honest structural-only check."""

    def test_valid_policy_passes_structural_check(self):
        with patch("agents.opa_admin.shutil.which", return_value=None):
            result = opa_admin.validate_policy("package mlops.governance\n\ndefault allow = false\n")
        assert result.valid is True
        assert result.checked_with == "structural (opa binary unavailable)"
        assert result.errors == []

    def test_missing_package_declaration_fails(self):
        with patch("agents.opa_admin.shutil.which", return_value=None):
            result = opa_admin.validate_policy("default allow = false\n")
        assert result.valid is False
        assert any("package" in e.lower() for e in result.errors)

    def test_unbalanced_braces_fails(self):
        with patch("agents.opa_admin.shutil.which", return_value=None):
            result = opa_admin.validate_policy("package mlops.governance\n\nallow { true\n")
        assert result.valid is False
        assert any("brace" in e.lower() for e in result.errors)


class TestValidatePolicyWithOpaBinary:
    """opa binary present -> a real `opa check` subprocess call, mocked at the subprocess boundary."""

    def test_opa_check_success(self):
        mock_proc = MagicMock(returncode=0, stdout="", stderr="")
        with patch("agents.opa_admin.shutil.which", return_value="/usr/bin/opa"), \
             patch("agents.opa_admin.subprocess.run", return_value=mock_proc):
            result = opa_admin.validate_policy("package mlops.governance\n")
        assert result.valid is True
        assert result.checked_with == "opa check"

    def test_opa_check_failure_surfaces_stderr(self):
        mock_proc = MagicMock(returncode=1, stdout="", stderr="1 error occurred: policy.rego:1: rego_parse_error\n")
        with patch("agents.opa_admin.shutil.which", return_value="/usr/bin/opa"), \
             patch("agents.opa_admin.subprocess.run", return_value=mock_proc):
            result = opa_admin.validate_policy("not valid rego {{{")
        assert result.valid is False
        assert result.checked_with == "opa check"
        assert any("rego_parse_error" in e for e in result.errors)


class TestReadWritePolicy:
    def test_read_policy_returns_configmap_data(self):
        mock_cm = MagicMock()
        mock_cm.data = {"mlops_policy.rego": "package mlops.governance\n"}
        mock_core_v1 = MagicMock()
        mock_core_v1.read_namespaced_config_map.return_value = mock_cm
        mock_k8s_client_mod = MagicMock()
        mock_k8s_client_mod.CoreV1Api.return_value = mock_core_v1

        with patch("agents.opa_admin._k8s_client", return_value=mock_k8s_client_mod):
            text = opa_admin.read_policy()
        assert text == "package mlops.governance\n"

    def test_write_policy_patches_configmap_when_valid(self):
        mock_core_v1 = MagicMock()
        mock_k8s_client_mod = MagicMock()
        mock_k8s_client_mod.CoreV1Api.return_value = mock_core_v1

        with patch("agents.opa_admin.shutil.which", return_value=None), \
             patch("agents.opa_admin._k8s_client", return_value=mock_k8s_client_mod):
            result = opa_admin.write_policy("package mlops.governance\n")

        assert result.valid is True
        mock_core_v1.patch_namespaced_config_map.assert_called_once()
        _, kwargs = mock_core_v1.patch_namespaced_config_map.call_args
        assert kwargs["name"] == "mlops-opa-policy"
        assert kwargs["body"]["data"]["mlops_policy.rego"] == "package mlops.governance\n"

    def test_write_policy_skips_patch_when_invalid(self):
        mock_core_v1 = MagicMock()
        mock_k8s_client_mod = MagicMock()
        mock_k8s_client_mod.CoreV1Api.return_value = mock_core_v1

        with patch("agents.opa_admin.shutil.which", return_value=None), \
             patch("agents.opa_admin._k8s_client", return_value=mock_k8s_client_mod):
            result = opa_admin.write_policy("no package declaration here")

        assert result.valid is False
        mock_core_v1.patch_namespaced_config_map.assert_not_called()
