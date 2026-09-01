"""
agents/packaging/cosign_signer.py

Phase 4 — Cosign Keyless Image Signing.

Signs every production-bound OCI image using Sigstore's keyless signing
protocol (OIDC-based, no private key management required).  The signature is
recorded in the Rekor public transparency log, making it independently
verifiable.

Signing flow:
  cosign sign --yes <image>@<digest>
    → OIDC token fetched from the ambient provider (GitHub Actions, GKE WI, etc.)
    → Sigstore Fulcio issues a short-lived certificate for the OIDC identity
    → The signature + certificate are uploaded to Rekor
    → Rekor returns a transparency log entry URL
    → SignatureAttestation is persisted to cosign_attestations

Verification (by a consumer):
  cosign verify \
    --certificate-identity-regexp <signing_identity> \
    --certificate-oidc-issuer <oidc_issuer> \
    <image>@<digest>

Dev mode (settings.cosign_enabled=False or no ambient OIDC token):
  Signing is skipped.  A cosign_attestations row is inserted with signed=False
  so the gate does not block — local development doesn't have a Sigstore OIDC
  token.  The gate enforces signing only when settings.cosign_enabled=True.

Error contract:
  - subprocess.CalledProcessError raises CosignSigningError.
  - Missing cosign binary raises CosignSigningError (not FileNotFoundError).
  - No exceptions are swallowed — signed=False is only set when the caller
    explicitly opts out via settings.cosign_enabled=False.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

from configs.settings import settings

logger = logging.getLogger(__name__)

# Regex to extract the Rekor transparency log entry URL from cosign output
_REKOR_URL_RE = re.compile(
    r"https://rekor\.sigstore\.dev/api/v1/log/entries[^\s]+"
)


class CosignSigningError(RuntimeError):
    """Raised when Cosign fails to sign an image."""


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class SignatureAttestation:
    """
    Record of one Cosign signing event.

    Attributes:
        image_digest:       sha256 digest of the signed image.
        signed:             True if Cosign actually signed the image.
        oidc_issuer:        OIDC provider URL (e.g. GitHub Actions token issuer).
        signing_identity:   OIDC subject / workflow ref used as the signing identity.
        transparency_log_url: Rekor entry URL for independent verification.
        signature_bundle:   Full Cosign bundle dict (cert + sig JSON), empty if unsigned.
        skip_reason:        Why signing was skipped (only set when signed=False).
    """
    image_digest:         str
    signed:               bool
    oidc_issuer:          Optional[str]  = None
    signing_identity:     Optional[str]  = None
    transparency_log_url: Optional[str]  = None
    signature_bundle:     dict[str, Any] = field(default_factory=dict)
    skip_reason:          Optional[str]  = None


# ── Signer ────────────────────────────────────────────────────────────────────

class CosignSigner:
    """
    Signs an OCI image digest using Cosign keyless signing.

    Args:
        model_name:    For logging and DB record annotation.
        model_version: For logging and DB record annotation.
        pool:          asyncpg pool for cosign_attestations persistence.
    """

    def __init__(
        self,
        model_name:    str,
        model_version: str,
        pool:          Any,
    ) -> None:
        self._model_name    = model_name
        self._model_version = model_version
        self._pool          = pool

    async def sign(
        self,
        image_uri:    str,
        image_digest: str,
        workflow_id:  Optional[str] = None,
    ) -> SignatureAttestation:
        """
        Sign the image and persist the attestation.

        When settings.cosign_enabled=False (local dev), signing is skipped and
        a record with signed=False is written so the gate can make an informed
        decision (rather than blocking local development entirely).

        Args:
            image_uri:    Full digest-pinned image reference (tag@sha256:...).
            image_digest: sha256 digest string.
            workflow_id:  Optional FK to workflows table.

        Returns:
            SignatureAttestation with signing outcome.

        Raises:
            CosignSigningError: if cosign is enabled and the signing subprocess fails.
        """
        if not settings.cosign_enabled:
            logger.warning(
                "Cosign signing disabled (cosign_enabled=False) — "
                "skipping for %s v%s. Set cosign_enabled=true in production.",
                self._model_name,
                self._model_version,
            )
            attest = SignatureAttestation(
                image_digest= image_digest,
                signed=       False,
                skip_reason=  "cosign_enabled=False in settings",
            )
            await self._persist(attest, workflow_id)
            return attest

        # Canonical digest-pinned reference that Cosign signs
        digest_ref = f"{image_uri.split('@')[0]}@{image_digest}"
        logger.info("Cosign signing: %s", digest_ref)

        attest = self._run_cosign_sign(digest_ref, image_digest)
        await self._persist(attest, workflow_id)

        if attest.signed:
            logger.info(
                "Image signed: digest=%s rekor=%s",
                image_digest, attest.transparency_log_url,
            )
        else:
            logger.error(
                "Cosign signing failed for %s: %s",
                image_digest, attest.skip_reason,
            )

        return attest

    # ── Cosign subprocess ─────────────────────────────────────────────────────

    def _run_cosign_sign(
        self, digest_ref: str, image_digest: str
    ) -> SignatureAttestation:
        """
        Execute `cosign sign --yes <digest_ref>` and parse the output.

        cosign reads an ambient OIDC token from:
          - $ACTIONS_ID_TOKEN_REQUEST_TOKEN (GitHub Actions)
          - GKE Workload Identity (in-cluster)
          - SPIFFE/SPIRE sidecar

        Raises:
            CosignSigningError: if the binary is missing or exits non-zero.
        """
        cmd = [
            "cosign", "sign",
            "--yes",           # non-interactive — required for CI
            digest_ref,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
        except FileNotFoundError as exc:
            raise CosignSigningError(
                "cosign binary not found on PATH. "
                "Install cosign or set cosign_enabled=false for local dev."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise CosignSigningError(
                f"cosign sign exited {exc.returncode}.\n"
                f"stdout: {exc.stdout[:500]}\n"
                f"stderr: {exc.stderr[:500]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CosignSigningError(
                f"cosign sign timed out after 120s for {digest_ref!r}"
            ) from exc

        combined_output = proc.stdout + proc.stderr
        transparency_url = self._extract_rekor_url(combined_output)
        oidc_issuer, signing_identity = self._extract_identity(combined_output)

        return SignatureAttestation(
            image_digest=         image_digest,
            signed=               True,
            oidc_issuer=          oidc_issuer,
            signing_identity=     signing_identity,
            transparency_log_url= transparency_url,
            signature_bundle={
                "cosign_output": combined_output[:2000],
                "digest_ref":    digest_ref,
            },
        )

    # ── Output parsing ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_rekor_url(output: str) -> Optional[str]:
        """
        Extract the Rekor transparency log entry URL from cosign output.

        Cosign emits a line like:
          tlog entry created with index: 12345678
          Rekor entry: https://rekor.sigstore.dev/api/v1/log/entries?logIndex=12345678
        or the URL directly in the output.
        """
        m = _REKOR_URL_RE.search(output)
        if m:
            return m.group(0)

        # Alternative: extract from "tlog entry created with index: N"
        for line in output.splitlines():
            if "tlog entry created" in line.lower() or "rekor" in line.lower():
                parts = line.split()
                for part in parts:
                    if part.startswith("https://rekor"):
                        return part
        return None

    @staticmethod
    def _extract_identity(output: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract the OIDC issuer and signing identity from cosign output.

        Cosign 2.x prints lines like:
          Signing with identity: https://github.com/org/repo/.github/workflows/...
          with issuer: https://token.actions.githubusercontent.com
        """
        issuer   = None
        identity = None
        for line in output.splitlines():
            lower = line.lower()
            if "issuer:" in lower:
                issuer = line.split(":", 1)[-1].strip()
            elif "identity:" in lower or "signing with" in lower:
                identity = line.split(":", 1)[-1].strip()
        return issuer, identity

    # ── Persistence ────────────────────────────────────────────────────────────

    async def _persist(
        self, attest: SignatureAttestation, workflow_id: Optional[str]
    ) -> None:
        """INSERT into cosign_attestations."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO cosign_attestations (
                    image_digest, model_name, model_version,
                    signed, oidc_issuer, signing_identity,
                    transparency_log_url, signature_bundle, workflow_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
                """,
                attest.image_digest,
                self._model_name,
                self._model_version,
                attest.signed,
                attest.oidc_issuer,
                attest.signing_identity,
                attest.transparency_log_url,
                json.dumps(attest.signature_bundle),
                workflow_id,
            )