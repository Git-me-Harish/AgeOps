"""
serving/ — Standardised inference server embedded in every model OCI image.

This package is copied wholesale into every model container at build time.
It must remain self-contained — no imports from agents/, configs/, or scripts/.

Modules:
  inference_server  — FastAPI app (ENTRYPOINT target)
  health            — Liveness / readiness state singleton
  metrics           — Prometheus metrics singleton
"""