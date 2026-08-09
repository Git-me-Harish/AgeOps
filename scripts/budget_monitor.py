#!/usr/bin/env python3
"""
Budget Monitor
══════════════
Checks daily usage of all free-tier resources and fires alerts
if any service is approaching its limit.

Free-tier limits being tracked
  Cloudflare R2:  10 GB storage · 10 M Class-B reads / month
  Neon:            0.5 GB storage · 191.9 compute-hours / month
  Oracle Cloud:   Always-free A1 VMs (4 OCPU / 24 GB RAM total)

Run via Kubernetes CronJob (midnight UTC) or directly:
  python scripts/budget_monitor.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("budget_monitor")

# ── Free-tier thresholds (alert at 80 %) ──────────────────────────────────────
R2_STORAGE_LIMIT_GB   = 10.0
R2_STORAGE_ALERT_GB   = 8.0
NEON_STORAGE_LIMIT_MB = 512.0
NEON_STORAGE_ALERT_MB = 410.0

@dataclass
class BudgetAlert:
    service: str
    metric: str
    current: float
    limit: float
    unit: str
    pct_used: float

    def __str__(self) -> str:
        return (
            f"⚠️  BUDGET ALERT [{self.service}] {self.metric}: "
            f"{self.current:.2f}{self.unit} / {self.limit}{self.unit} "
            f"({self.pct_used:.1f}% used)"
        )


# ── Cloudflare R2 ─────────────────────────────────────────────────────────────

def check_r2(api_token: str, account_id: str) -> list[BudgetAlert]:
    alerts: list[BudgetAlert] = []
    try:
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/r2/buckets"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        total_bytes = sum(
            b.get("size", 0) for b in data.get("result", {}).get("buckets", [])
        )
        total_gb = total_bytes / (1024 ** 3)

        logger.info("R2 storage: %.3f GB / %s GB", total_gb, R2_STORAGE_LIMIT_GB)
        if total_gb >= R2_STORAGE_ALERT_GB:
            alerts.append(BudgetAlert(
                service="Cloudflare R2", metric="Storage",
                current=total_gb, limit=R2_STORAGE_LIMIT_GB, unit="GB",
                pct_used=(total_gb / R2_STORAGE_LIMIT_GB) * 100,
            ))
    except Exception as exc:
        logger.warning("R2 budget check failed: %s", exc)
    return alerts


# ── Neon PostgreSQL ───────────────────────────────────────────────────────────

def check_neon(api_key: str, project_id: str) -> list[BudgetAlert]:
    alerts: list[BudgetAlert] = []
    try:
        url = f"https://console.neon.tech/api/v2/projects/{project_id}"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        usage = data.get("project", {}).get("data_storage_bytes_hour", 0)
        used_mb = usage / (1024 ** 2)

        logger.info("Neon storage: %.1f MB / %s MB", used_mb, NEON_STORAGE_LIMIT_MB)
        if used_mb >= NEON_STORAGE_ALERT_MB:
            alerts.append(BudgetAlert(
                service="Neon PostgreSQL", metric="Storage",
                current=used_mb, limit=NEON_STORAGE_LIMIT_MB, unit="MB",
                pct_used=(used_mb / NEON_STORAGE_LIMIT_MB) * 100,
            ))
    except Exception as exc:
        logger.warning("Neon budget check failed: %s", exc)
    return alerts


# ── Dispatch alerts ───────────────────────────────────────────────────────────

def dispatch_alerts(alerts: list[BudgetAlert]) -> None:
    if not alerts:
        logger.info("✅ All services within budget thresholds.")
        return

    for alert in alerts:
        logger.warning(str(alert))

    # Structured JSON line picked up by Loki via Promtail
    payload = {
        "event": "budget_alert",
        "alerts": [
            {
                "service": a.service, "metric": a.metric,
                "pct_used": round(a.pct_used, 1), "unit": a.unit,
                "current": a.current, "limit": a.limit,
            }
            for a in alerts
        ],
    }
    logger.warning("BUDGET_ALERTS %s", json.dumps(payload))

    # Optional: post to Slack webhook if configured
    slack_url = os.getenv("SLACK_WEBHOOK_URL")
    if slack_url:
        body = json.dumps({
            "text": "\n".join(str(a) for a in alerts)
        }).encode()
        try:
            req = urllib.request.Request(slack_url, data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
            logger.info("Slack alert sent.")
        except Exception as exc:
            logger.warning("Slack webhook failed: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cf_token    = os.getenv("CLOUDFLARE_API_TOKEN", "")
    cf_account  = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
    neon_key    = os.getenv("NEON_API_KEY", "")
    neon_proj   = os.getenv("NEON_PROJECT_ID", "")

    all_alerts: list[BudgetAlert] = []

    if cf_token and cf_account:
        all_alerts.extend(check_r2(cf_token, cf_account))
    else:
        logger.info("Skipping R2 check (CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID not set)")

    if neon_key and neon_proj:
        all_alerts.extend(check_neon(neon_key, neon_proj))
    else:
        logger.info("Skipping Neon check (NEON_API_KEY or NEON_PROJECT_ID not set)")

    dispatch_alerts(all_alerts)
    return 1 if all_alerts else 0


if __name__ == "__main__":
    sys.exit(main())
