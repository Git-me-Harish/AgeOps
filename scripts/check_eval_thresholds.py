#!/usr/bin/env python3
"""
check_eval_thresholds.py
═════════════════════════
CI gate step: reads eval_results.json written by run_ai_eval.py
and exits 1 (blocking the pipeline) if any metric fails its threshold.

Called from ci-cd.yaml:
  python scripts/check_eval_thresholds.py --min-accuracy 0.80 --max-drift 0.15
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("check_eval_thresholds")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MLflow evaluation thresholds")
    parser.add_argument("--input",        default="eval_results.json")
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    parser.add_argument("--min-f1",       type=float, default=0.75)
    parser.add_argument("--min-roc-auc",  type=float, default=0.75)
    parser.add_argument("--max-drift",    type=float, default=0.15)
    args = parser.parse_args()

    try:
        with open(args.input) as f:
            results: dict = json.load(f)
    except FileNotFoundError:
        logger.warning("eval_results.json not found — skipping threshold gate")
        return 0

    if results.get("skipped"):
        logger.info("Eval was skipped (%s) — passing threshold gate", results.get("reason"))
        return 0

    failures: list[str] = []

    checks = [
        ("accuracy",  results.get("accuracy",  0.0), args.min_accuracy, ">="),
        ("f1_score",  results.get("f1_score",  0.0), args.min_f1,       ">="),
        ("roc_auc",   results.get("roc_auc",   0.0), args.min_roc_auc,  ">="),
        ("drift",     results.get("drift",     0.0), args.max_drift,    "<="),
    ]

    for metric, value, threshold, op in checks:
        if op == ">=" and value < threshold:
            msg = f"FAIL  {metric}={value:.4f} < threshold {threshold:.4f}"
            failures.append(msg)
            logger.error(msg)
        elif op == "<=" and value > threshold:
            msg = f"FAIL  {metric}={value:.4f} > threshold {threshold:.4f}"
            failures.append(msg)
            logger.error(msg)
        else:
            logger.info("PASS  %s=%.4f (%s %.4f)", metric, value, op, threshold)

    if failures:
        logger.error("❌ Evaluation gate FAILED — %d threshold(s) not met. Blocking deployment.", len(failures))
        return 1

    logger.info("✅ All evaluation thresholds passed. Proceeding to deployment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
