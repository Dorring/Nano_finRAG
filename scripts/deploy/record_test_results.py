#!/usr/bin/env python3
"""Run pytest and write a summary JSON for the Phase 7 acceptance report.

Usage::

    python scripts/deploy/record_test_results.py

The script runs ``pytest -q`` (excluding torch-only modules that cannot be
collected on the deployment server), parses the terminal summary line, and
writes ``runtime/phase7/test-results.json``.  ``collect_deployment_report.py``
reads this file to set acceptance criteria #37-#39.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_PATH = REPO_ROOT / "runtime" / "phase7" / "test-results.json"

# Modules that require torch (not available on the deployment server).
IGNORE_ARGS = [
    "--ignore=tests/test_attention_fallback.py",
    "--ignore=tests/test_engine.py",
]


def _ensure_output_dir() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _parse_summary(text: str) -> dict:
    """Extract pass/skip/fail/error counts from pytest terminal output."""
    passed = 0
    skipped = 0
    failed = 0
    errors = 0
    for m in re.finditer(
        r"(\d+)\s+(passed|skipped|failed|error|errors)\b", text
    ):
        count = int(m.group(1))
        kind = m.group(2)
        if kind == "passed":
            passed = count
        elif kind == "skipped":
            skipped = count
        elif kind == "failed":
            failed = count
        elif kind in ("error", "errors"):
            errors = count
    return {
        "total": passed + skipped + failed + errors,
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
    }


def main() -> int:
    _ensure_output_dir()

    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short"] + IGNORE_ARGS
    print(f"[record_test_results] Running: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=1800,
    )

    output = proc.stdout + "\n" + proc.stderr
    summary = _parse_summary(output)
    summary["exit_code"] = proc.returncode
    summary["command"] = " ".join(cmd)

    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[record_test_results] Wrote {OUTPUT_PATH}")
    print(
        f"[record_test_results] "
        f"{summary['passed']} passed, {summary['skipped']} skipped, "
        f"{summary['failed']} failed, {summary['errors']} errors"
    )
    return 0 if summary["failed"] == 0 and summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
