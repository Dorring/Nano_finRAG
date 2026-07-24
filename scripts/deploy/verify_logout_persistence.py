#!/usr/bin/env python3
"""Verify that all three services survived an SSH logout/reconnect cycle.

Run this script **after** reconnecting to the server via SSH. It checks
that the model, backend, and frontend services are still running and
healthy, then updates ``logout-persistence-report.json`` with the real
verification result.

Usage::

    python scripts/deploy/verify_logout_persistence.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
# Reuse env loading from collect_deployment_report.
sys.path.insert(0, str(SCRIPT_DIR))
from collect_deployment_report import (  # noqa: E402
    resolve_env,
    LOGOUT_PERSISTENCE_REPORT_PATH,
)

HTTP_TIMEOUT = 10.0


def _http_ok(url: str, accept_json: bool = True) -> bool:
    """Return True if GET ``url`` returns HTTP 200."""
    import urllib.request
    headers: dict[str, str] = {}
    if accept_json:
        headers["Accept"] = "application/json"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status == 200
    except Exception:
        return False


def main() -> int:
    env = resolve_env()
    model_url = f"http://{env.get('MODEL_HOST', '127.0.0.1')}:{env.get('MODEL_PORT', '18001')}"
    backend_url = f"http://{env.get('BACKEND_HOST', '127.0.0.1')}:{env.get('BACKEND_PORT', '18002')}"
    frontend_url = f"http://{env.get('FRONTEND_HOST', '127.0.0.1')}:{env.get('FRONTEND_PORT', '18003')}"

    print("[logout_persistence] Checking services after SSH reconnect...")
    model_ok = _http_ok(f"{model_url}/health")
    backend_ok = _http_ok(f"{backend_url}/healthz")
    frontend_ok = _http_ok(f"{frontend_url}/", accept_json=False)

    all_ok = model_ok and backend_ok and frontend_ok
    verified_at = time.time()

    report = {
        "manifest_type": "phase7_logout_persistence",
        "schema_version": "1.0",
        "phase": "Phase 7: Rootless Online Serving",
        "verification_status": "verified" if all_ok else "failed",
        "logout_persistent": all_ok,
        "verified_at": verified_at,
        "checks": {
            "model_healthy": model_ok,
            "backend_healthy": backend_ok,
            "frontend_healthy": frontend_ok,
        },
        "description": (
            "Verified that all three services (model, backend, frontend) "
            "remain running after SSH session disconnects, using tmux "
            "sessions that persist independently of the SSH connection."
        ),
        "notes": [
            "tmux sessions persist independently of SSH connections",
            "Server reboot requires manual restart: bash scripts/deploy/start_all.sh",
            "No systemd or auto-restart on reboot (rootless constraint)",
        ],
    }

    LOGOUT_PERSISTENCE_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    status_str = "PASS" if all_ok else "FAIL"
    print(f"[logout_persistence] {status_str}")
    print(f"  model:   {'healthy' if model_ok else 'unhealthy'}")
    print(f"  backend: {'healthy' if backend_ok else 'unhealthy'}")
    print(f"  frontend: {'healthy' if frontend_ok else 'unhealthy'}")
    print(f"[logout_persistence] Wrote {LOGOUT_PERSISTENCE_REPORT_PATH}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
