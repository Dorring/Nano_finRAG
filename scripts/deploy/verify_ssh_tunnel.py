#!/usr/bin/env python3
"""Verify SSH tunnel access to the frontend and backend.

Run this script **on the local machine** after establishing an SSH tunnel::

    ssh -N -L 18003:127.0.0.1:18003 -L 18002:127.0.0.1:18002 <user>@<server>

Then::

    python scripts/deploy/verify_ssh_tunnel.py

The script checks that:
1. The frontend page loads via the tunnel.
2. A real backend API request completes (login + query).
3. A ``trace_id`` is present in the query response.

Results are written to ``artifacts/deployment/phase7/ssh-tunnel-report.json``.
No full questions or answers are recorded.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional
import urllib.request
import urllib.error

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "deployment" / "phase7"
REPORT_PATH = ARTIFACT_DIR / "ssh-tunnel-report.json"

HTTP_TIMEOUT = 15.0

# Default tunnel ports (local end of the SSH tunnel).
FRONTEND_PORT = os.environ.get("FRONTEND_PORT", "18003")
BACKEND_PORT = os.environ.get("BACKEND_PORT", "18002")
FRONTEND_HOST = os.environ.get("FRONTEND_HOST", "127.0.0.1")
BACKEND_HOST = os.environ.get("BACKEND_HOST", "127.0.0.1")

# Test credentials — must match the backend's test user.
TEST_EMAIL = os.environ.get("TEST_EMAIL", "test@example.com")
TEST_PASSWORD = os.environ.get("TEST_PASSWORD", "testpassword123")


def _post_json(url: str, body: dict[str, Any]) -> tuple[int, Optional[dict]]:
    """POST JSON and return ``(status_code, parsed_body)``."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp else ""
        try:
            return exc.code, json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return exc.code, None
    except Exception:
        return 0, None


def _get(url: str, accept_json: bool = False) -> int:
    """Return HTTP status code for a GET request."""
    headers: dict[str, str] = {}
    if accept_json:
        headers["Accept"] = "application/json"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:
        return 0


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    frontend_url = f"http://{FRONTEND_HOST}:{FRONTEND_PORT}"
    backend_url = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

    print("[ssh_tunnel] Verifying SSH tunnel access...")

    # 1. Frontend page accessible via tunnel.
    frontend_status = _get(f"{frontend_url}/")
    frontend_accessible = frontend_status == 200
    print(f"  frontend: HTTP {frontend_status} ({'OK' if frontend_accessible else 'FAIL'})")

    # 2. Backend login via tunnel.
    login_status, login_body = _post_json(
        f"{backend_url}/token",
        {"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    token = None
    if login_status == 200 and isinstance(login_body, dict):
        token = login_body.get("access_token")
    backend_login_ok = token is not None
    print(f"  backend login: HTTP {login_status} ({'OK' if backend_login_ok else 'FAIL'})")

    # 3. Backend query via tunnel (records only metadata, not content).
    query_ok = False
    http_status = 0
    trace_id_present = False
    if token:
        query_status, query_body = _post_json(
            f"{backend_url}/query",
            {
                "question": "测试问题",
                "n_results": 1,
            },
        )
        http_status = query_status
        if query_status == 200 and isinstance(query_body, dict):
            query_ok = "answer" in query_body
            trace_id = query_body.get("trace_id")
            trace_id_present = bool(trace_id and isinstance(trace_id, str) and len(trace_id) > 0)
        print(f"  backend query: HTTP {query_status} ({'OK' if query_ok else 'FAIL'}), trace_id={'yes' if trace_id_present else 'no'}")

    tunnel_ok = frontend_accessible and query_ok and trace_id_present

    report = {
        "manifest_type": "phase7_ssh_tunnel",
        "schema_version": "1.0",
        "phase": "Phase 7: Rootless Online Serving",
        "verified_at": time.time(),
        "tunnel_established": True,
        "frontend_accessible": frontend_accessible,
        "backend_login_ok": backend_login_ok,
        "request_completed": query_ok,
        "http_status": http_status if query_ok else None,
        "trace_id_present": trace_id_present,
        "tunnel_ok": tunnel_ok,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    status_str = "PASS" if tunnel_ok else "FAIL"
    print(f"[ssh_tunnel] {status_str}")
    print(f"[ssh_tunnel] Wrote {REPORT_PATH}")
    return 0 if tunnel_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
