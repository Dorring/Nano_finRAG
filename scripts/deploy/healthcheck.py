#!/usr/bin/env python3
"""Phase 7 rootless online serving health check.

Standalone script (stdlib + urllib only) that probes the model, backend,
and frontend services. Prints a JSON report to stdout and writes a copy
to ``artifacts/deployment/phase7/health-report.json``.

Exit code 0 when all services are healthy, 1 otherwise.

Usage::

    python scripts/deploy/healthcheck.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CONFIG_DIR = REPO_ROOT / "config" / "deployment"
ENV_FILE_PRIMARY = CONFIG_DIR / "online.env"
ENV_FILE_FALLBACK = CONFIG_DIR / "online.env.example"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "deployment" / "phase7"
HEALTH_REPORT_PATH = ARTIFACT_DIR / "health-report.json"
HTTP_TIMEOUT_SECONDS = 10.0


def load_env_config(env_file_path: Path) -> Mapping[str, str]:
    """Load ``KEY=VALUE`` pairs from an env file into a dict.

    Skips blank lines and comments (lines starting with ``#``). Values are
    taken verbatim without shell expansion.
    """
    config: dict[str, str] = {}
    if not env_file_path.is_file():
        return config
    with env_file_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            config[key] = value.strip()
    return config


def resolve_env() -> Mapping[str, str]:
    """Resolve the effective environment, preferring ``online.env`` over ``.example``.

    File values override process environment variables to mirror the
    ``set -a; . file`` semantics used by ``scripts/deploy/load_env.sh``.
    """
    env_file = ENV_FILE_PRIMARY if ENV_FILE_PRIMARY.is_file() else ENV_FILE_FALLBACK
    merged: dict[str, str] = dict(os.environ)
    merged.update(load_env_config(env_file))
    return merged


def http_get_json(url: str, timeout: float = HTTP_TIMEOUT_SECONDS, accept_json: bool = True) -> tuple[int, Optional[Any]]:
    """Perform an HTTP GET and return ``(status_code, parsed_json_or_None)``.

    Returns ``(0, None)`` on network/timeout errors so callers can mark a
    service unhealthy without propagating exceptions or stack traces.

    When ``accept_json`` is True (default), sends ``Accept: application/json``.
    Set to False for endpoints that serve HTML (e.g. Vite dev server root),
    since some dev servers return 404 for JSON content negotiation.
    """
    headers: dict[str, str] = {}
    if accept_json:
        headers["Accept"] = "application/json"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200))
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, None
    parsed: Optional[Any] = None
    if body:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
    return status, parsed


def http_get_status(url: str, timeout: float = HTTP_TIMEOUT_SECONDS, accept_json: bool = True) -> int:
    """Return only the HTTP status code for a GET request (0 on failure)."""
    status, _ = http_get_json(url, timeout=timeout, accept_json=accept_json)
    return status


def sanitize_url(url: str) -> str:
    """Return a URL safe for inclusion in reports.

    Loopback / localhost URLs are retained (per spec example output).
    Any other host is masked as ``<host>``.
    """
    if "127.0.0.1" in url or "localhost" in url or "0.0.0.0" in url:
        return url
    if "://" in url:
        scheme, _, rest = url.partition("://")
        host, sep, path = rest.partition("/")
        return f"{scheme}://<host>/{path}" if sep else f"{scheme}://<host>"
    return url


def check_model_service(base_url: str, expected_model_name: str) -> dict[str, Any]:
    """Check model service ``/health`` and ``/v1/models`` endpoints."""
    report: dict[str, Any] = {"url": sanitize_url(base_url)}
    health_status, _ = http_get_json(f"{base_url}/health")
    models_status, models_body = http_get_json(f"{base_url}/v1/models")
    healthy = health_status == 200 and models_status == 200
    model_name: Optional[str] = None
    if isinstance(models_body, dict):
        data = models_body.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                model_name = first.get("id") or first.get("name")
    if expected_model_name and model_name != expected_model_name:
        healthy = False
    report["status"] = "healthy" if healthy else "unhealthy"
    if model_name:
        report["model_name"] = model_name
    return report


def check_backend_service(base_url: str) -> dict[str, Any]:
    """Check backend service ``/healthz`` endpoint."""
    report: dict[str, Any] = {"url": sanitize_url(base_url)}
    status = http_get_status(f"{base_url}/healthz")
    report["status"] = "healthy" if status == 200 else "unhealthy"
    return report


def check_frontend_service(base_url: str) -> dict[str, Any]:
    """Check frontend service root page."""
    report: dict[str, Any] = {"url": sanitize_url(base_url)}
    # Vite dev server returns 404 for Accept: application/json on root;
    # use a plain GET without content negotiation.
    status = http_get_status(f"{base_url}/", accept_json=False)
    report["status"] = "healthy" if status == 200 else "unhealthy"
    return report


def build_report(env: Mapping[str, str]) -> dict[str, Any]:
    """Build the full health report dict from resolved env vars."""
    model_host = env.get("MODEL_HOST", "127.0.0.1")
    model_port = env.get("MODEL_PORT", "18001")
    backend_host = env.get("BACKEND_HOST", "127.0.0.1")
    backend_port = env.get("BACKEND_PORT", "18002")
    frontend_host = env.get("FRONTEND_HOST", "127.0.0.1")
    frontend_port = env.get("FRONTEND_PORT", "18003")
    expected_model = env.get("MODEL_NAME", env.get("LLM_MODEL_NAME", ""))

    model_url = f"http://{model_host}:{model_port}"
    backend_url = f"http://{backend_host}:{backend_port}"
    frontend_url = f"http://{frontend_host}:{frontend_port}"

    model_report = check_model_service(model_url, expected_model)
    backend_report = check_backend_service(backend_url)
    frontend_report = check_frontend_service(frontend_url)

    overall = "healthy"
    if (
        model_report["status"] != "healthy"
        or backend_report["status"] != "healthy"
        or frontend_report["status"] != "healthy"
    ):
        overall = "unhealthy"

    return {
        "model": model_report,
        "backend": backend_report,
        "frontend": frontend_report,
        "overall": overall,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    """Write the report JSON to ``path``, creating parent dirs as needed.

    Failures are swallowed so artifact IO never changes the exit code.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except OSError:
        pass


def main() -> int:
    """Run all health checks, print report, return exit code."""
    env = resolve_env()
    report = build_report(env)
    print(json.dumps(report, indent=2, sort_keys=True))
    write_report(report, HEALTH_REPORT_PATH)
    return 0 if report["overall"] == "healthy" else 1


if __name__ == "__main__":
    sys.exit(main())
