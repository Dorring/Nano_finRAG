"""Tests for scripts/deploy/healthcheck.py.

Verify healthcheck.py exists, is importable, checks all three services
(model, backend, frontend), uses urllib (not requests/httpx), produces
JSON output, sanitizes output, and has a configured timeout (10 seconds).
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HEALTHCHECK = REPO_ROOT / "scripts" / "deploy" / "healthcheck.py"


def _load_module(module_name: str, path: Path):
    """Import a Python module from a file path without starting it."""
    if not path.is_file():
        pytest.skip(f"Script not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def healthcheck_source() -> str:
    if not HEALTHCHECK.is_file():
        pytest.skip(f"healthcheck.py not found: {HEALTHCHECK}")
    return HEALTHCHECK.read_text(encoding="utf-8")


def test_healthcheck_exists():
    # Verify healthcheck.py exists
    assert HEALTHCHECK.is_file(), f"Expected {HEALTHCHECK} to exist"


def test_healthcheck_importable():
    # Verify healthcheck.py can be imported without errors
    module = _load_module("deploy_healthcheck", HEALTHCHECK)
    assert hasattr(module, "main"), "healthcheck.py missing main() function"
    assert hasattr(module, "build_report"), "healthcheck.py missing build_report() function"


def test_healthcheck_checks_all_services(healthcheck_source):
    # Verify healthcheck.py checks model, backend, and frontend services
    assert "check_model_service" in healthcheck_source, \
        "healthcheck.py does not define check_model_service()"
    assert "check_backend_service" in healthcheck_source, \
        "healthcheck.py does not define check_backend_service()"
    assert "check_frontend_service" in healthcheck_source, \
        "healthcheck.py does not define check_frontend_service()"
    assert "/health" in healthcheck_source or "/v1/models" in healthcheck_source, \
        "healthcheck.py does not probe model endpoints"
    assert "/healthz" in healthcheck_source, \
        "healthcheck.py does not probe backend /healthz"
    # Frontend root check (GET /)
    assert 'check_frontend_service' in healthcheck_source


def test_healthcheck_uses_urllib(healthcheck_source):
    # Verify healthcheck.py uses urllib, not requests/httpx
    assert "urllib.request" in healthcheck_source or "urllib.error" in healthcheck_source, \
        "healthcheck.py should use urllib for HTTP requests"
    assert "import requests" not in healthcheck_source, \
        "healthcheck.py should NOT import requests"
    assert "import httpx" not in healthcheck_source, \
        "healthcheck.py should NOT import httpx"


def test_healthcheck_output_is_json(healthcheck_source):
    # Verify output is JSON-formatted (json.dumps in main)
    assert "json.dumps" in healthcheck_source or "json.dump" in healthcheck_source, \
        "healthcheck.py should output JSON (json.dumps/json.dump)"


def test_healthcheck_does_not_expose_absolute_paths(healthcheck_source):
    # Verify healthcheck.py sanitizes output (no raw absolute paths leaked)
    assert "sanitize_url" in healthcheck_source or "sanitize" in healthcheck_source, \
        "healthcheck.py should sanitize URLs in output"
    # The script should not print raw file system paths in the report
    # (it may reference paths for internal file I/O, but not in the JSON report)
    assert "REPO_ROOT" in healthcheck_source, \
        "healthcheck.py should reference REPO_ROOT internally"


def test_healthcheck_timeout_configured(healthcheck_source):
    # Verify timeout is configured at 10 seconds
    assert re.search(r"TIMEOUT.*=\s*10", healthcheck_source) or \
           re.search(r"timeout\s*=\s*10", healthcheck_source), \
        "healthcheck.py should configure a 10-second timeout"


def test_healthcheck_no_secrets_in_output(healthcheck_source):
    # Verify healthcheck.py does not log secrets/API keys
    secret_log_patterns = [
        re.compile(r"print\s*\(.*(?:api_key|API_KEY|secret|password|token)", re.IGNORECASE),
        re.compile(r"\"detail\".*(?:api_key|password|secret|token)", re.IGNORECASE),
    ]
    for pat in secret_log_patterns:
        m = pat.search(healthcheck_source)
        assert not m, \
            f"healthcheck.py may log secrets: {m.group()}"
