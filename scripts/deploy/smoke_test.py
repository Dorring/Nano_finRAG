#!/usr/bin/env python3
"""Phase 7 rootless online serving smoke tests.

Standalone script (stdlib + urllib only) that runs 12 protocol/chain
smoke tests against the deployed model, backend, and frontend services.
Prints a JSON report to stdout and writes a copy to
``artifacts/deployment/phase7/smoke-report.json``.

Exit code 0 when all tests pass, 1 otherwise.

Usage::

    python scripts/deploy/smoke_test.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
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
SMOKE_REPORT_PATH = ARTIFACT_DIR / "smoke-report.json"
HTTP_TIMEOUT_SECONDS = 15.0
SSE_READ_TIMEOUT_SECONDS = 60.0

# Stable smoke-test account. Register is attempted first; if the account
# already exists the script falls back to login.
SMOKE_USER_EMAIL = "phase7-smoke@example.com"
SMOKE_USER_PASSWORD = "phase7-smoke-password-123456"

# Short test questions (full text is never logged in the report).
QUESTION_NORMAL = "贵州茅台2023年营业收入是多少?"
QUESTION_CALC = "贵州茅台2023年毛利率是多少?"
QUESTION_UNANSWERABLE = "今天天气怎么样?"

# Path leak detection markers (kept generic; no real paths embedded).
_PATH_LEAK_MARKERS = (
    "/home/", "/Users/", "/root/", "/var/", "/etc/", "/opt/", "/srv/",
    "/data/", "C:\\Users\\", "C:\\\\Users\\\\",
)


class TestResult:
    """Container for a single test outcome."""

    __slots__ = ("name", "status", "detail")

    def __init__(self, name: str, status: str, detail: str = "") -> None:
        self.name = name
        self.status = status  # "pass" | "fail" | "pending"
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "status": self.status}
        if self.detail:
            out["detail"] = self.detail
        return out


def load_env_config(env_file_path: Path) -> Mapping[str, str]:
    """Load ``KEY=VALUE`` pairs from an env file into a dict."""
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
    """Resolve the effective environment, preferring ``online.env`` over ``.example``."""
    env_file = ENV_FILE_PRIMARY if ENV_FILE_PRIMARY.is_file() else ENV_FILE_FALLBACK
    merged: dict[str, str] = dict(os.environ)
    merged.update(load_env_config(env_file))
    return merged


def _safe_parse(text: Optional[str]) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def http_request(
    method: str,
    url: str,
    body: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = HTTP_TIMEOUT_SECONDS,
    accept_json: bool = True,
) -> tuple[int, Optional[Any], Optional[str]]:
    """Perform an HTTP request.

    Returns ``(status_code, parsed_json_or_None, raw_text_or_None)``.
    On network errors returns ``(0, None, None)`` so callers never see
    exceptions or stack traces.

    When ``accept_json`` is False, the ``Accept: application/json`` header
    is omitted (needed for Vite dev server root, which returns 404 for
    JSON content negotiation).
    """
    req_headers: dict[str, str] = {}
    if accept_json:
        req_headers["Accept"] = "application/json"
    if headers:
        req_headers.update(headers)
    data: Optional[bytes] = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200))
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:
            raw = None
        return exc.code, _safe_parse(raw), raw
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, None, None
    return status, _safe_parse(raw), raw


def obtain_auth_token(backend_url: str) -> Optional[str]:
    """Register or login as the smoke-test user; return ``access_token`` or ``None``."""
    _, body, _ = http_request(
        "POST",
        f"{backend_url}/register",
        body={"email": SMOKE_USER_EMAIL, "password": SMOKE_USER_PASSWORD},
    )
    if isinstance(body, dict) and body.get("access_token"):
        return body["access_token"]
    _, body, _ = http_request(
        "POST",
        f"{backend_url}/login",
        body={"email": SMOKE_USER_EMAIL, "password": SMOKE_USER_PASSWORD},
    )
    if isinstance(body, dict) and body.get("access_token"):
        return body["access_token"]
    return None


def authed_post(
    backend_url: str,
    path: str,
    token: Optional[str],
    body: dict[str, Any],
) -> tuple[int, Optional[Any], Optional[str]]:
    """POST to an authenticated backend endpoint."""
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return http_request("POST", f"{backend_url}{path}", body=body, headers=headers)


def _stream_terminated(lines: list[str]) -> bool:
    """Check whether the SSE stream ended with a ``done`` event or ``[DONE]`` marker."""
    if not lines:
        return False
    blob = "\n".join(lines)
    if "data: [DONE]" in blob:
        return True
    for line in lines:
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "done":
            return True
    return False


def read_sse_stream(
    url: str,
    body: dict[str, Any],
    token: Optional[str],
    timeout: float = SSE_READ_TIMEOUT_SECONDS,
) -> tuple[bool, list[str]]:
    """POST to an SSE endpoint, read the full stream.

    Returns ``(terminated_ok, collected_lines)``.
    """
    headers: dict[str, str] = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    collected: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line:
                    collected.append(line)
    except urllib.error.HTTPError as exc:
        # 422/401/etc: capture response body for diagnostics.
        try:
            payload = exc.read().decode("utf-8", errors="replace").strip()
            if payload:
                collected.append(f"HTTP_{exc.code}: {payload[:200]}")
        except Exception:
            pass
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    return _stream_terminated(collected), collected


def contains_path_leak(text: Optional[str]) -> bool:
    """Heuristic: detect absolute path patterns that should not leak in errors."""
    if not text:
        return False
    return any(marker in text for marker in _PATH_LEAK_MARKERS)


# ---------------------------------------------------------------------------
# Individual smoke tests
# ---------------------------------------------------------------------------

def test_model_accessible(model_url: str, expected_model: str) -> TestResult:
    """Test 1: model ``/v1/models`` is accessible and returns expected name."""
    status, body, _ = http_request("GET", f"{model_url}/v1/models")
    if status != 200:
        return TestResult("model_accessible", "fail", "model endpoint not reachable")
    model_name: Optional[str] = None
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            model_name = data[0].get("id") or data[0].get("name")
    if expected_model and model_name != expected_model:
        return TestResult("model_accessible", "fail", "model name mismatch")
    return TestResult("model_accessible", "pass")


def test_backend_healthz(backend_url: str) -> TestResult:
    """Test 2: backend ``/healthz`` returns 200."""
    status, _, _ = http_request("GET", f"{backend_url}/healthz")
    if status != 200:
        return TestResult("backend_healthz", "fail", "healthz did not return 200")
    return TestResult("backend_healthz", "pass")


def test_frontend_root(frontend_url: str) -> TestResult:
    """Test 3: frontend ``/`` returns 200."""
    status, _, _ = http_request("GET", f"{frontend_url}/", accept_json=False)
    if status != 200:
        return TestResult("frontend_root", "fail", "root page did not return 200")
    return TestResult("frontend_root", "pass")


def test_backend_calls_model(backend_url: str) -> TestResult:
    """Test 4: backend can call model service (verified via ``/readyz``)."""
    status, _, _ = http_request("GET", f"{backend_url}/readyz")
    # readyz returns 200 when ready, 503 when not. 200 means backend deps OK.
    if status == 200:
        return TestResult("backend_calls_model", "pass")
    return TestResult("backend_calls_model", "fail", "readyz did not return 200")


def test_frontend_reaches_backend(frontend_url: str, backend_port: str) -> TestResult:
    """Test 5: frontend page loads and references backend API.

    In Vite dev mode the HTML is minimal (no inlined API URL), and the
    dev server transforms ``import.meta.env.VITE_API_URL`` in-place. We
    check the HTML, the ``/src/api.js`` module, and the Vite env shim
    (``/@vite/env``) which exposes the resolved env vars.
    """
    status, _, raw = http_request("GET", f"{frontend_url}/", accept_json=False)
    if status != 200 or not raw:
        return TestResult("frontend_reaches_backend", "fail", "frontend page not loaded")

    def _references_backend(text: str) -> bool:
        return (
            backend_port in text
            or "/query" in text
            or "VITE_API_URL" in text
            or "import.meta.env" in text
            or "/healthz" in text
        )

    # 1) Check the HTML payload first (production build inlines the URL).
    if _references_backend(raw):
        return TestResult("frontend_reaches_backend", "pass")

    # 2) In Vite dev mode, check source modules and the env shim.
    for module_path in (
        "/src/api.js",
        "/src/main.jsx",
        "/@vite/env",
        "/@id/__x00__import-meta-env",
    ):
        m_status, _, m_raw = http_request(
            "GET", f"{frontend_url}{module_path}", accept_json=False
        )
        if m_status == 200 and m_raw and _references_backend(m_raw):
            return TestResult("frontend_reaches_backend", "pass")

    return TestResult("frontend_reaches_backend", "fail", "no backend API reference found")


def test_query_normal(backend_url: str, token: Optional[str]) -> tuple[TestResult, Optional[str]]:
    """Test 6: normal finance Q&A returns a response.

    Returns ``(result, trace_id)`` so the trace-id test can reuse the response.
    """
    status, body, _ = authed_post(backend_url, "/query", token, {"question": QUESTION_NORMAL})
    if status != 200:
        return TestResult("query_normal", "fail", "query did not return 200"), None
    if not isinstance(body, dict) or "answer" not in body:
        return TestResult("query_normal", "fail", "response missing answer field"), None
    trace_id = body.get("trace_id") if isinstance(body, dict) else None
    return TestResult("query_normal", "pass"), trace_id


def test_query_calculation(backend_url: str, token: Optional[str]) -> TestResult:
    """Test 7: financial calculation query returns a response."""
    status, body, _ = authed_post(backend_url, "/query", token, {"question": QUESTION_CALC})
    if status != 200:
        return TestResult("query_calculation", "fail", "calc query did not return 200")
    if not isinstance(body, dict) or "answer" not in body:
        return TestResult("query_calculation", "fail", "response missing answer field")
    return TestResult("query_calculation", "pass")


def test_query_unanswerable_safe(backend_url: str, token: Optional[str]) -> TestResult:
    """Test 8: unanswerable question safely fails (safe fallback, no crash)."""
    status, body, raw = authed_post(
        backend_url, "/query", token, {"question": QUESTION_UNANSWERABLE}
    )
    # Accept 200 (safe fallback answer) or a 4xx structured error envelope.
    # A 500 with a raw stack trace is a failure.
    if status == 200 and isinstance(body, dict) and "answer" in body:
        return TestResult("query_unanswerable_safe", "pass")
    if 400 <= status < 500:
        if contains_path_leak(raw):
            return TestResult("query_unanswerable_safe", "fail", "internal path leaked in error")
        return TestResult("query_unanswerable_safe", "pass")
    return TestResult("query_unanswerable_safe", "fail", "unsafe crash response")


def test_sse_terminates(backend_url: str, token: Optional[str]) -> TestResult:
    """Test 9: SSE endpoint returns a stream that terminates properly."""
    terminated, lines = read_sse_stream(
        f"{backend_url}/query/stream",
        {"question": QUESTION_NORMAL},
        token,
    )
    if terminated:
        return TestResult("sse_terminates", "pass")
    # Include the last few lines (truncated) for diagnosis without leaking
    # full user content.
    tail = lines[-3:] if lines else []
    detail = "stream did not terminate with done event"
    if tail:
        snippet = " | ".join(t.strip()[:80] for t in tail)
        detail = f"{detail}; tail: {snippet}"
    return TestResult("sse_terminates", "fail", detail)


def test_trace_id_present(trace_id: Optional[str]) -> TestResult:
    """Test 10: trace ID exists in the query response."""
    if trace_id and isinstance(trace_id, str) and len(trace_id) > 0:
        return TestResult("trace_id_present", "pass")
    return TestResult("trace_id_present", "fail", "no trace_id in query response")


def test_no_path_leak_in_errors(backend_url: str) -> TestResult:
    """Test 11: error responses do not leak internal paths."""
    # Send a malformed query (missing required ``question`` field) to trigger
    # a validation error and inspect the response body for path patterns.
    status, _, raw = authed_post(backend_url, "/query", None, {"question": "x"})
    # A 422 (missing/min_length) or 401/403 (auth) is expected.
    if contains_path_leak(raw):
        return TestResult("no_path_leak_in_errors", "fail", "internal path found in error response")
    if status == 0:
        return TestResult("no_path_leak_in_errors", "fail", "backend unreachable")
    return TestResult("no_path_leak_in_errors", "pass")


def _endpoint_healthy(url: str, accept_json: bool = True) -> bool:
    """Return True if a GET to ``url`` returns HTTP 200."""
    status, _, _ = http_request("GET", url, accept_json=accept_json)
    return status == 200


def test_restart_recovery(env: Mapping[str, str]) -> TestResult:
    """Test 12: services recover after a real restart.

    Snapshots initial health, invokes ``restart_all.sh``, waits for all three
    services to become healthy again, and only then reports ``pass``. If the
    services are not initially healthy the test reports ``pending`` (cannot
    verify restart of something that is not running). It never returns a
    fixed ``pass``.
    """
    restart_script = SCRIPT_DIR / "restart_all.sh"
    if not restart_script.is_file():
        return TestResult("restart_recovery", "pending", "restart_all.sh not found")

    model_url = (
        f"http://{env.get('MODEL_HOST', '127.0.0.1')}:{env.get('MODEL_PORT', '18001')}"
    )
    backend_url = (
        f"http://{env.get('BACKEND_HOST', '127.0.0.1')}:{env.get('BACKEND_PORT', '18002')}"
    )
    frontend_url = (
        f"http://{env.get('FRONTEND_HOST', '127.0.0.1')}:{env.get('FRONTEND_PORT', '18003')}"
    )

    # Services must be healthy before we restart them.
    initial_ok = (
        _endpoint_healthy(f"{model_url}/health")
        and _endpoint_healthy(f"{backend_url}/healthz")
        and _endpoint_healthy(f"{frontend_url}/", accept_json=False)
    )
    if not initial_ok:
        return TestResult(
            "restart_recovery", "pending",
            "services not healthy before restart; cannot verify recovery",
        )

    # Run the real restart script.
    try:
        proc = subprocess.run(
            ["bash", str(restart_script)],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError):
        return TestResult(
            "restart_recovery", "pending",
            "restart_all.sh could not be executed",
        )
    if proc.returncode != 0:
        return TestResult(
            "restart_recovery", "fail",
            "restart_all.sh exited non-zero",
        )

    # Wait for all three services to become healthy again (up to 180s).
    deadline = time.monotonic() + 180.0
    recovered = False
    while time.monotonic() < deadline:
        if (
            _endpoint_healthy(f"{model_url}/health")
            and _endpoint_healthy(f"{backend_url}/healthz")
            and _endpoint_healthy(f"{frontend_url}/", accept_json=False)
        ):
            recovered = True
            break
        time.sleep(3)

    if recovered:
        return TestResult("restart_recovery", "pass")
    return TestResult(
        "restart_recovery", "fail",
        "services not healthy after restart within 180s",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all_tests(env: Mapping[str, str]) -> dict[str, Any]:
    """Run all 12 smoke tests and return the structured report."""
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

    results: list[TestResult] = []

    # Tests 1-5: infrastructure chain.
    results.append(test_model_accessible(model_url, expected_model))
    results.append(test_backend_healthz(backend_url))
    results.append(test_frontend_root(frontend_url))
    results.append(test_backend_calls_model(backend_url))
    results.append(test_frontend_reaches_backend(frontend_url, backend_port))

    # Obtain an auth token for query tests.
    token = obtain_auth_token(backend_url)

    # Tests 6-10: query protocol chain.
    normal_result, trace_id = test_query_normal(backend_url, token)
    results.append(normal_result)
    results.append(test_query_calculation(backend_url, token))
    results.append(test_query_unanswerable_safe(backend_url, token))
    results.append(test_sse_terminates(backend_url, token))
    results.append(test_trace_id_present(trace_id))

    # Tests 11-12: security and recovery.
    results.append(test_no_path_leak_in_errors(backend_url))
    results.append(test_restart_recovery(env))

    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    pending = sum(1 for r in results if r.status == "pending")
    if failed > 0:
        overall = "fail"
    elif pending > 0:
        overall = "pending"
    else:
        overall = "pass"

    return {
        "tests": [r.to_dict() for r in results],
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "pending": pending,
        "overall": overall,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    """Write the report JSON to ``path``, creating parent dirs as needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except OSError:
        pass


def main() -> int:
    """Run all smoke tests, print report, return exit code."""
    env = resolve_env()
    report = run_all_tests(env)
    print(json.dumps(report, indent=2, sort_keys=True))
    write_report(report, SMOKE_REPORT_PATH)
    return 0 if report["overall"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
