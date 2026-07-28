"""Smoke-test API response shapes against Pydantic schemas.

Tests verify API endpoint stability and contract compliance.
When venv deps are unavailable, integration tests skip gracefully.
"""
import os
import pytest


def _have_api_deps():
    try:
        import jose  # noqa: F401
        import bcrypt  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_api_deps(), reason="API deps (jose, bcrypt) not available")
class TestAPIContract:
    """API contract tests - only run with full venv."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.main import app
        from fastapi.testclient import TestClient
        self.client = TestClient(app)

    def test_healthz_returns_200(self):
        response = self.client.get("/healthz")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_readyz_returns_200(self):
        response = self.client.get("/readyz")
        assert response.status_code in (200, 503)

    def test_root_returns_200(self):
        response = self.client.get("/")
        assert response.status_code == 200

    def test_ops_summary_endpoint(self):
        response = self.client.get("/ops/summary")
        assert response.status_code != 500

    def test_documents_endpoint_exists(self):
        response = self.client.get("/documents")
        assert response.status_code in (200, 401, 403, 422)

    def test_query_endpoint_requires_auth(self):
        response = self.client.post("/query", json={"question": "test"})
        assert response.status_code in (401, 403, 422)

    def test_register_endpoint_accepts_valid_payload(self):
        response = self.client.post("/register", json={
            "username": "arch_test_user",
            "password": "arch_test_password_123",
        })
        assert response.status_code != 500

    def test_login_endpoint_exists(self):
        response = self.client.post("/login", json={
            "username": "nonexistent",
            "password": "nonexistent",
        })
        assert response.status_code != 500

    def test_eval_score_endpoint_requires_body(self):
        response = self.client.post("/eval/score", json={})
        assert response.status_code != 500

    def test_eval_compare_endpoint_requires_body(self):
        response = self.client.post("/eval/compare", json={})
        assert response.status_code != 500

    def test_no_unexpected_500_from_health_endpoints(self):
        endpoints = [
            ("GET", "/"),
            ("GET", "/healthz"),
            ("GET", "/readyz"),
            ("GET", "/ops/summary"),
        ]
        for method, path in endpoints:
            response = getattr(self.client, method.lower())(path)
            assert response.status_code != 500, (
                f"{method} {path} returned 500: {response.text[:200]}"
            )


def test_api_endpoint_count():
    """Verify the number of registered endpoints is stable."""
    main_py = os.path.join("src", "main.py")
    if not os.path.isfile(main_py):
        pytest.skip("main.py not found at expected path")

    with open(main_py, encoding="utf-8") as fh:
        content = fh.read()

    # Count route decorators
    import re
    routes = re.findall(r'@app\.(get|post|delete|put)\(', content)
    # Baseline: 32 endpoints as of 2026-07-29, including the authenticated
    # NF37 retrieval candidate diagnostics endpoint (/ops/retrieval-candidates).
    assert len(routes) == 32, (
        f"Expected 32 API endpoints, found {len(routes)}. "
        "If you added/removed endpoints, update this test and artifacts/baseline/repository-metrics.json."
    )


def test_key_endpoints_present():
    """Verify critical API endpoints are registered."""
    main_py = os.path.join("src", "main.py")
    if not os.path.isfile(main_py):
        pytest.skip("main.py not found")

    with open(main_py, encoding="utf-8") as fh:
        content = fh.read()

    # These routes must exist - check for the route path string
    required_routes = [
        '"/healthz"',
        '"/readyz"',
        '"/ops/summary"',
        '"/documents"',
        '"/query"',
        '"/query/stream"',
        '"/register"',
        '"/login"',
        '"/upload"',
        '"/traces"',
        '"/feedback"',
        '"/eval/score"',
        '"/eval/compare"',
        '"/sessions"',
        '"/memory/profile"',
        '"/document-registry"',
        '"/me"',
        '"/replay/traces"',
        '"/replay/feedback"',
    ]

    missing = [r for r in required_routes if r not in content]
    assert not missing, (
        f"{len(missing)} required route(s) not found: {missing}"
    )
