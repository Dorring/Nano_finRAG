"""Phase 3 real HTTP endpoint tests for /query with calculations.

Uses FastAPI TestClient to exercise the real /query route with dependency
overrides for auth and patched get_rag_engine. Verifies that calculations
are present in the HTTP response, FAILED does not expose internal errors,
and old fields remain compatible.
"""
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock heavy imports before importing app modules.
for _mod in [
    "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
    "camelot", "pymudf", "pymupdf", "langchain", "langchain_core",
    "langchain_core.documents", "langchain_text_splitters", "jieba_fast",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["jieba_fast"].cut_for_search = lambda text: [text]


def _have_api_deps():
    try:
        import jose  # noqa: F401
        import bcrypt  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_api_deps(), reason="API deps (jose, bcrypt) not available")
class TestCalculationHTTPEndpoint:
    """Real FastAPI /query endpoint tests with calculation support."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from src.main import app
        from src.services.auth import get_current_user
        from fastapi.testclient import TestClient

        # Override auth to bypass JWT.
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.email = "test@test.com"
        app.dependency_overrides[get_current_user] = lambda: mock_user

        self.client = TestClient(app)
        self.app = app
        self.mock_user = mock_user
        yield
        app.dependency_overrides.clear()

    def _mock_engine_result(self, *, status="executed", answer="calc answer",
                            calculations=None, include_calculations=True):
        """Build a mock engine.query() return dict."""
        result = {
            "answer": answer,
            "sources": [{"doc_id": "chunk_001", "page": 12}],
            "searched_docs": ["annual_report.pdf"],
            "confidence": 1.0 if status == "executed" else 0.0,
            "context_sufficient": True,
            "intent": "financial_calculation",
            "intent_confidence": 0.9,
            "trace_id": "trace_http_001",
            "rewritten_question": None,
            "retrieved_chunks": [],
            "retrieval_debug": {},
        }
        if include_calculations and calculations is not None:
            result["calculations"] = calculations
        return result

    def _calculation_dict(self, status="executed", error_code=None):
        """Build a public-dict calculation result."""
        calc = {
            "status": status,
            "operation": "gross_margin",
            "value": "0.4000" if status == "executed" else None,
            "unit": "ratio" if status == "executed" else None,
            "formula": "(revenue - cogs) / revenue" if status == "executed" else None,
            "formula_version": "gross_margin.v1",
            "target_metric": "gross_margin",
            "operands": (
                [{
                    "name": "revenue",
                    "value": "1000000",
                    "unit": "base",
                    "scale": None,
                    "evidence_chunk_id": "chunk_001",
                    "document_name": "annual_report.pdf",
                    "page": 12,
                    "evidence_excerpt": "Total revenue was $1,000,000.",
                }]
                if status == "executed" else []
            ),
            "error_code": error_code,
        }
        return calc

    def _patch_engine(self, result_dict):
        """Patch get_rag_engine and related helpers for one request."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=result_dict)
        return (
            patch("src.main.get_rag_engine", return_value=mock_engine),
            patch("src.main._resolve_query_document_names_for_user", return_value=["annual_report.pdf"]),
            patch("src.main.memory_store"),
        )

    def test_post_query_ordinary_question(self):
        """Point 1: POST /query ordinary question (no calculations)."""
        result_dict = self._mock_engine_result(
            answer="The company makes widgets.",
            include_calculations=False,
        )
        with self._patch_engine(result_dict)[0], \
             self._patch_engine(result_dict)[1], \
             patch("src.main.memory_store") as mock_mem:
            mock_mem.get_profile = MagicMock(return_value=None)
            response = self.client.post("/query", json={
                "question": "What does the company do?",
                "n_results": 3,
            })
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The company makes widgets."
        assert data["calculations"] == []

    def test_post_query_calculation_success(self):
        """Point 2: POST /query calculation success."""
        calc = self._calculation_dict("executed")
        result_dict = self._mock_engine_result(calculations=[calc])
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            response = self.client.post("/query", json={
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        assert response.status_code == 200
        data = response.json()
        assert len(data["calculations"]) == 1
        assert data["calculations"][0]["status"] == "executed"

    def test_post_query_blocked(self):
        """Point 3: POST /query BLOCKED."""
        calc = self._calculation_dict("blocked", error_code="INSUFFICIENT_OPERANDS")
        result_dict = self._mock_engine_result(
            answer="Unable to compute Gross Margin: insufficient operands",
            calculations=[calc],
        )
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            response = self.client.post("/query", json={
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        assert response.status_code == 200
        data = response.json()
        assert data["calculations"][0]["status"] == "blocked"

    def test_post_query_failed(self):
        """Point 4: POST /query FAILED."""
        calc = self._calculation_dict("failed", error_code="PRIMITIVE_EXCEPTION")
        result_dict = self._mock_engine_result(
            answer="Unable to compute Gross Margin due to an internal error.",
            calculations=[calc],
        )
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            response = self.client.post("/query", json={
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        assert response.status_code == 200
        data = response.json()
        assert data["calculations"][0]["status"] == "failed"

    def test_http_200(self):
        """Point 5: HTTP 200 for all calculation responses."""
        for status in ("executed", "blocked", "failed"):
            calc = self._calculation_dict(status, error_code="ERR" if status != "executed" else None)
            result_dict = self._mock_engine_result(calculations=[calc])
            p1, p2, p3 = self._patch_engine(result_dict)
            with p1, p2, p3:
                p3.get_profile = MagicMock(return_value=None)
                response = self.client.post("/query", json={
                    "question": "Calculate the gross margin.",
                    "n_results": 3,
                })
            assert response.status_code == 200

    def test_response_schema_accepts_calculations(self):
        """Point 6: Response schema accepts calculations."""
        calc = self._calculation_dict("executed")
        result_dict = self._mock_engine_result(calculations=[calc])
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            response = self.client.post("/query", json={
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        data = response.json()
        assert "calculations" in data
        assert isinstance(data["calculations"], list)

    def test_success_response_has_formula_version_and_evidence(self):
        """Point 7: Success response includes formula_version and evidence."""
        calc = self._calculation_dict("executed")
        result_dict = self._mock_engine_result(calculations=[calc])
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            response = self.client.post("/query", json={
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        data = response.json()
        c = data["calculations"][0]
        assert c["formula_version"] == "gross_margin.v1"
        assert len(c["operands"]) == 1
        assert c["operands"][0]["evidence_chunk_id"] == "chunk_001"

    def test_failed_no_internal_error_message(self):
        """Point 8: FAILED response does not include internal error text."""
        calc = self._calculation_dict("failed", error_code="PRIMITIVE_EXCEPTION")
        result_dict = self._mock_engine_result(calculations=[calc])
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            response = self.client.post("/query", json={
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        data = response.json()
        c = data["calculations"][0]
        assert "error_message" not in c
        assert c["error_code"] == "PRIMITIVE_EXCEPTION"

    def test_ordinary_response_keeps_original_fields(self):
        """Point 9: Ordinary response keeps original fields."""
        result_dict = self._mock_engine_result(
            answer="Some answer.",
            include_calculations=False,
        )
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            response = self.client.post("/query", json={
                "question": "What is the company name?",
                "n_results": 3,
            })
        data = response.json()
        for field in ("answer", "sources", "question", "searched_docs",
                       "confidence", "context_sufficient", "intent",
                       "intent_confidence", "trace_id"):
            assert field in data

    def test_old_client_ignores_calculations(self):
        """Point 10: Old client ignoring calculations is not affected."""
        calc = self._calculation_dict("executed")
        result_dict = self._mock_engine_result(calculations=[calc])
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            response = self.client.post("/query", json={
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        data = response.json()
        # Old fields still present and correct.
        assert data["answer"]
        assert isinstance(data["sources"], list)
        # calculations is additive and does not break old field access.
        assert isinstance(data["calculations"], list)
        # Response is JSON-serializable (TestClient already parsed it).
        json.dumps(data)


# Declared, not inherited.  See `legacy_single_turn_endpoint` in conftest.py for
# why a test that patches the RAG engine must say which runtime it exercises
# rather than depend on whichever mode is currently the default.


@pytest.fixture(autouse=True)
def _declared_runtime(legacy_single_turn_endpoint: None) -> None:
    """This module patches get_rag_engine to assert the legacy /query response
    shape, which only the V1 lifecycle produces."""
