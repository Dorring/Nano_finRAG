"""Phase 4 HTTP and SSE endpoint tests for validation exposure.

Exercises the real /query and /query/stream endpoints with FastAPI
TestClient, verifying that:

1. When the engine result carries ``answerability`` / ``validation`` /
   ``repair`` dicts, they appear in both the HTTP JSON response and the
   SSE ``done`` event.
2. When the engine result omits them (validation pipeline disabled),
   the keys are absent from both surfaces — preserving the legacy shape.
3. Internal fields (``error_message``, ``message``, ``evidence_ids``,
   ``repair_notes``) never leak into public responses.
4. The HTTP response uses ``response_model_exclude_none`` so None
   optional fields do not appear as ``null`` in the JSON body.
"""
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock heavy imports before importing app modules.
for _mod in [
    "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
    "camelot", "pymupdf", "langchain", "langchain_core",
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


def _parse_sse_lines(text: str) -> list[dict]:
    """Parse SSE response text into a list of event dicts."""
    events = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


def _answerability_dict(status="answerable"):
    return {
        "status": status,
        "reason_codes": ["sufficient_evidence"],
        "evidence_count": 3,
        "document_count": 2,
        "missing_requirements": [],
    }


def _validation_dict(status="passed"):
    return {
        "status": status,
        "checked_claim_count": 5,
        "supported_claim_count": 5,
        "unsupported_claim_count": 0,
        "issues": [
            {
                "code": "CITATION_WARNING",
                "severity": "warning",
                "public_message": "A citation could not be resolved.",
            },
        ],
    }


def _repair_dict(was_repaired=True, fallback_used=False):
    return {
        "was_repaired": was_repaired,
        "fallback_used": fallback_used,
    }


def _base_result(*, answer="LLM answer", with_validation=False):
    result = {
        "answer": answer,
        "sources": [{"doc_id": "chunk_001", "page": 12}],
        "searched_docs": ["annual_report.pdf"],
        "confidence": 0.9,
        "context_sufficient": True,
        "intent": "document_qa",
        "intent_confidence": 0.8,
        "trace_id": "trace_val_001",
        "rewritten_question": None,
        "retrieved_chunks": [],
        "retrieval_debug": {},
        "calculations": [],
    }
    if with_validation:
        result["answerability"] = _answerability_dict()
        result["validation"] = _validation_dict()
        result["repair"] = _repair_dict()
    return result


@pytest.mark.skipif(not _have_api_deps(), reason="API deps (jose, bcrypt) not available")
class TestValidationHTTPEndpoint:
    """Real FastAPI /query endpoint tests with validation fields."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from src.main import app
        from src.services.auth import get_current_user
        from fastapi.testclient import TestClient

        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.email = "test@test.com"
        app.dependency_overrides[get_current_user] = lambda: mock_user

        self.client = TestClient(app)
        self.app = app
        self.mock_user = mock_user
        yield
        app.dependency_overrides.clear()

    def _patch_engine(self, result_dict):
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=result_dict)
        return (
            patch("src.main.get_rag_engine", return_value=mock_engine),
            patch("src.main._resolve_query_document_names_for_user",
                  return_value=["annual_report.pdf"]),
            patch("src.main.memory_store"),
        )

    # ------------------------------------------------------------------
    # Validation fields present
    # ------------------------------------------------------------------

    def test_http_response_includes_answerability(self):
        result_dict = _base_result(with_validation=True)
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            resp = self.client.post("/query", json={
                "question": "What was revenue?",
                "n_results": 3,
            })
        assert resp.status_code == 200
        body = resp.json()
        assert "answerability" in body
        assert body["answerability"]["status"] == "answerable"
        assert body["answerability"]["evidence_count"] == 3
        assert body["answerability"]["reason_codes"] == ["sufficient_evidence"]

    def test_http_response_includes_validation(self):
        result_dict = _base_result(with_validation=True)
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            resp = self.client.post("/query", json={
                "question": "What was revenue?",
                "n_results": 3,
            })
        body = resp.json()
        assert "validation" in body
        assert body["validation"]["status"] == "passed"
        assert body["validation"]["checked_claim_count"] == 5
        assert len(body["validation"]["issues"]) == 1
        issue = body["validation"]["issues"][0]
        assert issue["code"] == "CITATION_WARNING"
        assert issue["severity"] == "warning"
        assert issue["public_message"] is not None

    def test_http_response_includes_repair(self):
        result_dict = _base_result(with_validation=True)
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            resp = self.client.post("/query", json={
                "question": "What was revenue?",
                "n_results": 3,
            })
        body = resp.json()
        assert "repair" in body
        assert body["repair"]["was_repaired"] is True
        assert body["repair"]["fallback_used"] is False

    def test_http_validation_excludes_internal_fields(self):
        """Public validation payload must not expose internal message/evidence_ids."""
        result_dict = _base_result(with_validation=True)
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            resp = self.client.post("/query", json={
                "question": "What was revenue?",
                "n_results": 3,
            })
        body = resp.json()
        # Internal issue fields must NOT appear.
        for issue in body["validation"]["issues"]:
            assert "message" not in issue
            assert "evidence_ids" not in issue
        # Answerability must not expose scores.
        assert "best_score" not in body["answerability"]
        assert "average_score" not in body["answerability"]
        # Repair must not expose repair_notes.
        assert "repair_notes" not in body["repair"]

    # ------------------------------------------------------------------
    # Validation fields absent (pipeline disabled)
    # ------------------------------------------------------------------

    def test_http_response_omits_validation_when_disabled(self):
        result_dict = _base_result(with_validation=False)
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            resp = self.client.post("/query", json={
                "question": "What was revenue?",
                "n_results": 3,
            })
        body = resp.json()
        assert "answerability" not in body
        assert "validation" not in body
        assert "repair" not in body
        # Legacy fields remain.
        assert body["answer"] == "LLM answer"
        assert body["trace_id"] == "trace_val_001"

    def test_http_response_excludes_none_optional_fields(self):
        """response_model_exclude_none must omit None optional fields."""
        result_dict = _base_result(with_validation=False)
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            resp = self.client.post("/query", json={
                "question": "What was revenue?",
                "n_results": 3,
            })
        body = resp.json()
        # rewritten_question is None in the mock → must be absent.
        assert "rewritten_question" not in body
        # session_id is None (no session) → must be absent.
        assert "session_id" not in body

    # ------------------------------------------------------------------
    # Old fields still present
    # ------------------------------------------------------------------

    def test_http_old_fields_remain_with_validation(self):
        result_dict = _base_result(with_validation=True)
        p1, p2, p3 = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            resp = self.client.post("/query", json={
                "question": "What was revenue?",
                "n_results": 3,
            })
        body = resp.json()
        for field in ("answer", "sources", "question", "searched_docs",
                       "confidence", "context_sufficient", "intent",
                       "intent_confidence", "trace_id", "calculations"):
            assert field in body, f"missing legacy field: {field}"


@pytest.mark.skipif(not _have_api_deps(), reason="API deps (jose, bcrypt) not available")
class TestValidationSSEEndpoint:
    """Real FastAPI /query/stream endpoint tests with validation fields."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from src.main import app
        from src.services.auth import get_current_user
        from fastapi.testclient import TestClient

        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.email = "test@test.com"
        app.dependency_overrides[get_current_user] = lambda: mock_user

        self.client = TestClient(app)
        self.app = app
        yield
        app.dependency_overrides.clear()

    def _patch_engine(self, result_dict):
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=result_dict)
        mock_engine.generate_answer_stream = AsyncMock()
        mock_engine.trace_logger = MagicMock()
        return (
            patch("src.main.get_rag_engine", return_value=mock_engine),
            patch("src.main._resolve_query_document_names_for_user",
                  return_value=["annual_report.pdf"]),
            patch("src.main.memory_store"),
            mock_engine,
        )

    def _stream_query(self, json_body):
        response = self.client.post("/query/stream", json=json_body)
        assert response.status_code == 200
        return _parse_sse_lines(response.text)

    # ------------------------------------------------------------------
    # Validation fields present in SSE done event
    # ------------------------------------------------------------------

    def test_sse_done_includes_answerability(self):
        result_dict = _base_result(
            answer="validated answer",
            with_validation=True,
        )
        p1, p2, p3, _ = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            events = self._stream_query({
                "question": "What was revenue?",
                "n_results": 3,
            })
        done = [e for e in events if e.get("type") == "done"]
        assert done
        assert "answerability" in done[0]
        assert done[0]["answerability"]["status"] == "answerable"

    def test_sse_done_includes_validation(self):
        result_dict = _base_result(with_validation=True)
        p1, p2, p3, _ = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            events = self._stream_query({
                "question": "What was revenue?",
                "n_results": 3,
            })
        done = [e for e in events if e.get("type") == "done"]
        assert done
        assert "validation" in done[0]
        assert done[0]["validation"]["status"] == "passed"
        assert len(done[0]["validation"]["issues"]) == 1

    def test_sse_done_includes_repair(self):
        result_dict = _base_result(with_validation=True)
        p1, p2, p3, _ = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            events = self._stream_query({
                "question": "What was revenue?",
                "n_results": 3,
            })
        done = [e for e in events if e.get("type") == "done"]
        assert done
        assert "repair" in done[0]
        assert done[0]["repair"]["was_repaired"] is True

    def test_sse_done_validation_excludes_internal_fields(self):
        result_dict = _base_result(with_validation=True)
        p1, p2, p3, _ = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            events = self._stream_query({
                "question": "What was revenue?",
                "n_results": 3,
            })
        done = [e for e in events if e.get("type") == "done"][0]
        for issue in done["validation"]["issues"]:
            assert "message" not in issue
            assert "evidence_ids" not in issue
        assert "best_score" not in done["answerability"]
        assert "repair_notes" not in done["repair"]

    # ------------------------------------------------------------------
    # Validation fields absent from SSE when pipeline disabled
    # ------------------------------------------------------------------

    def test_sse_done_omits_validation_when_disabled(self):
        result_dict = _base_result(with_validation=False)
        p1, p2, p3, _ = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            events = self._stream_query({
                "question": "What was revenue?",
                "n_results": 3,
            })
        done = [e for e in events if e.get("type") == "done"]
        assert done
        assert "answerability" not in done[0]
        assert "validation" not in done[0]
        assert "repair" not in done[0]
        # Legacy fields remain.
        assert done[0]["trace_id"] == "trace_val_001"

    # ------------------------------------------------------------------
    # Token event carries the (possibly repaired) answer
    # ------------------------------------------------------------------

    def test_sse_token_carries_answer(self):
        result_dict = _base_result(
            answer="Repaired answer text.",
            with_validation=True,
        )
        p1, p2, p3, _ = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            events = self._stream_query({
                "question": "What was revenue?",
                "n_results": 3,
            })
        token = [e for e in events if e.get("type") == "token"]
        assert token
        assert token[0]["content"] == "Repaired answer text."

    # ------------------------------------------------------------------
    # Strict validation path: blocked answer does not stream partial tokens
    # ------------------------------------------------------------------

    def test_sse_blocked_answer_emits_safe_message_as_token(self):
        """When validation BLOCKS, the safe fallback is emitted as the
        single token event — no partial LLM tokens leak."""
        result_dict = _base_result(
            answer="I cannot answer this question based on the available evidence.",
            with_validation=True,
        )
        # Override validation to blocked status.
        result_dict["validation"] = _validation_dict(status="blocked")
        result_dict["repair"] = _repair_dict(was_repaired=False, fallback_used=True)
        p1, p2, p3, mock_engine = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            events = self._stream_query({
                "question": "What was revenue?",
                "n_results": 3,
            })
        token = [e for e in events if e.get("type") == "token"]
        done = [e for e in events if e.get("type") == "done"]
        assert token
        assert done
        # The token must be the safe fallback, not a partial LLM stream.
        assert "cannot answer" in token[0]["content"].lower()
        # LLM stream must NOT be invoked.
        mock_engine.generate_answer_stream.assert_not_called()
        # Done event carries the blocked validation status.
        assert done[0]["validation"]["status"] == "blocked"
        assert done[0]["repair"]["fallback_used"] is True


# Declared, not inherited.  See `legacy_single_turn_endpoint` in conftest.py for
# why a test that patches the RAG engine must say which runtime it exercises
# rather than depend on whichever mode is currently the default.


@pytest.fixture(autouse=True)
def _declared_runtime(legacy_single_turn_endpoint: None) -> None:
    """This module drives the real /query HTTP and SSE routes with a patched
    RAG engine. Only the V1 lifecycle calls get_rag_engine, so the runtime
    under test is V1 single-turn."""
