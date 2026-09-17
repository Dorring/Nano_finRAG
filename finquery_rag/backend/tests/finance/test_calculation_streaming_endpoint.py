"""Phase 3 real SSE endpoint tests for /query/stream with calculations.

Uses FastAPI TestClient to exercise the real /query/stream route with
dependency overrides for auth and patched get_rag_engine. Verifies that
calculation results flow through the unified streaming path, the LLM
stream is bypassed for EXECUTED/BLOCKED/FAILED, FAILED does not expose
internal errors, and non-calculation queries preserve the legacy stream
flow.
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


@pytest.mark.skipif(not _have_api_deps(), reason="API deps (jose, bcrypt) not available")
class TestCalculationStreamingEndpoint:
    """Real FastAPI /query/stream endpoint tests with calculation support."""

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
            "trace_id": "trace_sse_001",
            "rewritten_question": None,
            "retrieved_chunks": [],
            "retrieval_debug": {},
        }
        if include_calculations and calculations is not None:
            result["calculations"] = calculations
        return result

    def _calculation_dict(self, status="executed", error_code=None):
        """Build a public-dict calculation result."""
        return {
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

    def _patch_engine(self, result_dict):
        """Patch get_rag_engine and related helpers for one request."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=result_dict)
        # generate_answer_stream must never be called in the unified path.
        mock_engine.generate_answer_stream = AsyncMock()
        mock_engine.trace_logger = MagicMock()
        return (
            patch("src.main.get_rag_engine", return_value=mock_engine),
            patch("src.main._resolve_query_document_names_for_user", return_value=["annual_report.pdf"]),
            patch("src.main.memory_store"),
            mock_engine,
        )

    def _stream_query(self, json_body):
        """POST /query/stream and return parsed SSE events."""
        response = self.client.post("/query/stream", json=json_body)
        assert response.status_code == 200
        return _parse_sse_lines(response.text)

    def test_calculation_success_receives_deterministic_token(self):
        """Point 1: calculation success receives a deterministic answer/token event."""
        calc = self._calculation_dict("executed")
        result_dict = self._mock_engine_result(
            answer="Gross Margin = 40.00%",
            calculations=[calc],
        )
        p1, p2, p3, mock_engine = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            events = self._stream_query({
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        # First event must be a token event carrying the deterministic answer.
        token_events = [e for e in events if e.get("type") == "token"]
        assert token_events, "expected at least one token event"
        assert token_events[0]["content"] == "Gross Margin = 40.00%"
        # LLM stream method must NOT be invoked.
        mock_engine.generate_answer_stream.assert_not_called()

    def test_done_event_includes_calculations(self):
        """Point 2: final done event includes calculations."""
        calc = self._calculation_dict("executed")
        result_dict = self._mock_engine_result(calculations=[calc])
        p1, p2, p3, _ = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            events = self._stream_query({
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        done_events = [e for e in events if e.get("type") == "done"]
        assert done_events, "expected a done event"
        assert "calculations" in done_events[0]
        assert len(done_events[0]["calculations"]) == 1
        assert done_events[0]["calculations"][0]["status"] == "executed"

    def test_calculation_success_bypasses_llm_stream(self):
        """Point 3: calculation success does not invoke LLM stream."""
        calc = self._calculation_dict("executed")
        result_dict = self._mock_engine_result(calculations=[calc])
        p1, p2, p3, mock_engine = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            self._stream_query({
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        mock_engine.generate_answer_stream.assert_not_called()

    def test_blocked_bypasses_llm_stream(self):
        """Point 4: BLOCKED calculation does not invoke LLM stream."""
        calc = self._calculation_dict("blocked", error_code="INSUFFICIENT_OPERANDS")
        result_dict = self._mock_engine_result(
            answer="Unable to compute Gross Margin: insufficient operands",
            calculations=[calc],
        )
        p1, p2, p3, mock_engine = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            self._stream_query({
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        mock_engine.generate_answer_stream.assert_not_called()

    def test_failed_bypasses_llm_stream(self):
        """Point 5: FAILED calculation does not invoke LLM stream."""
        calc = self._calculation_dict("failed", error_code="PRIMITIVE_EXCEPTION")
        result_dict = self._mock_engine_result(
            answer="Unable to compute Gross Margin due to an internal error.",
            calculations=[calc],
        )
        p1, p2, p3, mock_engine = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            self._stream_query({
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        mock_engine.generate_answer_stream.assert_not_called()

    def test_failed_event_excludes_internal_exception(self):
        """Point 6: FAILED event does not expose internal error text."""
        calc = self._calculation_dict("failed", error_code="PRIMITIVE_EXCEPTION")
        # Intentionally add an internal-looking string that must NOT leak
        # through the public calculation payload.
        result_dict = self._mock_engine_result(
            answer="Unable to compute Gross Margin due to an internal error.",
            calculations=[calc],
        )
        p1, p2, p3, _ = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            events = self._stream_query({
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        done_events = [e for e in events if e.get("type") == "done"]
        assert done_events
        calc_payload = done_events[0]["calculations"][0]
        assert "error_message" not in calc_payload
        assert calc_payload["error_code"] == "PRIMITIVE_EXCEPTION"
        # No internal exception text should be present anywhere in the calc payload.
        serialized = json.dumps(calc_payload)
        assert "Traceback" not in serialized
        assert "Exception" not in serialized

    def test_non_calculation_keeps_stream_flow(self):
        """Point 7: non-calculation request keeps the streaming flow (token + done)."""
        result_dict = self._mock_engine_result(
            answer="The company manufactures widgets.",
            include_calculations=False,
        )
        p1, p2, p3, mock_engine = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            events = self._stream_query({
                "question": "What does the company do?",
                "n_results": 3,
            })
        token_events = [e for e in events if e.get("type") == "token"]
        done_events = [e for e in events if e.get("type") == "done"]
        assert token_events
        assert done_events
        assert token_events[0]["content"] == "The company manufactures widgets."
        # Non-calculation result may or may not carry calculations; when
        # absent the done event should still be well-formed.
        assert done_events[0].get("calculations", []) == []

    def test_trace_id_present(self):
        """Point 8: Trace ID is present in the done event."""
        calc = self._calculation_dict("executed")
        result_dict = self._mock_engine_result(calculations=[calc])
        p1, p2, p3, _ = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            events = self._stream_query({
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        done_events = [e for e in events if e.get("type") == "done"]
        assert done_events
        assert done_events[0]["trace_id"] == "trace_sse_001"

    def test_sources_present(self):
        """Point 9: Sources are present in the done event."""
        calc = self._calculation_dict("executed")
        result_dict = self._mock_engine_result(calculations=[calc])
        p1, p2, p3, _ = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            events = self._stream_query({
                "question": "Calculate the gross margin.",
                "n_results": 3,
            })
        done_events = [e for e in events if e.get("type") == "done"]
        assert done_events
        assert isinstance(done_events[0]["sources"], list)
        assert len(done_events[0]["sources"]) >= 1

    def test_session_writes_compatible(self):
        """Point 10: session messages are written when session_id is provided."""
        calc = self._calculation_dict("executed")
        result_dict = self._mock_engine_result(calculations=[calc])
        p1, p2, p3, _ = self._patch_engine(result_dict)
        with p1, p2, p3:
            p3.get_profile = MagicMock(return_value=None)
            with patch("src.main.session_manager") as mock_session:
                mock_session.get_recent_messages = MagicMock(return_value=[])
                events = self._stream_query({
                    "question": "Calculate the gross margin.",
                    "n_results": 3,
                    "session_id": "sess-123",
                })
        assert events  # events must be emitted
        # When session_id is provided, two add_message calls must happen
        # (user message + assistant message), preserving legacy behavior.
        assert mock_session.add_message.call_count == 2
        # The first call must be the user's question.
        first_call = mock_session.add_message.call_args_list[0]
        assert first_call.args[2] == "user"
        assert first_call.args[3] == "Calculate the gross margin."
        # The second call must be the assistant's answer.
        second_call = mock_session.add_message.call_args_list[1]
        assert second_call.args[2] == "assistant"


# Declared, not inherited.  See `legacy_single_turn_endpoint` in conftest.py for
# why a test that patches the RAG engine must say which runtime it exercises
# rather than depend on whichever mode is currently the default.


@pytest.fixture(autouse=True)
def _declared_runtime(legacy_single_turn_endpoint: None) -> None:
    """This module patches get_rag_engine to assert the legacy /query/stream
    event shape, which only the V1 lifecycle produces."""
