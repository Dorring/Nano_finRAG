"""I3 API-level parity tests for routing /query through the V1 adapter."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from src.runtime import (
    FinancialQueryRequest,
    LegacyFinancialRuntimeAdapter,
    QueryExecutionService,
    to_legacy_query_dict,
)


def test_contract_service_maps_legacy_payload_without_api_dependencies():
    engine = FakeRAGEngine()
    request = FinancialQueryRequest(
        request_id="i3-unit",
        user_id="42",
        session_id="__stateless__:i3-unit",
        original_query="What was revenue?",
        standalone_query="What was revenue?",
        request_metadata={
            "document_names": ["annual_report.pdf"],
            "n_results": 5,
            "conversation_history": [],
            "memory_profile": {},
        },
    )
    result = asyncio.run(
        QueryExecutionService(LegacyFinancialRuntimeAdapter(engine)).execute(request),
    )

    assert to_legacy_query_dict(result) == _legacy_result()
    assert engine.calls[0]["user_id"] == 42


def _have_api_deps() -> bool:
    try:
        import bcrypt  # noqa: F401
        import jose  # noqa: F401
    except ImportError:
        return False
    return True


class FakeRAGEngine:
    def __init__(
        self, result: dict[str, Any] | None = None, error: Exception | None = None
    ):
        self.result = result or _legacy_result()
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def query(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def _legacy_result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "answer": "Revenue was $100B.",
        "sources": [
            {
                "filename": "annual_report.pdf",
                "page": 4,
                "chunk_id": "chunk-revenue",
            },
        ],
        "searched_docs": ["annual_report.pdf"],
        "rewritten_question": None,
        "confidence": 0.91,
        "context_sufficient": True,
        "intent": "document_qa",
        "intent_confidence": 0.88,
        "trace_id": "trace-i3",
        "retrieved_chunks": [{"chunk_id": "chunk-revenue"}],
        "retrieval_debug": {"candidate_count": 1},
        "calculations": [],
    }
    result.update(overrides)
    return result


@pytest.mark.skipif(
    not _have_api_deps(), reason="API deps (jose, bcrypt) not available"
)
class TestQueryRuntimeIntegration:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        from fastapi.testclient import TestClient

        from src.main import app
        from src.services.auth import get_current_user

        self.app = app
        self.auth_dependency = get_current_user
        self.user = SimpleNamespace(id=42, email="i3@example.test")
        app.dependency_overrides[get_current_user] = lambda: self.user
        self.client = TestClient(app)
        yield
        app.dependency_overrides.clear()

    def _post(
        self,
        *,
        enabled: bool,
        engine: FakeRAGEngine,
        session_id: str | None = None,
        session_manager: Any | None = None,
    ):
        memory_store = SimpleNamespace(get_profile=lambda user_id: {})
        patches = [
            patch("src.main.get_rag_engine", return_value=engine),
            patch(
                "src.main._resolve_query_document_names_for_user",
                return_value=["annual_report.pdf"],
            ),
            patch("src.main.memory_store", memory_store),
            patch.dict(
                os.environ,
                {"FINANCIAL_RUNTIME_ADAPTER_ENABLED": "true" if enabled else "false"},
                clear=False,
            ),
        ]
        if session_manager is not None:
            patches.append(patch("src.main.session_manager", session_manager))
        if session_manager is None:
            with patches[0], patches[1], patches[2], patches[3]:
                return self.client.post(
                    "/query",
                    json={
                        "question": "What was revenue?",
                        "document_names": ["annual_report.pdf"],
                        "n_results": 5,
                        **({"session_id": session_id} if session_id else {}),
                    },
                )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            return self.client.post(
                "/query",
                json={
                    "question": "What was revenue?",
                    "document_names": ["annual_report.pdf"],
                    "n_results": 5,
                    **({"session_id": session_id} if session_id else {}),
                },
            )

    def test_adapter_and_direct_paths_have_identical_public_payload(self):
        direct_engine = FakeRAGEngine()
        adapter_engine = FakeRAGEngine()
        direct = self._post(enabled=False, engine=direct_engine)
        adapted = self._post(enabled=True, engine=adapter_engine)

        assert direct.status_code == 200
        assert adapted.status_code == 200
        assert adapted.json() == direct.json()
        assert adapter_engine.calls == direct_engine.calls

    def test_calculation_and_blocked_payloads_preserve_legacy_shape(self):
        calculation = _legacy_result(
            answer="Growth was 10%.",
            calculations=[
                {
                    "status": "executed",
                    "operation": "growth_rate",
                    "value": "0.10",
                    "unit": "ratio",
                    "formula": "(current-prior)/prior",
                    "formula_version": "growth_rate.v1",
                    "target_metric": "growth",
                    "operands": [],
                    "error_code": None,
                },
            ],
        )
        direct = self._post(enabled=False, engine=FakeRAGEngine(calculation))
        adapted = self._post(enabled=True, engine=FakeRAGEngine(calculation))
        assert direct.status_code == adapted.status_code == 200
        assert adapted.json() == direct.json()
        assert adapted.json()["calculations"][0]["value"] == "0.10"

        blocked = _legacy_result(
            answer="I cannot answer this safely.",
            answerability={
                "status": "not_answerable",
                "reason_codes": ["MISSING_METRIC"],
                "evidence_count": 0,
                "document_count": 0,
                "missing_requirements": ["revenue"],
            },
            validation={
                "status": "blocked",
                "checked_claim_count": 0,
                "supported_claim_count": 0,
                "unsupported_claim_count": 0,
                "issues": [],
            },
        )
        direct = self._post(enabled=False, engine=FakeRAGEngine(blocked))
        adapted = self._post(enabled=True, engine=FakeRAGEngine(blocked))
        assert direct.status_code == adapted.status_code == 200
        assert adapted.json() == direct.json()
        assert adapted.json()["answerability"]["status"] == "not_answerable"

    def test_session_messages_are_written_once_on_adapter_path(self):
        session = SimpleNamespace(
            get_recent_messages=Mock(
                return_value=[{"role": "user", "content": "Revenue?"}]
            ),
            add_message=Mock(),
        )
        engine = FakeRAGEngine()
        response = self._post(
            enabled=True,
            engine=engine,
            session_id="session-i3",
            session_manager=session,
        )

        assert response.status_code == 200
        assert session.add_message.call_count == 2
        assert [call.args[2] for call in session.add_message.call_args_list] == [
            "user",
            "assistant",
        ]
        assert engine.calls == [
            {
                "question": "What was revenue?",
                "doc_names": ["annual_report.pdf"],
                "user_id": 42,
                "n_results": 5,
                "conversation_history": [{"role": "user", "content": "Revenue?"}],
                "memory_profile": {},
            },
        ]

    def test_endpoint_request_is_unresolved_and_preserves_identity_scope(self):
        captured: dict[str, Any] = {}
        from src.main import QueryExecutionService as RealQueryExecutionService

        class SpyExecutionService:
            def __init__(self, runtime):
                self.runtime = runtime

            async def execute(self, request):
                captured["request"] = request
                return await RealQueryExecutionService(self.runtime).execute(request)

        with patch("src.main.QueryExecutionService", SpyExecutionService):
            response = self._post(enabled=True, engine=FakeRAGEngine())

        assert response.status_code == 200
        request = captured["request"]
        assert request.user_id == "42"
        assert request.session_id.startswith("__stateless__:")
        assert request.original_query == "What was revenue?"
        assert request.standalone_query == request.original_query
        assert request.query_as_resolved is False
        assert request.request_metadata["n_results"] == 5

    def test_engine_error_keeps_http_error_semantics(self):
        direct = self._post(
            enabled=False,
            engine=FakeRAGEngine(error=RuntimeError("direct failure")),
        )
        adapted = self._post(
            enabled=True,
            engine=FakeRAGEngine(error=RuntimeError("adapter failure")),
        )

        assert direct.status_code == adapted.status_code == 500
        assert direct.json()["detail"]["error_code"] == "query_error"
        assert adapted.json()["detail"]["error_code"] == "query_error"
