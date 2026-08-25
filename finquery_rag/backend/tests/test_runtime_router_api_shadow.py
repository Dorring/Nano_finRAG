"""TV2-06 endpoint smoke with an offline test-only embedding constructor."""
from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from src.runtime import ReleaseStatus, V2ExecutionOutcome, V2ExecutionStatus


class _FakeRAGEngine:
    async def query(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "answer": "Revenue was $100B.",
            "sources": [],
            "searched_docs": ["annual_report.pdf"],
            "rewritten_question": None,
            "confidence": 0.9,
            "context_sufficient": True,
            "intent": "document_qa",
            "intent_confidence": 0.9,
            "trace_id": "tv2-06-api",
            "retrieved_chunks": [],
            "retrieval_debug": {},
            "calculations": [],
        }


class _Coordinator:
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[Any] = []

    async def execute(self, request: Any) -> V2ExecutionOutcome:
        self.calls += 1
        self.requests.append(request)
        return V2ExecutionOutcome(
            status=V2ExecutionStatus.READY_FOR_RELEASE,
            answer="V2 shadow answer",
            release_status=ReleaseStatus.RELEASED,
            route="STRUCTURED_SINGLE",
            evidence_ids=["E-shadow"],
        )


def _load_main(monkeypatch: Any, tmp_path: Any) -> Any:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("SESSIONS_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setenv("FINANCIAL_RUNTIME_ADAPTER_ENABLED", "true")
    from chromadb.utils import embedding_functions

    class OfflineEmbedding:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, inputs: list[str]) -> list[list[float]]:
            return [[0.0] * 384 for _ in inputs]

    monkeypatch.setattr(
        embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        OfflineEmbedding,
    )
    if "src.services.vector_store" in sys.modules:
        raise RuntimeError("run this endpoint smoke in a fresh Python process")
    return importlib.import_module("src.main")


def test_query_and_stream_share_v1_primary_shadow_transport(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    main = _load_main(monkeypatch, tmp_path)
    user = SimpleNamespace(id=42, email="tv2-06@example.test")
    main.app.dependency_overrides[main.get_current_user] = lambda: user
    client = TestClient(main.app)
    engine = _FakeRAGEngine()
    monkeypatch.setattr(main, "get_rag_engine", lambda: engine)
    monkeypatch.setattr(
        main,
        "_resolve_query_document_names_for_user",
        lambda *args, **kwargs: ["annual_report.pdf"],
    )
    try:
        monkeypatch.setenv("FINANCIAL_RUNTIME_MODE", "v1")
        main.configure_trusted_v2_shadow_runtime_builder(None)
        v1_json = client.post(
            "/query",
            json={"question": "What was revenue?", "document_names": ["annual_report.pdf"]},
        )
        v1_stream = client.post(
            "/query/stream",
            json={"question": "What was revenue?", "document_names": ["annual_report.pdf"]},
        )
        assert v1_json.status_code == 200, v1_json.text
        assert v1_stream.status_code == 200, v1_stream.text

        coordinator = _Coordinator()
        from src.runtime import TrustedFinancialRuntimeV2

        main.configure_trusted_v2_shadow_runtime_builder(
            lambda engine, request: TrustedFinancialRuntimeV2(coordinator),
        )
        monkeypatch.setenv("FINANCIAL_RUNTIME_MODE", "shadow")
        shadow_json = client.post(
            "/query",
            json={"question": "What was revenue?", "document_names": ["annual_report.pdf"]},
        )
        shadow_stream = client.post(
            "/query/stream",
            json={"question": "What was revenue?", "document_names": ["annual_report.pdf"]},
        )
        assert shadow_json.status_code == 200
        assert shadow_stream.status_code == 200
        assert shadow_json.json() == v1_json.json()
        assert shadow_stream.text == v1_stream.text
        assert coordinator.calls == 2
        # The V2 adapter strips raw context before the coordinator boundary.
        # The original FinancialQueryRequest itself remains shared upstream.
        assert all(
            request.standalone_query == "What was revenue?"
            and "conversation_history" not in request.request_metadata
            and "memory_profile" not in request.request_metadata
            for request in coordinator.requests
        )
        monkeypatch.setenv("FINANCIAL_RUNTIME_MODE", "v2")
        official_json = client.post(
            "/query",
            json={"question": "What was revenue?", "document_names": ["annual_report.pdf"]},
        )
        official_stream = client.post(
            "/query/stream",
            json={"question": "What was revenue?", "document_names": ["annual_report.pdf"]},
        )
        assert official_json.status_code == 200
        assert official_stream.status_code == 200
        assert official_json.json()["answer"] == "V2 shadow answer"
        assert "V2 shadow answer" in official_stream.text
        assert coordinator.calls == 4
    finally:
        main.configure_trusted_v2_shadow_runtime_builder(None)
        main.app.dependency_overrides.clear()
