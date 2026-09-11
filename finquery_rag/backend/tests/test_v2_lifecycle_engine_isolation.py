"""Verify that official V2 execution does not construct the legacy RAG engine."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from src.runtime.query_lifecycle import QueryLifecycleService, UserTurnExecutionRequest
from src.runtime.runtime_contract import (
    FinancialQueryResult,
    ReleaseStatus,
    RuntimeStatus,
    RuntimeVersion,
)


class _V2Runtime:
    async def execute(self, request: Any) -> FinancialQueryResult:
        assert request.standalone_query == "What was Apple FY2024 revenue?"
        return FinancialQueryResult(
            status=RuntimeStatus.ANSWER,
            answer="391 USD billion",
            runtime_version=RuntimeVersion.V2,
            release_status=ReleaseStatus.RELEASED,
            evidence_ids=["E1"],
        )


def _service(
    *,
    get_rag_engine: Any,
    runtime_factory: Any,
    requires_engine: Any,
    adapter_enabled: bool = True,
) -> QueryLifecycleService:
    return QueryLifecycleService(
        session_manager=SimpleNamespace(),
        memory_store=SimpleNamespace(get_profile=lambda _user_id: None),
        get_rag_engine=get_rag_engine,
        get_conversation_service=lambda: None,
        financial_runtime_adapter_enabled=lambda: adapter_enabled,
        active_query_requires_context=lambda _query: False,
        active_query_is_out_of_scope=lambda _query: False,
        assistant_session_metadata=lambda **_kwargs: {},
        financial_runtime_factory=runtime_factory,
        financial_runtime_requires_engine=requires_engine,
    )


def _request() -> UserTurnExecutionRequest:
    return UserTurnExecutionRequest(
        request_id="v2-engine-isolation",
        user_id=7,
        original_query="What was Apple FY2024 revenue?",
        conversation_mode="off",
    )


def test_official_v2_does_not_construct_legacy_rag_engine() -> None:
    captured: dict[str, Any] = {}

    def fail_if_constructed() -> Any:
        raise AssertionError("official V2 must not construct the legacy RAG engine")

    def build_runtime(engine: Any, _request: Any) -> _V2Runtime:
        captured["engine"] = engine
        return _V2Runtime()

    result = asyncio.run(
        _service(
            get_rag_engine=fail_if_constructed,
            runtime_factory=build_runtime,
            requires_engine=lambda: False,
        ).execute_user_turn(_request())
    )

    assert captured["engine"] is None
    assert result.runtime_version == "V2"
    assert result.status == "ANSWER"
    assert result.legacy_result["trace_id"] == "v2-engine-isolation"


def test_legacy_engine_is_still_required_for_direct_engine_execution() -> None:
    service = _service(
        get_rag_engine=lambda: None,
        runtime_factory=None,
        requires_engine=lambda: False,
        adapter_enabled=False,
    )

    try:
        asyncio.run(service.execute_user_turn(_request()))
    except RuntimeError as exc:
        assert str(exc) == "legacy query execution requires the RAG engine"
    else:
        raise AssertionError("direct legacy execution unexpectedly proceeded without an engine")


def test_official_v2_document_scope_does_not_load_legacy_vector_store(monkeypatch: Any) -> None:
    import src.main as main

    monkeypatch.setenv("FINANCIAL_RUNTIME_MODE", "v2")
    monkeypatch.setattr(main.document_registry, "list_documents", lambda _user_id: [])

    assert main._resolve_query_document_names_for_user(7, None) == []
    assert main._resolve_query_document_names_for_user(
        7,
        ["annual_report.pdf", "annual_report.pdf"],
    ) == ["annual_report.pdf"]
