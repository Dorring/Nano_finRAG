"""Unit tests for the I2 legacy V1 runtime adapter."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.runtime import (
    FinancialQARuntime,
    FinancialQueryRequest,
    FinancialQueryResult,
    ReleaseStatus,
    RuntimeStatus,
    RuntimeVersion,
    UnsupportedResolvedQueryError,
)
from src.runtime.runtime_adapters import LegacyFinancialRuntimeAdapter


class FakeRAGEngine:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result if result is not None else {"answer": "ok"}
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def query(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def _request(**metadata: Any) -> FinancialQueryRequest:
    return FinancialQueryRequest(
        request_id="req-1",
        user_id="7",
        session_id="session-1",
        original_query="What is revenue?",
        request_metadata=metadata,
    )


def test_adapter_implements_runtime_protocol_and_maps_v1_answer() -> None:
    engine = FakeRAGEngine(
        {
            "answer": "Revenue was $100B.",
            "sources": [
                {
                    "filename": "report.pdf",
                    "page": 4,
                    "chunk_id": "user_7_report.pdf::chunk-1",
                },
            ],
            "validation": {
                "status": "passed",
                "issues": [],
            },
            "trace_id": "trace-1",
        },
    )
    adapter = LegacyFinancialRuntimeAdapter(engine)

    result = asyncio.run(adapter.execute(_request(document_names=["report.pdf"])))

    assert isinstance(adapter, FinancialQARuntime)
    assert isinstance(result, FinancialQueryResult)
    assert result.status is RuntimeStatus.ANSWER
    assert result.release_status is ReleaseStatus.RELEASED
    assert result.runtime_version is RuntimeVersion.V1
    assert result.answer == "Revenue was $100B."
    assert result.evidence_ids == ["user_7_report.pdf::chunk-1"]
    assert result.citation_ids == []
    assert result.calculation_ids == []
    assert result.citations[0]["filename"] == "report.pdf"
    assert engine.calls == [
        {
            "question": "What is revenue?",
            "doc_names": ["report.pdf"],
            "user_id": 7,
            "n_results": 3,
            "conversation_history": [],
            "memory_profile": None,
        },
    ]


def test_adapter_parity_with_direct_v1_engine_call() -> None:
    engine = FakeRAGEngine(
        {
            "answer": "Revenue was $100B.",
            "sources": [{"chunk_id": "chunk-1", "filename": "report.pdf"}],
            "validation": {"status": "passed", "issues": []},
        },
    )
    raw_result = asyncio.run(
        engine.query(
            question="What is revenue?",
            doc_names=["report.pdf"],
            user_id=7,
            n_results=3,
            conversation_history=[],
            memory_profile=None,
        ),
    )
    adapted_result = asyncio.run(
        adapter_result(engine, _request(document_names=["report.pdf"])),
    )

    assert adapted_result.answer == raw_result["answer"]
    assert adapted_result.citations == raw_result["sources"]
    assert adapted_result.evidence_ids == ["chunk-1"]
    assert adapted_result.status is RuntimeStatus.ANSWER
    assert adapted_result.release_status is ReleaseStatus.RELEASED


def test_adapter_maps_calculation_operand_provenance_without_inventing_id() -> None:
    engine = FakeRAGEngine(
        {
            "answer": "Growth was 10%.",
            "calculations": [
                {
                    "status": "executed",
                    "operands": [
                        {"name": "current", "evidence_chunk_id": "chunk-current"},
                        {"name": "prior", "evidence_chunk_id": "chunk-prior"},
                    ],
                },
            ],
            "validation": {"status": "passed", "issues": []},
        },
    )

    result = asyncio.run(adapter_result(engine, _request()))

    assert result.status is RuntimeStatus.ANSWER
    assert result.release_status is ReleaseStatus.RELEASED
    assert result.evidence_ids == ["chunk-current", "chunk-prior"]
    assert result.calculation_ids == []


def test_adapter_preserves_request_options_without_session_lifecycle() -> None:
    engine = FakeRAGEngine()
    request = _request(
        document_names=["a.pdf", "b.pdf"],
        n_results=5,
        conversation_history=[{"role": "user", "content": "Revenue?"}],
        memory_profile={"active_metric": "revenue"},
    )

    asyncio.run(adapter_result(engine, request))

    assert engine.calls[0] == {
        "question": "What is revenue?",
        "doc_names": ["a.pdf", "b.pdf"],
        "user_id": 7,
        "n_results": 5,
        "conversation_history": [{"role": "user", "content": "Revenue?"}],
        "memory_profile": {"active_metric": "revenue"},
    }


def test_query_as_resolved_fails_fast_before_engine_call() -> None:
    engine = FakeRAGEngine()
    request = FinancialQueryRequest(
        request_id="req-2",
        user_id="7",
        session_id="session-1",
        original_query="What about last year?",
        standalone_query="Apple FY2023 revenue",
        query_as_resolved=True,
    )

    with pytest.raises(UnsupportedResolvedQueryError, match="rewrite bypass"):
        asyncio.run(adapter_result(engine, request))
    assert engine.calls == []


def test_changed_standalone_query_without_flag_also_fails_fast() -> None:
    engine = FakeRAGEngine()
    request = FinancialQueryRequest(
        request_id="req-3",
        user_id="7",
        session_id="session-1",
        original_query="What about last year?",
        standalone_query="Apple FY2023 revenue",
    )

    with pytest.raises(UnsupportedResolvedQueryError):
        asyncio.run(adapter_result(engine, request))
    assert engine.calls == []


def test_engine_exception_becomes_explicit_error_result() -> None:
    engine = FakeRAGEngine(error=RuntimeError("secret internal error"))
    result = asyncio.run(adapter_result(engine, _request()))

    assert result.status is RuntimeStatus.ERROR
    assert result.release_status is ReleaseStatus.NOT_RELEASED
    assert result.answer is None
    assert result.reason_codes == ["LEGACY_RUNTIME_EXCEPTION"]
    assert result.debug_metadata["exception_type"] == "RuntimeError"
    assert "secret internal error" not in result.to_json()


def test_blocked_v1_result_is_fail_closed_without_answer_parsing() -> None:
    engine = FakeRAGEngine(
        {
            "answer": "I cannot answer this safely.",
            "answerability": {
                "status": "not_answerable",
                "reason_codes": ["MISSING_METRIC"],
            },
            "validation": {"status": "blocked", "issues": []},
        },
    )
    result = asyncio.run(adapter_result(engine, _request()))

    assert result.status is RuntimeStatus.FAIL_CLOSED
    assert result.release_status is ReleaseStatus.NOT_RELEASED
    assert result.reason_codes == ["MISSING_METRIC"]


def test_missing_validation_does_not_claim_release() -> None:
    engine = FakeRAGEngine({"answer": "Revenue was $100B."})
    result = asyncio.run(adapter_result(engine, _request()))

    assert result.status is RuntimeStatus.ANSWER
    assert result.release_status is ReleaseStatus.NOT_APPLICABLE
    assert result.evidence_ids == []


def test_malformed_v1_result_is_explicit_error() -> None:
    engine = FakeRAGEngine({"sources": []})
    result = asyncio.run(adapter_result(engine, _request()))

    assert result.status is RuntimeStatus.ERROR
    assert result.reason_codes == ["LEGACY_RESULT_INVALID"]


def test_result_contract_round_trip_after_adaptation() -> None:
    engine = FakeRAGEngine(
        {
            "answer": "Revenue was $100B.",
            "sources": [{"chunk_id": "chunk-1"}],
            "validation": {"status": "passed", "issues": []},
        },
    )
    result = asyncio.run(adapter_result(engine, _request()))
    assert FinancialQueryResult.from_json(result.to_json()) == result


async def adapter_result(
    engine: FakeRAGEngine,
    request: FinancialQueryRequest,
) -> FinancialQueryResult:
    return await LegacyFinancialRuntimeAdapter(engine).execute(request)
