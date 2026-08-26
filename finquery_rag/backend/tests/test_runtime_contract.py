"""Unit tests for the I1 unified financial runtime contract."""

from __future__ import annotations

import asyncio
import json
from dataclasses import fields

import pytest

from src.runtime import (
    ClarificationPayload,
    FinancialQARuntime,
    FinancialQueryRequest,
    FinancialQueryResult,
    ReleaseStatus,
    RuntimeRouterMode,
    RuntimeStatus,
    RuntimeVersion,
)


def test_request_preserves_original_and_defaults_standalone_query() -> None:
    request = FinancialQueryRequest(
        request_id="req-1",
        user_id="user-7",
        session_id="session-9",
        original_query="What about last year?",
    )

    assert request.original_query == "What about last year?"
    assert request.standalone_query == request.original_query
    assert request.query_as_resolved is False
    assert request.conversation_metadata == {}
    assert request.request_metadata == {}


def test_request_supports_resolved_query_without_legacy_rewrite_signal() -> None:
    request = FinancialQueryRequest(
        request_id="req-2",
        user_id="user-7",
        session_id="session-9",
        original_query="What about last year?",
        standalone_query="Apple FY2023 revenue",
        query_as_resolved=True,
        conversation_metadata={"active_entity": "AAPL"},
    )

    assert request.original_query == "What about last year?"
    assert request.standalone_query == "Apple FY2023 revenue"
    assert request.query_as_resolved is True
    assert request.conversation_metadata == {"active_entity": "AAPL"}


@pytest.mark.parametrize(
    "field_name",
    ["request_id", "user_id", "session_id", "original_query"],
)
def test_request_requires_identity_and_original_query(field_name: str) -> None:
    values = {
        "request_id": "req-1",
        "user_id": "user-7",
        "session_id": "session-9",
        "original_query": "Revenue?",
    }
    values[field_name] = ""

    with pytest.raises((TypeError, ValueError)):
        FinancialQueryRequest(**values)


def test_request_json_round_trip_is_stable() -> None:
    request = FinancialQueryRequest(
        request_id="req-3",
        user_id="user-7",
        session_id="session-9",
        original_query="Revenue?",
        standalone_query="AAPL FY2025 revenue",
        query_as_resolved=True,
        conversation_metadata={"period": "FY2025"},
        request_metadata={"source": "test"},
    )

    encoded = request.to_json()
    assert encoded == request.to_json()
    assert FinancialQueryRequest.from_json(encoded).to_dict() == request.to_dict()
    assert json.loads(encoded)["query_as_resolved"] is True


def test_all_runtime_statuses_are_serializable() -> None:
    for status in (
        RuntimeStatus.ANSWER,
        RuntimeStatus.OUT_OF_SCOPE,
        RuntimeStatus.FAIL_CLOSED,
        RuntimeStatus.ERROR,
    ):
        result = FinancialQueryResult(
            status=status,
            runtime_version=RuntimeVersion.V1,
            release_status=ReleaseStatus.NOT_APPLICABLE,
        )
        decoded = FinancialQueryResult.from_json(result.to_json())
        assert decoded.status is status
        assert decoded.runtime_version is RuntimeVersion.V1


def test_clarification_requires_structured_payload_and_no_answer_text() -> None:
    clarification = ClarificationPayload(
        question="Which period do you mean?",
        reason_codes=["AMBIGUOUS_PERIOD"],
        options=["FY2024", "FY2025"],
    )
    result = FinancialQueryResult(
        status=RuntimeStatus.CLARIFICATION_REQUIRED,
        clarification=clarification,
    )

    assert result.answer is None
    assert result.clarification == clarification
    assert FinancialQueryResult.from_dict(result.to_dict()) == result

    with pytest.raises(ValueError):
        FinancialQueryResult(status=RuntimeStatus.CLARIFICATION_REQUIRED)

    with pytest.raises(ValueError):
        FinancialQueryResult(
            status=RuntimeStatus.CLARIFICATION_REQUIRED,
            answer="FY2025",
            clarification=clarification,
        )


def test_version_and_router_mode_are_separate() -> None:
    result = FinancialQueryResult(
        status=RuntimeStatus.ANSWER,
        answer="answer",
        runtime_version=RuntimeVersion.V2,
        router_mode=RuntimeRouterMode.SHADOW,
        release_status=ReleaseStatus.NOT_RELEASED,
        runtime_metadata={"implementation": "trusted_v2"},
        evidence_ids=["structured:fact:1"],
        citation_ids=["citation:1"],
        calculation_ids=["calculation:1"],
        citations=[{"evidence_id": "structured:fact:1"}],
    )

    assert result.runtime_version is RuntimeVersion.V2
    assert result.router_mode is RuntimeRouterMode.SHADOW
    assert result.release_status is ReleaseStatus.NOT_RELEASED
    assert result.evidence_ids == ["structured:fact:1"]
    assert result.runtime_metadata is not None
    assert result.to_dict()["router_mode"] == "SHADOW"


def test_non_clarification_results_do_not_carry_clarification() -> None:
    with pytest.raises(ValueError):
        FinancialQueryResult(
            status=RuntimeStatus.ANSWER,
            clarification=ClarificationPayload(question="Clarify"),
        )


def test_contract_has_no_answer_derived_provenance_fields() -> None:
    result = FinancialQueryResult(status=RuntimeStatus.FAIL_CLOSED)

    assert result.evidence_ids == []
    assert result.citation_ids == []
    assert result.calculation_ids == []
    assert not hasattr(result, "parsed_answer_value")
    assert not hasattr(result, "assistant_answer_as_evidence")
    assert not hasattr(result, "historical_financial_fact")


def test_runtime_protocol_is_async_and_structurally_checkable() -> None:
    class StubRuntime:
        async def execute(
            self,
            request: FinancialQueryRequest,
        ) -> FinancialQueryResult:
            await asyncio.sleep(0)
            return FinancialQueryResult(status=RuntimeStatus.FAIL_CLOSED)

    assert isinstance(StubRuntime(), FinancialQARuntime)
    assert len(fields(FinancialQueryRequest)) == 8
