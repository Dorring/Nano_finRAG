"""TV2-01 tests for the Trusted Financial Runtime V2 adapter shell."""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest

from src.runtime import (
    FinancialQARuntime,
    FinancialQueryRequest,
    ReleaseStatus,
    RuntimeStatus,
    RuntimeVersion,
    TrustedFinancialRuntimeV2,
    V2ExecutionOutcome,
    V2ExecutionRequest,
    V2ExecutionStatus,
)


class FakeCoordinator:
    def __init__(
        self,
        outcome: V2ExecutionOutcome | None = None,
        error: Exception | None = None,
    ) -> None:
        self.outcome = outcome
        self.error = error
        self.requests: list[V2ExecutionRequest] = []

    async def execute(self, request: V2ExecutionRequest) -> V2ExecutionOutcome:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.outcome is None:
            raise AssertionError("fake coordinator outcome was not configured")
        return self.outcome


def _request(**kwargs: Any) -> FinancialQueryRequest:
    return FinancialQueryRequest(
        request_id="req-v2-1",
        user_id="user-7",
        session_id="session-1",
        original_query=kwargs.pop("original_query", "What is revenue?"),
        standalone_query=kwargs.pop("standalone_query", None),
        query_as_resolved=kwargs.pop("query_as_resolved", False),
        request_metadata=kwargs.pop("request_metadata", {}),
        conversation_metadata=kwargs.pop("conversation_metadata", {}),
    )


def _released_outcome() -> V2ExecutionOutcome:
    return V2ExecutionOutcome(
        status=V2ExecutionStatus.READY_FOR_RELEASE,
        answer="Revenue was $100B.",
        citations=[{"citation_id": "cite-1", "page": 4}],
        evidence_ids=["fact-1", "fact-1"],
        citation_ids=["cite-1"],
        calculation_ids=["calc-1"],
        reason_codes=["READY"],
        release_status=ReleaseStatus.RELEASED,
        route="DIRECT_FACT",
        validator_status="PASSED",
        runtime_metadata={"stage": "coordinator"},
        latency_metadata={"total_ms": 12},
    )


def test_adapter_is_runtime_port_and_maps_released_outcome() -> None:
    coordinator = FakeCoordinator(_released_outcome())
    adapter = TrustedFinancialRuntimeV2(coordinator)

    result = asyncio.run(adapter.execute(_request()))

    assert isinstance(adapter, FinancialQARuntime)
    assert result.status is RuntimeStatus.ANSWER
    assert result.release_status is ReleaseStatus.RELEASED
    assert result.runtime_version is RuntimeVersion.V2
    assert result.answer == "Revenue was $100B."
    assert result.evidence_ids == ["fact-1"]
    assert result.citation_ids == ["cite-1"]
    assert result.calculation_ids == ["calc-1"]
    assert result.citations == [{"citation_id": "cite-1", "page": 4}]
    assert result.runtime_metadata is not None
    assert result.runtime_metadata.attributes["production_routing"] is False
    assert result.runtime_metadata.attributes["route"] == "DIRECT_FACT"


def test_adapter_passes_standalone_query_and_does_not_reuse_legacy_rewrite_flag() -> None:
    coordinator = FakeCoordinator(_released_outcome())
    adapter = TrustedFinancialRuntimeV2(coordinator)
    request = _request(
        original_query="What about last year?",
        standalone_query="What was Apple FY2023 revenue?",
        query_as_resolved=True,
    )

    asyncio.run(adapter.execute(request))

    received = coordinator.requests[0]
    assert received.original_query == "What about last year?"
    assert received.standalone_query == "What was Apple FY2023 revenue?"
    assert received.conversation_resolved is True


def test_adapter_accepts_off_style_request_without_resolution() -> None:
    coordinator = FakeCoordinator(_released_outcome())
    adapter = TrustedFinancialRuntimeV2(coordinator)

    asyncio.run(adapter.execute(_request()))

    received = coordinator.requests[0]
    assert received.standalone_query == received.original_query
    assert received.conversation_resolved is False


def test_v2_request_drops_uncontrolled_raw_conversation_context() -> None:
    coordinator = FakeCoordinator(_released_outcome())
    adapter = TrustedFinancialRuntimeV2(coordinator)
    request = _request(
        request_metadata={
            "document_names": ["report.pdf"],
            "n_results": 5,
            "conversation_history": [{"role": "user", "content": "old"}],
            "memory_profile": {"active_metric": "revenue"},
        },
        conversation_metadata={
            "active_metric": "revenue",
            "recent_turns": [{"role": "user", "content": "old"}],
        },
    )

    asyncio.run(adapter.execute(request))

    received = coordinator.requests[0]
    assert received.request_metadata == {
        "document_names": ["report.pdf"],
        "n_results": 5,
    }
    assert received.conversation_metadata == {"active_metric": "revenue"}


def test_fail_closed_candidate_answer_never_maps_to_answer_status() -> None:
    outcome = V2ExecutionOutcome(
        status=V2ExecutionStatus.FAIL_CLOSED,
        answer="candidate text that was rejected",
        reason_codes=["NUMERIC_VALIDATION_FAILED"],
        release_status=ReleaseStatus.NOT_RELEASED,
    )
    result = asyncio.run(
        TrustedFinancialRuntimeV2(FakeCoordinator(outcome)).execute(_request()),
    )

    assert result.status is RuntimeStatus.FAIL_CLOSED
    assert result.release_status is ReleaseStatus.NOT_RELEASED
    assert result.answer == "candidate text that was rejected"


def test_coordinator_exception_is_error_not_trust_policy_fail_closed() -> None:
    result = asyncio.run(
        TrustedFinancialRuntimeV2(
            FakeCoordinator(error=RuntimeError("secret")),
        ).execute(_request()),
    )

    assert result.status is RuntimeStatus.ERROR
    assert result.release_status is ReleaseStatus.NOT_RELEASED
    assert result.reason_codes == ["V2_COORDINATOR_EXCEPTION"]
    assert result.debug_metadata["exception_type"] == "RuntimeError"
    assert "secret" not in result.to_json()


def test_invalid_coordinator_output_is_explicit_error() -> None:
    class InvalidCoordinator:
        async def execute(self, request: V2ExecutionRequest) -> dict[str, Any]:
            return {"answer": "not an outcome"}

    result = asyncio.run(
        TrustedFinancialRuntimeV2(InvalidCoordinator()).execute(_request()),
    )

    assert result.status is RuntimeStatus.ERROR
    assert result.release_status is ReleaseStatus.NOT_RELEASED
    assert result.reason_codes == ["V2_OUTCOME_INVALID"]


def test_answer_text_does_not_create_provenance() -> None:
    outcome = V2ExecutionOutcome(
        status=V2ExecutionStatus.READY_FOR_RELEASE,
        answer="Revenue was $999B [chunk-abc].",
        release_status=ReleaseStatus.RELEASED,
    )
    result = asyncio.run(
        TrustedFinancialRuntimeV2(FakeCoordinator(outcome)).execute(_request()),
    )

    assert result.status is RuntimeStatus.ANSWER
    assert result.evidence_ids == []
    assert result.citation_ids == []
    assert result.calculation_ids == []


def test_outcome_contract_rejects_invalid_status_release_combinations() -> None:
    with pytest.raises(ValueError, match="READY_FOR_RELEASE"):
        V2ExecutionOutcome(
            status=V2ExecutionStatus.READY_FOR_RELEASE,
            answer="answer",
            release_status=ReleaseStatus.NOT_RELEASED,
        )
    with pytest.raises(ValueError, match="FAIL_CLOSED"):
        V2ExecutionOutcome(
            status=V2ExecutionStatus.FAIL_CLOSED,
            release_status=ReleaseStatus.RELEASED,
        )


def test_v2_request_and_outcome_json_round_trip() -> None:
    request = V2ExecutionRequest.from_financial_request(
        _request(
            standalone_query="Apple FY2023 revenue",
            query_as_resolved=True,
            request_metadata={"document_names": ["report.pdf"]},
        ),
    )
    outcome = _released_outcome()

    assert V2ExecutionRequest.from_json(request.to_json()) == request
    assert V2ExecutionOutcome.from_json(outcome.to_json()) == outcome


def test_adapter_does_not_construct_trusted_rag_runtime() -> None:
    module = importlib.import_module("src.runtime.trusted_v2_adapter")
    assert not hasattr(module, "TrustedRAGRuntimeV2")
