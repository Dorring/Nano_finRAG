"""TV2-08 official V2 routing and default-activation tests."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.runtime import (
    FinancialQueryRequest,
    FinancialRuntimeRouter,
    FinancialRuntimeModeError,
    FinancialQueryResult,
    ReleaseStatus,
    RuntimeStatus,
    RuntimeVersion,
    TrustedFinancialRuntimeV2,
    V2ExecutionOutcome,
    V2ExecutionStatus,
    resolve_financial_runtime_mode,
)
from src.conversation.config import resolve_multiturn_context_mode


class _Runtime:
    def __init__(
        self,
        result: FinancialQueryResult,
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls = 0
        self.requests: list[FinancialQueryRequest] = []

    async def execute(self, request: FinancialQueryRequest) -> FinancialQueryResult:
        self.calls += 1
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


class _Coordinator:
    def __init__(self, outcome: V2ExecutionOutcome) -> None:
        self.outcome = outcome
        self.calls = 0
        self.requests: list[Any] = []

    async def execute(self, request: Any) -> V2ExecutionOutcome:
        self.calls += 1
        self.requests.append(request)
        return self.outcome


def _request() -> FinancialQueryRequest:
    return FinancialQueryRequest(
        request_id="tv2-08",
        user_id="42",
        session_id="tv2-08-session",
        original_query="What about last year?",
        standalone_query="What was Apple FY2023 revenue?",
        query_as_resolved=True,
        request_metadata={
            "conversation_history": [{"role": "assistant", "content": "old"}],
        },
    )


def _result(
    answer: str | None,
    *,
    version: RuntimeVersion,
    status: RuntimeStatus = RuntimeStatus.ANSWER,
    release: ReleaseStatus = ReleaseStatus.RELEASED,
) -> FinancialQueryResult:
    return FinancialQueryResult(
        status=status,
        answer=answer,
        runtime_version=version,
        release_status=release,
    )


def _v2(
    *,
    status: V2ExecutionStatus = V2ExecutionStatus.READY_FOR_RELEASE,
    answer: str | None = "V2 official",
) -> TrustedFinancialRuntimeV2:
    outcome = V2ExecutionOutcome(
        status=status,
        answer=answer,
        release_status=(
            ReleaseStatus.RELEASED
            if status is V2ExecutionStatus.READY_FOR_RELEASE
            else ReleaseStatus.NOT_RELEASED
        ),
        evidence_ids=["E1"] if status is V2ExecutionStatus.READY_FOR_RELEASE else [],
    )
    return TrustedFinancialRuntimeV2(_Coordinator(outcome))


def test_runtime_mode_defaults_to_v2_and_supports_rollback_modes() -> None:
    assert resolve_financial_runtime_mode(environ={}) == "v2"
    assert resolve_financial_runtime_mode("v1") == "v1"
    assert resolve_financial_runtime_mode("shadow") == "shadow"
    assert resolve_financial_runtime_mode("v2") == "v2"


def test_default_conversation_mode_is_active() -> None:
    assert resolve_multiturn_context_mode(environ={}) == "on"
    assert resolve_multiturn_context_mode(environ={"MULTITURN_CONTEXT_MODE": "off"}) == "off"


def test_v2_router_is_official_and_never_calls_v1() -> None:
    v1 = _Runtime(_result("V1", version=RuntimeVersion.V1))
    v2 = _Runtime(_result("V2", version=RuntimeVersion.V2))
    router = FinancialRuntimeRouter(v1, v2_runtime=v2, mode="v2")

    result = asyncio.run(router.execute(_request()))

    assert result is v2.result
    assert router.mode == "v2"
    assert router.primary_runtime is v1
    assert router.v1_calls == 0
    assert router.v2_calls == 1
    assert v1.calls == 0
    assert v2.calls == 1
    assert v2.requests[0].standalone_query == "What was Apple FY2023 revenue?"


def test_v2_router_does_not_fallback_when_v2_errors() -> None:
    v1 = _Runtime(_result("V1", version=RuntimeVersion.V1))
    v2 = _Runtime(_result("unused", version=RuntimeVersion.V2), error=RuntimeError("v2 failed"))
    router = FinancialRuntimeRouter(v1, v2_runtime=v2, mode="v2")

    with pytest.raises(RuntimeError, match="v2 failed"):
        asyncio.run(router.execute(_request()))

    assert v1.calls == 0
    assert router.v1_calls == 0
    assert v2.calls == 1


def test_v2_runtime_can_return_fail_closed_as_official_result() -> None:
    v1 = _Runtime(_result("V1", version=RuntimeVersion.V1))
    coordinator = _Coordinator(
        V2ExecutionOutcome(
            status=V2ExecutionStatus.FAIL_CLOSED,
            reason_codes=["MISSING_SLOT"],
            release_status=ReleaseStatus.NOT_RELEASED,
        ),
    )
    v2 = TrustedFinancialRuntimeV2(coordinator)
    router = FinancialRuntimeRouter(v1, v2_runtime=v2, mode="v2")

    result = asyncio.run(router.execute(_request()))

    assert result.status is RuntimeStatus.FAIL_CLOSED
    assert result.release_status is ReleaseStatus.NOT_RELEASED
    assert result.runtime_version is RuntimeVersion.V2
    assert v1.calls == 0
    assert coordinator.calls == 1


def test_v2_router_requires_explicit_real_runtime() -> None:
    with pytest.raises(ValueError, match="v2 mode"):
        FinancialRuntimeRouter(mode="v2")
    with pytest.raises(FinancialRuntimeModeError):
        resolve_financial_runtime_mode("canary")


def test_v2_structured_result_maps_to_legacy_transport_without_text_provenance() -> None:
    from src.runtime.response_mapper import to_legacy_query_dict

    released = FinancialQueryResult(
        status=RuntimeStatus.ANSWER,
        answer="Revenue: 391 USD billion [chunk-text]",
        runtime_version=RuntimeVersion.V2,
        release_status=ReleaseStatus.RELEASED,
        evidence_ids=["E1"],
        citation_ids=["C1"],
        calculation_ids=["CALC-1"],
        citations=[{"citation_id": "C1", "evidence_id": "E1"}],
    )
    payload = to_legacy_query_dict(released)
    assert payload["answer"] == released.answer
    assert payload["sources"] == released.citations
    assert payload["searched_docs"] == []
    assert payload["retrieved_chunks"] == []

    refused = FinancialQueryResult(
        status=RuntimeStatus.FAIL_CLOSED,
        runtime_version=RuntimeVersion.V2,
        release_status=ReleaseStatus.NOT_RELEASED,
        reason_codes=["MISSING_SLOT"],
    )
    refused_payload = to_legacy_query_dict(refused)
    assert refused_payload["answer"]
    assert refused_payload["sources"] == []
