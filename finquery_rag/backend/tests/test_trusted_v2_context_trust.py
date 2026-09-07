"""Context trust and structured trace invariants for Trusted V2."""

from __future__ import annotations

from rag_v2.adaptive import AdaptiveRAGStateV1
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from src.runtime import (
    ContextTrustLevel,
    FinancialQueryRequest,
    V2ExecutionRequest,
)
from src.runtime.trusted_v2_capabilities import TrustedV2CapabilityPorts
from src.runtime.trusted_v2_contracts import V2ExecutionRequest as ContractRequest
from src.runtime.trusted_v2_coordinator import (
    BoundedTrustedV2Coordinator,
    V2ExecutionTrace,
)


def _request(**kwargs: object) -> V2ExecutionRequest:
    payload: dict[str, object] = {
        "request_id": "context-trust-request",
        "user_id": "user-1",
        "session_id": "session-1",
        "original_query": "What was revenue?",
        "standalone_query": "What was revenue?",
    }
    payload.update(kwargs)
    return V2ExecutionRequest(**payload)


def test_only_binder_admitted_evidence_can_cross_fact_boundary() -> None:
    assert ContextTrustLevel.USER_EXPLICIT_QUERY.can_enter_financial_fact_chain is False
    assert ContextTrustLevel.ASSISTANT_TEXT.can_enter_financial_fact_chain is False
    assert ContextTrustLevel.MODEL_GENERATED_SUMMARY.can_enter_financial_fact_chain is False
    assert ContextTrustLevel.RETRIEVED_CANDIDATE.can_enter_financial_fact_chain is False
    assert ContextTrustLevel.BINDER_ADMITTED_EVIDENCE.can_enter_financial_fact_chain is True


def test_request_context_levels_are_semantic_only_and_round_trip() -> None:
    request = ContractRequest(
        request_id="request-1",
        user_id="user-1",
        session_id="session-1",
        original_query="What about last year?",
        standalone_query="What was Apple FY2023 revenue?",
        conversation_resolved=True,
        context_trust_levels=(
            ContextTrustLevel.ASSISTANT_TEXT,
            ContextTrustLevel.MODEL_GENERATED_SUMMARY,
        ),
    )

    assert request.context_trust_levels == (
        ContextTrustLevel.USER_EXPLICIT_QUERY,
        ContextTrustLevel.ASSISTANT_TEXT,
        ContextTrustLevel.MODEL_GENERATED_SUMMARY,
        ContextTrustLevel.STRUCTURED_DIALOGUE_STATE,
    )
    assert all(
        not level.can_enter_financial_fact_chain
        for level in request.context_trust_levels
    )
    assert ContractRequest.from_json(request.to_json()) == request


def test_request_cannot_self_attest_binder_admission() -> None:
    try:
        ContractRequest(
            request_id="request-1",
            user_id="user-1",
            session_id="session-1",
            original_query="Revenue?",
            standalone_query="Revenue?",
            context_trust_levels=(ContextTrustLevel.BINDER_ADMITTED_EVIDENCE,),
        )
    except ValueError as exc:
        assert "Binder admission" in str(exc)
    else:
        raise AssertionError("request boundary accepted Binder authority")


def test_financial_request_can_label_semantic_context_without_granting_authority() -> None:
    financial_request = FinancialQueryRequest(
        request_id="request-1",
        user_id="user-1",
        session_id="session-1",
        original_query="What about last year?",
        standalone_query="What was Apple FY2023 revenue?",
        query_as_resolved=True,
        conversation_metadata={
            "context_trust_levels": [
                ContextTrustLevel.COMPRESSED_HISTORY,
                ContextTrustLevel.ASSISTANT_TEXT,
            ],
        },
    )

    request = ContractRequest.from_financial_request(financial_request)

    assert ContextTrustLevel.COMPRESSED_HISTORY in request.context_trust_levels
    assert ContextTrustLevel.ASSISTANT_TEXT in request.context_trust_levels
    assert ContextTrustLevel.BINDER_ADMITTED_EVIDENCE not in request.context_trust_levels


def test_trace_records_context_authority_and_stable_execution_id() -> None:
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(DeterministicFallbackProvider({})),
        capabilities=TrustedV2CapabilityPorts(),
    )
    request = _request(
        conversation_resolved=True,
        context_trust_levels=(ContextTrustLevel.ASSISTANT_TEXT,),
    )
    state = AdaptiveRAGStateV1.new(
        request_id=request.request_id,
        query=request.standalone_query,
        intent="DIRECT_FACT",
        task_type="DIRECT_FACT",
        required_slots=[],
    )
    state.evidence_packets = [{"evidence_id": "candidate-1"}]
    state.bound_evidence_ids = ["admitted-1"]

    first = coordinator._trace(
        request,
        "plan-1",
        state,
        (),
        "READY_TO_GENERATE",
    )
    second = coordinator._trace(
        request,
        "plan-1",
        state,
        (),
        "READY_TO_GENERATE",
    )
    trace = first.to_dict()

    assert first.execution_id == second.execution_id
    assert trace["plan_id"] == "plan-1"
    assert trace["execution_id"]
    assert trace["context_trust_levels"] == [
        "USER_EXPLICIT_QUERY",
        "ASSISTANT_TEXT",
        "STRUCTURED_DIALOGUE_STATE",
        "RETRIEVED_CANDIDATE",
        "BINDER_ADMITTED_EVIDENCE",
    ]
    assert trace["financial_fact_context_levels"] == [
        "BINDER_ADMITTED_EVIDENCE",
    ]


def test_trace_drops_provider_private_reasoning_recursively() -> None:
    state = AdaptiveRAGStateV1.new(
        request_id="trace-request",
        query="Revenue?",
        intent="DIRECT_FACT",
        task_type="DIRECT_FACT",
        required_slots=[],
    )
    trace = V2ExecutionTrace.from_state(
        request_id="trace-request",
        plan_id="plan-1",
        state=state,
        reason_codes=(),
        capability_trace={
            "retrieval": {
                "route": "R4",
                "chain_of_thought": "secret",
                "nested": {"model_reasoning": "secret"},
            },
        },
    ).to_dict()

    serialized = str(trace).casefold()
    assert trace["retrieval_rounds"] == []
    assert "secret" not in serialized
    assert "chain_of_thought" not in serialized
    assert "model_reasoning" not in serialized


def test_trace_financial_authority_rejects_non_binder_level() -> None:
    state = AdaptiveRAGStateV1.new(
        request_id="trace-request",
        query="Revenue?",
        intent="DIRECT_FACT",
        task_type="DIRECT_FACT",
        required_slots=[],
    )
    try:
        V2ExecutionTrace.from_state(
            request_id="trace-request",
            plan_id="plan-1",
            state=state,
            reason_codes=(),
            financial_fact_context_levels=(
                ContextTrustLevel.ASSISTANT_TEXT,
            ),
        )
    except ValueError as exc:
        assert "BINDER_ADMITTED_EVIDENCE" in str(exc)
    else:
        raise AssertionError("trace accepted non-Binder financial authority")
