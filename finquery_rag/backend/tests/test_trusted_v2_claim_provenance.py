"""Focused tests for structured claim-level Trusted V2 provenance."""

from __future__ import annotations

import asyncio

import pytest

from rag_v2.adaptive import AdaptiveRAGStateV1
from rag_v2.contracts import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from src.runtime import (
    ClaimProvenance,
    FinancialQueryRequest,
    FinancialQueryResult,
    ReleaseStatus,
    RuntimeStatus,
    RuntimeVersion,
    TrustedFinancialRuntimeV2,
    V2ExecutionOutcome,
    V2ExecutionRequest,
    V2ExecutionStatus,
    build_claim_provenance,
)
from src.runtime.trusted_v2_capabilities import TrustedV2CapabilityPorts
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator


def _plan(*, calculation: bool = False) -> SupervisorPlan:
    slots = (
        RequiredSlot(
            "current",
            "Revenue",
            "FY2024",
            "current" if calculation else "value",
            "numeric",
            None,
        ),
    )
    if calculation:
        slots += (
            RequiredSlot("prior", "Revenue", "FY2023", "prior", "numeric", None),
        )
    return SupervisorPlan(
        intent=Intent.CALCULATION if calculation else Intent.DIRECT_FACT,
        required_slots=slots,
        operation="growth_rate" if calculation else None,
        next_action=Action.RETRIEVE,
    )


def _state(plan: SupervisorPlan) -> AdaptiveRAGStateV1:
    state = AdaptiveRAGStateV1.new(
        request_id="claim-test",
        query="What was revenue?",
        intent=plan.intent.value,
        task_type=plan.intent.value,
        required_slots=[slot.to_dict() for slot in plan.required_slots],
        plan={"supervisor_plan": plan.to_dict()},
    )
    state.evidence_packets = [
        {
            "evidence_id": "candidate-only",
            "citation_id": "CANDIDATE-CITATION",
        },
        {
            "evidence_id": "E1",
            "citation_id": "C1",
        },
        {
            "evidence_id": "E2",
            "citation_id": "C2",
        },
    ]
    state.bound_evidence_ids = ["E1", "E2"]
    state.bound_slot_bindings = {"current": ["E1"]}
    if len(plan.required_slots) > 1:
        state.bound_slot_bindings["prior"] = ["E2"]
    return state


def _request() -> V2ExecutionRequest:
    return V2ExecutionRequest(
        request_id="claim-test",
        user_id="user-1",
        session_id="session-1",
        original_query="What was revenue?",
        standalone_query="What was revenue?",
    )


def test_claim_builder_uses_bound_slots_not_candidate_pool() -> None:
    plan = _plan()
    claims = build_claim_provenance(
        plan=plan,
        state=_state(plan),
        evidence_ids=["candidate-only", "E1"],
        citation_ids=["CANDIDATE-CITATION", "C1"],
        release_status=ReleaseStatus.RELEASED,
        validator_status="PASS",
    )

    assert [claim.claim_id for claim in claims] == ["slot:current"]
    assert claims[0].bound_evidence_ids == ("E1",)
    assert claims[0].citation_ids == ("C1",)
    assert "candidate-only" not in claims[0].bound_evidence_ids


def test_calculation_claim_links_all_bound_slots_to_structured_result() -> None:
    plan = _plan(calculation=True)
    claims = build_claim_provenance(
        plan=plan,
        state=_state(plan),
        evidence_ids=["E1", "E2"],
        citation_ids=["C1", "C2"],
        calculation_ids=["CALC-1", "CALC-1"],
        release_status=ReleaseStatus.RELEASED,
        validator_status="PASS",
    )

    assert [claim.claim_id for claim in claims] == [
        "slot:current",
        "slot:prior",
        "calculation:CALC-1",
    ]
    calculation = claims[-1]
    assert calculation.required_slot_ids == ("current", "prior")
    assert calculation.bound_evidence_ids == ("E1", "E2")
    assert calculation.citation_ids == ("C1", "C2")
    assert calculation.calculation_ids == ("CALC-1",)


def test_answer_text_cannot_create_claim_provenance() -> None:
    result = FinancialQueryResult(
        status=RuntimeStatus.ANSWER,
        answer="$999B [chunk_xyz]",
        runtime_version=RuntimeVersion.V2,
        release_status=ReleaseStatus.RELEASED,
    )

    assert result.claim_provenance == ()
    assert result.evidence_ids == []
    assert result.to_dict()["claim_provenance"] == []


def test_v2_outcome_round_trip_preserves_typed_claims() -> None:
    claim = ClaimProvenance(
        claim_id="slot:revenue",
        required_slot_ids=("revenue",),
        bound_evidence_ids=("E1",),
        citation_ids=("C1",),
        release_status=ReleaseStatus.RELEASED,
        validator_status="PASS",
    )
    outcome = V2ExecutionOutcome(
        status=V2ExecutionStatus.READY_FOR_RELEASE,
        answer="Revenue was $100B.",
        evidence_ids=["E1"],
        citation_ids=["C1"],
        release_status=ReleaseStatus.RELEASED,
        claim_provenance=(claim,),
    )

    restored = V2ExecutionOutcome.from_json(outcome.to_json())

    assert restored == outcome
    assert restored.claim_provenance[0] == claim


def test_claim_release_status_must_match_v2_outcome() -> None:
    claim = ClaimProvenance(claim_id="slot:value")
    with pytest.raises(ValueError, match="claim provenance release status"):
        V2ExecutionOutcome(
            status=V2ExecutionStatus.READY_FOR_RELEASE,
            answer="candidate",
            release_status=ReleaseStatus.RELEASED,
            claim_provenance=(claim,),
        )


def test_coordinator_emits_claims_at_final_structured_outcome_boundary() -> None:
    plan = _plan()
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(DeterministicFallbackProvider({})),
        capabilities=TrustedV2CapabilityPorts(),
    )
    outcome = coordinator._outcome(
        request=_request(),
        plan=plan,
        plan_id="plan-1",
        state=_state(plan),
        reason_codes=["FINAL_VALIDATION_NOT_WIRED"],
        status=V2ExecutionStatus.FAIL_CLOSED,
        answer="$999B [chunk_xyz]",
        evidence_ids=["candidate-only", "E1"],
        citation_ids=["CANDIDATE-CITATION", "C1"],
        terminal_state="CANDIDATE_READY_FOR_VALIDATION",
        validator_status="FAIL",
    )

    assert outcome.claim_provenance[0].claim_id == "slot:current"
    assert outcome.claim_provenance[0].bound_evidence_ids == ("E1",)
    assert outcome.claim_provenance[0].release_status is ReleaseStatus.NOT_RELEASED
    assert outcome.debug_metadata["trace"]["claim_provenance"] == [
        outcome.claim_provenance[0].to_dict()
    ]


def test_adapter_transmits_claims_without_parsing_answer() -> None:
    claim = ClaimProvenance(
        claim_id="slot:value",
        required_slot_ids=("value",),
        bound_evidence_ids=("E1",),
        citation_ids=("C1",),
        release_status=ReleaseStatus.RELEASED,
    )

    class Coordinator:
        async def execute(self, request: V2ExecutionRequest) -> V2ExecutionOutcome:
            del request
            return V2ExecutionOutcome(
                status=V2ExecutionStatus.READY_FOR_RELEASE,
                answer="$999B [chunk_xyz]",
                release_status=ReleaseStatus.RELEASED,
                claim_provenance=(claim,),
            )

    request = FinancialQueryRequest(
        request_id="claim-adapter",
        user_id="user-1",
        session_id="session-1",
        original_query="What was revenue?",
    )
    result = asyncio.run(TrustedFinancialRuntimeV2(Coordinator()).execute(request))

    assert result.claim_provenance == (claim,)
    assert result.evidence_ids == []
