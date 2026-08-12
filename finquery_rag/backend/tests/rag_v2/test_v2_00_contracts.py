from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_v2.contracts import (
    Action,
    AnswerEnvelope,
    BindingStatus,
    BoundFact,
    CanonicalAnswer,
    CanonicalSource,
    CheckStatus,
    ContractError,
    EvidenceBinding,
    Intent,
    RequiredSlot,
    SupervisorPlan,
    ValidationDecision,
    ValidationResult,
    VerifiedEvidencePacket,
)
from rag_v2.contracts.calculation import CalculationResultPacket, CalculationStatus
from rag_v2.contracts.errors import StateTransitionError
from rag_v2.orchestration import RepairBudget, State, StateMachine, load_question_envelopes
from rag_v2.supervisor import validate_plan


ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"


def make_slot(slot_id: str = "slot_1", metric: str = "revenue", period: str = "FY2025") -> RequiredSlot:
    return RequiredSlot(slot_id, metric, period, "current", "currency", "USD")


def make_binding() -> EvidenceBinding:
    return EvidenceBinding(BindingStatus.BOUND, {"slot_1": ("fact_1",)})


def make_fact() -> BoundFact:
    return BoundFact(
        fact_id="fact_1",
        candidate_id="candidate_1",
        physical_source_id="source_1",
        document_id="doc_1",
        pdf_page=1,
        metric="revenue",
        period="FY2025",
        value="100",
        currency="USD",
        scale="1",
        unit="currency",
        citation_id="source_1",
        slot_id="slot_1",
    )


def make_validation(decision: ValidationDecision) -> ValidationResult:
    status = CheckStatus.PASS if decision is ValidationDecision.PASS else CheckStatus.FAIL
    return ValidationResult(status, status, status, status, status, status, status, decision, () if status is CheckStatus.PASS else ("citation",))


def test_v2_00_loads_all_72_question_envelopes_without_answer_fields() -> None:
    envelopes = load_question_envelopes(QUESTIONS)
    assert len(envelopes) == 72
    assert len({item.question_id for item in envelopes}) == 72
    assert all(set(item.to_dict()) == {"question_id", "question", "document_scope"} for item in envelopes)


def test_supervisor_plan_schema_has_no_no_answer_intent() -> None:
    plan = SupervisorPlan(Intent.DIRECT_FACT, (make_slot(),), None, Action.RETRIEVE)
    assert validate_plan(plan) is plan
    payload = plan.to_dict()
    assert "NO_ANSWER" not in {item.value for item in Intent}
    assert SupervisorPlan.from_dict(json.loads(json.dumps(payload))) == plan
    with pytest.raises(ContractError):
        SupervisorPlan.from_dict({**payload, "intent": "NO_ANSWER"})
    with pytest.raises(ContractError):
        validate_plan(SupervisorPlan(Intent.DIRECT_FACT, (make_slot(),), None, Action.CALCULATE))


def test_verified_packet_and_answer_cannot_escape_canonical_sources() -> None:
    packet = VerifiedEvidencePacket("What was revenue?", Intent.DIRECT_FACT, (make_fact(),), None, ("source_1",))
    assert packet.to_dict()["allowed_citations"] == ["source_1"]
    canonical = CanonicalAnswer("100", "FY2025", "USD", "1", "currency", CanonicalSource.FINANCIAL_FACT, ("source_1",))
    envelope = AnswerEnvelope(canonical, "Revenue was $100.", ("source_1",))
    assert envelope.to_dict()["canonical_answer"]["value"] == "100"
    with pytest.raises(ContractError):
        AnswerEnvelope(canonical, "Revenue was $100.", ("other_source",))


def test_state_machine_blocks_generation_without_bound_evidence() -> None:
    machine = StateMachine()
    machine.accept_plan(SupervisorPlan(Intent.DIRECT_FACT, (make_slot(),), None, Action.RETRIEVE))
    machine.execute(Action.RETRIEVE)
    machine.record_materialized()
    with pytest.raises(StateTransitionError):
        machine.execute(Action.GENERATE)
    machine.record_binding(make_binding())
    machine.execute(Action.BIND)
    machine.execute(Action.GENERATE)
    machine.record_validation(make_validation(ValidationDecision.PASS))
    machine.release()
    assert machine.state is State.RELEASED
    assert machine.can_generate is False


def test_calculation_requires_binding_and_preserves_result_contract() -> None:
    machine = StateMachine()
    machine.accept_plan(SupervisorPlan(Intent.CALCULATION, (make_slot("current"), make_slot("prior", period="FY2024")), "growth_rate", Action.RETRIEVE))
    machine.execute(Action.RETRIEVE)
    machine.record_materialized()
    machine.record_binding(EvidenceBinding(BindingStatus.BOUND, {"current": ("fact_1",), "prior": ("fact_2",)}))
    machine.execute(Action.BIND)
    machine.execute(Action.CALCULATE)
    result = CalculationResultPacket(CalculationStatus.EXECUTED, "growth_rate", "0.10", "FY2025", "percentage", "1", None, ("source_1", "source_2"), "growth_rate.v1")
    machine.record_calculation(result)
    machine.execute(Action.GENERATE)
    assert machine.state is State.GENERATED
    assert machine.calculation_result == result


def test_repair_budgets_fail_closed() -> None:
    machine = StateMachine(RepairBudget(retrieval_repair_max=1, generation_repair_max=1, total_tool_steps_max=8))
    machine.accept_plan(SupervisorPlan(Intent.DIRECT_FACT, (make_slot(),), None, Action.RETRIEVE))
    machine.execute(Action.RETRIEVE)
    machine.record_binding(EvidenceBinding(BindingStatus.MISSING, {}, missing_slots=("slot_1",)))
    machine.begin_repair("retrieval")
    machine.execute(Action.REPAIR_RETRIEVAL)
    machine.record_binding(EvidenceBinding(BindingStatus.MISSING, {}, missing_slots=("slot_1",)))
    machine.begin_repair("retrieval")
    assert machine.state is State.ABSTAINED
    assert machine.retrieval_repairs == 1
