from __future__ import annotations

from rag_v2.contracts.evidence import BindingStatus
from rag_v2.contracts.plan import SupervisorPlan
from rag_v2.evidence.binder_provider import BinderCallMetadata, BinderProviderResult, _binding_from_payload
from rag_v2.evidence.binder_service import BinderRequest, SemanticBinderService
from rag_v2.evidence.binding_validator import validate_binding


def plan(intent: str = "DIRECT_FACT") -> SupervisorPlan:
    return SupervisorPlan.from_dict({
        "intent": intent,
        "required_slots": [
            {"slot_id": "slot_1", "metric": "revenue", "period": "FY2025", "role": "value", "value_type": "numeric", "unit": None},
        ],
        "operation": None,
        "next_action": "RETRIEVE",
    })


def fact(fact_id: str = "f1") -> dict:
    return {"fact_id": fact_id, "provenance_complete": True, "candidate_id": "candidate:1"}


class StubProvider:
    provider_name = "stub"
    model_name = "stub-model"
    last_call = None

    def __init__(self, binding):
        self.binding = binding
        self.calls = 0

    def bind(self, request):
        self.calls += 1
        return BinderProviderResult(self.binding, BinderCallMetadata("stub", "stub-model", "evidence_binder", "strong_general_llm", 1.0, True, True))


def binding(status="BOUND", slot_bindings=None, missing=None, ambiguous=None):
    return _binding_from_payload({
        "status": status,
        "slot_bindings": slot_bindings or {},
        "missing_slots": missing or [],
        "ambiguous_slots": ambiguous or [],
        "invalid_reasons": [],
    })


def test_binder_only_uses_frozen_slots_and_packet() -> None:
    provider = StubProvider(binding(slot_bindings={"slot_1": ["f1"]}))
    request = BinderRequest("q", "question", plan(), (fact(),))
    result = SemanticBinderService(provider).bind(request)
    assert provider.calls == 1
    assert result.validation.passed
    assert result.validation.selected_fact_ids == ("f1",)
    assert "financial_facts" in request.to_dict()
    assert request.to_dict()["required_slots"] == [request.plan.required_slots[0].to_dict()]


def test_unknown_fact_and_slot_are_rejected() -> None:
    invalid = binding(slot_bindings={"new_slot": ["invented"]})
    result = validate_binding(invalid, plan(), (fact(),))
    assert not result.passed
    assert result.final_status == BindingStatus.INVALID.value
    assert any("unknown_fact" in reason for reason in result.reasons)
    assert any("unknown_slot" in reason for reason in result.reasons)


def test_missing_ambiguous_and_empty_packet_are_safe() -> None:
    missing = SemanticBinderService(StubProvider(binding("MISSING", missing=["slot_1"]))).bind(BinderRequest("q", "question", plan(), (fact(),)))
    assert missing.binding.status == BindingStatus.MISSING.value
    ambiguous = SemanticBinderService(StubProvider(binding("AMBIGUOUS", ambiguous=["slot_1"]))).bind(BinderRequest("q", "question", plan(), (fact(), fact("f2"))))
    assert ambiguous.binding.status == BindingStatus.AMBIGUOUS.value
    empty_provider = StubProvider(binding(slot_bindings={"slot_1": ["f1"]}))
    empty = SemanticBinderService(empty_provider).bind(BinderRequest("q", "question", plan(), ()))
    assert empty.binding.status == BindingStatus.MISSING.value
    assert empty.skipped_no_fact_supply is True
    assert empty_provider.calls == 0


def test_calculation_roles_are_not_changed_by_binding() -> None:
    calc = SupervisorPlan.from_dict({
        "intent": "CALCULATION",
        "required_slots": [
            {"slot_id": "current", "metric": "revenue", "period": "FY2025", "role": "current", "value_type": "numeric", "unit": None},
            {"slot_id": "prior", "metric": "revenue", "period": "FY2024", "role": "prior", "value_type": "numeric", "unit": None},
        ],
        "operation": "growth_rate",
        "next_action": "RETRIEVE",
    })
    facts = (fact("f-current"), fact("f-prior"))
    result = SemanticBinderService(StubProvider(binding(slot_bindings={"current": ["f-current"], "prior": ["f-prior"]}))).bind(BinderRequest("q", "question", calc, facts))
    assert result.validation.passed
    assert calc.required_slots[0].role == "current"
    assert calc.required_slots[1].role == "prior"


def test_schema_is_strict_and_binder_has_no_calculator_or_retriever() -> None:
    from rag_v2.evidence.prompt import BINDER_RESPONSE_FORMAT, BINDER_SCHEMA

    assert BINDER_SCHEMA["additionalProperties"] is False
    assert BINDER_RESPONSE_FORMAT["json_schema"]["strict"] is True
    assert "value" not in BINDER_SCHEMA["properties"]
    assert "calculation_result" not in BINDER_SCHEMA["properties"]
