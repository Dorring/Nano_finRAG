from __future__ import annotations

import pytest

from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.evidence.binder_service import BinderRequest
from rag_v2.evidence.binder_selection import (
    build_selection_messages,
    fact_handle_map,
    parse_selection,
    provider_request,
    selection_to_binding,
)
from rag_v2.evidence.binding_validator import validate_binding


def plan(*slot_ids: str) -> SupervisorPlan:
    return SupervisorPlan(
        intent=Intent.CALCULATION if len(slot_ids) > 1 else Intent.DIRECT_FACT,
        required_slots=tuple(RequiredSlot(slot_id, "revenue", "FY2025", "current", "numeric", None) for slot_id in slot_ids),
        operation="growth_rate" if len(slot_ids) > 1 else None,
        next_action=Action.RETRIEVE,
    )


def fact(fact_id: str) -> dict[str, object]:
    return {
        "fact_id": fact_id,
        "candidate_id": "candidate:one",
        "physical_source_id": "source:one",
        "raw_metric": "revenue",
        "normalized_metric": "revenue",
        "raw_period": "FY2025",
        "normalized_period": "FY2025",
        "raw_value": "100",
        "parsed_numeric_value": "100",
        "provenance_complete": True,
    }


def request(*slot_ids: str) -> BinderRequest:
    return BinderRequest("synthetic", "synthetic question", plan(*slot_ids), tuple(fact(f"fact_{i}") for i in range(1, 4)))


def test_handles_are_one_to_one_and_deterministic() -> None:
    req = request("slot_1")
    assert fact_handle_map(req) == {"F01": "fact_1", "F02": "fact_2", "F03": "fact_3"}
    assert fact_handle_map(req) == fact_handle_map(req)


def test_dynamic_schema_has_exact_slots_and_handles() -> None:
    req = request("slot_1", "slot_2")
    _, handles, schema = provider_request(req)
    slots = schema["properties"]["slots"]
    assert slots["required"] == ["slot_1", "slot_2"]
    assert slots["additionalProperties"] is False
    assert slots["properties"]["slot_1"]["properties"]["fact_handles"]["items"]["enum"] == list(handles)


def test_bound_adapter_and_validator_pass() -> None:
    req = request("slot_1")
    dto = parse_selection({"slots": {"slot_1": {"status": "BOUND", "fact_handles": ["F01"]}}}, req.plan, fact_handle_map(req))
    binding = selection_to_binding(dto, req, fact_handle_map(req))
    result = validate_binding(binding, req.plan, req.facts)
    assert binding.status == "BOUND"
    assert binding.slot_bindings == {"slot_1": ("fact_1",)}
    assert result.passed


def test_missing_adapter_and_validator_pass() -> None:
    req = request("slot_1")
    dto = parse_selection({"slots": {"slot_1": {"status": "MISSING", "fact_handles": []}}}, req.plan, fact_handle_map(req))
    binding = selection_to_binding(dto, req, fact_handle_map(req))
    assert binding.status == "MISSING"
    assert validate_binding(binding, req.plan, req.facts).passed


def test_ambiguous_adapter_and_validator_pass() -> None:
    req = request("slot_1")
    dto = parse_selection({"slots": {"slot_1": {"status": "AMBIGUOUS", "fact_handles": ["F01", "F02"]}}}, req.plan, fact_handle_map(req))
    binding = selection_to_binding(dto, req, fact_handle_map(req))
    assert binding.status == "AMBIGUOUS"
    assert validate_binding(binding, req.plan, req.facts).passed


@pytest.mark.parametrize(
    "payload",
    [
        {"slots": {"wrong": {"status": "BOUND", "fact_handles": ["F01"]}}},
        {"slots": {"slot_1": {"status": "BOUND", "fact_handles": ["F99"]}}},
        {"slots": {"slot_1": {"status": "BOUND", "fact_handles": []}}},
        {"slots": {"slot_1": {"status": "MISSING", "fact_handles": ["F01"]}}},
        {"slots": {"slot_1": {"status": "AMBIGUOUS", "fact_handles": ["F01"]}}},
    ],
)
def test_invalid_provider_shape_fails_closed(payload: dict[str, object]) -> None:
    req = request("slot_1")
    with pytest.raises(ValueError):
        parse_selection(payload, req.plan, fact_handle_map(req))


def test_provider_prompt_has_no_answer_or_calculation_fields() -> None:
    req = request("slot_1")
    payload, _, _ = provider_request(req)
    messages = build_selection_messages(req, payload)
    assert "BinderSelectionDTOv1" in messages[0]["content"]
    assert "overall query status" in messages[0]["content"]
    assert "calculation result" in messages[0]["content"]
