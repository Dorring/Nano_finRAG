from __future__ import annotations

import pytest

from rag_v2.contracts.evidence import BindingStatus
from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.evidence.binder_service import BinderRequest
from rag_v2.evidence.binding_validator import validate_binding
from rag_v2.evidence.slotwise_binder import (
    DuplicateFactHandleError,
    SLOTWISE_SYSTEM_PROMPT,
    parse_slotwise_selection,
    slotwise_provider_request,
    slotwise_selection_to_binding,
)


def _request(*slot_ids: str) -> BinderRequest:
    plan = SupervisorPlan(
        intent=Intent.CALCULATION if len(slot_ids) > 1 else Intent.DIRECT_FACT,
        required_slots=tuple(
            RequiredSlot(slot_id, "revenue", "FY2025", "current", "numeric", None)
            for slot_id in slot_ids
        ),
        operation="growth_rate" if len(slot_ids) > 1 else None,
        next_action=Action.RETRIEVE,
    )
    facts = tuple(
        {
            "fact_id": f"fact_{index}",
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
        for index in range(1, 4)
    )
    return BinderRequest("slotwise-test", "synthetic question", plan, facts)


def test_slotwise_request_has_shared_facts_and_exact_slot_schema() -> None:
    request = _request("current", "prior")
    payload, handles, schema = slotwise_provider_request(request)

    assert set(payload) == {"question", "intent", "operation", "facts", "tasks"}
    assert set(payload["facts"]) == set(handles)
    assert set(payload["tasks"]) == {"current", "prior"}
    assert schema["additionalProperties"] is False
    task_schema = schema["properties"]["tasks"]
    assert task_schema["required"] == ["current", "prior"]
    assert task_schema["additionalProperties"] is False
    # Bailian's strict-schema endpoint rejects uniqueItems on arrays; the
    # deterministic parser still rejects duplicate handles after parsing.
    assert "uniqueItems" not in task_schema["properties"]["current"]
    assert task_schema["properties"]["current"]["items"]["enum"] == list(handles)


def test_slotwise_adapter_derives_status_independently() -> None:
    request = _request("current", "prior")
    _, handles, _ = slotwise_provider_request(request)

    bound = parse_slotwise_selection(
        {"tasks": {"current": ["F01"], "prior": ["F02"]}},
        request.plan,
        handles,
    )
    bound_binding = slotwise_selection_to_binding(bound, request, handles)
    assert bound_binding.status == BindingStatus.BOUND.value
    assert validate_binding(bound_binding, request.plan, request.facts).passed

    missing = parse_slotwise_selection(
        {"tasks": {"current": ["F01"], "prior": []}},
        request.plan,
        handles,
    )
    missing_binding = slotwise_selection_to_binding(missing, request, handles)
    assert missing_binding.status == BindingStatus.MISSING.value
    assert validate_binding(missing_binding, request.plan, request.facts).passed

    ambiguous = parse_slotwise_selection(
        {"tasks": {"current": ["F01", "F02"], "prior": ["F03"]}},
        request.plan,
        handles,
    )
    ambiguous_binding = slotwise_selection_to_binding(ambiguous, request, handles)
    assert ambiguous_binding.status == BindingStatus.AMBIGUOUS.value
    assert validate_binding(ambiguous_binding, request.plan, request.facts).passed


@pytest.mark.parametrize(
    "payload",
    [
        {"tasks": {"unknown": ["F01"]}},
        {"tasks": {"current": ["F99"], "prior": []}},
        {"tasks": {"current": ["F01", "F01"], "prior": []}},
        {"tasks": {"current": {"handles": ["F01"]}, "prior": []}},
    ],
)
def test_slotwise_schema_rejects_unknown_or_invalid_selection(payload: dict[str, object]) -> None:
    request = _request("current", "prior")
    _, handles, _ = slotwise_provider_request(request)
    with pytest.raises((ValueError, DuplicateFactHandleError)):
        parse_slotwise_selection(payload, request.plan, handles)


def test_slotwise_prompt_is_independent_and_answer_free() -> None:
    lowered = SLOTWISE_SYSTEM_PROMPT.casefold()
    assert "independent evidence-selection problem" in lowered
    assert "do not globally allocate facts" in lowered
    assert "do not calculate" in lowered
    assert "do not" in lowered and "answer" in lowered
    assert "gold" not in lowered
