from __future__ import annotations

import pytest

from rag_v2.contracts.evidence import BindingStatus
from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.evidence.binder_service import BinderRequest
from rag_v2.evidence.binding_validator import validate_binding
from rag_v2.evidence.pairwise_binder import (
    PAIRWISE_SYSTEM_PROMPT,
    pairwise_compatibility_to_binding,
    pairwise_provider_request,
    parse_pairwise_compatibility,
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
    return BinderRequest("pairwise-test", "synthetic question", plan, facts)


def _labels(handles: list[str], values: list[str]) -> dict[str, str]:
    return dict(zip(handles, values, strict=True))


def test_pairwise_schema_is_exact_and_shared() -> None:
    request = _request("slot_1", "slot_2")
    payload, handles, schema = pairwise_provider_request(request)

    assert set(payload) == {"question", "intent", "operation", "facts", "tasks"}
    assert set(payload["facts"]) == set(handles)
    tasks = schema["properties"]["tasks"]
    assert tasks["required"] == ["slot_1", "slot_2"]
    assert tasks["additionalProperties"] is False
    candidate_schema = tasks["properties"]["slot_1"]
    assert candidate_schema["required"] == list(handles)
    assert candidate_schema["additionalProperties"] is False
    assert candidate_schema["properties"]["F01"]["enum"] == ["MATCH", "REJECT", "INDETERMINATE"]


def test_reducer_derives_bound_missing_and_ambiguous() -> None:
    request = _request("slot_1")
    _, handles, _ = pairwise_provider_request(request)

    match = parse_pairwise_compatibility(
        {"tasks": {"slot_1": _labels(list(handles), ["MATCH", "REJECT", "REJECT"])}},
        request.plan,
        handles,
    )
    binding, outcomes = pairwise_compatibility_to_binding(match, request, handles)
    assert binding.status == BindingStatus.BOUND.value
    assert outcomes["slot_1"]["F01"] == "MATCH"
    assert validate_binding(binding, request.plan, request.facts).passed

    missing = parse_pairwise_compatibility(
        {"tasks": {"slot_1": _labels(list(handles), ["REJECT", "REJECT", "REJECT"])}},
        request.plan,
        handles,
    )
    missing_binding, _ = pairwise_compatibility_to_binding(missing, request, handles)
    assert missing_binding.status == BindingStatus.MISSING.value
    assert validate_binding(missing_binding, request.plan, request.facts).passed

    ambiguous = parse_pairwise_compatibility(
        {"tasks": {"slot_1": _labels(list(handles), ["REJECT", "INDETERMINATE", "REJECT"])}},
        request.plan,
        handles,
    )
    ambiguous_binding, _ = pairwise_compatibility_to_binding(ambiguous, request, handles)
    assert ambiguous_binding.status == BindingStatus.AMBIGUOUS.value
    assert validate_binding(ambiguous_binding, request.plan, request.facts).passed

    two_matches = parse_pairwise_compatibility(
        {"tasks": {"slot_1": _labels(list(handles), ["MATCH", "MATCH", "REJECT"])}},
        request.plan,
        handles,
    )
    two_match_binding, _ = pairwise_compatibility_to_binding(two_matches, request, handles)
    assert two_match_binding.status == BindingStatus.AMBIGUOUS.value
    assert validate_binding(two_match_binding, request.plan, request.facts).passed


@pytest.mark.parametrize(
    "payload",
    [
        {"tasks": {"unknown": {"F01": "MATCH", "F02": "REJECT", "F03": "REJECT"}}},
        {"tasks": {"slot_1": {"F01": "MAYBE", "F02": "REJECT", "F03": "REJECT"}}},
        {"tasks": {"slot_1": {"F01": "MATCH", "F02": "REJECT"}}},
    ],
)
def test_pairwise_parser_fails_closed(payload: dict[str, object]) -> None:
    request = _request("slot_1")
    _, handles, _ = pairwise_provider_request(request)
    with pytest.raises(ValueError):
        parse_pairwise_compatibility(payload, request.plan, handles)


def test_pairwise_prompt_is_enum_only_and_answer_free() -> None:
    lowered = PAIRWISE_SYSTEM_PROMPT.casefold()
    assert "match" in lowered
    assert "reject" in lowered
    assert "indeterminate" in lowered
    assert "do not calculate" in lowered
    assert "reasoning" in lowered
    assert "selection" in lowered
