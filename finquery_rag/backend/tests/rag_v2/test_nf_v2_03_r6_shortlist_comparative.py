from __future__ import annotations

import pytest

from rag_v2.contracts.evidence import BindingStatus
from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.evidence.binder_service import BinderRequest
from rag_v2.evidence.binding_validator import validate_binding
from rag_v2.evidence.shortlist_comparative_binder import (
    COMPARATIVE_SYSTEM_PROMPT,
    build_shortlists,
    comparative_decisions_to_binding,
    parse_comparative_decisions,
    shortlist_provider_request,
)


def _request() -> BinderRequest:
    plan = SupervisorPlan(
        intent=Intent.DIRECT_FACT,
        required_slots=(RequiredSlot("slot_1", "revenue", "FY2025", "value", "numeric", None),),
        operation=None,
        next_action=Action.RETRIEVE,
    )
    facts = (
        {"fact_id": "fact_good", "candidate_id": "candidate:good", "physical_source_id": "source:good", "raw_metric": "revenue", "normalized_metric": "revenue", "raw_period": "FY2025", "normalized_period": "FY2025", "raw_value": "100", "parsed_numeric_value": "100", "provenance_complete": True, "row_label": "Revenue", "statement_title": "Operations"},
        {"fact_id": "fact_old", "candidate_id": "candidate:old", "physical_source_id": "source:old", "raw_metric": "revenue", "normalized_metric": "revenue", "raw_period": "FY2024", "normalized_period": "FY2024", "raw_value": "90", "parsed_numeric_value": "90", "provenance_complete": True},
        {"fact_id": "fact_bad", "candidate_id": "candidate:bad", "physical_source_id": "source:bad", "raw_metric": "expenses", "normalized_metric": "expenses", "raw_period": "FY2025", "normalized_period": "FY2025", "raw_value": "10", "parsed_numeric_value": "10", "provenance_complete": True},
    )
    return BinderRequest("shortlist-test", "synthetic question", plan, facts)


def test_shortlist_rejects_only_explicit_period_and_is_bounded() -> None:
    shortlists, _, _ = build_shortlists(_request())
    shortlist = shortlists["slot_1"]
    assert len(shortlist.candidates) <= 5
    assert any(item["handle"] == "F02" and item["reason"] == "explicit_period_conflict" for item in shortlist.hard_rejected)
    assert [item["handle"] for item in shortlist.candidates] == ["F01", "F03"]


def test_comparative_schema_and_reducer_are_deterministic() -> None:
    request = _request()
    payload, shortlists, handles, schema = shortlist_provider_request(request)
    assert set(payload) == {"question", "intent", "operation", "facts", "tasks"}
    assert schema["additionalProperties"] is False
    task = schema["properties"]["tasks"]["properties"]["slot_1"]
    assert task["required"] == ["decision", "selected_handle"]
    assert task["properties"]["decision"]["enum"] == ["SELECT", "NONE", "AMBIGUOUS"]
    assert task["properties"]["selected_handle"]["enum"] == ["F01", "F03", None]

    select = parse_comparative_decisions({"tasks": {"slot_1": {"decision": "SELECT", "selected_handle": "F01"}}}, request.plan, shortlists)
    binding, outcomes = comparative_decisions_to_binding(select, request, shortlists, handles)
    assert outcomes["slot_1"]["decision"] == "SELECT"
    assert binding.status == BindingStatus.BOUND.value
    assert validate_binding(binding, request.plan, request.facts).passed

    none = parse_comparative_decisions({"tasks": {"slot_1": {"decision": "NONE", "selected_handle": None}}}, request.plan, shortlists)
    none_binding, _ = comparative_decisions_to_binding(none, request, shortlists, handles)
    assert none_binding.status == BindingStatus.MISSING.value
    assert validate_binding(none_binding, request.plan, request.facts).passed

    ambiguous = parse_comparative_decisions({"tasks": {"slot_1": {"decision": "AMBIGUOUS", "selected_handle": None}}}, request.plan, shortlists)
    ambiguous_binding, _ = comparative_decisions_to_binding(ambiguous, request, shortlists, handles)
    assert ambiguous_binding.status == BindingStatus.AMBIGUOUS.value
    assert validate_binding(ambiguous_binding, request.plan, request.facts).passed


@pytest.mark.parametrize(
    "payload",
    [
        {"tasks": {"unknown": {"decision": "SELECT", "selected_handle": "F01"}}},
        {"tasks": {"slot_1": {"decision": "MAYBE", "selected_handle": None}}},
        {"tasks": {"slot_1": {"decision": "SELECT", "selected_handle": "F99"}}},
        {"tasks": {"slot_1": {"decision": "NONE", "selected_handle": "F01"}}},
    ],
)
def test_comparative_parser_fails_closed(payload: dict[str, object]) -> None:
    request = _request()
    _, shortlists, _, _ = shortlist_provider_request(request)
    with pytest.raises(ValueError):
        parse_comparative_decisions(payload, request.plan, shortlists)


def test_prompt_is_comparative_and_fail_closed() -> None:
    lowered = COMPARATIVE_SYSTEM_PROMPT.casefold()
    assert "compare all shortlisted candidates jointly" in lowered
    assert "none" in lowered and "ambiguous" in lowered and "select" in lowered
    assert "do not select a closest lexical match" in lowered
    assert "reasoning" in lowered
