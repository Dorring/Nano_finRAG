from __future__ import annotations

import pytest

from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.evidence.binder_service import BinderRequest
from rag_v2.evidence.binder_fact_view import (
    binder_fact_view_v2_field_provenance,
    build_binder_fact_view,
    build_binder_fact_view_v2,
)
from rag_v2.evidence.binder_selection import (
    DuplicateFactHandleError,
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
    assert slots["properties"]["slot_1"]["type"] == "array"
    assert "uniqueItems" not in slots["properties"]["slot_1"]
    assert slots["properties"]["slot_1"]["items"]["enum"] == list(handles)
    assert "status" not in slots["properties"]["slot_1"]


def test_bound_adapter_and_validator_pass() -> None:
    req = request("slot_1")
    dto = parse_selection({"slots": {"slot_1": ["F01"]}}, req.plan, fact_handle_map(req))
    binding = selection_to_binding(dto, req, fact_handle_map(req))
    result = validate_binding(binding, req.plan, req.facts)
    assert binding.status == "BOUND"
    assert binding.slot_bindings == {"slot_1": ("fact_1",)}
    assert result.passed


def test_missing_adapter_and_validator_pass() -> None:
    req = request("slot_1")
    dto = parse_selection({"slots": {"slot_1": []}}, req.plan, fact_handle_map(req))
    binding = selection_to_binding(dto, req, fact_handle_map(req))
    assert binding.status == "MISSING"
    assert validate_binding(binding, req.plan, req.facts).passed


def test_ambiguous_adapter_and_validator_pass() -> None:
    req = request("slot_1")
    dto = parse_selection({"slots": {"slot_1": ["F01", "F02"]}}, req.plan, fact_handle_map(req))
    binding = selection_to_binding(dto, req, fact_handle_map(req))
    assert binding.status == "AMBIGUOUS"
    assert validate_binding(binding, req.plan, req.facts).passed


@pytest.mark.parametrize(
    "payload",
    [
        {"slots": {"wrong": ["F01"]}},
        {"slots": {"slot_1": ["F99"]}},
        {"slots": {"slot_1": {"fact_handles": ["F01"]}}},
        {"slots": {"slot_1": "F01"}},
    ],
)
def test_invalid_provider_shape_fails_closed(payload: dict[str, object]) -> None:
    req = request("slot_1")
    with pytest.raises(ValueError):
        parse_selection(payload, req.plan, fact_handle_map(req))


def test_duplicate_handles_parse_but_adapter_fails_closed() -> None:
    req = request("slot_1")
    dto = parse_selection({"slots": {"slot_1": ["F01", "F01"]}}, req.plan, fact_handle_map(req))
    with pytest.raises(DuplicateFactHandleError, match="duplicate_fact_handle"):
        selection_to_binding(dto, req, fact_handle_map(req))


def test_provider_prompt_has_no_answer_or_calculation_fields() -> None:
    req = request("slot_1")
    payload, _, _ = provider_request(req)
    messages = build_selection_messages(req, payload)
    assert "BinderSelectionDTOv1" in messages[0]["content"]
    assert "overall query status" in messages[0]["content"]
    assert "calculation result" in messages[0]["content"]


def test_fact_view_preserves_source_context_without_new_metric_label() -> None:
    view = build_binder_fact_view(
        {"fact_id": "secret-internal-id", "raw_metric": "Revenue", "normalized_metric": "revenue", "raw_period": "FY2025", "parsed_numeric_value": "100", "provenance_complete": True},
        "F01",
        {"row_label": "Data Center", "column_header": ["FY2025"], "table_title": "Revenue by segment"},
    )
    assert view["fact_handle"] == "F01"
    assert view["row_label"] == "Data Center"
    assert view["column_header"] == ["FY2025"]
    assert "fact_id" not in view
    assert "canonical_metric" not in view


def test_fact_view_v2_exposes_only_linked_structural_context() -> None:
    source = {
        "candidate_id": "candidate:one",
        "statement_id": "income_statement",
        "table_title": "Revenue by segment",
        "row_label": "Data Center",
        "column_header": ["FY2025", "Revenue"],
        "source_text": "[STRUCTURE]\nStatement: income_statement\nMetric Path: Data Center > Revenue\nColumn Headers: FY2025 | Revenue\n",
        "table_id": "table:one",
        "pdf_page": 7,
    }
    source_fact = fact("fact_1")
    view = build_binder_fact_view_v2(source_fact, source, "F01")
    assert view["fact_handle"] == "F01"
    assert view["statement_title"] == "income_statement"
    assert view["row_path"] == ["Data Center", "Revenue"]
    assert view["column_header_path"] == ["FY2025", "Revenue"]
    assert view["table_title"] == "Revenue by segment"
    assert "fact_id" not in view
    assert "canonical_metric" not in view
    provenance = binder_fact_view_v2_field_provenance(source_fact, source)
    assert provenance["statement_title"]["source_candidate_id"] == "candidate:one"
    assert all(item["source_candidate_id"] == "candidate:one" for item in provenance.values() if item["origin"] == "source_candidate")


def test_fact_view_v2_does_not_accept_question_or_gold() -> None:
    source_fact = fact("fact_1")
    with pytest.raises(TypeError):
        build_binder_fact_view_v2(source_fact, {"candidate_id": "candidate:one"}, "F01", "question")  # type: ignore[call-arg]


def test_v2_provider_request_keeps_exact_handles_and_slots() -> None:
    req = request("slot_1")
    payload, handles, schema = provider_request(req, fact_view_version="v2", source_by_candidate={"candidate:one": {"statement_id": "income_statement", "table_title": "Revenue"}})
    assert [item["fact_handle"] for item in payload["binder_fact_views"]] == ["F01", "F02", "F03"]
    assert payload["binder_fact_views"][0]["statement_title"] == "income_statement"
    assert schema["properties"]["slots"]["required"] == ["slot_1"]
    assert handles == fact_handle_map(req)
