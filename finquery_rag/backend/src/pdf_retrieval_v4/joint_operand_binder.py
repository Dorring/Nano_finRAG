"""Joint semantic operand assignment with structural coherence."""

from __future__ import annotations

from itertools import product
from typing import Any

from .metric_binding_contract_v2 import normalize
from .operation_unit_contract import evaluate_operation_units


def _lineage(semantic_class: dict[str, Any], field: str) -> set[str]:
    return {
        str(item[field])
        for item in semantic_class.get("physical_provenance") or []
        if item.get(field)
    }


def _coherent(assignment: tuple[dict[str, Any], ...], field: str) -> bool:
    sets = [_lineage(item, field) for item in assignment]
    return bool(sets) and all(sets) and bool(set.intersection(*sets))


def _operand_key(semantic_class: dict[str, Any]) -> tuple[str, ...]:
    """Return runtime operand semantics, excluding physical source identity."""
    unit = semantic_class.get("unit_context") or {}
    return (
        normalize(semantic_class.get("document_id")),
        normalize(semantic_class.get("metric")),
        normalize(semantic_class.get("period")),
        normalize(semantic_class.get("segment")),
        normalize(semantic_class.get("bucket")),
        normalize(semantic_class.get("value")),
        normalize(semantic_class.get("measurement_kind")),
        normalize(unit.get("scale")),
        normalize(unit.get("currency")),
    )


def _merge_equivalent_slot(classes: list[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(min(classes, key=lambda item: item["semantic_fact_id"]))
    merged["equivalent_semantic_fact_ids"] = sorted(
        {item["semantic_fact_id"] for item in classes}
    )
    for field in ("supporting_candidate_keys", "supporting_evidence_ids"):
        merged[field] = sorted({value for item in classes for value in item.get(field) or []})
    provenance = {
        tuple(sorted(item.items()))
        for semantic_class in classes
        for item in semantic_class.get("physical_provenance") or []
    }
    merged["physical_provenance"] = [dict(item) for item in sorted(provenance)]
    return merged


def bind_joint_operands(
    plan: dict[str, Any],
    slot_options: list[dict[str, Any]],
) -> dict[str, Any]:
    if any(not item["compatible_classes"] for item in slot_options):
        return {
            "binding_status": "undercovered",
            "assignment_count": 0,
            "selected_assignment": None,
            "same_row_filter_applied": False,
            "same_table_filter_applied": False,
        }
    assignments = list(product(*(item["compatible_classes"] for item in slot_options)))
    assignments = [item for item in assignments if len({value["document_id"] for value in item}) == 1]
    constraints = plan.get("constraints") or {}
    raw_metrics = {str(item["slot"].get("raw_metric_phrase") or "").casefold() for item in slot_options}
    require_row_preference = len(raw_metrics) == 1 or bool(constraints.get("prefer_same_row"))
    same_row_assignments = [item for item in assignments if _coherent(item, "row_id")]
    same_row_filter = bool(require_row_preference and same_row_assignments)
    if same_row_filter:
        assignments = same_row_assignments
    same_table_assignments = [item for item in assignments if _coherent(item, "table_fragment_id")]
    same_table_filter = bool(constraints.get("prefer_same_logical_table") and same_table_assignments)
    if same_table_filter:
        assignments = same_table_assignments
    by_tuple: dict[tuple[tuple[str, ...], ...], list[tuple[dict[str, Any], ...]]] = {}
    for assignment in assignments:
        key = tuple(_operand_key(item) for item in assignment)
        by_tuple.setdefault(key, []).append(assignment)
    if not by_tuple:
        status, selected = "undercovered", None
    elif len(by_tuple) == 1:
        status = "deterministic"
        equivalent_assignments = next(iter(by_tuple.values()))
        selected = tuple(
            _merge_equivalent_slot([assignment[index] for assignment in equivalent_assignments])
            for index in range(len(slot_options))
        )
    else:
        status, selected = "runtime_operand_ambiguity", None
    if selected:
        if str(plan.get("task_type") or "") == "calculation_multi_operand":
            unit_result = evaluate_operation_units(
                plan.get("operation"),
                list(selected),
                same_row=_coherent(selected, "row_id"),
                same_table=_coherent(selected, "table_fragment_id"),
            )
        else:
            unit_result = {
                "ready": True,
                "reason": None,
                "normalized_values": [str(item.get("value") or "") for item in selected],
                "scale_contract": "not_applicable",
                "currency_contract": "not_applicable",
            }
        final_status = "deterministic_ready" if unit_result["ready"] else "deterministic_unit_blocked"
        selected_assignment = {
            "semantic_fact_ids": [item["semantic_fact_id"] for item in selected],
            "equivalent_semantic_fact_ids": [
                item.get("equivalent_semantic_fact_ids") or [item["semantic_fact_id"]]
                for item in selected
            ],
            "same_row": _coherent(selected, "row_id"),
            "same_table": _coherent(selected, "table_fragment_id"),
            "unit_contract": unit_result,
        }
    else:
        final_status, selected_assignment = status, None
    return {
        "binding_status": final_status,
        "assignment_count": len(by_tuple),
        "physical_assignment_count": len(assignments),
        "selected_assignment": selected_assignment,
        "same_row_filter_applied": same_row_filter,
        "same_table_filter_applied": same_table_filter,
        "rank_used_to_resolve_ambiguity": False,
    }
