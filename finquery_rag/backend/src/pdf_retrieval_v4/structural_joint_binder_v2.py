"""B2 structural joint operand binding with frozen M0/M1 candidates."""

from __future__ import annotations

import re
import unicodedata
from itertools import product
from typing import Any

from .metric_binding_contract_v2 import normalize
from .operation_unit_contract import evaluate_operation_units

ROW_MORPHOLOGY = {
    "assets": "asset",
    "operations": "operation",
    "revenues": "revenue",
    "sales": "sale",
    "services": "service",
}


def canonical_row_label(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    tokens = re.sub(r"[^a-z0-9]+", " ", text).split()
    return " ".join(ROW_MORPHOLOGY.get(token, token) for token in tokens)


def hydrate_structural_provenance(
    semantic_class: dict[str, Any],
    fragment_to_logical: dict[str, str],
    row_labels: dict[str, str],
) -> dict[str, Any]:
    hydrated = dict(semantic_class)
    provenance: list[dict[str, Any]] = []
    for source in semantic_class.get("physical_provenance") or []:
        item = dict(source)
        fragment_id = str(item.get("table_fragment_id") or "")
        row_id = str(item.get("row_id") or "")
        logical_id = fragment_to_logical.get(fragment_id)
        raw_label = row_labels.get(row_id)
        normalized_label = canonical_row_label(raw_label)
        item.update(
            {
                "logical_table_id": logical_id,
                "logical_table_status": "resolved" if logical_id else "unresolved",
                "raw_row_label": raw_label,
                "canonical_row_label": normalized_label or None,
                "canonical_row_identity": [logical_id, normalized_label]
                if logical_id and normalized_label
                else None,
            }
        )
        provenance.append(item)
    hydrated["physical_provenance"] = provenance
    return hydrated


def _lineage(semantic_class: dict[str, Any], field: str) -> set[str]:
    return {
        str(item[field])
        for item in semantic_class.get("physical_provenance") or []
        if item.get(field)
    }


def _row_identities(semantic_class: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        tuple(item["canonical_row_identity"])
        for item in semantic_class.get("physical_provenance") or []
        if item.get("canonical_row_identity")
    }


def _coherent(assignment: tuple[dict[str, Any], ...], field: str) -> bool:
    values = [_lineage(item, field) for item in assignment]
    return bool(values) and all(values) and bool(set.intersection(*values))


def _canonical_row_coherent(assignment: tuple[dict[str, Any], ...]) -> bool:
    values = [_row_identities(item) for item in assignment]
    return bool(values) and all(values) and bool(set.intersection(*values))


def _operand_key(semantic_class: dict[str, Any]) -> tuple[str, ...]:
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


def _tuple_count(assignments: list[tuple[dict[str, Any], ...]]) -> int:
    return len({tuple(_operand_key(item) for item in assignment) for assignment in assignments})


def _merge_equivalent_slot(classes: list[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(min(classes, key=lambda item: item["semantic_fact_id"]))
    merged["equivalent_semantic_fact_ids"] = sorted(
        {item["semantic_fact_id"] for item in classes}
    )
    for field in ("supporting_candidate_keys", "supporting_evidence_ids"):
        merged[field] = sorted({value for item in classes for value in item.get(field) or []})
    provenance_by_payload = {
        tuple(sorted((key, str(value)) for key, value in source.items())): source
        for item in classes
        for source in item.get("physical_provenance") or []
    }
    merged["physical_provenance"] = [
        provenance_by_payload[key] for key in sorted(provenance_by_payload)
    ]
    return merged


def bind_structural_operands_b2(
    plan: dict[str, Any],
    slot_options: list[dict[str, Any]],
) -> dict[str, Any]:
    empty_lineage = {
        "before": {"physical_assignments": 0, "operand_tuples": 0},
        "after_same_canonical_row": {"physical_assignments": 0, "operand_tuples": 0},
        "after_same_logical_table": {"physical_assignments": 0, "operand_tuples": 0},
        "after_semantic_tuple_collapse": {"operand_tuples": 0},
    }
    if any(not item["compatible_classes"] for item in slot_options):
        return {
            "binding_status": "undercovered",
            "assignment_count": 0,
            "physical_assignment_count": 0,
            "selected_assignment": None,
            "canonical_row_filter_applied": False,
            "logical_table_filter_applied": False,
            "rank_used_to_resolve_ambiguity": False,
            "assignment_lineage": empty_lineage,
        }
    assignments = [
        assignment
        for assignment in product(*(item["compatible_classes"] for item in slot_options))
        if len({item["document_id"] for item in assignment}) == 1
    ]
    before = list(assignments)
    constraints = plan.get("constraints") or {}
    raw_metrics = {normalize(item["slot"].get("raw_metric_phrase")) for item in slot_options}
    same_metric_multi_period = len(raw_metrics) == 1 and len(slot_options) > 1
    row_requested = same_metric_multi_period or bool(constraints.get("prefer_same_row"))
    coherent_rows = [assignment for assignment in assignments if _canonical_row_coherent(assignment)]
    canonical_row_filter = bool(row_requested and coherent_rows)
    if canonical_row_filter:
        assignments = coherent_rows
    after_row = list(assignments)
    table_requested = same_metric_multi_period or bool(
        constraints.get("prefer_same_logical_table")
    )
    coherent_tables = [
        assignment for assignment in assignments if _coherent(assignment, "logical_table_id")
    ]
    logical_table_filter = bool(table_requested and coherent_tables)
    if logical_table_filter:
        assignments = coherent_tables
    after_table = list(assignments)
    by_tuple: dict[tuple[tuple[str, ...], ...], list[tuple[dict[str, Any], ...]]] = {}
    for assignment in assignments:
        by_tuple.setdefault(tuple(_operand_key(item) for item in assignment), []).append(assignment)
    if not by_tuple:
        final_status, selected = "undercovered", None
    elif len(by_tuple) == 1:
        equivalent_assignments = next(iter(by_tuple.values()))
        selected = tuple(
            _merge_equivalent_slot([assignment[index] for assignment in equivalent_assignments])
            for index in range(len(slot_options))
        )
        if plan.get("task_type") == "calculation_multi_operand":
            unit_result = evaluate_operation_units(
                plan.get("operation"),
                list(selected),
                # Unit compatibility remains byte-for-byte R5.1 semantics.
                # Canonical row/logical table affect assignment filtering only.
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
        final_status = (
            "deterministic_ready" if unit_result["ready"] else "deterministic_unit_blocked"
        )
    else:
        final_status, selected = "runtime_operand_ambiguity", None
        unit_result = None
    selected_assignment = (
        {
            "semantic_fact_ids": [item["semantic_fact_id"] for item in selected],
            "equivalent_semantic_fact_ids": [
                item.get("equivalent_semantic_fact_ids") or [item["semantic_fact_id"]]
                for item in selected
            ],
            "same_canonical_row": _canonical_row_coherent(selected),
            "same_logical_table": _coherent(selected, "logical_table_id"),
            "same_table_fragment": _coherent(selected, "table_fragment_id"),
            "unit_contract": unit_result,
            "supporting_candidate_keys": sorted(
                {key for item in selected for key in item.get("supporting_candidate_keys") or []}
            ),
            "supporting_evidence_ids": sorted(
                {key for item in selected for key in item.get("supporting_evidence_ids") or []}
            ),
            "physical_provenance": [
                source
                for item in selected
                for source in item.get("physical_provenance") or []
            ],
        }
        if selected
        else None
    )
    return {
        "binding_status": final_status,
        "assignment_count": len(by_tuple),
        "physical_assignment_count": len(after_table),
        "selected_assignment": selected_assignment,
        "canonical_row_filter_applied": canonical_row_filter,
        "logical_table_filter_applied": logical_table_filter,
        "rank_used_to_resolve_ambiguity": False,
        "assignment_lineage": {
            "before": {
                "physical_assignments": len(before),
                "operand_tuples": _tuple_count(before),
            },
            "after_same_canonical_row": {
                "physical_assignments": len(after_row),
                "operand_tuples": _tuple_count(after_row),
            },
            "after_same_logical_table": {
                "physical_assignments": len(after_table),
                "operand_tuples": _tuple_count(after_table),
            },
            "after_semantic_tuple_collapse": {"operand_tuples": len(by_tuple)},
        },
    }
