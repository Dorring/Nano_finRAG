from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any

from .evidence_slot_matcher_v3 import match_slot

GRADE = {
    "M0_exact_leaf": 4,
    "M1_exact_path": 3,
    "M2_path_segment": 2,
    "M3_concept": 1,
    "raw_fallback": 0,
}


def _id(
    plan_id: str, group: tuple[str, ...], mapping: dict[str, dict[str, Any]]
) -> str:
    value = [
        "pdf-retrieval-v4/evidence-set/v3",
        plan_id,
        group,
        sorted(
            (
                key,
                item["evidence_id"],
                (item.get("matrix_dimension") or {}).get("dimension_identity"),
            )
            for key, item in mapping.items()
        ),
    ]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def build_sets(plan: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    slots = list(plan.get("operand_slots") or [])
    slot_ids = [item["slot_id"] for item in slots]
    by_evidence: dict[str, dict[str, dict[str, Any]]] = {}
    slot_matches = {slot_id: [] for slot_id in slot_ids}
    for item in evidence:
        for slot in slots:
            match = match_slot(plan, slot, item)
            if match:
                by_evidence.setdefault(item["evidence_id"], {})[slot["slot_id"]] = match
                slot_matches[slot["slot_id"]].append(match)
    ids = sorted(
        by_evidence,
        key=lambda key: (
            min(item["candidate_rank"] for item in by_evidence[key].values()),
            key,
        ),
    )
    groups = [(key,) for key in ids if set(by_evidence[key]) == set(slot_ids)]
    if not groups and len(slot_ids) > 1:
        groups = [
            (a, b)
            for a, b in combinations(ids, 2)
            if set(by_evidence[a]) | set(by_evidence[b]) == set(slot_ids)
        ]
    if not groups:
        groups = [(key,) for key in ids]
    records = []
    constraints = plan.get("constraints") or {}
    prefer_row = bool(constraints.get("prefer_same_row"))
    prefer_table = bool(constraints.get("prefer_same_logical_table"))
    for group in groups:
        mapping = {}
        for slot_id in slot_ids:
            options = [
                by_evidence[key][slot_id]
                for key in group
                if slot_id in by_evidence[key]
            ]
            if options:
                mapping[slot_id] = min(
                    options,
                    key=lambda item: (
                        -GRADE[item["metric_grade"]],
                        -item["statement_context_score"],
                        item["candidate_rank"],
                        item["evidence_id"],
                    ),
                )
        complete = len(mapping) == len(slot_ids)
        values = list(mapping.values())
        typed = sum(item["typed"] for item in values)
        exact = sum(
            item["metric_grade"] in {"M0_exact_leaf", "M1_exact_path"}
            for item in values
        )
        dimensions = sum(item["dimension_exact"] for item in values)
        context = sum(item["statement_context_score"] for item in values)
        row_matrix = (
            len(group) == 1
            and complete
            and values
            and values[0]["evidence_type"] == "row_matrix"
        )
        rows = {item.get("row_id") for item in values if item.get("row_id")}
        tables = {
            item.get("table_fragment_id")
            for item in values
            if item.get("table_fragment_id")
        }
        same_row = bool(values) and len(rows) == 1
        same_table = bool(values) and len(tables) == 1
        candidate_keys = {item["candidate_key"] for item in values}
        ranks = [item["candidate_rank"] for item in values] or [10**6]
        semantic_tuple = (
            -int(complete),
            -typed,
            -exact,
            -dimensions,
            -context,
            -int(row_matrix),
            -int(prefer_row and same_row),
            -int(prefer_table and same_table),
            len(group),
            len(candidate_keys),
        )
        rank_tuple = (*semantic_tuple, max(ranks), sum(ranks))
        set_id = _id(plan["plan_id"], group, mapping)
        records.append(
            {
                "evidence_set_id": set_id,
                "evidence_ids": list(group),
                "slot_mapping": mapping,
                "complete_slot_count": len(mapping),
                "typed_slot_count": typed,
                "exact_metric_count": exact,
                "exact_dimension_count": dimensions,
                "statement_context_score": context,
                "row_matrix_full_cover": bool(row_matrix),
                "same_canonical_row": same_row,
                "same_logical_table": same_table,
                "evidence_count": len(group),
                "candidate_count": len(candidate_keys),
                "worst_candidate_rank": max(ranks),
                "sum_candidate_ranks": sum(ranks),
                "semantic_rank_tuple": list(semantic_tuple),
                "rank_tuple": list(rank_tuple),
            }
        )
    records.sort(key=lambda item: (tuple(item["rank_tuple"]), item["evidence_set_id"]))
    [item.update(rank=index) for index, item in enumerate(records, 1)]
    if not records:
        return {
            "primary_status": "empty",
            "primary_set_id": None,
            "co_primary_set_ids": [],
            "sets": [],
            "slot_matches": slot_matches,
            "planner_complete": False,
        }
    best_semantic = tuple(records[0]["semantic_rank_tuple"])
    semantic_peers = [
        item for item in records if tuple(item["semantic_rank_tuple"]) == best_semantic
    ]
    complete = records[0]["complete_slot_count"] == len(slot_ids)
    if not complete:
        status = "partial"
    elif len(semantic_peers) > 1:
        status = "ambiguous"
    else:
        status = "unique"
    return {
        "primary_status": status,
        "primary_set_id": records[0]["evidence_set_id"] if status == "unique" else None,
        "co_primary_set_ids": [item["evidence_set_id"] for item in semantic_peers]
        if status == "ambiguous"
        else [],
        "sets": records,
        "slot_matches": slot_matches,
        "planner_complete": complete,
    }
