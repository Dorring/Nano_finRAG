from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any

from .evidence_slot_matcher_v2 import match_slot

SCHEMA = "pdf-retrieval-v4/evidence-set/v2"
GRADE = {
    "M0_exact_leaf": 4,
    "M1_exact_path": 3,
    "M2_path_segment": 2,
    "M3_concept": 1,
    "raw_fallback": 0,
}


def _identity(
    plan_id: str, evidence_ids: tuple[str, ...], mapping: dict[str, str]
) -> str:
    value = [SCHEMA, plan_id, evidence_ids, sorted(mapping.items())]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _intersection(values: list[list[str]]) -> bool:
    nonempty = [set(value) for value in values if value]
    return bool(nonempty) and bool(set.intersection(*nonempty))


def build_sets(plan: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    slots = list(plan.get("operand_slots") or [])
    slot_ids = [slot["slot_id"] for slot in slots]
    matches: dict[str, list[dict[str, Any]]] = {slot_id: [] for slot_id in slot_ids}
    by_evidence: dict[str, dict[str, dict[str, Any]]] = {}
    for item in evidence:
        for slot in slots:
            match = match_slot(slot, item)
            if match:
                matches[slot["slot_id"]].append(match)
                by_evidence.setdefault(item["evidence_id"], {})[slot["slot_id"]] = match
    evidence_ids = sorted(
        by_evidence, key=lambda key: (evidence_by_rank(evidence, key), key)
    )
    candidate_groups: list[tuple[str, ...]] = []
    complete_singletons = [
        key for key in evidence_ids if set(by_evidence[key]) == set(slot_ids)
    ]
    candidate_groups.extend((key,) for key in complete_singletons)
    if not complete_singletons and len(slot_ids) > 1:
        for left, right in combinations(evidence_ids, 2):
            if set(by_evidence[left]) | set(by_evidence[right]) == set(slot_ids):
                candidate_groups.append((left, right))
    if not candidate_groups:
        candidate_groups.extend((key,) for key in evidence_ids)
    records = []
    for group in candidate_groups:
        mapping: dict[str, dict[str, Any]] = {}
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
                        item["candidate_rank"],
                        item["evidence_id"],
                    ),
                )
        complete = len(mapping) == len(slot_ids)
        typed = sum(item["typed"] for item in mapping.values())
        exact = sum(
            item["metric_grade"] in {"M0_exact_leaf", "M1_exact_path"}
            for item in mapping.values()
        )
        row_matrix_cover = (
            len(group) == 1
            and complete
            and next(iter(mapping.values()))["evidence_type"] == "row_matrix"
        )
        rows = [item["row_ids"] for item in mapping.values()]
        tables = [item["table_ids"] for item in mapping.values()]
        ranks = [item["candidate_rank"] for item in mapping.values()] or [10**6]
        candidate_keys = {item["candidate_key"] for item in mapping.values()}
        semantic_tuple = (
            -int(complete),
            -typed,
            -exact,
            -int(row_matrix_cover),
            -int(_intersection(rows)),
            -int(_intersection(tables)),
            len(group),
            len(candidate_keys),
            max(ranks),
            sum(ranks),
        )
        mapping_ids = {
            slot_id: item["evidence_id"] for slot_id, item in mapping.items()
        }
        set_id = _identity(plan["plan_id"], tuple(sorted(group)), mapping_ids)
        records.append(
            {
                "evidence_set_id": set_id,
                "evidence_ids": list(group),
                "slot_mapping": mapping,
                "complete_slot_count": len(mapping),
                "typed_slot_count": typed,
                "exact_match_count": exact,
                "row_matrix_full_cover": row_matrix_cover,
                "same_canonical_row": _intersection(rows),
                "same_logical_table": _intersection(tables),
                "evidence_count": len(group),
                "candidate_count": len(candidate_keys),
                "worst_candidate_rank": max(ranks),
                "sum_candidate_ranks": sum(ranks),
                "semantic_rank_tuple": list(semantic_tuple),
            }
        )
    records.sort(
        key=lambda item: (tuple(item["semantic_rank_tuple"]), item["evidence_set_id"])
    )
    for rank, item in enumerate(records, 1):
        item["rank"] = rank
    if not records:
        return {
            "primary_status": "empty",
            "primary_set_id": None,
            "co_primary_set_ids": [],
            "sets": [],
            "slot_matches": matches,
            "planner_complete": False,
        }
    best = tuple(records[0]["semantic_rank_tuple"])
    co_primary = [
        item for item in records if tuple(item["semantic_rank_tuple"]) == best
    ]
    complete = records[0]["complete_slot_count"] == len(slot_ids)
    if not complete:
        status = "partial"
    elif len(co_primary) == 1:
        status = "unique"
    else:
        status = "ambiguous"
    return {
        "primary_status": status,
        "primary_set_id": co_primary[0]["evidence_set_id"]
        if status == "unique"
        else None,
        "co_primary_set_ids": [item["evidence_set_id"] for item in co_primary]
        if status == "ambiguous"
        else [],
        "sets": records,
        "slot_matches": matches,
        "planner_complete": complete,
    }


def evidence_by_rank(evidence: list[dict[str, Any]], evidence_id: str) -> int:
    return min(
        item["candidate_rank"]
        for item in evidence
        if item["evidence_id"] == evidence_id
    )
