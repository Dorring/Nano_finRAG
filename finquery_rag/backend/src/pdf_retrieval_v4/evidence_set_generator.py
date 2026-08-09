from __future__ import annotations

import hashlib
import json
from itertools import product
from typing import Any

from .evidence_slot_matcher import match_slot

SCHEMA = "pdf-retrieval-v4/evidence-set/v1"
GRADE = {"A_exact": 3, "B_concept": 2, "C_matrix_cover": 1, "raw_fallback": 0}


def _set_id(plan_id: str, mapping: dict[str, dict[str, Any]]) -> str:
    payload = [
        SCHEMA,
        plan_id,
        sorted((slot, value["evidence_id"]) for slot, value in mapping.items()),
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def generate_evidence_sets(
    plan: dict[str, Any], evidence: list[dict[str, Any]], limit: int = 20
) -> dict[str, Any]:
    slots = list(plan.get("operand_slots") or [])
    matches = {
        slot["slot_id"]: [
            match for item in evidence if (match := match_slot(slot, item))
        ]
        for slot in slots
    }
    for values in matches.values():
        values.sort(
            key=lambda item: (
                -GRADE[item["metric_grade"]],
                not item["typed"],
                item["candidate_rank"],
                item["evidence_id"],
            )
        )
    if not slots:
        status = (
            "raw_fallback_only"
            if any(item["evidence_type"] == "raw_candidate" for item in evidence)
            else "empty"
        )
        return {
            "status": status,
            "slot_matches": matches,
            "sets": [],
            "primary_set_ids": [],
            "planner_complete": False,
        }
    options = [matches[slot["slot_id"]][:limit] or [None] for slot in slots]
    sets = []
    for combination in product(*options):
        mapping = {
            slot["slot_id"]: value
            for slot, value in zip(slots, combination, strict=True)
            if value
        }
        complete = len(mapping)
        typed = sum(bool(value["typed"]) for value in mapping.values())
        exact = sum(value["metric_grade"] == "A_exact" for value in mapping.values())
        candidate_keys = {value["candidate_key"] for value in mapping.values()}
        evidence_ids = {value["evidence_id"] for value in mapping.values()}
        ranks = [value["candidate_rank"] for value in mapping.values()] or [10**6]
        set_id = _set_id(plan["plan_id"], mapping)
        key = (
            -complete,
            -typed,
            -exact,
            len(evidence_ids),
            len(candidate_keys),
            max(ranks),
            sum(ranks),
            set_id,
        )
        sets.append(
            {
                "evidence_set_id": set_id,
                "slot_mapping": mapping,
                "complete_slot_count": complete,
                "typed_slot_count": typed,
                "exact_match_count": exact,
                "evidence_count": len(evidence_ids),
                "candidate_count": len(candidate_keys),
                "worst_candidate_rank": max(ranks),
                "sum_candidate_ranks": sum(ranks),
                "_rank_key": key,
            }
        )
    sets.sort(key=lambda item: item["_rank_key"])
    best_key = sets[0]["_rank_key"][:-1]
    primary = [item for item in sets if item["_rank_key"][:-1] == best_key]
    for rank, item in enumerate(sets[:limit], 1):
        item.pop("_rank_key", None)
        item["rank"] = rank
    complete = primary[0]["complete_slot_count"] == len(slots)
    typed_complete = complete and primary[0]["typed_slot_count"] == len(slots)
    return {
        "status": "complete"
        if typed_complete
        else "partial"
        if primary[0]["complete_slot_count"]
        else "empty",
        "slot_matches": matches,
        "sets": sets[:limit],
        "primary_set_ids": [item["evidence_set_id"] for item in primary],
        "ambiguous_primary": len(primary) > 1,
        "planner_complete": complete,
    }
