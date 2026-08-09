"""Coverage-preserving slot composition for Gate 08 R6."""

from __future__ import annotations

from typing import Any

RRF_K = 60
FINAL_POOL_K = 40
SLOT_MIN_BUDGET = 10
SLOT_CANDIDATE_HORIZON = 40


def compose_slot_candidates(
    slot_rankings: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    slot_order = list(slot_rankings)
    support: dict[str, dict[str, int]] = {}
    for slot_id in slot_order:
        for rank, item in enumerate(slot_rankings[slot_id][:SLOT_CANDIDATE_HORIZON], 1):
            key = str(item.get("candidate_key") or "")
            if key:
                support.setdefault(key, {})[slot_id] = rank
    selected: list[str] = []
    selected_set: set[str] = set()
    coverage = {slot_id: 0 for slot_id in slot_order}
    minimum_selected: set[str] = set()
    while any(value < SLOT_MIN_BUDGET for value in coverage.values()):
        progressed = False
        for slot_id in slot_order:
            if coverage[slot_id] >= SLOT_MIN_BUDGET:
                continue
            candidate = next(
                (
                    str(item.get("candidate_key") or "")
                    for item in slot_rankings[slot_id][:SLOT_CANDIDATE_HORIZON]
                    if str(item.get("candidate_key") or "") not in selected_set
                ),
                "",
            )
            if not candidate or len(selected) >= FINAL_POOL_K:
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            minimum_selected.add(candidate)
            for supported_slot in support[candidate]:
                coverage[supported_slot] += 1
            progressed = True
        if not progressed or len(selected) >= FINAL_POOL_K:
            break
    scores = {
        key: sum(1.0 / (RRF_K + rank) for rank in ranks.values())
        for key, ranks in support.items()
    }
    residual = sorted(
        (key for key in support if key not in selected_set),
        key=lambda key: (-scores[key], key),
    )
    for key in residual:
        if len(selected) >= FINAL_POOL_K:
            break
        selected.append(key)
        selected_set.add(key)
    result = [
        {
            "candidate_key": key,
            "slot_ranks": support[key],
            "supporting_slots": [slot for slot in slot_order if slot in support[key]],
            "minimum_coverage_selected": key in minimum_selected,
            "slot_support_rrf_score": scores[key],
            "final_rank": rank,
        }
        for rank, key in enumerate(selected, 1)
    ]
    audit = {
        "slot_order": slot_order,
        "slot_coverage": coverage,
        "minimum_coverage_available": all(value >= SLOT_MIN_BUDGET for value in coverage.values()),
        "union_size": len(support),
        "selected_count": len(result),
    }
    return result, audit
