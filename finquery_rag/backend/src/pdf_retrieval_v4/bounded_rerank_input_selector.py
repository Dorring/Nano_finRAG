"""Rank-provenance-preserving bounded Top100 rerank input selection."""

from __future__ import annotations

from typing import Any

RERANK_INPUT_BUDGET = 100
SLOT_COMPOSITION_HORIZON = 100
SLOT_MIN_BUDGET = 10
INF_RANK = 10**9


def build_priority_ranking(
    raw_family: list[dict[str, Any]], structured_family: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    raw = {item["candidate_key"]: item for item in raw_family}
    structured = {item["candidate_key"]: item for item in structured_family}
    records = []
    for candidate_key in set(raw) | set(structured):
        raw_item = raw.get(candidate_key, {})
        structured_item = structured.get(candidate_key, {})
        raw_priority = raw_item.get("best_rank") or raw_item.get("rank")
        structured_priority = structured_item.get("best_rank") or structured_item.get("rank")
        priorities = sorted(value for value in (raw_priority, structured_priority) if value is not None)
        records.append(
            {
                "candidate_key": candidate_key,
                "top_priority_rank": priorities[0],
                "second_priority_rank": priorities[1] if len(priorities) > 1 else None,
                "raw_priority_rank": raw_priority,
                "structured_priority_rank": structured_priority,
            }
        )
    records.sort(
        key=lambda item: (
            item["top_priority_rank"],
            item["second_priority_rank"] if item["second_priority_rank"] is not None else INF_RANK,
            item["candidate_key"],
        )
    )
    return [{**item, "rank": rank} for rank, item in enumerate(records, 1)]


def select_single_slot_top100(
    priority_ranking: list[dict[str, Any]], *, budget: int = RERANK_INPUT_BUDGET
) -> list[dict[str, Any]]:
    if budget != RERANK_INPUT_BUDGET:
        raise ValueError("rerank_input_budget_must_equal_100")
    return [
        {**item, "final_candidate_rank": rank, "supporting_slots": [], "minimum_coverage_selected": False}
        for rank, item in enumerate(priority_ranking[:budget], 1)
    ]


def _residual_priority(
    main_ranking: list[dict[str, Any]],
    slot_rankings: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    ranks: dict[str, dict[str, int]] = {}
    lanes = {"main": main_ranking[:SLOT_COMPOSITION_HORIZON]}
    lanes.update(
        {
            f"slot:{slot_id}": items[:SLOT_COMPOSITION_HORIZON]
            for slot_id, items in slot_rankings.items()
        }
    )
    for lane, items in lanes.items():
        for position, item in enumerate(items, 1):
            ranks.setdefault(item["candidate_key"], {})[lane] = int(item.get("rank") or position)
    records = []
    for candidate_key, lane_ranks in ranks.items():
        ordered = sorted(lane_ranks.values())
        records.append(
            {
                "candidate_key": candidate_key,
                "residual_best_rank": ordered[0],
                "residual_second_rank": ordered[1] if len(ordered) > 1 else None,
                "residual_lane_ranks": lane_ranks,
            }
        )
    records.sort(
        key=lambda item: (
            item["residual_best_rank"],
            item["residual_second_rank"] if item["residual_second_rank"] is not None else INF_RANK,
            item["candidate_key"],
        )
    )
    return records


def select_multi_slot_top100(
    slot_rankings: dict[str, list[dict[str, Any]]],
    main_ranking: list[dict[str, Any]],
    *,
    budget: int = RERANK_INPUT_BUDGET,
    slot_horizon: int = SLOT_COMPOSITION_HORIZON,
    slot_min_budget: int = SLOT_MIN_BUDGET,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if budget != RERANK_INPUT_BUDGET:
        raise ValueError("rerank_input_budget_must_equal_100")
    if slot_horizon != SLOT_COMPOSITION_HORIZON:
        raise ValueError("slot_composition_horizon_must_equal_100")
    if slot_min_budget != SLOT_MIN_BUDGET:
        raise ValueError("slot_min_budget_must_equal_10")
    slot_order = list(slot_rankings)
    support: dict[str, dict[str, int]] = {}
    for slot_id, items in slot_rankings.items():
        for position, item in enumerate(items[:slot_horizon], 1):
            support.setdefault(item["candidate_key"], {})[slot_id] = int(item.get("rank") or position)
    selected: list[str] = []
    selected_set: set[str] = set()
    minimum_selected: set[str] = set()
    coverage = {slot_id: 0 for slot_id in slot_order}
    while any(value < slot_min_budget for value in coverage.values()):
        progressed = False
        for slot_id in slot_order:
            if coverage[slot_id] >= slot_min_budget:
                continue
            candidate_key = next(
                (
                    item["candidate_key"]
                    for item in slot_rankings[slot_id][:slot_horizon]
                    if item["candidate_key"] not in selected_set
                ),
                None,
            )
            if candidate_key is None or len(selected) >= budget:
                continue
            selected.append(candidate_key)
            selected_set.add(candidate_key)
            minimum_selected.add(candidate_key)
            for supported_slot in support[candidate_key]:
                coverage[supported_slot] += 1
            progressed = True
        if not progressed or len(selected) >= budget:
            break
    residual = _residual_priority(main_ranking, slot_rankings)
    residual_by_key = {item["candidate_key"]: item for item in residual}
    for item in residual:
        candidate_key = item["candidate_key"]
        if candidate_key not in selected_set:
            selected.append(candidate_key)
            selected_set.add(candidate_key)
        if len(selected) >= budget:
            break
    main_by_key = {item["candidate_key"]: item for item in main_ranking}
    result = []
    for final_rank, candidate_key in enumerate(selected, 1):
        main_item = main_by_key.get(candidate_key, {})
        residual_item = residual_by_key.get(candidate_key, {})
        result.append(
            {
                "candidate_key": candidate_key,
                "final_candidate_rank": final_rank,
                "top_priority_rank": main_item.get("top_priority_rank"),
                "second_priority_rank": main_item.get("second_priority_rank"),
                "raw_priority_rank": main_item.get("raw_priority_rank"),
                "structured_priority_rank": main_item.get("structured_priority_rank"),
                "slot_priority_ranks": support.get(candidate_key, {}),
                "supporting_slots": [slot for slot in slot_order if slot in support.get(candidate_key, {})],
                "minimum_coverage_selected": candidate_key in minimum_selected,
                "residual_best_rank": residual_item.get("residual_best_rank"),
                "residual_second_rank": residual_item.get("residual_second_rank"),
                "residual_lane_ranks": residual_item.get("residual_lane_ranks", {}),
            }
        )
    return result, {
        "slot_order": slot_order,
        "slot_coverage": coverage,
        "minimum_coverage_available": all(value >= slot_min_budget for value in coverage.values()),
        "residual_union_size": len(residual),
        "selected_count": len(result),
    }
