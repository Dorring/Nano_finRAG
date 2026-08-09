"""Pure bounded candidate-family selection for Gate 08 R8-R1."""

from __future__ import annotations

from typing import Any

RRF_K = 60
CANDIDATE_BUDGET = 50
SLOT_MIN_BUDGET = 10
SLOT_CANDIDATE_HORIZON = 50


def _key(item: dict[str, Any]) -> str:
    return str(item.get("candidate_key") or item.get("original_candidate_identity") or "")


def _rank(item: dict[str, Any], position: int) -> int:
    return int(
        item.get("rank")
        or item.get("stage_rank")
        or item.get("structured_rank")
        or item.get("fused_rank")
        or position
    )


def fuse_ranked_families(
    lanes: dict[str, list[dict[str, Any]]], *, rrf_k: int = RRF_K
) -> list[dict[str, Any]]:
    """Fuse explicit source-local rankings with deterministic candidate dedup."""
    if rrf_k != RRF_K:
        raise ValueError("rrf_k_must_equal_60")
    scores: dict[str, float] = {}
    support: dict[str, dict[str, int]] = {}
    for lane_name, items in lanes.items():
        seen: set[str] = set()
        for position, item in enumerate(items, 1):
            candidate_key = _key(item)
            if not candidate_key or candidate_key in seen:
                continue
            seen.add(candidate_key)
            source_rank = _rank(item, position)
            scores[candidate_key] = scores.get(candidate_key, 0.0) + 1.0 / (
                rrf_k + source_rank
            )
            support.setdefault(candidate_key, {})[lane_name] = source_rank
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [
        {
            "candidate_key": candidate_key,
            "rank": rank,
            "rrf_score": scores[candidate_key],
            "lane_ranks": support[candidate_key],
        }
        for rank, candidate_key in enumerate(ordered, 1)
    ]


def build_raw_family(
    production_raw: list[dict[str, Any]], candidate_raw: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return fuse_ranked_families(
        {"production_raw": production_raw, "candidate_raw": candidate_raw}
    )


def build_structured_family(
    structured_h1: list[dict[str, Any]],
    metric: list[dict[str, Any]],
    existing_structured: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    lanes = {"structured_h1": structured_h1, "structured_metric": metric}
    if existing_structured is not None:
        lanes["existing_structured"] = existing_structured
    return fuse_ranked_families(lanes)


def select_single_slot_top50(
    raw_family: list[dict[str, Any]], structured_family: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    fused = fuse_ranked_families(
        {"raw_family": raw_family, "structured_family": structured_family}
    )
    return [
        {**item, "final_candidate_rank": rank}
        for rank, item in enumerate(fused[:CANDIDATE_BUDGET], 1)
    ]


def select_multi_slot_top50(
    slot_rankings: dict[str, list[dict[str, Any]]],
    main_ranking: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Protect required slots, then fill from slot and main-query residual RRF."""
    slot_order = list(slot_rankings)
    support: dict[str, dict[str, int]] = {}
    for slot_id, items in slot_rankings.items():
        for position, item in enumerate(items[:SLOT_CANDIDATE_HORIZON], 1):
            candidate_key = _key(item)
            if candidate_key:
                support.setdefault(candidate_key, {})[slot_id] = _rank(item, position)
    selected: list[str] = []
    selected_set: set[str] = set()
    minimum_selected: set[str] = set()
    coverage = {slot_id: 0 for slot_id in slot_order}
    while any(value < SLOT_MIN_BUDGET for value in coverage.values()):
        progressed = False
        for slot_id in slot_order:
            if coverage[slot_id] >= SLOT_MIN_BUDGET:
                continue
            candidate_key = next(
                (
                    _key(item)
                    for item in slot_rankings[slot_id][:SLOT_CANDIDATE_HORIZON]
                    if _key(item) not in selected_set
                ),
                "",
            )
            if not candidate_key or len(selected) >= CANDIDATE_BUDGET:
                continue
            selected.append(candidate_key)
            selected_set.add(candidate_key)
            minimum_selected.add(candidate_key)
            for supported_slot in support.get(candidate_key, {}):
                coverage[supported_slot] += 1
            progressed = True
        if not progressed or len(selected) >= CANDIDATE_BUDGET:
            break
    residual_lanes = {
        f"slot:{slot_id}": items[:SLOT_CANDIDATE_HORIZON]
        for slot_id, items in slot_rankings.items()
    }
    residual_lanes["main"] = main_ranking
    residual = fuse_ranked_families(residual_lanes)
    residual_by_key = {_key(item): item for item in residual}
    for item in residual:
        candidate_key = _key(item)
        if candidate_key not in selected_set:
            selected.append(candidate_key)
            selected_set.add(candidate_key)
        if len(selected) >= CANDIDATE_BUDGET:
            break
    result = []
    for final_rank, candidate_key in enumerate(selected, 1):
        residual_item = residual_by_key.get(candidate_key, {})
        result.append(
            {
                "candidate_key": candidate_key,
                "slot_ranks": support.get(candidate_key, {}),
                "supporting_slots": [
                    slot for slot in slot_order if slot in support.get(candidate_key, {})
                ],
                "minimum_coverage_selected": candidate_key in minimum_selected,
                "residual_rrf_score": residual_item.get("rrf_score", 0.0),
                "residual_lane_ranks": residual_item.get("lane_ranks", {}),
                "final_candidate_rank": final_rank,
            }
        )
    return result, {
        "slot_order": slot_order,
        "slot_coverage": coverage,
        "minimum_coverage_available": all(
            value >= SLOT_MIN_BUDGET for value in coverage.values()
        ),
        "slot_union_size": len(support),
        "residual_union_size": len(residual),
        "selected_count": len(result),
    }
