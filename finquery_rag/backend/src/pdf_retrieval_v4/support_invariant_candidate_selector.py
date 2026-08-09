"""Support-count-invariant family fusion for Gate 08 R8-R1.2."""

from __future__ import annotations

from typing import Any

RRF_K = 60
INF_RANK = 10**9


def _candidate_key(item: dict[str, Any]) -> str:
    return str(item.get("candidate_key") or item.get("original_candidate_identity") or "")


def _source_rank(item: dict[str, Any], position: int) -> int:
    return int(
        item.get("rank")
        or item.get("stage_rank")
        or item.get("structured_rank")
        or item.get("fused_rank")
        or position
    )


def rank_support_invariant_family(
    lanes: dict[str, list[dict[str, Any]]], *, rrf_k: int = RRF_K
) -> list[dict[str, Any]]:
    """Rank by best lane, using the second-best lane only as a tie-break."""
    if rrf_k != RRF_K:
        raise ValueError("rrf_k_must_equal_60")
    lane_ranks: dict[str, dict[str, int]] = {}
    for lane_name, items in lanes.items():
        seen: set[str] = set()
        for position, item in enumerate(items, 1):
            key = _candidate_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            lane_ranks.setdefault(key, {})[lane_name] = _source_rank(item, position)
    ordering: dict[str, tuple[int, int, str]] = {}
    for key, ranks_by_lane in lane_ranks.items():
        ranks = sorted(ranks_by_lane.values())
        ordering[key] = (ranks[0], ranks[1] if len(ranks) > 1 else INF_RANK, key)
    ordered = sorted(ordering, key=ordering.__getitem__)
    result = []
    for rank, key in enumerate(ordered, 1):
        best, second, _ = ordering[key]
        result.append(
            {
                "candidate_key": key,
                "rank": rank,
                "best_rank": best,
                "second_best_rank": None if second == INF_RANK else second,
                "best_rank_score": 1.0 / (rrf_k + best),
                "lane_ranks": lane_ranks[key],
                "support_count": len(lane_ranks[key]),
            }
        )
    return result


def build_raw_family_v2(
    production_raw: list[dict[str, Any]], candidate_raw: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return rank_support_invariant_family(
        {"production_raw": production_raw, "candidate_raw": candidate_raw}
    )


def build_structured_family_v2(
    structured_h1: list[dict[str, Any]],
    metric: list[dict[str, Any]],
    existing_structured: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    lanes = {"structured_h1": structured_h1, "structured_metric": metric}
    if existing_structured is not None:
        lanes["existing_structured"] = existing_structured
    return rank_support_invariant_family(lanes)


def fuse_main_families_v2(
    raw_family: list[dict[str, Any]], structured_family: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return rank_support_invariant_family(
        {"raw_family": raw_family, "structured_family": structured_family}
    )
