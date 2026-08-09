"""Pure hierarchical field-family rank fusion for Gate 08 R7."""

from __future__ import annotations

from typing import Any

RRF_K = 60
FIELD_LANES = (
    "structured_metric_bm25",
    "structured_axis_bm25",
    "structured_context_bm25",
    "structured_evidence_bm25",
)
GENERAL_LANES = ("candidate_structured_bm25", "candidate_structured_dense")


def _rank_map(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(item["candidate_key"]): int(item.get("rank") or position)
        for position, item in enumerate(items, 1)
        if item.get("candidate_key")
    }


def fuse_ranked_lanes(
    lanes: dict[str, list[dict[str, Any]]], *, rrf_k: int = RRF_K
) -> list[dict[str, Any]]:
    if rrf_k != RRF_K:
        raise ValueError("rrf_k_must_equal_60")
    rank_maps = {name: _rank_map(items) for name, items in lanes.items() if items}
    scores: dict[str, float] = {}
    support: dict[str, dict[str, int]] = {}
    for lane, ranks in rank_maps.items():
        for key, rank in ranks.items():
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            support.setdefault(key, {})[lane] = rank
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [
        {
            "candidate_key": key,
            "rank": rank,
            "rrf_score": scores[key],
            "lane_ranks": support[key],
        }
        for rank, key in enumerate(ordered, 1)
    ]


def fuse_field_family(
    lane_hits: dict[str, list[dict[str, Any]]], *, rrf_k: int = RRF_K
) -> list[dict[str, Any]]:
    return fuse_ranked_lanes(
        {lane: lane_hits.get(lane, []) for lane in FIELD_LANES if lane_hits.get(lane)},
        rrf_k=rrf_k,
    )


def fuse_flat_h0(
    lane_hits: dict[str, list[dict[str, Any]]], *, rrf_k: int = RRF_K
) -> list[dict[str, Any]]:
    return fuse_ranked_lanes(
        {
            lane: lane_hits.get(lane, [])
            for lane in (*GENERAL_LANES, *FIELD_LANES)
            if lane_hits.get(lane)
        },
        rrf_k=rrf_k,
    )


def fuse_hierarchical_structured(
    lane_hits: dict[str, list[dict[str, Any]]], *, rrf_k: int = RRF_K
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    field_family = fuse_field_family(lane_hits, rrf_k=rrf_k)
    top = {
        lane: lane_hits.get(lane, [])
        for lane in GENERAL_LANES
        if lane_hits.get(lane)
    }
    if field_family:
        top["structured_field_family"] = field_family
    return field_family, fuse_ranked_lanes(top, rrf_k=rrf_k)
