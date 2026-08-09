"""Pure lane-preserving candidate-family fusion for Gate 08 R4."""

from __future__ import annotations

from typing import Any

from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit
from src.pdf_retrieval_v4.candidate_slot_pool import build_slot_pool


def _rank_map(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(item["candidate_key"]): int(item.get("rank") or position)
        for position, item in enumerate(items, 1)
        if item.get("candidate_key")
    }


def fuse_candidate_families(
    raw_family: list[dict[str, Any]],
    structured_family: list[dict[str, Any]],
    *,
    protected_structured_k: int,
    total_k: int = 40,
    rrf_k: int = 60,
    slot_id: str | None = None,
) -> list[dict[str, Any]]:
    """Protect Structured prefix, then fill by cross-family residual RRF."""
    raw_ranks = _rank_map(raw_family)
    structured_ranks = _rank_map(structured_family)
    protected: list[str] = []
    seen: set[str] = set()
    for item in structured_family[:protected_structured_k]:
        key = str(item.get("candidate_key") or "")
        if key and key not in seen:
            protected.append(key)
            seen.add(key)

    residual_scores: dict[str, float] = {}
    for key, rank in raw_ranks.items():
        if key not in seen:
            residual_scores[key] = residual_scores.get(key, 0.0) + 1.0 / (
                rrf_k + rank
            )
    for key, rank in structured_ranks.items():
        if rank <= protected_structured_k or key in seen:
            continue
        residual_scores[key] = residual_scores.get(key, 0.0) + 1.0 / (
            rrf_k + rank
        )
    residual = sorted(residual_scores, key=lambda key: (-residual_scores[key], key))
    selected = (protected + residual)[:total_k]
    residual_rank = {key: rank for rank, key in enumerate(residual, 1)}
    result: list[dict[str, Any]] = []
    for final_rank, key in enumerate(selected, 1):
        support = [
            family
            for family, ranks in (
                ("raw", raw_ranks),
                ("structured", structured_ranks),
            )
            if key in ranks
        ]
        result.append(
            {
                "candidate_key": key,
                "raw_family_rank": raw_ranks.get(key),
                "structured_family_rank": structured_ranks.get(key),
                "protected_structured": key in protected,
                "residual_rrf_rank": residual_rank.get(key),
                "residual_rrf_score": residual_scores.get(key),
                "slot_id": slot_id,
                "final_r4_rank": final_rank,
                "family_support": support,
            }
        )
    return result


def fuse_single_slot_families(
    raw_family: list[dict[str, Any]],
    structured_family: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return fuse_candidate_families(
        raw_family,
        structured_family,
        protected_structured_k=20,
        total_k=40,
        rrf_k=60,
    )


def fuse_multi_slot_families(
    slot_family_rankings: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Fuse each slot with one Structured opportunity, then old round-robin."""
    slot_traces: dict[str, list[dict[str, Any]]] = {}
    slot_hits: dict[str, list[CandidateRRFHit]] = {}
    for slot_id, ranking in slot_family_rankings.items():
        trace = fuse_candidate_families(
            (ranking.get("raw") or {}).get("fused") or [],
            (ranking.get("structured") or {}).get("fused") or [],
            protected_structured_k=1,
            total_k=40,
            rrf_k=60,
            slot_id=slot_id,
        )
        slot_traces[slot_id] = trace
        slot_hits[slot_id] = [
            CandidateRRFHit(
                candidate_key=item["candidate_key"],
                rrf_score=float(item.get("residual_rrf_score") or 0.0),
                lane_ranks={
                    family: rank
                    for family, rank in (
                        ("raw_family", item.get("raw_family_rank")),
                        ("structured_family", item.get("structured_family_rank")),
                    )
                    if rank is not None
                },
                supporting_view_ids={},
            )
            for item in trace
        ]
    pool = build_slot_pool(
        slot_hits,
        slot_top_k=20,
        slot_min_budget=10,
        total_k=40,
    )
    enriched: list[dict[str, Any]] = []
    trace_maps = {
        slot_id: {item["candidate_key"]: item for item in items}
        for slot_id, items in slot_traces.items()
    }
    for final_rank, item in enumerate(pool, 1):
        trace = dict(trace_maps[item["slot_id"]][item["candidate_key"]])
        trace.update(item)
        trace["final_r4_rank"] = final_rank
        enriched.append(trace)
    return enriched, slot_traces
