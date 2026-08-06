"""Gate 08 R2 candidate-level RRF fusion across 4 lanes.

Fuses hits from the 4 candidate-aligned lanes (candidate_raw_bm25,
candidate_raw_dense, candidate_structured_bm25,
candidate_structured_dense) into a single candidate-ranked list using
Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.pdf_retrieval_v4.candidate_view_index import CandidateSearchHit


@dataclass(frozen=True)
class CandidateRRFHit:
    """A single candidate-level RRF-fused hit."""

    candidate_key: str
    rrf_score: float
    lane_ranks: dict[str, int]  # lane_name -> rank (1-based)
    supporting_view_ids: dict[str, str]  # lane_name -> view_id


def fuse_candidate_hits(
    lane_hits: dict[str, list[CandidateSearchHit]],
    *,
    rrf_k: int = 60,
    lane_weights: dict[str, float] | None = None,
) -> list[CandidateRRFHit]:
    """Fuse candidate hits across lanes using RRF.

    For each lane, build a candidate_key -> rank mapping (1-based).
    For each candidate that appears in any lane, compute::

        score = sum(weight * 1/(rrf_k + rank) for each lane where candidate appears)

    Default all weights = 1.0.  Sort by (-rrf_score, candidate_key) for
    determinism.
    """
    weights = dict(lane_weights) if lane_weights else {}

    # Build candidate_key -> {lane: (rank, view_id)} mapping.
    candidate_lanes: dict[str, dict[str, tuple[int, str]]] = {}
    for lane, hits in lane_hits.items():
        if weights.get(lane, 1.0) == 0.0:
            continue
        for position, hit in enumerate(hits, 1):
            if not hit.candidate_key:
                continue
            # Use the hit's actual rank field (bm25_rank or dense_rank);
            # fall back to list position if neither is set.
            actual_rank = hit.bm25_rank or hit.dense_rank or position
            candidate_lanes.setdefault(hit.candidate_key, {})[lane] = (
                actual_rank,
                hit.view_id,
            )

    # Compute RRF scores.
    results: list[CandidateRRFHit] = []
    for candidate_key, lane_data in candidate_lanes.items():
        score = 0.0
        lane_ranks: dict[str, int] = {}
        supporting_view_ids: dict[str, str] = {}
        for lane, (rank, view_id) in lane_data.items():
            weight = weights.get(lane, 1.0)
            score += weight * (1.0 / (rrf_k + rank))
            lane_ranks[lane] = rank
            supporting_view_ids[lane] = view_id
        results.append(
            CandidateRRFHit(
                candidate_key=candidate_key,
                rrf_score=score,
                lane_ranks=lane_ranks,
                supporting_view_ids=supporting_view_ids,
            )
        )

    results.sort(key=lambda h: (-h.rrf_score, h.candidate_key))
    return results
