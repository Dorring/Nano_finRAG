"""Lane contribution ablation helpers for Gate 08 R2.1.

Pure functions for offline ablation of sealed Gate 08 R2 lane hits.
No re-encoding, no re-indexing, no re-retrieval — only re-combination
of existing lane hit lists.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

RRF_K = 60
LANE_K = 50
POOL_K = 40

RAW_LANES = ("candidate_raw_bm25", "candidate_raw_dense")
STRUCTURED_LANES = (
    "candidate_structured_bm25",
    "candidate_structured_dense",
)
ALL_LANES = RAW_LANES + STRUCTURED_LANES


def find_rank_in_lane(
    lane_hits: list[dict[str, Any]], candidate_key: str
) -> int | None:
    """Find rank of candidate_key in a lane's hit list.

    Returns the rank (1-based) if found, None otherwise.
    """
    for hit in lane_hits:
        if str(hit.get("candidate_key")) == candidate_key:
            return int(hit.get("rank"))
    return None


def find_rrf_rank(
    rrf_hits: list[dict[str, Any]], candidate_key: str
) -> int | None:
    """Find RRF rank of candidate_key in the fused result list."""
    for hit in rrf_hits:
        if str(hit.get("candidate_key")) == candidate_key:
            return int(hit.get("rank"))
    return None


def rrf_fuse(
    lane_hits_dict: dict[str, list[dict[str, Any]]],
    lanes: tuple[str, ...],
    rrf_k: int = RRF_K,
    top_k: int = POOL_K,
) -> list[str]:
    """Fuse lane hits using RRF, return sorted candidate keys.

    score(candidate) = sum(1 / (rrf_k + rank)) over selected lanes.
    Missing lanes contribute 0.  Ties broken by candidate_key for
    determinism.
    """
    scores: dict[str, float] = defaultdict(float)
    for lane in lanes:
        hits = lane_hits_dict.get(lane, [])
        for hit in hits:
            key = str(hit.get("candidate_key") or "")
            if not key:
                continue
            rank = int(hit.get("rank") or 0)
            if rank > 0:
                scores[key] += 1.0 / (rrf_k + rank)
    sorted_keys = sorted(scores.keys(), key=lambda k: (-scores[k], k))
    return sorted_keys[:top_k]


def build_e0_pool(
    raw_case: dict[str, Any],
    gate08_prediction: dict[str, Any],
) -> set[str]:
    """Build E0 pool: raw_full_rrf + existing structured source pool."""
    pool: set[str] = set()
    for item in raw_case.get("raw_full_rrf_candidates") or []:
        key = str(item.get("candidate_key") or "")
        if key:
            pool.add(key)
    for item in (
        gate08_prediction.get("structured_strict_source_pool") or []
    ):
        key = str(item.get("original_candidate_identity") or "")
        if key:
            pool.add(key)
    return pool


def build_combined_pool_keys(prediction: dict[str, Any]) -> set[str]:
    """Extract all candidate keys from R2 combined_pool."""
    return {
        str(item.get("candidate_key") or "")
        for item in prediction.get("combined_pool") or []
        if item.get("candidate_key")
    }


def build_raw_pool_keys(raw_case: dict[str, Any]) -> set[str]:
    """Extract raw pool candidate keys for raw-gold-retained metric."""
    return {
        str(item.get("candidate_key") or "")
        for item in raw_case.get("raw_full_rrf_candidates") or []
        if item.get("candidate_key")
    }


def classify_lane_support(
    *,
    recovered: bool,
    raw_bm25_rank: int | None,
    raw_dense_rank: int | None,
    structured_bm25_rank: int | None,
    structured_dense_rank: int | None,
) -> dict[str, bool]:
    """Classify which lane(s) supported recovery of a Gold source.

    Returns a dict with recovered_by_raw_lane, recovered_by_structured_lane,
    and recovered_by_fusion_only flags.
    """
    has_raw = raw_bm25_rank is not None or raw_dense_rank is not None
    has_structured = (
        structured_bm25_rank is not None
        or structured_dense_rank is not None
    )
    return {
        "recovered_by_raw_lane": recovered and has_raw,
        "recovered_by_structured_lane": recovered and has_structured,
        "recovered_by_fusion_only": (
            recovered and not has_raw and not has_structured
        ),
    }
