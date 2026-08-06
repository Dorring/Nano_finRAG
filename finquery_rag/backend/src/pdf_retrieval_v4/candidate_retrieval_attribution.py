"""Gate 08 R2 failure attribution for unrecovered B-class candidates.

Classifies why a B-class candidate was not recovered in the candidate
direct retrieval pool, producing a structured attribution result for
diagnostics and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit
from src.pdf_retrieval_v4.candidate_view_index import CandidateSearchHit


class BClassFailureStage(str, Enum):
    """Failure stages for B-class candidate recovery attribution."""

    CANDIDATE_NOT_IN_ANY_TOP50 = "candidate_not_in_any_top50"
    RAW_VIEW_QUERY_MISMATCH = "raw_view_query_mismatch"
    STRUCTURED_VIEW_QUERY_MISMATCH = "structured_view_query_mismatch"
    METRIC_CONFLICT_FILTER = "metric_conflict_filter"
    PERIOD_CONFLICT_FILTER = "period_conflict_filter"
    CANDIDATE_RANK_41_TO_50 = "candidate_rank_41_to_50"
    CANDIDATE_POOL_BUDGET_TRUNCATED = "candidate_pool_budget_truncated"
    MULTI_SLOT_BUDGET_TRUNCATED = "multi_slot_budget_truncated"
    RECOVERED = "recovered"


@dataclass(frozen=True)
class BClassAttributionResult:
    """Attribution result for a single B-class candidate."""

    candidate_key: str
    case_id: str
    first_failure_stage: BClassFailureStage
    best_rank: int | None
    in_top50: bool
    in_top40: bool
    detail: str = ""


def classify_b_class_failure(
    *,
    candidate_key: str,
    case_id: str,
    candidate_direct_pool: list[dict[str, Any]],
    lane_hits: dict[str, list[CandidateSearchHit]],
    rrf_hits: list[CandidateRRFHit],
    is_multi_slot: bool,
) -> BClassAttributionResult:
    """Classify the failure stage for an unrecovered B-class candidate.

    Classification priority:

    1.  If candidate in top-40 pool -> RECOVERED.
    2.  If multi-slot and candidate in RRF but truncated by slot budget
        -> MULTI_SLOT_BUDGET_TRUNCATED.
    3.  If candidate in RRF top-50 (rank 41-50) -> CANDIDATE_RANK_41_TO_50.
    4.  If candidate appears in any lane's top-50 but not in RRF top-50
        -> CANDIDATE_POOL_BUDGET_TRUNCATED.
    5.  If candidate not in any lane's top-50:
        - If candidate has structured view but not in structured lanes
          -> STRUCTURED_VIEW_QUERY_MISMATCH.
        - If candidate only has raw view and not in raw lanes
          -> RAW_VIEW_QUERY_MISMATCH.
    6.  Default -> CANDIDATE_NOT_IN_ANY_TOP50.
    """
    # Build lookup structures.
    pool_rank_map: dict[str, int] = {}
    for rank, item in enumerate(candidate_direct_pool, 1):
        key = str(item.get("candidate_key") or "")
        if key and key not in pool_rank_map:
            pool_rank_map[key] = rank

    rrf_rank_map: dict[str, int] = {}
    for rank, hit in enumerate(rrf_hits, 1):
        if hit.candidate_key and hit.candidate_key not in rrf_rank_map:
            rrf_rank_map[hit.candidate_key] = rank

    # 1. If candidate in top-40 pool -> RECOVERED.
    if candidate_key in pool_rank_map:
        return BClassAttributionResult(
            candidate_key=candidate_key,
            case_id=case_id,
            first_failure_stage=BClassFailureStage.RECOVERED,
            best_rank=pool_rank_map[candidate_key],
            in_top50=True,
            in_top40=True,
        )

    # 2. If multi-slot and candidate in RRF but not in pool
    #    -> MULTI_SLOT_BUDGET_TRUNCATED.
    if is_multi_slot and candidate_key in rrf_rank_map:
        rank = rrf_rank_map[candidate_key]
        return BClassAttributionResult(
            candidate_key=candidate_key,
            case_id=case_id,
            first_failure_stage=BClassFailureStage.MULTI_SLOT_BUDGET_TRUNCATED,
            best_rank=rank,
            in_top50=True,
            in_top40=False,
            detail=f"rrf_rank_{rank}_truncated_by_slot_budget",
        )

    # 3. If candidate in RRF top-50 (rank 41-50) -> CANDIDATE_RANK_41_TO_50.
    if candidate_key in rrf_rank_map:
        rank = rrf_rank_map[candidate_key]
        return BClassAttributionResult(
            candidate_key=candidate_key,
            case_id=case_id,
            first_failure_stage=BClassFailureStage.CANDIDATE_RANK_41_TO_50,
            best_rank=rank,
            in_top50=True,
            in_top40=False,
            detail=f"rrf_rank_{rank}",
        )

    # 4. If candidate appears in any lane's top-50 but not in RRF top-50
    #    -> CANDIDATE_POOL_BUDGET_TRUNCATED.
    in_structured = False
    in_raw = False
    best_lane_rank: int | None = None
    for lane, hits in lane_hits.items():
        for rank, hit in enumerate(hits, 1):
            if hit.candidate_key == candidate_key:
                if "structured" in lane:
                    in_structured = True
                if "raw" in lane:
                    in_raw = True
                if best_lane_rank is None or rank < best_lane_rank:
                    best_lane_rank = rank

    if in_structured or in_raw:
        return BClassAttributionResult(
            candidate_key=candidate_key,
            case_id=case_id,
            first_failure_stage=BClassFailureStage.CANDIDATE_POOL_BUDGET_TRUNCATED,
            best_rank=best_lane_rank,
            in_top50=True,
            in_top40=False,
            detail="in_lane_top50_not_in_rrf_top50",
        )

    # 5. Not in any lane's top-50.
    #    If candidate has structured view but not in structured lanes
    #    -> STRUCTURED_VIEW_QUERY_MISMATCH.
    #    If candidate only has raw view and not in raw lanes
    #    -> RAW_VIEW_QUERY_MISMATCH.
    #    (For candidates absent from all lane hits, view type cannot be
    #     determined from search results alone; these checks act as a
    #     safety net for partial-lane-presence cases.)
    if in_structured:
        return BClassAttributionResult(
            candidate_key=candidate_key,
            case_id=case_id,
            first_failure_stage=BClassFailureStage.STRUCTURED_VIEW_QUERY_MISMATCH,
            best_rank=best_lane_rank,
            in_top50=False,
            in_top40=False,
        )
    if in_raw:
        return BClassAttributionResult(
            candidate_key=candidate_key,
            case_id=case_id,
            first_failure_stage=BClassFailureStage.RAW_VIEW_QUERY_MISMATCH,
            best_rank=best_lane_rank,
            in_top50=False,
            in_top40=False,
        )

    # 6. Default -> CANDIDATE_NOT_IN_ANY_TOP50.
    return BClassAttributionResult(
        candidate_key=candidate_key,
        case_id=case_id,
        first_failure_stage=BClassFailureStage.CANDIDATE_NOT_IN_ANY_TOP50,
        best_rank=None,
        in_top50=False,
        in_top40=False,
    )
