"""Pure helpers for NF-EVAL-04 candidate-recall attribution.

The module deliberately contains no retrieval or production dependencies.  It
only classifies already observed index/rank records so the diagnostic runner
and its unit tests share one fail-closed vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class CandidateRecallFailureStage(str, Enum):
    GOLD_IDENTITY_INVALID = "gold_identity_invalid"
    MISSING_FROM_BOTH_INDEXES = "missing_from_both_indexes"
    BM25_ONLY_INDEXED = "bm25_only_indexed"
    DENSE_ONLY_INDEXED = "dense_only_indexed"
    NOT_RETRIEVED_BY_BM25_TOP200 = "not_retrieved_by_bm25_top200"
    NOT_RETRIEVED_BY_DENSE_TOP200 = "not_retrieved_by_dense_top200"
    MISSED_BY_BOTH_RETRIEVERS = "missed_by_both_retrievers"
    BM25_WINDOW_TRUNCATION = "bm25_window_truncation"
    DENSE_WINDOW_TRUNCATION = "dense_window_truncation"
    BOTH_WINDOWS_TRUNCATED = "both_windows_truncated"
    LOST_DURING_NORMALIZATION = "lost_during_normalization"
    LOST_DURING_RRF_FUSION = "lost_during_rrf_fusion"
    PARENT_CHILD_IDENTITY_MISMATCH = "parent_child_identity_mismatch"
    ENTERED_RRF_POOL = "entered_rrf_pool"


class RecallGate(str, Enum):
    BM25_QUERY_TERMINOLOGY = "bm25_query_terminology"
    CANDIDATE_WINDOW = "candidate_window"
    DENSE_COVERAGE = "dense_coverage"
    PARENT_CHILD = "parent_child"
    RRF_FUSION = "rrf_fusion"
    NO_CONCENTRATED_BOTTLENECK = "no_concentrated_bottleneck"


@dataclass(frozen=True)
class VerifiedCandidateEquivalence:
    gold_candidate_key: str
    retrievable_candidate_key: str
    relation: str
    verification_source: str


def rank_bucket(rank: int | None) -> str:
    """Return the requested diagnostic rank bucket."""

    if rank is None:
        return "not_retrieved"
    if rank <= 20:
        return "top20"
    if rank <= 40:
        return "21_40"
    if rank <= 100:
        return "41_100"
    if rank <= 200:
        return "101_200"
    return "not_retrieved"


def classify_first_recall_failure(
    *,
    identity_valid: bool,
    present_in_bm25_index: bool,
    present_in_dense_index: bool,
    bm25_rank: int | None,
    dense_rank: int | None,
    bm25_production_limit: int | None,
    dense_production_limit: int | None,
    entered_production_union: bool,
    entered_production_rrf: bool,
    normalization_lost: bool = False,
    rrf_lost: bool = False,
    parent_child_mismatch: bool = False,
) -> CandidateRecallFailureStage:
    """Classify the first observable loss for one Gold source.

    Index presence is reported separately from query retrieval.  A source
    that is indexed but absent from Top-200 is therefore a retriever miss,
    not a window truncation.  Window truncation requires a known rank within
    the diagnostic Top-200 and above the actual production request limit.
    """

    if not identity_valid:
        return CandidateRecallFailureStage.GOLD_IDENTITY_INVALID
    if not present_in_bm25_index and not present_in_dense_index:
        return CandidateRecallFailureStage.MISSING_FROM_BOTH_INDEXES
    if bm25_rank is None and dense_rank is None:
        if present_in_bm25_index and not present_in_dense_index:
            return CandidateRecallFailureStage.NOT_RETRIEVED_BY_BM25_TOP200
        if present_in_dense_index and not present_in_bm25_index:
            return CandidateRecallFailureStage.NOT_RETRIEVED_BY_DENSE_TOP200
        return CandidateRecallFailureStage.MISSED_BY_BOTH_RETRIEVERS

    bm25_truncated = (
        bm25_rank is not None
        and bm25_production_limit is not None
        and bm25_rank > bm25_production_limit
    )
    dense_truncated = (
        dense_rank is not None
        and dense_production_limit is not None
        and dense_rank > dense_production_limit
    )
    if bm25_truncated and dense_truncated:
        return CandidateRecallFailureStage.BOTH_WINDOWS_TRUNCATED
    if bm25_truncated and (dense_rank is None or dense_truncated):
        return CandidateRecallFailureStage.BM25_WINDOW_TRUNCATION
    if dense_truncated and (bm25_rank is None or bm25_truncated):
        return CandidateRecallFailureStage.DENSE_WINDOW_TRUNCATION

    if normalization_lost:
        return CandidateRecallFailureStage.LOST_DURING_NORMALIZATION
    if rrf_lost:
        return CandidateRecallFailureStage.LOST_DURING_RRF_FUSION
    if parent_child_mismatch:
        return CandidateRecallFailureStage.PARENT_CHILD_IDENTITY_MISMATCH
    if entered_production_rrf:
        return CandidateRecallFailureStage.ENTERED_RRF_POOL
    if entered_production_union:
        return CandidateRecallFailureStage.LOST_DURING_RRF_FUSION

    # A source may have been returned by only one production channel but then
    # omitted by an adapter.  Keep the explanation explicit rather than
    # silently calling it a retriever miss.
    if bm25_rank is not None and dense_rank is None:
        return CandidateRecallFailureStage.BM25_ONLY_INDEXED
    if dense_rank is not None and bm25_rank is None:
        return CandidateRecallFailureStage.DENSE_ONLY_INDEXED
    return CandidateRecallFailureStage.LOST_DURING_NORMALIZATION


def classify_index_presence(
    *, present_in_bm25_index: bool, present_in_dense_index: bool
) -> str:
    if present_in_bm25_index and present_in_dense_index:
        return "both_indexes"
    if present_in_bm25_index:
        return "bm25_only_indexed"
    if present_in_dense_index:
        return "dense_only_indexed"
    return "missing_from_both_indexes"


def require_verified_equivalence(
    equivalence: VerifiedCandidateEquivalence | None,
    *,
    relation: str,
) -> bool:
    """Accept only a pre-verified relation; same-page evidence is invalid."""

    if equivalence is None:
        return False
    return (
        equivalence.relation == relation
        and bool(equivalence.gold_candidate_key)
        and bool(equivalence.retrievable_candidate_key)
        and bool(equivalence.verification_source)
    )


def stage_key_set(candidates: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str(item.get("candidate_key"))
        for item in candidates
        if str(item.get("candidate_key") or "").strip()
    }


def candidate_in_scope(candidate: Mapping[str, Any], whitelist: set[str]) -> bool:
    document = candidate.get("canonical_document_id") or candidate.get("document_id")
    return str(document or "") in whitelist


def source_coverage(
    expected_keys: Sequence[str], candidate_keys: Iterable[str]
) -> str:
    expected = {str(key) for key in expected_keys if str(key)}
    present = expected & {str(key) for key in candidate_keys if str(key)}
    if not present:
        return "none"
    if present == expected:
        return "all"
    return "partial"


def choose_next_gate(
    *,
    terminology_cases: int,
    window_cases: int,
    dense_coverage_cases: int,
    parent_child_cases: int,
    rrf_fusion_cases: int,
) -> dict[str, Any]:
    """Choose exactly one direction using unique Case counts."""

    candidates: list[tuple[str, int, int]] = [
        (RecallGate.BM25_QUERY_TERMINOLOGY.value, terminology_cases, 12),
        (RecallGate.CANDIDATE_WINDOW.value, window_cases, 12),
        (RecallGate.DENSE_COVERAGE.value, dense_coverage_cases, 12),
        (RecallGate.PARENT_CHILD.value, parent_child_cases, 8),
        (RecallGate.RRF_FUSION.value, rrf_fusion_cases, 8),
    ]
    passing = [item for item in candidates if item[1] >= item[2]]
    if not passing:
        selected = RecallGate.NO_CONCENTRATED_BOTTLENECK.value
    else:
        # Stable order breaks equal-count ties without returning multiple
        # possible experiments.
        selected = max(passing, key=lambda item: (item[1], -candidates.index(item)))[0]
    return {
        "selected_gate": selected,
        "optimization_allowed": False,
        "thresholds": {name: threshold for name, _, threshold in candidates},
        "case_counts": {name: count for name, count, _ in candidates},
        "passing_gates": [name for name, _, _ in passing],
    }


def unique_case_count(rows: Iterable[Mapping[str, Any]], key: str) -> int:
    return len({str(row.get(key)) for row in rows if row.get(key) is not None})
