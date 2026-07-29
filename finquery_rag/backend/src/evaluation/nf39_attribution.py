"""NF39 RRF-to-Final attribution and rank-preserving fusion support.

This module provides the core data structures and classification logic to
answer:

> Did labeled evidence disappear at RRF Top-40, Reranker Input truncation,
> Reranker ranking, Final Context selection, or generation?

It also provides redundancy statistics for the Final Top-5 context.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.evaluation.evaluation import EvaluationCase, ExpectedSource
from src.evaluation.nf37_metrics import (
    all_source_coverage_at_k,
    candidate_to_source,
    case_hit_rate_at_k,
    source_recall_at_k,
)


class EvaluationIntegrityError(ValueError):
    """Raised when candidate identity integrity is violated."""


class FinalLossStage(str, Enum):
    """Unified failure stage for gold evidence loss attribution."""

    NOT_IN_RRF_40 = "not_in_rrf_40"
    TRUNCATED_BEFORE_RERANKER = "truncated_before_reranker"
    DEMOTED_BY_RERANKER = "demoted_by_reranker"
    DROPPED_BY_FINAL_SELECTOR = "dropped_by_final_selector"
    PRESENT_IN_FINAL_ANSWER_FAILED = "present_in_final_answer_failed"
    PASSED = "passed"


# ---------------------------------------------------------------------------
# Stage candidate normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageCandidate:
    """Normalized candidate carrying unified identity fields.

    For ``table_cell`` blocks, ``parent_row_id`` is derived from ``parent_id``
    (the parent row's evidence_id).  For ``table_row`` blocks, ``row_id`` is
    the row's own ``evidence_id``.  This lets ``canonical_candidate_key``
    produce the same key for a cell and its parent row.
    """

    evidence_id: str
    document_id: str
    page: int | None
    block_type: str
    parent_id: str | None = None
    table_id: str | None = None
    parent_row_id: str | None = None
    row_id: str | None = None


def to_stage_candidate(summary: dict[str, Any]) -> StageCandidate:
    """Convert a ``summarize_candidates`` row into a :class:`StageCandidate`."""
    block_type = summary.get("block_type", "text")
    parent_id = summary.get("parent_id")
    evidence_id = summary.get("evidence_id") or summary.get("candidate_id") or ""
    parent_row_id = parent_id if block_type == "table_cell" else None
    row_id = evidence_id if block_type == "table_row" else None
    return StageCandidate(
        evidence_id=evidence_id,
        document_id=summary.get("document_id", ""),
        page=summary.get("page"),
        block_type=block_type,
        parent_id=parent_id,
        table_id=summary.get("table_id"),
        parent_row_id=parent_row_id,
        row_id=row_id,
    )


# ---------------------------------------------------------------------------
# Unified candidate identity
# ---------------------------------------------------------------------------


def canonical_candidate_key(candidate: StageCandidate) -> str:
    """Return the canonical identity key for a candidate.

    - ``table_cell`` maps to its parent row so cells never act as independent
      final candidates.
    - ``table_row`` uses its own row identity.
    - All other blocks use ``block:{document_id}:{evidence_id}``.

    A ``table_cell`` without ``parent_row_id`` raises
    :class:`EvaluationIntegrityError`.
    """
    if candidate.block_type == "table_cell":
        if not candidate.parent_row_id:
            raise EvaluationIntegrityError(
                "table_cell has no parent row"
            )
        return (
            f"table_row:"
            f"{candidate.document_id}:"
            f"{candidate.parent_row_id}"
        )

    if candidate.block_type == "table_row":
        return (
            f"table_row:"
            f"{candidate.document_id}:"
            f"{candidate.row_id}"
        )

    return (
        f"block:"
        f"{candidate.document_id}:"
        f"{candidate.evidence_id}"
    )


def evidence_family_key(candidate: StageCandidate) -> str:
    """Return the evidence family key for redundancy counting.

    - ``table_cell`` / ``table_row`` → ``row:{document_id}:{parent_row_id|row_id}``
    - blocks with ``parent_id`` → ``parent:{document_id}:{parent_id}``
    - otherwise → :func:`canonical_candidate_key`
    """
    if candidate.block_type in {"table_cell", "table_row"}:
        return (
            f"row:"
            f"{candidate.document_id}:"
            f"{candidate.parent_row_id or candidate.row_id}"
        )

    if candidate.parent_id:
        return (
            f"parent:"
            f"{candidate.document_id}:"
            f"{candidate.parent_id}"
        )

    return canonical_candidate_key(candidate)


# ---------------------------------------------------------------------------
# Candidate stage position tracking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateStagePosition:
    """Track a single candidate's rank/score across every pipeline stage."""

    candidate_id: str
    document_id: str
    page: int | None
    block_type: str

    rrf_rank: int | None = None
    reranker_input_rank: int | None = None
    reranker_rank: int | None = None
    final_input_rank: int | None = None
    final_rank: int | None = None

    rrf_score: float | None = None
    reranker_score: float | None = None


def build_stage_positions(
    rrf_top40: list[dict[str, Any]],
    reranker_input: list[dict[str, Any]],
    reranker_ranked: list[dict[str, Any]],
    final_input: list[dict[str, Any]],
    final_top5: list[dict[str, Any]],
) -> dict[str, CandidateStagePosition]:
    """Build a map of ``canonical_candidate_key`` → :class:`CandidateStagePosition`.

    All stage lists are in ``summarize_candidates`` dict format.
    Ranks are 1-based; absent candidates get ``None``.
    """
    positions: dict[str, CandidateStagePosition] = {}

    def _register(summary: dict[str, Any]) -> StageCandidate:
        return to_stage_candidate(summary)

    # RRF stage
    for rank, row in enumerate(rrf_top40, start=1):
        cand = _register(row)
        key = canonical_candidate_key(cand)
        if key not in positions:
            positions[key] = CandidateStagePosition(
                candidate_id=cand.evidence_id,
                document_id=cand.document_id,
                page=cand.page,
                block_type=cand.block_type,
            )
        pos = positions[key]
        positions[key] = CandidateStagePosition(
            candidate_id=pos.candidate_id,
            document_id=pos.document_id,
            page=pos.page,
            block_type=pos.block_type,
            rrf_rank=rank,
            reranker_input_rank=pos.reranker_input_rank,
            reranker_rank=pos.reranker_rank,
            final_input_rank=pos.final_input_rank,
            final_rank=pos.final_rank,
            rrf_score=row.get("rrf_score") or row.get("score"),
            reranker_score=pos.reranker_score,
        )

    # Reranker input stage
    for rank, row in enumerate(reranker_input, start=1):
        cand = _register(row)
        key = canonical_candidate_key(cand)
        if key not in positions:
            positions[key] = CandidateStagePosition(
                candidate_id=cand.evidence_id,
                document_id=cand.document_id,
                page=cand.page,
                block_type=cand.block_type,
            )
        pos = positions[key]
        positions[key] = CandidateStagePosition(
            candidate_id=pos.candidate_id,
            document_id=pos.document_id,
            page=pos.page,
            block_type=pos.block_type,
            rrf_rank=pos.rrf_rank,
            reranker_input_rank=rank,
            reranker_rank=pos.reranker_rank,
            final_input_rank=pos.final_input_rank,
            final_rank=pos.final_rank,
            rrf_score=pos.rrf_score,
            reranker_score=pos.reranker_score,
        )

    # Reranker ranked stage
    for rank, row in enumerate(reranker_ranked, start=1):
        cand = _register(row)
        key = canonical_candidate_key(cand)
        if key not in positions:
            positions[key] = CandidateStagePosition(
                candidate_id=cand.evidence_id,
                document_id=cand.document_id,
                page=cand.page,
                block_type=cand.block_type,
            )
        pos = positions[key]
        positions[key] = CandidateStagePosition(
            candidate_id=pos.candidate_id,
            document_id=pos.document_id,
            page=pos.page,
            block_type=pos.block_type,
            rrf_rank=pos.rrf_rank,
            reranker_input_rank=pos.reranker_input_rank,
            reranker_rank=rank,
            final_input_rank=pos.final_input_rank,
            final_rank=pos.final_rank,
            rrf_score=pos.rrf_score,
            reranker_score=row.get("reranker_score") or row.get("score"),
        )

    # Final input stage
    for rank, row in enumerate(final_input, start=1):
        cand = _register(row)
        key = canonical_candidate_key(cand)
        if key not in positions:
            positions[key] = CandidateStagePosition(
                candidate_id=cand.evidence_id,
                document_id=cand.document_id,
                page=cand.page,
                block_type=cand.block_type,
            )
        pos = positions[key]
        positions[key] = CandidateStagePosition(
            candidate_id=pos.candidate_id,
            document_id=pos.document_id,
            page=pos.page,
            block_type=pos.block_type,
            rrf_rank=pos.rrf_rank,
            reranker_input_rank=pos.reranker_input_rank,
            reranker_rank=pos.reranker_rank,
            final_input_rank=rank,
            final_rank=pos.final_rank,
            rrf_score=pos.rrf_score,
            reranker_score=pos.reranker_score,
        )

    # Final Top-5 stage
    for rank, row in enumerate(final_top5, start=1):
        cand = _register(row)
        key = canonical_candidate_key(cand)
        if key not in positions:
            positions[key] = CandidateStagePosition(
                candidate_id=cand.evidence_id,
                document_id=cand.document_id,
                page=cand.page,
                block_type=cand.block_type,
            )
        pos = positions[key]
        positions[key] = CandidateStagePosition(
            candidate_id=pos.candidate_id,
            document_id=pos.document_id,
            page=pos.page,
            block_type=pos.block_type,
            rrf_rank=pos.rrf_rank,
            reranker_input_rank=pos.reranker_input_rank,
            reranker_rank=pos.reranker_rank,
            final_input_rank=pos.final_input_rank,
            final_rank=rank,
            rrf_score=pos.rrf_score,
            reranker_score=pos.reranker_score,
        )

    return positions


# ---------------------------------------------------------------------------
# Gold matching helpers
# ---------------------------------------------------------------------------


def _candidate_summary_to_source(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a summarize_candidates row to the source-matching format."""
    return {
        "chunk_id": row.get("evidence_id") or row.get("candidate_id"),
        "filename": row.get("document_id"),
        "page": row.get("page"),
    }


def contains_gold(
    candidates: list[dict[str, Any]],
    expected_sources: Iterable[ExpectedSource],
) -> bool:
    """Return True if any expected source matches any candidate in the list."""
    expected_list = list(expected_sources)
    if not expected_list:
        return True
    for row in candidates:
        source = _candidate_summary_to_source(row)
        for expected in expected_list:
            if expected.matches(source):
                return True
    return False


def gold_ranks(
    candidates: list[dict[str, Any]],
    expected_sources: Iterable[ExpectedSource],
) -> list[int]:
    """Return 1-based ranks of all gold matches in ``candidates``."""
    expected_list = list(expected_sources)
    ranks: list[int] = []
    for rank, row in enumerate(candidates, start=1):
        source = _candidate_summary_to_source(row)
        for expected in expected_list:
            if expected.matches(source):
                ranks.append(rank)
                break
    return ranks


# ---------------------------------------------------------------------------
# Final loss classification
# ---------------------------------------------------------------------------


def classify_final_loss(
    *,
    expected_sources: Iterable[ExpectedSource],
    rrf_top40: list[dict[str, Any]],
    reranker_input: list[dict[str, Any]],
    reranker_ranked: list[dict[str, Any]],
    final_top5: list[dict[str, Any]],
    golden_pass: bool,
) -> FinalLossStage:
    """Classify where gold evidence was lost between RRF and Final Context.

    The classification is sequential:

    1. ``NOT_IN_RRF_40`` – gold absent from RRF Top-40.
    2. ``TRUNCATED_BEFORE_RERANKER`` – gold in RRF but not in reranker input.
    3. ``DEMOTED_BY_RERANKER`` – gold in reranker input but not in reranker Top-5.
    4. ``DROPPED_BY_FINAL_SELECTOR`` – gold in reranker Top-5 but not in Final Top-5.
    5. ``PRESENT_IN_FINAL_ANSWER_FAILED`` – gold in Final Top-5 but answer failed.
    6. ``PASSED`` – gold present and answer passed.
    """
    expected_list = list(expected_sources)
    if not expected_list:
        return FinalLossStage.PASSED

    if not contains_gold(rrf_top40, expected_list):
        return FinalLossStage.NOT_IN_RRF_40

    if not contains_gold(reranker_input, expected_list):
        return FinalLossStage.TRUNCATED_BEFORE_RERANKER

    if not contains_gold(reranker_ranked[:5], expected_list):
        return FinalLossStage.DEMOTED_BY_RERANKER

    if not contains_gold(final_top5, expected_list):
        return FinalLossStage.DROPPED_BY_FINAL_SELECTOR

    if not golden_pass:
        return FinalLossStage.PRESENT_IN_FINAL_ANSWER_FAILED

    return FinalLossStage.PASSED


# ---------------------------------------------------------------------------
# Stage metrics (no-answer cases excluded from denominator)
# ---------------------------------------------------------------------------


def _filter_retrieval_cases(
    cases: Iterable[EvaluationCase],
) -> list[EvaluationCase]:
    """Return only cases that participate in retrieval metric denominators.

    No-answer cases (``expected_no_answer=True``) and cases without
    ``expected_sources`` are excluded.
    """
    return [
        case
        for case in cases
        if case.expected_sources and not case.expected_no_answer
    ]


def compute_stage_metrics(
    *,
    cases: Iterable[EvaluationCase],
    rankings: dict[str, list[dict[str, Any]]],
    ks: tuple[int, ...] = (5, 8, 20, 40),
) -> dict[str, Any]:
    """Compute Case Hit, Source Recall, All-source Coverage, and MRR.

    No-answer cases are excluded from the denominator.
    """
    retrieval_cases = _filter_retrieval_cases(cases)
    result: dict[str, Any] = {}
    for k in ks:
        result[f"case_hit_rate_at_{k}"] = case_hit_rate_at_k(
            retrieval_cases, rankings, k
        )
        result[f"source_recall_at_{k}"] = source_recall_at_k(
            retrieval_cases, rankings, k
        )
        result[f"all_source_coverage_at_{k}"] = all_source_coverage_at_k(
            retrieval_cases, rankings, k
        )

    reciprocal: list[float] = []
    for case in retrieval_cases:
        rank = next(
            (
                index
                for index, candidate in enumerate(
                    rankings.get(case.case_id, []), 1
                )
                if any(
                    expected.matches(candidate_to_source(candidate))
                    for expected in case.expected_sources
                )
            ),
            None,
        )
        reciprocal.append(1 / rank if rank else 0)
    result["mrr"] = (
        sum(reciprocal) / len(reciprocal) if reciprocal else 1.0
    )
    result["case_count"] = len(retrieval_cases)
    return result


# ---------------------------------------------------------------------------
# Redundancy statistics
# ---------------------------------------------------------------------------


@dataclass
class RedundancyReport:
    """Aggregate redundancy statistics for Final Top-5 across all cases."""

    unique_documents_at_5: int = 0
    unique_pages_at_5: int = 0
    unique_evidence_families_at_5: int = 0
    same_parent_duplicate_count_at_5: int = 0
    same_table_row_duplicate_count_at_5: int = 0
    same_page_candidate_ratio_at_5: float = 0.0
    total_candidates_at_5: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "unique_documents_at_5": self.unique_documents_at_5,
            "unique_pages_at_5": self.unique_pages_at_5,
            "unique_evidence_families_at_5": self.unique_evidence_families_at_5,
            "same_parent_duplicate_count_at_5": self.same_parent_duplicate_count_at_5,
            "same_table_row_duplicate_count_at_5": self.same_table_row_duplicate_count_at_5,
            "same_page_candidate_ratio_at_5": self.same_page_candidate_ratio_at_5,
            "total_candidates_at_5": self.total_candidates_at_5,
        }


def compute_redundancy(
    final_rankings: dict[str, list[dict[str, Any]]],
    *,
    top_k: int = 5,
) -> RedundancyReport:
    """Compute redundancy statistics for Final Top-K across all cases.

    Counts duplicates by parent block, table row, and page.  A candidate
    counts as a same-parent duplicate when its evidence family key has
    already been seen within the same case's Top-K.
    """
    report = RedundancyReport()
    all_documents: set[str] = set()
    all_pages: set[tuple[str, int | None]] = set()
    all_families: set[str] = set()
    total_same_parent = 0
    total_same_table_row = 0
    total_candidates = 0
    total_same_page = 0

    for candidates in final_rankings.values():
        top = candidates[:top_k]
        if not top:
            continue
        stage_candidates = [to_stage_candidate(row) for row in top]
        family_keys: list[str] = []
        parent_keys: list[str] = []
        row_keys: list[str] = []
        pages: list[int | None] = []

        for cand in stage_candidates:
            all_documents.add(cand.document_id)
            all_pages.add((cand.document_id, cand.page))
            fk = evidence_family_key(cand)
            all_families.add(fk)
            family_keys.append(fk)
            parent_keys.append(
                f"parent:{cand.document_id}:{cand.parent_id}"
                if cand.parent_id
                else fk
            )
            row_keys.append(canonical_candidate_key(cand))
            pages.append(cand.page)

        # Count duplicates within this case's Top-K
        parent_counter = Counter(parent_keys)
        row_counter = Counter(row_keys)

        total_same_parent += sum(
            count - 1 for count in parent_counter.values() if count > 1
        )
        total_same_table_row += sum(
            count - 1
            for key, count in row_counter.items()
            if key.startswith("table_row:") and count > 1
        )
        total_candidates += len(top)

        # Same-page ratio: within each case, how many candidates share a page
        page_counter = Counter(pages)
        # For each case, count candidates that share a page with another
        same_in_case = sum(
            count for count in page_counter.values() if count > 1
        )
        total_same_page += same_in_case

    report.unique_documents_at_5 = len(all_documents)
    report.unique_pages_at_5 = len(all_pages)
    report.unique_evidence_families_at_5 = len(all_families)
    report.same_parent_duplicate_count_at_5 = total_same_parent
    report.same_table_row_duplicate_count_at_5 = total_same_table_row
    report.total_candidates_at_5 = total_candidates
    report.same_page_candidate_ratio_at_5 = (
        total_same_page / total_candidates if total_candidates else 0.0
    )
    return report


# ---------------------------------------------------------------------------
# Case-level attribution record
# ---------------------------------------------------------------------------


@dataclass
class CaseAttribution:
    """Per-case attribution result."""

    case_id: str
    bucket: str
    loss_stage: FinalLossStage
    gold_rrf_ranks: list[int] = field(default_factory=list)
    gold_reranker_ranks: list[int] = field(default_factory=list)
    gold_final_ranks: list[int] = field(default_factory=list)
    rrf_top5_hit: bool = False
    reranker_top5_hit: bool = False
    final_top5_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "bucket": self.bucket,
            "loss_stage": self.loss_stage.value,
            "gold_rrf_ranks": self.gold_rrf_ranks,
            "gold_reranker_ranks": self.gold_reranker_ranks,
            "gold_final_ranks": self.gold_final_ranks,
            "rrf_top5_hit": self.rrf_top5_hit,
            "reranker_top5_hit": self.reranker_top5_hit,
            "final_top5_hit": self.final_top5_hit,
        }


def build_case_attribution(
    *,
    case: EvaluationCase,
    rrf_top40: list[dict[str, Any]],
    reranker_input: list[dict[str, Any]],
    reranker_ranked: list[dict[str, Any]],
    final_top5: list[dict[str, Any]],
    golden_pass: bool,
) -> CaseAttribution:
    """Build a :class:`CaseAttribution` for a single case."""
    expected = case.expected_sources
    loss_stage = classify_final_loss(
        expected_sources=expected,
        rrf_top40=rrf_top40,
        reranker_input=reranker_input,
        reranker_ranked=reranker_ranked,
        final_top5=final_top5,
        golden_pass=golden_pass,
    )

    bucket = "no_answer"
    if case.expected_no_answer:
        bucket = "no_answer"
    elif case.expected_intent:
        bucket = case.expected_intent
    elif expected:
        bucket = "retrieval"

    return CaseAttribution(
        case_id=case.case_id,
        bucket=bucket,
        loss_stage=loss_stage,
        gold_rrf_ranks=gold_ranks(rrf_top40, expected),
        gold_reranker_ranks=gold_ranks(reranker_ranked, expected),
        gold_final_ranks=gold_ranks(final_top5, expected),
        rrf_top5_hit=contains_gold(rrf_top40[:5], expected),
        reranker_top5_hit=contains_gold(reranker_ranked[:5], expected),
        final_top5_hit=contains_gold(final_top5, expected),
    )
