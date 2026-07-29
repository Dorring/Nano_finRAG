"""Tests for NF39 stage attribution and final loss classification."""
from __future__ import annotations

from src.evaluation.evaluation import EvaluationCase, ExpectedSource
from src.evaluation.nf39_attribution import (
    EvaluationIntegrityError,
    FinalLossStage,
    canonical_candidate_key,
    classify_final_loss,
    contains_gold,
    evidence_family_key,
    to_stage_candidate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_expected(
    chunk_id: str = "ev_1",
    filename: str = "doc_a",
    page: int | None = None,
) -> ExpectedSource:
    return ExpectedSource(filename=filename, page=page, chunk_id=chunk_id)


def _make_case(
    case_id: str = "case_1",
    question: str = "What is the revenue?",
    expected_sources: tuple[ExpectedSource, ...] = (),
    expected_no_answer: bool = False,
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question=question,
        expected_sources=expected_sources,
        expected_answer_contains=(),
        expected_numbers=(),
        expected_no_answer=expected_no_answer,
        expected_calculations=(),
        expected_intent=None,
        document_names=("doc_a",),
        tags=(),
        metadata={},
    )


def _make_candidate(
    eid: str,
    doc: str = "doc_a",
    page: int = 1,
    block_type: str = "text",
    parent_id: str | None = None,
    table_id: str | None = None,
) -> dict:
    return {
        "evidence_id": eid,
        "candidate_id": eid,
        "document_id": doc,
        "page": page,
        "block_type": block_type,
        "parent_id": parent_id,
        "table_id": table_id,
    }


def _make_candidates(n: int, doc: str = "doc_a") -> list[dict]:
    return [_make_candidate(f"ev_{i}", doc=doc, page=i) for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


def test_truncated_before_reranker_classification():
    """Gold in RRF Top-40 but not in reranker input → TRUNCATED_BEFORE_RERANKER."""
    expected = _make_expected(chunk_id="ev_5")
    rrf_top40 = _make_candidates(10)
    reranker_input = _make_candidates(4)  # ev_5 is missing
    reranker_ranked = _make_candidates(4)
    final_top5 = _make_candidates(4)

    stage = classify_final_loss(
        expected_sources=[expected],
        rrf_top40=rrf_top40,
        reranker_input=reranker_input,
        reranker_ranked=reranker_ranked,
        final_top5=final_top5,
        golden_pass=True,
    )
    assert stage == FinalLossStage.TRUNCATED_BEFORE_RERANKER


def test_demoted_by_reranker_classification():
    """Gold in reranker input but not in reranker Top-5 → DEMOTED_BY_RERANKER."""
    expected = _make_expected(chunk_id="ev_6")
    rrf_top40 = _make_candidates(10)
    reranker_input = _make_candidates(10)
    # ev_6 is at rank 6 in reranker output (not in Top-5)
    reranker_ranked = _make_candidates(10)
    final_top5 = _make_candidates(5)  # ev_6 not in final

    stage = classify_final_loss(
        expected_sources=[expected],
        rrf_top40=rrf_top40,
        reranker_input=reranker_input,
        reranker_ranked=reranker_ranked,
        final_top5=final_top5,
        golden_pass=True,
    )
    assert stage == FinalLossStage.DEMOTED_BY_RERANKER


def test_dropped_by_final_selector_classification():
    """Gold in reranker Top-5 but not in Final Top-5 → DROPPED_BY_FINAL_SELECTOR."""
    expected = _make_expected(chunk_id="ev_3")
    rrf_top40 = _make_candidates(10)
    reranker_input = _make_candidates(10)
    # ev_3 is in reranker Top-5 (rank 3)
    reranker_ranked = _make_candidates(10)
    # ev_3 is NOT in final Top-5
    final_top5 = [
        _make_candidate("ev_1"),
        _make_candidate("ev_2"),
        _make_candidate("ev_4"),
        _make_candidate("ev_5"),
        _make_candidate("ev_6"),
    ]

    stage = classify_final_loss(
        expected_sources=[expected],
        rrf_top40=rrf_top40,
        reranker_input=reranker_input,
        reranker_ranked=reranker_ranked,
        final_top5=final_top5,
        golden_pass=True,
    )
    assert stage == FinalLossStage.DROPPED_BY_FINAL_SELECTOR


def test_present_in_final_answer_failed_classification():
    """Gold in Final Top-5 but golden_pass=False → PRESENT_IN_FINAL_ANSWER_FAILED."""
    expected = _make_expected(chunk_id="ev_3")
    rrf_top40 = _make_candidates(10)
    reranker_input = _make_candidates(10)
    reranker_ranked = _make_candidates(10)
    final_top5 = _make_candidates(5)  # ev_3 is in Top-5

    stage = classify_final_loss(
        expected_sources=[expected],
        rrf_top40=rrf_top40,
        reranker_input=reranker_input,
        reranker_ranked=reranker_ranked,
        final_top5=final_top5,
        golden_pass=False,
    )
    assert stage == FinalLossStage.PRESENT_IN_FINAL_ANSWER_FAILED


def test_not_in_rrf_40_classification():
    """Gold not in RRF Top-40 → NOT_IN_RRF_40."""
    expected = _make_expected(chunk_id="ev_999")
    rrf_top40 = _make_candidates(10)
    reranker_input = _make_candidates(10)
    reranker_ranked = _make_candidates(10)
    final_top5 = _make_candidates(5)

    stage = classify_final_loss(
        expected_sources=[expected],
        rrf_top40=rrf_top40,
        reranker_input=reranker_input,
        reranker_ranked=reranker_ranked,
        final_top5=final_top5,
        golden_pass=True,
    )
    assert stage == FinalLossStage.NOT_IN_RRF_40


def test_passed_classification():
    """Gold in Final Top-5 and golden_pass=True → PASSED."""
    expected = _make_expected(chunk_id="ev_3")
    rrf_top40 = _make_candidates(10)
    reranker_input = _make_candidates(10)
    reranker_ranked = _make_candidates(10)
    final_top5 = _make_candidates(5)

    stage = classify_final_loss(
        expected_sources=[expected],
        rrf_top40=rrf_top40,
        reranker_input=reranker_input,
        reranker_ranked=reranker_ranked,
        final_top5=final_top5,
        golden_pass=True,
    )
    assert stage == FinalLossStage.PASSED


def test_no_expected_sources_returns_passed():
    """Cases without expected_sources are automatically PASSED."""
    stage = classify_final_loss(
        expected_sources=[],
        rrf_top40=[],
        reranker_input=[],
        reranker_ranked=[],
        final_top5=[],
        golden_pass=False,
    )
    assert stage == FinalLossStage.PASSED


# ---------------------------------------------------------------------------
# Candidate identity tests
# ---------------------------------------------------------------------------


def test_candidate_identity_is_stable_across_stages():
    """The same candidate must produce the same key regardless of stage context."""
    candidate_dict = _make_candidate("ev_1", block_type="text")
    cand = to_stage_candidate(candidate_dict)
    key1 = canonical_candidate_key(cand)

    # The same candidate appears in different stage lists but keeps its key
    cand2 = to_stage_candidate(
        {**candidate_dict, "rank": 1, "score": 0.5}
    )
    key2 = canonical_candidate_key(cand2)

    assert key1 == key2, "Identity must be stable across stages"


def test_table_cell_maps_to_parent_row_identity():
    """table_cell must produce the same key as its parent table_row."""
    cell = to_stage_candidate(
        _make_candidate("cell_1", block_type="table_cell", parent_id="row_1")
    )
    row = to_stage_candidate(
        _make_candidate("row_1", block_type="table_row")
    )

    assert canonical_candidate_key(cell) == canonical_candidate_key(row), (
        "table_cell and parent table_row must share canonical key"
    )
    assert canonical_candidate_key(cell).startswith("table_row:")


def test_table_cell_without_parent_raises():
    """table_cell without parent_row_id must raise EvaluationIntegrityError."""
    import pytest

    cell = to_stage_candidate(
        _make_candidate("cell_1", block_type="table_cell", parent_id=None)
    )
    with pytest.raises(EvaluationIntegrityError, match="table_cell has no parent row"):
        canonical_candidate_key(cell)


def test_evidence_family_key_for_table_row():
    """table_row evidence family uses row identity."""
    row = to_stage_candidate(
        _make_candidate("row_1", block_type="table_row")
    )
    assert evidence_family_key(row) == "row:doc_a:row_1"


def test_evidence_family_key_for_block_with_parent():
    """Blocks with parent_id use parent family key."""
    block = to_stage_candidate(
        _make_candidate("ev_1", block_type="text", parent_id="parent_1")
    )
    assert evidence_family_key(block) == "parent:doc_a:parent_1"


def test_evidence_family_key_for_orphan_block():
    """Blocks without parent_id use canonical key."""
    block = to_stage_candidate(
        _make_candidate("ev_1", block_type="text", parent_id=None)
    )
    assert evidence_family_key(block) == "block:doc_a:ev_1"


def test_contains_gold_matches_by_chunk_id():
    """contains_gold must match by chunk_id/evidence_id."""
    expected = _make_expected(chunk_id="ev_3")
    candidates = _make_candidates(5)
    assert contains_gold(candidates, [expected]) is True

    expected_miss = _make_expected(chunk_id="ev_999")
    assert contains_gold(candidates, [expected_miss]) is False


def test_contains_gold_matches_by_filename_and_page():
    """contains_gold must match by filename and page when chunk_id doesn't match."""
    expected = ExpectedSource(filename="doc_a", page=3, chunk_id="")
    candidates = _make_candidates(5)
    assert contains_gold(candidates, [expected]) is True
