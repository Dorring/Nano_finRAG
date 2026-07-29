"""Tests for rank-preserving fusion (NF39 Section 十二-十三)."""
from __future__ import annotations

import inspect

from src.retrieval.rank_preserving_fusion import (
    candidate_key,
    rank_preserving_fusion,
    reciprocal_rank_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    eid: str,
    doc: str = "doc_a",
    page: int = 1,
    block_type: str = "text",
    parent_id: str | None = None,
) -> dict:
    return {
        "candidate_id": eid,
        "evidence_id": eid,
        "document_id": doc,
        "page": page,
        "block_type": block_type,
        "parent_id": parent_id,
    }


def _make_rrf_candidates(n: int = 10) -> list[dict]:
    return [_make_candidate(f"ev_{i}", page=i) for i in range(n)]


def _reverse_reranker(candidates: list[dict]) -> list[dict]:
    """Return candidates in reverse order to simulate reranker re-ranking."""
    return list(reversed(candidates))


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_rank_fusion_preserves_candidate_set():
    """The fusion output must contain exactly the same candidates as the RRF input."""
    rrf = _make_rrf_candidates(10)
    reranked = _reverse_reranker(rrf[:8])  # reranker only outputs 8

    fused = rank_preserving_fusion(
        rrf_candidates=rrf,
        reranked_candidates=reranked,
    )

    rrf_keys = {candidate_key(c) for c in rrf}
    fused_keys = {candidate_key(c) for c in fused}
    assert fused_keys == rrf_keys, "Fusion must not add or remove candidates"
    assert len(fused) == len(rrf), "Fusion must not duplicate candidates"


def test_rank_fusion_is_deterministic():
    """Same inputs must produce identical outputs."""
    rrf = _make_rrf_candidates(10)
    reranked = _reverse_reranker(rrf[:8])

    run1 = rank_preserving_fusion(
        rrf_candidates=rrf, reranked_candidates=reranked
    )
    run2 = rank_preserving_fusion(
        rrf_candidates=rrf, reranked_candidates=reranked
    )

    keys1 = [candidate_key(c) for c in run1]
    keys2 = [candidate_key(c) for c in run2]
    assert keys1 == keys2, "Fusion must be deterministic"


def test_equal_rank_ties_use_rrf_rank():
    """When two candidates have the same fused score, RRF rank breaks the tie.

    This happens when both candidates are absent from the reranker output
    (both get missing_rank).  In that case, the one with the better (lower)
    RRF rank must come first.
    """
    rrf = _make_rrf_candidates(5)
    # Empty reranker output: all candidates get missing_rank
    reranked: list[dict] = []

    fused = rank_preserving_fusion(
        rrf_candidates=rrf, reranked_candidates=reranked
    )

    # All candidates have the same reranker term (missing_rank).
    # Tie-break by RRF rank ascending.
    fused_keys = [candidate_key(c) for c in fused]
    rrf_keys = [candidate_key(c) for c in rrf]
    assert fused_keys == rrf_keys, (
        "When fused scores are equal, order must follow RRF rank"
    )


def test_rank_fusion_does_not_use_raw_scores():
    """The fusion function must not reference raw score fields.

    Only ranks (positional indices) drive the fused score.
    """
    source = inspect.getsource(rank_preserving_fusion)
    # Must not reference .get("score") or ["score"] or .score
    assert '"score"' not in source, (
        "rank_preserving_fusion must not use raw scores"
    )
    assert "'score'" not in source, (
        "rank_preserving_fusion must not use raw scores"
    )

    source_rrs = inspect.getsource(reciprocal_rank_score)
    assert '"score"' not in source_rrs, (
        "reciprocal_rank_score must not use raw scores"
    )


def test_rank_fusion_uses_fixed_equal_weights():
    """The fusion must use fixed 1:1 equal weights with no weight parameter."""
    source = inspect.getsource(rank_preserving_fusion)
    # Must not accept a weight parameter
    assert "weight" not in source.lower(), (
        "Fusion must not accept a weight parameter"
    )
    # Must not reference 0.2, 0.4, 0.6, 0.8 weights
    for w in ("0.2", "0.4", "0.6", "0.8"):
        assert w not in source, f"Fusion must not use weight {w}"

    # The score must be a simple sum of two reciprocal_rank_score calls
    assert source.count("reciprocal_rank_score(") == 2, (
        "Fusion must use exactly two reciprocal_rank_score calls (equal weight)"
    )


def test_reciprocal_rank_score_handles_missing_rank():
    """When rank is None, missing_rank must be used."""
    score_present = reciprocal_rank_score(1, k=60, missing_rank=100)
    score_missing = reciprocal_rank_score(None, k=60, missing_rank=100)

    assert score_present == 1.0 / (60 + 1)
    assert score_missing == 1.0 / (60 + 100)
    assert score_present > score_missing


def test_candidate_key_for_table_cell_and_row():
    """table_cell must map to parent_row identity; table_row uses its own id."""
    cell = _make_candidate("cell_1", block_type="table_cell", parent_id="row_1")
    row = _make_candidate("row_1", block_type="table_row")

    assert candidate_key(cell) == "table_row:doc_a:row_1"
    assert candidate_key(row) == "table_row:doc_a:row_1"
    assert candidate_key(cell) == candidate_key(row), (
        "table_cell and its parent table_row must share the same key"
    )


def test_candidate_key_for_text_block():
    """text blocks use block:{document_id}:{evidence_id}."""
    text = _make_candidate("ev_1", block_type="text")
    assert candidate_key(text) == "block:doc_a:ev_1"


def test_candidate_key_table_cell_without_parent_raises():
    """table_cell without parent_id must raise."""
    import pytest

    cell = {
        "candidate_id": "cell_1",
        "evidence_id": "cell_1",
        "document_id": "doc_a",
        "page": 1,
        "block_type": "table_cell",
        "parent_id": None,
    }
    with pytest.raises(ValueError, match="table_cell has no parent row"):
        candidate_key(cell)


def test_fusion_candidate_not_in_reranker_gets_missing_rank():
    """Candidates in RRF but not in reranker output get a lower score."""
    rrf = _make_rrf_candidates(5)
    # Reranker only outputs the first candidate
    reranked = [rrf[0]]

    fused = rank_preserving_fusion(
        rrf_candidates=rrf, reranked_candidates=reranked
    )

    # The first candidate (in both RRF and reranker) should rank highest
    assert candidate_key(fused[0]) == candidate_key(rrf[0]), (
        "Candidate present in both lists must rank highest"
    )


def test_fusion_preserves_order_when_reranker_agrees_with_rrf():
    """When reranker agrees with RRF order, fusion preserves that order."""
    rrf = _make_rrf_candidates(5)
    reranked = list(rrf)  # same order

    fused = rank_preserving_fusion(
        rrf_candidates=rrf, reranked_candidates=reranked
    )

    fused_keys = [candidate_key(c) for c in fused]
    rrf_keys = [candidate_key(c) for c in rrf]
    assert fused_keys == rrf_keys
