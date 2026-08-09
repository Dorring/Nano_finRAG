from __future__ import annotations

from src.pdf_retrieval_v4.field_family_normalizer import (
    RRF_K,
    fuse_field_family,
    fuse_flat_h0,
    fuse_hierarchical_structured,
)


def _hits(prefix: str, count: int = 4):
    return [{"candidate_key": f"{prefix}{i}", "rank": i + 1} for i in range(count)]


def test_field_family_uses_only_active_fields() -> None:
    hits = {"structured_metric_bm25": _hits("m"), "structured_axis_bm25": []}
    family = fuse_field_family(hits)
    assert family
    assert all(set(item["lane_ranks"]) == {"structured_metric_bm25"} for item in family)


def test_top_level_has_three_votes_and_rank_not_score() -> None:
    hits = {
        "candidate_structured_bm25": _hits("x"),
        "candidate_structured_dense": _hits("x"),
        "structured_metric_bm25": _hits("x"),
        "structured_axis_bm25": _hits("x"),
    }
    family, result = fuse_hierarchical_structured(hits)
    assert RRF_K == 60
    assert family
    assert set(result[0]["lane_ranks"]) == {
        "candidate_structured_bm25",
        "candidate_structured_dense",
        "structured_field_family",
    }


def test_candidate_dedup_and_determinism() -> None:
    hits = {
        "candidate_structured_bm25": _hits("x"),
        "candidate_structured_dense": _hits("x"),
        "structured_metric_bm25": _hits("x"),
    }
    assert fuse_flat_h0(hits) == fuse_flat_h0(hits)
    assert len({item["candidate_key"] for item in fuse_flat_h0(hits)}) == 4


def test_field_count_does_not_change_top_level_vote_count() -> None:
    base = {
        "candidate_structured_bm25": _hits("x"),
        "candidate_structured_dense": _hits("x"),
        "structured_metric_bm25": _hits("x"),
    }
    _, one = fuse_hierarchical_structured(base)
    _, four = fuse_hierarchical_structured(
        {
            **base,
            "structured_axis_bm25": _hits("x"),
            "structured_context_bm25": _hits("x"),
            "structured_evidence_bm25": _hits("x"),
        }
    )
    assert len(one[0]["lane_ranks"]) == len(four[0]["lane_ranks"]) == 3
