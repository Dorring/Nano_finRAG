import numpy as np
import pytest

from src.evaluation.nf_opt_16 import (
    assert_query_has_no_expected_fields,
    build_sparse_inverted_index,
    rank_scores,
    sparse_rank,
    stable_smoke_sample,
    validate_model_output,
)


def test_bge_m3_sparse_and_multi_vector_output_is_required():
    result = validate_model_output(
        {
            "dense_vecs": np.zeros((2, 4)),
            "lexical_weights": [{"12": 0.4}, {"15": 0.2, "18": 0.1}],
            "colbert_vecs": [np.zeros((3, 4)), np.zeros((2, 4))],
        },
        expected_rows=2,
    )
    assert result["sparse_nonzero_count"] == 3
    assert result["colbert_token_count"] == 5
    assert result["dense_output_ignored"] is True


def test_model_query_cannot_carry_expected_fields():
    with pytest.raises(ValueError, match="excluded expected fields"):
        assert_query_has_no_expected_fields({"question": "What was revenue?", "expected_answer": "not allowed"})


def test_smoke_sample_is_content_only_and_deterministic():
    rows = [
        {"doc_name": "b.pdf", "doc_id": "b", "content": "B"},
        {"doc_name": "a.pdf", "doc_id": "a2", "content": "A2"},
        {"doc_name": "a.pdf", "doc_id": "a1", "content": "A1"},
    ]
    assert [item["doc_id"] for item in stable_smoke_sample(rows, limit=2)] == ["a1", "a2"]


def test_sparse_rank_uses_weighted_overlap_and_stable_candidate_key_tie_break():
    index = build_sparse_inverted_index([{"a": 0.4, "b": 0.2}, {"a": 0.4}, {"b": 0.8}])
    assert sparse_rank(
        query_weights={"a": 1.0, "b": 1.0},
        inverted_index=index,
        candidate_keys=["candidate:z", "candidate:a", "candidate:b"],
        limit=3,
    ) == [(2, 0.8), (0, 0.6000000000000001), (1, 0.4)]


def test_sparse_rank_never_returns_non_overlapping_candidate():
    index = build_sparse_inverted_index([{"known": 0.5}, {"other": 0.8}])
    assert sparse_rank(
        query_weights={"known": 1.0},
        inverted_index=index,
        candidate_keys=["candidate:a", "candidate:b"],
        limit=10,
    ) == [(0, 0.5)]


def test_late_interaction_scores_have_stable_identity_tie_break():
    assert rank_scores([0.8, 0.8, 0.9], ["candidate:z", "candidate:a", "candidate:b"], limit=3) == [
        (2, 0.9),
        (1, 0.8),
        (0, 0.8),
    ]
