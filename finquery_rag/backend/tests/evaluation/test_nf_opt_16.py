import numpy as np
import pytest

from src.evaluation.nf_opt_16 import (
    assert_query_has_no_expected_fields,
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
