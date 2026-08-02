from src.evaluation.nf_eval_04 import (
    CandidateRecallFailureStage,
    RecallGate,
    VerifiedCandidateEquivalence,
    candidate_in_scope,
    choose_next_gate,
    classify_first_recall_failure,
    classify_index_presence,
    rank_bucket,
    require_verified_equivalence,
    source_coverage,
)


def test_index_presence_is_distinct_from_query_recall():
    assert classify_index_presence(present_in_bm25_index=True, present_in_dense_index=False) == "bm25_only_indexed"
    assert classify_first_recall_failure(
        identity_valid=True,
        present_in_bm25_index=True,
        present_in_dense_index=False,
        bm25_rank=None,
        dense_rank=None,
        bm25_production_limit=40,
        dense_production_limit=40,
        entered_production_union=False,
        entered_production_rrf=False,
    ) == CandidateRecallFailureStage.NOT_RETRIEVED_BY_BM25_TOP200


def test_rank_buckets_and_window_truncation():
    assert [rank_bucket(item) for item in (1, 20, 21, 40, 41, 100, 101, 200, None)] == [
        "top20", "top20", "21_40", "21_40", "41_100", "41_100", "101_200", "101_200", "not_retrieved"
    ]
    assert classify_first_recall_failure(
        identity_valid=True,
        present_in_bm25_index=True,
        present_in_dense_index=False,
        bm25_rank=73,
        dense_rank=None,
        bm25_production_limit=40,
        dense_production_limit=40,
        entered_production_union=False,
        entered_production_rrf=False,
    ) == CandidateRecallFailureStage.BM25_WINDOW_TRUNCATION


def test_dense_window_requires_rank_above_actual_limit():
    assert classify_first_recall_failure(
        identity_valid=True,
        present_in_bm25_index=False,
        present_in_dense_index=True,
        bm25_rank=None,
        dense_rank=20,
        bm25_production_limit=40,
        dense_production_limit=20,
        entered_production_union=True,
        entered_production_rrf=True,
    ) == CandidateRecallFailureStage.ENTERED_RRF_POOL


def test_parent_child_mapping_requires_verified_relation():
    assert not require_verified_equivalence(None, relation="row_to_chunk")
    assert not require_verified_equivalence(
        VerifiedCandidateEquivalence("g", "r", "same_page", "heuristic"),
        relation="row_to_chunk",
    )
    assert require_verified_equivalence(
        VerifiedCandidateEquivalence("g", "r", "row_to_chunk", "golden_binding"),
        relation="row_to_chunk",
    )


def test_same_page_is_not_identity_equivalence():
    assert candidate_in_scope({"document_id": "legacy"}, {"benchmark"}) is False


def test_source_coverage_and_gate_use_unique_cases():
    assert source_coverage(["a", "b"], ["a"]) == "partial"
    assert source_coverage(["a", "b"], ["a", "b"]) == "all"
    result = choose_next_gate(
        terminology_cases=12,
        window_cases=15,
        dense_coverage_cases=15,
        parent_child_cases=20,
        rrf_fusion_cases=20,
    )
    assert result["selected_gate"] == RecallGate.PARENT_CHILD.value
    assert result["optimization_allowed"] is False


def test_no_production_answer_generation_in_helper_layer():
    # Pure attribution helpers have no callable model/retrieval entry point.
    assert not hasattr(classify_first_recall_failure, "model")
