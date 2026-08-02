from src.evaluation.nf_eval_04 import (
    CandidateRecallFailureStage,
    classify_first_recall_failure,
    classify_index_presence,
)


def test_index_presence_is_distinct_from_query_recall():
    assert classify_index_presence(
        present_in_bm25_index=True,
        present_in_dense_index=False,
    ) == "bm25_only_indexed"
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


def test_missing_from_both_indexes_is_not_a_retriever_miss():
    assert classify_first_recall_failure(
        identity_valid=True,
        present_in_bm25_index=False,
        present_in_dense_index=False,
        bm25_rank=None,
        dense_rank=None,
        bm25_production_limit=40,
        dense_production_limit=40,
        entered_production_union=False,
        entered_production_rrf=False,
    ) == CandidateRecallFailureStage.MISSING_FROM_BOTH_INDEXES
