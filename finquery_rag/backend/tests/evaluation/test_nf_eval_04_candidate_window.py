from src.evaluation.nf_eval_04 import (
    CandidateRecallFailureStage,
    classify_first_recall_failure,
)


def test_window_truncation_requires_gold_below_production_limit():
    kwargs = dict(
        identity_valid=True,
        present_in_bm25_index=True,
        present_in_dense_index=False,
        dense_rank=None,
        dense_production_limit=40,
        entered_production_union=False,
        entered_production_rrf=False,
    )
    assert classify_first_recall_failure(
        **kwargs, bm25_rank=41, bm25_production_limit=40
    ) == CandidateRecallFailureStage.BM25_WINDOW_TRUNCATION
    assert classify_first_recall_failure(
        **kwargs, bm25_rank=40, bm25_production_limit=40
    ) != CandidateRecallFailureStage.BM25_WINDOW_TRUNCATION
