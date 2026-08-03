from src.evaluation.nf_opt_03 import window_gate


def test_total_gain_cannot_hide_source_regression():
    gate = window_gate(
        complete=True, prefix_passed=True, lineage_passed=True, scope_passed=True,
        model_calls=0, answer_generation_calls=0,
        bm25_source_gain=10, bm25_all_gold_gain=6, rrf40_source_gain=6,
        reranker20_source_gain=5, final5_source_gain=3, final_all_gold_gain=2,
        bm25_source_regression=0, rrf40_source_regression=2,
        reranker20_source_regression=0, final5_source_regression=0,
        rrf_all_gold_regression=0, reranker_all_gold_regression=0,
        final_all_gold_regression=0, latency_passed=True,
    )
    assert gate["regression_passed"] is False
    assert gate["passed"] is False


def test_all_gold_regression_fails_gate():
    gate = window_gate(
        complete=True, prefix_passed=True, lineage_passed=True, scope_passed=True,
        model_calls=0, answer_generation_calls=0,
        bm25_source_gain=10, bm25_all_gold_gain=6, rrf40_source_gain=6,
        reranker20_source_gain=5, final5_source_gain=3, final_all_gold_gain=2,
        bm25_source_regression=0, rrf40_source_regression=0,
        reranker20_source_regression=0, final5_source_regression=0,
        rrf_all_gold_regression=1, reranker_all_gold_regression=0,
        final_all_gold_regression=0, latency_passed=True,
    )
    assert gate["passed"] is False
