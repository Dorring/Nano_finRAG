from src.evaluation.nf_opt_03 import select_smallest_passing_window, window_gate


def test_smallest_passing_window_is_selected():
    gates = {"B80": {"passed": False}, "B120": {"passed": True}, "B200": {"passed": True}}
    assert select_smallest_passing_window(gates) == "B120"


def test_latency_failure_blocks_switch():
    gate = window_gate(
        complete=True, prefix_passed=True, lineage_passed=True, scope_passed=True,
        model_calls=0, answer_generation_calls=0,
        bm25_source_gain=10, bm25_all_gold_gain=6, rrf40_source_gain=6,
        reranker20_source_gain=5, final5_source_gain=3, final_all_gold_gain=2,
        bm25_source_regression=0, rrf40_source_regression=0,
        reranker20_source_regression=0, final5_source_regression=0,
        rrf_all_gold_regression=0, reranker_all_gold_regression=0,
        final_all_gold_regression=0, latency_passed=False,
    )
    assert gate["latency_gate_passed"] is False
    assert gate["passed"] is False


def test_production_default_is_unchanged():
    assert select_smallest_passing_window({"B80": {"passed": False}, "B120": {"passed": False}, "B200": {"passed": False}}) is None
