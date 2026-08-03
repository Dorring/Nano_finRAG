from src.evaluation.nf_opt_02 import protected_residual_gate


def _gate(**overrides):
    values = dict(case_count=64, source_count=80, overlap_count=0, identity_conflict_count=0, out_of_scope_count=0, base_missing_count=0, base_order_changed_count=0, dense_gold_regressions=0, union_source_regressions=0, rrf_full_source_regressions=0, rrf_top40_source_regressions=0, rrf_full_all_gold_regressions=0, rrf_top40_all_gold_regressions=0, union_source_hits=28, rrf_full_source_hits=36, rrf_top40_source_hits=30, rrf_top40_all_gold_cases=24, dense_latency_ratio=0.1, retrieval_latency_ratio=0.1)
    values.update(overrides)
    return protected_residual_gate(**values)


def test_current_gold_sources_cannot_regress():
    assert _gate(dense_gold_regressions=1)["passed"] is False


def test_all_gold_case_regression_fails_gate():
    assert _gate(rrf_top40_all_gold_regressions=1)["passed"] is False
