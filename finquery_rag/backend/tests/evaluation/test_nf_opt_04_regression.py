from src.evaluation.nf_opt_04 import final_budget_gate

def gate(**overrides):
    values = dict(integrity_passed=True, source_hit_count=20, all_gold_case_count=15, multi_evidence_all_gold_count=4, new_source_count=7, new_all_gold_case_count=5, source_regression_count=0, all_gold_regression_count=0, multi_evidence_regression_count=0, conflicting_period_case_increase=0, conflicting_value_case_increase=0, duplicate_case_rate=0.0, context_token_p95=100.0, total_latency_ratio=0.0)
    values.update(overrides)
    return final_budget_gate(**values)

def test_source_regression_fails_closed():
    assert gate(source_regression_count=1)["passed"] is False

def test_all_gold_regression_fails_closed():
    assert gate(all_gold_regression_count=1)["passed"] is False
