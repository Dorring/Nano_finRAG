from src.evaluation.nf_opt_02_r1 import select_smallest_passing_variant, transfer_gate

def _gate(**overrides):
    values = dict(completeness_passed=True, lineage_passed=True, model_calls=0, answer_generation_calls=0, reranker20_source_gain=8, reranker10_source_gain=6, final5_source_gain=4, reranker20_all_gold_gain=5, final_all_gold_gain=3, reranker20_source_regression=0, reranker20_all_gold_regression=0, final_source_regression=0, final_all_gold_regression=0, latency_gate_passed=True)
    values.update(overrides)
    return transfer_gate(**values)

def test_smallest_passing_residual_budget_is_selected():
    assert select_smallest_passing_variant({"C10": {"passed": False}, "C20": {"passed": True}, "C40": {"passed": True}}) == "C20"

def test_latency_failure_blocks_production_switch():
    result = _gate(latency_gate_passed=False)
    assert result["passed"] is False
    assert result["decision"] == "protected_residual_transfer_validated_latency_blocked"
    assert result["production_switch_allowed"] is False

def test_transfer_gate_requires_zero_all_gold_regression():
    result = _gate(reranker20_all_gold_regression=1)
    assert result["passed"] is False
    assert result["regression_passed"] is False
