from src.evaluation.nf_opt_01 import dense_coverage_gate


def test_gate_requires_source_and_case_improvement():
    result = dense_coverage_gate(
        shadow_gold_identity_presence=80,
        unsupported_candidate_count=0,
        out_of_scope_candidate_count=0,
        dense_source_gain_at_200=10,
        production_union_source_gain=8,
        rrf_source_gain_at_40=5,
        rrf_all_case_gain=5,
        dense_regressed_sources=0,
        rrf_regressed_sources_at_40=0,
        rrf_regressed_all_cases=0,
        latency_increase_ratio=0.1,
    )
    assert result["passed"] is True
    assert result["production_switch_allowed"] is False


def test_average_gain_cannot_hide_case_regression():
    result = dense_coverage_gate(
        shadow_gold_identity_presence=80,
        unsupported_candidate_count=0,
        out_of_scope_candidate_count=0,
        dense_source_gain_at_200=20,
        production_union_source_gain=12,
        rrf_source_gain_at_40=8,
        rrf_all_case_gain=6,
        dense_regressed_sources=2,
        rrf_regressed_sources_at_40=0,
        rrf_regressed_all_cases=0,
        latency_increase_ratio=0.1,
    )
    assert result["passed"] is False
    assert result["next_gate"] == "stop_and_analyze_regression"


def test_positive_rrf_gain_recommends_window_diagnostic_without_switch():
    result = dense_coverage_gate(
        shadow_gold_identity_presence=80,
        unsupported_candidate_count=0,
        out_of_scope_candidate_count=0,
        dense_source_gain_at_200=38,
        production_union_source_gain=5,
        rrf_source_gain_at_40=13,
        rrf_all_case_gain=12,
        dense_regressed_sources=1,
        rrf_regressed_sources_at_40=0,
        rrf_regressed_all_cases=0,
        latency_increase_ratio=0.12,
    )
    assert result["passed"] is False
    assert result["next_gate"] == "candidate_window_expansion"
    assert result["production_switch_allowed"] is False
