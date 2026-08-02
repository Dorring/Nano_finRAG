from src.evaluation.nf_eval_04 import RecallGate, choose_next_gate


def test_gate_uses_unique_case_count_and_selects_one_direction():
    result = choose_next_gate(
        terminology_cases=0,
        window_cases=13,
        dense_coverage_cases=53,
        parent_child_cases=0,
        rrf_fusion_cases=0,
    )
    assert result["selected_gate"] == RecallGate.DENSE_COVERAGE.value
    assert result["optimization_allowed"] is False
    assert len(result["passing_gates"]) == 2


def test_gate_stops_when_no_concentrated_bottleneck_exists():
    result = choose_next_gate(
        terminology_cases=1,
        window_cases=2,
        dense_coverage_cases=3,
        parent_child_cases=1,
        rrf_fusion_cases=0,
    )
    assert result["selected_gate"] == RecallGate.NO_CONCENTRATED_BOTTLENECK.value
