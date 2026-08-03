from src.evaluation.nf_opt_04 import final_budget_decision, select_smallest_passing_variant

def test_smallest_passing_budget_is_selected():
    gates = {"F8": {"passed": True}, "FT8": {"passed": True}, "F10": {"passed": True}, "FT10": {"passed": True}}
    assert select_smallest_passing_variant(gates) == "F8"

def test_production_default_is_unchanged_by_gate():
    gates = {"F8": {"passed": False}, "FT8": {"passed": False}, "F10": {"passed": False}, "FT10": {"passed": False}}
    assert select_smallest_passing_variant(gates) is None


def test_insufficient_gain_is_not_labeled_a_context_conflict():
    assert final_budget_decision(
        selected_variant=None,
        context_quality_blocked=False,
    ) == (
        "final_budget_gain_insufficient",
        "stop_context_expansion_and_start_calculation_route",
    )
