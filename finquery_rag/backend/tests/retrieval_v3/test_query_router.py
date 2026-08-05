from src.retrieval_v3.query_router import route_question


def test_single_table_fact_routes_without_calculation() -> None:
    profile = route_question("What was total net sales in FY2025?", document_scope=("aapl_fy2025",))
    assert profile.task_type == "table_single_fact"
    assert profile.issuer == "aapl_fy2025"
    assert profile.operation is None


def test_narrative_signal_does_not_override_calculation() -> None:
    profile = route_question("Explain the growth rate of revenue from FY2024 to FY2025.")
    assert profile.task_type == "calculation_multi_operand"
    assert profile.operation == "growth_rate"
