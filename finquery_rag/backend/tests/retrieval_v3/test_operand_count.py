from src.retrieval_v3.query_router import route_question


def test_growth_requires_two_operands() -> None:
    profile = route_question("What was the growth rate of revenue from FY2024 to FY2025?")
    assert profile.expected_operand_count == 2
    assert profile.requires_multiple_sources is True


def test_difference_requires_two_operands() -> None:
    profile = route_question("Which was higher: automotive revenue or energy revenue, and by how much?")
    assert profile.task_type == "calculation_multi_operand"
    assert profile.expected_operand_count == 2
