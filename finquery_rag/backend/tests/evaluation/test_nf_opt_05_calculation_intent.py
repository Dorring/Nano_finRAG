from src.domain.calculation import CalculationOperation
from src.finance.calculation_intent import detect_calculation_intent


def test_growth_rate_query_routes_to_calculation():
    result = detect_calculation_intent(
        "What was the year-over-year growth rate of revenue from FY2024 to FY2025?"
    )
    assert result.requires_calculation
    assert result.operation is CalculationOperation.GROWTH_RATE


def test_difference_query_routes_to_calculation():
    result = detect_calculation_intent(
        "Which segment was higher, A or B, and by how much?"
    )
    assert result.requires_calculation
    assert result.operation is CalculationOperation.DIFFERENCE


def test_percentage_share_routes_to_calculation():
    result = detect_calculation_intent(
        "What percentage of total revenue came from Services?"
    )
    assert result.requires_calculation
    assert result.operation is CalculationOperation.PERCENTAGE_SHARE


def test_sum_query_routes_to_calculation():
    result = detect_calculation_intent(
        "What was the combined total of product and services revenue?"
    )
    assert result.requires_calculation
    assert result.operation is CalculationOperation.SUM
