from decimal import Decimal

from src.domain.financial_fact import BoundOperand, FinancialFact
from src.finance.structured_operand_binding import adapt_bound_operands


def _bound(role, value):
    fact = FinancialFact(
        role,
        Decimal(value),
        "FY2025",
        None,
        None,
        "million",
        role,
        role,
        "doc",
        1,
        role,
        "FY2025",
        None,
        "test",
        1.0,
    )
    return BoundOperand(role, fact, Decimal(value))


def test_difference_preserves_question_direction():
    operands = adapt_bound_operands(
        (_bound("minuend", "10"), _bound("subtrahend", "3")), "difference"
    )
    assert [operand.value for operand in operands] == [Decimal("10"), Decimal("3")]


def test_generic_difference_no_longer_returns_empty():
    operands = adapt_bound_operands(
        (_bound("minuend", "10"), _bound("subtrahend", "3")), "difference"
    )
    assert len(operands) == 2
