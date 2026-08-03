from decimal import Decimal

from src.domain.financial_fact import FinancialFact, OperandBindingSpec
from src.finance.structured_operand_binding import bind_operands


def _fact(period):
    return FinancialFact(
        "Total revenue",
        Decimal("100"),
        period,
        None,
        None,
        "million",
        "key",
        "chunk",
        "doc",
        1,
        "Total revenue",
        period,
        None,
        "test",
        1.0,
    )


def test_growth_binds_same_metric_two_periods():
    result = bind_operands(
        (
            OperandBindingSpec("previous", "total revenue", "FY2024", None, None, None),
            OperandBindingSpec("current", "total revenue", "FY2025", None, None, None),
        ),
        (_fact("FY2024"), _fact("FY2025")),
    )
    assert result.success


def test_growth_rejects_wrong_period():
    result = bind_operands(
        (OperandBindingSpec("previous", "total revenue", "FY2023", None, None, None),),
        (_fact("FY2024"),),
    )
    assert result.block_reason == "OPERAND_MISSING"
