from decimal import Decimal

from src.domain.financial_fact import FinancialFact, OperandBindingSpec
from src.finance.structured_operand_binding import bind_operands


def _fact(metric, value):
    return FinancialFact(
        metric,
        Decimal(value),
        "FY2025",
        None,
        None,
        "million",
        metric,
        metric,
        "doc",
        1,
        metric,
        "FY2025",
        None,
        "test",
        1.0,
    )


def test_percentage_share_binds_specific_part_and_total():
    result = bind_operands(
        (
            OperandBindingSpec(
                "part", "EMEA third party revenue", "FY2025", None, None, None
            ),
            OperandBindingSpec("total", "total revenue", "FY2025", None, None, None),
        ),
        (_fact("EMEA third party revenue", "10"), _fact("Total revenue", "100")),
    )
    assert result.success


def test_percentage_share_rejects_wrong_segment_measure():
    result = bind_operands(
        (
            OperandBindingSpec(
                "part", "EMEA third party revenue", "FY2025", None, None, None
            ),
        ),
        (_fact("EMEA total segment revenue", "10"),),
    )
    assert result.block_reason == "OPERAND_MISSING"
