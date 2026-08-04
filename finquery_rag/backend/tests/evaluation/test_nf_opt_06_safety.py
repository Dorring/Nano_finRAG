from decimal import Decimal

from src.domain.financial_fact import FinancialFact, OperandBindingSpec
from src.finance.calculation_pipeline import CalculationPipeline
from src.finance.structured_operand_binding import bind_operands


def _fact(key):
    return FinancialFact(
        "Revenue",
        Decimal("100"),
        "FY2025",
        None,
        None,
        None,
        key,
        key,
        "doc",
        1,
        "Revenue",
        "FY2025",
        None,
        "test",
        1.0,
    )


def test_multiple_values_do_not_select_first_silently():
    result = bind_operands(
        (OperandBindingSpec("current", "revenue", "FY2025", None, None, None),),
        (_fact("a"), _fact("b")),
    )
    assert result.block_reason == "OPERAND_AMBIGUOUS"


def test_evidence_identity_is_required():
    missing = FinancialFact(
        "Revenue",
        Decimal("100"),
        "FY2025",
        None,
        None,
        None,
        None,
        "",
        "doc",
        1,
        "Revenue",
        "FY2025",
        None,
        "test",
        1.0,
    )
    result = bind_operands(
        (OperandBindingSpec("current", "revenue", "FY2025", None, None, None),),
        (missing,),
    )
    assert result.block_reason == "OPERAND_MISSING"


def test_shadow_flag_defaults_off():
    assert CalculationPipeline()._enable_structured_operand_binding is False


def test_scale_ambiguous_control_blocks():
    fact = FinancialFact(
        "Revenue",
        Decimal("100"),
        "FY2025",
        None,
        None,
        None,
        "key",
        "chunk",
        "doc",
        1,
        "Revenue",
        "FY2025",
        None,
        "test",
        1.0,
    )
    result = bind_operands(
        (OperandBindingSpec("current", "revenue", "FY2025", None, None, "million"),),
        (fact,),
    )
    assert result.block_reason == "OPERAND_MISSING"
