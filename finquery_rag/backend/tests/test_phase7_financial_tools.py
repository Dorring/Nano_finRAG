"""Phase 7A tests: deterministic financial calculation tools.

Every assertion names the reading it means.  ``points_value`` is the number as
the input stated it -- 12.5 for ``12.5%`` -- and ``ratio_value`` is the
multiplicative form, 0.125.  They differ for a percentage and only for a
percentage, and a test that reaches for the wrong one is the defect this naming
exists to make visible.
"""
import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.financial_tools import (
    convert_scale,
    format_ratio_percent,
    growth_rate,
    parse_financial_number,
    percentage_share,
    sum_values,
    verify_sum,
)


def test_parse_plain_and_comma_number():
    assert parse_financial_number("1,234.50").points_value == Decimal("1234.50")


def test_parse_accounting_negative():
    assert parse_financial_number("(1,234)").points_value == Decimal("-1234")


def test_a_percentage_keeps_both_of_its_readings():
    """``12.5%`` is 12.5 points and the factor 0.125, and both are returned.

    The old contract returned only the second, under a field named ``value``, so
    a caller that meant the first had no way to say so -- and the gold arithmetic
    for a percentage share means the first.
    """
    result = parse_financial_number("12.5%")

    assert result.ok is True
    assert result.points_value == Decimal("12.5")
    assert result.ratio_value == Decimal("0.125")
    assert result.details["is_percent"] is True


def test_whitespace_does_not_change_the_quantity():
    """The defect this contract exists to remove.

    ``12.5%`` and ``12.5 %`` are one quantity written two ways.  Detection used
    to depend on the sign being written against the digits, so these came out
    100x apart -- and the fact store writes both forms, 305 glued and 222 spaced.
    """
    glued = parse_financial_number("12.5%")
    spaced = parse_financial_number("12.5 %")

    assert (glued.points_value, glued.ratio_value) == (
        spaced.points_value,
        spaced.ratio_value,
    )
    assert (glued.points_value, glued.ratio_value) == (
        Decimal("12.5"),
        Decimal("0.125"),
    )


def test_an_accounting_negative_percentage_is_read_at_all():
    """``(0.3)%`` -- a negative rate as a filing writes it.

    It has no digits-adjacent sign at any position the old grammar looked, so it
    did not parse.  The three ways of writing one percentage now agree.
    """
    result = parse_financial_number("(0.3)%")

    assert result.ok is True
    assert result.points_value == Decimal("-0.3")
    assert result.ratio_value == Decimal("-0.003")


def test_a_percentage_is_not_an_absolute_quantity():
    """Same digits, different kind -- the `$5` against `5 %` case."""
    assert (
        parse_financial_number("12.5%").representation
        != parse_financial_number("12.5").representation
    )


def test_a_ratio_is_not_its_percentage_form():
    """0.125 and 12.5% are convertible and are not the same claim here."""
    assert parse_financial_number("0.125").ratio_value == Decimal("0.125")
    assert parse_financial_number("0.125").points_value == Decimal("0.125")
    assert (
        parse_financial_number("0.125").representation
        != parse_financial_number("12.5%").representation
    )


def test_parse_scale_words():
    assert parse_financial_number("$1.2 million").points_value == Decimal("1200000.0")
    assert parse_financial_number("3亿元").points_value == Decimal("300000000")


def test_parse_unknown_returns_error():
    result = parse_financial_number("not a number")
    assert result.ok is False
    assert "Cannot parse" in result.error


def test_growth_rate():
    result = growth_rate("120", "100", precision=4)
    assert result.ok is True
    assert result.points_value == Decimal("0.2000")
    assert result.unit == "ratio"


def test_growth_rate_zero_previous_error():
    result = growth_rate("120", "0")
    assert result.ok is False
    assert "zero previous" in result.error


def test_percentage_share():
    result = percentage_share("25", "200", precision=4)
    assert result.points_value == Decimal("0.1250")


def test_a_percentage_share_reads_the_stated_number():
    """``21%`` of ``(486)`` is ``21 / -486``.

    The benchmark gold states -0.0432 for exactly this pair.  Reading the ratio
    form would answer -0.000432 -- the whole reason the two readings are named.
    """
    result = percentage_share("21%", "(486)", precision=4)

    assert result.ok is True
    assert result.points_value == Decimal("-0.0432")
    assert result.points_value != Decimal("-0.000432")


def test_sum_values_quantized():
    result = sum_values(["1.111", "2.222"], precision=2)
    assert result.points_value == Decimal("3.33")
    assert result.details["count"] == 2


def test_a_sum_of_percentages_is_a_percentage():
    result = sum_values(["30 %", "2 %"], precision=2)

    assert result.ok is True
    assert result.points_value == Decimal("32")
    assert result.ratio_value == Decimal("0.32")


def test_a_sum_of_mixed_kinds_is_refused():
    """A percentage plus an amount is not a quantity either reading describes."""
    result = sum_values(["32 %", "486"])

    assert result.ok is False
    assert "different kinds" in result.error


def test_verify_sum_passes_with_tolerance():
    result = verify_sum(["1.00", "2.00"], "3.004", tolerance="0.01")
    assert result.ok is True
    assert result.details["computed_total"] == "3.00"


def test_verify_sum_fails_outside_tolerance():
    result = verify_sum(["1.00", "2.00"], "3.50", tolerance="0.01")
    assert result.ok is False
    assert "does not match" in result.error


def test_verify_sum_against_another_kind_is_refused():
    """The comparison asks the shared contract before it compares.

    Without this the difference is computable and meaningless, and would be
    reported as a failed check against a number that was never comparable.
    """
    result = verify_sum(["32 %", "32 %"], "486")

    assert result.ok is False
    assert "another kind" in result.error


def test_convert_scale_million_to_billion():
    result = convert_scale("1500", "million", "billion", precision=4)
    assert result.ok is True
    assert result.points_value == Decimal("1.5000")
    assert result.unit == "billion"


def test_convert_scale_unknown_error():
    result = convert_scale("1", "weird", "million")
    assert result.ok is False
    assert "Unknown source scale" in result.error


def test_format_ratio_percent():
    result = format_ratio_percent("0.125", precision=2)
    assert result.ok is True
    assert result.points_value == Decimal("12.50")
    assert result.ratio_value == Decimal("0.125")
    assert result.unit == "percent"


def test_tool_result_to_dict_serializes_every_reading():
    result = growth_rate("120", "100", precision=2).to_dict()

    assert result["points_value"] == "0.20"
    assert result["ratio_value"] == "0.20"
    assert result["representation"] == "ratio"
    assert result["unit"] == "ratio"
    assert "value" not in result
