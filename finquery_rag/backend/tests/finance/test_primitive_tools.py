"""Tests for migrated financial primitives (Phase 3 Commit 3).

Verifies:
- All original primitives are accessible at ``src.finance.primitive_tools``.
- ``src.services.financial_tools`` re-exports them (backward compat).
- The 5 new primitives (difference, average_values, gross_margin,
  net_margin, debt_ratio) produce correct Decimal results.
- NaN/Infinity are rejected by Decimal construction.
- ``src.finance`` does not import from ``src.services`` (layer purity).
"""

import inspect
import os
import sys
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.finance.primitive_tools import (
    average_values,
    debt_ratio,
    difference,
    gross_margin,
    net_margin,
    parse_financial_number,
)


# ---------------------------------------------------------------------------
# Migration / backward-compat
# ---------------------------------------------------------------------------


class TestMigration:
    """Original primitives are accessible at the new canonical location."""

    @pytest.mark.parametrize(
        "fn_name",
        [
            "ToolResult",
            "parse_financial_number",
            "growth_rate",
            "percentage_share",
            "sum_values",
            "verify_sum",
            "convert_scale",
            "format_ratio_percent",
        ],
    )
    def test_canonical_location_has_original_primitives(self, fn_name):
        import src.finance.primitive_tools as pt

        assert hasattr(pt, fn_name)

    @pytest.mark.parametrize(
        "fn_name",
        [
            "difference",
            "average_values",
            "gross_margin",
            "net_margin",
            "debt_ratio",
        ],
    )
    def test_new_primitives_exist(self, fn_name):
        import src.finance.primitive_tools as pt

        assert hasattr(pt, fn_name)

    @pytest.mark.parametrize(
        "fn_name",
        [
            "ToolResult",
            "parse_financial_number",
            "growth_rate",
            "percentage_share",
            "sum_values",
            "verify_sum",
            "convert_scale",
            "format_ratio_percent",
            "difference",
            "average_values",
            "gross_margin",
            "net_margin",
            "debt_ratio",
        ],
    )
    def test_backward_compat_shim_re_exports(self, fn_name):
        """``src.services.financial_tools`` must still expose every primitive."""
        import src.services.financial_tools as legacy

        assert hasattr(legacy, fn_name), f"legacy shim missing {fn_name}"

    def test_backward_compat_shim_re_exports_scale_factors(self):
        """Architecture test reads ``_SCALE_FACTORS`` from the legacy module."""
        import src.services.financial_tools as legacy

        assert legacy._SCALE_FACTORS is not None
        assert "million" in legacy._SCALE_FACTORS

    def test_legacy_and_canonical_are_same_object(self):
        """The shim must re-export the *same* function objects, not copies."""
        import src.services.financial_tools as legacy
        import src.finance.primitive_tools as pt

        assert legacy.growth_rate is pt.growth_rate
        assert legacy.parse_financial_number is pt.parse_financial_number
        assert legacy.ToolResult is pt.ToolResult


# ---------------------------------------------------------------------------
# New primitive: difference
# ---------------------------------------------------------------------------


class TestDifference:
    def test_basic_subtraction(self):
        r = difference("110", "100")
        assert r.ok
        assert r.points_value == Decimal("10")

    def test_negative_result(self):
        r = difference("90", "110")
        assert r.ok
        assert r.points_value == Decimal("-20")

    def test_precision(self):
        r = difference("1.123456", "0", precision=2)
        assert r.ok
        assert r.points_value == Decimal("1.12")

    def test_rejects_invalid(self):
        r = difference("abc", "100")
        assert not r.ok
        assert r.error is not None


# ---------------------------------------------------------------------------
# New primitive: average_values
# ---------------------------------------------------------------------------


class TestAverageValues:
    def test_basic_average(self):
        r = average_values(["10", "20", "30"])
        assert r.ok
        assert r.points_value == Decimal("20")

    def test_precision(self):
        r = average_values(["1", "2"], precision=4)
        assert r.ok
        assert r.points_value == Decimal("1.5000")

    def test_empty_list_returns_error(self):
        r = average_values([])
        assert not r.ok
        assert "empty" in r.error.lower()

    def test_rejects_invalid_member(self):
        r = average_values(["10", "abc"])
        assert not r.ok


# ---------------------------------------------------------------------------
# New primitive: gross_margin
# ---------------------------------------------------------------------------


class TestGrossMargin:
    def test_basic_margin(self):
        r = gross_margin("1000", "600")
        assert r.ok
        assert r.points_value == Decimal("0.4")

    def test_full_precision(self):
        r = gross_margin("1000", "333", precision=4)
        assert r.ok
        assert r.points_value == Decimal("0.6670")

    def test_zero_revenue_returns_error(self):
        r = gross_margin("0", "100")
        assert not r.ok
        assert "zero revenue" in r.error.lower()

    def test_negative_cogs_yields_margin_above_one(self):
        """If cogs is negative (refund), margin can exceed 1 — that's valid arithmetic."""
        r = gross_margin("1000", "-200")
        assert r.ok
        assert r.points_value == Decimal("1.2")


# ---------------------------------------------------------------------------
# New primitive: net_margin
# ---------------------------------------------------------------------------


class TestNetMargin:
    def test_basic_margin(self):
        r = net_margin("1000", "80")
        assert r.ok
        assert r.points_value == Decimal("0.08")

    def test_zero_revenue_returns_error(self):
        r = net_margin("0", "10")
        assert not r.ok

    def test_negative_net_income(self):
        r = net_margin("1000", "-50")
        assert r.ok
        assert r.points_value == Decimal("-0.05")


# ---------------------------------------------------------------------------
# New primitive: debt_ratio
# ---------------------------------------------------------------------------


class TestDebtRatio:
    def test_basic_ratio(self):
        r = debt_ratio("600", "1000")
        assert r.ok
        assert r.points_value == Decimal("0.6")

    def test_zero_assets_returns_error(self):
        r = debt_ratio("100", "0")
        assert not r.ok
        assert "zero total assets" in r.error.lower()

    def test_liabilities_exceed_assets(self):
        r = debt_ratio("1200", "1000")
        assert r.ok
        assert r.points_value == Decimal("1.2")


# ---------------------------------------------------------------------------
# NaN / Infinity rejection
# ---------------------------------------------------------------------------


class TestNaNRejection:
    """Decimal construction must reject NaN/Infinity inputs."""

    def test_nan_string_rejected(self):
        r = parse_financial_number("NaN")
        assert not r.ok

    def test_infinity_string_rejected(self):
        r = parse_financial_number("Infinity")
        assert not r.ok


# ---------------------------------------------------------------------------
# Finance-layer purity
# ---------------------------------------------------------------------------


class TestFinanceLayerPurity:
    """``src.finance`` must not import from ``src.services`` or ``src.application``."""

    def test_primitive_tools_does_not_import_services(self):
        import src.finance.primitive_tools as pt

        source = inspect.getsource(pt)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "from src.services" not in stripped, f"forbidden import: {stripped}"
            assert "from src.application" not in stripped, (
                f"forbidden import: {stripped}"
            )
            assert "import src.services" not in stripped, (
                f"forbidden import: {stripped}"
            )
            assert "import src.application" not in stripped, (
                f"forbidden import: {stripped}"
            )

    def test_finance_init_does_not_import_services(self):
        import src.finance as fin

        source = inspect.getsource(fin)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "from src.services" not in stripped, f"forbidden import: {stripped}"
            assert "from src.application" not in stripped, (
                f"forbidden import: {stripped}"
            )
