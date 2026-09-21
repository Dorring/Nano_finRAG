"""Tests for financial calculation domain objects (Phase 3 Commit 2)."""

import os
import sys
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.calculation import (
    NOT_APPLICABLE_RESULT,
    CalculationOperation,
    CalculationOperand,
    CalculationPlan,
    CalculationResult,
    CalculationStatus,
)


# ---------------------------------------------------------------------------
# CalculationOperation enum
# ---------------------------------------------------------------------------


#: The nine arithmetic operations as they stood before relational operations
#: were added.  Frozen as an explicit set rather than asserted by count: a count
#: moves for any addition, so it would be rewritten on every extension and would
#: stop saying anything.  This says which operations must not move, and is
#: imported by the other guards that want the same answer rather than restated
#: in each -- a second copy would be a second thing to forget to update.
LEGACY_ARITHMETIC_OPERATIONS = frozenset(
    {
        "difference",
        "growth_rate",
        "percentage_share",
        "sum",
        "average",
        "gross_margin",
        "net_margin",
        "debt_ratio",
        "scale_conversion",
    }
)


class TestCalculationOperation:
    def test_the_legacy_arithmetic_operations_are_still_present(self):
        assert {op.value for op in CalculationOperation} >= LEGACY_ARITHMETIC_OPERATIONS

    def test_relational_operations_were_added(self):
        """Their answer is a relation between operands, not a quantity."""

        assert CalculationOperation.COMPARISON.value == "comparison"
        assert CalculationOperation.RANKING.value == "ranking"

    @pytest.mark.parametrize(
        "name,value",
        [
            ("DIFFERENCE", "difference"),
            ("GROWTH_RATE", "growth_rate"),
            ("PERCENTAGE_SHARE", "percentage_share"),
            ("SUM", "sum"),
            ("AVERAGE", "average"),
            ("GROSS_MARGIN", "gross_margin"),
            ("NET_MARGIN", "net_margin"),
            ("DEBT_RATIO", "debt_ratio"),
            ("SCALE_CONVERSION", "scale_conversion"),
        ],
    )
    def test_member_names_and_values(self, name, value):
        op = CalculationOperation[name]
        assert op.value == value
        # str Enum also allows value-based lookup
        assert CalculationOperation(value) is op

    def test_roe_is_not_included(self):
        """ROE is excluded from v1 (requires average equity disambiguation)."""
        assert not hasattr(CalculationOperation, "ROE")

    def test_cagr_is_not_included(self):
        """CAGR is excluded from v1 (requires multi-period compounding)."""
        assert not hasattr(CalculationOperation, "CAGR")

    def test_is_str_enum(self):
        assert isinstance(CalculationOperation.DIFFERENCE, str)
        assert CalculationOperation.DIFFERENCE == "difference"


# ---------------------------------------------------------------------------
# CalculationStatus enum
# ---------------------------------------------------------------------------


class TestCalculationStatus:
    def test_has_five_statuses(self):
        assert len(CalculationStatus) == 5

    @pytest.mark.parametrize(
        "name,value",
        [
            ("NOT_APPLICABLE", "not_applicable"),
            ("READY", "ready"),
            ("EXECUTED", "executed"),
            ("BLOCKED", "blocked"),
            ("FAILED", "failed"),
        ],
    )
    def test_member_names_and_values(self, name, value):
        assert CalculationStatus[name].value == value
        assert CalculationStatus(value) is CalculationStatus[name]

    def test_is_str_enum(self):
        assert isinstance(CalculationStatus.EXECUTED, str)


# ---------------------------------------------------------------------------
# CalculationOperand
# ---------------------------------------------------------------------------


class TestCalculationOperand:
    def test_minimal_construction(self):
        op = CalculationOperand(
            name="revenue",
            value=Decimal("1000000"),
        )
        assert op.name == "revenue"
        assert op.value == Decimal("1000000")
        assert op.unit is None
        assert op.scale is None
        assert op.source_text == ""
        assert op.evidence_chunk_id == ""

    def test_full_construction_binds_evidence(self):
        op = CalculationOperand(
            name="revenue",
            value=Decimal("1200000"),
            unit="USD",
            scale="million",
            source_text="Revenue for FY2025 was $1.2 million",
            evidence_chunk_id="chunk_001",
            document_name="annual_report.pdf",
            page=12,
        )
        assert op.evidence_chunk_id == "chunk_001"
        assert op.source_text == "Revenue for FY2025 was $1.2 million"
        assert op.page == 12

    def test_is_frozen(self):
        op = CalculationOperand(name="x", value=Decimal("1"))
        with pytest.raises(Exception):
            op.value = Decimal("2")  # type: ignore[misc]

    def test_page_zero_is_preserved(self):
        """page=0 must be kept (truthy checks would silently drop it)."""
        op = CalculationOperand(name="x", value=Decimal("1"), page=0)
        assert op.page == 0

    def test_to_dict_serializes_value_as_string(self):
        op = CalculationOperand(
            name="revenue",
            value=Decimal("1234.56"),
            unit="USD",
            source_text="Revenue was 1,234.56",
            evidence_chunk_id="c1",
        )
        d = op.to_dict()
        assert d["value"] == "1234.56"
        assert d["name"] == "revenue"
        assert d["unit"] == "USD"
        assert d["evidence_chunk_id"] == "c1"
        assert d["source_text"] == "Revenue was 1,234.56"


# ---------------------------------------------------------------------------
# CalculationPlan
# ---------------------------------------------------------------------------


class TestCalculationPlan:
    def _make_operand(self, name: str, value: str) -> CalculationOperand:
        return CalculationOperand(
            name=name,
            value=Decimal(value),
            source_text=f"{name} = {value}",
            evidence_chunk_id=f"chunk_{name}",
        )

    def test_minimal_plan(self):
        plan = CalculationPlan(
            operation=CalculationOperation.GROSS_MARGIN,
            operands=(
                self._make_operand("revenue", "1000"),
                self._make_operand("cogs", "600"),
            ),
            formula_version="gross_margin.v1",
            target_metric="gross_margin",
        )
        assert plan.operation is CalculationOperation.GROSS_MARGIN
        assert len(plan.operands) == 2
        assert plan.formula_version == "gross_margin.v1"
        assert plan.precision == 4
        assert plan.status is CalculationStatus.READY
        assert plan.block_reason is None

    def test_is_frozen(self):
        plan = CalculationPlan(
            operation=CalculationOperation.SUM,
            operands=(),
            formula_version="sum.v1",
            target_metric="sum",
        )
        with pytest.raises(Exception):
            plan.precision = 2  # type: ignore[misc]

    def test_to_dict_round_trips_operands(self):
        plan = CalculationPlan(
            operation=CalculationOperation.GROWTH_RATE,
            operands=(self._make_operand("current", "110"),),
            formula_version="growth_rate.v1",
            target_metric="growth_rate",
            precision=2,
        )
        d = plan.to_dict()
        assert d["operation"] == "growth_rate"
        assert d["formula_version"] == "growth_rate.v1"
        assert d["precision"] == 2
        assert len(d["operands"]) == 1
        assert d["operands"][0]["name"] == "current"


# ---------------------------------------------------------------------------
# CalculationResult
# ---------------------------------------------------------------------------


class TestCalculationResult:
    def test_executed_result(self):
        r = CalculationResult(
            status=CalculationStatus.EXECUTED,
            operation=CalculationOperation.GROSS_MARGIN,
            value=Decimal("0.4"),
            unit="ratio",
            formula="(revenue - cogs) / revenue",
            formula_version="gross_margin.v1",
            target_metric="gross_margin",
        )
        assert r.status is CalculationStatus.EXECUTED
        assert r.value == Decimal("0.4")
        assert r.error_code is None

    def test_blocked_result_has_error_message(self):
        r = CalculationResult(
            status=CalculationStatus.BLOCKED,
            error_code="INSUFFICIENT_EVIDENCE",
            error_message="Could not find COGS figure in retrieved evidence",
        )
        assert r.value is None
        assert r.error_code == "INSUFFICIENT_EVIDENCE"

    def test_to_dict_includes_all_fields(self):
        op = CalculationOperand(
            name="revenue",
            value=Decimal("1000"),
            source_text="Revenue was 1000",
            evidence_chunk_id="c1",
        )
        r = CalculationResult(
            status=CalculationStatus.EXECUTED,
            operation=CalculationOperation.GROSS_MARGIN,
            value=Decimal("0.4"),
            formula="(revenue - cogs) / revenue",
            formula_version="gross_margin.v1",
            target_metric="gross_margin",
            operands=(op,),
        )
        d = r.to_dict()
        assert d["status"] == "executed"
        assert d["operation"] == "gross_margin"
        assert d["value"] == "0.4"
        assert d["formula_version"] == "gross_margin.v1"
        assert len(d["operands"]) == 1
        assert d["operands"][0]["name"] == "revenue"
        assert d["error_code"] is None

    def test_to_dict_not_applicable(self):
        d = NOT_APPLICABLE_RESULT.to_dict()
        assert d["status"] == "not_applicable"
        assert d["operation"] is None
        assert d["value"] is None
        assert d["operands"] == []

    def test_is_frozen(self):
        r = CalculationResult(status=CalculationStatus.FAILED)
        with pytest.raises(Exception):
            r.value = Decimal("1")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Domain-layer dependency purity
# ---------------------------------------------------------------------------


class TestDomainPurity:
    """The domain module must not import from finance/application/services."""

    def test_calculation_module_does_not_import_services(self):
        import inspect

        from src.domain import calculation as calc_mod

        source = inspect.getsource(calc_mod)
        # The module docstring mentions these layers, but no actual imports.
        # Check for import statements that would couple layers.
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith(('"""', "'''")) or stripped.startswith(
                "from __future__"
            ):
                continue
            # Allow: from dataclasses, from decimal, from enum, from typing
            assert "from src.services" not in stripped, f"forbidden import: {stripped}"
            assert "from src.finance" not in stripped, f"forbidden import: {stripped}"
            assert "from src.application" not in stripped, (
                f"forbidden import: {stripped}"
            )
            assert "import src.services" not in stripped, (
                f"forbidden import: {stripped}"
            )
            assert "import src.finance" not in stripped, f"forbidden import: {stripped}"
            assert "import src.application" not in stripped, (
                f"forbidden import: {stripped}"
            )
