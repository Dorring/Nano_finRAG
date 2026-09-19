"""Tests for the calculation registry and executor (Phase 3 Commit 6)."""

import os
import sys
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
    CalculationPlan,
    CalculationStatus,
)
from src.finance.calculation_executor import execute_plan
from src.finance.calculation_registry import (
    CALCULATION_REGISTRY,
    OperationEntry,
    get_operation_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _operand(name: str, value: str) -> CalculationOperand:
    return CalculationOperand(
        name=name,
        value=Decimal(value),
        source_text=f"{name} = {value}",
        evidence_chunk_id=f"chunk_{name}",
    )


def _ready_plan(
    operation: CalculationOperation,
    operands: tuple[CalculationOperand, ...],
    target_metric: str | None = None,
    precision: int = 4,
) -> CalculationPlan:
    entry = get_operation_entry(operation)
    return CalculationPlan(
        operation=operation,
        operands=operands,
        formula_version=entry.formula_version if entry else "unknown.v1",
        target_metric=target_metric or operation.value,
        precision=precision,
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestCalculationRegistry:
    def test_every_operation_has_a_registry_entry(self):
        """The registry is the executable-capability authority.

        An operation with no entry cannot be executed, so this is the invariant
        that says the enum and the registry describe the same set -- adding an
        enum member without an entry is a half-wired change, and this is what
        catches it.
        """

        registered = set(CALCULATION_REGISTRY.keys())
        expected = set(CalculationOperation)
        assert registered == expected, (
            f"missing: {expected - registered}, extra: {registered - expected}"
        )

    @pytest.mark.parametrize(
        "operation,formula_version,unit",
        [
            (CalculationOperation.DIFFERENCE, "difference.v1", "base"),
            (CalculationOperation.GROWTH_RATE, "growth_rate.v1", "ratio"),
            (CalculationOperation.PERCENTAGE_SHARE, "percentage_share.v1", "ratio"),
            (CalculationOperation.SUM, "sum.v1", "base"),
            (CalculationOperation.AVERAGE, "average.v1", "base"),
            (CalculationOperation.GROSS_MARGIN, "gross_margin.v1", "ratio"),
            (CalculationOperation.NET_MARGIN, "net_margin.v1", "ratio"),
            (CalculationOperation.DEBT_RATIO, "debt_ratio.v1", "ratio"),
            (CalculationOperation.SCALE_CONVERSION, "scale_conversion.v1", "base"),
        ],
    )
    def test_formula_version_and_unit(self, operation, formula_version, unit):
        entry = CALCULATION_REGISTRY[operation]
        assert entry.formula_version == formula_version
        assert entry.unit == unit

    @pytest.mark.parametrize(
        "operation,roles",
        [
            (CalculationOperation.DIFFERENCE, ("current", "previous")),
            (CalculationOperation.GROWTH_RATE, ("current", "previous")),
            (CalculationOperation.PERCENTAGE_SHARE, ("part", "total")),
            (CalculationOperation.GROSS_MARGIN, ("revenue", "cogs")),
            (CalculationOperation.NET_MARGIN, ("revenue", "net_income")),
            (CalculationOperation.DEBT_RATIO, ("total_liabilities", "total_assets")),
            (CalculationOperation.SCALE_CONVERSION, ("value",)),
        ],
    )
    def test_operand_roles_for_two_operand_operations(self, operation, roles):
        entry = CALCULATION_REGISTRY[operation]
        assert entry.operand_roles == roles

    def test_sum_and_average_have_empty_roles(self):
        assert CALCULATION_REGISTRY[CalculationOperation.SUM].operand_roles == ()
        assert CALCULATION_REGISTRY[CalculationOperation.AVERAGE].operand_roles == ()

    def test_two_operand_operations_require_exactly_two(self):
        for op in (
            CalculationOperation.DIFFERENCE,
            CalculationOperation.GROWTH_RATE,
            CalculationOperation.PERCENTAGE_SHARE,
            CalculationOperation.GROSS_MARGIN,
            CalculationOperation.NET_MARGIN,
            CalculationOperation.DEBT_RATIO,
        ):
            entry = CALCULATION_REGISTRY[op]
            assert entry.min_operands == 2
            assert entry.max_operands == 2

    def test_sum_accepts_variable_operands(self):
        entry = CALCULATION_REGISTRY[CalculationOperation.SUM]
        assert entry.min_operands == 1
        assert entry.max_operands >= 2

    def test_average_accepts_variable_operands(self):
        entry = CALCULATION_REGISTRY[CalculationOperation.AVERAGE]
        assert entry.min_operands == 1
        assert entry.max_operands >= 2

    def test_get_operation_entry_returns_entry_for_known(self):
        entry = get_operation_entry(CalculationOperation.GROSS_MARGIN)
        assert entry is not None
        assert entry.operation is CalculationOperation.GROSS_MARGIN

    def test_get_operation_entry_returns_none_for_unknown(self):
        # All enum members are registered; simulate unknown by deleting.
        # Instead, verify the function handles missing keys gracefully.
        entry = CALCULATION_REGISTRY.pop(CalculationOperation.SUM, None)
        try:
            assert get_operation_entry(CalculationOperation.SUM) is None
        finally:
            if entry is not None:
                CALCULATION_REGISTRY[CalculationOperation.SUM] = entry

    def test_every_entry_is_frozen(self):
        for entry in CALCULATION_REGISTRY.values():
            with pytest.raises(Exception):
                entry.formula = "tampered"  # type: ignore[misc]

    def test_every_entry_has_nonempty_formula(self):
        for entry in CALCULATION_REGISTRY.values():
            assert entry.formula, f"empty formula for {entry.operation}"

    def test_formula_versions_match_metric_lexicon(self):
        """Registry formula_versions must match NAMED_METRICS / GENERIC_OPERATIONS."""
        from src.finance.metric_lexicon import GENERIC_OPERATIONS, NAMED_METRICS

        for metric in NAMED_METRICS.values():
            entry = CALCULATION_REGISTRY[metric.operation]
            assert entry.formula_version == metric.formula_version, (
                f"version mismatch for {metric.metric}: "
                f"registry={entry.formula_version} lexicon={metric.formula_version}"
            )
        for generic in GENERIC_OPERATIONS:
            entry = CALCULATION_REGISTRY[generic.operation]
            assert entry.formula_version == generic.formula_version


# ---------------------------------------------------------------------------
# Executor: status pass-through
# ---------------------------------------------------------------------------


class TestExecutorStatusPassThrough:
    def test_not_applicable_plan_returns_not_applicable(self):
        plan = CalculationPlan(
            operation=CalculationOperation.GROSS_MARGIN,
            operands=(),
            formula_version="gross_margin.v1",
            target_metric="gross_margin",
            status=CalculationStatus.NOT_APPLICABLE,
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.NOT_APPLICABLE
        assert result.value is None
        assert result.error_code is None

    def test_blocked_plan_returns_blocked_with_error_code(self):
        plan = CalculationPlan(
            operation=CalculationOperation.GROSS_MARGIN,
            operands=(),
            formula_version="gross_margin.v1",
            target_metric="gross_margin",
            status=CalculationStatus.BLOCKED,
            block_reason="missing COGS operand",
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.BLOCKED
        assert result.error_code == "PLAN_BLOCKED"
        assert "missing COGS operand" in result.error_message


# ---------------------------------------------------------------------------
# Executor: successful execution per operation
# ---------------------------------------------------------------------------


class TestExecutorSuccess:
    def test_gross_margin_executed(self):
        plan = _ready_plan(
            CalculationOperation.GROSS_MARGIN,
            (_operand("revenue", "1000"), _operand("cogs", "600")),
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.EXECUTED
        assert result.value == Decimal("0.4000")
        assert result.unit == "ratio"
        assert result.formula_version == "gross_margin.v1"
        assert result.target_metric == "gross_margin"
        assert len(result.operands) == 2

    def test_net_margin_executed(self):
        plan = _ready_plan(
            CalculationOperation.NET_MARGIN,
            (_operand("revenue", "1000"), _operand("net_income", "150")),
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.EXECUTED
        assert result.value == Decimal("0.1500")
        assert result.unit == "ratio"

    def test_debt_ratio_executed(self):
        plan = _ready_plan(
            CalculationOperation.DEBT_RATIO,
            (
                _operand("total_liabilities", "400"),
                _operand("total_assets", "1000"),
            ),
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.EXECUTED
        assert result.value == Decimal("0.4000")
        assert result.unit == "ratio"

    def test_growth_rate_executed(self):
        plan = _ready_plan(
            CalculationOperation.GROWTH_RATE,
            (_operand("current", "120"), _operand("previous", "100")),
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.EXECUTED
        assert result.value == Decimal("0.2000")
        assert result.unit == "ratio"

    def test_percentage_share_executed(self):
        plan = _ready_plan(
            CalculationOperation.PERCENTAGE_SHARE,
            (_operand("part", "30"), _operand("total", "120")),
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.EXECUTED
        assert result.value == Decimal("0.2500")
        assert result.unit == "ratio"

    def test_difference_executed(self):
        plan = _ready_plan(
            CalculationOperation.DIFFERENCE,
            (_operand("current", "150"), _operand("previous", "100")),
            precision=2,
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.EXECUTED
        assert result.value == Decimal("50.00")
        assert result.unit == "base"

    def test_sum_executed(self):
        plan = _ready_plan(
            CalculationOperation.SUM,
            (_operand("a", "100"), _operand("b", "200"), _operand("c", "300")),
            precision=2,
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.EXECUTED
        assert result.value == Decimal("600.00")
        assert result.unit == "base"

    def test_average_executed(self):
        plan = _ready_plan(
            CalculationOperation.AVERAGE,
            (_operand("a", "100"), _operand("b", "200"), _operand("c", "300")),
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.EXECUTED
        assert result.value == Decimal("200.0000")
        assert result.unit == "base"

    def test_executed_result_carries_formula(self):
        plan = _ready_plan(
            CalculationOperation.GROSS_MARGIN,
            (_operand("revenue", "1000"), _operand("cogs", "600")),
        )
        result = execute_plan(plan)
        assert result.formula == "(revenue - cogs) / revenue"

    def test_executed_result_carries_operands(self):
        ops = (_operand("revenue", "1000"), _operand("cogs", "600"))
        plan = _ready_plan(CalculationOperation.GROSS_MARGIN, ops)
        result = execute_plan(plan)
        assert result.operands == ops


# ---------------------------------------------------------------------------
# Executor: operand count validation
# ---------------------------------------------------------------------------


class TestExecutorOperandValidation:
    def test_insufficient_operands_blocks(self):
        plan = _ready_plan(
            CalculationOperation.GROSS_MARGIN,
            (_operand("revenue", "1000"),),  # missing cogs
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.BLOCKED
        assert result.error_code == "INSUFFICIENT_OPERANDS"
        assert "at least 2" in result.error_message

    def test_too_many_operands_blocks(self):
        plan = _ready_plan(
            CalculationOperation.GROSS_MARGIN,
            (
                _operand("revenue", "1000"),
                _operand("cogs", "600"),
                _operand("extra", "100"),
            ),
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.BLOCKED
        assert result.error_code == "TOO_MANY_OPERANDS"
        assert "at most 2" in result.error_message

    def test_sum_with_zero_operands_blocks(self):
        plan = _ready_plan(CalculationOperation.SUM, ())
        result = execute_plan(plan)
        assert result.status is CalculationStatus.BLOCKED
        assert result.error_code == "INSUFFICIENT_OPERANDS"

    def test_average_with_zero_operands_blocks(self):
        plan = _ready_plan(CalculationOperation.AVERAGE, ())
        result = execute_plan(plan)
        assert result.status is CalculationStatus.BLOCKED
        assert result.error_code == "INSUFFICIENT_OPERANDS"


# ---------------------------------------------------------------------------
# Executor: primitive decline (BLOCKED)
# ---------------------------------------------------------------------------


class TestExecutorPrimitiveDecline:
    def test_growth_rate_zero_previous_blocks(self):
        plan = _ready_plan(
            CalculationOperation.GROWTH_RATE,
            (_operand("current", "100"), _operand("previous", "0")),
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.BLOCKED
        assert result.error_code == "PRIMITIVE_DECLINED"
        assert "zero previous" in result.error_message

    def test_gross_margin_zero_revenue_blocks(self):
        plan = _ready_plan(
            CalculationOperation.GROSS_MARGIN,
            (_operand("revenue", "0"), _operand("cogs", "600")),
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.BLOCKED
        assert result.error_code == "PRIMITIVE_DECLINED"

    def test_debt_ratio_zero_assets_blocks(self):
        plan = _ready_plan(
            CalculationOperation.DEBT_RATIO,
            (
                _operand("total_liabilities", "400"),
                _operand("total_assets", "0"),
            ),
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.BLOCKED
        assert result.error_code == "PRIMITIVE_DECLINED"

    def test_percentage_share_zero_total_blocks(self):
        plan = _ready_plan(
            CalculationOperation.PERCENTAGE_SHARE,
            (_operand("part", "30"), _operand("total", "0")),
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.BLOCKED
        assert result.error_code == "PRIMITIVE_DECLINED"

    def test_scale_conversion_without_scales_blocks_unit_ambiguous(self):
        """Scale conversion without source/target scale blocks with UNIT_AMBIGUOUS."""
        plan = _ready_plan(
            CalculationOperation.SCALE_CONVERSION,
            (_operand("value", "100"),),
        )
        result = execute_plan(plan)
        assert result.status is CalculationStatus.BLOCKED
        assert result.error_code == "UNIT_AMBIGUOUS"
        assert "scale is missing" in result.error_message


# ---------------------------------------------------------------------------
# Executor: FAILED path (exception)
# ---------------------------------------------------------------------------


class TestExecutorFailure:
    def test_unknown_operation_returns_failed(self):
        # Simulate unknown operation by temporarily removing it from registry.
        entry = CALCULATION_REGISTRY.pop(CalculationOperation.SUM, None)
        try:
            plan = _ready_plan(
                CalculationOperation.SUM,
                (_operand("a", "1"),),
            )
            result = execute_plan(plan)
            assert result.status is CalculationStatus.FAILED
            assert result.error_code == "UNKNOWN_OPERATION"
        finally:
            if entry is not None:
                CALCULATION_REGISTRY[CalculationOperation.SUM] = entry

    def test_primitive_exception_returns_failed(self):
        """If the primitive raises, the executor returns FAILED (not BLOCKED)."""
        # Inject a poisonous entry that raises.
        original = CALCULATION_REGISTRY[CalculationOperation.SUM]

        def _poisonous(operands, precision):
            raise RuntimeError("boom")

        CALCULATION_REGISTRY[CalculationOperation.SUM] = OperationEntry(
            operation=CalculationOperation.SUM,
            func=_poisonous,
            formula="boom()",
            formula_version="sum.v1",
            unit="base",
            min_operands=1,
            max_operands=100,
            operand_roles=(),
        )
        try:
            plan = _ready_plan(
                CalculationOperation.SUM,
                (_operand("a", "1"),),
            )
            result = execute_plan(plan)
            assert result.status is CalculationStatus.FAILED
            assert result.error_code == "PRIMITIVE_EXCEPTION"
            assert "boom" in result.error_message
        finally:
            CALCULATION_REGISTRY[CalculationOperation.SUM] = original


# ---------------------------------------------------------------------------
# Executor: result serialization
# ---------------------------------------------------------------------------


class TestExecutorResultSerialization:
    def test_executed_to_dict_round_trip(self):
        plan = _ready_plan(
            CalculationOperation.GROSS_MARGIN,
            (_operand("revenue", "1000"), _operand("cogs", "600")),
        )
        result = execute_plan(plan)
        d = result.to_dict()
        assert d["status"] == "executed"
        assert d["operation"] == "gross_margin"
        assert d["value"] == "0.4000"
        assert d["formula_version"] == "gross_margin.v1"
        assert d["formula"] == "(revenue - cogs) / revenue"
        assert d["unit"] == "ratio"
        assert len(d["operands"]) == 2
        assert d["operands"][0]["name"] == "revenue"
        assert d["error_code"] is None

    def test_blocked_to_dict_includes_error(self):
        plan = _ready_plan(
            CalculationOperation.GROWTH_RATE,
            (_operand("current", "100"), _operand("previous", "0")),
        )
        result = execute_plan(plan)
        d = result.to_dict()
        assert d["status"] == "blocked"
        assert d["error_code"] == "PRIMITIVE_DECLINED"
        assert d["value"] is None


# ---------------------------------------------------------------------------
# Layer purity
# ---------------------------------------------------------------------------


class TestExecutorLayerPurity:
    def test_executor_does_not_import_services_or_application(self):
        import inspect

        from src.finance import calculation_executor as mod

        source = inspect.getsource(mod)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith(('"""', "'''")) or stripped.startswith(
                "from __future__"
            ):
                continue
            assert "from src.services" not in stripped, f"forbidden: {stripped}"
            assert "from src.application" not in stripped, f"forbidden: {stripped}"

    def test_registry_does_not_import_services_or_application(self):
        import inspect

        from src.finance import calculation_registry as mod

        source = inspect.getsource(mod)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith(('"""', "'''")) or stripped.startswith(
                "from __future__"
            ):
                continue
            assert "from src.services" not in stripped, f"forbidden: {stripped}"
            assert "from src.application" not in stripped, f"forbidden: {stripped}"
