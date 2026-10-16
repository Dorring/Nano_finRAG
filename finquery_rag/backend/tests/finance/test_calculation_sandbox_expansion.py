"""Comprehensive test suite for calculation sandbox expansion.

Tests:
1. Multi-source scale harmonization (SQL million, Market API units, PDF thousands).
2. Pure Decimal high-order primitives:
   - compute_cagr: multi-period compounding, fail-closed on non-positive base/periods
   - valuation_multiple: price/eps, market_cap/net_income, fail-closed on zero/negative base
   - spread: two-operand delta and multi-operand range (max - min)
   - median_values: odd/even lists, sorted deterministic middle
3. Registration in CALCULATION_REGISTRY.
4. ExecutionDAG execution with heterogeneous multi-source operands.
5. End-to-end integration with SqlFactStore and SandboxMarketProvider.
"""

from decimal import Decimal
import pytest

from rag_v2.contracts.financial_semantics import RepresentationKind
from rag_v2.contracts.plan import ExecutionDAG, OperandRef, OperationStep
from rag_v2.sources.market_api import SandboxMarketProvider
from rag_v2.sources.sql_fact_store import SqlFactStore
from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
    CalculationPlan,
    CalculationStatus,
)
from src.finance.calculation_executor import (
    _harmonize_operand_scales,
    execute_dag,
    execute_plan,
)
from src.finance.calculation_registry import (
    CALCULATION_REGISTRY,
    get_operation_entry,
    supports,
)
from src.finance.primitive_tools import (
    compute_cagr,
    median_values,
    spread,
    valuation_multiple,
)


# ---------------------------------------------------------------------------
# 1. Primitives Unit Tests
# ---------------------------------------------------------------------------

class TestHighOrderPrimitives:
    """Test deterministic pure Decimal implementations of high-order financial operators."""

    def test_compute_cagr_standard(self):
        # 100 to 200 over 4 periods: (200 / 100) ** (1/4) - 1 = 2 ** 0.25 - 1 ~ 0.1892
        res = compute_cagr(Decimal("200"), Decimal("100"), periods=Decimal("4"), precision=4)
        assert res.ok is True
        assert res.points_value == Decimal("0.1892")
        assert res.ratio_value == Decimal("0.1892")
        assert res.representation == RepresentationKind.RATIO
        assert res.unit == "ratio"

    def test_compute_cagr_one_period_equals_growth_rate(self):
        # 110 to 121 over 1 period = 10%
        res = compute_cagr(Decimal("121"), Decimal("110"), periods=Decimal("1"), precision=4)
        assert res.ok is True
        assert res.points_value == Decimal("0.1000")

    def test_compute_cagr_fail_closed_non_positive_start(self):
        # Starting from negative or zero is mathematically invalid for CAGR
        res_zero = compute_cagr(Decimal("100"), Decimal("0"), periods=Decimal("3"))
        assert res_zero.ok is False
        assert "non-positive start value" in (res_zero.error or "")

        res_neg = compute_cagr(Decimal("100"), Decimal("-50"), periods=Decimal("3"))
        assert res_neg.ok is False
        assert "non-positive start value" in (res_neg.error or "")

    def test_compute_cagr_fail_closed_negative_end(self):
        res = compute_cagr(Decimal("-20"), Decimal("50"), periods=Decimal("3"))
        assert res.ok is False
        assert "negative end value" in (res.error or "")

    def test_compute_cagr_end_zero(self):
        # Value drops to 0 over 3 years: -100% (-1.0000)
        res = compute_cagr(Decimal("0"), Decimal("50"), periods=Decimal("3"), precision=4)
        assert res.ok is True
        assert res.points_value == Decimal("-1.0000")

    def test_compute_cagr_fail_closed_invalid_periods(self):
        res_zero = compute_cagr(Decimal("120"), Decimal("100"), periods=Decimal("0"))
        assert res_zero.ok is False
        assert "non-positive period count" in (res_zero.error or "")

        res_neg = compute_cagr(Decimal("120"), Decimal("100"), periods=Decimal("-2"))
        assert res_neg.ok is False

    def test_valuation_multiple_price_to_earnings(self):
        # Stock Price $231.45, Diluted EPS $6.15 -> P/E = 37.6341
        res = valuation_multiple(Decimal("231.45"), Decimal("6.15"), precision=4)
        assert res.ok is True
        assert res.points_value == Decimal("37.6341")
        assert res.unit == "multiple"

    def test_valuation_multiple_fail_closed_zero_denominator(self):
        res = valuation_multiple(Decimal("100"), Decimal("0"))
        assert res.ok is False
        assert "zero fundamental metric" in (res.error or "")

    def test_valuation_multiple_fail_closed_negative_denominator(self):
        # Loss-making firms have undefined P/E multiple
        res = valuation_multiple(Decimal("100"), Decimal("-5.2"))
        assert res.ok is False
        assert "undefined for non-positive fundamental metric" in (res.error or "")

    def test_spread_two_operands(self):
        # Apple Gross Margin 46.21% vs Microsoft 44.60% -> spread = 1.61%
        res = spread([Decimal("46.21"), Decimal("44.60")], precision=2)
        assert res.ok is True
        assert res.points_value == Decimal("1.61")

    def test_spread_multi_operands_range(self):
        # Set of margins: [12.5, 34.0, 48.2, 19.1] -> max - min = 48.2 - 12.5 = 35.7
        res = spread([Decimal("12.5"), Decimal("34.0"), Decimal("48.2"), Decimal("19.1")], precision=2)
        assert res.ok is True
        assert res.points_value == Decimal("35.70")
        assert res.details["max"] == "48.2"
        assert res.details["min"] == "12.5"

    def test_spread_insufficient_operands(self):
        res = spread([Decimal("12.5")])
        assert res.ok is False
        assert "requires at least 2 values" in (res.error or "")

    def test_median_odd_count(self):
        # [10, 50, 20] -> sorted [10, 20, 50] -> median is 20
        res = median_values([Decimal("10"), Decimal("50"), Decimal("20")], precision=2)
        assert res.ok is True
        assert res.points_value == Decimal("20.00")

    def test_median_even_count(self):
        # [10, 40, 20, 30] -> sorted [10, 20, 30, 40] -> median = (20 + 30) / 2 = 25
        res = median_values([Decimal("10"), Decimal("40"), Decimal("20"), Decimal("30")], precision=2)
        assert res.ok is True
        assert res.points_value == Decimal("25.00")


# ---------------------------------------------------------------------------
# 2. Scale Harmonization Tests
# ---------------------------------------------------------------------------

class TestScaleHarmonization:
    """Test automatic scale harmonization across heterogeneous sources."""

    def test_harmonize_million_and_units_difference(self):
        op1 = CalculationOperand(
            name="total_revenue",
            value=Decimal("391035"),
            scale="million",
            unit="USD",
        )
        op2 = CalculationOperand(
            name="cash_receipt",
            value=Decimal("5000000"),
            scale="units",
            unit="USD",
        )
        harmonized = _harmonize_operand_scales((op1, op2), CalculationOperation.DIFFERENCE)
        # op1 should be converted to units: 391035 * 1,000,000 = 391,035,000,000
        # op2 should remain 5,000,000 units
        assert harmonized[0].value == Decimal("391035000000")
        assert harmonized[0].scale == "units"
        assert harmonized[1].value == Decimal("5000000")
        assert harmonized[1].scale == "units"

    def test_harmonize_same_scale_remains_untouched(self):
        op1 = CalculationOperand(name="rev1", value=Decimal("100"), scale="million", unit="USD")
        op2 = CalculationOperand(name="rev2", value=Decimal("80"), scale="million", unit="USD")
        harmonized = _harmonize_operand_scales((op1, op2), CalculationOperation.DIFFERENCE)
        assert harmonized[0].value == Decimal("100")
        assert harmonized[0].scale == "million"
        assert harmonized[1].value == Decimal("80")
        assert harmonized[1].scale == "million"

    def test_harmonize_cagr_preserves_period_count(self):
        # End value in million, Start value in thousands, Periods = 4
        op_end = CalculationOperand(name="end_rev", value=Decimal("100"), scale="million", unit="USD")
        op_start = CalculationOperand(name="start_rev", value=Decimal("50000"), scale="thousands", unit="USD")
        op_periods = CalculationOperand(name="n_years", value=Decimal("4"), scale="units", unit=None)

        harmonized = _harmonize_operand_scales((op_end, op_start, op_periods), CalculationOperation.CAGR)
        # op_end: 100 million -> 100,000 thousands (finest scale among end & start is thousands)
        # op_start: 50,000 thousands
        # op_periods: remains 4! periods must NOT be scale-multiplied
        assert harmonized[0].value == Decimal("100000")
        assert harmonized[0].scale == "thousands"
        assert harmonized[1].value == Decimal("50000")
        assert harmonized[1].scale == "thousands"
        assert harmonized[2].value == Decimal("4")


# ---------------------------------------------------------------------------
# 3. Calculation Registry & Plan Execution Tests
# ---------------------------------------------------------------------------

class TestRegistryAndPlanExecution:
    """Test registry metadata and execute_plan integration."""

    def test_all_new_operations_supported(self):
        for op in (
            CalculationOperation.CAGR,
            CalculationOperation.VALUATION_MULTIPLE,
            CalculationOperation.SPREAD,
            CalculationOperation.MEDIAN,
        ):
            assert supports(op) is True
            entry = get_operation_entry(op)
            assert entry is not None
            assert entry.formula_version.endswith(".v1")

    def test_execute_plan_cagr_three_operands(self):
        op1 = CalculationOperand(name="rev_2024", value=Decimal("200"), scale="million")
        op2 = CalculationOperand(name="rev_2020", value=Decimal("100"), scale="million")
        op3 = CalculationOperand(name="periods", value=Decimal("4"), scale="units")

        plan = CalculationPlan(
            operation=CalculationOperation.CAGR,
            operands=(op1, op2, op3),
            formula_version="cagr.v1",
            target_metric="revenue_cagr",
            precision=4,
        )
        res = execute_plan(plan)
        assert res.status == CalculationStatus.EXECUTED
        assert res.value == Decimal("0.1892")
        assert res.unit == "ratio"

    def test_execute_plan_valuation_multiple_harmonized(self):
        # Market Cap = 3,520,000 million USD; Net Income = 93,736 million USD
        op_mc = CalculationOperand(name="market_cap", value=Decimal("3520000"), scale="million", unit="USD")
        op_ni = CalculationOperand(name="net_income", value=Decimal("93736"), scale="million", unit="USD")

        plan = CalculationPlan(
            operation=CalculationOperation.VALUATION_MULTIPLE,
            operands=(op_mc, op_ni),
            formula_version="valuation_multiple.v1",
            target_metric="pe_ratio",
            precision=2,
        )
        res = execute_plan(plan)
        assert res.status == CalculationStatus.EXECUTED
        # 3520000 / 93736 = 37.55
        assert res.value == Decimal("37.55")
        assert res.unit == "multiple"

    def test_execute_plan_spread_and_median(self):
        op1 = CalculationOperand(name="m1", value=Decimal("15.2"), unit="percent")
        op2 = CalculationOperand(name="m2", value=Decimal("22.8"), unit="percent")
        op3 = CalculationOperand(name="m3", value=Decimal("18.0"), unit="percent")

        plan_spread = CalculationPlan(
            operation=CalculationOperation.SPREAD,
            operands=(op1, op2, op3),
            formula_version="spread.v1",
            target_metric="margin_spread",
            precision=2,
        )
        res_spread = execute_plan(plan_spread)
        assert res_spread.status == CalculationStatus.EXECUTED
        # 22.8 - 15.2 = 7.6
        assert res_spread.value == Decimal("7.60")

        plan_median = CalculationPlan(
            operation=CalculationOperation.MEDIAN,
            operands=(op1, op2, op3),
            formula_version="median.v1",
            target_metric="margin_median",
            precision=2,
        )
        res_median = execute_plan(plan_median)
        assert res_median.status == CalculationStatus.EXECUTED
        # sorted [15.2, 18.0, 22.8] -> 18.0
        assert res_median.value == Decimal("18.00")


# ---------------------------------------------------------------------------
# 4. Multi-Step DAG Execution with Multi-Source Data
# ---------------------------------------------------------------------------

class TestMultiSourceDAGExecution:
    """Test ExecutionDAG combining SQL wide table and Market API data."""

    def test_coca_cola_5yr_revenue_cagr_from_sqlite(self):
        """Query real audited 5-year series from SqlFactStore and compute 4-period CAGR."""
        store = SqlFactStore(":memory:")
        store.seed_benchmark_data()

        # KO FY24 Total Revenue = 46011 M, KO FY20 Total Revenue = 33014 M
        fact_2024 = store.query_fact("KO", "total revenue", "FY2024")
        fact_2020 = store.query_fact("KO", "total revenue", "FY2020")
        assert fact_2024 is not None
        assert fact_2020 is not None

        op_end = CalculationOperand(
            name="KO_FY2024_Revenue",
            value=Decimal(str(fact_2024["value"])),
            scale=fact_2024.get("scale", "million"),
            unit=fact_2024.get("unit", "USD"),
        )
        op_start = CalculationOperand(
            name="KO_FY2020_Revenue",
            value=Decimal(str(fact_2020["value"])),
            scale=fact_2020.get("scale", "million"),
            unit=fact_2020.get("unit", "USD"),
        )
        op_periods = CalculationOperand(
            name="periods",
            value=Decimal("4"),
            scale="units",
        )

        dag = ExecutionDAG(
            steps=(
                OperationStep(
                    step_id="step_cagr",
                    operation="cagr",
                    operands=(
                        OperandRef(ref_type="slot", ref_id="slot_end"),
                        OperandRef(ref_type="slot", ref_id="slot_start"),
                        OperandRef(ref_type="slot", ref_id="slot_periods"),
                    ),
                    output_unit="ratio",
                ),
            ),
            output_step_id="step_cagr",
        )

        leaf_operands = {
            "slot_end": op_end,
            "slot_start": op_start,
            "slot_periods": op_periods,
        }

        result = execute_dag(dag, leaf_operands, precision=4)
        assert result.status == CalculationStatus.EXECUTED
        # (47061 / 33014) ** 0.25 - 1 = 0.0927
        assert result.value == Decimal("0.0927")
        assert result.unit == "ratio"

    def test_realtime_pe_multiple_derivation(self):
        """Combine real-time stock price from Market API with 10-K diluted EPS from SqlFactStore."""
        market_provider = SandboxMarketProvider()
        quote = market_provider.get_quote("Apple", "stock_price")
        assert quote is not None

        # Apple real-time quote is $231.45 (scale: units, unit: USD)
        op_price = CalculationOperand(
            name="AAPL_Stock_Price",
            value=Decimal(str(quote["value"])),
            scale=quote.get("scale", "units"),
            unit=quote.get("unit", "USD"),
            slot_id="slot_price",
        )

        # 10-K Audited Diluted EPS: $6.11 (scale: units, unit: USD)
        op_eps = CalculationOperand(
            name="AAPL_Diluted_EPS",
            value=Decimal("6.11"),
            scale="units",
            unit="USD",
            slot_id="slot_eps",
        )

        dag = ExecutionDAG(
            steps=(
                OperationStep(
                    step_id="step_pe",
                    operation="valuation_multiple",
                    operands=(
                        OperandRef(ref_type="slot", ref_id="slot_price"),
                        OperandRef(ref_type="slot", ref_id="slot_eps"),
                    ),
                    output_unit="multiple",
                ),
            ),
            output_step_id="step_pe",
        )

        leaf_operands = {
            "slot_price": op_price,
            "slot_eps": op_eps,
        }

        result = execute_dag(dag, leaf_operands, precision=2)
        assert result.status == CalculationStatus.EXECUTED
        # 231.45 / 6.11 = 37.88
        assert result.value == Decimal("37.88")
        assert result.unit == "multiple"

    def test_heterogeneous_dag_cross_source_scale_harmonization(self):
        """Test multi-step DAG:
        Step 1: Compute Net Margin = Net Income / Revenue (both in million from SQL).
        Step 2: Compute P/E Multiple = Market Cap (million from Market API) / Net Income (million from SQL).
        """
        op_rev = CalculationOperand(name="revenue", value=Decimal("391035"), scale="million", unit="USD")
        op_ni = CalculationOperand(name="net_income", value=Decimal("93736"), scale="million", unit="USD")
        op_mc = CalculationOperand(name="market_cap", value=Decimal("3520000"), scale="million", unit="USD")

        dag = ExecutionDAG(
            steps=(
                OperationStep(
                    step_id="calc_net_margin",
                    operation="net_margin",
                    operands=(
                        OperandRef(ref_type="slot", ref_id="rev_slot"),
                        OperandRef(ref_type="slot", ref_id="ni_slot"),
                    ),
                    output_unit="ratio",
                ),
                OperationStep(
                    step_id="calc_pe_multiple",
                    operation="valuation_multiple",
                    operands=(
                        OperandRef(ref_type="slot", ref_id="mc_slot"),
                        OperandRef(ref_type="slot", ref_id="ni_slot"),
                    ),
                    output_unit="multiple",
                ),
            ),
            output_step_id="calc_pe_multiple",
        )

        leaf_operands = {
            "rev_slot": op_rev,
            "ni_slot": op_ni,
            "mc_slot": op_mc,
        }

        result = execute_dag(dag, leaf_operands, precision=4)
        assert result.status == CalculationStatus.EXECUTED
        assert len(result.steps) == 2
        # Net margin = 93736 / 391035 = 0.2397
        assert result.steps[0].value == Decimal("0.2397")
        # PE multiple = 3520000 / 93736 = 37.5523
        assert result.value == Decimal("37.5523")
