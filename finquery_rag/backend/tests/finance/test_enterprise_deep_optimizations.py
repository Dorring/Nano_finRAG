"""Comprehensive test suite for Phase 8 enterprise optimizations.

Tests:
1. Deterministic DuPont 3-Factor decomposition primitive and adapter
2. Deterministic FX Currency Conversion primitive and adapter
3. Deterministic Scenario Growth Projection primitive and adapter
4. CalculationOperation enum, plan contracts, and registry completeness
5. Table Cell Coordinates and PDF Visual Bounding Box passthrough in citations
6. US-GAAP Taxonomy namespace stripping and canonical accounting resolution
7. SEC EDGAR 8-K filing recognition, item parsing, and idempotent recording
"""

from decimal import Decimal
import tempfile
from pathlib import Path
import pytest

from src.domain.calculation import CalculationOperand, CalculationOperation
from rag_v2.contracts.plan import _OPERATION_VALUES
from src.finance.calculation_registry import (
    CALCULATION_REGISTRY,
    get_operation_entry,
    supports,
)
from src.finance.primitive_tools import (
    compute_dupont_3factor,
    compute_fx_conversion,
    project_scenario_growth,
)
from src.runtime.trusted_v2_coordinator import _structured_citations
from rag_v2.sources.sql_fact_store import normalize_metric
from src.finance.edgar_feed_monitor import EdgarFeedMonitor, SEC_8K_ITEM_DESCRIPTIONS


# ---------------------------------------------------------------------------
# 1. DuPont 3-Factor Decomposition
# ---------------------------------------------------------------------------

def test_dupont_3factor_happy_path():
    # Apple FY2024 ballpark: Net Income ~93736, Revenue ~391035, Assets ~364980, Equity ~66782
    res = compute_dupont_3factor(
        net_income="93736",
        revenue="391035",
        total_assets="364980",
        equity="66782",
        precision=4,
    )
    assert res.ok is True
    assert res.points_value is not None
    # Direct ROE = 93736 / 66782 = 1.4036
    assert abs(res.points_value - Decimal("1.4036")) < Decimal("0.001")
    assert res.details is not None
    assert "net_margin" in res.details
    assert "asset_turnover" in res.details
    assert "equity_multiplier" in res.details


def test_dupont_3factor_division_by_zero():
    res = compute_dupont_3factor("100", "0", "500", "200")
    assert res.ok is False
    assert "non-zero revenue" in (res.error or "")

    res2 = compute_dupont_3factor("100", "1000", "0", "200")
    assert res2.ok is False
    assert "non-zero assets" in (res2.error or "")

    res3 = compute_dupont_3factor("100", "1000", "500", "0")
    assert res3.ok is False
    assert "non-zero equity" in (res3.error or "")


def test_dupont_adapter_in_registry():
    entry = get_operation_entry(CalculationOperation.DUPONT_3FACTOR)
    assert entry is not None
    assert supports(CalculationOperation.DUPONT_3FACTOR) is True
    assert "dupont_3factor" in _OPERATION_VALUES

    operands = (
        CalculationOperand("s1", Decimal("93736"), "USD", "c1", "net income"),
        CalculationOperand("s2", Decimal("391035"), "USD", "c2", "revenue"),
        CalculationOperand("s3", Decimal("364980"), "USD", "c3", "total assets"),
        CalculationOperand("s4", Decimal("66782"), "USD", "c4", "equity"),
    )
    res = entry.func(operands, 4)
    assert res.ok is True
    assert abs(res.points_value - Decimal("1.4036")) < Decimal("0.001")


# ---------------------------------------------------------------------------
# 2. FX Currency Conversion
# ---------------------------------------------------------------------------

def test_compute_fx_conversion_benchmark_rates():
    # 100 EUR to USD (1 EUR = 1.08 USD) -> 108 USD
    res = compute_fx_conversion("100", from_currency="EUR", to_currency="USD", precision=2)
    assert res.ok is True
    assert res.points_value == Decimal("108.00")
    assert res.unit == "USD"

    # 100 USD to CNY (1 USD ~ 7.14 CNY -> 0.14 rate)
    res_cny = compute_fx_conversion("100", from_currency="USD", to_currency="CNY", precision=2)
    assert res_cny.ok is True
    assert res_cny.points_value is not None


def test_compute_fx_conversion_custom_rate():
    # 1000 USD to EUR with custom rate 0.92 -> 920.00 EUR
    res = compute_fx_conversion("1000", from_currency="USD", to_currency="EUR", fx_rate="0.92")
    assert res.ok is True
    assert res.points_value == Decimal("920.00")
    assert res.unit == "EUR"


def test_fx_conversion_adapter_in_registry():
    entry = get_operation_entry(CalculationOperation.FX_CONVERSION)
    assert entry is not None
    assert supports(CalculationOperation.FX_CONVERSION) is True
    assert "fx_conversion" in _OPERATION_VALUES

    operands = (
        CalculationOperand("s1", Decimal("100"), "EUR", "c1", "revenue"),
        CalculationOperand("s2", Decimal("1.08"), "USD", "c2", "rate"),
    )
    res = entry.func(operands, 2)
    assert res.ok is True
    assert res.points_value == Decimal("108.00")


# ---------------------------------------------------------------------------
# 3. Scenario Growth Projection
# ---------------------------------------------------------------------------

def test_project_scenario_growth_compounding():
    # 100 growing at 10% for 2 periods -> 100 * 1.1^2 = 121
    res = project_scenario_growth("100", "0.10", periods=2, precision=2)
    assert res.ok is True
    assert res.points_value == Decimal("121.00")

    # 100 growing at 5% (written as 5%) for 1 period -> 105
    res2 = project_scenario_growth("100", "5%", periods=1, precision=2)
    assert res2.ok is True
    assert res2.points_value == Decimal("105.00")


def test_scenario_projection_adapter_in_registry():
    entry = get_operation_entry(CalculationOperation.SCENARIO_PROJECTION)
    assert entry is not None
    assert supports(CalculationOperation.SCENARIO_PROJECTION) is True
    assert "scenario_projection" in _OPERATION_VALUES

    operands = (
        CalculationOperand("s1", Decimal("100"), "USD", "c1", "revenue"),
        CalculationOperand("s2", Decimal("0.10"), "ratio", "c2", "growth rate"),
        CalculationOperand("s3", Decimal("3"), "base", "c3", "periods"),
    )
    # 100 * 1.1^3 = 133.10
    res = entry.func(operands, 2)
    assert res.ok is True
    assert res.points_value == Decimal("133.10")


# ---------------------------------------------------------------------------
# 4. Table Cell Coordinates and PDF Visual Bounding Box
# ---------------------------------------------------------------------------

def test_structured_citations_cell_coordinates_and_bbox():
    packet = {
        "evidence_id": "test_ev_01",
        "citation_id": "test_cite_01",
        "filename": "apple-10k-2024.pdf",
        "page": 43,
        "type": "table_cell",
        "row_idx": 12,
        "col_idx": 2,
        "row_label": "Deferred Revenue",
        "col_label": "FY2024",
        "table_title": "Consolidated Balance Sheets",
        "bbox": [150, 40, 165, 320, 43],
    }

    class MockState:
        def __init__(self, packets):
            self.evidence_packets = packets

    state = MockState([packet])
    citations = _structured_citations(state, ["test_ev_01"], {"test_cite_01"})
    assert len(citations) == 1
    c = citations[0]
    assert c["citation_id"] == "test_cite_01"
    assert c["filename"] == "apple-10k-2024.pdf"
    assert c["page"] == 43
    assert "cell_coordinates" in c
    coords = c["cell_coordinates"]
    assert coords["row_idx"] == 12
    assert coords["col_idx"] == 2
    assert coords["row_label"] == "Deferred Revenue"
    assert coords["col_label"] == "FY2024"
    assert coords["table_title"] == "Consolidated Balance Sheets"
    assert c.get("bbox") == [150, 40, 165, 320, 43]


# ---------------------------------------------------------------------------
# 5. US-GAAP Taxonomy Mapping
# ---------------------------------------------------------------------------

def test_usgaap_taxonomy_normalization():
    assert normalize_metric("us-gaap:Revenues") == "total revenue"
    assert normalize_metric("us-gaap:SalesRevenueNet") == "total revenue"
    assert normalize_metric("us-gaap:OperatingIncomeLoss") == "operating income"
    assert normalize_metric("us-gaap:NetIncomeLoss") == "net income"
    assert normalize_metric("us-gaap:GrossProfit") == "gross profit"
    assert normalize_metric("us-gaap:Assets") == "total assets"
    assert normalize_metric("us-gaap:Liabilities") == "total liabilities"
    assert normalize_metric("us-gaap:StockholdersEquity") == "stockholders' equity"
    assert normalize_metric("us-gaap:DeferredRevenueCurrent") == "deferred revenue"
    assert normalize_metric("us-gaap:ContractWithCustomerLiabilityCurrent") == "deferred revenue"
    # Plain text also remains preserved
    assert normalize_metric("Net Income") == "net income"
    assert normalize_metric("Revenue") == "total revenue"


# ---------------------------------------------------------------------------
# 6. SEC 8-K Event Stream & Monitor
# ---------------------------------------------------------------------------

def test_edgar_monitor_8k_ingestion():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_facts.sqlite"
        monitor = EdgarFeedMonitor(db_path=db_file)

        # Mark an 8-K filing as ingested with item summary
        event_summary = f"Item 2.02: {SEC_8K_ITEM_DESCRIPTIONS['2.02']}"
        monitor.mark_filing_ingested(
            ticker="AAPL",
            accession_number="0000320193-24-000099",
            form="8-K",
            filing_date="2024-10-31",
            facts_count=0,
            event_summary=event_summary,
        )

        ingested = monitor.get_ingested_accession_numbers()
        assert "0000320193-24-000099" in ingested

        # Idempotent re-run
        monitor.mark_filing_ingested(
            ticker="AAPL",
            accession_number="0000320193-24-000099",
            form="8-K",
            filing_date="2024-10-31",
            facts_count=0,
            event_summary=event_summary,
        )
        assert len(monitor.get_ingested_accession_numbers()) == 1
