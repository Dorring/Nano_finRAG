"""Unit tests for MultiSourceRetrievalAdapter unified dispatch."""

from typing import Any
from collections.abc import Mapping

from rag_v2.adaptive.adaptive_contracts import (
    AdaptiveRAGStateV1,
    ReasonCode,
    ReplanActionV1,
    ToolCapability,
)
from rag_v2.routing.source_router import DataSourceType
from rag_v2.sources.market_api import SandboxMarketProvider
from rag_v2.sources.multi_source_adapter import MultiSourceRetrievalAdapter
from rag_v2.sources.sql_fact_store import SqlFactStore


class FakePdfRetriever:
    """Mock PDF retriever for testing multi-source delegation and fallback."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_target_slots: tuple[str, ...] = ()

    def retrieve(self, action: ReplanActionV1, state: AdaptiveRAGStateV1) -> list[dict[str, Any]]:
        self.call_count += 1
        self.last_target_slots = tuple(action.target_slots)
        results = []
        for sid in action.target_slots:
            results.append({
                "evidence_id": f"pdf_doc_{sid}",
                "metric": "pdf_fallback_metric",
                "value": "12345",
                "period": "FY2024",
                "entity": "Apple",
                "slots": (sid,),
                "source": "PDF_RETRIEVAL",
            })
        return results


def _make_state(slots: list[dict[str, Any]]) -> AdaptiveRAGStateV1:
    return AdaptiveRAGStateV1.new(
        request_id="req_multi_01",
        query="Apple multi-source query",
        required_slots=slots,
    )


def test_sql_routing_and_hit() -> None:
    sql_store = SqlFactStore(":memory:")
    sql_store.seed_benchmark_data()
    pdf_retriever = FakePdfRetriever()

    adapter = MultiSourceRetrievalAdapter(
        sql_store=sql_store,
        market_provider=None,
        pdf_retriever=pdf_retriever,
    )

    state = _make_state([
        {
            "slot_id": "s1",
            "entity": "Apple",
            "metric": "Total Revenue",
            "period": "FY2024",
            "source_preference": DataSourceType.STRUCTURED_STORE.value,
        }
    ])

    action = ReplanActionV1(
        capability=ToolCapability.STRUCTURED_FINANCIAL_LOOKUP,
        query="Apple FY2024 Total Revenue",
        reason_code=ReasonCode.MISSING_SLOT,
        target_slots=("s1",),
        constraints={"source_preference": DataSourceType.STRUCTURED_STORE.value},
    )

    results = adapter.retrieve(action, state)
    assert len(results) == 1
    assert results[0]["value"] == "391035"
    assert results[0]["source"] == "SQL_FACT_STORE"
    assert pdf_retriever.call_count == 0
    assert adapter.last_dispatch_stats["sql_hits"] == 1

    sql_store.close()


def test_market_api_routing() -> None:
    market_provider = SandboxMarketProvider()
    adapter = MultiSourceRetrievalAdapter(
        sql_store=None,
        market_provider=market_provider,
        pdf_retriever=None,
    )

    state = _make_state([
        {
            "slot_id": "s_pe",
            "entity": "Apple",
            "metric": "P/E",
            "period": "REALTIME",
            "source_preference": DataSourceType.MARKET_API.value,
        }
    ])

    action = ReplanActionV1(
        capability=ToolCapability.EXTERNAL_WEB_SEARCH,
        query="Apple PE Ratio",
        reason_code=ReasonCode.MISSING_SLOT,
        target_slots=("s_pe",),
        constraints={"source_preference": DataSourceType.MARKET_API.value},
    )

    results = adapter.retrieve(action, state)
    assert len(results) == 1
    assert results[0]["value"] == "34.5"
    assert results[0]["source"] == "MARKET_API_SANDBOX"
    assert adapter.last_dispatch_stats["market_hits"] == 1


def test_mixed_heterogeneous_plan_dispatch() -> None:
    sql_store = SqlFactStore(":memory:")
    sql_store.seed_benchmark_data()
    market_provider = SandboxMarketProvider()
    pdf_retriever = FakePdfRetriever()

    adapter = MultiSourceRetrievalAdapter(
        sql_store=sql_store,
        market_provider=market_provider,
        pdf_retriever=pdf_retriever,
    )

    state = _make_state([
        {
            "slot_id": "s_sql",
            "entity": "Apple",
            "metric": "Net Income",
            "period": "FY2024",
            "source_preference": DataSourceType.STRUCTURED_STORE.value,
        },
        {
            "slot_id": "s_mkt",
            "entity": "Apple",
            "metric": "Stock Price",
            "period": "REALTIME",
            "source_preference": DataSourceType.MARKET_API.value,
        },
        {
            "slot_id": "s_pdf",
            "entity": "Apple",
            "metric": "Services Segment Revenue",
            "period": "FY2024",
            "source_preference": DataSourceType.FOOTNOTE_DISCLOSURES.value,
        },
    ])

    action = ReplanActionV1(
        capability=ToolCapability.SEMANTIC_RETRIEVAL,
        query="Mixed query",
        reason_code=ReasonCode.MISSING_SLOT,
        target_slots=("s_sql", "s_mkt", "s_pdf"),
    )

    results = adapter.retrieve(action, state)
    assert len(results) == 3

    sources = {r["source"] for r in results}
    assert "SQL_FACT_STORE" in sources
    assert "MARKET_API_SANDBOX" in sources
    assert "PDF_RETRIEVAL" in sources

    assert adapter.last_dispatch_stats["sql_hits"] == 1
    assert adapter.last_dispatch_stats["market_hits"] == 1
    assert adapter.last_dispatch_stats["pdf_calls"] == 1
    assert pdf_retriever.last_target_slots == ("s_pdf",)

    sql_store.close()


def test_sql_miss_fallback_to_pdf() -> None:
    sql_store = SqlFactStore(":memory:")
    # Empty store, no benchmark seeded
    pdf_retriever = FakePdfRetriever()

    adapter = MultiSourceRetrievalAdapter(
        sql_store=sql_store,
        market_provider=None,
        pdf_retriever=pdf_retriever,
        enable_sql_fallback=True,
    )

    state = _make_state([
        {
            "slot_id": "s_unknown",
            "entity": "Apple",
            "metric": "Nonexistent Metric",
            "period": "FY2024",
            "source_preference": DataSourceType.STRUCTURED_STORE.value,
        }
    ])

    action = ReplanActionV1(
        capability=ToolCapability.STRUCTURED_FINANCIAL_LOOKUP,
        query="Apple Nonexistent Metric",
        reason_code=ReasonCode.MISSING_SLOT,
        target_slots=("s_unknown",),
    )

    results = adapter.retrieve(action, state)
    assert len(results) == 1
    assert results[0]["source"] == "PDF_RETRIEVAL"
    assert adapter.last_dispatch_stats["sql_misses"] == 1
    assert adapter.last_dispatch_stats["pdf_calls"] == 1
    assert pdf_retriever.last_target_slots == ("s_unknown",)

    sql_store.close()
