"""Unit tests for SqlFactStore structured financial fact store."""

from rag_v2.sources.sql_fact_store import SqlFactStore, normalize_metric, normalize_entity, normalize_period


def test_normalization_functions() -> None:
    assert normalize_metric("Total Net Sales") == "total revenue"
    assert normalize_metric("Revenues") == "total revenue"
    assert normalize_metric("Operating Profit") == "operating income"
    assert normalize_metric("R&D") == "research and development"

    ent, ticker = normalize_entity("Apple Inc.")
    assert ticker == "aapl"
    assert ent == "Apple Inc."

    assert normalize_period("FY2024") == "FY2024"
    assert normalize_period("2024") == "FY2024"
    assert normalize_period("Fiscal 2023") == "FY2023"


def test_sql_fact_store_upsert_and_query() -> None:
    store = SqlFactStore(":memory:")
    store.upsert_fact(
        entity="Apple",
        metric="Total Net Sales",
        period="FY2024",
        value="391035",
        unit="USD",
        scale="million",
    )

    # Query with alias
    res = store.query_fact(entity="AAPL", metric="Revenue", period="2024")
    assert res is not None
    assert res["entity"] == "Apple"
    assert res["value"] == "391035"
    assert res["metric"] == "total revenue"
    assert res["period"] == "FY2024"

    # Query missing fact
    missing = store.query_fact(entity="Apple", metric="Inventory", period="FY2020")
    assert missing is None

    store.close()


def test_sql_fact_store_seed_benchmark() -> None:
    store = SqlFactStore(":memory:")
    store.seed_benchmark_data()

    # Verify seeded Apple FY2024 net sales and net income
    rev = store.query_fact(entity="Apple", metric="Total Revenue", period="FY2024")
    assert rev is not None
    assert rev["value"] == "391035"

    ni = store.query_fact(entity="AAPL", metric="Net Income", period="FY2024")
    assert ni is not None
    assert ni["value"] == "93736"

    # Verify seeded Microsoft FY2024 total revenue
    msft_rev = store.query_fact(entity="MSFT", metric="Revenue", period="FY2024")
    assert msft_rev is not None
    assert msft_rev["value"] == "245122"

    store.close()


def test_as_evidence_packet_schema() -> None:
    store = SqlFactStore(":memory:")
    store.seed_benchmark_data()

    fact = store.query_fact("Apple", "Total Revenue", "FY2024")
    assert fact is not None
    packet = store.as_evidence_packet(fact, slot_id="slot_rev_01")

    assert packet["evidence_id"].startswith("sql_fact_")
    assert packet["metric"] == "total revenue"
    assert packet["value"] == "391035"
    assert packet["period"] == "FY2024"
    assert packet["source"] == "SQL_FACT_STORE"
    assert packet["slots"] == ("slot_rev_01",)
    assert packet["temporal"]["period_semantics"] == "ANNUAL"
    assert "391035" in packet["source_text"]

    store.close()
