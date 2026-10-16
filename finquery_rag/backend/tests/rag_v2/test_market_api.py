"""Unit tests for MarketDataProvider and quote retrieval."""

from rag_v2.sources.market_api import (
    LiveMarketProvider,
    SandboxMarketProvider,
    normalize_market_metric,
    normalize_ticker,
)


def test_market_normalizations() -> None:
    assert normalize_market_metric("P/E") == "pe_ratio"
    assert normalize_market_metric("PE Ratio") == "pe_ratio"
    assert normalize_market_metric("Stock Price") == "stock_price"
    assert normalize_market_metric("Market Cap") == "market_cap"

    assert normalize_ticker("Apple Inc.") == "AAPL"
    assert normalize_ticker("Tesla") == "TSLA"
    assert normalize_ticker("NVDA") == "NVDA"


def test_sandbox_market_provider() -> None:
    provider = SandboxMarketProvider()

    # Query existing quote
    quote_pe = provider.get_quote("Apple", "P/E")
    assert quote_pe is not None
    assert quote_pe["metric"] == "pe_ratio"
    assert quote_pe["ticker"] == "AAPL"
    assert quote_pe["value"] == "34.5"
    assert quote_pe["unit"] == "ratio"

    # Query custom registered quote
    provider.set_quote(entity="Nvidia", metric="Stock Price", value="135.20", ticker="NVDA")
    quote_nvda = provider.get_quote("NVDA", "stock price")
    assert quote_nvda is not None
    assert quote_nvda["value"] == "135.20"

    # Convert to EvidencePacket
    packet = provider.as_evidence_packet(quote_pe, slot_id="slot_pe_1")
    assert packet["evidence_id"].startswith("market_quote_")
    assert packet["source"] == "MARKET_API_SANDBOX"
    assert packet["period"] == "REALTIME"
    assert packet["temporal"]["period_semantics"] == "INSTANT"
    assert packet["slots"] == ("slot_pe_1",)


def test_live_market_provider_fallback() -> None:
    # Set a tiny timeout to guarantee timeout/fallback behavior in offline unit testing
    provider = LiveMarketProvider(timeout_seconds=0.001)

    quote = provider.get_quote("Tesla", "PE Ratio")
    assert quote is not None
    assert quote["metric"] == "pe_ratio"
    assert quote["ticker"] == "TSLA"
    assert quote["value"] == "62.4"
