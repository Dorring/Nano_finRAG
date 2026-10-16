"""Market API provider for real-time and market quote retrieval."""

from __future__ import annotations

import re
import urllib.request
import json
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


_MARKET_METRIC_CANONICAL: dict[str, str] = {
    "pe ratio": "pe_ratio",
    "pe": "pe_ratio",
    "p/e": "pe_ratio",
    "trailing pe": "pe_ratio",
    "forward pe": "pe_ratio",
    "stock price": "stock_price",
    "share price": "stock_price",
    "last price": "stock_price",
    "price": "stock_price",
    "market cap": "market_cap",
    "market capitalization": "market_cap",
    "exchange rate": "exchange_rate",
    "fx rate": "exchange_rate",
    "spot rate": "exchange_rate",
}

_ENTITY_TICKER_MAP: dict[str, str] = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "apple inc.": "AAPL",
    "aapl": "AAPL",
    "microsoft": "MSFT",
    "microsoft corp": "MSFT",
    "microsoft corporation": "MSFT",
    "msft": "MSFT",
    "tesla": "TSLA",
    "tesla inc": "TSLA",
    "tesla inc.": "TSLA",
    "tsla": "TSLA",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "amazon": "AMZN",
    "amazon com": "AMZN",
    "amazon.com inc": "AMZN",
    "amzn": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "alphabet inc": "GOOGL",
    "googl": "GOOGL",
    "goog": "GOOGL",
    "meta": "META",
    "meta platforms": "META",
    "facebook": "META",
    "jpmorgan": "JPM",
    "jpmorgan chase": "JPM",
    "jpmorgan chase & co": "JPM",
    "jpm": "JPM",
    "coca-cola": "KO",
    "coca cola": "KO",
    "coke": "KO",
    "ko": "KO",
    "walmart": "WMT",
    "walmart inc": "WMT",
    "wmt": "WMT",
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "exxon mobil": "XOM",
    "xom": "XOM",
    "netflix": "NFLX",
    "nflx": "NFLX",
}


def register_ticker(entity_name: str, ticker: str) -> None:
    """Register custom entity name to standard ticker mapping dynamically."""
    _ENTITY_TICKER_MAP[entity_name.strip().lower()] = ticker.strip().upper()


def normalize_market_metric(metric: str) -> str:
    """Normalize market metric name into canonical key."""
    cleaned = re.sub(r"[^\w\s/]", "", metric.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return _MARKET_METRIC_CANONICAL.get(cleaned, cleaned)


def normalize_ticker(entity: str) -> str:
    """Resolve entity string to standard uppercase ticker."""
    clean = entity.strip().lower()
    return _ENTITY_TICKER_MAP.get(clean, entity.strip().upper())


@runtime_checkable

class MarketDataProvider(Protocol):
    """Protocol for financial market data providers."""

    def get_quote(self, entity: str, metric: str) -> dict[str, Any] | None:
        """Fetch quote for entity and metric."""
        ...

    def as_evidence_packet(
        self,
        quote: Mapping[str, Any],
        slot_id: str,
        candidate_id: str | None = None,
    ) -> dict[str, Any]:
        """Convert quote into standard EvidencePacket."""
        ...


class BaseMarketProvider:
    """Base helper providing EvidencePacket formatting."""

    def as_evidence_packet(
        self,
        quote: Mapping[str, Any],
        slot_id: str,
        candidate_id: str | None = None,
    ) -> dict[str, Any]:
        entity = quote["entity"]
        ticker = quote.get("ticker", entity)
        metric = quote["metric"]
        value = str(quote["value"])
        unit = quote.get("unit", "USD")
        cid = candidate_id or f"market_quote_{ticker}_{metric}"
        source_label = quote.get("source", "MARKET_API")

        return {
            "evidence_id": cid,
            "candidate_id": cid,
            "metric": metric,
            "value": value,
            "period": "REALTIME",
            "entity": entity,
            "entity_id": ticker.lower(),
            "scope": "consolidated",
            "unit": unit,
            "currency": unit if unit in ("USD", "EUR", "CNY", "GBP") else "USD",
            "scale": quote.get("scale", "units"),
            "source": source_label,
            "document_id": "Market Data Provider",
            "citation_id": f"[Market Data Provider] {entity} ({ticker}) Real-time {metric}",
            "page": 0,
            "slots": (str(slot_id),),
            "slot_ids": (str(slot_id),),
            "source_text": (
                f"{entity} ({ticker}) real-time {metric} is {value} {unit}."
            ),
            "temporal": {
                "entity": entity,
                "metric": metric,
                "value": value,
                "period_semantics": "INSTANT",
                "source": source_label,
            },
            "metadata": {
                "provider": "market_data_api",
                "ticker": ticker,
                "timestamp": quote.get("timestamp"),
            },
        }


class SandboxMarketProvider(BaseMarketProvider):
    """Deterministic yet configurable market provider for testing and sandbox execution."""

    def __init__(self) -> None:
        self._quotes: dict[tuple[str, str], dict[str, Any]] = {}
        self._seed_default_quotes()

    def _seed_default_quotes(self) -> None:
        default_data = [
            # AAPL
            {"entity": "Apple", "ticker": "AAPL", "metric": "stock_price", "value": "231.45", "unit": "USD"},
            {"entity": "Apple", "ticker": "AAPL", "metric": "pe_ratio", "value": "34.5", "unit": "ratio"},
            {"entity": "Apple", "ticker": "AAPL", "metric": "market_cap", "value": "3520000", "unit": "USD", "scale": "million"},
            # MSFT
            {"entity": "Microsoft", "ticker": "MSFT", "metric": "stock_price", "value": "428.10", "unit": "USD"},
            {"entity": "Microsoft", "ticker": "MSFT", "metric": "pe_ratio", "value": "36.2", "unit": "ratio"},
            {"entity": "Microsoft", "ticker": "MSFT", "metric": "market_cap", "value": "3180000", "unit": "USD", "scale": "million"},
            # TSLA
            {"entity": "Tesla", "ticker": "TSLA", "metric": "stock_price", "value": "218.80", "unit": "USD"},
            {"entity": "Tesla", "ticker": "TSLA", "metric": "pe_ratio", "value": "62.4", "unit": "ratio"},
            {"entity": "Tesla", "ticker": "TSLA", "metric": "market_cap", "value": "698000", "unit": "USD", "scale": "million"},
        ]
        for item in default_data:
            self.set_quote(
                entity=item["entity"],
                metric=item["metric"],
                value=item["value"],
                ticker=item.get("ticker"),
                unit=item.get("unit", "USD"),
                scale=item.get("scale", "units"),
            )

    def set_quote(
        self,
        entity: str,
        metric: str,
        value: str | int | float,
        ticker: str | None = None,
        unit: str = "USD",
        scale: str = "units",
    ) -> None:
        """Register or update a quote in the sandbox."""
        resolved_ticker = ticker or normalize_ticker(entity)
        norm_m = normalize_market_metric(metric)
        self._quotes[(resolved_ticker, norm_m)] = {
            "entity": entity,
            "ticker": resolved_ticker,
            "metric": norm_m,
            "value": str(value),
            "unit": unit,
            "scale": scale,
            "source": "MARKET_API_SANDBOX",
            "timestamp": "2026-10-16T12:00:00Z",
        }

    def get_quote(self, entity: str, metric: str) -> dict[str, Any] | None:
        """Retrieve market quote."""
        resolved_ticker = normalize_ticker(entity)
        norm_m = normalize_market_metric(metric)
        return self._quotes.get((resolved_ticker, norm_m))


class LiveMarketProvider(BaseMarketProvider):
    """External live market provider with graceful timeout and sandbox fallback."""

    def __init__(
        self,
        timeout_seconds: float = 2.0,
        fallback_provider: MarketDataProvider | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.fallback_provider = fallback_provider or SandboxMarketProvider()

    def get_quote(self, entity: str, metric: str) -> dict[str, Any] | None:
        """Fetch quote from live endpoint with timeout and fallback."""
        ticker = normalize_ticker(entity)
        norm_m = normalize_market_metric(metric)

        # Attempt public market lookup if network is reachable
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Nano_finRAG/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result = data.get("chart", {}).get("result", [{}])[0]
                meta = result.get("meta", {})
                regular_price = meta.get("regularMarketPrice")

                if regular_price is not None and norm_m == "stock_price":
                    return {
                        "entity": entity,
                        "ticker": ticker,
                        "metric": "stock_price",
                        "value": str(regular_price),
                        "unit": meta.get("currency", "USD"),
                        "scale": "units",
                        "source": "MARKET_API_LIVE",
                        "timestamp": "REALTIME",
                    }
        except Exception:
            # On network disconnect / timeout, safely fall back to the fallback provider
            pass

        if self.fallback_provider is not None:
            return self.fallback_provider.get_quote(entity, metric)
        return None
