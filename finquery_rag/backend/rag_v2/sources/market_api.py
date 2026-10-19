"""Market API provider for real-time and market quote retrieval."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable


_MARKET_METRIC_CANONICAL: dict[str, str] = {
    "pe ratio": "pe_ratio",
    "pe": "pe_ratio",
    "p/e": "pe_ratio",
    "trailing pe": "pe_ratio",
    "forward pe": "pe_ratio",
    "市盈率": "pe_ratio",
    "动态市盈率": "pe_ratio",
    "静态市盈率": "pe_ratio",
    "stock price": "stock_price",
    "share price": "stock_price",
    "last price": "stock_price",
    "price": "stock_price",
    "股价": "stock_price",
    "股票价格": "stock_price",
    "现价": "stock_price",
    "最新价": "stock_price",
    "收盘价": "stock_price",
    "开盘价": "stock_price",
    "market cap": "market_cap",
    "market capitalization": "market_cap",
    "市值": "market_cap",
    "总市值": "market_cap",
    "exchange rate": "exchange_rate",
    "fx rate": "exchange_rate",
    "spot rate": "exchange_rate",
    "汇率": "exchange_rate",
}

_ENTITY_TICKER_MAP: dict[str, str] = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "apple inc.": "AAPL",
    "aapl": "AAPL",
    "苹果": "AAPL",
    "microsoft": "MSFT",
    "microsoft corp": "MSFT",
    "microsoft corporation": "MSFT",
    "msft": "MSFT",
    "微软": "MSFT",
    "tesla": "TSLA",
    "tesla inc": "TSLA",
    "tesla inc.": "TSLA",
    "tsla": "TSLA",
    "特斯拉": "TSLA",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "英伟达": "NVDA",
    "byd": "1211.HK",
    "byd inc": "1211.HK",
    "比亚迪": "1211.HK",
    "byddy": "BYDDY",
    "amazon": "AMZN",
    "amazon com": "AMZN",
    "amazon.com inc": "AMZN",
    "amzn": "AMZN",
    "亚马逊": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "alphabet inc": "GOOGL",
    "googl": "GOOGL",
    "goog": "GOOGL",
    "谷歌": "GOOGL",
    "meta": "META",
    "meta platforms": "META",
    "facebook": "META",
    "tencent": "0700.HK",
    "腾讯": "0700.HK",
    "alibaba": "BABA",
    "baba": "BABA",
    "阿里巴巴": "BABA",
    "阿里": "BABA",
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


def register_market_metric(alias: str, canonical: str) -> None:
    """Register custom market metric alias dynamically."""
    _MARKET_METRIC_CANONICAL[alias.strip().lower()] = canonical.strip().lower()



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
            "currency": (
                unit.upper()
                if (len(unit) == 3 and unit.isalpha())
                else ("USD" if unit not in ("USD", "EUR", "CNY", "GBP", "JPY", "HKD", "CAD", "AUD", "CHF") else unit)
            ),
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
            {"entity": "苹果", "ticker": "AAPL", "metric": "stock_price", "value": "231.45", "unit": "USD"},
            {"entity": "苹果", "ticker": "AAPL", "metric": "pe_ratio", "value": "34.5", "unit": "ratio"},
            {"entity": "苹果", "ticker": "AAPL", "metric": "market_cap", "value": "3520000", "unit": "USD", "scale": "million"},
            # MSFT
            {"entity": "Microsoft", "ticker": "MSFT", "metric": "stock_price", "value": "428.10", "unit": "USD"},
            {"entity": "Microsoft", "ticker": "MSFT", "metric": "pe_ratio", "value": "36.2", "unit": "ratio"},
            {"entity": "Microsoft", "ticker": "MSFT", "metric": "market_cap", "value": "3180000", "unit": "USD", "scale": "million"},
            {"entity": "微软", "ticker": "MSFT", "metric": "stock_price", "value": "428.10", "unit": "USD"},
            {"entity": "微软", "ticker": "MSFT", "metric": "pe_ratio", "value": "36.2", "unit": "ratio"},
            {"entity": "微软", "ticker": "MSFT", "metric": "market_cap", "value": "3180000", "unit": "USD", "scale": "million"},
            # TSLA
            {"entity": "Tesla", "ticker": "TSLA", "metric": "stock_price", "value": "218.80", "unit": "USD"},
            {"entity": "Tesla", "ticker": "TSLA", "metric": "pe_ratio", "value": "62.4", "unit": "ratio"},
            {"entity": "Tesla", "ticker": "TSLA", "metric": "market_cap", "value": "698000", "unit": "USD", "scale": "million"},
            {"entity": "特斯拉", "ticker": "TSLA", "metric": "stock_price", "value": "218.80", "unit": "USD"},
            {"entity": "特斯拉", "ticker": "TSLA", "metric": "pe_ratio", "value": "62.4", "unit": "ratio"},
            {"entity": "特斯拉", "ticker": "TSLA", "metric": "market_cap", "value": "698000", "unit": "USD", "scale": "million"},
            # NVDA
            {"entity": "NVIDIA", "ticker": "NVDA", "metric": "stock_price", "value": "135.20", "unit": "USD"},
            {"entity": "NVIDIA", "ticker": "NVDA", "metric": "pe_ratio", "value": "48.2", "unit": "ratio"},
            {"entity": "NVIDIA", "ticker": "NVDA", "metric": "market_cap", "value": "3320000", "unit": "USD", "scale": "million"},
            {"entity": "英伟达", "ticker": "NVDA", "metric": "stock_price", "value": "135.20", "unit": "USD"},
            {"entity": "英伟达", "ticker": "NVDA", "metric": "pe_ratio", "value": "48.2", "unit": "ratio"},
            {"entity": "英伟达", "ticker": "NVDA", "metric": "market_cap", "value": "3320000", "unit": "USD", "scale": "million"},
            # BYD / 比亚迪
            {"entity": "BYD", "ticker": "1211.HK", "metric": "stock_price", "value": "73.75", "unit": "HKD"},
            {"entity": "BYD", "ticker": "1211.HK", "metric": "pe_ratio", "value": "21.8", "unit": "ratio"},
            {"entity": "BYD", "ticker": "1211.HK", "metric": "market_cap", "value": "765000", "unit": "HKD", "scale": "million"},
            {"entity": "比亚迪", "ticker": "1211.HK", "metric": "stock_price", "value": "73.75", "unit": "HKD"},
            {"entity": "比亚迪", "ticker": "1211.HK", "metric": "pe_ratio", "value": "21.8", "unit": "ratio"},
            {"entity": "比亚迪", "ticker": "1211.HK", "metric": "market_cap", "value": "765000", "unit": "HKD", "scale": "million"},
            # Tencent / 腾讯
            {"entity": "Tencent", "ticker": "0700.HK", "metric": "stock_price", "value": "425.00", "unit": "HKD"},
            {"entity": "Tencent", "ticker": "0700.HK", "metric": "pe_ratio", "value": "22.5", "unit": "ratio"},
            {"entity": "Tencent", "ticker": "0700.HK", "metric": "market_cap", "value": "3980000", "unit": "HKD", "scale": "million"},
            {"entity": "腾讯", "ticker": "0700.HK", "metric": "stock_price", "value": "425.00", "unit": "HKD"},
            # Alibaba / 阿里巴巴
            {"entity": "Alibaba", "ticker": "BABA", "metric": "stock_price", "value": "102.50", "unit": "USD"},
            {"entity": "Alibaba", "ticker": "BABA", "metric": "pe_ratio", "value": "16.4", "unit": "ratio"},
            {"entity": "Alibaba", "ticker": "BABA", "metric": "market_cap", "value": "245000", "unit": "USD", "scale": "million"},
            {"entity": "阿里巴巴", "ticker": "BABA", "metric": "stock_price", "value": "102.50", "unit": "USD"},
            {"entity": "阿里", "ticker": "BABA", "metric": "stock_price", "value": "102.50", "unit": "USD"},
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


# =========================================================================
# FX Rate Provider & Cross-Currency Support
# =========================================================================

_DEFAULT_FX_RATES: dict[tuple[str, str], Decimal] = {
    ("EUR", "USD"): Decimal("1.08"),
    ("USD", "EUR"): Decimal("0.9259"),
    ("GBP", "USD"): Decimal("1.27"),
    ("USD", "GBP"): Decimal("0.7874"),
    ("JPY", "USD"): Decimal("0.0067"),
    ("USD", "JPY"): Decimal("149.25"),
    ("CNY", "USD"): Decimal("0.138"),
    ("USD", "CNY"): Decimal("7.246"),
    ("HKD", "USD"): Decimal("0.128"),
    ("USD", "HKD"): Decimal("7.8125"),
    ("CAD", "USD"): Decimal("0.73"),
    ("USD", "CAD"): Decimal("1.3698"),
    ("AUD", "USD"): Decimal("0.66"),
    ("USD", "AUD"): Decimal("1.5152"),
    ("CHF", "USD"): Decimal("1.12"),
    ("USD", "CHF"): Decimal("0.8928"),
}

_CUSTOM_FX_RATES: dict[tuple[str, str], Decimal] = {}


def register_fx_rate(from_curr: str, to_curr: str, rate: Decimal | str | float) -> None:
    """Register custom FX conversion rate dynamically."""
    f = from_curr.strip().upper()
    t = to_curr.strip().upper()
    dec_rate = Decimal(str(rate))
    _CUSTOM_FX_RATES[(f, t)] = dec_rate
    if dec_rate > Decimal("0"):
        _CUSTOM_FX_RATES[(t, f)] = Decimal("1") / dec_rate


def get_fx_rate(from_curr: str, to_curr: str) -> Decimal | None:
    """Get currency exchange rate from from_curr to to_curr with triangular cross-rate resolution."""
    f = from_curr.strip().upper()
    t = to_curr.strip().upper()
    if f == t:
        return Decimal("1")
    if (f, t) in _CUSTOM_FX_RATES:
        return _CUSTOM_FX_RATES[(f, t)]
    if (f, t) in _DEFAULT_FX_RATES:
        return _DEFAULT_FX_RATES[(f, t)]

    # Standard triangular cross-rate conversion via USD base currency
    if f != "USD" and t != "USD":
        rate_to_usd = get_fx_rate(f, "USD")
        rate_from_usd = get_fx_rate("USD", t)
        if rate_to_usd is not None and rate_from_usd is not None:
            return rate_to_usd * rate_from_usd

    return None


# =========================================================================
# Market API Resilience: TTL Cache & Circuit Breaker
# =========================================================================

class CachedMarketProvider(BaseMarketProvider):
    """In-memory thread-safe TTL cache wrapping any MarketDataProvider."""

    def __init__(
        self,
        provider: MarketDataProvider,
        ttl_seconds: float = 300.0,
    ) -> None:
        self.provider = provider
        self.ttl_seconds = ttl_seconds
        self._cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}

    def get_quote(self, entity: str, metric: str) -> dict[str, Any] | None:
        key = (normalize_ticker(entity), normalize_market_metric(metric))
        now = time.time()
        if key in self._cache:
            ts, quote = self._cache[key]
            if now - ts <= self.ttl_seconds:
                # Return cached quote with cache indicator
                cached_copy = dict(quote)
                cached_copy["is_cached"] = True
                return cached_copy

        # Cache miss or expired
        quote = self.provider.get_quote(entity, metric)
        if quote is not None:
            self._cache[key] = (now, quote)
        return quote

    def clear_cache(self) -> None:
        self._cache.clear()


class CircuitBreakerMarketProvider(BaseMarketProvider):
    """Circuit breaker wrapper protecting pipeline from external API downtime.

    States:
      - CLOSED: Normal operation, routing requests to provider.
      - OPEN: Failures exceeded threshold. Bypasses external calls for recovery_timeout.
      - HALF_OPEN: Probe call allowed to check recovery.
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        fallback_provider: MarketDataProvider | None = None,
    ) -> None:
        self.provider = provider
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.fallback_provider = fallback_provider or SandboxMarketProvider()
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def get_quote(self, entity: str, metric: str) -> dict[str, Any] | None:
        now = time.time()

        if self.state == "OPEN":
            if now - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                # Breaker open: route directly to fallback
                res = self.fallback_provider.get_quote(entity, metric)
                if res:
                    res = dict(res)
                    res["circuit_breaker"] = "OPEN_FALLBACK"
                return res

        try:
            quote = self.provider.get_quote(entity, metric)
            if quote is not None:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return quote
            else:
                self._record_failure(now)
        except Exception:
            self._record_failure(now)

        # Fallback invocation
        return self.fallback_provider.get_quote(entity, metric)

    def _record_failure(self, now: float) -> None:
        self.failure_count += 1
        self.last_failure_time = now
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

