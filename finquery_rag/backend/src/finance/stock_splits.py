"""Stock split and share dilution adjustment for per-share financial metrics.

Manages corporate stock splits (e.g. NVDA 10:1 in June 2024, AAPL 4:1 in August 2020)
and computes split adjustment factors to prevent 10x valuation multiple / P/E distortions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence


@dataclass(frozen=True)
class StockSplitEvent:
    """Historical stock split event."""

    ticker: str
    effective_date: str  # YYYY-MM-DD
    ratio: Decimal       # e.g. 10 for 10-for-1 split, 4 for 4-for-1 split


# Default known historical stock splits (expandable dynamically)
_KNOWN_STOCK_SPLITS: dict[str, list[StockSplitEvent]] = {
    "NVDA": [StockSplitEvent("NVDA", "2024-06-07", Decimal("10")), StockSplitEvent("NVDA", "2021-07-20", Decimal("4"))],
    "AAPL": [StockSplitEvent("AAPL", "2020-08-31", Decimal("4")), StockSplitEvent("AAPL", "2014-06-09", Decimal("7"))],
    "TSLA": [StockSplitEvent("TSLA", "2022-08-25", Decimal("3")), StockSplitEvent("TSLA", "2020-08-31", Decimal("5"))],
    "GOOGL": [StockSplitEvent("GOOGL", "2022-07-15", Decimal("20"))],
    "AMZN": [StockSplitEvent("AMZN", "2022-06-06", Decimal("20"))],
}

_CUSTOM_STOCK_SPLITS: dict[str, list[StockSplitEvent]] = {}


def register_stock_split(ticker: str, effective_date: str, split_ratio: Decimal | int | str) -> None:
    """Register custom stock split dynamically."""
    t = ticker.strip().upper()
    event = StockSplitEvent(t, effective_date.strip(), Decimal(str(split_ratio)))
    if t not in _CUSTOM_STOCK_SPLITS:
        _CUSTOM_STOCK_SPLITS[t] = []
    _CUSTOM_STOCK_SPLITS[t].append(event)


def get_split_adjustment_factor(
    ticker: str,
    from_date: str,
    to_date: str,
) -> Decimal:
    """Calculate cumulative split adjustment factor between two dates.

    If a 10-for-1 split occurred between from_date and to_date, factor is 10.
    Historical per-share metric (pre-split EPS) must be divided by this factor
    to compare with post-split market stock price.
    """
    t = ticker.strip().upper()
    splits = _CUSTOM_STOCK_SPLITS.get(t, []) + _KNOWN_STOCK_SPLITS.get(t, [])
    if not splits:
        return Decimal("1")

    start = min(from_date, to_date)
    end = max(from_date, to_date)

    cumulative_factor = Decimal("1")
    for event in splits:
        if start <= event.effective_date <= end:
            cumulative_factor *= event.ratio

    return cumulative_factor


def adjust_per_share_metric(
    ticker: str,
    metric_name: str,
    raw_value: Decimal,
    from_date: str,
    to_date: str,
) -> tuple[Decimal, Decimal, bool]:
    """Adjust per-share fundamental metric (e.g. EPS, DPS) for stock splits.

    Returns:
        (adjusted_value, split_factor, has_split)
    """
    is_per_share = any(k in metric_name.lower() for k in ("eps", "per share", "dividend per share", "earnings per share"))
    if not is_per_share:
        return raw_value, Decimal("1"), False

    factor = get_split_adjustment_factor(ticker, from_date, to_date)
    if factor != Decimal("1"):
        # Pre-split EPS -> Post-split comparable EPS
        adjusted_value = raw_value / factor
        return adjusted_value, factor, True

    return raw_value, Decimal("1"), False
