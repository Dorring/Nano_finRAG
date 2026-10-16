"""Fiscal Calendar and Temporal Alignment for Financial RAG.

Handles differing corporate fiscal year calendars (e.g. Apple ends in Sept,
Microsoft in June, NVIDIA in January) and diagnoses temporal drift in
cross-company comparisons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


# Standard fiscal year end months (1-12) for major public entities
_DEFAULT_FISCAL_YEAR_END_MONTHS: dict[str, int] = {
    "aapl": 9,      # September
    "apple": 9,
    "msft": 6,      # June
    "microsoft": 6,
    "nvda": 1,      # January (FY2024 ended Jan 2024 / Jan 2025)
    "nvidia": 1,
    "wmt": 1,       # January
    "walmart": 1,
    "amzn": 12,     # December
    "amazon": 12,
    "googl": 12,    # December
    "google": 12,
    "alphabet": 12,
    "meta": 12,     # December
    "tsla": 12,     # December
    "tesla": 12,
    "ko": 12,       # December
    "coca-cola": 12,
    "nflx": 12,     # December
    "netflix": 12,
    "jpm": 12,      # December
    "jpmorgan": 12,
}

_CUSTOM_FISCAL_YEAR_END_MONTHS: dict[str, int] = {}


def register_fiscal_year_end(entity: str, end_month: int) -> None:
    """Register custom entity fiscal year end month (1 to 12) dynamically."""
    if not (1 <= end_month <= 12):
        raise ValueError(f"Fiscal year end month must be between 1 and 12, got {end_month}")
    _CUSTOM_FISCAL_YEAR_END_MONTHS[entity.strip().lower()] = end_month


def get_fiscal_year_end_month(entity: str) -> int:
    """Get the fiscal year end month for an entity (default 12 if unknown)."""
    clean = entity.strip().lower()
    if clean in _CUSTOM_FISCAL_YEAR_END_MONTHS:
        return _CUSTOM_FISCAL_YEAR_END_MONTHS[clean]
    return _DEFAULT_FISCAL_YEAR_END_MONTHS.get(clean, 12)


def extract_year(period: str) -> int | None:
    """Extract numeric year from period string (e.g. FY2024, 2024, FY24, CY23, '24)."""
    p = period.strip()
    m4 = re.search(r"\b((?:19|20)\d{2})\b", p)
    if m4:
        return int(m4.group(1))
    m2 = re.search(r"(?:fy|cy|['’])\s*(\d{2})\b", p, re.IGNORECASE)
    if m2:
        val = int(m2.group(1))
        return 2000 + val if val < 70 else 1900 + val
    return None


@dataclass(frozen=True)
class FiscalAlignmentReport:
    """Report detailing temporal alignment between two financial entities/periods."""

    entity_a: str
    period_a: str
    month_a: int
    entity_b: str
    period_b: str
    month_b: int
    drift_months: int
    is_aligned: bool
    warning: str | None = None
    recommendation: str | None = None

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return {
            "entity_a": self.entity_a,
            "period_a": self.period_a,
            "month_a": self.month_a,
            "entity_b": self.entity_b,
            "period_b": self.period_b,
            "month_b": self.month_b,
            "drift_months": self.drift_months,
            "is_aligned": self.is_aligned,
            "warning": self.warning,
            "recommendation": self.recommendation,
        }


def check_fiscal_alignment(
    entity_a: str,
    period_a: str,
    entity_b: str,
    period_b: str,
    *,
    drift_tolerance_months: int = 2,
) -> FiscalAlignmentReport:
    """Check whether two entities and periods are temporally comparable.

    Computes the calendar drift in months. If drift exceeds tolerance (default 2 months),
    flags an alignment warning and provides TTM recommendation.
    """
    m_a = get_fiscal_year_end_month(entity_a)
    m_b = get_fiscal_year_end_month(entity_b)

    year_a = extract_year(period_a)
    year_b = extract_year(period_b)
    if year_a is None and year_b is None:
        year_a = 0
        year_b = 0
    elif year_a is None:
        year_a = year_b
    elif year_b is None:
        year_b = year_a

    # Calculate absolute calendar month timestamp (year * 12 + month)
    abs_month_a = year_a * 12 + m_a
    abs_month_b = year_b * 12 + m_b
    drift_months = abs(abs_month_a - abs_month_b)

    is_aligned = drift_months <= drift_tolerance_months
    warning = None
    recommendation = None

    if not is_aligned:
        warning = (
            f"Temporal mismatch: {entity_a} {period_a} ends in month {m_a:02d}, "
            f"while {entity_b} {period_b} ends in month {m_b:02d} (drift of {drift_months} months)."
        )
        recommendation = (
            f"Consider aligning {entity_a} and {entity_b} using standardized TTM "
            f"(Trailing Twelve Months) or matching calendar quarter data."
        )

    return FiscalAlignmentReport(
        entity_a=entity_a,
        period_a=period_a,
        month_a=m_a,
        entity_b=entity_b,
        period_b=period_b,
        month_b=m_b,
        drift_months=drift_months,
        is_aligned=is_aligned,
        warning=warning,
        recommendation=recommendation,
    )
