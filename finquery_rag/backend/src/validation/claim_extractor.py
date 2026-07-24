"""Deterministic claim extraction from generated answers (Phase 4 Commit 5).

The ``ClaimExtractor`` scans the LLM-generated answer text and extracts
claims that can be deterministically validated:

- ``amount``       — currency amounts with optional scale suffix.
- ``percent``      — percentage values.
- ``ratio``        — decimal ratios (0.4) and colon ratios (3:2).
- ``period``       — fiscal years, quarters, calendar years.
- ``citation_ref`` — bracket-style source citations ([1], [doc.pdf, p.12]).

Free-form natural-language propositions are NOT extracted — their
unverifiability is a documented Phase 4 limitation.

The extractor is purely regex-based and deterministic. No LLM, no
retrieval, no side effects.

Layer dependency: ``domain <- validation``. Imports only from ``src.domain``
and stdlib.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from src.domain.validation import ExtractedClaim


# ---------------------------------------------------------------------------
# Known metric keywords (lowercase, used for proximity matching)
# ---------------------------------------------------------------------------

_METRIC_KEYWORDS: tuple[str, ...] = (
    "revenue", "total revenue", "net revenue", "sales", "net sales",
    "cost of goods sold", "cogs", "cost of sales",
    "gross profit", "gross margin",
    "net income", "net profit", "net loss",
    "operating income", "operating profit", "operating loss",
    "total assets", "total liabilities", "total equity",
    "shareholders equity", "stockholders equity",
    "ebitda", "ebit",
    "debt ratio", "debt to equity", "debt-to-equity",
    "profit margin", "net margin", "operating margin",
    "earnings per share", "eps",
    "cash flow", "operating cash flow",
    "dividend", "dividends",
    "liabilities", "assets", "equity",
    "income", "profit", "loss", "expense", "expenses",
    "growth rate", "growth", "increase", "decrease",
    "market share", "percentage share",
)

# Map keyword to canonical metric key (for CalculationResult comparison).
_METRIC_CANONICAL: dict[str, str] = {
    "revenue": "revenue",
    "total revenue": "revenue",
    "net revenue": "revenue",
    "sales": "revenue",
    "net sales": "revenue",
    "cost of goods sold": "cost_of_goods_sold",
    "cogs": "cost_of_goods_sold",
    "cost of sales": "cost_of_goods_sold",
    "gross profit": "gross_profit",
    "gross margin": "gross_margin",
    "net income": "net_income",
    "net profit": "net_income",
    "net loss": "net_loss",
    "operating income": "operating_income",
    "operating profit": "operating_income",
    "total assets": "total_assets",
    "total liabilities": "total_liabilities",
    "total equity": "total_equity",
    "debt ratio": "debt_ratio",
    "profit margin": "profit_margin",
    "net margin": "net_margin",
    "operating margin": "operating_margin",
}


# ---------------------------------------------------------------------------
# Currency and scale maps
# ---------------------------------------------------------------------------

_CURRENCY_MAP: dict[str, str] = {
    "$": "USD",
    "¥": "CNY",
    "€": "EUR",
    "£": "GBP",
    "usd": "USD",
    "cny": "CNY",
    "rmb": "CNY",
    "eur": "EUR",
    "gbp": "GBP",
}

_SCALE_MAP: dict[str, Decimal] = {
    "thousand": Decimal("1000"),
    "million": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "trillion": Decimal("1000000000000"),
    "k": Decimal("1000"),
    "m": Decimal("1000000"),
    "b": Decimal("1000000000"),
    "万": Decimal("10000"),
    "亿": Decimal("100000000"),
    # Not a financial scale.  Treat it as a raw amount so an invented unit
    # such as "8888 zillion" is still extracted and rejected by grounding
    # validation instead of silently bypassing the safety gate.
    "zillion": Decimal("1"),
}


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Currency amount: $1,000,000 or $1.5M or ¥100万
_RE_CURRENCY_AMOUNT = re.compile(
    r"([$¥€£])\s*"                       # currency symbol
    r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*"
    r"(trillion|billion|million|thousand|k|m|b|万|亿)?",
    re.IGNORECASE,
)

# Plain amount with scale: 1.5 million or 1.5M (no currency symbol)
_RE_SCALED_AMOUNT = re.compile(
    r"(?<![$¥€£\w])"
    r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*"
    r"(trillion|billion|million|thousand|zillion|k|m|b|万|亿)\b",
    re.IGNORECASE,
)

# Percentage: 40% or 40.00%
_RE_PERCENTAGE = re.compile(
    r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*%",
)

# Colon ratio: 3:2
_RE_COLON_RATIO = re.compile(
    r"(?<!\d)(\d+)\s*:\s*(\d+)(?!\d)",
)

# Fiscal year: FY2025 or FY 2025
_RE_FISCAL_YEAR = re.compile(
    r"\bFY\s?(\d{4})\b",
    re.IGNORECASE,
)

# Quarter: Q1 2024 or Q3 2024
_RE_QUARTER = re.compile(
    r"\bQ([1-4])\s+(\d{4})\b",
    re.IGNORECASE,
)

# Bare year: 2024 (but not years like 1234 or 9999)
_RE_YEAR = re.compile(
    r"\b(20[0-2]\d|19[89]\d)\b",
)

# Citation reference: [1] or [doc.pdf, p.12] or [source 1]
_RE_CITATION = re.compile(
    r"\[([^\]]{1,80})\]",
)


class ClaimExtractor:
    """Extract deterministic claims from a generated answer.

    The extractor is stateless and deterministic. The same input always
    produces the same output.
    """

    def extract(self, answer: str) -> tuple[ExtractedClaim, ...]:
        """Extract all deterministic claims from the answer text.

        Returns a tuple of ``ExtractedClaim`` objects sorted by their
        position in the answer text.
        """
        if not answer or not answer.strip():
            return ()

        claims: list[tuple[int, ExtractedClaim]] = []
        counter = 0

        for match in _RE_CURRENCY_AMOUNT.finditer(answer):
            counter += 1
            claim = self._build_currency_claim(counter, match, answer)
            if claim is not None:
                claims.append((match.start(), claim))

        for match in _RE_SCALED_AMOUNT.finditer(answer):
            counter += 1
            claim = self._build_scaled_claim(counter, match, answer)
            if claim is not None:
                # Avoid duplicating claims already caught by currency regex.
                if not self._overlaps_existing(match, claims):
                    claims.append((match.start(), claim))

        for match in _RE_PERCENTAGE.finditer(answer):
            counter += 1
            claim = self._build_percent_claim(counter, match, answer)
            if claim is not None:
                if not self._overlaps_existing(match, claims):
                    claims.append((match.start(), claim))

        for match in _RE_COLON_RATIO.finditer(answer):
            counter += 1
            claim = self._build_ratio_claim(counter, match, answer)
            if claim is not None:
                if not self._overlaps_existing(match, claims):
                    claims.append((match.start(), claim))

        for match in _RE_FISCAL_YEAR.finditer(answer):
            counter += 1
            claim = self._build_fiscal_year_claim(counter, match)
            if claim is not None:
                claims.append((match.start(), claim))

        for match in _RE_QUARTER.finditer(answer):
            counter += 1
            claim = self._build_quarter_claim(counter, match)
            if claim is not None:
                claims.append((match.start(), claim))

        # Bare years (only if not already captured by FY/quarter patterns).
        for match in _RE_YEAR.finditer(answer):
            if not self._overlaps_existing(match, claims):
                counter += 1
                claim = self._build_year_claim(counter, match)
                claims.append((match.start(), claim))

        # Citation references.
        for match in _RE_CITATION.finditer(answer):
            counter += 1
            claim = self._build_citation_claim(counter, match)
            if claim is not None:
                claims.append((match.start(), claim))

        # Sort by position and return.
        claims.sort(key=lambda x: x[0])
        return tuple(c for _, c in claims)

    # -----------------------------------------------------------------
    # Claim builders
    # -----------------------------------------------------------------

    @staticmethod
    def _build_currency_claim(
        claim_num: int,
        match: re.Match,
        answer: str,
    ) -> ExtractedClaim | None:
        """Build a claim for a currency amount match."""
        symbol = match.group(1)
        number_str = match.group(2).replace(",", "")
        scale_str = match.group(3)

        try:
            value = Decimal(number_str)
        except InvalidOperation:
            return None

        currency = _CURRENCY_MAP.get(symbol, None)
        scale = None
        if scale_str:
            scale_lower = scale_str.lower()
            if scale_lower in _SCALE_MAP:
                value = value * _SCALE_MAP[scale_lower]
                scale = scale_lower

        metric = ClaimExtractor._find_nearby_metric(answer, match.start(), match.end())
        period = ClaimExtractor._find_nearby_period(answer, match.start(), match.end())

        return ExtractedClaim(
            claim_id=f"claim_{claim_num:03d}",
            claim_type="amount",
            raw_text=match.group(0),
            metric=metric,
            value=value,
            unit="base",
            scale=scale,
            currency=currency,
            period=period,
        )

    @staticmethod
    def _build_scaled_claim(
        claim_num: int,
        match: re.Match,
        answer: str,
    ) -> ExtractedClaim | None:
        """Build a claim for a plain amount with scale suffix."""
        number_str = match.group(1).replace(",", "")
        scale_str = match.group(2).lower()

        try:
            value = Decimal(number_str)
        except InvalidOperation:
            return None

        if scale_str not in _SCALE_MAP:
            return None

        value = value * _SCALE_MAP[scale_str]
        metric = ClaimExtractor._find_nearby_metric(answer, match.start(), match.end())
        period = ClaimExtractor._find_nearby_period(answer, match.start(), match.end())

        return ExtractedClaim(
            claim_id=f"claim_{claim_num:03d}",
            claim_type="amount",
            raw_text=match.group(0),
            metric=metric,
            value=value,
            unit="base",
            scale=scale_str,
            period=period,
        )

    @staticmethod
    def _build_percent_claim(
        claim_num: int,
        match: re.Match,
        answer: str,
    ) -> ExtractedClaim | None:
        """Build a claim for a percentage match."""
        number_str = match.group(1).replace(",", "")
        try:
            value = Decimal(number_str)
        except InvalidOperation:
            return None

        metric = ClaimExtractor._find_nearby_metric(answer, match.start(), match.end())
        period = ClaimExtractor._find_nearby_period(answer, match.start(), match.end())

        return ExtractedClaim(
            claim_id=f"claim_{claim_num:03d}",
            claim_type="percent",
            raw_text=match.group(0),
            metric=metric,
            value=value,
            unit="percent",
            period=period,
        )

    @staticmethod
    def _build_ratio_claim(
        claim_num: int,
        match: re.Match,
        answer: str,
    ) -> ExtractedClaim | None:
        """Build a claim for a colon ratio (3:2)."""
        left = match.group(1)
        right = match.group(2)
        try:
            value = Decimal(left) / Decimal(right)
        except (InvalidOperation, ZeroDivisionError):
            return None

        metric = ClaimExtractor._find_nearby_metric(answer, match.start(), match.end())
        period = ClaimExtractor._find_nearby_period(answer, match.start(), match.end())

        return ExtractedClaim(
            claim_id=f"claim_{claim_num:03d}",
            claim_type="ratio",
            raw_text=match.group(0),
            metric=metric,
            value=value,
            unit="ratio",
            period=period,
        )

    @staticmethod
    def _build_fiscal_year_claim(claim_num: int, match: re.Match) -> ExtractedClaim:
        """Build a claim for a fiscal year (FY2025)."""
        year = match.group(1)
        return ExtractedClaim(
            claim_id=f"claim_{claim_num:03d}",
            claim_type="period",
            raw_text=match.group(0),
            period=f"FY{year}",
        )

    @staticmethod
    def _build_quarter_claim(claim_num: int, match: re.Match) -> ExtractedClaim:
        """Build a claim for a quarter (Q3 2024)."""
        quarter = match.group(1)
        year = match.group(2)
        return ExtractedClaim(
            claim_id=f"claim_{claim_num:03d}",
            claim_type="period",
            raw_text=match.group(0),
            period=f"Q{quarter} {year}",
        )

    @staticmethod
    def _build_year_claim(claim_num: int, match: re.Match) -> ExtractedClaim:
        """Build a claim for a bare year (2024)."""
        year = match.group(1)
        return ExtractedClaim(
            claim_id=f"claim_{claim_num:03d}",
            claim_type="period",
            raw_text=match.group(0),
            period=year,
        )

    @staticmethod
    def _build_citation_claim(claim_num: int, match: re.Match) -> ExtractedClaim:
        """Build a claim for a citation reference ([1])."""
        return ExtractedClaim(
            claim_id=f"claim_{claim_num:03d}",
            claim_type="citation_ref",
            raw_text=match.group(0),
            citation_refs=(match.group(1).strip(),),
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _find_nearby_metric(
        answer: str,
        match_start: int,
        match_end: int,
        window: int = 60,
    ) -> str | None:
        """Find a metric keyword near the matched numeric value.

        Searches a window of ``window`` characters before and after the
        match for known metric keywords. Returns the canonical metric key
        or ``None``.
        """
        start = max(0, match_start - window)
        end = min(len(answer), match_end + window)
        context = answer[start:end].lower()

        # Sort keywords by length (longest first) to match multi-word terms.
        for keyword in sorted(_METRIC_KEYWORDS, key=len, reverse=True):
            if keyword in context:
                return _METRIC_CANONICAL.get(keyword, keyword)
        return None

    @staticmethod
    def _find_nearby_period(
        answer: str,
        match_start: int,
        match_end: int,
        window: int = 80,
    ) -> str | None:
        """Find a period (year/FY/quarter) near the matched numeric value.

        Searches a window of ``window`` characters before and after the
        match for fiscal years (FY2025), quarters (Q3 2024), or bare
        years (2024). Returns the period string or ``None``.

        If multiple distinct years are found in the window, returns
        ``None`` (period is ambiguous — the validator will flag
        ``PERIOD_AMBIGUOUS``).
        """
        start = max(0, match_start - window)
        end = min(len(answer), match_end + window)
        context = answer[start:end]

        # Try fiscal year first (most specific).
        fy_matches = _RE_FISCAL_YEAR.findall(context)
        if fy_matches:
            return f"FY{fy_matches[0]}"

        # Try quarter.
        q_matches = _RE_QUARTER.findall(context)
        if q_matches:
            return f"Q{q_matches[0][0]} {q_matches[0][1]}"

        # Try bare years.
        year_matches = _RE_YEAR.findall(context)
        if len(year_matches) == 1:
            return year_matches[0]
        if len(year_matches) > 1:
            # Multiple years in window — ambiguous.
            return None

        return None

    @staticmethod
    def _overlaps_existing(
        match: re.Match,
        existing: list[tuple[int, ExtractedClaim]],
    ) -> bool:
        """Check if a match overlaps with an already-extracted claim."""
        match_start = match.start()
        match_end = match.end()
        for pos, claim in existing:
            # Approximate overlap check: if the existing claim starts
            # within the match range (or vice versa), consider it overlapping.
            claim_len = len(claim.raw_text)
            claim_start = pos
            claim_end = pos + claim_len
            if match_start <= claim_start < match_end:
                return True
            if claim_start <= match_start < claim_end:
                return True
        return False
