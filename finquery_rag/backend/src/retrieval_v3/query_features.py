"""Deterministic, question-only feature extraction for Retrieval V3."""

from __future__ import annotations

import re
import unicodedata

from src.retrieval_v3.models import MetricPhrase, PeriodExpression


_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_PERIOD = re.compile(
    rf"\b(?:"
    rf"(?:fy\s*)?(?P<iso>(?:19|20)\d{{2}}[-/]\d{{1,2}}[-/]\d{{1,2}})"
    rf"|(?:(?:fy\s*|fiscal\s+(?:year\s*)?|year\s+ended\s+)?"
    rf"(?:{_MONTH}\s+\d{{1,2}},?\s+)?)?"
    rf"(?P<year>(?:19|20)\d{{2}})"
    rf")\b",
    re.I,
)
_STATEMENTS = (("consolidated statements of income", "income_statement"), ("income statement", "income_statement"), ("balance sheet", "balance_sheet"), ("cash flow statement", "cash_flow_statement"), ("operating segments", "operating_segments"), ("products and services", "products_and_services"))
_QUESTION_PREFIX = re.compile(r"^(?:what|which|how much|how many|please|could you|can you)\s+(?:was|were|is|are|did|does|do)?\s*", re.I)
_TRAILING = re.compile(r"\s*(?:reported|report|according to|for|in|during|from)\s*$", re.I)
_OPERATIONS = re.compile(r"\b(?:growth rate|percentage growth|percent change|year[- ]over[- ]year|yoy|difference between|how much higher|how much lower|percentage of|share of total|sum of|average of|mean of|combined total)\b", re.I)
_SOURCE_WRAPPERS = re.compile(
    r"\b(?:"
    r"(?:[a-z]{1,6}\s+)?filing|annual\s+report|10[- ]k|report|disclosure|"
    r"statement|line\s+item|row|source\s+occurrence\s+\d+"
    r")\b",
    re.I,
)
_TICKERS = re.compile(r"\b(?:aapl|amzn|googl|jpm|msft|nvda|pfe|tsla|v|ko)\b", re.I)
_TICKER_CONTEXT = re.compile(
    r"\b(?:for|from)\s+(?:aapl|amzn|googl|jpm|msft|nvda|pfe|tsla|v|ko)\b",
    re.I,
)
_QUOTED_TEXT = re.compile(r"(?<![A-Za-z0-9_])(['\"])(.*?)(?<!\\)\1(?![A-Za-z0-9_])")
_QUERY_NOISE = re.compile(
    r"\b(?:what|which|how much|how many|was|were|is|are|did|does|do|calculate|compute|"
    r"find|tell|give|show|the|a|an|company(?:'s)?|given|associated|using|requested|"
    r"disclosures?|retrieve|answer|temporal|scope|represent|their|it|compare|"
    r"comparison|versus|vs)\b",
    re.I,
)


def _mask_quoted_text(text: str) -> str:
    """Mask quoted row/label text while extracting query-level periods."""
    return _QUOTED_TEXT.sub(lambda match: " " * len(match.group(0)), text)


def _protect_quoted_text(text: str) -> tuple[str, dict[str, str]]:
    """Protect quoted metric labels from temporal cleanup and return them."""
    protected: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"quotedmetric{len(protected)}"
        protected[token] = match.group(2)
        return f" {token} "

    return _QUOTED_TEXT.sub(replace, text), protected


def normalize_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question or "").replace("–", "-").split())


def extract_periods(question: str) -> tuple[tuple[PeriodExpression, ...], tuple[str, ...]]:
    normalized = normalize_question(question)
    matches = list(_PERIOD.finditer(_mask_quoted_text(normalized)))
    if "rather than" in normalized.lower() and matches:
        matches = matches[:1]
    periods_list: list[PeriodExpression] = []
    seen: set[str] = set()
    for match in matches:
        year = match.group("iso")[:4] if match.group("iso") else match.group("year")
        if not year:
            continue
        normalized_period = f"FY{year}"
        if normalized_period in seen:
            continue
        seen.add(normalized_period)
        periods_list.append(PeriodExpression(raw_text=match.group(0), normalized_period=normalized_period))
    periods = tuple(periods_list)
    unresolved = ()
    if any(marker in normalized.lower() for marker in ("current year", "prior year", "previous year")) and not periods:
        unresolved = ("relative_period_without_filing_context",)
    return periods, unresolved


def extract_statement_hint(question: str) -> str | None:
    lowered = normalize_question(question).lower()
    return next((hint for phrase, hint in _STATEMENTS if phrase in lowered), None)


def extract_metric_phrases(question: str, periods: tuple[PeriodExpression, ...]) -> tuple[MetricPhrase, ...]:
    text, quoted = _protect_quoted_text(normalize_question(question))
    # Remove every recognized temporal expression, not just the first-seen
    # normalized period. This prevents ISO dates from leaving ``-12-31`` in
    # the retrieval phrase when duplicate FY expressions were deduplicated.
    text = _PERIOD.sub(" ", text)
    text = re.sub(r"^in the .*?table,\s*", "", text, flags=re.I)
    text = _QUESTION_PREFIX.sub("", text)
    text = re.sub(r"\b(?:apple|microsoft|nvidia|jpmorgan|jpmorgan chase|tesla|coca-cola|visa|pfizer)\'?s?\b", " ", text, flags=re.I)
    text = _TICKER_CONTEXT.sub(" ", text)
    text = _TICKERS.sub(" ", text)
    text = re.sub(r"\bin(?=\s+q[1-4]\b)", " ", text, flags=re.I)
    text = _SOURCE_WRAPPERS.sub(" ", text)
    text = _QUERY_NOISE.sub(" ", text)
    text = re.sub(r"\brather than\b.*", "", text, flags=re.I)
    text = re.sub(r"\breported by\b.*", "", text, flags=re.I)
    text = _OPERATIONS.sub(" ", text)
    comparison = re.search(r"\bboth\s+(.+?)\s+and\s+(.+)$", text, re.I)
    higher = re.search(r":\s*(.+?)\s+(?:or|versus|vs\.?)\s+(.+?)(?:,|$)", text, re.I)
    if comparison:
        parts = [comparison.group(1), comparison.group(2)]
    elif higher:
        parts = [higher.group(1), higher.group(2)]
    else:
        parts = [text]
    if quoted:
        # In the evaluation and filing-query grammars, quoted labels are the
        # most precise metric signal (including date-like row labels). Prefer
        # them over surrounding prose such as ``given`` or ``temporal scope``.
        quoted_parts = []
        for value in quoted.values():
            clean = " ".join(re.sub(r"[^A-Za-z0-9&/-]+", " ", value).split())
            if len(clean) >= 3 and re.search(r"[A-Za-z]", clean):
                quoted_parts.append(clean)
        if quoted_parts:
            parts = quoted_parts
        else:
            for token, value in quoted.items():
                parts = [part.replace(token, value) for part in parts]
    seen, values = set(), []
    for part in parts:
        clean = _TRAILING.sub("", " ".join(re.sub(r"[^A-Za-z0-9&/-]+", " ", part).split()))
        if len(clean) >= 3 and re.search(r"[A-Za-z]", clean) and clean.lower() not in seen:
            seen.add(clean.lower())
            values.append(MetricPhrase(raw_text=clean, normalized_text=clean.lower()))
    return tuple(values[:3])
