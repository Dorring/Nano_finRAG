"""Deterministic, question-only feature extraction for Retrieval V3."""

from __future__ import annotations

import re
import unicodedata

from src.retrieval_v3.models import MetricPhrase, PeriodExpression


_PERIOD = re.compile(r"\b(?:fy\s*|fiscal\s+|year ended\s+)?((?:19|20)\d{2})\b", re.I)
_STATEMENTS = (("consolidated statements of income", "income_statement"), ("income statement", "income_statement"), ("balance sheet", "balance_sheet"), ("cash flow statement", "cash_flow_statement"), ("operating segments", "operating_segments"), ("products and services", "products_and_services"))
_QUESTION_PREFIX = re.compile(r"^(?:what|which|how much|how many|please|could you|can you)\s+(?:was|were|is|are|did|does|do)?\s*", re.I)
_TRAILING = re.compile(r"\s*(?:reported|report|according to|for|in|during|from)\s*$", re.I)
_OPERATIONS = re.compile(r"\b(?:growth rate|percentage growth|percent change|year[- ]over[- ]year|yoy|difference between|how much higher|how much lower|percentage of|share of total|sum of|average of|mean of|combined total)\b", re.I)


def normalize_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question or "").replace("–", "-").split())


def extract_periods(question: str) -> tuple[tuple[PeriodExpression, ...], tuple[str, ...]]:
    normalized = normalize_question(question)
    matches = list(_PERIOD.finditer(normalized))
    if "rather than" in normalized.lower() and matches:
        matches = matches[:1]
    periods = tuple(PeriodExpression(raw_text=match.group(0), normalized_period=f"FY{match.group(1)}") for match in matches)
    unresolved = ()
    if any(marker in normalized.lower() for marker in ("current year", "prior year", "previous year")) and not periods:
        unresolved = ("relative_period_without_filing_context",)
    return periods, unresolved


def extract_statement_hint(question: str) -> str | None:
    lowered = normalize_question(question).lower()
    return next((hint for phrase, hint in _STATEMENTS if phrase in lowered), None)


def extract_metric_phrases(question: str, periods: tuple[PeriodExpression, ...]) -> tuple[MetricPhrase, ...]:
    text = normalize_question(question)
    for period in periods:
        text = re.sub(re.escape(period.raw_text), " ", text, flags=re.I)
    text = re.sub(r"^in the .*?table,\s*", "", text, flags=re.I)
    text = _QUESTION_PREFIX.sub("", text)
    text = re.sub(r"\b(?:apple|microsoft|nvidia|jpmorgan|jpmorgan chase|tesla|coca-cola|visa|pfizer)\'?s?\b", " ", text, flags=re.I)
    text = re.sub(r"\b(?:what|which|how much|how many|was|were|is|are|did|does|the|a|an|company)\b", " ", text, flags=re.I)
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
    seen, values = set(), []
    for part in parts:
        clean = _TRAILING.sub("", " ".join(re.sub(r"[^A-Za-z0-9&/-]+", " ", part).split()))
        if len(clean) >= 3 and re.search(r"[A-Za-z]", clean) and clean.lower() not in seen:
            seen.add(clean.lower())
            values.append(MetricPhrase(raw_text=clean, normalized_text=clean.lower()))
    return tuple(values[:3])
