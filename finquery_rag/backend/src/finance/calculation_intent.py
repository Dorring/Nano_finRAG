"""Question-only deterministic derived-value calculation intent detection."""

from __future__ import annotations

from dataclasses import dataclass
import re

from src.domain.calculation import CalculationOperation


_YEAR_RE = re.compile(r"\b(?:fy\s*)?(?:19|20)\d{2}\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{1,}", re.IGNORECASE)
_GROWTH = (
    "growth rate",
    "percentage growth",
    "percentage increase",
    "percentage decrease",
    "percent change",
    "year-over-year growth",
    "year over year growth",
    "yoy",
    "qoq",
    "increase from",
    "decrease from",
)
_SHARE = (
    "what percentage of",
    "percentage share",
    "share of total",
    "portion of total",
    "as a percentage of",
    "represented what percentage",
    "came from",
)
_DIFFERENCE = (
    "difference between",
    "by how much",
    "how much higher",
    "how much lower",
    "exceeded",
    "gap between",
    "subtract",
    "minus",
)
_SUM = ("combined total", "sum of", "total of", "together", "in aggregate")
_AVERAGE = ("average of", "mean of", "average across")
_EXPLICIT = ("calculate", "compute", "derive", "work out")
_RATIO = (
    ("gross margin", CalculationOperation.GROSS_MARGIN),
    ("net margin", CalculationOperation.NET_MARGIN),
    ("debt ratio", CalculationOperation.DEBT_RATIO),
)
_SCALE = ("convert", "expressed in", "in terms of")
_STOPWORDS = {
    "what",
    "was",
    "were",
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "report",
    "reported",
    "company",
    "fiscal",
    "year",
    "rate",
    "total",
    "how",
    "much",
    "percentage",
    "between",
}


@dataclass(frozen=True)
class CalculationIntent:
    requires_calculation: bool
    operation: CalculationOperation | None
    metric_terms: tuple[str, ...]
    period_terms: tuple[str, ...]
    entity_terms: tuple[str, ...]
    expected_operand_count: int | None
    derived_value_requested: bool
    confidence: float
    matched_signals: tuple[str, ...]
    rejection_reason: str | None


def _metadata(
    question: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    tokens = tuple(
        dict.fromkeys(token.lower() for token in _TOKEN_RE.findall(question))
    )
    metrics = tuple(token for token in tokens if token not in _STOPWORDS)
    entities = tuple(
        token
        for token in tokens
        if token
        in {"apple", "microsoft", "nvidia", "tesla", "visa", "pfizer", "jpmorgan"}
    )
    periods = tuple(
        dict.fromkeys(
            match.group(0).upper().replace(" ", "")
            for match in _YEAR_RE.finditer(question)
        )
    )
    return metrics, periods, entities


def _result(
    question: str,
    operation: CalculationOperation | None,
    operands: int | None,
    signals: tuple[str, ...],
    rejection: str | None = None,
) -> CalculationIntent:
    metrics, periods, entities = _metadata(question)
    accepted = operation is not None and rejection is None
    return CalculationIntent(
        requires_calculation=accepted,
        operation=operation if accepted else None,
        metric_terms=metrics,
        period_terms=periods,
        entity_terms=entities,
        expected_operand_count=operands if accepted else None,
        derived_value_requested=accepted,
        confidence=0.98 if accepted else 0.0,
        matched_signals=signals,
        rejection_reason=rejection,
    )


def _has_any(text: str, values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value for value in values if value in text)


def detect_calculation_intent(question: str) -> CalculationIntent:
    """Accept only explicit requests for a derived value and its operands."""
    normalized = " ".join((question or "").lower().split())
    if not normalized:
        return _result(question or "", None, None, (), "empty_question")
    growth = _has_any(normalized, _GROWTH)
    if growth:
        years = {match.group(0) for match in _YEAR_RE.finditer(normalized)}
        valid = len(years) >= 2 or (" from " in normalized and " to " in normalized)
        return _result(
            question,
            CalculationOperation.GROWTH_RATE if valid else None,
            2,
            growth,
            None if valid else "growth_requires_two_periods",
        )
    share = _has_any(normalized, _SHARE)
    if share:
        valid = (" of " in normalized or "from " in normalized) and (
            "total" in normalized or "percentage" in normalized
        )
        return _result(
            question,
            CalculationOperation.PERCENTAGE_SHARE if valid else None,
            2,
            share,
            None if valid else "share_requires_part_and_total",
        )
    difference = _has_any(normalized, _DIFFERENCE)
    if difference:
        valid = any(
            marker in normalized for marker in (" between ", " or ", " and ", "from ")
        )
        return _result(
            question,
            CalculationOperation.DIFFERENCE if valid else None,
            2,
            difference,
            None if valid else "difference_requires_two_operands",
        )
    summed = _has_any(normalized, _SUM)
    if summed:
        valid = " and " in normalized or "," in normalized
        return _result(
            question,
            CalculationOperation.SUM if valid else None,
            2,
            summed,
            None if valid else "sum_requires_two_operands",
        )
    average = _has_any(normalized, _AVERAGE)
    if average:
        valid = " and " in normalized or " across " in normalized
        return _result(
            question,
            CalculationOperation.AVERAGE if valid else None,
            2,
            average,
            None if valid else "average_requires_two_operands",
        )
    explicit = _has_any(normalized, _EXPLICIT)
    if explicit:
        for metric, operation in _RATIO:
            if metric in normalized:
                has_operands = (" from " in normalized and " and " in normalized) or (
                    "\u6839\u636e" in question and "\u548c" in question
                )
                return _result(
                    question,
                    operation if has_operands else None,
                    2,
                    (*explicit, metric),
                    None if has_operands else "ratio_requires_two_operands",
                )
        if _has_any(normalized, _SCALE):
            return _result(
                question,
                CalculationOperation.SCALE_CONVERSION,
                1,
                (*explicit, *_has_any(normalized, _SCALE)),
            )
    return _result(question, None, None, (), "no_explicit_derived_value_signal")
