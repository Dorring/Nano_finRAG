"""Deterministic, audit-only metric matching contract for NF-V2-01.

This module is intentionally independent of question IDs, benchmark values,
LLM calls, embeddings, and ranking scores.  It is an evaluation contract;
the frozen SupervisorPlan and prediction artifacts are not modified.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class MetricMatchType(str, Enum):
    EXACT = "EXACT"
    SURFACE_NORMALIZED = "SURFACE_NORMALIZED"
    CANONICAL_EQUIVALENT = "CANONICAL_EQUIVALENT"
    NON_CONFLICTING_QUALIFIER_EQUIVALENT = "NON_CONFLICTING_QUALIFIER_EQUIVALENT"
    NOT_EQUIVALENT = "NOT_EQUIVALENT"


@dataclass(frozen=True)
class MetricMatchResult:
    matched: bool
    match_type: str
    predicted_normalized: str
    reference_normalized: str
    rule_id: str | None = None


SURFACE_RULE_IDS = (
    "surface_nfkc_case_whitespace",
    "surface_punctuation_normalization",
    "surface_simple_singular_plural",
    "surface_grammatical_article_elision",
)

# These are general semantic forms, not aliases for any benchmark example.
CANONICAL_RULE_IDS = (
    "canonical_disclosure_predicate_elision",
    "canonical_terminal_function_word_elision",
)

QUALIFIER_RULE_IDS = (
    "qualifier_margin_percentage",
    "qualifier_segment_revenue",
    "qualifier_possessive_scope",
    "qualifier_period_token",
)

_PUNCTUATION_RE = re.compile(r"[^\w\s']+", flags=re.UNICODE)
_PERIOD_TOKEN_RE = re.compile(r"\b(?:fy)?20\d{2}\b", flags=re.IGNORECASE)
_DISCLOSURE_PREFIX_RE = re.compile(r"^(?:s\s+)?disclose\s+", flags=re.IGNORECASE)
_GENERIC_METRIC_HEADS = frozenset({
    "assets", "expense", "expenses", "income", "liabilities", "margin",
    "revenue", "sales", "total revenue", "net sales",
})
_GRAMMATICAL_ARTICLES = frozenset({"a", "an", "the"})


def _collapse(value: str) -> str:
    return " ".join(value.split())


def _singularize(token: str) -> str:
    if token.endswith("'s"):
        return token
    if len(token) <= 3 or token.endswith("ss"):
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("ses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("us"):
        return token[:-1]
    return token


def surface_normalize(value: Any) -> str:
    """Apply only query-independent lexical normalization."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("’", "'")
    text = _PUNCTUATION_RE.sub(" ", text)
    tokens = [_singularize(token) for token in _collapse(text).split() if token not in _GRAMMATICAL_ARTICLES]
    return " ".join(tokens)


def _strict_surface(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("’", "'")
    return _collapse(text)


def _strip_disclosure(value: str) -> str:
    return _collapse(_DISCLOSURE_PREFIX_RE.sub("", value))


def _strip_terminal_of(value: str) -> str:
    return re.sub(r"\s+(?:of|for)$", "", value).strip()


def _strip_period_tokens(value: str) -> str:
    return _collapse(_PERIOD_TOKEN_RE.sub(" ", value))


def _strip_possessive_scope(value: str) -> str:
    return _collapse(re.sub(r"\b[\w-]+'s\b", " ", value, flags=re.IGNORECASE))


def _is_segment_revenue_pair(predicted: str, reference: str, raw_reference: str) -> bool:
    if not predicted.endswith(" revenue") or predicted == reference:
        return False
    base = predicted[: -len(" revenue")].strip()
    if base != reference or len(reference.split()) < 2:
        return False
    if reference in _GENERIC_METRIC_HEADS or reference.split()[-1] in _GENERIC_METRIC_HEADS:
        return False
    # Segment labels in the frozen review are proper-name-like; this generic
    # shape check avoids treating "operating income" as "operating income
    # revenue" without adding a case-specific alias.
    raw_tokens = raw_reference.split()
    return bool(raw_tokens) and sum(token[:1].isupper() for token in raw_tokens) >= 2


def _qualifier_match(predicted: str, reference: str, raw_reference: str) -> str | None:
    if predicted.endswith(" percentage") and predicted[: -len(" percentage")].strip() == reference and reference.endswith("margin"):
        return "qualifier_margin_percentage"
    if _is_segment_revenue_pair(predicted, reference, raw_reference):
        return "qualifier_segment_revenue"
    if _strip_possessive_scope(predicted) == reference or _strip_possessive_scope(reference) == predicted:
        if predicted != reference:
            return "qualifier_possessive_scope"
    if _strip_terminal_of(_strip_period_tokens(predicted)) == _strip_terminal_of(_strip_period_tokens(reference)) and predicted != reference:
        return "qualifier_period_token"
    return None


def match_metric(
    predicted: Any,
    reference: Any,
    *,
    predicted_value_type: str | None = None,
    reference_value_type: str | None = None,
) -> MetricMatchResult:
    """Return a deterministic match result with no fuzzy or semantic search."""

    predicted_str = str(predicted or "")
    reference_str = str(reference or "")
    predicted_normalized = surface_normalize(predicted_str)
    reference_normalized = surface_normalize(reference_str)
    if not predicted_normalized or not reference_normalized:
        return MetricMatchResult(False, MetricMatchType.NOT_EQUIVALENT.value, predicted_normalized, reference_normalized)
    if _strict_surface(predicted_str) == _strict_surface(reference_str):
        return MetricMatchResult(True, MetricMatchType.EXACT.value, predicted_normalized, reference_normalized)
    if predicted_normalized == reference_normalized:
        strict_predicted = _strict_surface(predicted_str)
        strict_reference = _strict_surface(reference_str)
        rule_id = "surface_grammatical_article_elision" if strict_predicted != strict_reference and all(token not in _GRAMMATICAL_ARTICLES for token in strict_predicted.split() if token not in strict_reference.split()) else "surface_nfkc_case_whitespace"
        return MetricMatchResult(True, MetricMatchType.SURFACE_NORMALIZED.value, predicted_normalized, reference_normalized, rule_id)

    canonical_predicted = _strip_disclosure(predicted_normalized)
    canonical_reference = _strip_disclosure(reference_normalized)
    if canonical_predicted == canonical_reference:
        return MetricMatchResult(True, MetricMatchType.CANONICAL_EQUIVALENT.value, predicted_normalized, reference_normalized, "canonical_disclosure_predicate_elision")
    if _strip_terminal_of(canonical_predicted) == _strip_terminal_of(canonical_reference):
        return MetricMatchResult(True, MetricMatchType.CANONICAL_EQUIVALENT.value, predicted_normalized, reference_normalized, "canonical_terminal_function_word_elision")

    qualifier_rule = _qualifier_match(canonical_predicted, canonical_reference, reference_str)
    if qualifier_rule == "qualifier_margin_percentage":
        compatible_types = {predicted_value_type, reference_value_type} - {None, ""}
        if not compatible_types or compatible_types <= {"percentage"}:
            return MetricMatchResult(True, MetricMatchType.NON_CONFLICTING_QUALIFIER_EQUIVALENT.value, predicted_normalized, reference_normalized, qualifier_rule)
    elif qualifier_rule is not None:
        return MetricMatchResult(True, MetricMatchType.NON_CONFLICTING_QUALIFIER_EQUIVALENT.value, predicted_normalized, reference_normalized, qualifier_rule)
    return MetricMatchResult(False, MetricMatchType.NOT_EQUIVALENT.value, predicted_normalized, reference_normalized)


OPERATIONAL_ROLES = frozenset({"current", "prior", "numerator", "denominator", "minuend", "subtrahend"})


@dataclass(frozen=True)
class SlotMatch:
    predicted_index: int
    reference_index: int
    metric: MetricMatchResult
    period_match: bool
    operational_role_match: bool


@dataclass(frozen=True)
class SlotSetMatchResult:
    complete: bool
    matches: tuple[SlotMatch, ...]
    unmatched_predicted: tuple[int, ...]
    unmatched_reference: tuple[int, ...]


def match_slots(predicted_slots: Sequence[Mapping[str, Any]], reference_slots: Sequence[Mapping[str, Any]]) -> SlotSetMatchResult:
    """Match slots as a deterministic set, preserving strict calculator roles."""

    candidates: list[tuple[int, int, SlotMatch]] = []
    for predicted_index, predicted in enumerate(predicted_slots):
        for reference_index, reference in enumerate(reference_slots):
            metric = match_metric(
                predicted.get("metric"),
                reference.get("target", reference.get("metric")),
                predicted_value_type=predicted.get("value_type"),
                reference_value_type=reference.get("value_type"),
            )
            period_match = surface_normalize(predicted.get("period")) == surface_normalize(reference.get("period"))
            predicted_role = str(predicted.get("role") or "")
            reference_role = str(reference.get("role") or "")
            operational_role_match = (
                predicted_role == reference_role
                if predicted_role in OPERATIONAL_ROLES or reference_role in OPERATIONAL_ROLES
                else True
            )
            if metric.matched and period_match and operational_role_match:
                priority = {
                    MetricMatchType.EXACT.value: 0,
                    MetricMatchType.SURFACE_NORMALIZED.value: 1,
                    MetricMatchType.CANONICAL_EQUIVALENT.value: 2,
                    MetricMatchType.NON_CONFLICTING_QUALIFIER_EQUIVALENT.value: 3,
                }.get(metric.match_type, 9)
                candidates.append((priority, predicted_index * max(len(reference_slots), 1) + reference_index, SlotMatch(predicted_index, reference_index, metric, period_match, operational_role_match)))
    used_predicted: set[int] = set()
    used_reference: set[int] = set()
    matches: list[SlotMatch] = []
    for _, _, candidate in sorted(candidates, key=lambda item: (item[0], item[1])):
        if candidate.predicted_index in used_predicted or candidate.reference_index in used_reference:
            continue
        used_predicted.add(candidate.predicted_index)
        used_reference.add(candidate.reference_index)
        matches.append(candidate)
    return SlotSetMatchResult(
        complete=len(matches) == len(predicted_slots) == len(reference_slots),
        matches=tuple(sorted(matches, key=lambda item: item.reference_index)),
        unmatched_predicted=tuple(index for index in range(len(predicted_slots)) if index not in used_predicted),
        unmatched_reference=tuple(index for index in range(len(reference_slots)) if index not in used_reference),
    )
