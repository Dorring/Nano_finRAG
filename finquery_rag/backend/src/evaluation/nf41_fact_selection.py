"""Offline structured-fact tracing for the NF41 frozen-context experiment.

The module deliberately works only with already-rendered NF39 R2 candidates.
It neither retrieves evidence nor calls a model.  Production answer selection
is unchanged; the constraint-aware selector is evaluation-only until enabled
explicitly by a later reviewed configuration change.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Iterable

from src.evaluation.evaluation import EvaluationCase
from src.evaluation.nf40_frozen_context import FrozenCaseContext


class AnswerExecutionMode(StrEnum):
    """The production route observed for an answer, not a model judgement."""

    DETERMINISTIC_FACT = "deterministic_fact"
    DETERMINISTIC_CALCULATION = "deterministic_calculation"
    SAFE_RESPONSE = "safe_response"
    LLM_GENERATION = "llm_generation"


class DeterministicFactFailure(StrEnum):
    GOLD_SOURCE_NOT_FACTIZED = "gold_source_not_factized"
    CORRECT_FACT_NOT_EXTRACTED = "correct_fact_not_extracted"
    WRONG_CANDIDATE_SELECTED = "wrong_candidate_selected"
    WRONG_METRIC_SELECTED = "wrong_metric_selected"
    WRONG_PERIOD_SELECTED = "wrong_period_selected"
    WRONG_UNIT_OR_SCALE_SELECTED = "wrong_unit_or_scale_selected"
    WRONG_TABLE_COLUMN_SELECTED = "wrong_table_column_selected"
    FACT_CORRECT_RENDERING_WRONG = "fact_correct_rendering_wrong"
    CALCULATION_WRONG_OPERATION = "calculation_wrong_operation"
    CALCULATION_WRONG_OPERAND = "calculation_wrong_operand"
    UNCLASSIFIED = "unclassified"


SELECTOR_FAILURE_TYPES = frozenset(
    {
        DeterministicFactFailure.WRONG_CANDIDATE_SELECTED,
        DeterministicFactFailure.WRONG_METRIC_SELECTED,
        DeterministicFactFailure.WRONG_PERIOD_SELECTED,
        DeterministicFactFailure.WRONG_UNIT_OR_SCALE_SELECTED,
        DeterministicFactFailure.WRONG_TABLE_COLUMN_SELECTED,
    }
)


@dataclass(frozen=True)
class StructuredFactCandidate:
    fact_id: str
    candidate_key: str
    candidate_rank: int
    document_id: str
    page: int | None
    text: str
    value: Decimal | None
    source_expression: str | None
    unit: str | None
    period: str | None
    metric_text: str
    extraction_source: str
    extraction_confidence: float


@dataclass(frozen=True)
class FactQueryConstraints:
    entities: tuple[str, ...]
    metric_terms: tuple[str, ...]
    periods: tuple[str, ...]
    expected_unit_family: str | None
    expected_answer_type: str
    requires_entity_match: bool
    requires_period_match: bool
    requires_numeric_answer: bool


_NUMBER = re.compile(
    r"(?<![\w.])(?:[$€£]|(?:usd|rs\.?)\s*)?\(?\d[\d,]*(?:\.\d+)?\)?"
    r"\s*(?:%|per\s+cent|million|billion|thousand(?:s)?(?:\s+of\s+swiss\s+francs)?|swiss\s+francs|francs)?",
    re.IGNORECASE,
)
_PERIOD = re.compile(r"\b(?:19|20)\d{2}\b")
_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9&/-]{1,}")
_STOPWORDS = frozenset(
    {
        "what", "was", "were", "the", "and", "for", "did", "does",
        "have", "how", "much", "many", "as", "of", "in", "on", "by",
        "to", "from", "with", "which", "a", "an", "at", "is", "its",
        "shown", "amount", "reported", "according", "illustration", "annual",
        "financial", "statement", "statements", "year", "net", "provided",
    }
)


def _tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN.findall(value or "")
        if token.lower() not in _STOPWORDS
    }


def _unit_family(query: str) -> str | None:
    normalized = (query or "").lower()
    if any(marker in normalized for marker in ("percentage", "percent", "per cent", "share", "margin", "rate")):
        return "percentage"
    if any(marker in normalized for marker in ("revenue", "cash", "income", "assets", "expense", "amount", "facility")):
        return "currency_or_amount"
    return None


def build_constraints(query: str) -> FactQueryConstraints:
    """Build deterministic query constraints without document-specific aliases."""
    normalized = (query or "").lower()
    periods = tuple(dict.fromkeys(_PERIOD.findall(normalized)))
    unit = _unit_family(normalized)
    numeric = unit is not None or bool(re.search(r"\b(?:how much|how many)\b", normalized))
    tokens = _tokens(query)
    entities = tuple(
        item.lower()
        for item in re.findall(r"\b(?:[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b", query or "")
        if item.lower() not in {"what", "which", "how", "in", "the"}
    )
    answer_type = "numeric" if numeric else "fact"
    if unit == "percentage":
        answer_type = "percentage"
    if "compare" in normalized:
        answer_type = "comparison"
    return FactQueryConstraints(
        entities=tuple(dict.fromkeys(entities)),
        metric_terms=tuple(sorted(tokens)),
        periods=periods,
        expected_unit_family=unit,
        expected_answer_type=answer_type,
        requires_entity_match=bool(entities),
        requires_period_match=bool(periods),
        requires_numeric_answer=numeric,
    )


def _parse_number(expression: str) -> Decimal | None:
    raw = re.sub(r"[^0-9.]", "", expression or "")
    if not raw or raw.count(".") > 1:
        return None
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _extract_unit(expression: str) -> str | None:
    lowered = (expression or "").lower()
    if "%" in lowered or "per cent" in lowered:
        return "percentage"
    if "billion" in lowered:
        return "billion"
    if "million" in lowered:
        return "million"
    if "thousand" in lowered:
        return "thousand"
    if any(marker in lowered for marker in ("$", "usd", "franc")):
        return "currency"
    return None


def extract_structured_facts(context: FrozenCaseContext) -> list[StructuredFactCandidate]:
    """Extract sentence/line-level facts from the immutable rendered payload."""
    facts: list[StructuredFactCandidate] = []
    for candidate in context.candidates:
        text = candidate.rendered_content.partition("\n")[2]
        spans = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", text) if item.strip()]
        for span_index, span in enumerate(spans):
            period_match = _PERIOD.search(span)
            numbers = list(_NUMBER.finditer(span))
            if numbers:
                for number_index, match in enumerate(numbers):
                    expression = re.sub(r"\s+", " ", match.group(0)).strip()
                    value = _parse_number(expression)
                    if value is None:
                        continue
                    facts.append(
                        StructuredFactCandidate(
                            fact_id=f"{candidate.candidate_key}:{span_index}:{number_index}",
                            candidate_key=candidate.candidate_key,
                            candidate_rank=candidate.rank,
                            document_id=candidate.document_id,
                            page=candidate.page,
                            text=span[:700],
                            value=value,
                            source_expression=expression,
                            unit=_extract_unit(expression),
                            period=period_match.group(0) if period_match else None,
                            metric_text=span,
                            extraction_source="numeric_span",
                            extraction_confidence=1.0,
                        )
                    )
            elif len(span) >= 12:
                facts.append(
                    StructuredFactCandidate(
                        fact_id=f"{candidate.candidate_key}:{span_index}:fact",
                        candidate_key=candidate.candidate_key,
                        candidate_rank=candidate.rank,
                        document_id=candidate.document_id,
                        page=candidate.page,
                        text=span[:700],
                        value=None,
                        source_expression=None,
                        unit=None,
                        period=period_match.group(0) if period_match else None,
                        metric_text=span,
                        extraction_source="text_span",
                        extraction_confidence=0.7,
                    )
                )
    return facts


def _expected_values(case: EvaluationCase) -> set[Decimal]:
    values = set()
    for item in case.expected_numbers:
        parsed = _parse_number(item)
        if parsed is not None:
            values.add(parsed)
    return values


def fact_matches_expected_source(fact: StructuredFactCandidate, case: EvaluationCase) -> bool:
    return any(
        source.filename == fact.document_id and (source.page is None or source.page == fact.page)
        for source in case.expected_sources
    )


def fact_matches_expected_answer(fact: StructuredFactCandidate, case: EvaluationCase) -> bool:
    expected_values = _expected_values(case)
    if expected_values and fact.value is not None:
        return fact.value in expected_values
    normalized = fact.text.lower()
    return any(expected.lower() in normalized for expected in case.expected_answer_contains)


def fact_matches_rendered_answer(fact: StructuredFactCandidate, answer: str) -> bool:
    normalized = (answer or "").lower()
    if fact.source_expression and fact.source_expression.lower() in normalized:
        return True
    if fact.value is not None:
        return str(fact.value).rstrip("0").rstrip(".") in normalized.replace(",", "")
    return bool(fact.text and fact.text.lower()[:30] in normalized)


def _score_fact(fact: StructuredFactCandidate, constraints: FactQueryConstraints) -> tuple[int, int, int, int, int, int]:
    metric_matches = len(_tokens(fact.metric_text) & set(constraints.metric_terms))
    period_matches = int(bool(fact.period and fact.period in constraints.periods))
    entity_matches = sum(entity in fact.metric_text.lower() for entity in constraints.entities)
    unit_matches = int(
        constraints.expected_unit_family is None
        or fact.unit is None
        or (constraints.expected_unit_family == "percentage" and fact.unit == "percentage")
        or (constraints.expected_unit_family == "currency_or_amount" and fact.unit != "percentage")
    )
    explicit = period_matches + entity_matches + unit_matches
    return (explicit, metric_matches, period_matches, entity_matches, unit_matches, -fact.candidate_rank)


def has_explicit_conflict(fact: StructuredFactCandidate, constraints: FactQueryConstraints) -> bool:
    if constraints.requires_period_match and fact.period and fact.period not in constraints.periods:
        return True
    if constraints.requires_entity_match and fact.text and not any(entity in fact.text.lower() for entity in constraints.entities):
        return True
    if constraints.expected_unit_family == "percentage" and fact.unit and fact.unit != "percentage":
        return True
    return False


def select_constraint_aware_fact(
    facts: Iterable[StructuredFactCandidate], constraints: FactQueryConstraints
) -> StructuredFactCandidate | None:
    """Select with explicit conflict filtering and stable lexical tie-breaking."""
    eligible = [fact for fact in facts if not has_explicit_conflict(fact, constraints)]
    if constraints.requires_numeric_answer:
        numeric = [fact for fact in eligible if fact.value is not None]
        eligible = numeric or eligible
    if not eligible:
        return None
    return max(eligible, key=lambda fact: (_score_fact(fact, constraints), fact.fact_id))


def classify_fact_failure(
    *, case: EvaluationCase, facts: list[StructuredFactCandidate], selected: StructuredFactCandidate | None,
    rendered_answer: str,
) -> DeterministicFactFailure:
    gold_facts = [fact for fact in facts if fact_matches_expected_source(fact, case)]
    if not gold_facts:
        return DeterministicFactFailure.GOLD_SOURCE_NOT_FACTIZED
    correct = [fact for fact in gold_facts if fact_matches_expected_answer(fact, case)]
    if not correct:
        return DeterministicFactFailure.CORRECT_FACT_NOT_EXTRACTED
    if selected is None:
        return DeterministicFactFailure.WRONG_CANDIDATE_SELECTED
    if selected.fact_id not in {fact.fact_id for fact in correct}:
        constraints = build_constraints(case.question)
        if constraints.periods and selected.period and selected.period not in constraints.periods:
            return DeterministicFactFailure.WRONG_PERIOD_SELECTED
        if constraints.expected_unit_family == "percentage" and selected.unit != "percentage":
            return DeterministicFactFailure.WRONG_UNIT_OR_SCALE_SELECTED
        if constraints.requires_numeric_answer and selected.candidate_key == correct[0].candidate_key:
            return DeterministicFactFailure.WRONG_TABLE_COLUMN_SELECTED
        if _tokens(selected.metric_text) & set(constraints.metric_terms):
            return DeterministicFactFailure.WRONG_CANDIDATE_SELECTED
        return DeterministicFactFailure.WRONG_METRIC_SELECTED
    if not fact_matches_rendered_answer(selected, rendered_answer):
        return DeterministicFactFailure.FACT_CORRECT_RENDERING_WRONG
    return DeterministicFactFailure.UNCLASSIFIED


def summarize_failures(rows: Iterable[dict]) -> dict[str, int]:
    counts = Counter(row["failure"] for row in rows)
    return {kind.value: counts[kind.value] for kind in DeterministicFactFailure}
