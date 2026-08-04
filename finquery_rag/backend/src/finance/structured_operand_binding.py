"""Default-off structured binding for deterministic financial calculations.

This module turns evidence into metric-period-value facts before binding
calculation operands. It deliberately blocks when table structure is absent
or ambiguous; it never chooses the first number merely because it appears
near a keyword.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re
from typing import Iterable

from src.domain.calculation import CalculationOperand
from src.domain.evidence import EvidenceItem
from src.domain.financial_fact import BoundOperand, FinancialFact, OperandBindingSpec
from src.finance.calculation_intent import CalculationIntent
from src.finance.operation_router import RoutingDecision
from src.finance.primitive_tools import parse_financial_number

_PERIOD_RE = re.compile(r"\b(?:FY\s*)?(20\d{2})\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\(?-?\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?")
_ROW_SPLIT_RE = re.compile(r"\s*\|\s*")
_SPACE_RE = re.compile(r"\s+")
_SCALE_WORDS = ("thousand", "million", "billion", "trillion")


@dataclass(frozen=True)
class BindingResult:
    operands: tuple[BoundOperand, ...]
    specs: tuple[OperandBindingSpec, ...]
    block_reason: str | None = None

    @property
    def success(self) -> bool:
        return self.block_reason is None and len(self.operands) == len(self.specs)


def normalize_metric_name(value: str) -> str:
    """Normalize generic financial metric text without company-specific aliases."""
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    value = re.sub(r"\btotal\s+net\s+sales\b", "total revenue", value)
    value = re.sub(r"\bnet\s+sales\b", "revenue", value)
    value = re.sub(r"\btotal\s+revenues?\b", "total revenue", value)
    value = re.sub(r"\brevenues?\b", "revenue", value)
    return _SPACE_RE.sub(" ", value).strip()


def normalize_period(value: str | None) -> str | None:
    """Normalize annual period labels while preserving annual-only semantics."""
    if not value:
        return None
    text = value.strip()
    if re.search(r"\b(?:q[1-4]|quarter)\b", text, re.IGNORECASE):
        return None
    match = _PERIOD_RE.search(text)
    return f"FY{match.group(1)}" if match else None


def _metadata_value(metadata: dict, *names: str):
    for name in names:
        value = metadata.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


def _candidate_key(item: EvidenceItem) -> str | None:
    return _metadata_value(item.metadata, "candidate_key")


def _scale(metadata: dict) -> str | None:
    value = _metadata_value(metadata, "scale", "unit_scale", "table_scale")
    if value:
        return str(value).lower()
    title = str(_metadata_value(metadata, "table_title", "title") or "").lower()
    for scale in _SCALE_WORDS:
        if f"in {scale}" in title or f"({scale}" in title:
            return scale
    return None


def _currency(metadata: dict) -> str | None:
    value = _metadata_value(metadata, "currency", "currency_code")
    return str(value).upper() if value else None


def _parse_value(value: object, scale: str | None) -> Decimal | None:
    parsed = parse_financial_number(value, scale=scale)
    return parsed.value if parsed.ok else None


def _facts_from_structured_row(item: EvidenceItem) -> tuple[FinancialFact, ...]:
    metadata = item.metadata
    row = _metadata_value(metadata, "row_label", "metric", "label")
    headers = _metadata_value(metadata, "column_headers", "headers")
    cells = _metadata_value(metadata, "cells", "values")
    if (
        not row
        or not isinstance(headers, (list, tuple))
        or not isinstance(cells, (list, tuple))
    ):
        return ()
    if len(headers) != len(cells):
        return ()
    scale = _scale(metadata)
    facts = []
    for header, cell in zip(headers, cells, strict=True):
        period = normalize_period(str(header))
        if period is None:
            continue
        value = _parse_value(cell, scale)
        if value is None:
            continue
        facts.append(
            FinancialFact(
                metric=str(row),
                value=value,
                period=period,
                entity=str(_metadata_value(metadata, "entity", "segment") or "")
                or None,
                currency=_currency(metadata),
                scale=scale,
                candidate_key=_candidate_key(item),
                evidence_chunk_id=item.chunk_id,
                document_name=item.document_name,
                page=item.page,
                row_label=str(row),
                column_header=str(header),
                table_title=str(_metadata_value(metadata, "table_title", "title") or "")
                or None,
                extraction_method="structured_table_row",
                confidence=1.0,
            )
        )
    return tuple(facts)


def _facts_from_serialized_row(item: EvidenceItem) -> tuple[FinancialFact, ...]:
    """Extract only self-describing serialized rows with explicit period context."""
    metadata = item.metadata
    text = item.content
    if "|" not in text:
        return ()
    context = re.search(
        r"Table column context:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE
    )
    period = normalize_period(context.group(1)) if context else None
    # Without headers or a single explicit period, multiple cells are ambiguous.
    if period is None:
        return ()
    first_line = text.splitlines()[0]
    cells = [cell.strip() for cell in _ROW_SPLIT_RE.split(first_line) if cell.strip()]
    if len(cells) < 2:
        return ()
    metric = cells[0]
    numeric = []
    for cell in cells[1:]:
        value = _parse_value(cell, _scale(metadata))
        if value is not None:
            numeric.append((cell, value))
    if len(numeric) != 1:
        return ()
    raw, value = numeric[0]
    return (
        FinancialFact(
            metric=metric,
            value=value,
            period=period,
            entity=str(_metadata_value(metadata, "entity", "segment") or "") or None,
            currency=_currency(metadata),
            scale=_scale(metadata),
            candidate_key=_candidate_key(item),
            evidence_chunk_id=item.chunk_id,
            document_name=item.document_name,
            page=item.page,
            row_label=metric,
            column_header=period,
            table_title=str(_metadata_value(metadata, "table_title", "title") or "")
            or None,
            extraction_method="serialized_table_row",
            confidence=0.9,
        ),
    )


def _facts_from_plain_text(item: EvidenceItem) -> tuple[FinancialFact, ...]:
    """Extract a fact only when metric, period and exactly one value co-occur."""
    text = item.content
    periods = {normalize_period(match.group(0)) for match in _PERIOD_RE.finditer(text)}
    periods.discard(None)
    if len(periods) != 1:
        return ()
    values = []
    for token in _NUMBER_RE.findall(text):
        value = _parse_value(token, _scale(item.metadata))
        if value is not None:
            values.append((token, value))
    if len(values) != 1:
        return ()
    metric = _metadata_value(item.metadata, "row_label", "metric", "label")
    if not metric:
        return ()
    raw, value = values[0]
    return (
        FinancialFact(
            metric=str(metric),
            value=value,
            period=next(iter(periods)),
            entity=str(_metadata_value(item.metadata, "entity", "segment") or "")
            or None,
            currency=_currency(item.metadata),
            scale=_scale(item.metadata),
            candidate_key=_candidate_key(item),
            evidence_chunk_id=item.chunk_id,
            document_name=item.document_name,
            page=item.page,
            row_label=str(metric),
            column_header=None,
            table_title=str(
                _metadata_value(item.metadata, "table_title", "title") or ""
            )
            or None,
            extraction_method="plain_text_metric_period_value",
            confidence=0.8,
        ),
    )


def extract_financial_facts(
    evidence: tuple[EvidenceItem, ...],
) -> tuple[FinancialFact, ...]:
    """Extract high-confidence facts without falling back to first-number logic."""
    facts = []
    for item in evidence:
        extracted = _facts_from_structured_row(item)
        if not extracted:
            extracted = _facts_from_serialized_row(item)
        if not extracted:
            extracted = _facts_from_plain_text(item)
        facts.extend(extracted)
    return tuple(facts)


def _question_periods(intent: CalculationIntent) -> tuple[str | None, str | None]:
    periods = tuple(normalize_period(value) for value in intent.period_terms)
    periods = tuple(value for value in periods if value)
    return (periods[0], periods[1]) if len(periods) >= 2 else (None, None)


def _metric_before(question: str, marker: str) -> str | None:
    normalized = " ".join(question.split())
    lowered = normalized.lower()
    index = lowered.find(marker)
    if index < 0:
        return None
    value = normalized[:index]
    value = re.sub(r"^what was the .*? of ", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"^what percentage of .*?\bFY\s*\d{4}\s+", "", value, flags=re.IGNORECASE
    )
    return value.strip(" :?") or None


def build_operand_specs(
    *,
    question: str,
    routing_decision: RoutingDecision,
    calculation_intent: CalculationIntent,
) -> tuple[OperandBindingSpec, ...]:
    """Build semantic operand constraints from a routed derived-value question."""
    operation = routing_decision.operation
    if operation is None:
        return ()
    if operation.value == "growth_rate":
        old, new = _question_periods(calculation_intent)
        metric = _metric_before(question, " reported by") or _metric_before(
            question, " from"
        )
        return (
            OperandBindingSpec("previous", metric, old, None, None, None),
            OperandBindingSpec("current", metric, new, None, None, None),
        )
    if operation.value == "percentage_share":
        lowered = question.lower()
        marker = " came from "
        if marker not in lowered:
            return ()
        index = lowered.index(marker)
        total = _metric_before(question, marker)
        part = question[index + len(marker) :].rstrip(" ?.")
        period = normalize_period(next(iter(calculation_intent.period_terms), ""))
        return (
            OperandBindingSpec("part", part, period, None, None, None),
            OperandBindingSpec("total", total, period, None, None, None),
        )
    if operation.value == "difference":
        match = re.search(r":\s*(.+?)\s+or\s+(.+?)(?:,|\?|$)", question, re.IGNORECASE)
        period = normalize_period(next(iter(calculation_intent.period_terms), ""))
        if not match:
            return ()
        return (
            OperandBindingSpec(
                "minuend", match.group(1).strip(), period, None, None, None
            ),
            OperandBindingSpec(
                "subtrahend", match.group(2).strip(), period, None, None, None
            ),
        )
    return ()


def _matches(spec: OperandBindingSpec, fact: FinancialFact) -> bool:
    if spec.metric and normalize_metric_name(spec.metric) != normalize_metric_name(
        fact.metric
    ):
        return False
    if spec.period and normalize_period(spec.period) != normalize_period(fact.period):
        return False
    if spec.entity and normalize_metric_name(spec.entity) != normalize_metric_name(
        fact.entity or ""
    ):
        return False
    if spec.currency and spec.currency != fact.currency:
        return False
    if spec.expected_scale and spec.expected_scale != fact.scale:
        return False
    return fact.candidate_key is not None and bool(fact.evidence_chunk_id)


def bind_operands(
    specs: tuple[OperandBindingSpec, ...],
    facts: Iterable[FinancialFact],
) -> BindingResult:
    """Bind each spec to one unique, unambiguous fact or return a block reason."""
    facts = tuple(facts)
    bound = []
    used: set[tuple[str | None, str | None, Decimal]] = set()
    for spec in specs:
        matches = [fact for fact in facts if _matches(spec, fact)]
        unique = {
            (fact.candidate_key, fact.period, fact.value): fact for fact in matches
        }
        if not unique:
            return BindingResult((), specs, "OPERAND_MISSING")
        if len(unique) != 1:
            return BindingResult((), specs, "OPERAND_AMBIGUOUS")
        fact = next(iter(unique.values()))
        identity = (fact.candidate_key, fact.period, fact.value)
        if identity in used:
            return BindingResult((), specs, "ROLE_ASSIGNMENT_AMBIGUOUS")
        used.add(identity)
        bound.append(BoundOperand(spec.role, fact, fact.value))
    return BindingResult(tuple(bound), specs)


def adapt_bound_operands(
    bound: tuple[BoundOperand, ...],
    operation: str,
) -> tuple[CalculationOperand, ...]:
    """Adapt semantic roles to the existing unchanged calculator operand order."""
    by_role = {item.role: item for item in bound}
    if operation == "growth_rate":
        ordered = (by_role["current"], by_role["previous"])
    elif operation == "difference":
        ordered = (by_role["minuend"], by_role["subtrahend"])
    else:
        ordered = bound
    return tuple(
        CalculationOperand(
            name=item.role,
            value=item.normalized_value,
            unit=item.fact.currency,
            scale=item.fact.scale,
            source_text=item.fact.row_label or item.fact.metric,
            evidence_chunk_id=item.fact.evidence_chunk_id,
            document_name=item.fact.document_name,
            page=item.fact.page,
        )
        for item in ordered
    )
