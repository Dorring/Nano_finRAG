"""Structured financial fact contracts for calculation operand binding."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class FinancialFact:
    """One metric-period-value observation tied to a single evidence item."""

    metric: str
    value: Decimal
    period: str | None
    entity: str | None
    currency: str | None
    scale: str | None
    candidate_key: str | None
    evidence_chunk_id: str
    document_name: str | None
    page: int | None
    row_label: str | None
    column_header: str | None
    table_title: str | None
    extraction_method: str
    confidence: float


@dataclass(frozen=True)
class OperandBindingSpec:
    """The semantic constraints required to bind one calculation operand."""

    role: str
    metric: str | None
    period: str | None
    entity: str | None
    currency: str | None
    expected_scale: str | None


@dataclass(frozen=True)
class BoundOperand:
    """A role assigned to exactly one structured financial fact."""

    role: str
    fact: FinancialFact
    normalized_value: Decimal
