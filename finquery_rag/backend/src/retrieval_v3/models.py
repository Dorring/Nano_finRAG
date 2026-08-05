from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


QueryTaskType = Literal[
    "table_single_fact",
    "general_single_fact",
    "single_metric_multi_period",
    "multi_metric_comparison",
    "calculation_multi_operand",
    "narrative_or_note",
    "unsupported",
]


@dataclass(frozen=True)
class MetricPhrase:
    raw_text: str
    normalized_text: str
    role: str | None = None


@dataclass(frozen=True)
class PeriodExpression:
    raw_text: str
    normalized_period: str | None
    role: str | None = None


@dataclass(frozen=True)
class QueryProfile:
    task_type: QueryTaskType
    issuer: str | None
    metric_phrases: tuple[MetricPhrase, ...]
    periods: tuple[PeriodExpression, ...]
    operation: str | None
    expected_operand_count: int
    requires_multiple_sources: bool
    statement_hint: str | None
    narrative_intent: bool
    answerability_check_required: bool
    routing_reasons: tuple[str, ...]
    unresolved_reasons: tuple[str, ...]
