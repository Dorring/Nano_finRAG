"""Question-only Retrieval V3 profile router; it has no benchmark dependencies."""

from __future__ import annotations

import re

from src.finance.calculation_intent import detect_calculation_intent
from src.retrieval_v3.models import QueryProfile
from src.retrieval_v3.query_features import extract_metric_phrases, extract_periods, extract_statement_hint, normalize_question


_NARRATIVE = ("why", "explain", "describe", "reason", "risk", "strategy", "discussion", "policy", "note")
_COMPARISON = ("compare", "which was higher", "difference between", " versus ", " vs ", "both ", "how much more", "how much less")
_TABLE_FACT = ("what was", "what were", "how much", "how many", "amount", "percentage", "reported", "net income", "revenue", "sales", "assets", "margin")


def route_question(question: str, *, document_scope: tuple[str, ...] = ()) -> QueryProfile:
    normalized = normalize_question(question)
    issuer = document_scope[0] if len(document_scope) == 1 else None
    periods, unresolved = extract_periods(normalized)
    metrics = extract_metric_phrases(normalized, periods)
    statement_hint = extract_statement_hint(normalized)
    if not normalized:
        return QueryProfile("unsupported", issuer, (), (), None, 0, False, None, False, True, (), ("empty_question",))
    calculation = detect_calculation_intent(normalized)
    if calculation.requires_calculation and calculation.operation is not None:
        count = int(calculation.expected_operand_count or 0)
        return QueryProfile("calculation_multi_operand" if count else "unsupported", issuer, metrics, periods, calculation.operation.value if count else None, count, count > 1, statement_hint, False, True, ("calculation_intent", *calculation.matched_signals), unresolved if count else (*unresolved, "calculation_operand_count_unresolved"))
    lowered = normalized.lower()
    if any(signal in lowered for signal in _COMPARISON) and len(metrics) >= 2:
        return QueryProfile("multi_metric_comparison", issuer, metrics, periods, None, len(metrics), True, statement_hint, False, True, ("explicit_comparison",), unresolved)
    if len(periods) >= 2 and len(metrics) == 1:
        return QueryProfile("single_metric_multi_period", issuer, metrics, periods, None, len(periods), True, statement_hint, False, True, ("multiple_explicit_periods",), unresolved)
    if any(re.search(rf"\b{re.escape(signal)}\b", lowered) for signal in _NARRATIVE):
        return QueryProfile("narrative_or_note", issuer, metrics, periods, None, 1, False, statement_hint, True, True, ("narrative_signal",), unresolved)
    if len(metrics) != 1:
        return QueryProfile("unsupported", issuer, metrics, periods, None, 0, False, statement_hint, False, True, (), (*unresolved, "metric_count_unresolved"))
    task_type = "table_single_fact" if any(signal in lowered for signal in _TABLE_FACT) else "general_single_fact"
    return QueryProfile(task_type, issuer, metrics, periods, None, 1, False, statement_hint, False, True, ("single_fact",), unresolved)
