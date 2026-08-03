"""Operation router: decides whether a question should enter the calculation pipeline.

The router is deliberately *conservative*: it only routes to the deterministic
calculation pipeline when ALL of the following are true:

1. ``intent["intent"] == "financial_calculation"`` (first gate, from
   ``src.services.intent.classify_query_intent``).
2. A recognized metric alias or generic operation keyword is present in the
   question text.
3. An explicit calculation verb/pattern is present (e.g. "calculate",
   "compute", "计算", "根据...计算", "derive"). This prevents reported-metric
   lookups like "毛利率是多少" from entering the calculation pipeline even
   when the intent classifier labels them ``financial_calculation``.

If any gate fails, the router returns ``NOT_APPLICABLE`` and the orchestrator
continues the normal RAG flow (LLM / deterministic answer extractor).

Layer dependency: ``domain -> finance -> application -> services``. This
module imports from ``src.domain.calculation`` and ``src.finance.metric_lexicon``
(both allowed) and stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain.calculation import CalculationOperation, CalculationStatus
from src.finance.calculation_intent import detect_calculation_intent
from src.finance.metric_lexicon import (
    GenericOperationEntry,
    MetricDefinition,
    find_generic_operation,
    find_metric_by_alias,
)


# Explicit calculation verbs/patterns. At least one must be present for the
# router to enter the calculation pipeline. This is the conservative gate
# that prevents reported-metric lookups from being routed to calculation.
EXPLICIT_CALCULATION_PATTERNS: tuple[str, ...] = (
    "calculate",
    "compute",
    "derive",
    "work out",
    "convert",
    "根据",
    "计算",
    "求",
    "换算",
)


@dataclass(frozen=True)
class RoutingDecision:
    """The outcome of routing a question through the calculation pipeline gate.

    - ``READY``: the question should enter the calculation pipeline. The
      router has identified the operation, metric, formula version, and
      expected operand roles.
    - ``NOT_APPLICABLE``: the question should continue the normal RAG flow.
      ``reason`` explains why routing was declined.
    """

    status: CalculationStatus
    operation: CalculationOperation | None = None
    metric: str | None = None
    formula_version: str | None = None
    operand_roles: tuple[str, ...] = ()
    formula_template: str | None = None
    unit: str | None = None
    target_scale: str | None = None
    reason: str = ""

    @classmethod
    def not_applicable(cls, reason: str) -> "RoutingDecision":
        return cls(status=CalculationStatus.NOT_APPLICABLE, reason=reason)

    @classmethod
    def from_metric(cls, definition: MetricDefinition) -> "RoutingDecision":
        return cls(
            status=CalculationStatus.READY,
            operation=definition.operation,
            metric=definition.metric,
            formula_version=definition.formula_version,
            operand_roles=definition.operand_roles,
            formula_template=definition.formula_template,
            unit=definition.unit,
            reason="named_metric_matched",
        )

    @classmethod
    def from_generic(
        cls, entry: GenericOperationEntry, target_scale: str | None = None
    ) -> "RoutingDecision":
        return cls(
            status=CalculationStatus.READY,
            operation=entry.operation,
            metric=entry.operation.value,
            formula_version=entry.formula_version,
            operand_roles=(),
            formula_template=entry.formula_template,
            unit=entry.unit,
            target_scale=target_scale,
            reason="generic_operation_matched",
        )


def _has_explicit_calculation_pattern(text: str) -> bool:
    """Check whether the question contains an explicit calculation verb."""
    lowered = text.lower()
    return any(
        pattern in lowered or pattern in text
        for pattern in EXPLICIT_CALCULATION_PATTERNS
    )


_TARGET_SCALE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("\u4ebf\u5143", "\u4ebf\u5143"),
    ("\u767e\u4e07\u5143", "\u767e\u4e07\u5143"),
    ("\u5343\u4e07\u5143", "\u5343\u4e07\u5143"),
    ("\u4e07\u5143", "\u4e07\u5143"),
    ("\u4ebf", "\u4ebf"),
    ("\u767e\u4e07", "\u767e\u4e07"),
    ("\u5343\u4e07", "\u5343\u4e07"),
    ("\u4e07", "\u4e07"),
    ("billion", "billion"),
    ("million", "million"),
    ("thousand", "thousand"),
)

_CURRENCY_KEYWORDS: tuple[str, ...] = (
    "\u7f8e\u5143",
    "\u4eba\u6c11\u5e01",
    "\u6b27\u5143",
    "\u65e5\u5143",
    "\u82f1\u9551",
    "\u6e2f\u5e01",
    "\u6fb3\u5143",
    "usd",
    "cny",
    "rmb",
    "eur",
    "jpy",
    "gbp",
    "hkd",
    "aud",
)


def _extract_target_scale(question: str) -> str | None:
    """Extract the target scale from a scale-conversion question.

    Returns the canonical scale name, or None if not found.
    Returns "__CURRENCY__" if a currency keyword is detected.
    """
    lowered = question.lower()

    for kw in _CURRENCY_KEYWORDS:
        if kw in lowered or kw in question:
            return "__CURRENCY__"

    has_conversion_verb = any(
        marker in question or marker in lowered
        for marker in (
            "\u6362\u7b97",
            "\u8f6c\u6362",
            "convert",
            "in terms of",
            "expressed in",
        )
    )
    if not has_conversion_verb:
        return None

    for pattern, canonical in _TARGET_SCALE_PATTERNS:
        if pattern in question or pattern in lowered:
            return canonical

    return None


def route_calculation(
    question: str,
    intent: dict[str, Any],
    *,
    allow_derived_document_qa: bool = False,
) -> RoutingDecision:
    """Route a question to the calculation pipeline or decline (NOT_APPLICABLE).

    Args:
        question: The user's (possibly rewritten) question text.
        intent: The intent dict returned by ``classify_query_intent``.

    Returns:
        A ``RoutingDecision`` with ``status`` READY or NOT_APPLICABLE.
    """
    calculation_intent = detect_calculation_intent(question)
    if (
        allow_derived_document_qa
        and calculation_intent.requires_calculation
        and calculation_intent.derived_value_requested
        and calculation_intent.operation is not None
    ):
        # A fully structured derived-value request can override document_qa.
        # This is the NF-OPT-05 addition; no direct reported metric qualifies.
        metric_def = find_metric_by_alias(question)
        if (
            metric_def is not None
            and metric_def.operation is calculation_intent.operation
        ):
            decision = RoutingDecision.from_metric(metric_def)
            return RoutingDecision(
                **{**decision.__dict__, "reason": "derived_value_named_metric"}
            )
        generic_entry = find_generic_operation(question)
        if (
            generic_entry is not None
            and generic_entry.operation is calculation_intent.operation
        ):
            target_scale = (
                _extract_target_scale(question)
                if generic_entry.operation is CalculationOperation.SCALE_CONVERSION
                else None
            )
            decision = RoutingDecision.from_generic(
                generic_entry, target_scale=target_scale
            )
            return RoutingDecision(
                **{**decision.__dict__, "reason": "derived_value_generic_operation"}
            )
        generic_by_operation = {
            CalculationOperation.DIFFERENCE: (
                "difference.v1",
                "current - previous",
                "base",
            ),
            CalculationOperation.SUM: ("sum.v1", "sum(operands)", "base"),
            CalculationOperation.AVERAGE: (
                "average.v1",
                "sum(operands) / count",
                "base",
            ),
        }
        fallback = generic_by_operation.get(calculation_intent.operation)
        if fallback is not None:
            formula_version, formula_template, unit = fallback
            return RoutingDecision(
                status=CalculationStatus.READY,
                operation=calculation_intent.operation,
                metric=calculation_intent.operation.value,
                formula_version=formula_version,
                formula_template=formula_template,
                unit=unit,
                reason="derived_value_intent",
            )

    # Preserve the production calculator's explicit-intent contract for
    # callers already classified as financial_calculation.
    if intent.get("intent") != "financial_calculation":
        return RoutingDecision.not_applicable("intent_not_financial_calculation")
    if not _has_explicit_calculation_pattern(question):
        return RoutingDecision.not_applicable("no_explicit_calculation_verb")
    metric_def = find_metric_by_alias(question)
    if metric_def is not None:
        return RoutingDecision.from_metric(metric_def)
    generic_entry = find_generic_operation(question)
    if generic_entry is not None:
        target_scale = None
        if generic_entry.operation is CalculationOperation.SCALE_CONVERSION:
            target_scale = _extract_target_scale(question)
        return RoutingDecision.from_generic(generic_entry, target_scale=target_scale)
    return RoutingDecision.not_applicable("no_metric_or_operation_matched")
