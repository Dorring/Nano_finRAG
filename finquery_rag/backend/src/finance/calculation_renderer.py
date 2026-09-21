"""Deterministic calculation renderer: turn a CalculationResult into text.

The renderer converts a ``CalculationResult`` into a human-readable answer
string suitable for direct return to the user (when the orchestrator
bypasses the LLM). It is purely deterministic — no LLM calls, no
retrieval, no side effects.

Rendering rules:
- ``EXECUTED``  -> a multi-line answer with the metric name, computed
  value, formula, and per-operand evidence citations.
- ``BLOCKED``   -> a single-line deterministic refusal explaining why the
  calculation could not be performed.
- ``FAILED``    -> a safe failure message (no internal stack exposed).
- ``NOT_APPLICABLE`` -> empty string (orchestrator continues to LLM).

Value formatting:
- ``ratio`` unit -> percentage with 2 decimal places (0.4 -> "40.00%").
- ``base`` unit  -> thousands-separated decimal with 2 decimal places.
- other units    -> raw Decimal string.

Layer dependency: ``domain -> finance -> application -> services``. This
module imports from ``src.domain.calculation`` and its own layer's
``src.finance.relational_directory`` -- both allowed -- plus stdlib. It
deliberately does not import the supervisor: a renderer that could read the plan
would be able to build a second `slot_id -> entity` lookup, and one that could
recompute would be a second executor.
"""

from __future__ import annotations

from decimal import Decimal

from src.domain.calculation import (
    RELATIONAL_OPERATIONS,
    CalculationOperand,
    CalculationResult,
    CalculationStatus,
)
from src.finance.relational_directory import (
    RelationalOperandDirectory,
    render_relational_result,
    resolve_relational_result,
)


def render_calculation_result(
    result: CalculationResult,
    *,
    directory: "RelationalOperandDirectory | None" = None,
) -> str:
    """Render a ``CalculationResult`` into a human-readable answer string.

    Returns a non-empty string for EXECUTED and BLOCKED results.
    Returns an empty string for NOT_APPLICABLE and FAILED (the LLM
    handles those).

    A **relational** result is different in kind and says so.  Its answer is an
    ordering over ``slot_id``s, which cannot be stated without names for them,
    so it needs a `RelationalOperandDirectory`; without one this returns ``""``
    rather than falling through to the quantity path.  Falling through would
    format ``value``, and a relational result has none -- so the failure would
    be a crash, or worse, whatever ``None`` happened to render as.
    """

    if result.operation in RELATIONAL_OPERATIONS:
        if directory is None:
            return ""
        resolved = resolve_relational_result(result, directory)
        return "" if resolved is None else render_relational_result(resolved)

    if result.status is CalculationStatus.EXECUTED:
        return _render_executed(result)
    if result.status is CalculationStatus.BLOCKED:
        return _render_blocked(result)
    if result.status is CalculationStatus.FAILED:
        return _render_failed(result)
    # NOT_APPLICABLE: the LLM handles this.
    return ""


def _render_executed(result: CalculationResult) -> str:
    """Render an EXECUTED result with value, formula, and evidence."""
    metric_label = _metric_label(result.target_metric)
    value_str = _format_value(result.value, result.unit)
    formula_line = _format_formula_line(result)

    lines: list[str] = [
        f"{metric_label}: {value_str}",
        "",
        formula_line,
    ]

    if result.operands:
        lines.append("Inputs:")
        for operand in result.operands:
            lines.append(f"  - {_format_operand(operand)}")

    return "\n".join(lines)


def _render_blocked(result: CalculationResult) -> str:
    """Render a BLOCKED result as a deterministic refusal."""
    metric_label = _metric_label(result.target_metric)
    reason = result.error_message or "calculation could not be completed"
    return f"Unable to compute {metric_label}: {reason}"


def _render_failed(result: CalculationResult) -> str:
    """Render a FAILED result as a safe failure message.

    The internal error/stack is never exposed to the user.
    """
    metric_label = _metric_label(result.target_metric)
    return (
        f"Unable to compute {metric_label} due to an internal calculation "
        f"error. Please try rephrasing the question or consult the source "
        f"documents directly."
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _metric_label(target_metric: str | None) -> str:
    """Convert a metric key (e.g. 'gross_margin') to a display label."""
    if not target_metric:
        return "Result"
    return target_metric.replace("_", " ").title()


def _format_value(value: Decimal | None, unit: str | None) -> str:
    """Format a Decimal value according to its unit."""
    if value is None:
        return "N/A"
    if unit == "ratio":
        percentage = value * Decimal("100")
        return f"{_format_decimal(percentage, 2)}%"
    if unit == "base":
        return _format_decimal(value, 2)
    return str(value)


def _format_decimal(value: Decimal, precision: int) -> str:
    """Format a Decimal with thousands separator and fixed precision."""
    quantized = value.quantize(Decimal("1").scaleb(-precision))
    # Use comma as thousands separator.
    s = f"{quantized:,.{precision}f}"
    return s


def _format_formula_line(result: CalculationResult) -> str:
    """Build the 'Formula: ... (version)' line."""
    parts: list[str] = []
    if result.formula:
        parts.append(f"Formula: {result.formula}")
    if result.formula_version:
        parts.append(f"({result.formula_version})")
    if not parts:
        return ""
    return " ".join(parts)


def _format_operand(operand: CalculationOperand) -> str:
    """Format a single operand with its value and evidence citation."""
    value_str = _format_decimal(operand.value, 2)
    citation = _format_citation(operand)
    if citation:
        return f"{operand.name} = {value_str} — {citation}"
    return f"{operand.name} = {value_str}"


def _format_citation(operand: CalculationOperand) -> str:
    """Build an evidence citation string from operand metadata."""
    parts: list[str] = []
    if operand.document_name:
        parts.append(operand.document_name)
    if operand.page is not None:
        parts.append(f"p.{operand.page}")
    if not parts and operand.evidence_chunk_id:
        parts.append(f"chunk {operand.evidence_chunk_id}")
    return ", ".join(parts)
