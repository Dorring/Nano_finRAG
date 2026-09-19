"""Calculation executor: runs a CalculationPlan to produce a CalculationResult.

The executor is the final step in the deterministic calculation pipeline.
It takes a ``CalculationPlan`` (built by the plan builder from a routing
decision and extracted operands) and produces a ``CalculationResult``:

- If the plan is already ``BLOCKED`` or ``NOT_APPLICABLE``, the result
  mirrors that status without invoking any primitive.
- If the plan is ``READY``, the executor validates operand count, looks
  up the primitive function from ``CALCULATION_REGISTRY``, and calls it.
  - Success -> ``EXECUTED`` with the computed value.
  - Primitive returns ``ok=False`` -> ``BLOCKED`` (deterministic refusal;
    e.g. division by zero, insufficient operands from adapter).
  - Unexpected exception -> ``FAILED`` (bypass LLM, return safe failure;
    never fall back to LLM to avoid reintroducing numeric hallucinations).

The executor NEVER calls the LLM and NEVER accesses retrieval. It is
purely deterministic and side-effect-free.

Layer dependency: ``domain -> finance -> application -> services``. This
module imports from ``src.domain.calculation`` and
``src.finance.calculation_registry`` (both allowed) and stdlib only.
"""

from __future__ import annotations

import logging
import re

from src.domain.calculation import (
    CalculationOperation,
    CalculationPlan,
    CalculationResult,
    CalculationStatus,
)
from src.finance.calculation_registry import RelationalToolResult, get_operation_entry
from src.finance.primitive_tools import convert_scale

logger = logging.getLogger(__name__)


def _execute_scale_conversion(plan: CalculationPlan) -> CalculationResult:
    """Execute a SCALE_CONVERSION plan using convert_scale primitive."""

    if not plan.source_scale or not plan.target_scale:
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=CalculationOperation.SCALE_CONVERSION,
            formula_version=plan.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="UNIT_AMBIGUOUS",
            error_message="source or target scale is missing",
        )

    if plan.target_scale == "__CURRENCY__":
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=CalculationOperation.SCALE_CONVERSION,
            formula_version=plan.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="CURRENCY_NOT_SUPPORTED",
            error_message="currency conversion is not supported",
        )

    if not plan.operands:
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=CalculationOperation.SCALE_CONVERSION,
            formula_version=plan.formula_version,
            target_metric=plan.target_metric,
            operands=(),
            error_code="INSUFFICIENT_OPERANDS",
            error_message="no numeric operand provided",
        )

    operand = plan.operands[0]
    try:
        tool_result = convert_scale(
            operand.value,
            plan.source_scale,
            plan.target_scale,
            precision=plan.precision,
        )
    except Exception as exc:
        return CalculationResult(
            status=CalculationStatus.FAILED,
            operation=CalculationOperation.SCALE_CONVERSION,
            formula_version=plan.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="PRIMITIVE_EXCEPTION",
            error_message=str(exc),
        )

    if not tool_result.ok or tool_result.points_value is None:
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=CalculationOperation.SCALE_CONVERSION,
            formula_version=plan.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="PRIMITIVE_DECLINED",
            error_message=tool_result.error or "scale conversion failed",
        )

    formula = "value * from_factor / to_factor"
    return CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation.SCALE_CONVERSION,
        value=tool_result.points_value,
        unit=plan.target_scale,
        formula=formula,
        formula_version=plan.formula_version,
        target_metric=plan.target_metric,
        operands=plan.operands,
    )


def execute_plan(plan: CalculationPlan) -> CalculationResult:
    """Execute a ``CalculationPlan`` and return a ``CalculationResult``.

    Args:
        plan: The calculation plan to execute. May be READY, BLOCKED, or
            NOT_APPLICABLE.

    Returns:
        A ``CalculationResult`` with status EXECUTED, BLOCKED, FAILED, or
        NOT_APPLICABLE.
    """
    # Pass-through for non-READY plans.
    if plan.status is CalculationStatus.NOT_APPLICABLE:
        return CalculationResult(status=CalculationStatus.NOT_APPLICABLE)

    if plan.status is CalculationStatus.BLOCKED:
        reason = plan.block_reason or "plan was blocked before execution"
        # Extract error code from block_reason if it matches ERROR_CODE: format.
        # Only all-caps identifiers before ":" are treated as error codes.
        m = re.match(r"^([A-Z][A-Z_]+):\s*(.*)", reason)
        if m:
            error_code = m.group(1)
            error_message = m.group(2) or reason
        else:
            error_code = "PLAN_BLOCKED"
            error_message = reason
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=plan.operation,
            formula_version=plan.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code=error_code,
            error_message=error_message,
        )

    if plan.status is not CalculationStatus.READY:
        # FAILED or EXECUTED plans should not reach the executor; mirror as-is.
        return CalculationResult(
            status=plan.status,
            operation=plan.operation,
            formula_version=plan.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
        )

    # SCALE_CONVERSION is handled specially via convert_scale primitive.
    if plan.operation is CalculationOperation.SCALE_CONVERSION:
        return _execute_scale_conversion(plan)

    # READY plan: look up the registry entry.
    entry = get_operation_entry(plan.operation)
    if entry is None:
        return CalculationResult(
            status=CalculationStatus.FAILED,
            operation=plan.operation,
            formula_version=plan.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="UNKNOWN_OPERATION",
            error_message=f"no registry entry for operation {plan.operation!r}",
        )

    # Validate operand count.
    operand_count = len(plan.operands)
    if operand_count < entry.min_operands:
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=plan.operation,
            formula=entry.formula,
            formula_version=entry.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="INSUFFICIENT_OPERANDS",
            error_message=(
                f"{plan.operation.value} requires at least {entry.min_operands} "
                f"operands, got {operand_count}"
            ),
        )
    if operand_count > entry.max_operands:
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=plan.operation,
            formula=entry.formula,
            formula_version=entry.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="TOO_MANY_OPERANDS",
            error_message=(
                f"{plan.operation.value} accepts at most {entry.max_operands} "
                f"operands, got {operand_count}"
            ),
        )

    # Execute the primitive function.
    try:
        result = entry.func(plan.operands, plan.precision)
    except Exception as exc:  # noqa: BLE001 - executor must not raise
        logger.warning("calculation primitive raised: %s", exc, exc_info=True)
        return CalculationResult(
            status=CalculationStatus.FAILED,
            operation=plan.operation,
            formula=entry.formula,
            formula_version=entry.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="PRIMITIVE_EXCEPTION",
            error_message=str(exc),
        )

    # The stated reading, and it is the same number the old unnamed field held
    # for every operation whose result is not a percentage: a ratio result
    # carries ``points_value == ratio_value``, and ``_format_value`` applies the
    # x100 for ``unit == "ratio"`` itself.  It differs only for a percentage
    # operand, where the stated number is the one the gold arithmetic uses.
    # Dispatch on what the adapter *returned*, never on the operation's name.
    # A name-based branch would be a second authority for "which operation
    # produces which shape", and it would start disagreeing with the adapter the
    # first time either changed.
    if isinstance(result, RelationalToolResult):
        if not result.ok:
            # Same BLOCKED path a declining primitive takes: operands that may
            # not be ordered against each other are a deterministic refusal, so
            # the orchestrator bypasses the LLM with a refusal rather than
            # letting prose answer a comparison nothing could verify.
            return CalculationResult(
                status=CalculationStatus.BLOCKED,
                operation=plan.operation,
                formula=entry.formula,
                formula_version=entry.formula_version,
                target_metric=plan.target_metric,
                operands=plan.operands,
                error_code="PRIMITIVE_DECLINED",
                error_message=result.error or "relational primitive declined",
            )
        # No `value`.  The relation *is* the answer, and a sign derived from it
        # would be a second field that can disagree with the first -- which is
        # the shape of defect this phase exists to remove.  A caller wanting the
        # sign can read it off `relation`.
        return CalculationResult(
            status=CalculationStatus.EXECUTED,
            operation=plan.operation,
            unit=entry.unit,
            formula=entry.formula,
            formula_version=entry.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            relation=result.relation,
            ordering_groups=result.ordering_groups,
        )

    if not result.ok or result.points_value is None:
        # Primitive declined (e.g. division by zero, missing scale params).
        # This is a deterministic refusal, so we BLOCK rather than FAILED
        # so the orchestrator bypasses the LLM with a deterministic refusal.
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=plan.operation,
            formula=entry.formula,
            formula_version=entry.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="PRIMITIVE_DECLINED",
            error_message=result.error or "primitive returned without a value",
        )

    # Success.
    return CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=plan.operation,
        value=result.points_value,
        unit=entry.unit,
        formula=entry.formula,
        formula_version=entry.formula_version,
        target_metric=plan.target_metric,
        operands=plan.operands,
    )
