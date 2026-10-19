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

from collections.abc import Mapping
import dataclasses
from decimal import Decimal
import logging
import re
from typing import Any

from rag_v2.contracts.plan import ExecutionDAG
from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
    CalculationPlan,
    CalculationResult,
    CalculationStatus,
    CalculationStepRecord,
    IntermediateValue,
)
from src.finance.calculation_registry import (
    RelationalToolResult,
    _canonical_unit,
    get_operation_entry,
)
from src.finance.calibrated_abstention import format_fail_closed_explanation
from src.finance.primitive_tools import _SCALE_FACTORS, convert_scale

logger = logging.getLogger(__name__)


_SCALE_FACTOR_TO_NAME: dict[Decimal, str] = {
    Decimal("1"): "units",
    Decimal("1000"): "thousands",
    Decimal("10000"): "万",
    Decimal("1000000"): "million",
    Decimal("100000000"): "亿",
    Decimal("1000000000"): "billion",
    Decimal("1000000000000"): "trillion",
}


def _get_operand_scale_factor(op: Any) -> Decimal:
    """Extract the magnitude scale factor for an operand."""
    scale = getattr(op, "scale", None)
    if not scale:
        return Decimal("1")
    key = str(scale).strip().lower()
    return _SCALE_FACTORS.get(key, Decimal("1"))


def _harmonize_operand_scales(
    operands: tuple[Any, ...],
    operation: CalculationOperation | str | None = None,
) -> tuple[Any, ...]:
    """Harmonize differing scale magnitudes across operands into a unified base scale.

    For example, if one operand has scale='million' ($391,035M) and another has
    scale='units' ($342) or scale='thousands' ($383,285K), they are converted
    to the finest common scale (e.g. units) to eliminate 10^6 magnitude errors
    and avoid false-positive unit mismatch rejections.

    Preserves Decimal precision with zero floating point drift.
    """
    if len(operands) <= 1:
        return operands

    op_enum = None
    if isinstance(operation, CalculationOperation):
        op_enum = operation
    elif operation is not None:
        try:
            op_enum = CalculationOperation(str(operation))
        except ValueError:
            op_enum = None

    # For CAGR: If 3 operands are passed (current, previous, periods), only harmonize current & previous
    if op_enum is CalculationOperation.CAGR:
        if len(operands) >= 3:
            value_ops = operands[:2]
            tail_ops = operands[2:]
        else:
            value_ops = operands
            tail_ops = ()
    else:
        value_ops = operands
        tail_ops = ()

    factors = [_get_operand_scale_factor(op) for op in value_ops]
    distinct_factors = set(factors)
    if len(distinct_factors) <= 1:
        # All operands already have the same scale factor (or none)
        return operands

    # Pick the finest scale (minimum factor) to convert by multiplication without fractional loss
    target_factor = min(distinct_factors)
    target_scale = _SCALE_FACTOR_TO_NAME.get(target_factor, "units")

    harmonized_values: list[Any] = []
    for op, factor in zip(value_ops, factors):
        if factor == target_factor:
            harmonized_values.append(op)
        else:
            multiplier = factor / target_factor
            new_value = op.value * multiplier
            if hasattr(op, "_replace"):  # namedtuple
                new_op = op._replace(value=new_value, scale=target_scale)
            elif dataclasses.is_dataclass(op) and not isinstance(op, type):
                new_op = dataclasses.replace(op, value=new_value, scale=target_scale)
            else:
                new_op = op
            harmonized_values.append(new_op)

    return tuple(harmonized_values) + tuple(tail_ops)


def _harmonize_operand_currencies(
    operands: tuple[Any, ...],
    target_currency: str = "USD",
    operation: Any = None,
) -> tuple[Any, ...]:
    """Harmonize differing currencies across operands using deterministic FX conversion."""
    if len(operands) <= 1:
        return operands

    op_name = getattr(operation, "value", str(operation)).lower() if operation else ""
    if op_name in ("comparison", "ranking"):
        return operands

    from rag_v2.sources.market_api import get_fx_rate

    currencies = [getattr(op, "unit", None) for op in operands if getattr(op, "unit", None)]
    distinct_currencies = {
        str(c).strip().upper()
        for c in currencies
        if str(c).strip().upper() in ("USD", "EUR", "CNY", "GBP", "JPY", "HKD", "CAD", "AUD", "CHF")
    }

    if len(distinct_currencies) <= 1:
        return operands

    target_curr = target_currency.upper()
    harmonized = []
    for op in operands:
        curr = getattr(op, "unit", None)
        curr_clean = str(curr).strip().upper() if curr else ""
        if curr_clean and curr_clean in distinct_currencies and curr_clean != target_curr:
            rate = get_fx_rate(curr_clean, target_curr)
            if rate is not None:
                new_value = op.value * rate
                if hasattr(op, "_replace"):
                    new_op = op._replace(value=new_value, unit=target_curr)
                elif dataclasses.is_dataclass(op) and not isinstance(op, type):
                    new_op = dataclasses.replace(op, value=new_value, unit=target_curr)
                else:
                    new_op = op
                harmonized.append(new_op)
                continue
        harmonized.append(op)
    return tuple(harmonized)


def _build_proof_tree(
    operation: Any,
    operands: tuple[Any, ...],
    value: Decimal | None,
    unit: str | None,
    formula: str | None,
) -> dict[str, Any]:
    """Build financial audit certificate and proof tree for compliance verification."""
    op_name = getattr(operation, "value", str(operation)) if operation else "calculation"
    operands_info = []
    for op in operands:
        operands_info.append({
            "name": getattr(op, "name", str(op)),
            "value": str(getattr(op, "value", "")),
            "unit": getattr(op, "unit", None),
            "scale": getattr(op, "scale", None),
            "document": getattr(op, "document_name", None),
            "page": getattr(op, "page", None),
            "slot_id": getattr(op, "slot_id", None),
            "excerpt": getattr(op, "source_text", "")[:120] if getattr(op, "source_text", None) else "",
        })

    # LaTeX mathematical representation
    latex = ""
    ops_vals = [str(getattr(op, "value", "")) for op in operands]
    if op_name == "cagr" and len(ops_vals) >= 2:
        latex = f"\\text{{CAGR}} = \\left(\\frac{{{ops_vals[0]}}}{{{ops_vals[1]}}}\\right)^{{1/n}} - 1 = {value}"
    elif op_name == "valuation_multiple" and len(ops_vals) >= 2:
        latex = f"\\text{{Multiple}} = \\frac{{{ops_vals[0]}}}{{{ops_vals[1]}}} = {value}"
    elif op_name == "spread" and len(ops_vals) >= 2:
        latex = f"\\text{{Spread}} = {ops_vals[0]} - {ops_vals[1]} = {value}"
    elif op_name == "median":
        latex = f"\\text{{Median}}({', '.join(ops_vals)}) = {value}"
    elif formula:
        latex = f"\\text{{{op_name}}}: {formula} = {value} {unit or ''}".strip()

    return {
        "operation": op_name,
        "latex_formula": latex,
        "computed_value": str(value) if value is not None else None,
        "unit": unit,
        "formula": formula,
        "operands_count": len(operands),
        "operands": operands_info,
        "deterministic_verification": "PASS_FAIL_CLOSED_VALIDATED",
    }



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
        harmonized_operands = _harmonize_operand_scales(plan.operands, plan.operation)
        harmonized_operands = _harmonize_operand_currencies(harmonized_operands, operation=plan.operation)
        result = entry.func(harmonized_operands, plan.precision)
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
            err_msg = result.error or "relational primitive declined"
            diag = format_fail_closed_explanation(err_msg, operation=getattr(plan.operation, "value", str(plan.operation)))
            return CalculationResult(
                status=CalculationStatus.BLOCKED,
                operation=plan.operation,
                formula=entry.formula,
                formula_version=entry.formula_version,
                target_metric=plan.target_metric,
                operands=plan.operands,
                error_code="PRIMITIVE_DECLINED",
                error_message=err_msg,
                diagnostic_explanation=diag,
            )
        # No `value`.  The ordering *is* the answer, and a sign derived from it
        # would be a second field that can disagree with the first -- which is
        # the shape of defect this phase exists to remove.  A caller wanting the
        # comparison's direction reads it off `relation`, which is derived from
        # the ordering rather than stored beside it.
        proof_tree = _build_proof_tree(
            operation=plan.operation,
            operands=harmonized_operands,
            value=None,
            unit=entry.unit,
            formula=entry.formula,
        )
        return CalculationResult(
            status=CalculationStatus.EXECUTED,
            operation=plan.operation,
            unit=entry.unit,
            formula=entry.formula,
            formula_version=entry.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            ordering_groups=result.ordering_groups,
            proof_tree=proof_tree,
        )

    if not result.ok or result.points_value is None:
        # Primitive declined (e.g. division by zero, missing scale params).
        # This is a deterministic refusal, so we BLOCK rather than FAILED
        # so the orchestrator bypasses the LLM with a deterministic refusal.
        err_msg = result.error or "primitive returned without a value"
        diag = format_fail_closed_explanation(err_msg, operation=getattr(plan.operation, "value", str(plan.operation)))
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=plan.operation,
            formula=entry.formula,
            formula_version=entry.formula_version,
            target_metric=plan.target_metric,
            operands=plan.operands,
            error_code="PRIMITIVE_DECLINED",
            error_message=err_msg,
            diagnostic_explanation=diag,
        )

    # Success.
    proof_tree = _build_proof_tree(
        operation=plan.operation,
        operands=harmonized_operands,
        value=result.points_value,
        unit=entry.unit,
        formula=entry.formula,
    )
    return CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=plan.operation,
        value=result.points_value,
        unit=entry.unit,
        formula=entry.formula,
        formula_version=entry.formula_version,
        target_metric=plan.target_metric,
        operands=plan.operands,
        proof_tree=proof_tree,
    )


def execute_dag(
    dag: ExecutionDAG,
    leaf_operands: Mapping[str, CalculationOperand],
    precision: int = 4,
    target_metric: str | None = None,
) -> CalculationResult:
    """Execute a structured ExecutionDAG deterministically using registered primitives.

    Args:
        dag: An immutable ExecutionDAG instance.
        leaf_operands: Mapping from slot_id to bound CalculationOperand.
        precision: Decimal precision for calculation steps.
        target_metric: Optional overriding target metric name.

    Returns:
        CalculationResult with EXECUTED, BLOCKED, or FAILED status,
        containing steps and leaf operands.
    """
    if not dag.steps:
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            error_code="EMPTY_DAG",
            error_message="ExecutionDAG has no steps",
        )

    # Topological sorting of DAG steps
    step_ids = [s.step_id for s in dag.steps]
    step_map = {s.step_id: s for s in dag.steps}
    in_degree = {s.step_id: 0 for s in dag.steps}
    adjacency: dict[str, list[str]] = {s.step_id: [] for s in dag.steps}

    for step in dag.steps:
        for op in step.operands:
            if op.is_step:
                if op.ref_id not in step_map:
                    return CalculationResult(
                        status=CalculationStatus.BLOCKED,
                        error_code="DANGLING_STEP_REF",
                        error_message=f"Step {step.step_id} references non-existent step {op.ref_id}",
                    )
                adjacency[op.ref_id].append(step.step_id)
                in_degree[step.step_id] += 1

    queue = [s_id for s_id in step_ids if in_degree[s_id] == 0]
    topo_order: list[str] = []
    while queue:
        curr = queue.pop(0)
        topo_order.append(curr)
        for nxt in adjacency[curr]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if len(topo_order) != len(step_ids):
        return CalculationResult(
            status=CalculationStatus.BLOCKED,
            error_code="DAG_CYCLE",
            error_message="ExecutionDAG contains cycles",
        )

    step_intermediates: dict[str, IntermediateValue] = {}
    executed_records: list[CalculationStepRecord] = []
    used_leaf_operands: dict[str, CalculationOperand] = {}

    for step_id in topo_order:
        step = step_map[step_id]
        try:
            step_op_enum = CalculationOperation(step.operation)
        except ValueError:
            return CalculationResult(
                status=CalculationStatus.FAILED,
                error_code="UNKNOWN_OPERATION",
                error_message=f"Unsupported operation {step.operation!r}",
                steps=tuple(executed_records),
            )

        entry = get_operation_entry(step_op_enum)
        if entry is None:
            return CalculationResult(
                status=CalculationStatus.FAILED,
                error_code="UNKNOWN_OPERATION",
                error_message=f"No registry entry for operation {step.operation!r}",
                steps=tuple(executed_records),
            )

        step_operands: list[Any] = []
        input_refs: list[str] = []

        for ref in step.operands:
            if ref.is_slot:
                op = leaf_operands.get(ref.ref_id)
                if op is None or op.value is None:
                    return CalculationResult(
                        status=CalculationStatus.BLOCKED,
                        operation=step_op_enum,
                        error_code="INSUFFICIENT_OPERANDS",
                        error_message=f"Missing leaf slot operand {ref.ref_id!r}",
                        steps=tuple(executed_records),
                    )
                used_leaf_operands[ref.ref_id] = op
                step_operands.append(op)
                input_refs.append(ref.ref_id)
            elif ref.is_step:
                interm = step_intermediates.get(ref.ref_id)
                if interm is None:
                    return CalculationResult(
                        status=CalculationStatus.BLOCKED,
                        operation=step_op_enum,
                        error_code="MISSING_STEP_OPERAND",
                        error_message=f"Missing intermediate step {ref.ref_id!r}",
                        steps=tuple(executed_records),
                    )
                step_operands.append(interm)
                input_refs.append(ref.ref_id)

        # Count check
        if len(step_operands) < entry.min_operands:
            return CalculationResult(
                status=CalculationStatus.BLOCKED,
                operation=step_op_enum,
                error_code="INSUFFICIENT_OPERANDS",
                error_message=f"Step {step.step_id} requires at least {entry.min_operands} operands",
                steps=tuple(executed_records),
            )
        if len(step_operands) > entry.max_operands:
            return CalculationResult(
                status=CalculationStatus.BLOCKED,
                operation=step_op_enum,
                error_code="TOO_MANY_OPERANDS",
                error_message=f"Step {step.step_id} accepts at most {entry.max_operands} operands",
                steps=tuple(executed_records),
            )

        # Check unit consistency for operations that require same units
        stated_units = {
            _canonical_unit(getattr(op, "unit", None))
            for op in step_operands
            if _canonical_unit(getattr(op, "unit", None)) not in (None, "base")
        }
        if (
            step_op_enum
            in {
                CalculationOperation.DIFFERENCE,
                CalculationOperation.SUM,
                CalculationOperation.AVERAGE,
                CalculationOperation.SPREAD,
                CalculationOperation.MEDIAN,
            }
            and len(stated_units) > 1
        ):
            return CalculationResult(
                status=CalculationStatus.BLOCKED,
                operation=step_op_enum,
                error_code="UNIT_MISMATCH",
                error_message=f"Step {step.step_id} operands have mismatched units: {sorted(stated_units)}",
                steps=tuple(executed_records),
            )

        try:
            harmonized_step_operands = _harmonize_operand_scales(tuple(step_operands), step_op_enum)
            tool_res = entry.func(harmonized_step_operands, precision)
        except Exception as exc:
            logger.warning("Calculation primitive raised in DAG step %s: %s", step.step_id, exc, exc_info=True)
            return CalculationResult(
                status=CalculationStatus.FAILED,
                operation=step_op_enum,
                error_code="PRIMITIVE_EXCEPTION",
                error_message=str(exc),
                steps=tuple(executed_records),
            )

        if not tool_res.ok or tool_res.points_value is None:
            return CalculationResult(
                status=CalculationStatus.BLOCKED,
                operation=step_op_enum,
                error_code="PRIMITIVE_DECLINED",
                error_message=tool_res.error or f"Primitive declined in step {step.step_id}",
                steps=tuple(executed_records),
            )

        # Inherit unit from operands if primitive returns base unit
        operand_unit = next((getattr(op, "unit", None) for op in step_operands if getattr(op, "unit", None) not in (None, "base")), None)
        if entry.unit != "base":
            step_unit = entry.unit
        elif step.output_unit and step.output_unit != "base":
            step_unit = step.output_unit
        else:
            step_unit = operand_unit or (step.output_unit if step.output_unit else "base")
        step_scale = getattr(harmonized_step_operands[0], "scale", None) if harmonized_step_operands else None
        intermediate = IntermediateValue(
            step_id=step.step_id,
            value=tool_res.points_value,
            unit=step_unit,
            scale=step_scale,
            formula_version=entry.formula_version,
            upstream_steps=tuple(input_refs),
            name=step.step_id,
        )
        step_intermediates[step.step_id] = intermediate

        rec = CalculationStepRecord(
            step_id=step.step_id,
            operation=step_op_enum,
            value=tool_res.points_value,
            unit=step_unit,
            formula=entry.formula,
            formula_version=entry.formula_version,
            target_metric=step.target_metric or step.step_id,
            status=CalculationStatus.EXECUTED,
            input_refs=tuple(input_refs),
        )
        executed_records.append(rec)

    final_intermediate = step_intermediates.get(dag.output_step_id)
    if final_intermediate is None:
        return CalculationResult(
            status=CalculationStatus.FAILED,
            error_code="OUTPUT_STEP_NOT_FOUND",
            error_message=f"Output step {dag.output_step_id!r} not executed",
            steps=tuple(executed_records),
        )

    out_step_def = step_map[dag.output_step_id]
    final_op = CalculationOperation(out_step_def.operation)
    final_target = target_metric or dag.target_metric or out_step_def.target_metric or dag.output_step_id
    final_version = dag.definition_version or f"dag.{dag.output_step_id}.v1"
    formula_str = f"DAG({' -> '.join(s.step_id for s in executed_records)})"

    proof_tree = _build_proof_tree(
        operation=final_op,
        operands=tuple(used_leaf_operands.values()),
        value=final_intermediate.value,
        unit=final_intermediate.unit,
        formula=formula_str,
    )
    return CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=final_op,
        value=final_intermediate.value,
        unit=final_intermediate.unit,
        formula=formula_str,
        formula_version=final_version,
        target_metric=final_target,
        operands=tuple(used_leaf_operands.values()),
        steps=tuple(executed_records),
        execution_dag=dag,
        proof_tree=proof_tree,
    )
