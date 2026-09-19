"""Calculation registry: maps CalculationOperation to primitive functions.

The registry is the single source of truth for how each deterministic
financial operation is executed. It binds each ``CalculationOperation``
to:

- The primitive function (via an adapter) that performs the arithmetic.
- The human-readable formula string.
- The formula version string (for traceability).
- The result unit.
- Operand count constraints (min/max).
- The expected operand roles (for validation by the plan builder).

The executor consults this registry to look up and dispatch calculations.

Layer dependency: ``domain -> finance -> application -> services``. This
module imports from ``src.domain.calculation`` and ``src.finance.primitive_tools``
(both allowed) and stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
    ComparisonRelation,
)
from src.finance.primitive_tools import (
    ToolResult,
    average_values,
    debt_ratio,
    difference,
    gross_margin,
    growth_rate,
    net_margin,
    percentage_share,
    sum_values,
)


#: A ranking's operand ceiling.  A bound rather than a limit anyone expects to
#: reach -- the largest question in the canonical set ranks four companies -- so
#: that `max_operands` stays a real constraint rather than a placeholder.
MAX_RANKING_OPERANDS = 64

# Type alias for the adapter functions that wrap primitive_tools calls.
# Each adapter takes a tuple of operands and a precision int and returns a
# ``ToolResult`` -- or, for a relational operation, a ``RelationalToolResult``.
# The union is what lets the nine arithmetic adapters stay byte-identical while
# two operations answer with a relation instead of a quantity.
CalculationFunc = Callable[
    [tuple[CalculationOperand, ...], int],
    "ToolResult | RelationalToolResult",
]


@dataclass(frozen=True)
class RelationalToolResult:
    """What a relational adapter returns, before it becomes a result.

    Executor-internal, exactly as ``ToolResult`` is: this is the adapter's
    return, not the Harness's artifact.  ``CalculationResult`` remains the one
    authority the validator reads, so nothing downstream has to know this type
    exists.

    ``ok`` mirrors ``ToolResult.ok`` so that an adapter can decline
    deterministically -- operands that may not be ordered against each other are
    a refusal, not an exception -- and the executor maps both kinds of decline
    through the same BLOCKED path.

    Exactly one of ``relation`` / ``ordering_groups`` is populated on success,
    and which one is fixed by the operation's own contract rather than by the
    executor reading the operation's name.  Dispatching on a name would put a
    second copy of "which operation expects which shape" beside the adapters
    that already embody it.
    """

    ok: bool
    relation: "ComparisonRelation | None" = None
    ordering_groups: tuple[tuple[str, ...], ...] | None = None
    error: str | None = None


@dataclass(frozen=True)
class OperationEntry:
    """Registry entry binding an operation to its execution metadata."""

    operation: CalculationOperation
    func: CalculationFunc
    formula: str
    formula_version: str
    unit: str
    min_operands: int
    max_operands: int
    operand_roles: tuple[str, ...]


# ---------------------------------------------------------------------------
# Adapter functions
# ---------------------------------------------------------------------------
# Each adapter extracts Decimal values from CalculationOperand instances
# and delegates to the corresponding primitive_tools function. Adapters
# never raise; they return ``ToolResult(ok=False, ...)`` on misuse so the
# executor can map the outcome to a BLOCKED result.


def _two_operand_adapter(
    primitive: Callable,
    precision: int,
    operands: tuple[CalculationOperand, ...],
    role_names: tuple[str, str],
) -> ToolResult:
    if len(operands) < 2:
        return ToolResult(
            False,
            error=f"requires 2 operands ({role_names[0]}, {role_names[1]}), "
            f"got {len(operands)}",
        )
    return primitive(operands[0].value, operands[1].value, precision=precision)


def _difference_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    return _two_operand_adapter(
        difference, precision, operands, ("current", "previous")
    )


def _growth_rate_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    return _two_operand_adapter(
        growth_rate, precision, operands, ("current", "previous")
    )


def _percentage_share_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    return _two_operand_adapter(
        percentage_share, precision, operands, ("part", "total")
    )


def _gross_margin_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    return _two_operand_adapter(gross_margin, precision, operands, ("revenue", "cogs"))


def _net_margin_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    return _two_operand_adapter(
        net_margin, precision, operands, ("revenue", "net_income")
    )


def _debt_ratio_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    return _two_operand_adapter(
        debt_ratio, precision, operands, ("total_liabilities", "total_assets")
    )


def _sum_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    if not operands:
        return ToolResult(False, error="sum requires at least 1 operand, got 0")
    values = [op.value for op in operands]
    return sum_values(values, precision=precision)


def _average_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    if not operands:
        return ToolResult(False, error="average requires at least 1 operand, got 0")
    values = [op.value for op in operands]
    return average_values(values, precision=precision)


def _scale_conversion_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    # Scale conversion requires explicit from_scale/to_scale parameters that
    # are not carried by CalculationOperand. For v1, this adapter exists for
    # registry completeness but always declines; the executor will map the
    # decline to a BLOCKED result.
    return ToolResult(
        False,
        error="scale_conversion requires explicit from/to scale parameters "
        "not available in the plan",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _comparison_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> RelationalToolResult:
    """Which of exactly two operands is larger, as a stated relation.

    ``difference`` is doing double duty here, deliberately.  It already refuses
    operands whose kinds or units do not agree, so ordering reuses that rule
    instead of restating compatibility -- a second rule would be a second
    authority for "may these two numbers be compared", and the two would drift.

    The sign becomes a ``ComparisonRelation`` rather than being left as a
    number: the caller wants "which operand is larger", and a bare ``-1`` does
    not answer that without also knowing which operand was first.
    """

    if len(operands) != 2:
        return RelationalToolResult(
            ok=False, error=f"comparison requires exactly 2 operands, got {len(operands)}"
        )
    delta = difference(operands[0].value, operands[1].value, precision=precision)
    if not delta.ok or delta.points_value is None:
        return RelationalToolResult(
            ok=False, error=delta.error or "operands_are_not_comparable"
        )
    if delta.points_value > 0:
        relation = ComparisonRelation.LHS_GT_RHS
    elif delta.points_value < 0:
        relation = ComparisonRelation.RHS_GT_LHS
    else:
        relation = ComparisonRelation.EQUAL
    return RelationalToolResult(ok=True, relation=relation)


def _ranking_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> RelationalToolResult:
    """Every operand, ordered descending, with equals grouped.

    ``average_values`` is the compatibility probe for the same reason
    ``difference`` is for a comparison: it refuses a mixture of kinds outright,
    so ranking inherits that rule rather than restating it.

    The refs are ``slot_id``.  A ranking is a claim about which *required
    operands* stand in what order; keying it on evidence would make the same
    ranking a different result whenever a different support of one canonical
    fact happened to be bound.
    """

    if len(operands) < 2:
        return RelationalToolResult(
            ok=False, error=f"ranking requires at least 2 operands, got {len(operands)}"
        )
    probe = average_values([operand.value for operand in operands], precision=precision)
    if not probe.ok:
        return RelationalToolResult(
            ok=False, error=probe.error or "operands_are_not_comparable"
        )

    # `slot_id` breaks ties in the *ordering of construction* only; equal values
    # still land in one group, so this cannot change the result, only the order
    # groups are emitted in -- and groups are compared as an ordering of sets.
    ordered = sorted(operands, key=lambda item: (-item.value, item.slot_id))
    groups: list[tuple[str, ...]] = []
    previous: object = object()
    for operand in ordered:
        if groups and operand.value == previous:
            groups[-1] = groups[-1] + (operand.slot_id,)
        else:
            groups.append((operand.slot_id,))
        previous = operand.value
    return RelationalToolResult(ok=True, ordering_groups=tuple(groups))


CALCULATION_REGISTRY: dict[CalculationOperation, OperationEntry] = {
    CalculationOperation.DIFFERENCE: OperationEntry(
        operation=CalculationOperation.DIFFERENCE,
        func=_difference_adapter,
        formula="current - previous",
        formula_version="difference.v1",
        unit="base",
        min_operands=2,
        max_operands=2,
        operand_roles=("current", "previous"),
    ),
    CalculationOperation.GROWTH_RATE: OperationEntry(
        operation=CalculationOperation.GROWTH_RATE,
        func=_growth_rate_adapter,
        formula="(current - previous) / previous",
        formula_version="growth_rate.v1",
        unit="ratio",
        min_operands=2,
        max_operands=2,
        operand_roles=("current", "previous"),
    ),
    CalculationOperation.PERCENTAGE_SHARE: OperationEntry(
        operation=CalculationOperation.PERCENTAGE_SHARE,
        func=_percentage_share_adapter,
        formula="part / total",
        formula_version="percentage_share.v1",
        unit="ratio",
        min_operands=2,
        max_operands=2,
        operand_roles=("part", "total"),
    ),
    CalculationOperation.SUM: OperationEntry(
        operation=CalculationOperation.SUM,
        func=_sum_adapter,
        formula="sum(operands)",
        formula_version="sum.v1",
        unit="base",
        min_operands=1,
        max_operands=100,
        operand_roles=(),
    ),
    CalculationOperation.AVERAGE: OperationEntry(
        operation=CalculationOperation.AVERAGE,
        func=_average_adapter,
        formula="sum(operands) / count",
        formula_version="average.v1",
        unit="base",
        min_operands=1,
        max_operands=100,
        operand_roles=(),
    ),
    CalculationOperation.GROSS_MARGIN: OperationEntry(
        operation=CalculationOperation.GROSS_MARGIN,
        func=_gross_margin_adapter,
        formula="(revenue - cogs) / revenue",
        formula_version="gross_margin.v1",
        unit="ratio",
        min_operands=2,
        max_operands=2,
        operand_roles=("revenue", "cogs"),
    ),
    CalculationOperation.NET_MARGIN: OperationEntry(
        operation=CalculationOperation.NET_MARGIN,
        func=_net_margin_adapter,
        formula="net_income / revenue",
        formula_version="net_margin.v1",
        unit="ratio",
        min_operands=2,
        max_operands=2,
        operand_roles=("revenue", "net_income"),
    ),
    CalculationOperation.DEBT_RATIO: OperationEntry(
        operation=CalculationOperation.DEBT_RATIO,
        func=_debt_ratio_adapter,
        formula="total_liabilities / total_assets",
        formula_version="debt_ratio.v1",
        unit="ratio",
        min_operands=2,
        max_operands=2,
        operand_roles=("total_liabilities", "total_assets"),
    ),
    CalculationOperation.SCALE_CONVERSION: OperationEntry(
        operation=CalculationOperation.SCALE_CONVERSION,
        func=_scale_conversion_adapter,
        formula="value * from_factor / to_factor",
        formula_version="scale_conversion.v1",
        unit="base",
        min_operands=1,
        max_operands=1,
        operand_roles=("value",),
    ),
    # Relational operations.  Their adapters return ``RelationalToolResult``
    # rather than ``ToolResult``, and the executor dispatches on the returned
    # type -- never on the operation's name, which would put a second copy of
    # "which operation expects which shape" beside the adapters that embody it.
    CalculationOperation.COMPARISON: OperationEntry(
        operation=CalculationOperation.COMPARISON,
        func=_comparison_adapter,
        formula="sign(lhs - rhs) as an explicit relation",
        formula_version="comparison.v1",
        unit="relation",
        min_operands=2,
        max_operands=2,
        operand_roles=("lhs", "rhs"),
    ),
    CalculationOperation.RANKING: OperationEntry(
        operation=CalculationOperation.RANKING,
        func=_ranking_adapter,
        formula="operands ordered descending, equals grouped",
        formula_version="ranking.v1",
        unit="ordering",
        min_operands=2,
        max_operands=MAX_RANKING_OPERANDS,
        # No fixed roles: a ranking is over however many slots the plan asked
        # for, and naming them here would be a second statement of the plan's
        # cardinality.
        operand_roles=(),
    ),
}


def get_operation_entry(
    operation: CalculationOperation,
) -> OperationEntry | None:
    """Return the registry entry for ``operation``, or None if not registered."""
    return CALCULATION_REGISTRY.get(operation)
