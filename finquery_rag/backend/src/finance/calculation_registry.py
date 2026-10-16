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
from decimal import Decimal
from typing import Callable

from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
)
from src.finance.primitive_tools import (
    ToolResult,
    average_values,
    compute_cagr,
    debt_ratio,
    difference,
    gross_margin,
    growth_rate,
    median_values,
    net_margin,
    percentage_share,
    spread,
    sum_values,
    valuation_multiple,
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


def _cagr_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    if len(operands) < 2:
        return ToolResult(
            False,
            error=f"cagr requires at least 2 operands (current, previous, [periods]), got {len(operands)}",
        )
    if len(operands) == 2:
        return compute_cagr(operands[0].value, operands[1].value, periods=Decimal("1"), precision=precision)
    if len(operands) == 3:
        return compute_cagr(operands[0].value, operands[1].value, periods=operands[2].value, precision=precision)
    # Chronological series: start is operands[0], end is operands[-1], periods is len - 1
    periods = Decimal(str(len(operands) - 1))
    return compute_cagr(operands[-1].value, operands[0].value, periods=periods, precision=precision)


def _valuation_multiple_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    return _two_operand_adapter(
        valuation_multiple, precision, operands, ("market_value", "fundamental_metric")
    )


def _spread_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    if not operands or len(operands) < 2:
        return ToolResult(False, error=f"spread requires at least 2 operands, got {len(operands)}")
    values = [op.value for op in operands]
    return spread(values, precision=precision)


def _median_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> ToolResult:
    if not operands:
        return ToolResult(False, error="median requires at least 1 operand, got 0")
    values = [op.value for op in operands]
    return median_values(values, precision=precision)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _canonical_unit(u: str | None) -> str | None:
    if not u:
        return None
    s = str(u).strip()
    if not s or s.lower() in ("base", "none"):
        return None
    s_lower = s.lower()
    if s in ("$", "US$") or s_lower in ("usd", "dollar", "dollars", "us dollar", "us dollars"):
        return "USD"
    if s in ("€",) or s_lower in ("eur", "euro", "euros"):
        return "EUR"
    if s in ("£",) or s_lower in ("gbp", "pound", "pounds"):
        return "GBP"
    if s in ("¥", "元") or s_lower in ("cny", "rmb", "yuan", "人民币"):
        return "CNY"
    if s_lower in ("%", "percent", "percentage"):
        return "percent"
    if s_lower in ("ratio",):
        return "ratio"
    if s_lower in ("shares", "share"):
        return "shares"
    return s.upper() if len(s) <= 4 and s.isalpha() else s_lower


def _units_agree(operands: tuple[CalculationOperand, ...]) -> bool:
    """Whether the operands are stated in a single canonical unit.

    Checked here, on the *operands*, because the primitives cannot: every
    adapter hands them ``operand.value``, so `difference` and `average_values`
    receive bare ``Decimal``s and a USD/EUR mismatch is invisible to them.

    ``None`` or "base" means the record did not state a unit, which is unknown
    rather than different, so it does not disagree with anything. Two *stated*
    units that differ after normalization always do.
    """
    stated = {_canonical_unit(operand.unit) for operand in operands if _canonical_unit(operand.unit) is not None}
    return len(stated) <= 1


def _comparison_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> RelationalToolResult:
    """Which of exactly two operands is larger, as an ordering over two.

    A comparison answers with the *same shape* a ranking does -- an ordering
    over ``slot_id`` -- rather than with a "left"/"right" relation of its own.
    That is not tidiness.  A relation naming a left and a right has to get them
    from somewhere, and the only somewhere available is the order of
    ``operands``, which is provenance and carries no meaning; comparison would
    then have kept exactly the implicit container-order authority that was
    removed from ranking.

    The operand order still reaches the subtraction below, and deliberately: it
    decides only which way round the sign comes out.  Both ways round produce the
    same ordering, so nothing of it survives into the result.
    """

    if len(operands) != 2:
        return RelationalToolResult(
            ok=False, error=f"comparison requires exactly 2 operands, got {len(operands)}"
        )
    if not _units_agree(operands):
        return RelationalToolResult(
            ok=False,
            error=(
                "operands_are_not_comparable: "
                f"{operands[0].unit!r} vs {operands[1].unit!r}"
            ),
        )

    first, second = operands
    delta = difference(first.value, second.value, precision=precision)
    if not delta.ok or delta.points_value is None:
        return RelationalToolResult(
            ok=False, error=delta.error or "operands_are_not_comparable"
        )

    if delta.points_value == 0:
        groups: tuple[tuple[str, ...], ...] = ((first.slot_id, second.slot_id),)
    elif delta.points_value > 0:
        groups = ((first.slot_id,), (second.slot_id,))
    else:
        groups = ((second.slot_id,), (first.slot_id,))
    return RelationalToolResult(ok=True, ordering_groups=groups)


def _ranking_adapter(
    operands: tuple[CalculationOperand, ...], precision: int
) -> RelationalToolResult:
    """Every operand, ordered descending, with equals grouped.

    The refs are ``slot_id``.  A ranking is a claim about which *required
    operands* stand in what order; keying it on evidence would make the same
    ranking a different result whenever a different support of one canonical
    fact happened to be bound.

    ``average_values`` is called as a cheap numeric sanity probe -- it declines
    on a mixture of kinds, which sorts would silently accept.  It cannot see
    units, so those are checked separately.
    """

    if len(operands) < 2:
        return RelationalToolResult(
            ok=False, error=f"ranking requires at least 2 operands, got {len(operands)}"
        )
    if not _units_agree(operands):
        return RelationalToolResult(
            ok=False, error="operands_are_not_comparable: mixed units"
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
    CalculationOperation.CAGR: OperationEntry(
        operation=CalculationOperation.CAGR,
        func=_cagr_adapter,
        formula="(current / previous) ** (1 / periods) - 1",
        formula_version="cagr.v1",
        unit="ratio",
        min_operands=2,
        max_operands=100,
        operand_roles=("current", "previous", "periods"),
    ),
    CalculationOperation.VALUATION_MULTIPLE: OperationEntry(
        operation=CalculationOperation.VALUATION_MULTIPLE,
        func=_valuation_multiple_adapter,
        formula="market_value / fundamental_metric",
        formula_version="valuation_multiple.v1",
        unit="multiple",
        min_operands=2,
        max_operands=2,
        operand_roles=("market_value", "fundamental_metric"),
    ),
    CalculationOperation.SPREAD: OperationEntry(
        operation=CalculationOperation.SPREAD,
        func=_spread_adapter,
        formula="max(operands) - min(operands)",
        formula_version="spread.v1",
        unit="base",
        min_operands=2,
        max_operands=100,
        operand_roles=(),
    ),
    CalculationOperation.MEDIAN: OperationEntry(
        operation=CalculationOperation.MEDIAN,
        func=_median_adapter,
        formula="median(operands)",
        formula_version="median.v1",
        unit="base",
        min_operands=1,
        max_operands=100,
        operand_roles=(),
    ),
}


def get_operation_entry(
    operation: CalculationOperation,
) -> OperationEntry | None:
    """Return the registry entry for ``operation``, or None if not registered."""
    return CALCULATION_REGISTRY.get(operation)


def supports(operation: "CalculationOperation | str | None") -> bool:
    """Whether this operation can be executed deterministically.

    The question "may the calculator run for this plan" is answered here, by the
    registry, and not by the plan's intent.  An operation is executable exactly
    when an entry describes how to execute it; intent describes what the
    question wanted, which is a routing concern.

    Those were conflated: the calculator refused anything that was not
    `Intent.CALCULATION`, and cross-entity comparison and ranking are planned as
    `MULTI_EVIDENCE`, so an entire stratum could never reach deterministic
    execution no matter what operation it named.

    ``None`` is not executable.  A plan that names no operation has nothing to
    execute, and saying so is not the same as calling it unsupported.
    """

    if operation is None:
        return False
    if isinstance(operation, CalculationOperation):
        return operation in CALCULATION_REGISTRY
    try:
        return CalculationOperation(str(operation)) in CALCULATION_REGISTRY
    except ValueError:
        return False
