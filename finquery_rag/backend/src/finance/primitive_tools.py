"""Deterministic financial calculation primitives for FinQuery.

Phase 3 Commit 3: migrated from ``src.services.financial_tools`` to
``src.finance.primitive_tools`` so the finance layer is the canonical home
for pure calculation logic. The original module is kept as a thin
re-export shim for backward compatibility.

These helpers are intentionally pure: they call no LLM and no external system,
and RAG answers can cite their structured outputs later. All arithmetic uses
``Decimal`` with ``ROUND_HALF_UP`` and rejects ``NaN``/``Infinity`` implicitly
via ``Decimal`` construction.

They do depend on ``rag_v2.contracts.financial_semantics`` for one thing, and it
is deliberate. ``RepresentationKind`` is the project's single statement of what a
percentage is, and ``quantities_are_comparable`` is its single statement of when
two quantities may be compared at all. A second opinion here -- particularly
about whether a percentage is 21 or 0.21 -- is how the two came to be 100x apart
in the first place. That module imports the standard library and nothing else,
so this stays a pure calculation layer.

New primitives added in Phase 3:
- ``difference`` — subtraction
- ``average_values`` — arithmetic mean
- ``gross_margin`` — (revenue - cogs) / revenue
- ``net_margin`` — net_income / revenue
- ``debt_ratio`` — total_liabilities / total_assets
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from rag_v2.contracts.financial_semantics import (
    CanonicalQuantity,
    MagnitudeScale,
    RepresentationKind,
    quantities_are_comparable,
)


__all__ = [
    "ToolResult",
    "parse_financial_number",
    "growth_rate",
    "percentage_share",
    "sum_values",
    "verify_sum",
    "convert_scale",
    "format_ratio_percent",
    "difference",
    "average_values",
    "gross_margin",
    "net_margin",
    "debt_ratio",
]


#: A sign, an optional accounting paren, a number, and -- either side of the
#: closing paren and either side of a space -- the percent sign.
#:
#: Percent detection used to be whitespace-sensitive, because ``(?P<percent>%?)``
#: sat immediately after the digits and a space pushed the sign into the suffix.
#: ``21%`` was therefore 0.21 and ``21 %`` was 21: one quantity, written two ways,
#: 100x apart, decided by a blank character.  The fact store writes both -- 305
#: glued, 222 spaced -- so this was not a corner.
#:
#: The percent sign is matched in two places because it legitimately appears in
#: two: after the number (``5.49 %``) and after the closing paren of an
#: accounting negative (``(0.3)%``).  ``suffix`` no longer accepts ``%`` so that
#: a trailing sign cannot be silently swallowed as scale text.
_NUMBER_RE = re.compile(
    r"^\s*(?P<prefix>[^\d\-\(]*)?"
    r"(?P<negative>\()?"
    r"(?P<number>[-+]?\d+(?:,\d{3})*(?:\.\d+)?)"
    r"\s*(?P<percent>%?)"
    r"(?P<suffix>[^\d\)%]*)?"
    r"(?P<close>\))?"
    r"\s*(?P<percent_after>%?)\s*$"
)

_SCALE_FACTORS = {
    "": Decimal("1"),
    "ones": Decimal("1"),
    "unit": Decimal("1"),
    "thousand": Decimal("1000"),
    "k": Decimal("1000"),
    "million": Decimal("1000000"),
    "m": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
    "万": Decimal("10000"),
    "万元": Decimal("10000"),
    "千万": Decimal("10000000"),
    "千万元": Decimal("10000000"),
    "百万": Decimal("1000000"),
    "百万元": Decimal("1000000"),
    "亿": Decimal("100000000"),
    "亿元": Decimal("100000000"),
}


@dataclass(frozen=True)
class ToolResult:
    """A calculation result whose number has a named reading.

    There is no ``value`` field.  A percentage is 21 points and also the factor
    0.21, and a single unnamed field can only carry one of them -- so every
    reader of the old one was quietly choosing a side, and the finance layer and
    the shared contract did not choose the same way.

    ``points_value`` is the number as the input stated it: ``21 %`` -> 21,
    ``$5`` -> 5.  Display, percentage-point arithmetic and the benchmark gold
    all mean this one.
    ``ratio_value`` is the multiplicative form: ``21 %`` -> 0.21, and equal to
    ``points_value`` for anything that is not a percentage.

    A result computed rather than parsed -- a sum, a mean, a margin -- carries
    the outcome in both fields, with ``representation`` saying which kind of
    quantity it is, so a caller can still ask the shared contract whether it may
    be compared with something else.
    """

    ok: bool
    points_value: Decimal | None = None
    ratio_value: Decimal | None = None
    representation: RepresentationKind = RepresentationKind.ABSOLUTE
    error: str | None = None
    unit: str | None = None
    details: dict[str, Any] | None = None

    def as_quantity(self) -> CanonicalQuantity | None:
        """This result as the shared quantity type, or ``None`` if it holds none.

        The rule for whether two quantities may be compared lives in one place
        and is written against ``CanonicalQuantity``.  This is how a primitive
        reaches it instead of restating it.
        """

        if self.points_value is None or self.ratio_value is None:
            return None
        return CanonicalQuantity(
            points_value=self.points_value,
            ratio_value=self.ratio_value,
            magnitude=MagnitudeScale.BASE,
            unit=self.unit or "",
            representation=self.representation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "points_value": (
                str(self.points_value) if self.points_value is not None else None
            ),
            "ratio_value": (
                str(self.ratio_value) if self.ratio_value is not None else None
            ),
            "representation": self.representation.value,
            "error": self.error,
            "unit": self.unit,
            "details": self.details or {},
        }


def _computed(
    value: Decimal,
    representation: RepresentationKind,
    *,
    unit: str,
    details: dict[str, Any] | None = None,
    ok: bool = True,
    error: str | None = None,
) -> ToolResult:
    """A computed result, both readings derived from one representation.

    The single place the division happens, so no operation can carry a
    percentage in one field and an amount in the other.
    """

    return ToolResult(
        ok,
        points_value=value,
        ratio_value=(
            value / Decimal("100")
            if representation is RepresentationKind.PERCENT
            else value
        ),
        representation=representation,
        error=error,
        unit=unit,
        details=details,
    )


def _share_one_representation(results: list[ToolResult]) -> RepresentationKind | None:
    """The kind a result computed *from* these carries, or ``None`` if mixed.

    A sum or a mean of percentages is a percentage and of amounts an amount.  A
    mixture is not a quantity either reading describes, so it is reported as no
    kind at all rather than silently as one of them.
    """

    representations = {item.representation for item in results}
    if len(representations) == 1:
        return representations.pop()
    return None


def parse_financial_number(value: Any, scale: str | None = None) -> ToolResult:
    """Parse numeric financial text into a quantity.

    Handles commas, percentages, surrounding currency text, accounting negatives
    like ``(1,234)``, and common English/Chinese scale words.

    A percentage returns both readings: ``12.5%`` is 12.5 points and the factor
    0.125.  Both come from one rule, and that rule cannot see whitespace --
    ``12.5%``, ``12.5 %`` and ``(0.3)%`` are one quantity written three ways.
    """

    if isinstance(value, ToolResult):
        return value
    if isinstance(value, Decimal):
        return ToolResult(True, points_value=value, ratio_value=value,
                          details={"scale": scale or ""})
    if isinstance(value, (int, float)):
        number = Decimal(str(value))
        return ToolResult(True, points_value=number, ratio_value=number,
                          details={"scale": scale or ""})

    text = str(value).strip()
    match = _NUMBER_RE.match(text)
    if not match:
        return ToolResult(False, error=f"Cannot parse financial number: {value!r}")

    number_text = match.group("number").replace(",", "")
    try:
        number = Decimal(number_text)
    except InvalidOperation:
        return ToolResult(False, error=f"Invalid numeric value: {value!r}")

    if match.group("negative") == "(" and match.group("close") == ")":
        number = -abs(number)

    suffix = (match.group("suffix") or "").strip()
    inferred_scale = scale or _infer_scale(suffix)
    factor = _scale_factor(inferred_scale)
    if factor is None:
        return ToolResult(False, error=f"Unknown scale: {inferred_scale!r}")

    is_percent = (match.group("percent") or match.group("percent_after")) == "%"
    points = number * factor
    return ToolResult(
        True,
        points_value=points,
        ratio_value=points / Decimal("100") if is_percent else points,
        representation=(
            RepresentationKind.PERCENT if is_percent else RepresentationKind.ABSOLUTE
        ),
        unit="base",
        details={
            "input": value,
            "scale": inferred_scale or "",
            "is_percent": is_percent,
        },
    )


def growth_rate(current: Any, previous: Any, precision: int = 4) -> ToolResult:
    """Return (current - previous) / previous.

    Reads ``points_value`` from both operands.  A growth rate is scale-invariant
    -- 11 and 13 give the same -0.1538 as 0.11 and 0.13 do -- so either reading
    computes the same number here.  One is still named, because an unnamed read
    is how the two readings came to be 100x apart in the first place.
    """
    cur = _ensure_decimal(current)
    prev = _ensure_decimal(previous)
    if not cur.ok:
        return cur
    if not prev.ok:
        return prev
    current_value = _points(cur)
    previous_value = _points(prev)
    if previous_value == 0:
        return ToolResult(
            False, error="Cannot compute growth rate with zero previous value"
        )
    value = (current_value - previous_value) / previous_value
    return _computed(
        _quantize(value, precision),
        RepresentationKind.RATIO,
        unit="ratio",
        details={"current": str(current_value), "previous": str(previous_value)},
    )


def percentage_share(part: Any, total: Any, precision: int = 4) -> ToolResult:
    """Return part / total.

    Reads ``points_value`` from both, and this is the operation the two readings
    actually disagree about.  The benchmark gold computes ``21 / (486)`` for a
    part stated as ``21%`` -- the percentage as written -- so reading the ratio
    form here would put the answer exactly 100x out.
    """
    p = _ensure_decimal(part)
    t = _ensure_decimal(total)
    if not p.ok:
        return p
    if not t.ok:
        return t
    part_value = _points(p)
    total_value = _points(t)
    if total_value == 0:
        return ToolResult(False, error="Cannot compute share with zero total")
    value = part_value / total_value
    return _computed(
        _quantize(value, precision),
        RepresentationKind.RATIO,
        unit="ratio",
        details={"part": str(part_value), "total": str(total_value)},
    )


def sum_values(values: list[Any], precision: int = 2) -> ToolResult:
    """Return sum(values).

    Reads ``points_value`` from each -- the sum of percentages is a sum in
    percentage points -- and refuses a mixture of kinds outright.  A percentage
    plus an amount is not a quantity either reading describes, and returning a
    number for it would be the failure this whole change is about.
    """
    parsed = []
    for value in values:
        item = _ensure_decimal(value)
        if not item.ok:
            return item
        parsed.append(item)
    representation = _share_one_representation(parsed)
    if representation is None:
        return ToolResult(False, error="Cannot sum quantities of different kinds")
    total = sum((_points(item) for item in parsed), Decimal("0"))
    return _computed(
        _quantize(total, precision),
        representation,
        unit="base",
        details={"count": len(parsed), "values": [str(_points(i)) for i in parsed]},
    )


def verify_sum(
    components: list[Any],
    reported_total: Any,
    tolerance: Any = Decimal("0.01"),
    precision: int = 2,
) -> ToolResult:
    """Check whether sum(components) matches reported_total within tolerance.

    The only comparison among the primitives, so it asks the shared contract
    whether the two sides may be compared at all before comparing them.  A
    percentage total checked against an amount sum is refused rather than
    answered: the difference is computable and means nothing.
    """
    summed = sum_values(components, precision=precision)
    if not summed.ok:
        return summed
    reported = _ensure_decimal(reported_total)
    tol = _ensure_decimal(tolerance)
    if not reported.ok:
        return reported
    if not tol.ok:
        return tol

    left = summed.as_quantity()
    right = reported.as_quantity()
    if left is None or right is None or not quantities_are_comparable(left, right):
        return ToolResult(
            False,
            error="Cannot compare a total with components of another kind",
        )

    diff = abs(_points(summed) - _points(reported))
    ok = diff <= abs(_points(tol))
    return _computed(
        _quantize(diff, precision),
        summed.representation,
        unit="base",
        ok=ok,
        error=None if ok else "Reported total does not match component sum",
        details={
            "computed_total": str(_points(summed)),
            "reported_total": str(_points(reported)),
            "tolerance": str(_points(tol)),
        },
    )


def convert_scale(
    value: Any, from_scale: str, to_scale: str, precision: int = 4
) -> ToolResult:
    """Convert a value from one scale to another, e.g. million -> billion.

    Reads ``points_value`` and returns the same reading.  A scale changes how
    large a number is, never what kind of quantity it is: a percentage converted
    from millions to billions is the same percentage.
    """
    val = _ensure_decimal(value)
    if not val.ok:
        return val
    from_factor = _scale_factor(from_scale)
    to_factor = _scale_factor(to_scale)
    if from_factor is None:
        return ToolResult(False, error=f"Unknown source scale: {from_scale!r}")
    if to_factor is None:
        return ToolResult(False, error=f"Unknown target scale: {to_scale!r}")
    converted = _points(val) * from_factor / to_factor
    return _computed(
        _quantize(converted, precision),
        val.representation,
        unit=to_scale,
        details={"from_scale": from_scale, "to_scale": to_scale},
    )


def format_ratio_percent(value: Any, precision: int = 2) -> ToolResult:
    """Format a ratio as a percentage, e.g. 0.125 -> 12.50.

    The one operation that multiplies, and therefore the one that reads
    ``ratio_value``.  Its result is a percentage, and both readings of the
    result are returned: 12.50 points, and the factor 0.125 it came from.
    """
    val = _ensure_decimal(value)
    if not val.ok:
        return val
    points = _quantize(_ratio(val) * Decimal("100"), precision)
    return ToolResult(
        True,
        points_value=points,
        ratio_value=points / Decimal("100"),
        representation=RepresentationKind.PERCENT,
        unit="percent",
    )


# ---------------------------------------------------------------------------
# New primitives added in Phase 3
# ---------------------------------------------------------------------------


def difference(current: Any, previous: Any, precision: int = 2) -> ToolResult:
    """Return current - previous.

    Reads ``points_value``: the difference of two percentages is a difference in
    percentage points, which is what a reader of the answer expects to see.

    Formula version: ``difference.v1``
    """
    cur = _ensure_decimal(current)
    prev = _ensure_decimal(previous)
    if not cur.ok:
        return cur
    if not prev.ok:
        return prev
    representation = _share_one_representation([cur, prev])
    if representation is None:
        return ToolResult(
            False, error="Cannot subtract quantities of different kinds"
        )
    value = _points(cur) - _points(prev)
    return _computed(
        _quantize(value, precision),
        representation,
        unit="base",
        details={"current": str(_points(cur)), "previous": str(_points(prev))},
    )


def average_values(values: list[Any], precision: int = 4) -> ToolResult:
    """Return the arithmetic mean of ``values``.

    Reads ``points_value``.  The mean of 32 % and 32 % is 32 percentage points,
    which is what the benchmark gold states for ``average-002``; taking the mean
    of the ratio forms would answer 0.32.

    Formula version: ``average.v1``
    """
    parsed = []
    for value in values:
        item = _ensure_decimal(value)
        if not item.ok:
            return item
        parsed.append(item)
    if not parsed:
        return ToolResult(False, error="Cannot compute average of empty list")
    representation = _share_one_representation(parsed)
    if representation is None:
        return ToolResult(
            False, error="Cannot average quantities of different kinds"
        )
    total = sum((_points(item) for item in parsed), Decimal("0"))
    avg = total / Decimal(len(parsed))
    return _computed(
        _quantize(avg, precision),
        representation,
        unit="base",
        details={"count": len(parsed), "values": [str(_points(i)) for i in parsed]},
    )


def gross_margin(revenue: Any, cogs: Any, precision: int = 4) -> ToolResult:
    """Return (revenue - cogs) / revenue.

    Reads ``points_value``.  Both operands are amounts in the same unit, so the
    two readings coincide; naming one keeps the rule uniform across the layer.

    Formula version: ``gross_margin.v1``
    """
    rev = _ensure_decimal(revenue)
    cog = _ensure_decimal(cogs)
    if not rev.ok:
        return rev
    if not cog.ok:
        return cog
    revenue_value = _points(rev)
    cogs_value = _points(cog)
    if revenue_value == 0:
        return ToolResult(False, error="Cannot compute gross margin with zero revenue")
    value = (revenue_value - cogs_value) / revenue_value
    return _computed(
        _quantize(value, precision),
        RepresentationKind.RATIO,
        unit="ratio",
        details={"revenue": str(revenue_value), "cogs": str(cogs_value)},
    )


def net_margin(revenue: Any, net_income: Any, precision: int = 4) -> ToolResult:
    """Return net_income / revenue.

    Reads ``points_value``, as ``gross_margin`` does and for the same reason.

    Formula version: ``net_margin.v1``
    """
    rev = _ensure_decimal(revenue)
    ni = _ensure_decimal(net_income)
    if not rev.ok:
        return rev
    if not ni.ok:
        return ni
    revenue_value = _points(rev)
    income_value = _points(ni)
    if revenue_value == 0:
        return ToolResult(False, error="Cannot compute net margin with zero revenue")
    value = income_value / revenue_value
    return _computed(
        _quantize(value, precision),
        RepresentationKind.RATIO,
        unit="ratio",
        details={"revenue": str(revenue_value), "net_income": str(income_value)},
    )


def debt_ratio(
    total_liabilities: Any, total_assets: Any, precision: int = 4
) -> ToolResult:
    """Return total_liabilities / total_assets.

    Reads ``points_value``, as the other ratio primitives do.

    Formula version: ``debt_ratio.v1``
    """
    tl = _ensure_decimal(total_liabilities)
    ta = _ensure_decimal(total_assets)
    if not tl.ok:
        return tl
    if not ta.ok:
        return ta
    liabilities = _points(tl)
    assets = _points(ta)
    if assets == 0:
        return ToolResult(
            False, error="Cannot compute debt ratio with zero total assets"
        )
    value = liabilities / assets
    return _computed(
        _quantize(value, precision),
        RepresentationKind.RATIO,
        unit="ratio",
        details={"total_liabilities": str(liabilities), "total_assets": str(assets)},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_decimal(value: Any) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    return parse_financial_number(value)


def _points(result: ToolResult) -> Decimal:
    """The stated reading of an operand.

    Raises rather than defaulting.  Every caller has already checked ``ok``, so a
    missing value here is a bug in the caller -- and a default would turn it into
    a silently wrong number instead of a loud failure.
    """

    if result.points_value is None:
        raise ValueError("operand carries no points_value")
    return result.points_value


def _ratio(result: ToolResult) -> Decimal:
    """The multiplicative reading of an operand."""

    if result.ratio_value is None:
        raise ValueError("operand carries no ratio_value")
    return result.ratio_value


def _scale_factor(scale: str | None) -> Decimal | None:
    return _SCALE_FACTORS.get((scale or "").strip().lower())


def _infer_scale(text: str) -> str:
    lowered = text.lower()
    for key in sorted(_SCALE_FACTORS, key=len, reverse=True):
        if key and key in lowered:
            return key
    return ""


def _quantize(value: Decimal, precision: int) -> Decimal:
    exponent = Decimal("1").scaleb(-precision)
    return value.quantize(exponent, rounding=ROUND_HALF_UP)
