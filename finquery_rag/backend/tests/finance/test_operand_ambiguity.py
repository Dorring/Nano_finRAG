"""P1.6-0: a bound operand must be uniquely identifiable, or it is not used.

`rank-005` released `Apple > Microsoft > The Coca-Cola Company` -- wrong -- and
every step of that chain was correct in isolation.  The store holds four
distinct values under The Coca-Cola Company / Interest rate contracts / FY2025
(`16`, `-3`, `-54`, `-2`), the packet surfaced only `-2`, the Binder bound it
lawfully, and the deterministic ranking rendered the ordering those operands
imply.  The operand was never identifiable and nothing asked.

The guard asks, of the **authoritative store**, and refuses when the answer is
ambiguous.  Two design choices carry it:

**The store, not the packet.**  A packet-scoped guard would be safe only on the
days top-K happens to surface a competitor -- `K=20` unsafe, `K=100` safe is not
a safety property.

**Comparable quantities, not raw distinct values.**  A percent and an amount are
different quantities rather than competing answers, and counting raw distinct
values would call `compare-002`'s Visa coordinate conflicting (`1926` and `21%`)
and destroy a correct release.

This blocks; it does not repair.  Which of `16` and `-2` is right is P1.6-A's
question, and until it is answered refusing is the honest response.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.finance.operand_ambiguity import CoordinateStatus, coordinate_status


def _fact(value: Any, **overrides: Any) -> dict[str, Any]:
    fact = {"value": value, "entity": "The Coca-Cola Company",
            "metric": "Interest rate contracts", "period": "FY2025"}
    fact.update(overrides)
    return fact


# --- the three states ---------------------------------------------------------


def test_one_value_is_unique() -> None:
    assert coordinate_status(_fact("1083"), [_fact("1083")]) is CoordinateStatus.UNIQUE


def test_several_facts_stating_one_value_are_a_consensus() -> None:
    """Distinct from UNIQUE only in what it says about the store; both pass."""

    status = coordinate_status(
        _fact("-8077"), [_fact("-8077", evidence_id="E1"), _fact("-8077", evidence_id="E2")]
    )

    assert status is CoordinateStatus.CONSENSUS_SAME_VALUE


def test_two_different_comparable_values_are_a_conflict() -> None:
    """`rank-005`: four values under one coordinate, one of them bound."""

    status = coordinate_status(
        _fact("-2"), [_fact("16"), _fact("-3"), _fact("-54"), _fact("-2")]
    )

    assert status is CoordinateStatus.CONFLICTING_VALUES


# --- comparability is the thing that keeps this from over-rejecting -----------


def test_a_percent_does_not_compete_with_an_amount() -> None:
    """`compare-002`'s Visa coordinate: `1926` and `21%`.

    Counting raw distinct values would call this conflicting and destroy a
    correct release.  A percent is a different quantity, not another answer to
    the same question -- which is what P1.4a established.
    """

    status = coordinate_status(_fact("1926"), [_fact("1926"), _fact("21%")])

    assert status is CoordinateStatus.UNIQUE


def test_a_percent_coordinate_that_conflicts_with_itself_still_conflicts() -> None:
    """Excluding incomparable quantities must not exempt percentages from the rule."""

    status = coordinate_status(_fact("21%"), [_fact("21%"), _fact("41%")])

    assert status is CoordinateStatus.CONFLICTING_VALUES


def test_a_conflict_among_comparable_values_survives_a_percent_bystander() -> None:
    """`rank-002`'s Tesla coordinate: 6411 / 1871 / 1332, with a `41%` alongside.

    The percent is excluded; the three amounts are not, and they conflict.  This
    is the case whose final order matched the gold only because `32488 > 1332 >
    -34550` has the same shape as `32488 > 6411 > -34550`.
    """

    status = coordinate_status(
        _fact("1332"), [_fact("6411"), _fact("1871"), _fact("41%"), _fact("1332")]
    )

    assert status is CoordinateStatus.CONFLICTING_VALUES


# --- refusing when it cannot read the value -----------------------------------


def test_a_bound_value_that_will_not_canonicalise_is_not_admissible() -> None:
    """A value that cannot be read cannot be shown to be unique.

    Defaulting it to admissible would make the guard fail open on exactly the
    malformed rows it exists to catch.
    """

    assert coordinate_status(_fact("n/a"), [_fact("1")]) is CoordinateStatus.CONFLICTING_VALUES


def test_a_bound_value_of_none_is_not_admissible() -> None:
    assert coordinate_status(_fact(None), [_fact("1")]) is CoordinateStatus.CONFLICTING_VALUES


def test_a_coordinate_that_returns_no_facts_is_not_admissible() -> None:
    """The bound fact is always one of its own coordinate's facts.

    So an empty result means the lookup did not find what it was handed, which
    is a fault rather than an unambiguous coordinate.  Failing closed here is
    what caught a mis-wired lookup during P1.6-0: it blocked every harness
    fixture instead of quietly admitting them, which is the direction a safety
    guard should break in.

    My first version of this test asserted UNIQUE -- the lenient reading -- and
    the implementation disagreed with it.  The implementation was right.
    """

    assert coordinate_status(_fact("1083"), []) is CoordinateStatus.CONFLICTING_VALUES


# --- the real coordinates, as measured from the store -------------------------


@pytest.mark.parametrize(
    ("label", "bound", "siblings", "expected"),
    [
        ("rank-005 Coca-Cola", "-2", ["16", "-3", "-54", "-2"], CoordinateStatus.CONFLICTING_VALUES),
        ("rank-002 Tesla", "1332", ["6411", "1871", "41%", "1332"], CoordinateStatus.CONFLICTING_VALUES),
        ("compare-002 Visa", "1926", ["1926", "21%"], CoordinateStatus.UNIQUE),
        ("compare-002 Apple", "-8077", ["-8077", "-8077"], CoordinateStatus.CONSENSUS_SAME_VALUE),
        ("rank-002 Microsoft", "32488", ["32488"], CoordinateStatus.UNIQUE),
        ("crossdiff-001 Tesla", "1083", ["1083"], CoordinateStatus.UNIQUE),
        ("crossdiff-001 NVIDIA", "1203", ["1203"], CoordinateStatus.UNIQUE),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_the_coordinates_measured_from_the_real_store(
    label: str, bound: str, siblings: list[str], expected: CoordinateStatus
) -> None:
    """The classification each real coordinate must get.

    The three that must keep releasing are here beside the two that must stop,
    so a future loosening of the guard fails loudly rather than silently
    restoring an ungrounded release.
    """

    assert coordinate_status(_fact(bound), [_fact(v) for v in siblings]) is expected, label
