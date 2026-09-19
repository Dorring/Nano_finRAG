"""Is a bound operand uniquely identifiable in the authoritative store?

`rank-005` bound The Coca-Cola Company / Interest rate contracts / FY2025 to `-2`
while the gold fact for that coordinate is `16`.  The store holds **four**
distinct values under that one coordinate -- `16`, `-3`, `-54`, `-2` -- and the
packet happened to surface only the one the Binder chose.

The deterministic ranking then executed faithfully and rendered
`Apple > Microsoft > The Coca-Cola Company`, which is wrong, and it released.
Nothing in that chain was broken: the executor did what it was given, the
renderer stated it, and the release invariant saw an admissible relational
result.  The operand was never uniquely *identifiable* and nothing asked.

**Why the authoritative store and not the packet.**  A guard scoped to the
packet would be safe only on the days top-K happens to surface a competitor:
`K=20` unsafe, `K=100` safe is not a safety property.  The question is whether
the fact is uniquely identifiable in the store, which does not depend on what
retrieval happened to return.

**Why comparable quantities and not raw distinct values.**  Two facts stating a
percentage and an amount are not competing answers to one question -- they are
different quantities, which P1.4a already established.  Counting raw distinct
values would call `compare-002`'s Visa coordinate conflicting (`1926` and `21%`)
and kill a correct release.  So the comparison is restricted to quantities
comparable with the bound one, using the shared contract's own predicate.

**This blocks rather than repairs.**  It cannot know which of `16` and `-2` is
right; recovering the dimension that separates them -- scope, table, row
hierarchy, unit -- is P1.6-A.  Until then the honest answer is to refuse, and
`rank-002` is the case that shows why: its final order matched the gold only
because `32488 > 1332 > -34550` has the same shape as `32488 > 6411 > -34550`.
A correct answer from an ungrounded operand is not a trusted success.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any

from rag_v2.contracts.financial_semantics import (
    CanonicalQuantity,
    canonical_quantity,
    quantities_are_comparable,
)


class CoordinateStatus(str, Enum):
    """Whether one semantic coordinate identifies one value."""

    #: Exactly one stored value is comparable with the bound operand.
    UNIQUE = "unique"
    #: Several stored facts, and they all state the same value.
    CONSENSUS_SAME_VALUE = "consensus_same_value"
    #: Several stored facts stating different values.  The coordinate does not
    #: identify a fact, so no operand read from it may be executed
    #: deterministically.
    CONFLICTING_VALUES = "conflicting_values"


def quantity_of(fact: Mapping[str, Any]) -> CanonicalQuantity | None:
    """The fact's value as a comparable quantity, or ``None`` if it has none.

    ``canonical_quantity`` reads the representation from the stated value --
    ``21 %`` is a percent, ``21`` is not -- which is the reading P1.4a built and
    the reason this file does not inspect text itself.
    """

    return canonical_quantity(
        fact.get("value"),
        unit=fact.get("unit"),
        scale=fact.get("scale"),
        currency=fact.get("currency"),
    )


def coordinate_status(
    bound: Mapping[str, Any],
    coordinate_facts: Iterable[Mapping[str, Any]],
) -> CoordinateStatus:
    """Classify the coordinate of ``bound`` against every fact at it.

    Restricted to facts whose quantity is *comparable with the bound fact's*.
    An incomparable one -- a percent where the bound value is an amount -- is a
    different quantity rather than a competing value, and counting it would
    refuse coordinates that are in fact unambiguous.

    A bound fact whose own value will not canonicalise is reported as
    ``CONFLICTING_VALUES``: a value that cannot be read cannot be shown to be
    unique, and defaulting it to admissible would make the guard fail open on
    exactly the malformed rows it exists to catch.
    """

    bound_quantity = quantity_of(bound)
    if bound_quantity is None:
        return CoordinateStatus.CONFLICTING_VALUES

    comparable: list[CanonicalQuantity] = []
    for fact in coordinate_facts:
        quantity = quantity_of(fact)
        if quantity is None:
            continue
        if not quantities_are_comparable(bound_quantity, quantity):
            continue
        comparable.append(quantity)

    identities = {quantity.identity for quantity in comparable}
    if len(identities) != 1:
        # Zero comparable facts is unreachable -- the bound fact is one -- and
        # more than one identity is the ambiguity this exists to catch.
        return CoordinateStatus.CONFLICTING_VALUES
    if len(comparable) == 1:
        return CoordinateStatus.UNIQUE
    # Several facts, one value.  Distinct from UNIQUE only in what it says about
    # the store; both are admissible, and the distinction is kept because a
    # consensus that later thins to a single fact is worth being able to see.
    return CoordinateStatus.CONSENSUS_SAME_VALUE
