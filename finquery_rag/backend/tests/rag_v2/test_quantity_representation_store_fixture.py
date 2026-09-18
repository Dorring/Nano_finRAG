"""The quantity store, replayed, so the split cannot drift outside a percentage.

Every coordinate tuple the fact store holds was run through the canonicaliser
twice -- once by the code before the representation split and once by the code
after -- and the fixture records what each said.  This replays it.

Two different things are asserted, and they are not equally strong:

* **The unmarked population reproduces its recorded identity.** That column was
  written by an implementation that no longer exists, so this is a real
  regression test rather than a tautology: 6,202 tuples across every amount,
  count, currency and scale the corpus writes must come out byte-identical, and
  the fixture is the only place that says what "identical" was.

* **The marked population is asserted structurally, not by copied constant.**
  Those 307 rows were produced by the implementation under test, so comparing
  them to the recorded string would prove nothing.  What is checked instead is
  the property the change is *for*: a value stating a percentage canonicalises,
  is typed PERCENT, keeps both readings one division apart, and never merges
  with the amount that shares its digits.  The literal identities are pinned by
  hand in ``test_financial_semantics_contract.py``.

The fixture excludes 18 tuples whose value is filing narrative rather than a
quantity (over 80 characters; the median value is 6).  They are corpus prose and
add nothing here -- a value that long never canonicalises, so their identity is
the text itself.

Provenance: ``scripts/evaluation/audit_percentage_representation.py``, run at
``feat/context-trust-runtime-trace`` with and without the split.  Re-run it after
any further change to the canonicaliser and regenerate the fixture deliberately.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from rag_v2.contracts import financial_semantics as fs

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/tv2_canonical_v1/quantity-coordinates-v1.jsonl"
)

#: The conclusion the differential reached, pinned so a future change that moves
#: one more tuple has to say so rather than passing quietly.
_EXPECTED_TUPLES = 6509
_EXPECTED_CHANGED = 307

_REPRESENTATION_MARKS = ("%", "percent", "percentage", "ratio")


def _rows() -> list[dict[str, Any]]:
    with _FIXTURE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


ROWS = _rows()
UNCHANGED = [row for row in ROWS if "identity_after_split" not in row]
CHANGED = [row for row in ROWS if "identity_after_split" in row]


def _identity(row: dict[str, Any]) -> str:
    return fs.quantity_identity(
        row["value"],
        scale=row["scale"],
        unit=row["unit"],
        currency=row["currency"],
    )


def _states_a_representation(value: str) -> bool:
    text = value.strip().casefold()
    return any(text.endswith(mark) for mark in _REPRESENTATION_MARKS)


def test_the_fixture_covers_the_store_it_claims_to() -> None:
    """A fixture that silently shrank would make every test below vacuous."""

    assert len(ROWS) == _EXPECTED_TUPLES
    assert len(CHANGED) == _EXPECTED_CHANGED
    assert len(UNCHANGED) + len(CHANGED) == _EXPECTED_TUPLES


def test_every_change_lands_on_a_value_that_states_a_representation() -> None:
    """The whole claim of the change, as a property rather than a count.

    If a later edit widened the grammar -- accepted a leading word, say -- the
    tuple for ``Revenue growth of 12 percent`` would move out of the literal
    population and appear here, naming itself.
    """

    strays = [row["value"] for row in CHANGED if not _states_a_representation(row["value"])]
    assert strays == []


def test_the_unmarked_population_is_untouched() -> None:
    """The drift guard.  6,202 tuples, every amount and unit the corpus writes.

    These identities were recorded by the pre-split implementation, so this is
    the one assertion here that cannot be satisfied by the code agreeing with
    itself.
    """

    drifted = [
        (row["value"], row["identity_before_split"], _identity(row))
        for row in UNCHANGED
        if _identity(row) != row["identity_before_split"]
    ]

    # Reported whole rather than truncated: the first few names are the useful
    # part, and a drifted fixture is a stop-and-look rather than a rerun.
    assert drifted == []


def test_a_percentage_now_canonicalises_and_carries_both_readings() -> None:
    for row in CHANGED:
        quantity = fs.canonical_quantity(
            row["value"],
            scale=row["scale"],
            unit=row["unit"],
            currency=row["currency"],
        )
        assert quantity is not None, row["value"]
        assert quantity.representation is fs.RepresentationKind.PERCENT, row["value"]
        assert quantity.ratio_value == quantity.points_value / Decimal("100"), row["value"]


def test_a_percentage_never_merges_with_the_amount_sharing_its_digits() -> None:
    """`$5` and `5 %` both cull to 5, and the identity keeps them apart."""

    for row in CHANGED:
        identity = _identity(row)
        assert identity.endswith("||percent"), row["value"]
        assert identity != fs.quantity_identity(str(row["value"]).strip().rstrip("%").strip())


@pytest.mark.parametrize(
    "value",
    ["Up 147%", "Up 114%", "Up 145%", "Up 45%"],
)
def test_prose_that_trails_the_store_stays_literal(value: str) -> None:
    """The five records that state a `%` and still did not canonicalise.

    They are the reason the change reaches 522 records rather than 527, and they
    are here so that number stays explained: a leading word is not a number, and
    reading one would be the widening this grammar exists to refuse.
    """

    assert fs.canonical_quantity(value) is None
    assert fs.quantity_identity(value).startswith("literal:")
