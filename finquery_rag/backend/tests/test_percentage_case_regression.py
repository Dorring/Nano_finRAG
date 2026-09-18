"""The benchmark's percentage cases, against the parser that has to read them.

Four cases in the canonical set involve a percentage, and all four were passing
or failing for reasons that had nothing to do with the percentage:

* ``pctshare-003`` expects ``21 / (486) = -0.0432``.  The parser read ``21%`` as
  0.21 -- because the sign was written against the digits -- and answered
  ``-0.000432``, exactly 100x out.  It was failing.
* ``average-002`` expects ``32.0000``.  Its operands are spaced (``32 %``), the
  spaced branch did not divide, and ``mean(32, 32)`` came out right by
  cancellation.  It was passing for a reason that would not have survived the
  operands being written ``32%``.
* the three ``factual_lookup`` cases state ``75.4%`` in the gold and ``75.4 %``
  in the store.  They agreed only because the scorer strips ``%`` and the parser
  happened not to divide the spaced form.

So each case is pinned against the reading it actually means -- ``points_value``,
the number as the row writes it -- and the glued/spaced pair is pinned as one
quantity, which is the property all four depend on and none of them was testing.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from src.finance.primitive_tools import (
    average_values,
    parse_financial_number,
    percentage_share,
)

_GOLD = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/tv2_canonical_v1/gold-evidence-v1.jsonl"
)

#: The cases whose gold states a percentage, with the number the gold states.
#: Authored here: reading them back out of the evaluator under test would pass
#: for any evaluator.
_PERCENT_CASES = [
    ("tv2f01-s1-aapl-003", "75.4%", Decimal("75.4")),
    ("tv2f01-s1-jpm-010", "1.83 %", Decimal("1.83")),
    ("tv2f01-s1-nvda-024", "12 %", Decimal("12")),
]


def _gold_rows() -> list[dict[str, Any]]:
    with _GOLD.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_the_three_factual_percentages_read_as_the_gold_states_them() -> None:
    for case_id, written, expected in _PERCENT_CASES:
        parsed = parse_financial_number(written)
        assert parsed.ok is True, case_id
        assert parsed.points_value == expected, case_id


def test_a_percentage_reads_the_same_written_either_way() -> None:
    """The property all four cases silently depend on.

    The gold writes ``75.4%`` and the store writes ``75.4 %``.  Under the old
    contract those were 100x apart for any value where the two sides differed in
    form -- so a case could pass or fail according to which spelling reached the
    parser.
    """
    for written in ("75.4%", "1.83 %", "12 %", "32 %", "21%"):
        glued = parse_financial_number(written.replace(" ", ""))
        spaced = parse_financial_number(written)

        assert glued.points_value == spaced.points_value, written
        assert glued.ratio_value == spaced.ratio_value, written


def test_pctshare_003_reads_the_stated_number() -> None:
    """``21%`` of ``(486)`` is ``21 / -486 = -0.0432``, which the gold states.

    Reading the ratio form answers -0.000432.  This is the case whose failure
    the two readings exist to make impossible to write by accident.
    """
    result = percentage_share("21%", "(486)", precision=4)

    assert result.ok is True
    assert result.points_value == Decimal("-0.0432")


def test_average_002_does_not_pass_by_cancellation() -> None:
    """``mean(32 %, 32 %)`` is 32 percentage points, however the operands are written.

    Both spellings must give 32.  Passing on one spelling and not the other is
    what "passing by cancellation" means, and it is what the old contract did.
    """
    spaced = average_values(["32 %", "32 %"], precision=4)
    glued = average_values(["32%", "32%"], precision=4)

    assert spaced.points_value == Decimal("32.0000")
    assert glued.points_value == spaced.points_value


def test_compare_007_stays_incomparable() -> None:
    """Pfizer ``$5`` against JPMorganChase ``5.49 %``.

    Five and 5.49 are comparable numbers and not comparable quantities: one is
    an amount, the other a rate.  The parser says which each is, so the refusal
    downstream is a type check rather than a coincidence of ordering.
    """
    amount = parse_financial_number("$5")
    rate = parse_financial_number("5.49 %")

    assert amount.representation is not rate.representation
    assert amount.ratio_value == amount.points_value == Decimal("5")
    assert rate.points_value == Decimal("5.49")
    assert rate.ratio_value == Decimal("0.0549")


@pytest.mark.parametrize("case_id", sorted(row["id"] for row in _gold_rows()))
def test_every_gold_percentage_parses_as_its_own_stated_number(case_id: str) -> None:
    """Structural sweep over the real gold file, not just the known cases.

    Every string the gold writes with a ``%`` must parse, and the number it
    parses to must be the number written -- the convention the gold's own scorer
    uses when it strips the sign.  A new percentage case added to the set is
    covered without anyone remembering to add it here.
    """
    row = next(r for r in _gold_rows() if r["id"] == case_id)
    candidates: list[str] = []
    for field in ("expected_value",):
        if isinstance(row.get(field), str):
            candidates.append(row[field])
    for field in ("values", "operands"):
        block = row.get(field)
        if isinstance(block, dict):
            candidates.extend(str(v) for v in block.values())

    for written in candidates:
        if "%" not in written:
            continue
        parsed = parse_financial_number(written)
        assert parsed.ok is True, f"{case_id}: {written!r} did not parse"
        stated = written.replace("%", "").replace(",", "").strip()
        assert parsed.points_value == Decimal(stated), f"{case_id}: {written!r}"
        assert parsed.representation.value == "percent", f"{case_id}: {written!r}"
