"""The percentage calculation path, replayed from the gold's own operands.

P1.4b changed the calculator's lexer.  Its evidence is a store-wide differential
over 6,527 quantities and the primitive unit tests -- both real, neither of them
a benchmark result.  This closes that gap deterministically, with no model in the
loop: take the operand values the gold names, run the calculator, and compare to
the value the gold states.

It separates two things the end-to-end runs cannot.  Where a case binds the
*facts the gold names*, the calculator is checked against the gold outright --
`pctshare-003` is exactly this, and the sealed P1.4c run corroborates it by
having produced -0.0432 from a real binding.  Where a case binds *different*
facts -- `pctshare-006` computes 0.4091 because the binder selected another
iPhone row -- the gold's own operands still replay to the gold's own answer,
which is what makes the discrepancy a binding result rather than an arithmetic
one.

Nothing here calls a model, so it cannot be flaky, and it says nothing about
whether the binder reaches these operands.  That is the separate measurement.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from src.finance.primitive_tools import percentage_share

GOLD = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/tv2_canonical_v1/gold-evidence-v1.jsonl"
)


def _share_cases() -> list[tuple[str, str, str, str]]:
    """``(case_id, part, total, expected)`` for every share the gold states."""

    cases: list[tuple[str, str, str, str]] = []
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("operation") != "percentage_share":
            continue
        operands = row.get("operands") or {}
        part, total = operands.get("part"), operands.get("total")
        expected = row.get("expected_value")
        if part is None or total is None or expected is None:
            continue
        cases.append((row["id"], str(part), str(total), str(expected)))
    return cases


SHARE_CASES = _share_cases()


def test_the_gold_states_shares_to_replay() -> None:
    """A fixture that silently emptied would make every case below vacuous."""

    assert len(SHARE_CASES) == 8


@pytest.mark.parametrize(
    "case_id,part,total,expected",
    SHARE_CASES,
    ids=[case[0] for case in SHARE_CASES],
)
def test_the_calculator_reproduces_the_gold_from_the_golds_own_operands(
    case_id: str, part: str, total: str, expected: str
) -> None:
    """`21%` of `(486)` is read as 21 over -486, which is the whole of P1.4b.

    The values come from the gold, not from a run, so this holds whether or not
    the binder reached them -- and it is the check that says the 100x is gone:
    reading the ratio form would put every one of these out by a factor of 100.
    """

    result = percentage_share(part, total, precision=4)

    assert result.ok is True, case_id
    assert result.points_value == Decimal(expected), case_id


def test_the_percentage_operand_is_read_as_written() -> None:
    """The single line that the whole lexer change exists for.

    Not inferred from the eight passing cases above -- one of them could pass by
    a compensating error.  This one states it directly.
    """

    result = percentage_share("21%", "(486)", precision=4)

    assert result.points_value == Decimal("-0.0432")
    # The ratio reading, which is what the old lexer used for a glued sign,
    # would produce this instead.
    assert result.points_value != Decimal("-0.000432")


def test_a_glued_and_a_spaced_percentage_share_are_one_operand() -> None:
    glued = percentage_share("21%", "(486)", precision=4)
    spaced = percentage_share("21 %", "(486)", precision=4)

    assert glued.points_value == spaced.points_value
    assert glued.ratio_value == spaced.ratio_value
