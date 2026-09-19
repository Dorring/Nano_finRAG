"""A cross-entity difference is scored, and was not.

`crossdiff-001` computed the right answer and released it deterministically:

    Difference: -120.00
    Formula: current - previous (difference.v1)

and the evaluator scored it a **false release**.  The `cross_entity_comparison`
branch of `_correct_by_stratum` consulted `expected_higher` and
`expected_ranking` and stopped:

    ranking = gold.get("expected_ranking") or ()
    return bool(ranking) and all(...)

A gold carrying `expected_value` and neither of those -- which is all five
`cross_entity_difference` cases -- fell through to `bool(())`, which is `False`
whatever the answer said.  So the stratum's one correct numeric release was
counted against it, and `false_release = 0` was unreachable by construction.

Same defect family as the circular release verdict removed from the canonical
scorer: an evaluator that cannot see the gold is not measuring the system.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from src.evaluation.p1_2_dual_track import _correct_by_stratum  # noqa: E402


def _row(**overrides):
    row = {
        "id": "c1",
        "stratum": "cross_entity_comparison",
        "answer": "",
        "calculations": [],
    }
    row.update(overrides)
    return row


def test_a_correct_difference_is_scored_correct() -> None:
    """The released form the deterministic renderer produces."""

    row = _row(answer="Difference: -120.00\n\nFormula: current - previous (difference.v1)")

    assert _correct_by_stratum(row, {"expected_value": -120, "tolerance": 0.0001}) is True


def test_a_wrong_difference_is_scored_wrong() -> None:
    """The fix must not make every difference correct."""

    row = _row(answer="Difference: 120.00")

    assert _correct_by_stratum(row, {"expected_value": -120, "tolerance": 0.0001}) is False


def test_a_difference_gold_with_no_answer_is_not_correct() -> None:
    assert _correct_by_stratum(_row(answer=""), {"expected_value": -120}) is False


def test_the_structured_calculation_also_scores() -> None:
    """A terse answer still scores when the calculation carries the value."""

    row = _row(answer="See the calculation.", calculations=[{"value": "-120.0000"}])

    assert _correct_by_stratum(row, {"expected_value": -120, "tolerance": 0.0001}) is True


# --- and the other two kinds are untouched ------------------------------------


def test_a_comparison_still_scores_on_the_higher_entity() -> None:
    assert _correct_by_stratum(
        _row(answer="Visa > Apple"), {"expected_higher": "Visa"}
    ) is True
    assert _correct_by_stratum(
        _row(answer="Apple > Visa"), {"expected_higher": "Visa"}
    ) is False


def test_a_ranking_still_scores_on_named_entities() -> None:
    gold = {"expected_ranking": ["Microsoft", "Tesla", "Apple"]}

    assert _correct_by_stratum(
        _row(answer="Microsoft > Tesla > Apple"), gold
    ) is True
    assert _correct_by_stratum(_row(answer="Microsoft > Apple"), gold) is False
