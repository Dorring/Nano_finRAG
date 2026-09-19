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

from src.evaluation.p1_2_dual_track import (  # noqa: E402
    _correct_by_stratum,
    _predicted_ordering,
)


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
        _row(answer="Apple > Microsoft"), {"expected_higher": "Visa"}
    ) is False


def test_a_comparison_naming_both_entities_passes_whatever_the_order() -> None:
    """A known limitation, pinned so it is not mistaken for a guarantee.

    The check is a substring test, so an answer naming both sides scores correct
    even when it asserts the wrong order.  `_correct_by_stratum` cannot do
    better without parsing the answer, and the release path no longer depends on
    it: a relational answer is now the deterministic rendering, so the order is
    stated by the renderer rather than by prose this check has to read.

    Asserted rather than fixed.  Changing it belongs with the other prose-scoring
    questions, not with a fix for a gold that was never consulted.
    """

    assert _correct_by_stratum(
        _row(answer="Apple > Visa"), {"expected_higher": "Visa"}
    ) is True


def test_a_ranking_still_scores_on_named_entities() -> None:
    gold = {"expected_ranking": ["Microsoft", "Tesla", "Apple"]}

    assert _correct_by_stratum(
        _row(answer="Microsoft > Tesla > Apple"), gold
    ) is True
    assert _correct_by_stratum(_row(answer="Microsoft > Apple"), gold) is False


# --- a ranking is an order, and ordering was never checked ---------------------


def test_a_ranking_in_the_wrong_order_is_not_correct() -> None:
    """`rank-005`: three names present, the order entirely wrong.

    The check used to be membership, so this scored correct and a real wrong
    release was counted as a correct one -- in the one stratum whose whole point
    is order.
    """

    gold = {"expected_ranking": ["Apple", "The Coca-Cola Company", "Microsoft"]}

    assert _correct_by_stratum(
        _row(answer="Apple > Microsoft > The Coca-Cola Company"), gold
    ) is False


def test_a_ranking_in_the_right_order_is_correct() -> None:
    gold = {"expected_ranking": ["Apple", "The Coca-Cola Company", "Microsoft"]}

    assert _correct_by_stratum(
        _row(answer="Apple > The Coca-Cola Company > Microsoft"), gold
    ) is True


@pytest.mark.parametrize(
    ("answer", "why"),
    [
        ("Apple > Microsoft", "a member is missing"),
        ("Apple > Microsoft > Coca-Cola > Tesla", "there is an extra member"),
        ("Apple", "there is no ordering at all"),
        ("", "nothing was answered"),
        ("The calculation could not be completed.", "prose, not a rendered order"),
    ],
    ids=["missing-member", "extra-member", "no-ordering", "empty", "prose"],
)
def test_a_ranking_that_does_not_state_the_order_is_not_correct(
    answer: str, why: str
) -> None:
    gold = {"expected_ranking": ["Apple", "The Coca-Cola Company", "Microsoft"]}

    assert _correct_by_stratum(_row(answer=answer), gold) is False, why


def test_a_tie_prediction_is_not_scored_correct() -> None:
    """`expected_ranking` cannot express a tie, so it is not guessed at.

    Scoring it either way would be inventing a gold the benchmark does not
    state.  Teaching the gold about ties is a benchmark change and is not made
    here; until then a tie prediction is incorrect rather than ambiguous.
    """

    gold = {"expected_ranking": ["Apple", "Microsoft"]}

    assert _correct_by_stratum(_row(answer="Apple = Microsoft"), gold) is False


def test_the_order_check_is_not_case_sensitive() -> None:
    gold = {"expected_ranking": ["Microsoft", "Tesla", "Apple"]}

    assert _correct_by_stratum(
        _row(answer="microsoft > tesla > apple"), gold
    ) is True


def test_a_rendered_ranking_parses_into_groups() -> None:
    """The parser reads the renderer's frozen format and nothing else."""

    assert _predicted_ordering("Apple > Microsoft") == (("Apple",), ("Microsoft",))
    assert _predicted_ordering("Apple = Microsoft > Tesla") == (
        ("Apple", "Microsoft"),
        ("Tesla",),
    )
    assert _predicted_ordering("just some prose") is None
    assert _predicted_ordering("") is None
