"""The canonical scorer must not let a release be its own evidence of correctness.

`score_predictions` had this as the last branch of its `cross_entity_comparison`
arm:

    elif pred_released:
        decision_correct = True

So any release scored as a correct decision, whatever the answer said.  The
evaluator endorsed the system with the system's own release verdict: no
cross-entity release could be counted wrong, and `decision_accuracy` in that
report was bounded below by the release rate rather than by anything the answer
stated.

It is the same shape as the circularity this codebase guards against elsewhere
-- expected values must never come from the code that produces the actual ones.
Here the *verdict* came from the system being judged.

The fallback was not covering an otherwise-unscorable case: all twenty
cross-entity golds in the canonical set carry exactly one of `expected_higher`,
`expected_ranking` or `expected_value`, which is asserted here so that the
precondition for removing it cannot quietly stop holding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = BACKEND_ROOT / "scripts" / "evaluation"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_tv2_canonical_benchmark import score_predictions  # noqa: E402

GOLD_PATH = BACKEND_ROOT / "benchmarks" / "tv2_canonical_v1" / "gold-evidence-v1.jsonl"

#: The expectation fields a cross-entity gold may carry.  Exactly one is present
#: per gold; that is what makes the answer scoreable without asking the runtime.
_EXPECTATION_FIELDS = ("expected_higher", "expected_ranking", "expected_value")


def _prediction(
    *,
    answer: str,
    released: bool = True,
    stratum: str = "cross_entity_comparison",
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": "case-1",
        "stratum": stratum,
        "answer": answer,
        "evidence_ids": evidence_ids or [],
        "release_status": "RELEASED" if released else "NOT_RELEASED",
        "reason_codes": [],
    }


def _gold(**kwargs: Any) -> dict[str, dict[str, Any]]:
    base: dict[str, Any] = {"id": "case-1", "fact_ids": []}
    base.update(kwargs)
    return {"case-1": base}


def _decision_accuracy(predictions: list[dict[str, Any]], gold: dict[str, Any]) -> float:
    """The cross-entity decision accuracy the scorer actually reports.

    `score_predictions` returns ``(overall_metrics, stratum_breakdown)`` -- its
    internal ``results_by_id`` is built and then not returned, so the per-case
    verdict is only observable through this aggregate.  That is why the
    assertions here are on a one-prediction population: a rate over one case is
    that case's verdict.
    """

    _overall, breakdown = score_predictions(predictions, gold)
    return breakdown["stratum_3_cross_entity_comparison"]["decision_accuracy"]


def test_a_release_with_a_wrong_answer_is_not_a_correct_decision() -> None:
    """The counter-example.

    ``compare-002`` is the live case that motivated this: `expected_higher` is
    Visa and the released answer states Apple's figure.  Before the fix this
    scored as a correct decision purely because it had been released.
    """

    accuracy = _decision_accuracy(
        [_prediction(answer="Apple reported a larger figure, 8,077.")],
        _gold(expected_higher="Visa"),
    )

    assert accuracy == 0.0


def test_a_release_that_states_the_gold_is_still_correct() -> None:
    """The fix must not make every release wrong -- only unscored ones."""

    accuracy = _decision_accuracy(
        [_prediction(answer="Visa reported the larger figure, 1,926.")],
        _gold(expected_higher="Visa"),
    )

    assert accuracy == 1.0


@pytest.mark.parametrize(
    ("gold_kwargs", "answer"),
    [
        ({"expected_ranking": ["Apple", "Microsoft", "Tesla"]}, "Apple, Microsoft, Tesla"),
        ({"expected_value": "1,926"}, "the figure is 1,926"),
    ],
    ids=["ranking", "value"],
)
def test_the_other_two_expectation_kinds_are_scored_on_their_own_terms(
    gold_kwargs: dict[str, Any], answer: str
) -> None:
    """`expected_ranking` and `expected_value` must carry correctness alone."""

    wrong = _decision_accuracy(
        [_prediction(answer="Something else entirely.")], _gold(**gold_kwargs)
    )
    assert wrong == 0.0

    right = _decision_accuracy([_prediction(answer=answer)], _gold(**gold_kwargs))
    assert right == 1.0


def test_a_non_release_is_unaffected() -> None:
    """A fail-closed case was never correct here, and must not become so."""

    accuracy = _decision_accuracy(
        [_prediction(answer="", released=False)],
        _gold(expected_higher="Visa"),
    )

    assert accuracy == 0.0


def test_the_fixture_is_not_vacuous() -> None:
    """Guards every assertion above: a stratum that scored nothing reports 0
    accuracy for the same reason an unanswerable case does."""

    _overall, breakdown = score_predictions(
        [_prediction(answer="Visa reported the larger figure, 1,926.")],
        _gold(expected_higher="Visa"),
    )

    assert breakdown["stratum_3_cross_entity_comparison"]["total"] == 1


def test_every_cross_entity_gold_carries_an_expectation() -> None:
    """The precondition for removing the fallback, asserted rather than assumed.

    If a cross-entity gold ever arrived with none of the three fields, its
    `decision_correct` would be permanently False -- a silent undercount rather
    than a visible error.  This is the guard that would catch that.
    """

    golds = [
        json.loads(line)
        for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cross_entity = [g for g in golds if g["id"].startswith("tv2f01-s3-")]

    assert cross_entity, "no cross-entity golds found; the guard would be vacuous"
    for gold in cross_entity:
        present = [name for name in _EXPECTATION_FIELDS if gold.get(name)]
        assert len(present) == 1, f"{gold['id']} carries {present}"
