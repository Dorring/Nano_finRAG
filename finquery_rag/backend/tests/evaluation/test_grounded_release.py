"""A correct answer from an unauthorised operand is not a trusted release.

`rank-002` released `Microsoft > Tesla > Apple` against a gold of
`[Microsoft, Tesla, Apple]`.  The ordering is right and the release is not: its
Tesla operand was `1,332` where the gold fact is `6,411`, and the order survived
only because `32488 > 1332 > -34550` has the same shape as
`32488 > 6411 > -34550`.

Every scorer in the repo reads the rendered relation and nothing else, so that
release was counted a success.  The defect is not that a wrong operand slipped
through -- the executor did exactly what it was given -- but that the *scorer
could not tell the difference* between a grounded release and one that got the
right shape by accident.

These tests pin the three claims apart, and pin the attributions that follow
from them: `rank-002` is UNGROUNDED, `compare-009` is GOLD_UNGRADEABLE rather
than wrong, and `crossdiff-001` -- grounded operands, valid gold -- is the one
that stays TRUSTED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from src.evaluation.grounded_release import (  # noqa: E402
    GoldValidity,
    ReleaseVerdict,
    executed_operands,
    gold_operand_values,
    load_gold_validity,
    operand_grounded_correct,
    release_verdict,
)
from src.evaluation.p1_2_dual_track import score_downstream  # noqa: E402


def _row(**overrides):
    """A released cross-entity row whose operands are the gold's, by default."""

    row = {
        "id": "c1",
        "stratum": "cross_entity_comparison",
        "answer": "A > B",
        "release_status": "RELEASED",
        "calculations": [{"operands": [{"value": "1"}, {"value": "2"}]}],
        "reason_codes": ["VALIDATED_RELEASE"],
    }
    row.update(overrides)
    return row


def _gold(**overrides):
    gold = {
        "expected_outcome": "ANSWER",
        "expected_ranking": ["A", "B"],
        "values": {"A": "1", "B": "2"},
    }
    gold.update(overrides)
    return gold


# --- the shape of the gold ----------------------------------------------------


def test_a_cross_entity_gold_authorises_its_values_by_entity() -> None:
    gold = _gold(values={"Apple": "(34,550)", "Tesla": "$ 6,411"})
    assert set(gold_operand_values(gold)) == {"(34,550)", "$ 6,411"}


def test_an_arithmetic_gold_authorises_its_operands_by_role() -> None:
    gold = {"operands": {"current": "325", "previous": "248"}}
    assert set(gold_operand_values(gold)) == {"325", "248"}


def test_a_gold_stating_no_operands_authorises_nothing() -> None:
    assert gold_operand_values({"expected_value": 604}) is None


# --- grounding ----------------------------------------------------------------


def test_operands_matching_the_gold_are_grounded() -> None:
    row = _row(calculations=[{"operands": [{"value": "1083"}, {"value": "1203"}]}])
    gold = _gold(values={"Tesla": "1,083", "NVIDIA": "1,203"})
    assert operand_grounded_correct(row, gold) is True


def test_one_operand_the_gold_does_not_authorise_is_not_grounded() -> None:
    """`rank-002`: the other two operands are exact and that does not help."""

    row = _row(
        calculations=[
            {"operands": [{"value": "-34550"}, {"value": "32488"}, {"value": "1332"}]}
        ]
    )
    gold = _gold(
        values={"Apple": "(34,550)", "Microsoft": "32,488", "Tesla": "$ 6,411"}
    )
    assert operand_grounded_correct(row, gold) is False


def test_a_repeated_operand_cannot_consume_one_gold_value_twice() -> None:
    """Matching is a multiset claim, so `2, 2` cannot be grounded by one `2`."""

    row = _row(calculations=[{"operands": [{"value": "5"}, {"value": "5"}]}])
    gold = _gold(values={"A": "5", "B": "7"})
    assert operand_grounded_correct(row, gold) is False


def test_a_row_with_no_deterministic_execution_is_unjudged() -> None:
    """Not `False`: a prose answer makes a different claim, not a false one."""

    assert operand_grounded_correct(_row(calculations=[]), _gold()) is None


def test_a_gold_without_operand_values_leaves_grounding_unjudged() -> None:
    row = _row()
    assert operand_grounded_correct(row, {"expected_value": 3}) is None


def test_grounding_honours_the_gold_tolerance() -> None:
    row = _row(calculations=[{"operands": [{"value": "1.02"}]}])
    gold = _gold(values={"A": "1.0"}, tolerance="0.05")
    assert operand_grounded_correct(row, gold) is True


def test_operands_are_compared_in_display_units_not_as_ratios() -> None:
    """A gold record carries two conventions at once, and operands use the other.

    `pctshare-003`'s operands are `21` and `-486` and its gold states `21 %` and
    `(486)` -- they agree exactly.  Reading the operands with the ratio
    convention, which is correct for `expected_value` and wrong here, turns the
    gold's `21 %` into `0.21` and reports a correct release as ungrounded.

    A grounding check that fails correct releases is worse than no check,
    because it is believed.
    """

    row = _row(
        stratum="arithmetic_calculation",
        calculations=[{"operands": [{"value": "21"}, {"value": "-486"}]}],
    )
    gold = _gold(values=None, operands={"part": "21 %", "total": "(486)"})
    assert operand_grounded_correct(row, gold) is True


def test_the_display_unit_convention_cannot_reach_expected_value() -> None:
    """`expected_value` is a computed ratio; grounding must only read operands.

    The two conventions coexist in one gold record, so the boundary between them
    is worth pinning: `gold_operand_values` returns `values`/`operands` and never
    `expected_value`, which is what keeps the display-unit fix from leaking into
    the correctness scorer's ratio handling.
    """

    gold = _gold(
        values=None,
        expected_value=0.3105,
        operands={"current": "325", "previous": "248"},
    )
    assert gold_operand_values(gold) == ("325", "248")


def test_operands_are_read_from_the_calculations_not_the_answer() -> None:
    """A prose answer that happens to name the gold value grounds nothing."""

    row = _row(answer="1083 and 1203", calculations=[])
    assert executed_operands(row) == ()
    assert operand_grounded_correct(row, _gold()) is None


# --- the verdict --------------------------------------------------------------


def test_a_grounded_correct_release_is_trusted() -> None:
    row = _row(calculations=[{"operands": [{"value": "1"}, {"value": "2"}]}])
    verdict = release_verdict(
        row, _gold(), decision_correct=True, gold_validity=GoldValidity.VALID
    )
    assert verdict is ReleaseVerdict.TRUSTED


def test_a_correct_relation_from_a_wrong_operand_is_ungrounded() -> None:
    """`rank-002`: SCORED correct, and still not a trusted release."""

    row = _row(
        answer="Microsoft > Tesla > Apple",
        release_status="RELEASED",
        calculations=[
            {"operands": [{"value": "-34550"}, {"value": "32488"}, {"value": "1332"}]}
        ],
    )
    gold = _gold(
        expected_ranking=["Microsoft", "Tesla", "Apple"],
        values={"Apple": "(34,550)", "Microsoft": "32,488", "Tesla": "$ 6,411"},
    )
    assert release_verdict(row, gold, decision_correct=True) is ReleaseVerdict.UNGROUNDED


@pytest.mark.parametrize(
    "validity", [GoldValidity.WRONG_SCOPE, GoldValidity.UNRESOLVED]
)
def test_a_discredited_gold_cannot_certify_a_release(validity) -> None:
    """`compare-009`: the operands are right and the gold is not, so the case
    says nothing about the system -- and must not be reported as a success."""

    verdict = release_verdict(_row(), _gold(), decision_correct=True, gold_validity=validity)
    assert verdict is ReleaseVerdict.GOLD_UNGRADEABLE


def test_a_released_wrong_answer_is_wrong_whatever_the_gold() -> None:
    row = _row()
    assert release_verdict(row, _gold(), decision_correct=False) is ReleaseVerdict.WRONG


def test_ungroundedness_outranks_a_discredited_gold() -> None:
    """The two findings are on opposite sides of the system/benchmark line.

    `rank-002` used an operand matching no gold value under any reading, so it
    is ungrounded whatever the audit later decides the gold should have been.
    Filing it under the benchmark's problem because the benchmark is *also*
    imperfect would lose a real finding about the system.
    """

    row = _row(calculations=[{"operands": [{"value": "1332"}]}])
    gold = _gold(values={"Tesla": "$ 6,411"})
    verdict = release_verdict(
        row, gold, decision_correct=True, gold_validity=GoldValidity.UNRESOLVED
    )
    assert verdict is ReleaseVerdict.UNGROUNDED


def test_a_grounded_release_under_a_discredited_gold_is_still_ungradeable() -> None:
    """`compare-009` the other way round: the operand matches and the gold is
    the thing that is wrong, so the case certifies nothing either way."""

    row = _row(calculations=[{"operands": [{"value": "4520"}]}])
    gold = _gold(values={"JPMorganChase": "$ 4,520"})
    verdict = release_verdict(
        row, gold, decision_correct=True, gold_validity=GoldValidity.WRONG_SCOPE
    )
    assert verdict is ReleaseVerdict.GOLD_UNGRADEABLE


def test_an_unreleased_row_makes_no_release_claim() -> None:
    row = _row(release_status="FAIL_CLOSED", answer="")
    assert (
        release_verdict(row, _gold(), decision_correct=True)
        is ReleaseVerdict.NOT_RELEASED
    )


def test_an_unaudited_gold_does_not_by_itself_block_trust() -> None:
    """The audit covers one stratum.  Silence about the rest is not a finding."""

    verdict = release_verdict(
        _row(), _gold(), decision_correct=True, gold_validity=GoldValidity.UNAUDITED
    )
    assert verdict is ReleaseVerdict.TRUSTED


# --- the artifact -------------------------------------------------------------


def test_gold_validity_is_loaded_from_the_audit_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "cross-entity-gold-validity.json"
    artifact.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "a", "case_classification": "VALID_GOLD"},
                    {"case_id": "b", "case_classification": "WRONG_SCOPE_GOLD"},
                    {"case_id": "c", "case_classification": "UNRESOLVED_GROUND_TRUTH"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_gold_validity(artifact) == {
        "a": GoldValidity.VALID,
        "b": GoldValidity.WRONG_SCOPE,
        "c": GoldValidity.UNRESOLVED,
    }


def test_a_missing_artifact_leaves_every_case_unaudited(tmp_path: Path) -> None:
    """A scorer that cannot find the audit must not promote its cases to valid."""

    assert load_gold_validity(tmp_path / "absent.json") == {}


def test_an_unrecognised_classification_is_not_treated_as_valid(tmp_path: Path) -> None:
    artifact = tmp_path / "a.json"
    artifact.write_text(
        json.dumps({"cases": [{"case_id": "a", "case_classification": "WHATEVER"}]}),
        encoding="utf-8",
    )
    assert load_gold_validity(artifact) == {}


# --- the counters -------------------------------------------------------------


def test_score_downstream_separates_trusted_from_ungrounded_release() -> None:
    """The whole point: `rank-002` must not land in `released_correct` alone.

    Both rows release a correct answer.  Only one of them used the gold's
    operands, and the counts have to say so -- otherwise the aggregate reports
    a coincidence as a success.
    """

    grounded = _row(
        id="grounded",
        calculations=[{"operands": [{"value": "1"}, {"value": "2"}]}],
    )
    ungrounded = _row(
        id="ungrounded",
        calculations=[{"operands": [{"value": "1"}, {"value": "99"}]}],
    )
    gold = {
        "grounded": _gold(),
        "ungrounded": _gold(values={"A": "1", "B": "2"}),
    }
    result = score_downstream([grounded, ungrounded], gold)
    counts = result["counts"]
    assert counts["released_correct"] == 2
    assert counts["trusted_release"] == 1
    assert counts["ungrounded_release"] == 1
    assert result["trusted_release_rate"] == 0.5


def test_score_downstream_reports_a_discredited_gold_as_ungradeable() -> None:
    row = _row(id="c9")
    gold = {"c9": _gold()}
    result = score_downstream(
        [row], gold, gold_validity={"c9": GoldValidity.WRONG_SCOPE}
    )
    counts = result["counts"]
    assert counts["gold_ungradeable_release"] == 1
    assert counts["trusted_release"] == 0
    assert counts["ungrounded_release"] == 0


def test_score_downstream_without_an_audit_still_counts_grounding() -> None:
    """Omitting the audit must not silently disable the grounding check."""

    row = _row(calculations=[{"operands": [{"value": "nonsense"}]}])
    result = score_downstream([row], {"c1": _gold()})
    assert result["counts"]["ungrounded_release"] == 1
    assert result["counts"]["trusted_release"] == 0
