"""A fixture may serialise ground truth.  It may not create it.

The first version of the fixture builder read the period for ``sum`` and
``average`` out of the question text -- using the *first* period it found, for
every operand.  Both slots then named the same quantity, the binder refused the
duplicate (correctly), and ten benchmark questions scored as binder failures for
what was a builder defect.  The failure was invisible because the slot count was
right and only the *values* were wrong.

Two things are pinned here.  The reader that replaced it fails loudly instead of
falling back, because a fallback is how the defect got in.  And the frozen
fixture is checked for distinguishable slots, with the archived defective
fixture checked to still *fail* that check -- a test that passes on both would
not be testing anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.build_p1_2_plan_fixtures import (
    _operand_periods_from_gold,
    load_fact_coordinates,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "artifacts/evaluation/p1-2-plan-fixtures"
)
FIXTURE_V1 = FIXTURE_DIR / "plan-fixtures-v1.jsonl"
FIXTURE_V2 = FIXTURE_DIR / "plan-fixtures-v2.jsonl"

#: The operations whose operands are one gold fact each.  Their periods come
#: from those facts; a collision here means the fixture asks for one quantity
#: twice.
MULTI_PERIOD_OPERATIONS = {"sum", "average"}

_ANSWER_MATERIAL = ("fact_ids", "expected_value", "operands", "values", "expected_higher")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _colliding_multi_period_cases(rows: list[dict]) -> list[str]:
    """Cases whose multi-period slots cannot be told apart."""

    colliding: list[str] = []
    for row in rows:
        plan = row["plan"]
        if str(plan.get("operation") or "") not in MULTI_PERIOD_OPERATIONS:
            continue
        signature = {
            (slot["role"], slot["metric"], slot["period"])
            for slot in plan["required_slots"]
        }
        if len(signature) != len(plan["required_slots"]):
            colliding.append(row["id"])
    return colliding


# --- the reader that replaced the fallback -----------------------------------------------


def _gold(fact_ids: list[str]) -> dict:
    return {"fact_ids": fact_ids, "operation": "sum"}


def _fact(period: str) -> dict:
    return {"period": period, "metric": "Cost of sales", "entity": "Apple"}


def test_operand_periods_come_from_the_facts_in_gold_order() -> None:
    periods = _operand_periods_from_gold(
        _gold(["a", "b"]),
        {"id": "q"},
        {"a": _fact("FY2024"), "b": _fact("FY2025")},
        arity=2,
        operation="sum",
    )

    assert periods == ["FY2024", "FY2025"]


def test_two_operands_at_one_period_are_refused() -> None:
    """The exact defect: the same requirement written twice."""

    with pytest.raises(ValueError, match="not distinct on period"):
        _operand_periods_from_gold(
            _gold(["a", "b"]),
            {"id": "q"},
            {"a": _fact("FY2024"), "b": _fact("FY2024")},
            arity=2,
            operation="sum",
        )


def test_a_missing_operand_fact_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="is not in the fact store"):
        _operand_periods_from_gold(
            _gold(["a", "b"]),
            {"id": "q"},
            {"a": _fact("FY2024")},
            arity=2,
            operation="sum",
        )


def test_a_gold_that_names_the_wrong_number_of_operands_is_refused() -> None:
    with pytest.raises(ValueError, match="needs 2 operand facts, gold names 1"):
        _operand_periods_from_gold(
            _gold(["a"]),
            {"id": "q"},
            {"a": _fact("FY2024")},
            arity=2,
            operation="sum",
        )


def test_an_operand_fact_without_a_period_is_refused() -> None:
    with pytest.raises(ValueError, match="states no period"):
        _operand_periods_from_gold(
            _gold(["a", "b"]),
            {"id": "q"},
            {"a": {"metric": "Cost of sales"}, "b": _fact("FY2025")},
            arity=2,
            operation="sum",
        )


def test_the_error_names_the_case_it_could_not_author() -> None:
    """A build failure has to say which question it could not author."""

    with pytest.raises(ValueError, match="tv2f01-s2-sum-003"):
        _operand_periods_from_gold(
            _gold(["a", "b"]),
            {"id": "tv2f01-s2-sum-003"},
            {"a": _fact("FY2024"), "b": _fact("FY2024")},
            arity=2,
            operation="sum",
        )


# --- reading the store -------------------------------------------------------------------


def test_coordinates_are_read_by_the_key_the_gold_uses(tmp_path: Path) -> None:
    store = tmp_path / "store.jsonl"
    store.write_text(
        json.dumps({"candidate_key": "v2fact:a", "period": "FY2024"})
        + "\n"
        + json.dumps({"candidate_key": "v2fact:unwanted", "period": "FY1999"})
        + "\n",
        encoding="utf-8",
    )

    found = load_fact_coordinates(store, ["v2fact:a"])

    assert set(found) == {"v2fact:a"}
    assert found["v2fact:a"]["period"] == "FY2024"


def test_the_first_record_for_an_id_wins(tmp_path: Path) -> None:
    """A candidate can have several extraction rows; they agree on the period."""

    store = tmp_path / "store.jsonl"
    store.write_text(
        json.dumps({"candidate_key": "v2fact:a", "period": "FY2024", "row_id": "r1"})
        + "\n"
        + json.dumps({"candidate_key": "v2fact:a", "period": "FY2024", "row_id": "r2"})
        + "\n",
        encoding="utf-8",
    )

    found = load_fact_coordinates(store, ["v2fact:a"])

    assert found["v2fact:a"]["row_id"] == "r1"


def test_reading_no_ids_does_not_open_the_store(tmp_path: Path) -> None:
    """A corpus with no multi-period operation must not require a store."""

    assert load_fact_coordinates(tmp_path / "does-not-exist.jsonl", []) == {}


# --- the frozen fixtures -----------------------------------------------------------------


@pytest.mark.skipif(not FIXTURE_V2.exists(), reason="the v2 fixture is not present")
def test_the_frozen_fixture_has_no_colliding_slots() -> None:
    assert _colliding_multi_period_cases(_read_jsonl(FIXTURE_V2)) == []


@pytest.mark.skipif(not FIXTURE_V1.exists(), reason="the v1 fixture is not present")
def test_the_check_finds_the_defect_it_was_written_for() -> None:
    """The archived fixture must still fail this, or the check proves nothing.

    v1 is kept rather than corrected so the two can be compared; a detector that
    passed on both would be measuring nothing.
    """

    colliding = _colliding_multi_period_cases(_read_jsonl(FIXTURE_V1))

    assert len(colliding) == 10
    assert "tv2f01-s2-sum-003" in colliding


@pytest.mark.skipif(not FIXTURE_V2.exists(), reason="the v2 fixture is not present")
def test_the_frozen_fixture_still_carries_no_answer_material() -> None:
    """Checked against the frozen bytes, so it holds without the fact store."""

    rendered = FIXTURE_V2.read_text(encoding="utf-8")

    for key in _ANSWER_MATERIAL:
        assert key not in rendered, f"{key} leaked into a fixture"
