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
    _operand_coordinates_from_gold,
    load_fact_coordinates,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "benchmarks/tv2_canonical_v1"
)
FIXTURE_V1 = FIXTURE_DIR / "plan-fixtures-v1.jsonl"
FIXTURE_V3 = FIXTURE_DIR / "plan-fixtures-v3.jsonl"

_ANSWER_MATERIAL = ("fact_ids", "expected_value", "operands", "values", "expected_higher")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _colliding_cases(rows: list[dict]) -> list[str]:
    """Cases whose slots cannot be told apart on the whole coordinate.

    The signature includes ``entity``: the contract lets a slot say whose fact
    it needs, so two slots naming two companies are two requirements however
    identical the rest of them is.  Reading it without entity would report the
    repaired comparison fixtures as still colliding.
    """

    colliding: list[str] = []
    for row in rows:
        slots = row["plan"]["required_slots"]
        signature = {
            (
                slot["role"],
                slot["metric"],
                slot["period"],
                slot.get("unit"),
                slot.get("entity"),
            )
            for slot in slots
        }
        if len(signature) != len(slots):
            colliding.append(row["id"])
    return colliding


# --- the reader that replaced the fallback -----------------------------------------------


def _gold(fact_ids: list[str]) -> dict:
    return {"fact_ids": fact_ids, "operation": "sum"}


def _fact(period: str, metric: str = "Cost of sales") -> dict:
    return {"period": period, "metric": metric, "entity": "Apple"}


def test_operand_coordinates_come_from_the_facts_in_gold_order() -> None:
    coordinates = _operand_coordinates_from_gold(
        _gold(["a", "b"]),
        {"id": "q"},
        {"a": _fact("FY2024"), "b": _fact("FY2025")},
        arity=2,
        operation="sum",
    )

    assert coordinates == [
        ("Cost of sales", "FY2024"),
        ("Cost of sales", "FY2025"),
    ]


def test_two_metrics_at_one_period_are_two_requirements() -> None:
    """The ``percentage_share`` shape, which the period-only check refused.

    ``sum`` and ``average`` add one metric across two periods; a share divides
    two metrics at one period.  Both are two operands.  A rule written for the
    first shape read the second as one requirement written twice, so every
    ``percentage_share`` case was unbuildable and the operation had no reachable
    coverage at all.
    """

    coordinates = _operand_coordinates_from_gold(
        _gold(["a", "b"]),
        {"id": "tv2f01-s2-pctshare-003"},
        {
            "a": _fact("FY2025", "Statutory federal income tax rate"),
            "b": _fact("FY2025", "Impact of the State Aid Decision"),
        },
        arity=2,
        operation="percentage_share",
    )

    assert coordinates == [
        ("Statutory federal income tax rate", "FY2025"),
        ("Impact of the State Aid Decision", "FY2025"),
    ]


def test_one_metric_at_one_period_is_still_refused() -> None:
    """The defect the check was written for, still caught after generalising."""

    with pytest.raises(ValueError, match=r"not distinct on \(metric, period\)"):
        _operand_coordinates_from_gold(
            _gold(["a", "b"]),
            {"id": "q"},
            {"a": _fact("FY2024"), "b": _fact("FY2024")},
            arity=2,
            operation="sum",
        )


def test_an_operand_fact_without_a_metric_is_refused() -> None:
    """The corpus must supply the coordinate, not the question."""

    with pytest.raises(ValueError, match="states no metric"):
        _operand_coordinates_from_gold(
            _gold(["a", "b"]),
            {"id": "q"},
            {"a": {"period": "FY2024"}, "b": _fact("FY2025")},
            arity=2,
            operation="percentage_share",
        )


def test_a_missing_operand_fact_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="is not in the fact store"):
        _operand_coordinates_from_gold(
            _gold(["a", "b"]),
            {"id": "q"},
            {"a": _fact("FY2024")},
            arity=2,
            operation="sum",
        )


def test_a_gold_that_names_the_wrong_number_of_operands_is_refused() -> None:
    with pytest.raises(ValueError, match="needs 2 operand facts, gold names 1"):
        _operand_coordinates_from_gold(
            _gold(["a"]),
            {"id": "q"},
            {"a": _fact("FY2024")},
            arity=2,
            operation="sum",
        )


def test_an_operand_fact_without_a_period_is_refused() -> None:
    with pytest.raises(ValueError, match="states no period"):
        _operand_coordinates_from_gold(
            _gold(["a", "b"]),
            {"id": "q"},
            {"a": {"metric": "Cost of sales"}, "b": _fact("FY2025")},
            arity=2,
            operation="sum",
        )


def test_the_error_names_the_case_it_could_not_author() -> None:
    """A build failure has to say which question it could not author."""

    with pytest.raises(ValueError, match="tv2f01-s2-sum-003"):
        _operand_coordinates_from_gold(
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


@pytest.mark.skipif(not FIXTURE_V3.exists(), reason="the v3 fixture is not present")
def test_the_frozen_fixture_has_no_colliding_slots() -> None:
    assert _colliding_cases(_read_jsonl(FIXTURE_V3)) == []


@pytest.mark.skipif(not FIXTURE_V1.exists(), reason="the v1 fixture is not present")
def test_the_check_finds_the_defects_it_was_written_for() -> None:
    """The archived fixture must still fail this, or the check proves nothing.

    v1 is kept rather than corrected so the two can be compared; a detector that
    passed on both would be measuring nothing.  Both defect classes are named
    here, so a check that caught only one of them would fail rather than pass.
    """

    colliding = _colliding_cases(_read_jsonl(FIXTURE_V1))

    assert len(colliding) == 30
    assert "tv2f01-s2-sum-003" in colliding  # the duplicated operand period
    assert "tv2f01-s3-compare-002" in colliding  # the indistinguishable entities


@pytest.mark.skipif(not FIXTURE_V3.exists(), reason="the v3 fixture is not present")
def test_a_comparison_is_repaired_by_entity_and_not_by_duplication() -> None:
    rows = {row["id"]: row for row in _read_jsonl(FIXTURE_V3)}

    slots = rows["tv2f01-s3-compare-002"]["plan"]["required_slots"]

    assert [slot.get("entity") for slot in slots] == ["Apple", "Visa"]
    assert len({slot["metric"] for slot in slots}) == 1


@pytest.mark.skipif(not FIXTURE_V3.exists(), reason="the v3 fixture is not present")
def test_a_ranking_plan_asks_for_every_company_it_ranks() -> None:
    """The arity is the gold's, not the literal two it used to be."""

    rows = {row["id"]: row for row in _read_jsonl(FIXTURE_V3)}

    slots = rows["tv2f01-s3-rank-003"]["plan"]["required_slots"]

    assert len(slots) == 4
    assert len({slot.get("entity") for slot in slots}) == 4


@pytest.mark.skipif(not FIXTURE_V3.exists(), reason="the v3 fixture is not present")
def test_the_fixture_carries_a_mention_and_never_an_identity() -> None:
    """`entity_id` is the Harness's to derive.  A fixture writing one would be
    authoring an identity rather than serialising a fact -- the mistake that
    produced the operand periods this builder was fixed for once already."""

    rendered = FIXTURE_V3.read_text(encoding="utf-8")

    assert "entity_id" not in rendered


@pytest.mark.skipif(not FIXTURE_V3.exists(), reason="the v3 fixture is not present")
def test_the_frozen_fixture_still_carries_no_answer_material() -> None:
    """Checked against the frozen bytes, so it holds without the fact store."""

    rendered = FIXTURE_V3.read_text(encoding="utf-8")

    for key in _ANSWER_MATERIAL:
        assert key not in rendered, f"{key} leaked into a fixture"
