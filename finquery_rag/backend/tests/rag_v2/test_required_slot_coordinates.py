"""The slot's evidence coordinates, and the one distinction they exist for.

``metric`` and ``period`` say what is wanted; ``entity`` says whose.  Without it
"Apple's FY2025 revenue" and "Visa's FY2025 revenue" are the same requirement,
which is why all twenty comparison fixtures were unanswerable and unscoreable.

``entity`` is an open-world mention and ``entity_id`` is an identity the Harness
*derived* from it, in that order and never the reverse.  The load-bearing
consequence is tested here: **``entity_id is None`` means the ontology cannot
name the mention, not that there is no constraint.**  Treating the two as the
same would silently widen a slot to any company whenever the vocabulary happens
not to cover it -- which is the failure that would be hardest to notice, because
it makes results look better.
"""

from __future__ import annotations

import pytest

from rag_v2.contracts import (
    Action,
    ContractError,
    Intent,
    RequiredSlot,
    SupervisorPlan,
    slot_key_error,
)
from rag_v2.supervisor import validate_plan_v2_01


def _v1_payload() -> dict:
    """A slot exactly as V2-00 wrote it: six keys, no coordinates."""

    return {
        "slot_id": "s1",
        "metric": "Total revenue",
        "period": "FY2025",
        "role": "value",
        "value_type": "numeric",
        "unit": None,
    }


def _slot(**overrides: object) -> RequiredSlot:
    fields = dict(_v1_payload())
    fields.update(overrides)
    return RequiredSlot(**fields)  # type: ignore[arg-type]


# --- the coordinates exist and stay out of the way ---------------------------------------


def test_a_slot_without_coordinates_round_trips_unchanged() -> None:
    slot = _slot()

    assert RequiredSlot.from_dict(slot.to_dict()) == slot


def test_a_v1_payload_still_parses() -> None:
    """Nothing written before these fields existed may stop parsing."""

    slot = RequiredSlot.from_dict(_v1_payload())

    assert slot.entity is None
    assert slot.entity_id is None
    assert slot.to_dict()["entity"] is None


def test_the_coordinates_survive_a_round_trip() -> None:
    slot = _slot(entity="The Coca-Cola Company", entity_id="ko")

    assert RequiredSlot.from_dict(slot.to_dict()) == slot


def test_the_plan_round_trip_keeps_them() -> None:
    plan = SupervisorPlan.from_dict(
        {
            "intent": "DIRECT_FACT",
            "required_slots": [_v1_payload() | {"entity": "Visa", "entity_id": "visa"}],
            "operation": None,
            "next_action": "RETRIEVE",
        }
    )

    restored = SupervisorPlan.from_dict(plan.to_dict())

    assert restored == plan
    assert restored.required_slots[0].entity_id == "visa"


# --- the distinction the two fields exist for --------------------------------------------


def test_an_unnamed_entity_is_not_the_same_as_no_entity() -> None:
    """The whole point of a mention beside an id.

    ``Pfizer`` is a real company the deployment's twelve-entry vocabulary does
    not know.  That must record as *a constrained slot the ontology cannot
    name*, and it must not read the same as a slot that names nobody.
    """

    constrained_but_unnamed = _slot(entity="Pfizer", entity_id=None)
    unconstrained = _slot(entity=None, entity_id=None)

    assert constrained_but_unnamed.entity == "Pfizer"
    assert constrained_but_unnamed.entity_id is None
    assert constrained_but_unnamed != unconstrained
    assert constrained_but_unnamed.to_dict() != unconstrained.to_dict()


def test_an_empty_string_is_not_a_silent_absence() -> None:
    """`""` would read as falsy everywhere and mean "no entity" by accident."""

    with pytest.raises(ContractError, match="entity must be None or"):
        _slot(entity="")


def test_a_whitespace_only_coordinate_is_refused_too() -> None:
    with pytest.raises(ContractError, match="entity_id must be None or"):
        _slot(entity_id="   ")


# --- what the providers accept ------------------------------------------------------------


def test_the_v1_key_set_is_still_a_slot() -> None:
    assert slot_key_error(_v1_payload()) is None


def test_the_coordinates_are_accepted() -> None:
    assert slot_key_error(_v1_payload() | {"entity": "Visa", "entity_id": "visa"}) is None


def test_a_missing_required_key_is_refused() -> None:
    payload = _v1_payload()
    del payload["role"]

    assert "missing required fields" in str(slot_key_error(payload))


def test_an_unknown_key_is_refused() -> None:
    """An unrecognised key is a misunderstanding, not an extension."""

    error = slot_key_error(_v1_payload() | {"company": "Visa"})

    assert error is not None
    assert "unknown fields" in error


def test_a_non_object_is_refused() -> None:
    assert slot_key_error(["not", "a", "slot"]) == "slot must be an object"  # type: ignore[arg-type]


# --- the validator ------------------------------------------------------------------------


def test_a_plan_carrying_coordinates_still_validates() -> None:
    plan = SupervisorPlan(
        intent=Intent.MULTI_EVIDENCE,
        required_slots=(
            _slot(slot_id="s1", entity="Apple", entity_id="aapl"),
            _slot(slot_id="s2", entity="Visa", entity_id="visa"),
        ),
        operation=None,
        next_action=Action.RETRIEVE,
    )

    assert validate_plan_v2_01(plan) is plan
