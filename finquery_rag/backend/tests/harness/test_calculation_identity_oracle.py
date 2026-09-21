"""H2A-2D-3A: calculation identity, pinned by a golden vector.

The audit found four stored copies of one derivable calculation identity. Before
any of them is demoted, the authority needs a guard that does not depend on any
of them -- otherwise removing the mirrors would be verified by the mirrors.

**This file never calls `CalculationResult.calculation_id` to produce an
expected value.** The literal below was derived once, by hand, from the
documented payload contract (`operation`, `formula_version`, `value`, `unit`,
and each operand's `name`/`value`/`evidence_chunk_id`, canonically serialised and
truncated to 16 hex characters), and is frozen here. `test_the_literal_is_derivable_from_the_documented_contract`
re-derives it from that specification so the magic string has a provenance a
future reader can check -- and it is still independent of production, because it
re-implements the contract rather than calling it.

The sensitivity cases are the point of a golden vector: a constant would pass the
equality test and fail every one of them.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal

from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
)

# --- the authored vector ------------------------------------------------------

OPERATION = CalculationOperation.GROWTH_RATE
VALUE = Decimal("0.0209")
UNIT = "ratio"
FORMULA_VERSION = "growth_rate.v1"
OPERANDS: tuple[tuple[str, Decimal, str], ...] = (
    ("current", Decimal("391000000"), "CURRENT"),
    ("previous", Decimal("383000000"), "PRIOR"),
)

#: Frozen literal. Derived from the contract above, not read from production.
GOLDEN_ID = "C1-b55ba5d38df1276c"


def _result(**overrides: object) -> CalculationResult:
    """Build the authored vector, with named fields overridable for sensitivity."""

    operands = overrides.pop("operands", OPERANDS)
    kwargs: dict[str, object] = {
        "status": CalculationStatus.EXECUTED,
        "operation": OPERATION,
        "value": VALUE,
        "unit": UNIT,
        "formula_version": FORMULA_VERSION,
        "operands": tuple(
            CalculationOperand(name, value, evidence_chunk_id=evidence)
            for name, value, evidence in operands  # type: ignore[misc]
        ),
    }
    kwargs.update(overrides)
    return CalculationResult(**kwargs)  # type: ignore[arg-type]


def _contract_payload(**overrides: object) -> dict[str, object]:
    """The documented payload, rebuilt from this file's literals."""

    operands = overrides.pop("operands", OPERANDS)
    payload: dict[str, object] = {
        "operation": OPERATION.value,
        "formula_version": FORMULA_VERSION,
        "value": str(VALUE),
        "unit": UNIT,
        "operands": [
            {"name": name, "value": str(value), "evidence_chunk_id": evidence}
            for name, value, evidence in operands  # type: ignore[misc]
        ],
    }
    payload.update(overrides)
    return payload


def _digest(payload: dict[str, object]) -> str:
    return "C1-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


# --- the golden vector --------------------------------------------------------


def test_the_authored_vector_has_the_frozen_identity() -> None:
    assert _result().calculation_id == GOLDEN_ID


def test_the_literal_is_derivable_from_the_documented_contract() -> None:
    """The magic string's provenance, made checkable.

    An independent implementation of the published contract -- not a call to the
    property it guards. If the contract's payload ever changes, this fails and
    the literal above has to be re-derived deliberately rather than silently
    becoming whatever production now emits.
    """

    assert _digest(_contract_payload()) == GOLDEN_ID


# --- sensitivity: identity inputs must matter --------------------------------


def test_changing_an_operand_value_changes_the_identity() -> None:
    moved = (("current", Decimal("392000000"), "CURRENT"), OPERANDS[1])
    assert _result(operands=moved).calculation_id != GOLDEN_ID


def test_changing_an_operand_evidence_changes_the_identity() -> None:
    """Identity covers operand *provenance*, not just arithmetic.

    The same number read from a different place is a different calculation.
    """

    moved = (("current", Decimal("391000000"), "OTHER"), OPERANDS[1])
    assert _result(operands=moved).calculation_id != GOLDEN_ID


def test_changing_the_operation_changes_the_identity() -> None:
    assert (
        _result(operation=CalculationOperation.DIFFERENCE).calculation_id != GOLDEN_ID
    )


def test_changing_the_result_value_changes_the_identity() -> None:
    assert _result(value=Decimal("0.0210")).calculation_id != GOLDEN_ID


def test_the_operand_order_is_part_of_the_identity() -> None:
    """Reversing operands is a different calculation, not a rewording."""

    assert _result(operands=tuple(reversed(OPERANDS))).calculation_id != GOLDEN_ID


# --- admissibility is a property of the contract ------------------------------


def test_a_blocked_result_has_no_identity() -> None:
    """H2A-1B: a result that is not admissible has no id to obtain.

    Demoting the state mirrors must not resurrect one -- and this is the test
    that would notice, because it asserts on the authority rather than on a
    mirror that could be assigned regardless.
    """

    blocked = CalculationResult(status=CalculationStatus.BLOCKED)
    assert blocked.calculation_id is None


def test_a_failed_result_has_no_identity() -> None:
    failed = CalculationResult(status=CalculationStatus.FAILED)
    assert failed.calculation_id is None


def test_a_result_without_an_operation_has_no_identity() -> None:
    assert (
        CalculationResult(status=CalculationStatus.EXECUTED, value=Decimal("1")).calculation_id
        is None
    )
