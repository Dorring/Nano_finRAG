"""P1.5-S2-B: executing a relational operation.

The contract exists (`tests/finance/test_relational_result_contract.py`); this
covers getting a relation *into* it.  Three things are being asserted, and the
second is the one that is easy to get wrong:

1. a comparison resolves to an explicit relation, and a ranking to an explicit
   ordering over `slot_id`;
2. the executor dispatches on **what the adapter returned**, not on the
   operation's name -- so there is no place where "ranking means an ordering"
   is written down twice;
3. an operand set that cannot be ordered declines through the *existing*
   `BLOCKED` path, which is the machinery that makes an incomplete comparison
   fail closed rather than reach the release validator as prose.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
    CalculationPlan,
    CalculationStatus,
    ComparisonRelation,
)
from src.finance.calculation_executor import execute_plan
from src.finance.calculation_registry import (
    CALCULATION_REGISTRY,
    RelationalToolResult,
    _comparison_adapter,
    _ranking_adapter,
)


def _operand(
    slot_id: str,
    value: str,
    *,
    unit: str | None = None,
    evidence: str | None = None,
) -> CalculationOperand:
    return CalculationOperand(
        name="value",
        value=Decimal(value),
        unit=unit,
        evidence_chunk_id=evidence or f"E-{slot_id}",
        slot_id=slot_id,
    )


def _plan(
    operation: CalculationOperation,
    operands: tuple[CalculationOperand, ...],
) -> CalculationPlan:
    return CalculationPlan(
        operation=operation,
        operands=operands,
        formula_version=CALCULATION_REGISTRY[operation].formula_version,
        target_metric=operation.value,
    )


# --- both operations are registered and executable ----------------------------


def test_both_relational_operations_are_registered() -> None:
    assert CalculationOperation.COMPARISON in CALCULATION_REGISTRY
    assert CalculationOperation.RANKING in CALCULATION_REGISTRY


def test_a_comparison_is_over_exactly_two_operands() -> None:
    entry = CALCULATION_REGISTRY[CalculationOperation.COMPARISON]

    assert entry.min_operands == 2
    assert entry.max_operands == 2


def test_a_ranking_accepts_however_many_the_plan_asked_for() -> None:
    """"No fixed roles" is the point: naming them would restate the plan's
    cardinality in a second place."""

    entry = CALCULATION_REGISTRY[CalculationOperation.RANKING]

    assert entry.min_operands == 2
    assert entry.max_operands > 2
    assert entry.operand_roles == ()


# --- comparison ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("30", "10", ComparisonRelation.LHS_GT_RHS),
        ("10", "30", ComparisonRelation.RHS_GT_LHS),
        ("10", "10", ComparisonRelation.EQUAL),
    ],
)
def test_a_comparison_states_which_side_is_larger(
    left: str, right: str, expected: ComparisonRelation
) -> None:
    result = _comparison_adapter((_operand("s1", left), _operand("s2", right)), 4)

    assert result.ok is True
    assert result.relation is expected
    assert result.ordering_groups is None


def test_a_negative_operand_is_larger_than_nothing_found_by_sign_alone() -> None:
    """`compare-002`'s shape: Apple is `(8,077)`, so Visa is the larger one.

    The relation says that; the sign would only say "-1".
    """

    result = _comparison_adapter(
        (_operand("s1", "-8077"), _operand("s2", "1926")), 4
    )

    assert result.relation is ComparisonRelation.RHS_GT_LHS


def test_a_comparison_needs_two_operands() -> None:
    result = _comparison_adapter((_operand("s1", "10"),), 4)

    assert result.ok is False
    assert result.error


def test_a_comparison_of_incomparable_units_declines() -> None:
    """Compatibility is asked of the primitive that already decides it.

    `difference` refuses operands whose units disagree, so ordering inherits
    that rule rather than restating it -- a second rule would be a second
    authority for "may these be compared" and the two would drift.
    """

    result = _comparison_adapter(
        (_operand("s1", "10", unit="USD"), _operand("s2", "4", unit="EUR")), 4
    )

    assert result.ok is False


# --- ranking ------------------------------------------------------------------


def test_a_ranking_orders_descending() -> None:
    result = _ranking_adapter(
        (_operand("s1", "10"), _operand("s2", "30"), _operand("s3", "20")), 4
    )

    assert result.ok is True
    assert result.ordering_groups == (("s2",), ("s3",), ("s1",))
    assert result.relation is None


def test_equal_operands_land_in_one_group() -> None:
    """`s2 > s1 = s3` -- the reason groups are used rather than a flat order."""

    result = _ranking_adapter(
        (_operand("s1", "10"), _operand("s2", "30"), _operand("s3", "10")), 4
    )

    assert result.ordering_groups == (("s2",), ("s1", "s3"))


def test_the_refs_are_slot_ids_not_evidence_ids() -> None:
    """One canonical fact, several supports: the ranking names the operand."""

    result = _ranking_adapter(
        (
            _operand("s1", "10", evidence="row-A"),
            _operand("s2", "30", evidence="row-B"),
        ),
        4,
    )

    flat = [ref for group in result.ordering_groups or () for ref in group]
    assert flat == ["s2", "s1"]
    assert not any("row" in ref for ref in flat)


def test_a_ranking_needs_at_least_two_operands() -> None:
    result = _ranking_adapter((_operand("s1", "10"),), 4)

    assert result.ok is False


# --- the executor -------------------------------------------------------------


def test_the_executor_turns_a_relation_into_a_result() -> None:
    plan = _plan(
        CalculationOperation.COMPARISON,
        (_operand("s1", "10"), _operand("s2", "30")),
    )

    result = execute_plan(plan)

    assert result.status is CalculationStatus.EXECUTED
    assert result.relation is ComparisonRelation.RHS_GT_LHS
    assert result.relational_result_is_well_formed is True


def test_the_executor_turns_an_ordering_into_a_result() -> None:
    plan = _plan(
        CalculationOperation.RANKING,
        (_operand("s1", "10"), _operand("s2", "30"), _operand("s3", "20")),
    )

    result = execute_plan(plan)

    assert result.status is CalculationStatus.EXECUTED
    assert result.ordering_groups == (("s2",), ("s3",), ("s1",))
    assert result.relational_result_is_well_formed is True


def test_a_relational_result_carries_no_value() -> None:
    """The relation is the answer; a derived sign would be a second field that
    can disagree with the first."""

    plan = _plan(
        CalculationOperation.COMPARISON,
        (_operand("s1", "10"), _operand("s2", "30")),
    )

    assert execute_plan(plan).value is None


def test_an_incomplete_comparison_blocks_through_the_existing_path() -> None:
    """The machinery that makes `compare-002`'s shape fail closed.

    This is not new behaviour: `min_operands` and `INSUFFICIENT_OPERANDS` are
    already written and tested for the arithmetic operations, and a relational
    operation inherits them by being registered with a minimum.
    """

    plan = _plan(CalculationOperation.COMPARISON, (_operand("s1", "10"),))

    result = execute_plan(plan)

    assert result.status is CalculationStatus.BLOCKED
    assert result.error_code == "INSUFFICIENT_OPERANDS"
    assert result.relational_result_is_well_formed is False


def test_an_incomparable_comparison_blocks_rather_than_answering() -> None:
    plan = _plan(
        CalculationOperation.COMPARISON,
        (_operand("s1", "10", unit="USD"), _operand("s2", "4", unit="EUR")),
    )

    result = execute_plan(plan)

    assert result.status is CalculationStatus.BLOCKED
    assert result.error_code == "PRIMITIVE_DECLINED"


def test_the_executor_dispatches_on_the_returned_type() -> None:
    """A relational adapter's return is handled by its own branch.

    Asserted by the effect rather than by reading the source: if the executor
    had fallen through to the `ToolResult` branch, a `RelationalToolResult` has
    no `points_value` and the result would have come back BLOCKED with
    `PRIMITIVE_DECLINED` instead of EXECUTED.
    """

    assert isinstance(
        _comparison_adapter((_operand("s1", "10"), _operand("s2", "30")), 4),
        RelationalToolResult,
    )

    result = execute_plan(
        _plan(
            CalculationOperation.COMPARISON,
            (_operand("s1", "10"), _operand("s2", "30")),
        )
    )

    assert result.status is CalculationStatus.EXECUTED
    assert result.error_code is None


def test_the_arithmetic_path_is_untouched() -> None:
    """A quantity still flows through `ToolResult` and carries a value."""

    plan = _plan(
        CalculationOperation.DIFFERENCE,
        (
            CalculationOperand(name="current", value=Decimal("10"),
                               evidence_chunk_id="E1", slot_id="s1"),
            CalculationOperand(name="previous", value=Decimal("4"),
                               evidence_chunk_id="E2", slot_id="s2"),
        ),
    )

    result = execute_plan(plan)

    assert result.status is CalculationStatus.EXECUTED
    assert result.value == Decimal("6")
    assert result.relation is None
    assert result.ordering_groups is None
