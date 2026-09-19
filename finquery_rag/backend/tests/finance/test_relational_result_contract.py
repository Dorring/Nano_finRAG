"""P1.5-S2-A: the contract for relational results.

`comparison` and `ranking` answer with a *relation between operands* rather than
a quantity, and the contract has to say so explicitly.  Two things drove the
shape, and both are asserted here rather than described:

**The result refers to semantic operands, never evidence ids.**  One canonical
fact may be supported by several independent evidence rows, so an identity keyed
on which support happened to be bound would change while the ranking it
describes did not.  The refs are `slot_id`.

**`operands` is provenance, and its order carries no meaning.**  Letting the
tuple's order double as the ranking result would make every incidental
reordering -- by provenance, by serialization, by a tidy-up -- silently rewrite
the answer.

This is the contract only.  Nothing sets these operations in production yet, so
no behaviour changes; the guard for that is the last test, which asserts the
identity of an ordinary quantity is still tied to its evidence.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.domain.calculation import (
    RELATIONAL_OPERATIONS,
    CalculationOperand,
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
    ComparisonRelation,
)


def _operand(slot_id: str, value: str, *, evidence: str = "E1", name: str = "value") -> CalculationOperand:
    return CalculationOperand(
        name=name,
        value=Decimal(value),
        evidence_chunk_id=evidence,
        slot_id=slot_id,
    )


def _ranking(
    operands: tuple[CalculationOperand, ...],
    groups: tuple[tuple[str, ...], ...] | None,
) -> CalculationResult:
    return CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation.RANKING,
        operands=operands,
        ordering_groups=groups,
    )


def _comparison(
    operands: tuple[CalculationOperand, ...],
    relation: ComparisonRelation | None,
) -> CalculationResult:
    return CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation.COMPARISON,
        operands=operands,
        relation=relation,
    )


# --- the two operations exist, and are named in one place ----------------------


def test_the_relational_operations_are_the_two_we_think_they_are() -> None:
    assert RELATIONAL_OPERATIONS == {
        CalculationOperation.COMPARISON,
        CalculationOperation.RANKING,
    }


# --- ranking: structure -------------------------------------------------------


def test_a_well_formed_ranking_is_admissible() -> None:
    result = _ranking(
        (_operand("s1", "10"), _operand("s2", "30"), _operand("s3", "20")),
        (("s2",), ("s3",), ("s1",)),
    )

    assert result.relational_result_is_well_formed is True


def test_a_tie_is_expressible() -> None:
    """`s2 > s1 = s3` -- the reason groups are used from the start."""

    result = _ranking(
        (_operand("s1", "10"), _operand("s2", "30"), _operand("s3", "10")),
        (("s2",), ("s1", "s3")),
    )

    assert result.relational_result_is_well_formed is True


def test_an_unknown_ref_is_inadmissible() -> None:
    result = _ranking(
        (_operand("s1", "10"), _operand("s2", "30")),
        (("s2",), ("s9",)),
    )

    assert result.relational_result_is_well_formed is False


def test_a_duplicated_ref_is_inadmissible() -> None:
    result = _ranking(
        (_operand("s1", "10"), _operand("s2", "30")),
        (("s2",), ("s2",), ("s1",)),
    )

    assert result.relational_result_is_well_formed is False


def test_a_missing_ref_is_inadmissible() -> None:
    result = _ranking(
        (_operand("s1", "10"), _operand("s2", "30"), _operand("s3", "20")),
        (("s2",), ("s1",)),
    )

    assert result.relational_result_is_well_formed is False


def test_a_missing_ordering_is_inadmissible() -> None:
    """A ranking with no ordering is prose with a status field."""

    result = _ranking((_operand("s1", "10"), _operand("s2", "30")), None)

    assert result.relational_result_is_well_formed is False


def test_an_empty_group_is_inadmissible() -> None:
    result = _ranking(
        (_operand("s1", "10"), _operand("s2", "30")),
        (("s2",), ()),
    )

    assert result.relational_result_is_well_formed is False


def test_an_operand_without_a_slot_id_is_inadmissible() -> None:
    """An operand that cannot be referred to cannot appear in a checkable order."""

    result = _ranking(
        (_operand("s1", "10"), _operand("", "30")),
        (("s1",), ("",)),
    )

    assert result.relational_result_is_well_formed is False


def test_a_blocked_ranking_is_not_well_formed() -> None:
    """Structure cannot rescue a result that produced nothing."""

    result = CalculationResult(
        status=CalculationStatus.BLOCKED,
        operation=CalculationOperation.RANKING,
        operands=(_operand("s1", "10"), _operand("s2", "30")),
        ordering_groups=(("s2",), ("s1",)),
        error_code="INSUFFICIENT_OPERANDS",
    )

    assert result.relational_result_is_well_formed is False


# --- comparison ---------------------------------------------------------------


def test_a_comparison_needs_a_relation() -> None:
    """`value = -1` does not say *which* operand was larger on its own."""

    operands = (_operand("s1", "10"), _operand("s2", "30"))

    assert _comparison(operands, ComparisonRelation.RHS_GT_LHS).relational_result_is_well_formed is True
    assert _comparison(operands, None).relational_result_is_well_formed is False


def test_a_comparison_is_over_exactly_two_operands() -> None:
    result = _comparison(
        (_operand("s1", "10"), _operand("s2", "30"), _operand("s3", "20")),
        ComparisonRelation.RHS_GT_LHS,
    )

    assert result.relational_result_is_well_formed is False


def test_equality_is_a_relation_not_an_absence() -> None:
    result = _comparison(
        (_operand("s1", "10"), _operand("s2", "10")),
        ComparisonRelation.EQUAL,
    )

    assert result.relational_result_is_well_formed is True


# --- identity: semantic, not evidence-bound -----------------------------------


def test_operand_order_does_not_change_a_ranking_identity() -> None:
    """`operands` is provenance; its order is not a claim."""

    first = _ranking(
        (_operand("s1", "10"), _operand("s2", "30"), _operand("s3", "20")),
        (("s2",), ("s3",), ("s1",)),
    )
    reordered = _ranking(
        (_operand("s3", "20"), _operand("s1", "10"), _operand("s2", "30")),
        (("s2",), ("s3",), ("s1",)),
    )

    assert first.calculation_id == reordered.calculation_id


def test_a_different_evidence_support_does_not_change_a_ranking_identity() -> None:
    """One canonical fact, several supports: the ranking is the same ranking.

    This is the reason the refs are slot ids.  The same two facts bound from a
    different physical row describe the same ordering, and an identity that
    moved would make "did the result change" unanswerable.
    """

    first = _ranking(
        (_operand("s1", "10", evidence="row-A"), _operand("s2", "30", evidence="row-B")),
        (("s2",), ("s1",)),
    )
    other_supports = _ranking(
        (_operand("s1", "10", evidence="row-C"), _operand("s2", "30", evidence="row-D")),
        (("s2",), ("s1",)),
    )

    assert first.calculation_id == other_supports.calculation_id


def test_a_different_value_does_change_a_ranking_identity() -> None:
    """Guards the two tests above: an identity that ignored values says nothing."""

    first = _ranking(
        (_operand("s1", "10"), _operand("s2", "30")),
        (("s2",), ("s1",)),
    )
    changed = _ranking(
        (_operand("s1", "10"), _operand("s2", "31")),
        (("s2",), ("s1",)),
    )

    assert first.calculation_id != changed.calculation_id


def test_a_different_order_does_change_a_ranking_identity() -> None:
    """And one that ignored the ordering would not describe the answer."""

    operands = (_operand("s1", "10"), _operand("s2", "30"))
    descending = _ranking(operands, (("s2",), ("s1",)))
    ascending = _ranking(operands, (("s1",), ("s2",)))

    assert descending.calculation_id != ascending.calculation_id


def test_a_quantity_identity_still_covers_its_evidence() -> None:
    """Production behaviour unchanged: no existing identity moved.

    A quantity's identity stays tied to the row it was read from and to the
    order the operands were built in.  Only relational results are semantic, and
    no operation is relational in production yet.
    """

    def difference(operands: tuple[CalculationOperand, ...]) -> CalculationResult:
        return CalculationResult(
            status=CalculationStatus.EXECUTED,
            operation=CalculationOperation.DIFFERENCE,
            value=Decimal("20"),
            operands=operands,
        )

    first = difference(
        (_operand("s1", "10", evidence="row-A", name="current"),
         _operand("s2", "30", evidence="row-B", name="previous"))
    )
    other_support = difference(
        (_operand("s1", "10", evidence="row-C", name="current"),
         _operand("s2", "30", evidence="row-B", name="previous"))
    )

    assert first.calculation_id != other_support.calculation_id


# --- serialization ------------------------------------------------------------


def test_the_ordering_survives_serialization_as_lists() -> None:
    result = _ranking(
        (_operand("s1", "10"), _operand("s2", "30")),
        (("s2",), ("s1",)),
    )

    assert result.to_dict()["ordering_groups"] == [["s2"], ["s1"]]
    assert result.to_public_dict()["ordering_groups"] == [["s2"], ["s1"]]


def test_an_absent_ordering_serializes_as_none_not_empty() -> None:
    """"No ordering" and "an empty ordering" are different claims."""

    result = CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation.RANKING,
        operands=(_operand("s1", "10"),),
    )

    assert result.to_dict()["ordering_groups"] is None


def test_the_relation_serializes_as_its_value() -> None:
    result = _comparison(
        (_operand("s1", "10"), _operand("s2", "30")),
        ComparisonRelation.RHS_GT_LHS,
    )

    assert result.to_dict()["relation"] == "rhs_gt_lhs"


def test_operands_carry_their_slot_id_across_serialization() -> None:
    operand = _operand("s2", "30")

    assert operand.to_dict()["slot_id"] == "s2"
    assert operand.to_public_dict()["slot_id"] == "s2"
