"""P1.5-S2-D1/D2/D3: naming a relational result, and stating it.

The result refers to `slot_id`s because that is the semantic identity that
survives a different evidence support being bound.  A reader needs names, and a
validator needs something to compare an answer against.  Both get it from one
`RelationalOperandDirectory`, built once from the plan -- so neither can grow a
`slot_id -> entity` lookup of its own that disagrees with the other's.

The renderer is only a writer.  It re-sorts nothing, reads no plan, and infers
no relation from anything but the groups it was handed; a renderer that did any
of those would be a second executor.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.domain.calculation import (
    CalculationOperand,
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
    ComparisonRelation,
)
from src.finance.relational_directory import (
    RelationalOperandDirectory,
    RelationalOperandRef,
    render_relational_result,
    resolve_relational_result,
)

APPLE = RelationalOperandRef(slot_id="s1", entity="Apple", metric="G&A", period="FY2025")
VISA = RelationalOperandRef(slot_id="s2", entity="Visa", metric="G&A", period="FY2025")
TESLA = RelationalOperandRef(slot_id="s3", entity="Tesla", metric="G&A", period="FY2025")

DIRECTORY = RelationalOperandDirectory(refs=(APPLE, VISA, TESLA))


def _operand(slot_id: str, value: str, *, evidence: str | None = None) -> CalculationOperand:
    return CalculationOperand(
        name="value",
        value=Decimal(value),
        evidence_chunk_id=evidence or f"E-{slot_id}",
        slot_id=slot_id,
    )


def _result(
    *,
    operation: CalculationOperation = CalculationOperation.RANKING,
    operands: tuple[CalculationOperand, ...] | None = None,
    groups: tuple[tuple[str, ...], ...] | None = (("s2",), ("s1",)),
) -> CalculationResult:
    return CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=operation,
        operands=operands if operands is not None else (_operand("s1", "10"), _operand("s2", "30")),
        ordering_groups=groups,
    )


# --- resolution ---------------------------------------------------------------


def test_a_valid_result_resolves_to_names() -> None:
    resolved = resolve_relational_result(_result(), DIRECTORY)

    assert resolved is not None
    assert resolved.labels() == (("Visa",), ("Apple",))


def test_an_unknown_ref_does_not_resolve() -> None:
    """There is no label to give it, and inventing one from the slot id would
    state the relation in terms the question never used."""

    assert resolve_relational_result(_result(groups=(("s2",), ("s9",))), DIRECTORY) is None


def test_a_duplicated_ref_does_not_resolve() -> None:
    result = _result(
        operands=(_operand("s1", "10"), _operand("s2", "30")),
        groups=(("s2",), ("s2",), ("s1",)),
    )

    assert resolve_relational_result(result, DIRECTORY) is None


def test_a_missing_operand_does_not_resolve() -> None:
    result = _result(
        operands=(_operand("s1", "10"), _operand("s2", "30"), _operand("s3", "20")),
        groups=(("s2",), ("s1",)),
    )

    assert resolve_relational_result(result, DIRECTORY) is None


def test_an_empty_directory_does_not_resolve() -> None:
    assert resolve_relational_result(_result(), RelationalOperandDirectory()) is None


def test_a_non_relational_result_does_not_resolve() -> None:
    result = CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation.DIFFERENCE,
        value=Decimal("20"),
        operands=(_operand("s1", "10"), _operand("s2", "30")),
    )

    assert resolve_relational_result(result, DIRECTORY) is None


def test_a_blocked_result_does_not_resolve() -> None:
    result = CalculationResult(
        status=CalculationStatus.BLOCKED,
        operation=CalculationOperation.COMPARISON,
        operands=(_operand("s1", "10"), _operand("s2", "30")),
        ordering_groups=(("s2",), ("s1",)),
        error_code="INSUFFICIENT_OPERANDS",
    )

    assert resolve_relational_result(result, DIRECTORY) is None


# --- identity does not leak evidence or order ---------------------------------


def test_a_different_evidence_support_renders_identically() -> None:
    """One canonical fact, several supports: the relation is the same relation."""

    first = _result(operands=(_operand("s1", "10", evidence="row-A"),
                              _operand("s2", "30", evidence="row-B")))
    other = _result(operands=(_operand("s1", "10", evidence="row-C"),
                              _operand("s2", "30", evidence="row-D")))

    assert render_relational_result(resolve_relational_result(first, DIRECTORY)) == \
           render_relational_result(resolve_relational_result(other, DIRECTORY))


def test_a_reordered_operand_list_renders_identically() -> None:
    """`operands` is provenance; nothing about its order reaches the answer."""

    forwards = _result(operands=(_operand("s1", "10"), _operand("s2", "30")))
    backwards = _result(operands=(_operand("s2", "30"), _operand("s1", "10")))

    assert render_relational_result(resolve_relational_result(forwards, DIRECTORY)) == \
           render_relational_result(resolve_relational_result(backwards, DIRECTORY))


# --- rendering ----------------------------------------------------------------


def test_a_comparison_renders_as_a_relation() -> None:
    resolved = resolve_relational_result(
        _result(operation=CalculationOperation.COMPARISON), DIRECTORY
    )

    assert render_relational_result(resolved) == "Visa > Apple"


def test_a_tie_renders_as_one_group() -> None:
    resolved = resolve_relational_result(
        _result(groups=(("s1", "s2"),), operation=CalculationOperation.COMPARISON),
        DIRECTORY,
    )

    assert render_relational_result(resolved) == "Apple = Visa"


def test_a_full_ranking_renders_in_order() -> None:
    resolved = resolve_relational_result(
        _result(
            operands=(_operand("s1", "10"), _operand("s2", "30"), _operand("s3", "20")),
            groups=(("s2",), ("s3",), ("s1",)),
        ),
        DIRECTORY,
    )

    assert render_relational_result(resolved) == "Visa > Tesla > Apple"


def test_a_ranking_with_a_tie_renders_both() -> None:
    """`A = B > C` -- ties and strict order in one statement."""

    resolved = resolve_relational_result(
        _result(
            operands=(_operand("s1", "10"), _operand("s2", "30"), _operand("s3", "30")),
            groups=(("s2", "s3"), ("s1",)),
        ),
        DIRECTORY,
    )

    assert render_relational_result(resolved) == "Visa = Tesla > Apple"


def test_a_ref_without_an_entity_falls_back_to_its_metric() -> None:
    """A slot the plan left unqualified is still an operand.

    Refusing to resolve it would invent a requirement the plan did not state;
    naming it by its slot id would state the relation in terms the question
    never used.  The metric is the middle ground, and the order of the fallbacks
    is fixed so an answer cannot silently use one where a reader expects
    another.
    """

    directory = RelationalOperandDirectory(
        refs=(RelationalOperandRef(slot_id="s1", entity=None, metric="Revenue"),)
    )
    resolved = resolve_relational_result(
        _result(operands=(_operand("s1", "10"),), groups=(("s1",),)), directory
    )

    assert render_relational_result(resolved) == "Revenue"


def test_a_ref_with_nothing_at_all_still_resolves_to_its_slot_id() -> None:
    """Traceable beats unresolvable; the slot id is at least the plan's own."""

    directory = RelationalOperandDirectory(refs=(RelationalOperandRef(slot_id="s1"),))
    resolved = resolve_relational_result(
        _result(operands=(_operand("s1", "10"),), groups=(("s1",),)), directory
    )

    assert render_relational_result(resolved) == "s1"


def test_an_empty_ordering_renders_as_nothing() -> None:
    """Not a refusal and not a guess -- there is simply nothing to state."""

    from src.finance.relational_directory import ResolvedRelationalResult

    empty = ResolvedRelationalResult(
        operation=CalculationOperation.RANKING, ordering_groups=()
    )

    assert render_relational_result(empty) == ""


# --- the derived relation agrees with the result's -----------------------------


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        ((("s2",), ("s1",)), ComparisonRelation.FIRST_GT_SECOND),
        ((("s1", "s2"),), ComparisonRelation.EQUAL),
    ],
)
def test_the_resolved_relation_matches_the_result_derivation(
    groups: tuple[tuple[str, ...], ...], expected: ComparisonRelation
) -> None:
    """One derivation rule, applied to the same ordering, twice."""

    resolved = resolve_relational_result(_result(groups=groups), DIRECTORY)

    assert resolved is not None
    assert resolved.relation is expected
