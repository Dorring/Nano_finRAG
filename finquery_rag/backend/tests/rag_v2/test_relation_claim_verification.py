"""Whether a comparison claim is supported, checked against the facts.

The rule this replaces asked only whether a calculation result existed.  A
comparison plan never produces one, so it refused every answer that *stated* a
comparison and released every answer that stated the two values and left the
comparison to the reader.  Release depended on the answer's wording, and the
more directly it answered the question the more likely it was to be refused.

Every case below is a real answer from the P1.3-D run against the real facts it
cited, including the one whose attribution is the wrong way round -- that one is
why a relation word with no direction does not release anything.
"""

from __future__ import annotations

from typing import Any

from rag_v2.generation.contracts import AnswerEnvelopeV1
from rag_v2.runtime.semantic_claims import (
    SemanticClaimDecision,
    SemanticClaimVerifierV1,
)

VERIFIER = SemanticClaimVerifierV1()


def _envelope(answer: str) -> AnswerEnvelopeV1:
    return AnswerEnvelopeV1(
        query_id="q",
        route="MULTI_EVIDENCE",
        answer_text=answer,
        citation_ids=(),
        generator_provider="test",
        generator_model="test",
    )


def _item(entity: str, value: str, metric: str = "Foreign exchange contracts") -> dict[str, Any]:
    return {
        "fact_id": f"atomic:{entity.casefold()}",
        "evidence_id": f"atomic:{entity.casefold()}",
        "citation_id": f"citation:v2:{entity.casefold()}",
        "entity": entity,
        "metric": metric,
        "period": "FY2025",
        "value": value,
    }


def _supported(answer: str, items: list[dict[str, Any]]) -> bool:
    return SemanticClaimVerifierV1._relation_supported(
        answer, {"evidence_items": items}
    )


# --- the three comparisons the D run refused ---------------------------------------------


def test_a_correct_directional_comparison_is_supported() -> None:
    """compare-004: Apple $109,079 against Microsoft $(44), and Apple said larger."""

    answer = (
        "The verified evidence reports that Apple's Foreign exchange contracts were "
        "larger in FY2025 compared to Microsoft's [citation:v2:apple]."
    )
    items = [_item("Apple", "$ 109,079"), _item("Microsoft", "$ (44)")]

    assert _supported(answer, items)


def test_a_comparison_between_incomparable_quantities_is_not_supported() -> None:
    """compare-007: Pfizer $5 against JPMorganChase 5.49 %.

    Re-derived, because the original pair could not distinguish the two rules.
    It stated the direction the *numbers* contradict -- it called Pfizer, whose
    5 is the smaller, the higher -- so the direction check refused it whatever
    the representation rule did.  It would have gone on passing with that rule
    deleted, which is the definition of a test that has stopped testing.

    Here the answer states the direction the numbers do show, so the relation is
    refused only if a percentage and a currency amount are not comparable, and
    the control below proves the direction path is reachable at all.
    """

    answer = (
        "The verified evidence reports that Pfizer's Discount rate was lower in "
        "FY2025 compared to JPMorganChase [citation:v2:pfizer]."
    )
    mixed = [
        _item("Pfizer", "$5", metric="Discount rate"),
        _item("JPMorganChase", "5.49 %", metric="Discount rate"),
    ]

    assert not _supported(answer, mixed)

    # The control.  Same direction, same ordering, both read as one kind of
    # quantity: this must be supported, or the assertion above says nothing
    # about comparability -- a rule that refused every comparison would pass it.
    same_kind = [
        _item("Pfizer", "5", metric="Discount rate"),
        _item("JPMorganChase", "5.49", metric="Discount rate"),
    ]

    assert _supported(answer, same_kind)


# --- percentages against percentages ------------------------------------------
#
# No benchmark question compares two percentages today, so nothing covered this
# before.  It is the capability the representation work exists to give, and it
# is covered here rather than left to the first question that needs it.


def test_a_correct_comparison_between_two_percentages_is_supported() -> None:
    answer = (
        "The verified evidence reports that Pfizer's Discount rate was higher in "
        "FY2025 compared to JPMorganChase [citation:v2:pfizer]."
    )
    items = [
        _item("Pfizer", "7.2 %", metric="Discount rate"),
        _item("JPMorganChase", "5.49 %", metric="Discount rate"),
    ]

    assert _supported(answer, items)


def test_an_inverted_percentage_comparison_is_not_supported() -> None:
    """compare-008's shape, now reachable for percentages.

    The facts show Pfizer higher and the answer says JPMorganChase is.  Both
    read as percentages, so the only thing that can refuse this is the direction
    check -- which is exactly what should refuse it.
    """

    answer = (
        "The verified evidence reports that JPMorganChase's Discount rate was "
        "higher in FY2025 compared to Pfizer [citation:v2:jpmorganchase]."
    )
    items = [
        _item("Pfizer", "7.2 %", metric="Discount rate"),
        _item("JPMorganChase", "5.49 %", metric="Discount rate"),
    ]

    assert not _supported(answer, items)


def test_a_relation_with_no_direction_releases_nothing() -> None:
    """compare-008, and this is the case that decides the rule.

    The answer reads "JPMorganChase's Diluted earnings per share was $ 7.46,
    compared to Apple's $ 20.02".  The facts are the other way round: Apple is
    $7.46 and JPMorganChase is 20.02.  The attribution is inverted.

    Judging only that a comparison was *establishable* would have released it,
    so a relation word without a direction supports nothing.
    """

    answer = (
        "The verified evidence reports that JPMorganChase's Diluted earnings per "
        "share was $ 7.46 in FY2025 [citation:v2:jpmorganchase], compared to "
        "Apple's $ 20.02 in FY2025 [citation:v2:apple]."
    )
    items = [
        _item("Apple", "$ 7.46", metric="Diluted earnings per share"),
        _item("JPMorganChase", "20.02", metric="Diluted earnings per share"),
    ]

    assert not _supported(answer, items)


# --- the direction has to be the one the evidence shows -----------------------------------


def test_the_wrong_direction_is_not_supported() -> None:
    """The check that makes this a verification rather than a permission."""

    answer = "Apple's Foreign exchange contracts were lower than Microsoft's."
    items = [_item("Apple", "$ 109,079"), _item("Microsoft", "$ (44)")]

    assert not _supported(answer, items)


def test_a_correct_downward_comparison_is_supported() -> None:
    answer = "Microsoft's Foreign exchange contracts were lower than Apple's."
    items = [_item("Apple", "$ 109,079"), _item("Microsoft", "$ (44)")]

    assert _supported(answer, items)


def test_a_tie_supports_neither_direction() -> None:
    answer = "Apple's Foreign exchange contracts were larger than Microsoft's."
    items = [_item("Apple", "$ 20"), _item("Microsoft", "20")]

    assert not _supported(answer, items)


def test_a_comparison_needs_two_companies() -> None:
    answer = "Apple's Foreign exchange contracts were larger."
    items = [_item("Apple", "$ 109,079")]

    assert not _supported(answer, items)


def test_a_company_with_two_disagreeing_values_is_not_usable() -> None:
    """Which of them the answer meant cannot be established, so neither is."""

    answer = "Apple's Foreign exchange contracts were larger than Microsoft's."
    items = [
        _item("Apple", "$ 109,079"),
        dict(_item("Apple", "$ 12"), fact_id="atomic:apple-2", evidence_id="atomic:apple-2"),
        _item("Microsoft", "$ (44)"),
    ]

    assert not _supported(answer, items)


def test_an_unlocatable_subject_is_not_supported() -> None:
    """A direction word with no company before it has no subject to check."""

    answer = "The larger Foreign exchange contracts value was in FY2025."
    items = [_item("Apple", "$ 109,079"), _item("Microsoft", "$ (44)")]

    assert not _supported(answer, items)


def test_a_value_that_does_not_canonicalise_is_dropped() -> None:
    answer = "Apple's Foreign exchange contracts were larger than Microsoft's."
    items = [_item("Apple", "$ 109,079"), _item("Microsoft", "not a number")]

    assert not _supported(answer, items)


def test_a_fact_without_a_company_is_dropped() -> None:
    answer = "Apple's Foreign exchange contracts were larger than Microsoft's."
    items = [_item("Apple", "$ 109,079"), _item("", "$ (44)")]

    assert not _supported(answer, items)


# --- the reason code itself ---------------------------------------------------------------


def test_the_reason_code_follows_the_check() -> None:
    """The claim is only raised when the evidence could not carry it."""

    supported_answer = (
        "The verified evidence reports that Apple's Foreign exchange contracts "
        "were larger in FY2025 compared to Microsoft's [citation:v2:apple]."
    )
    packet = {
        "evidence_items": [_item("Apple", "$ 109,079"), _item("Microsoft", "$ (44)")],
        "calculation_result": None,
        "allowed_citation_ids": [],
    }

    result = VERIFIER.verify(packet, _envelope(supported_answer))

    assert "SCV_RELATION_AMBIGUOUS" not in result.reason_codes
    assert result.decision is not SemanticClaimDecision.AMBIGUOUS


def test_a_relational_claim_the_evidence_cannot_carry_is_still_ambiguous() -> None:
    packet = {
        "evidence_items": [
            _item("Apple", "$ 7.46", metric="Diluted earnings per share"),
            _item("JPMorganChase", "20.02", metric="Diluted earnings per share"),
        ],
        "calculation_result": None,
        "allowed_citation_ids": [],
    }
    answer = (
        "The verified evidence reports that JPMorganChase's Diluted earnings per "
        "share was $ 7.46 in FY2025 [citation:v2:jpmorganchase], compared to "
        "Apple's $ 20.02 in FY2025 [citation:v2:apple]."
    )

    result = VERIFIER.verify(packet, _envelope(answer))

    assert "SCV_RELATION_AMBIGUOUS" in result.reason_codes
    assert result.decision is SemanticClaimDecision.AMBIGUOUS
