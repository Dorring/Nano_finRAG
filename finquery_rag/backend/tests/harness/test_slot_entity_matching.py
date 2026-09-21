"""Which fact a slot may claim, when the slot says whose it is.

Entity used to be matched once for the whole plan, against the companies the
*query* mentioned.  In a two-company question that let either company's fact
satisfy either slot -- so a comparison could bind one company twice, satisfy
both slots, and never be contradicted by anything.  That is the question the
question asks, and the old rule could not represent it.

The three cases are tested here with the fallback each of them forbids, because
every one of those fallbacks is a plausible-looking line of code that makes
results better and the answer wrong.
"""

from __future__ import annotations

import pytest

from rag_v2.contracts import RequiredSlot
from rag_v2.supervisor import extract_query_semantic_frame
from src.runtime.trusted_v2_binder import SemanticEvidenceEvaluationCapability


def _slot(**overrides: object) -> RequiredSlot:
    fields: dict[str, object] = {
        "slot_id": "s1",
        "metric": "Total revenue",
        "period": "FY2025",
        "role": "value",
        "value_type": "numeric",
    }
    fields.update(overrides)
    return RequiredSlot(**fields)  # type: ignore[arg-type]


def _fact(entity: str | None) -> dict:
    return {"entity": entity, "metric": "Total revenue", "period": "FY2025"}


def _frame(query: str):
    return extract_query_semantic_frame(query)


def _matches(slot: RequiredSlot, entity: str | None, query: str) -> bool:
    return SemanticEvidenceEvaluationCapability._entity_matches_slot(
        slot, _fact(entity), _frame(query)
    )


# --- case A: both sides carry a canonical identity ---------------------------------------


def test_a_canonical_slot_matches_the_same_canonical_company() -> None:
    slot = _slot(entity="Visa", entity_id="visa")

    assert _matches(slot, "Visa", "Compare Apple and Visa revenue in FY2025")


def test_a_canonical_slot_rejects_a_different_canonical_company() -> None:
    """The comparison case: the other company is not this slot's company."""

    slot = _slot(entity="Visa", entity_id="visa")

    assert not _matches(slot, "Apple", "Compare Apple and Visa revenue in FY2025")


def test_a_canonical_slot_is_not_gated_by_a_company_the_query_never_named() -> None:
    """A plan may not introduce a company the question did not mention."""

    slot = _slot(entity="Visa", entity_id="visa")

    assert not _matches(slot, "Visa", "What was Apple's revenue in FY2025?")


# --- case B: a declared identity must not degrade to a text guess ------------------------


def test_a_canonical_slot_refuses_a_fact_it_cannot_canonicalise() -> None:
    """`Pfizer` is a real company the twelve-entry vocabulary does not know.

    The slot declared `visa`; a fact that canonicalises to nothing has not been
    shown to be Visa, and comparing the *words* would answer a question the plan
    did not ask.
    """

    slot = _slot(entity="Visa", entity_id="visa")

    assert not _matches(slot, "Pfizer", "Compare Visa and Pfizer revenue in FY2025")


def test_a_canonical_slot_refuses_a_fact_with_no_entity_at_all() -> None:
    slot = _slot(entity="Visa", entity_id="visa")

    assert not _matches(slot, None, "What was Visa's revenue in FY2025?")


# --- case C: a mention with no identity, compared strictly -------------------------------


def test_a_mention_matches_the_same_mention() -> None:
    """The vocabulary does not know Pfizer, and the slot still means Pfizer."""

    slot = _slot(entity="Pfizer", entity_id=None)

    assert _matches(slot, "Pfizer", "What was Pfizer's revenue in FY2025?")


def test_a_mention_rejects_a_different_mention() -> None:
    slot = _slot(entity="Pfizer", entity_id=None)

    assert not _matches(slot, "Visa", "Compare Pfizer and Visa revenue in FY2025")


def test_a_mention_does_not_infer_an_alias() -> None:
    """`Pfizer Inc.` is not `Pfizer` here, on purpose.

    Inferring it is ontology work.  Doing it in the matcher would make the
    answer depend on which spellings happened to be written down, and the
    failure would look like a successful bind.
    """

    slot = _slot(entity="Pfizer", entity_id=None)

    assert not _matches(slot, "Pfizer Inc.", "What was Pfizer's revenue in FY2025?")


def test_a_mention_is_not_gated_by_the_query_vocabulary() -> None:
    """Pfizer is absent from the vocabulary; gating on it would reject the slot.

    This is the case the whole mention/identity split exists for, so the
    query-level test is deliberately not applied here.
    """

    slot = _slot(entity="Pfizer", entity_id=None)

    assert _matches(slot, "Pfizer", "Compare Pfizer and Visa revenue in FY2025")


# --- no coordinate: exactly the behaviour from before ------------------------------------


def test_a_slot_with_no_entity_keeps_the_query_level_rule() -> None:
    slot = _slot()

    assert _matches(slot, "Visa", "Compare Apple and Visa revenue in FY2025")
    assert _matches(slot, "Apple", "Compare Apple and Visa revenue in FY2025")
    assert not _matches(slot, "Tesla", "Compare Apple and Visa revenue in FY2025")


def test_a_query_naming_nobody_constrains_nobody() -> None:
    slot = _slot()

    assert _matches(slot, "Visa", "What was the total revenue in FY2025?")
