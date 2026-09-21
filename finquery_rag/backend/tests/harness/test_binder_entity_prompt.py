"""What the binder is told about a slot's entity.

The contract grew an entity coordinate and the slot serialization carried it to
the model, but the system prompt still said nothing about it: the only
disambiguation rule it had was about *scope*, and its ambiguity rule -- "use
AMBIGUOUS when multiple materially plausible facts cannot be safely
distinguished" -- read, to a model that had never been told entity is a
constraint, as an instruction to refuse every comparison.  It did refuse them,
and it was following the prompt.

So the prompt now says what the field means.  These tests pin the three claims
that make it work, and pin the scope paragraph **verbatim**, because this change
is scoped to entity and moving the scope wording while claiming otherwise is how
a single-variable change stops being one.
"""

from __future__ import annotations

from rag_v2.evidence.prompt import (
    BINDER_SYSTEM_PROMPT_V1,
    BINDER_RESPONSE_FORMAT,
    build_binder_messages,
)

#: The scope guidance as it stood before this change.  Pinned so that "entity
#: only" is checked rather than asserted.
_SCOPE_PARAGRAPH = (
    "For an unqualified direct financial fact, interpret the requested scope as the\n"
    "company-wide/consolidated total. Never bind a segment row merely because its\n"
    "metric and period match. If the packet contains only segment rows and no\n"
    "explicitly matching aggregate row, return MISSING. If the question explicitly\n"
    "requests a segment, bind only that segment; never substitute a consolidated\n"
    "total. Use the supplied row, metric-path, and scope metadata as source context."
)


def test_the_scope_guidance_is_untouched() -> None:
    assert _SCOPE_PARAGRAPH in BINDER_SYSTEM_PROMPT_V1


def test_the_binder_is_told_a_slots_entity_is_a_requirement() -> None:
    """Not a hint.  A fact from another company does not satisfy the slot."""

    prompt = BINDER_SYSTEM_PROMPT_V1

    assert "``entity``" in prompt
    assert "requirement of the slot and not a" in prompt
    assert "does not satisfy the slot" in prompt


def test_the_binder_is_told_another_companys_fact_is_not_a_candidate() -> None:
    """The exact reasoning that produced AMBIGUOUS on every comparison.

    A packet holding two companies' facts for one metric looks, under the old
    wording alone, like "multiple materially plausible facts".  It is not: each
    slot named the company it wants.
    """

    prompt = BINDER_SYSTEM_PROMPT_V1

    assert "do not\nreturn AMBIGUOUS because several companies report the same metric" in prompt
    assert "leave it out of the bindings rather than weighing it" in prompt


def test_ambiguity_is_narrowed_to_one_named_company() -> None:
    prompt = BINDER_SYSTEM_PROMPT_V1

    assert "more than one fact for\nthe *same* named company" in prompt


def test_a_slot_that_names_nobody_is_still_unconstrained() -> None:
    """The behaviour every existing plan depends on must not change."""

    assert "names no\ncompany is unconstrained as before" in BINDER_SYSTEM_PROMPT_V1


# --- what reaches the model -----------------------------------------------------------------


def test_both_slot_and_fact_entity_reach_the_model() -> None:
    """The fix is only sound because the model can see both sides.

    The slot says which company it wants and each fact says which company it is;
    if either were invisible the prompt could not reconcile them, and the right
    change would have been somewhere else entirely.
    """

    from rag_v2.evidence.binder_fact_view import build_runtime_binder_fact_view
    from rag_v2.contracts import RequiredSlot

    slot = RequiredSlot(
        slot_id="s1",
        metric="General and administrative",
        period="FY2025",
        role="value",
        value_type="numeric",
        entity="Visa",
        entity_id="visa",
    )
    fact = build_runtime_binder_fact_view(
        {
            "fact_id": "atomic:x",
            "evidence_id": "atomic:x",
            "entity": "Visa",
            "metric": "General and administrative",
            "period": "FY2025",
            "value": "1,926",
            "provenance_complete": True,
            "physical_source_id": "s",
        }
    )

    assert slot.to_dict()["entity"] == "Visa"
    assert fact["entity"] == "Visa"


def test_the_serialized_messages_are_unchanged_in_shape() -> None:
    """D changes what the binder is *told*, not what it is *sent*."""

    messages = build_binder_messages({"required_slots": [{"slot_id": "s1"}]})

    assert [message["role"] for message in messages] == ["system", "user"]
    # The system message *is* the prompt, so the three claims above are claims
    # about what reaches the model rather than about a string nobody reads.
    assert messages[0]["content"] == BINDER_SYSTEM_PROMPT_V1
    assert messages[1]["content"].startswith("BinderRequest JSON:")
    # The response contract is untouched by this change.
    assert BINDER_RESPONSE_FORMAT["type"] == "json_schema"
