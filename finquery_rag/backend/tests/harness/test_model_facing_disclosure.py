"""H2A-1A: one authority decides what a model may see.

Before this, the same evidence had two disclosure policies: the Binder's, an
allowlist, and the Specialist's, nothing.  The Specialist's safety came from the
*shape* of a packet dict -- its prompt asks for ``ev["source_text"]`` at the top
level and the text is nested under ``metadata``, so the branch is dead.  Safety
that holds because two shapes happen to disagree stops holding the moment either
changes.

These tests pin the properties that replace the accident:

- deny by default, for every boundary;
- an unknown field is invisible, including one the evidence contract gains later;
- the two profiles differ, and are produced by one implementation;
- a real production model boundary receives a projection, not an evidence object.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_v2.evidence.disclosure import (
    PROFILE_FIELDS,
    EvidenceDisclosureProfile,
    UnknownDisclosureProfile,
    allowed_fields,
    project,
)
from tests.harness.harness_support import packet

RAW_TEXT = "RAW SOURCE TEXT THAT MUST NOT CROSS"


def _evidence(**extra: Any) -> dict[str, Any]:
    """A packet-shaped evidence object, as the loop actually carries it."""

    from rag_v2.adaptive.adaptive_contracts import EvidencePacketV1

    return EvidencePacketV1.from_mapping(
        {**packet(), "source_text": RAW_TEXT, "content": RAW_TEXT, **extra}
    ).to_dict()


# --- deny by default ---------------------------------------------------------


def test_an_unlisted_field_never_crosses_any_profile() -> None:
    """The core rule.  A field the profile does not name is invisible."""

    evidence = _evidence()
    evidence["brand_new_field"] = "leaked?"

    for profile in EvidenceDisclosureProfile:
        view = project(evidence, profile=profile)
        assert "brand_new_field" not in view, profile


def test_a_field_added_to_the_evidence_contract_later_does_not_appear() -> None:
    """Future-field safety, which is the whole point of an allowlist.

    An exclude-list would leak this the moment the contract grew; an allowlist
    cannot.  Modelled by adding fields the profiles have never heard of.
    """

    evidence = _evidence()
    for name in ("internal_retrieval_score", "next_year_note", "board_commentary"):
        evidence[name] = RAW_TEXT

    for profile in EvidenceDisclosureProfile:
        view = project(evidence, profile=profile)
        assert set(view) <= set(allowed_fields(profile)), profile


def test_raw_source_text_never_crosses_any_profile() -> None:
    """The text lives in ``metadata``; no profile may reach into it."""

    evidence = _evidence()
    assert evidence["metadata"]["source_text"] == RAW_TEXT  # it really is there

    for profile in EvidenceDisclosureProfile:
        rendered = repr(project(evidence, profile=profile))
        assert RAW_TEXT not in rendered, profile
        assert "metadata" not in project(evidence, profile=profile), profile


def test_an_unknown_profile_is_refused_rather_than_defaulted() -> None:
    """Failing open on an unrecognised boundary would defeat the whole rule."""

    with pytest.raises(UnknownDisclosureProfile):
        allowed_fields("SUPERVISOR")


# --- the two profiles differ, under one authority ----------------------------


def test_the_profiles_are_generated_by_one_implementation() -> None:
    """Different field sets, one policy.

    The point is not that the profiles are equal -- they must not be -- but that
    a second ad-hoc allowlist was not written next to the first.
    """

    assert set(PROFILE_FIELDS) == set(EvidenceDisclosureProfile)
    for profile, fields in PROFILE_FIELDS.items():
        projected = project(_evidence(), profile=profile)
        assert set(projected) <= set(fields), profile


def test_the_binder_sees_more_than_the_specialist_and_both_are_closed() -> None:
    """The Binder selects one fact from many and needs the table provenance.

    The Specialist's profile is the field set its prompt actually reads, so it
    is smaller -- and it is a subset here, which is a property of these two
    prompts rather than a rule the authority enforces.
    """

    binder = set(allowed_fields(EvidenceDisclosureProfile.BINDER))
    specialist = set(allowed_fields(EvidenceDisclosureProfile.SPECIALIST))

    assert specialist < binder
    assert "table_title" in binder and "table_title" not in specialist
    assert "evidence_id" in specialist


def test_the_specialist_profile_is_what_its_prompt_reads() -> None:
    """Audited from ``LocalSpecialistGenerator.render_prompt`` field by field.

    Two fields the prompt *asks for* are deliberately absent:

    - ``source_text``: the prompt's ``Evidence: {ev['source_text']}`` branch has
      never fired on this path, because the text is nested under ``metadata``.
      Adding it would broaden model exposure while claiming to unify it.
    - ``page``: the prompt reads ``ev.get("page")`` at the top level and falls
      back to 1.  It joins the profile when H2A-1C makes it survive that far.
    """

    specialist = set(allowed_fields(EvidenceDisclosureProfile.SPECIALIST))

    assert specialist == {
        "evidence_id",
        "citation_id",
        "metric",
        "normalized_metric",
        "period",
        "value",
        "unit",
        "currency",
        "scale",
        "scope",
        "document_id",
    }
    assert "source_text" not in specialist
    assert "page" not in specialist


# --- a real production boundary ----------------------------------------------


class _RecordingSpecialist:
    """A specialist that records exactly what it was handed."""

    def __init__(self) -> None:
        self.payloads: list[list[dict[str, Any]]] = []

    def generate(
        self,
        question: str,
        evidence_items: list[dict[str, Any]],
        calculation_result: Any = None,
    ) -> str:
        self.payloads.append([dict(item) for item in evidence_items])
        return "recorded"


def test_a_real_specialist_call_receives_a_projection_not_an_evidence_object() -> None:
    """Tested through the production wiring, not the helper.

    The helper being correct says nothing about what actually reaches a model.
    This drives the real generation capability, which is what constructs the
    payload that crosses the boundary.
    """

    from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1
    from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability

    specialist = _RecordingSpecialist()
    capability = TrustedV2GenerationCapability(specialist=specialist)

    # Two bound facts, because the routing policy only reaches the specialist
    # for a multi-evidence route; one fact renders deterministically and never
    # crosses a model boundary at all.
    state = AdaptiveRAGStateV1.new("r", "What was revenue?")
    state.add_evidence(
        [
            EvidencePacketV1.from_mapping(_evidence()),
            EvidencePacketV1.from_mapping({**_evidence(), "evidence_id": "e2"}),
        ]
    )
    state.bound_evidence_ids = ["e1", "e2"]

    capability.generate(state)

    assert specialist.payloads, "the specialist was never called"
    for item in specialist.payloads[0]:
        assert "metadata" not in item
        assert "source_text" not in item
        assert "content" not in item
        assert RAW_TEXT not in repr(item)
        assert set(item) <= set(allowed_fields(EvidenceDisclosureProfile.SPECIALIST))


def test_the_disclosure_is_recorded_by_field_name_only() -> None:
    """A disclosure question must be answerable without leaking content."""

    from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1
    from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability

    capability = TrustedV2GenerationCapability(specialist=_RecordingSpecialist())
    state = AdaptiveRAGStateV1.new("r", "What was revenue?")
    state.add_evidence(
        [
            EvidencePacketV1.from_mapping(_evidence()),
            EvidencePacketV1.from_mapping({**_evidence(), "evidence_id": "e2"}),
        ]
    )
    state.bound_evidence_ids = ["e1", "e2"]

    capability.generate(state)

    snapshot = capability.trace_snapshot()
    assert "disclosed_fields" in snapshot
    assert set(snapshot["disclosed_fields"]) <= set(
        allowed_fields(EvidenceDisclosureProfile.SPECIALIST)
    )
    assert RAW_TEXT not in repr(snapshot)
