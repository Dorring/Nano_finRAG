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
    NESTED_FIELD_POLICY,
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
        view = project(evidence, profile=profile)
        assert RAW_TEXT not in repr(view), profile
        # A container may cross, but only with the keys its profile names --
        # V1_ANSWER permits ``metadata`` and must still filter it key by key.
        for name in ("metadata",):
            nested = view.get(name)
            if isinstance(nested, dict):
                assert set(nested) <= set(NESTED_FIELD_POLICY.get(profile, {}).get(name, ())), profile
                assert "source_text" not in nested, profile


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
    """Audited from ``src/generation/specialist_prompt.py`` field by field.

    One field the prompt *asks for* is still deliberately absent:

    - ``source_text``: the prompt's ``Evidence: {ev['source_text']}`` branch has
      never fired on this path, because the text is nested under ``metadata``.
      Adding it would broaden model exposure while claiming to unify it.

    ``page`` was the second such field and is now admitted -- H2A-3B0.  The
    deferral this test used to record ("it joins the profile when H2A-1C makes
    it survive that far") was met by H2A-2D-2B, which made
    ``EvidencePacketV1.page`` the single authority for page provenance.  Leaving
    it out was not the conservative choice it appeared to be: with ``page``
    absent from the profile, ``project`` never emitted it, so the prompt's
    ``ev.get("page") or 1`` fallback fired on every call and every source line
    asserted page 1.  The field being *missing* was what produced the false
    statement -- see ``test_specialist_page_disclosure.py``.
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
        "page",
    }
    assert "source_text" not in specialist


# --- a real production boundary ----------------------------------------------


class _RecordingSpecialist:
    """A specialist that records exactly what it was handed.

    H2A-3B3.  What it is handed is a rendered prompt now, so the assertions
    below read the compiled pack instead.  That is the stronger place to read
    from: the pack is what crossed the boundary under Disclosure Authority, and
    this records only that the call happened and what text it carried.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
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
    # for a state it cannot render: one fact renders deterministically and never
    # crosses a model boundary at all.
    #
    # H2A-2C-1 tightened what "cannot render" means.  This used to reach the
    # specialist by supplying two rows that differed only in `evidence_id` --
    # the same metric, period and value twice.  That is *one* canonical fact
    # stated twice, which the structured renderer states correctly, so it now
    # renders and never crosses a boundary.  The second fact differs in period,
    # which is what makes this a genuine multi-fact state -- the thing a
    # specialist is actually for -- and the disclosure assertions below are
    # unchanged by it.
    state = AdaptiveRAGStateV1.new("r", "What was revenue?")
    state.add_evidence(
        [
            EvidencePacketV1.from_mapping(_evidence()),
            EvidencePacketV1.from_mapping(
                {**_evidence(period="FY2023"), "evidence_id": "e2"}
            ),
        ]
    )
    state.bound_evidence_ids = ["e1", "e2"]

    capability.generate(state)

    assert specialist.prompts, "the specialist was never called"
    pack = capability.last_context_pack
    assert pack is not None, "the specialist was reached without a compiled context"
    for item in pack.evidence:
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
    # Namespaced by artifact type, so a calculation field cannot be mistaken for
    # an evidence field in the trace.
    disclosed = set(snapshot["disclosed_fields"])
    assert all(
        name.split(".", 1)[-1] in set(allowed_fields(EvidenceDisclosureProfile.SPECIALIST))
        or name.startswith("calculation.")
        for name in disclosed
    ), disclosed
    assert RAW_TEXT not in repr(snapshot)


# --- the V1 rollback runtime ---------------------------------------------
#
# V1 answers from assembled retrieval context rather than evidence packets, so
# it is a different authoritative source type with its own profile.  These drive
# the real chain -- retrieval chunk -> ContextBuilder -> LLMGateway -> provider
# -- and assert on the prompt the provider actually receives, not on the helper.


def _retrieval_chunk(**overrides: Any) -> dict[str, Any]:
    """A chunk shaped like the ones retrieval produces, private fields included."""

    chunk = {
        "content": "Apple FY2024 total net sales were 391,035 million.",
        "doc_id": "user_7::annual_report.pdf",
        "score": 0.87,
        "metadata": {
            "type": "text",
            "page": 12,
            "parent_id": "p-12",
            "section_path": ["Item 7", "Revenue"],
            "child_hit_count": 1,
            "internal_rerank_score": "PRIVATE_SCORE",
            "retrieval_debug": {"PRIVATE_DEBUG": True},
        },
    }
    chunk.update(overrides)
    return chunk


class _SpyLLMClient:
    """Captures exactly what the answer model is asked."""

    def __init__(self) -> None:
        self.messages: list[list[dict[str, str]]] = []
        outer = self

        class _Completions:
            def create(self, *, model: str, messages: list, **kw: Any) -> Any:
                outer.messages.append([dict(m) for m in messages])
                return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]})()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def _v1_prompt(chunks: list[dict[str, Any]]) -> tuple[str, _SpyLLMClient, Any]:
    """Run the real V1 chain and return the prompt the model was given."""

    import asyncio

    from src.generation.llm_gateway import LLMGateway
    from src.retrieval.context_builder import ContextBuilder

    builder = ContextBuilder()
    context, sources = builder.build(chunks)
    client = _SpyLLMClient()
    gateway = LLMGateway(llm_client=client, model_name="spy", max_new_tokens=64)
    asyncio.run(gateway.generate(context, "What was FY2024 revenue?"))
    return client.messages[-1][-1]["content"], client, builder


def test_v1_approved_chunk_text_still_reaches_the_answer_prompt() -> None:
    """Governed, not narrowed.  V1 answers from retrieved text; removing it
    would silently break the runtime this profile exists to govern."""

    prompt, _, builder = _v1_prompt([_retrieval_chunk()])

    assert "Apple FY2024 total net sales were 391,035 million." in prompt
    assert builder.last_disclosure_profile == "V1_ANSWER"
    assert "content" in builder.last_disclosed_fields


def test_v1_retrieval_internals_do_not_reach_the_answer_prompt() -> None:
    """A chunk is not a prompt field, and its metadata bag is not either."""

    prompt, _, _ = _v1_prompt([_retrieval_chunk()])

    assert "PRIVATE_SCORE" not in prompt
    assert "PRIVATE_DEBUG" not in prompt
    assert "internal_rerank_score" not in prompt
    assert "retrieval_debug" not in prompt


def test_a_field_added_to_chunks_later_does_not_reach_the_answer_prompt() -> None:
    """Future-field safety on the V1 path, the same property V2 has."""

    chunk = _retrieval_chunk()
    chunk["metadata"]["board_commentary"] = "PRIVATE_FUTURE_FIELD"
    chunk["future_top_level_field"] = "PRIVATE_FUTURE_TOP"

    prompt, _, _ = _v1_prompt([chunk])

    assert "PRIVATE_FUTURE_FIELD" not in prompt
    assert "PRIVATE_FUTURE_TOP" not in prompt


def test_v1_sources_are_still_built_from_the_projection() -> None:
    """The source list is a model-facing surface too, and must keep working."""

    import asyncio  # noqa: F401

    from src.retrieval.context_builder import ContextBuilder

    _, sources = ContextBuilder().build([_retrieval_chunk()])

    assert sources
    assert sources[0]["page"] == 12
    assert sources[0]["chunk_id"] == "user_7::annual_report.pdf"
    # The source list is itself model-facing, so it must be built from permitted
    # fields only.  ``filename`` is deliberately not asserted: it is derived from
    # the doc_id by V1's existing prefix-stripping, which is unrelated to
    # disclosure and should not be pinned here.
    assert set(sources[0]) == {
        "filename", "page", "type", "score", "chunk_id", "parent_id",
        "section_path", "child_hit_count",
    }
    assert "internal_rerank_score" not in sources[0]


def test_the_v1_answer_model_receives_strings_not_retrieval_objects() -> None:
    """The boundary's input type is the contract, not just its content."""

    _, client, _ = _v1_prompt([_retrieval_chunk()])

    for message in client.messages[0]:
        assert isinstance(message["content"], str)
    assert "doc_id" not in client.messages[0][-1]["content"]


def test_the_v1_profile_permits_what_the_formatter_reads_and_nothing_else() -> None:
    """Audited from ``ContextBuilder.build``, not from the chunk schema."""

    from rag_v2.evidence.disclosure import NESTED_FIELD_POLICY

    fields = set(allowed_fields(EvidenceDisclosureProfile.V1_ANSWER))
    nested = set(NESTED_FIELD_POLICY[EvidenceDisclosureProfile.V1_ANSWER]["metadata"])

    # Everything the formatter reads.
    assert {"content", "doc_id", "score", "metadata"} <= fields
    # And everything the *second* consumer of its output reads.  Narrowing this
    # back to what ContextBuilder alone needs is what silently added a "could
    # not verify these documents" suffix to V1 answers.
    assert {"document_name", "doc_name", "chunk_id", "page"} <= fields
    assert {"document_name", "doc_name", "filename"} <= nested
    assert {
        "type",
        "page",
        "parent_id",
        "section_path",
        "child_hit_count",
        "table_num",
        "parent_excerpt",
    } <= nested
    # And nothing it does not.
    assert "metadata" not in nested
    assert nested.isdisjoint({"internal_rerank_score", "retrieval_debug"})


def test_narrowing_the_v1_profile_to_the_formatter_alone_changes_answers() -> None:
    """The trap this profile already fell into once, pinned.

    ``ContextBuilder.build`` produces ``last_context_evidence``, and that has a
    second consumer: ``EvidenceItem.from_chunk``, whose ``document_name`` drives
    the answerability check.  An allowlist built from the formatter alone dropped
    it, the answerability check then reported the requested document as missing,
    and every V1 answer gained a "could not verify" suffix.

    So: the fields ``from_chunk`` reads must remain permitted, and this test
    fails the moment someone narrows the profile back to the formatter's needs.
    """

    from src.domain.evidence import EvidenceItem

    chunk = {
        "content": "text",
        "doc_id": "d1",
        "document_name": "report.pdf",
        "page": 3,
        "metadata": {"filename": "report.pdf"},
    }
    projected = project(chunk, profile=EvidenceDisclosureProfile.V1_ANSWER)
    item = EvidenceItem.from_chunk(projected)

    assert item.document_name == "report.pdf"
    assert item.page == 3


# --- calculation payloads: the second artifact type ---------------------------
#
# The audit found `_call_specialist` applying two policies in one call.  The
# evidence items beside it were projected; the calculation payload was
# `CalculationResult.to_dict()` -- the internal diagnostics form, which carries
# each operand's full `source_text` and the raw `error_message`, where
# `to_public_dict()` substitutes a bounded excerpt.
#
# The shipping router happens not to reach that branch, because `_route` forces
# the deterministic calculator when the plan is a calculation and the calculator
# refuses a plan that is not.  That is two facts staying true, not a boundary --
# which is the pattern this module was written to replace.

CALC_SECRET = "RAW OPERAND SOURCE TEXT FROM FILING PAGE 7"
CALC_ERROR = "INTERNAL EXCEPTION /var/lib/secret.py"


class _RecordingSpecialistWithCalculation:
    """Records that it was called, and the text it was handed.

    H2A-3B3.  The calculation payload used to arrive here as one of three
    arguments; it now arrives as part of the rendered prompt, and the governed
    projection that produced it lives in the pack.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "recorded"


def _state_with_calculation() -> Any:
    from decimal import Decimal

    from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1
    from src.domain.calculation import (
        CalculationOperation, CalculationOperand, CalculationResult, CalculationStatus,
    )

    calculation = CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation.DIFFERENCE,
        value=Decimal("8"),
        unit="USD",
        operands=(
            CalculationOperand(name="current", value=Decimal("391"), source_text=CALC_SECRET),
        ),
        error_message=CALC_ERROR,
    )
    state = AdaptiveRAGStateV1.new("r", "Why did revenue change?")
    # Not CALCULATION, so `_route` does not force the deterministic calculator --
    # i.e. the state a direct caller could construct.
    state.intent = "MULTI_EVIDENCE"
    state.add_evidence(
        [
            EvidencePacketV1.from_mapping({**_evidence(), "evidence_id": "e1"}),
            EvidencePacketV1.from_mapping({**_evidence(), "evidence_id": "e2"}),
        ]
    )
    state.bound_evidence_ids = ["e1", "e2"]
    state._calculation_result_obj = calculation
    # H2A-2D-3B: the state's `calculation_result_id` is derived from this
    # object, so restating the id here was a second copy of a truth the
    # result already owns -- and the setter no longer exists to allow it.
    return state


def test_the_specialist_calculation_payload_is_projected_not_raw() -> None:
    """The P0: one call, two policies, until now."""

    from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability

    specialist = _RecordingSpecialistWithCalculation()
    capability = TrustedV2GenerationCapability(specialist=specialist)
    capability.generate(_state_with_calculation())

    assert specialist.prompts, "the specialist must actually have been called"
    pack = capability.last_context_pack
    assert pack is not None
    payload = pack.calculation
    assert payload is not None, "the branch under test must be reached"

    assert CALC_SECRET not in repr(payload), "raw operand text crossed the boundary"
    assert CALC_ERROR not in repr(payload), "raw error text crossed the boundary"
    assert "operands" not in payload
    assert set(payload) == {"operation", "unit", "value"}

    # And none of it reached the text a model reads either.
    assert CALC_SECRET not in specialist.prompts[0]
    assert CALC_ERROR not in specialist.prompts[0]


def test_the_specialist_still_receives_the_calculation_fields_it_reads() -> None:
    """Governed, not narrowed: the prompt reads these three and still gets them."""

    from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability

    capability = TrustedV2GenerationCapability(
        specialist=_RecordingSpecialistWithCalculation()
    )
    capability.generate(_state_with_calculation())

    assert dict(capability.last_context_pack.calculation) == {
        "operation": "difference",
        "unit": "USD",
        "value": "8",
    }


def test_the_calculation_fields_are_named_in_the_trace() -> None:
    """The audit found the trace accounting for evidence but silent on calculation.

    "What was this model allowed to see" has to have a complete answer, and it
    has to be answerable by name without carrying values.
    """

    from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability

    capability = TrustedV2GenerationCapability(
        specialist=_RecordingSpecialistWithCalculation()
    )
    capability.generate(_state_with_calculation())
    fields = capability.trace_snapshot()["disclosed_fields"]

    assert "calculation.operation" in fields
    assert "calculation.value" in fields
    assert any(name.startswith("evidence.") for name in fields)
    assert CALC_SECRET not in repr(fields)


def test_a_profile_without_a_calculation_allowlist_receives_nothing() -> None:
    """Deny by default: an unconsidered boundary gets None, not the payload."""

    from rag_v2.evidence.disclosure import EvidenceDisclosureProfile, project_calculation

    payload = {"operation": "difference", "value": "8", "operands": [{"source_text": CALC_SECRET}]}

    for profile in EvidenceDisclosureProfile:
        view = project_calculation(payload, profile=profile)
        if profile is EvidenceDisclosureProfile.SPECIALIST:
            assert view == {"operation": "difference", "value": "8"}
        else:
            assert view is None, profile
