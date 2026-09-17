"""H2A-3B0: the specialist is told the page the evidence came from, or nothing.

F5, from the H2A-3A model context surface audit.  The specialist prompt read
``page = ev.get("page") or 1`` while ``page`` was absent from the SPECIALIST
disclosure profile.  The lookup therefore missed on every call, the fallback
always fired, and every ``Source:`` line asserted page 1 -- a fabricated
provenance value shown to a model, on the live path.

Two errors were stacked in that one expression, and the tests below keep them
apart because they fail differently:

* ``or`` treats a real page ``0`` as absent.  ``EvidencePacketV1.page`` keeps
  ``0`` valid and absence preserved as absence, so a truthiness test conflates
  two states the contract distinguishes.  The bug here is a *dropped* fact.
* the fallback invented a specific value rather than stating the absence.  ``1``
  is indistinguishable from a page the extractor found, which makes it a
  fabrication rather than an absence marker -- unlike ``unit``/``currency``,
  which fall back to "not specified" and say so.  The bug here is an *asserted*
  fact that is false.

These assert on the rendered prompt, and the last one asserts through the real
production boundary rather than the helper.  The renderer was extracted to
``src/generation/specialist_prompt.py`` in this phase for exactly that reason:
written beside ``import torch`` it would be excluded from collection on any
checkout without torch, and would have reported green while testing nothing.
"""

from __future__ import annotations

from typing import Any

from rag_v2.evidence.disclosure import (
    EvidenceDisclosureProfile,
    allowed_fields,
    project,
)
from src.generation.specialist_prompt import render_specialist_prompt

#: A page nobody recorded.  If it ever appears in a prompt, something reached
#: for a value the authoritative field did not have.
POISON_PAGE = 99

RAW_TEXT = "RAW SOURCE TEXT THAT MUST NOT CROSS"


def _packet(**extra: Any) -> dict[str, Any]:
    """One evidence packet, as the loop carries it -- through the real contract.

    Built with ``EvidencePacketV1.from_mapping().to_dict()`` rather than as a
    bare dict so that the test exercises the same normalisation production does
    (``pdf_page`` -> ``page``) instead of a shape no producer emits.
    """

    from rag_v2.adaptive.adaptive_contracts import EvidencePacketV1

    base: dict[str, Any] = {
        "evidence_id": "e1",
        "citation_id": "citation:a",
        "metric": "Revenue",
        "value": "120",
        "period": "FY2024",
        "scope": "consolidated",
        "unit": "USD",
        "currency": "USD",
        "scale": "million",
        "document_id": "doc-1",
        "source_text": RAW_TEXT,
        "content": RAW_TEXT,
    }
    return EvidencePacketV1.from_mapping({**base, **extra}).to_dict()


def _projected(**extra: Any) -> dict[str, Any]:
    return project(_packet(**extra), profile=EvidenceDisclosureProfile.SPECIALIST)


def _source_lines(prompt: str) -> list[str]:
    return [line for line in prompt.splitlines() if line.startswith("Source:")]


def _pack(items: list[dict[str, Any]]) -> Any:
    """Compile a pack from already-projected views.

    H2A-3B3 made the renderer take a pack; this is how a test that wants to
    render a particular projection gets one.  The compiler re-projects what it
    is handed, which on an already-projected SPECIALIST view is the identity --
    so this is the production projection applied twice, not a second one.
    """

    from rag_v2.context import (
        ContextCompilerV1,
        ContextRequestV1,
        ContextRoleV1,
        SpecialistContextPolicyV1,
    )

    return ContextCompilerV1(SpecialistContextPolicyV1()).compile(
        ContextRequestV1(
            role=ContextRoleV1.SPECIALIST,
            invocation_id="test-render",
            query="What was revenue?",
            admitted_evidence=tuple(items),
        )
    )


def _render(*items: dict[str, Any]) -> str:
    return render_specialist_prompt(_pack(list(items)))


# --- the page reaches the prompt -------------------------------------------------


def test_a_recorded_page_reaches_the_prompt() -> None:
    """The ordinary case, which was wrong on every call before this phase."""

    prompt = _render(_projected(page=7))

    assert "Source: doc-1:7" in prompt
    assert "Source: doc-1:1" not in prompt


def test_page_zero_is_a_page_and_survives_the_projection() -> None:
    """``0`` is a recorded page, not an absent one.

    ``EvidencePacketV1.page`` documents ``0`` as valid with absence preserved as
    absence.  A truthiness fallback collapses the two, so this is the case that
    proves the fallback is gone rather than merely reordered.
    """

    view = _projected(page=0)

    assert view["page"] == 0
    assert "Source: doc-1:0" in _render(view)


def test_a_missing_page_is_rendered_as_missing() -> None:
    """Absence is stated, not filled in.

    "not specified" is the same idiom ``unit`` and ``currency`` already use two
    lines above in the renderer, and unlike a number it cannot be mistaken for a
    value the extractor recorded.
    """

    prompt = _render(_projected())

    assert "Source: doc-1:not specified" in prompt
    assert "Source: doc-1:1" not in prompt


def test_an_explicit_none_page_is_absence_not_a_default() -> None:
    """A producer that emits ``page=None`` has said "no page", not "page 1"."""

    assert "page" not in _projected(page=None)
    assert "Source: doc-1:not specified" in _render(_projected(page=None))


# --- the page comes from the authority, and only from it -------------------------


def test_the_metadata_bag_cannot_supply_the_page() -> None:
    """The second-authority shape H2A-2D-2B closed for the Binder.

    ``metadata`` really does carry a page on live packets: ``from_mapping``
    promotes ``page`` to the canonical field *and* leaves it in the bag, because
    the bag is built from the keys the contract did not consume.  So this is not
    a hypothetical shape -- ``_packet(page=99)`` produces it.

    The SPECIALIST profile drops the container outright, so a page recorded only
    there is absent, and a page recorded there must not override the canonical
    field either.
    """

    # Canonical field unset, bag carrying a page: absence, not the bag's value.
    only_in_metadata = _packet(page=POISON_PAGE)
    assert only_in_metadata["metadata"]["page"] == POISON_PAGE
    only_in_metadata["page"] = None

    view = project(only_in_metadata, profile=EvidenceDisclosureProfile.SPECIALIST)
    assert "page" not in view
    prompt = _render(view)
    assert "Source: doc-1:not specified" in prompt
    assert str(POISON_PAGE) not in prompt
    assert "Source: doc-1:1" not in prompt

    # Canonical field and bag disagreeing: the canonical field decides.
    disagreeing = _packet(page=POISON_PAGE)
    disagreeing["page"] = 7

    view = project(disagreeing, profile=EvidenceDisclosureProfile.SPECIALIST)
    assert view["page"] == 7
    prompt = _render(view)
    assert "Source: doc-1:7" in prompt
    assert str(POISON_PAGE) not in prompt


def test_the_extractors_own_page_name_is_normalised_at_the_boundary() -> None:
    """``pdf_page`` is the extractor's name; ``page`` is the canonical field.

    ``EvidencePacketV1.from_mapping`` converts once, at ingestion, which is what
    makes reading the canonical field alone safe rather than lossy.
    """

    assert _projected(pdf_page=13)["page"] == 13
    assert "Source: doc-1:13" in _render(_projected(pdf_page=13))


def test_only_the_canonical_page_field_is_admitted() -> None:
    """Admitting ``page`` did not admit the alias it is normalised from.

    ``pdf_page`` reaching the model as well would be a second page field on the
    model-facing surface -- the duplication H2A-2 removed, reintroduced through
    the profile.
    """

    specialist = set(allowed_fields(EvidenceDisclosureProfile.SPECIALIST))

    assert "page" in specialist
    assert "pdf_page" not in specialist
    assert "metadata" not in specialist
    assert "source_text" not in specialist
    assert "content" not in specialist


# --- several sources ---------------------------------------------------------------


def test_each_source_carries_its_own_page() -> None:
    """Pages are per evidence item, in bound order, with no carry-over."""

    prompt = _render(_projected(page=7), _projected(page=0), _projected())

    assert _source_lines(prompt) == [
        "Source: doc-1:7",
        "Source: doc-1:0",
        "Source: doc-1:not specified",
    ]


# --- through the real production boundary ------------------------------------------


class _RenderingSpecialist:
    """A specialist that records the prompt the boundary rendered for it.

    H2A-3B3.  It used to render the prompt itself; the prompt now arrives
    already rendered from the compiled pack, so recording it is a recording of
    the production model input rather than a second rendering of it.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "rendered"


def _multi_fact_state():
    """A state the routing policy sends to the specialist.

    Two facts differing in period, because MULTI reaches the specialist only
    when the items are *distinct* canonical facts -- two rows of one fact render
    deterministically and never cross a model boundary.
    """

    from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1

    state = AdaptiveRAGStateV1.new("r", "What was revenue?")
    state.add_evidence(
        [
            EvidencePacketV1.from_mapping(_packet(page=7)),
            EvidencePacketV1.from_mapping(
                {**_packet(period="FY2023"), "evidence_id": "e2", "page": 0}
            ),
        ]
    )
    state.bound_evidence_ids = ["e1", "e2"]
    return state


def test_the_production_boundary_hands_the_model_the_true_page() -> None:
    """The fix asserted through the wiring, not through the helper.

    A correct renderer says nothing about what actually reaches a model.  This
    drives ``TrustedV2GenerationCapability``, which is what selects and projects
    the payload that crosses the boundary.
    """

    from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability

    specialist = _RenderingSpecialist()
    TrustedV2GenerationCapability(specialist=specialist).generate(_multi_fact_state())

    assert specialist.prompts, "the specialist was never called"
    prompt = specialist.prompts[0]

    assert "Source: doc-1:7" in prompt
    assert "Source: doc-1:0" in prompt
    assert "Source: doc-1:1" not in prompt
    assert RAW_TEXT not in prompt


def test_the_disclosed_page_is_recorded_in_the_trace_by_name() -> None:
    """Widening the profile changes what the disclosure trace reports.

    The trace names fields and never values, so the page itself must not appear
    in it -- but ``page`` becoming a field that crosses must.
    """

    from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability

    specialist = _RenderingSpecialist()
    capability = TrustedV2GenerationCapability(specialist=specialist)
    capability.generate(_multi_fact_state())

    disclosed = set(capability.trace_snapshot()["disclosed_fields"])

    assert "evidence.page" in disclosed
    assert "evidence.source_text" not in disclosed
    assert "7" not in " ".join(disclosed)
