"""H2A-3B2: a specialist prompt states what the evidence says, or states nothing.

F10, from the H2A-3A model context surface audit.  Five expressions in the
renderer answered a field the evidence did not carry with a value it made up:

    scale or "1"            scope or metric
    document_id or "filing"  metric or "Metric"     period or "Period"

They were worse than an omission, and the difference is worth stating because it
is the whole reason this phase exists.  A reader can tell ``Unit: not specified``
from a real unit -- the marker is not a unit, and ``currency`` two lines below
was already using it.  There is no such thing as a page ``1`` that nobody
recorded, no metric named ``Metric``, and no filing called ``filing``.  Those are
*values* where the record had none, and a model shown one has been handed
provenance that does not exist.

The last one was the worst of the five.  ``scope or metric`` answered "what
scope?" with "the scope is the metric" -- not a placeholder at all, but a new
fact about the world, and one that reads as authoritative because it is the same
word the model can see on the line above.

Why the assertions look like this
---------------------------------

They are **total and fixture-derived**, because a spot check cannot catch this
family.  The defect was five separate fabrications in five different fields, each
with its own literal, and a test asserting ``"Scale: 1" not in prompt`` would
have caught exactly one of them.  So the tests below derive every expected line
from the *authored evidence* and the *contract's absence rule*, never from the
renderer -- an expectation read out of the code under test is not an expectation.

The other half is the shape of the block: the frozen schema is a fixed set of
lines, so absence must be *stated* rather than omitted.  Omitting the line is
honest too, and it was the alternative the audit weighed; it loses because the
block's shape would then depend on the data, and answer rule 6 asks the model to
notice missing evidence, which a stated absence serves and a vanished line does
not.  The canonical renderer for this same SHA-pinned schema,
``rag_v2/generation/financial_view_v1.py``, spells absence the same way in its
``_value`` helper -- for every field, including metric, period and scope.

``not specified`` is therefore written out in this file rather than imported.
Importing the renderer's constant would make "absence is stated honestly" true by
definition, which is the circularity H2A-2's rule forbids.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from rag_v2.evidence.disclosure import (
    EvidenceDisclosureProfile,
    project,
)
from src.generation.specialist_prompt import render_specialist_prompt

#: The contract's absence marker.  Spelled out, not imported -- see the module
#: docstring.  If the renderer changes how it spells absence, this file should
#: go red and someone should decide whether that was meant.
ABSENT = "not specified"

#: Evidence with every field the SPECIALIST profile admits, present.
FULL: dict[str, Any] = {
    "evidence_id": "e1",
    "citation_id": "citation:a",
    "metric": "Revenue",
    "period": "FY2024",
    "scope": "consolidated",
    "value": "391",
    "unit": "USD",
    "currency": "USD",
    "scale": "million",
    "document_id": "doc-1",
    "page": 7,
}

#: Prompt label -> the evidence field it renders.  Every line of a block is
#: accounted for, so a field added later cannot quietly escape these tests.
LABEL_OF = {
    "metric": "Metric",
    "period": "Period",
    "scope": "Scope",
    "value": "Value",
    "unit": "Unit",
    "currency": "Currency",
    "scale": "Scale",
}

#: The two fields that share the ``Source`` line, and the fixtures that make
#: each of the four combinations.
REMOVABLE = tuple(LABEL_OF) + ("document_id", "page")

_EVIDENCE_HEADER = re.compile(r"^\[E\d+\]$")

#: Labels the pre-F10 renderer could produce with a value nobody recorded.
FABRICATED_FORMS = (
    "Metric: Metric",
    "Period: Period",
    "Scale: 1",
    "Source: filing",
)


def _view(*, drop: str | None = None, **overrides: Any) -> dict[str, Any]:
    """One evidence item, projected exactly as production projects it.

    Built through ``EvidencePacketV1.from_mapping().to_dict()`` and the real
    disclosure profile rather than as a bare dict, so the renderer is exercised
    on the shape it actually receives -- and so a test that forgot the profile
    would fail rather than pass for the wrong reason.
    """

    from rag_v2.adaptive.adaptive_contracts import EvidencePacketV1

    authored = {key: value for key, value in FULL.items() if key != drop}
    authored.update(overrides)
    packet = EvidencePacketV1.from_mapping(authored).to_dict()
    return project(packet, profile=EvidenceDisclosureProfile.SPECIALIST)


def _pack(items: list[dict[str, Any]], **kwargs: Any) -> Any:
    """Compile a pack from already-projected views.

    H2A-3B3 made the renderer take a pack, so a test that wants to render a
    particular projection compiles one.  The compiler re-projects what it is
    handed, which on an already-projected SPECIALIST view is the identity -- the
    production projection applied twice, not a second one.
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
            query=kwargs.pop("query", "What was revenue?"),
            admitted_evidence=tuple(items),
            calculation=kwargs.pop("calculation", None),
        )
    )


def _rendered(*items: dict[str, Any]) -> str:
    return render_specialist_prompt(_pack(list(items)))


def _block(prompt: str, handle: str = "E1") -> dict[str, str]:
    """The named ``[E#]`` block, as label -> rendered text.

    Parsed from the prompt text, so what is asserted is what a model reads.
    """

    found: dict[str, str] | None = None
    current: dict[str, str] | None = None
    for line in prompt.splitlines():
        if line.startswith("[") and line.endswith("]"):
            current = {} if line == f"[{handle}]" else None
            if current is not None:
                found = current
            continue
        if current is None or ": " not in line:
            continue
        label, _, text = line.partition(": ")
        current[label] = text
    assert found is not None, f"no [{handle}] block in the prompt"
    return found


# --- the field, one at a time ------------------------------------------------------


@pytest.mark.parametrize("field", REMOVABLE)
def test_a_field_the_evidence_lacks_is_stated_as_absent(field: str) -> None:
    """Drop one field, and only its own slot changes.

    Parametrized over every field rather than checked on one, because the defect
    was per-field: the fix for ``scale`` says nothing about ``scope``, and a
    renderer can acquire a new invented default in a field nobody re-checked.
    """

    block = _block(_rendered(_view(drop=field)))

    if field in LABEL_OF:
        assert block[LABEL_OF[field]] == ABSENT, field
        return

    # ``document_id`` and ``page`` share a line, and each is stated on its own.
    parts = block["Source"].split(":")
    index = 0 if field == "document_id" else 1
    assert parts[index] == ABSENT, (field, block["Source"])
    assert parts[1 - index] != ABSENT, (field, block["Source"])


@pytest.mark.parametrize("field", [f for f in REMOVABLE if f in LABEL_OF])
def test_a_field_the_evidence_has_arrives_unchanged(field: str) -> None:
    """The other direction: a recorded value is never replaced by the marker.

    Asserted beside the absence case so that "always print the marker" -- which
    would pass every absence test -- fails here instead.
    """

    block = _block(_rendered(_view()))

    assert block[LABEL_OF[field]] == str(FULL[field]), field


def test_every_line_of_a_full_item_is_the_authored_value() -> None:
    """The whole block at once, against the fixture rather than a snapshot."""

    block = _block(_rendered(_view()))

    for field, label in LABEL_OF.items():
        assert block[label] == str(FULL[field]), field
    assert block["Source"] == f"{FULL['document_id']}:{FULL['page']}"


# --- the fabrications, named --------------------------------------------------------


@pytest.mark.parametrize("literal", FABRICATED_FORMS)
def test_no_fabricated_form_survives_in_a_prompt(literal: str) -> None:
    """Each removed default, written the way the renderer used to write it.

    Redundant with the total assertions above and kept for the same reason the
    baseline keeps its own copy: when one of these fails, the failure should name
    the defect rather than a field offset.
    """

    # An item with nothing but a distinctive metric and value: every other slot
    # is absent, so every fallback that ever existed would fire here.
    bare = _view(
        drop="period", scope=None, unit=None, currency=None, scale=None,
        document_id=None, page=None,
    )
    assert literal not in _rendered(bare), literal

    # And the fully-populated item must not contain them either -- the check has
    # to hold for a prompt where every fallback would have been skipped, or the
    # test would only be proving that one particular input triggers the bug.
    assert literal not in _rendered(_view()), literal


def test_scope_is_never_answered_with_the_metric() -> None:
    """The one that invented a fact rather than a placeholder.

    ``scope or metric`` asserted that the scope *is* the metric.  A model reading
    ``Scope: Revenue`` beside ``Metric: Revenue`` has been told something about
    the world, and unlike ``not specified`` there is nothing in the line that
    marks it as an absence.

    The null case is the one that matters, and it is the reason the assertion is
    written as an inequality against the metric rather than against a literal.
    """

    bare = _view(scope=None)
    block = _block(_rendered(bare))

    assert block["Scope"] == ABSENT
    assert block["Scope"] != block["Metric"]
    assert block["Metric"] == FULL["metric"], "the metric itself must still cross"


@pytest.mark.parametrize("metric", ["Revenue", "Metric", "Period", "not specified"])
def test_no_field_echoes_a_neighbouring_value(metric: str) -> None:
    """Every absent slot is absent, whatever distinctive value sits beside it.

    ``Metric`` and ``Period`` are in the list because they are what the old
    fallbacks *printed*, and a metric genuinely named either one must still not
    make an absent period or scope look answered.
    """

    block = _block(
        _rendered(_view(metric=metric, period=None, scope=None))
    )

    assert block["Metric"] == metric
    assert block["Period"] == ABSENT, metric
    assert block["Scope"] == ABSENT, metric


# --- the shape of the block ---------------------------------------------------------


def test_an_item_with_one_value_still_renders_the_whole_schema() -> None:
    """Absence is stated, not omitted: the line set does not depend on the data.

    This is the alternative the audit rejected, made checkable.  If the renderer
    were to omit absent lines, a reader could no longer tell a field that was
    absent from a renderer version that stopped emitting it -- and answer rule 6
    asks the model to notice missing evidence.
    """

    block = _block(_rendered(_view(
        drop="period", scope=None, unit=None, currency=None, scale=None,
        document_id=None, page=None,
    )))

    assert set(block) == set(LABEL_OF.values()) | {"Source"}
    assert block["Value"] == "391"
    assert block["Metric"] == "Revenue"
    assert block["Source"] == ABSENT


@pytest.mark.parametrize(
    ("document", "page", "expected"),
    [
        ("doc-1", 7, "doc-1:7"),
        ("doc-1", None, "doc-1:not specified"),
        (None, 7, "not specified:7"),
        (None, None, "not specified"),
    ],
)
def test_the_source_line_states_what_is_known_about_each_component(
    document: str | None, page: int | None, expected: str
) -> None:
    """Two components, each stated on its own -- no component is invented.

    ``Source: not specified:7`` reads as "an unknown document, page 7", which is
    exactly what is known.  The single-marker case is what the canonical renderer
    emits when it has no source at all.
    """

    block = _block(_rendered(_view(document_id=document, page=page)))

    assert block["Source"] == expected


def test_page_zero_is_still_a_page() -> None:
    """The F5 property, re-asserted where F10 could have broken it.

    A truthiness test cannot tell a recorded page ``0`` from an absent one, and
    the F10 fix replaced every truthiness test in this renderer.
    """

    block = _block(_rendered(_view(page=0)))

    assert block["Source"] == "doc-1:0"
    assert block["Source"] != f"doc-1:{ABSENT}"


def test_an_empty_string_is_absence_not_a_blank_value() -> None:
    """``Value: `` with nothing after it is neither a value nor a statement.

    The disclosure projection drops ``None`` outright, so this shape does not
    occur in production -- but a producer that emitted ``""`` would otherwise
    render a blank where the schema has a slot, and the difference between "the
    evidence states the empty string" and "the evidence states nothing" is
    exactly the kind this phase is about.
    """

    from rag_v2.adaptive.adaptive_contracts import EvidencePacketV1

    raw = EvidencePacketV1.from_mapping({**FULL, "value": ""}).to_dict()
    view = project(raw, profile=EvidenceDisclosureProfile.SPECIALIST)
    block = _block(_rendered(view))

    assert block["Value"] == ABSENT


# --- the calculation block is not part of this ---------------------------------------


def test_the_calculation_block_is_unchanged_by_f10() -> None:
    """F10 was evidence-only, and this is what says so.

    The calculation renderer reads three fields from an already-projected
    payload, and none of them had a fabricated default.
    """

    prompt = render_specialist_prompt(
        _pack(
            [_view()],
            query="Why did revenue change?",
            calculation={"operation": "difference", "value": "8", "unit": "USD"},
        )
    )

    assert "[VERIFIED CALCULATION]" in prompt
    assert "Operation: difference" in prompt
    assert "Value: 8 USD" in prompt


# --- through the real production boundary ---------------------------------------------


class _RenderingSpecialist:
    """A specialist that records the prompt the boundary rendered for it.

    H2A-3B3.  It used to render the prompt itself; the prompt now arrives
    already rendered from the compiled pack, so this records the production
    model input rather than a second rendering of it.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "rendered"


def _sparse_state():
    """A state the routing policy sends to the specialist, missing most fields.

    Two facts differing in period, because MULTI reaches the specialist only when
    the items are distinct canonical facts, and neither carrying a scope, unit,
    currency, scale or page -- so every removed default would have fired.
    """

    from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1

    def packet(evidence_id: str, period: str) -> dict[str, Any]:
        return {
            "evidence_id": evidence_id,
            "citation_id": f"citation:{evidence_id}",
            "metric": "Revenue",
            "value": "391",
            "period": period,
            "document_id": "doc-1",
        }

    state = AdaptiveRAGStateV1.new("r", "Compare revenue year-over-year.")
    state.add_evidence(
        [
            EvidencePacketV1.from_mapping(packet("e1", "FY2024")),
            EvidencePacketV1.from_mapping(packet("e2", "FY2023")),
        ]
    )
    state.bound_evidence_ids = ["e1", "e2"]
    return state


def test_the_production_boundary_renders_no_fabricated_default() -> None:
    """The fix asserted through the wiring, not through the helper.

    A correct renderer says nothing about what actually reaches a model.  This
    drives ``TrustedV2GenerationCapability``, which is what selects and projects
    the payload that crosses the boundary.
    """

    from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability

    specialist = _RenderingSpecialist()
    TrustedV2GenerationCapability(specialist=specialist).generate(_sparse_state())

    assert specialist.prompts, "the specialist was never called"
    prompt = specialist.prompts[0]

    for literal in FABRICATED_FORMS + ("Scope: Revenue",):
        assert literal not in prompt, literal

    # The authored values did cross, so this is not passing by rendering nothing.
    assert "Metric: Revenue" in prompt
    assert "Period: FY2024" in prompt
    assert "Source: doc-1:not specified" in prompt
    # Two items, two honest markers in the slots nobody filled.
    assert prompt.count("Scope: not specified") == 2
