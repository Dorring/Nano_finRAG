"""H2A-3B2: the B3 legacy context baseline, and the record it superseded.

Four different kinds of assertion live here, and they are deliberately not the
same kind:

1. **Equivalence** -- the live path reproduces the frozen v2 values.  This is a
   change detector.  It says "nothing moved", not "this is correct".
2. **Contract properties, checked against the frozen record itself** -- every
   rendered line must be the authored value or a stated absence.  These are
   authored expectations, so they still hold if the code and the snapshot were
   regenerated together -- the one failure mode a pure snapshot comparison
   cannot catch.
3. **Reachability** -- every scenario must still reach the specialist.  Without
   this, a routing change that sent a scenario down the deterministic renderer
   would be recorded as a new shape rather than reported as a moved boundary.
4. **History** -- ``BASELINE_V1``, the pre-F10 record, must still show the
   defect it is kept to document, and the two records must differ *only* where
   a fabricated default was being rendered.

Kind 2 is the half that keeps this file from being the tautology H2A-2's
anti-circularity rule forbids.  A differential that only compared live output to
recorded output would still pass if someone re-captured the baseline after
breaking the page handling -- which is exactly what kind 1 did after H2A-3B0
changed the renderer, and why kind 2 has to be written from the inputs.

Kind 4 is new in H2A-3B2 and exists because a migration needs a target that does
not contain the defect being removed.  ``BASELINE_V3`` is that target; the v1
record is retained as evidence, and its own test fails if it is tidied up to
match.
"""

from __future__ import annotations

import hashlib
import re

import pytest

from tests.harness.b3_legacy_context_baseline import (
    BASELINE_V3,
    NOT_DETERMINED,
    SCENARIOS,
    observe,
)
from tests.harness.b3_legacy_context_baseline_v1 import BASELINE_V1
#: The record F10 produced, archived when the citation fix moved the live
#: baseline again.  The F10 blast-radius comparison below must run against
#: *this* one: measuring it against the current baseline would put two
#: independent changes into one comparison and it would stop answering the
#: question it asks.
from tests.harness.b3_legacy_context_baseline_v2 import BASELINE_V2 as BASELINE_V2_F10

SCENARIO_NAMES = sorted(SCENARIOS)

#: Fields that must never appear on the specialist's model-facing surface.
FORBIDDEN_IN_MODEL_INPUT = ("source_text", "metadata", "pdf_page", "content")

#: How the renderer states a field the evidence does not carry.  Spelled out
#: here rather than imported from the renderer: an expectation read out of the
#: code under test is not an expectation, and importing the constant would make
#: "absence is stated honestly" true by definition.
ABSENT = "not specified"

#: Prompt label -> the authored evidence field it renders.  Every line of an
#: evidence block is accounted for, so a field cannot be added to the prompt and
#: escape the property below by not being listed.
RENDERED_FIELDS = {
    "Metric": "metric",
    "Period": "period",
    "Scope": "scope",
    "Value": "value",
    "Unit": "unit",
    "Currency": "currency",
    "Scale": "scale",
}

#: The defaults F10 removed, written as the renderer spelled them.  Each one
#: asserted a value the evidence did not carry.
FABRICATED_LITERALS = (
    "Metric: Metric",
    "Period: Period",
    "Scale: 1",
    "Source: filing",
    "Scope: Revenue",
)

#: The sources rendered by the ``page_variants`` scenario, in bound order.  The
#: middle one is page ``0`` and the last has no page at all -- the pair the old
#: ``ev.get("page") or 1`` conflated.
PAGE_VARIANT_SOURCES = [
    "Source: doc-1:7",
    "Source: doc-1:0",
    "Source: doc-1:not specified",
]


#: The one bracketed header that opens an evidence block.
_EVIDENCE_HEADER = re.compile(r"^\[E\d+\]$")


def _rendered(value: object) -> str:
    return ABSENT if value is None else str(value)


def _blocks(prompt: str) -> list[dict[str, str]]:
    """Each ``[E#]`` block of a rendered prompt, as label -> rendered text.

    Written from the prompt text alone.  Reading the expected values out of the
    renderer instead would make the property below unfalsifiable.

    A block ends at the next bracketed header -- which is what keeps the
    calculation's ``Value`` line from overwriting the last evidence item's, the
    bug this parser had on its first run.
    """

    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in prompt.splitlines():
        if line.startswith("[") and line.endswith("]"):
            current = {} if _EVIDENCE_HEADER.match(line) else None
            if current is not None:
                blocks.append(current)
            continue
        if current is None or ": " not in line:
            continue
        label, _, text = line.partition(": ")
        current[label] = text
    return blocks


def test_every_scenario_is_frozen() -> None:
    """A scenario added without a capture would silently assert nothing."""

    assert SCENARIO_NAMES == sorted(BASELINE_V3)
    assert SCENARIO_NAMES, "the baseline must not be empty"
    assert SCENARIO_NAMES == sorted(BASELINE_V1)


# --- 1. equivalence --------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_the_live_context_path_has_not_moved(scenario: str) -> None:
    """The differential H2A-3B3 must reproduce exactly.

    Reported per key, and the prompt by digest, so a failure names what moved
    instead of printing two dicts.
    """

    live = observe(scenario)
    frozen = BASELINE_V3[scenario]

    assert set(live) == set(frozen), "the recorded shape itself changed"

    differences = [
        key for key in frozen
        if key != "prompt" and live[key] != frozen[key]
    ]
    assert not differences, {
        key: {"frozen": frozen[key], "live": live[key]} for key in differences
    }

    digest = hashlib.sha256(live["prompt"].encode("utf-8")).hexdigest()
    assert digest == frozen["prompt_sha256"], (
        "the model input changed but no recorded field did.\n"
        f"frozen digest: {frozen['prompt_sha256']}\n"
        f"live digest:   {digest}\n"
        f"--- frozen ---\n{frozen['prompt']}\n"
        f"--- live ---\n{live['prompt']}"
    )


# --- 3. reachability -------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_every_scenario_still_reaches_the_specialist(scenario: str) -> None:
    """The baseline only means something if B3 is what it exercises."""

    assert BASELINE_V3[scenario]["route_target"] == "LOCAL_SPECIALIST"


# --- 2. contract properties, held against the frozen artifact --------------------


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_no_forbidden_field_reaches_the_frozen_model_input(scenario: str) -> None:
    """Deny by default, asserted against the recorded model input.

    Checked on the frozen record rather than the live one on purpose: it has to
    hold for the artifact H2A-3B3 compares against, not only for today's run.
    """

    frozen = BASELINE_V3[scenario]

    for name in FORBIDDEN_IN_MODEL_INPUT:
        assert name not in frozen["disclosed_fields"], (scenario, name)

    assert frozen["projected_evidence"], scenario
    for item in frozen["projected_evidence"]:
        assert all(name not in item for name in FORBIDDEN_IN_MODEL_INPUT), (
            scenario, item,
        )
        # The profile governs no container, so nothing structured may cross.
        assert not any(
            isinstance(value, (dict, list)) for value in item.values()
        ), (scenario, item)

    assert "RAW SOURCE TEXT" not in frozen["prompt"], scenario


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_every_rendered_field_is_the_authored_value_or_a_stated_absence(
    scenario: str,
) -> None:
    """The F10 property, stated over the whole evidence block.

    Total on purpose, and derived from the inputs rather than from the renderer.
    A spot check against one known-bad literal would only catch the sample the
    author thought of -- and the defect this replaces was five separate
    fabrications, each in its own field, which is what a spot check misses.

    The two failure directions are different and both are caught here: a field
    the evidence carries must arrive *unchanged* (F5's dropped zero), and a
    field it does not carry must arrive as the absence marker and not as a
    value (F10's invented one).
    """

    authored = SCENARIOS[scenario]["evidence"]
    blocks = _blocks(BASELINE_V3[scenario]["prompt"])

    assert len(blocks) == len(authored), scenario

    for item, block in zip(authored, blocks):
        for label, field in RENDERED_FIELDS.items():
            # ``normalized_metric`` is the one declared alternative the renderer
            # reads; no authored scenario uses it, and the assertion is written
            # so that a scenario which did would be checked against it.
            value = item.get(field)
            if field == "metric" and value is None:
                value = item.get("normalized_metric")
            assert block[label] == _rendered(value), (scenario, field, block[label])

        document = item.get("document_id")
        page = item.get("page")
        expected_source = (
            ABSENT
            if document is None and page is None
            else f"{_rendered(document)}:{_rendered(page)}"
        )
        assert block["Source"] == expected_source, (scenario, item["evidence_id"])


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_the_frozen_model_input_carries_no_fabricated_default(scenario: str) -> None:
    """The named literals, asserted absent from the artifact that will be compared.

    Redundant with the total property above, and kept because a failure should
    name the defect rather than a field offset.  The two disagree only if the
    prompt grows a line neither test accounts for, which is itself worth a red.
    """

    prompt = BASELINE_V3[scenario]["prompt"]

    for literal in FABRICATED_LITERALS:
        assert literal not in prompt, (scenario, literal)


def test_the_page_variants_survive_in_the_frozen_record() -> None:
    """Page 7, page 0 and no page are three different things in the model input.

    This is the authored expectation for the shape H2A-3B0 fixed.  ``or 1``
    collapsed all three onto ``1``; a capture cannot be re-taken that way and
    still satisfy this.
    """

    source_lines = [
        line
        for line in BASELINE_V3["page_variants"]["prompt"].splitlines()
        if line.startswith("Source:")
    ]

    assert source_lines == PAGE_VARIANT_SOURCES


def test_the_frozen_token_count_is_recorded_as_undetermined() -> None:
    """B3's tokenizer is the checkpoint's, which the suite cannot reach.

    A number here would be an approximation presented as a measurement, which is
    what H2A-3A's tokenizer matrix exists to prevent.  When the canonical
    environment fills it in, this test forces the change to be a deliberate edit
    rather than a quiet one.
    """

    for scenario in SCENARIO_NAMES:
        assert BASELINE_V3[scenario]["input_tokens"] == NOT_DETERMINED, scenario


def test_the_calculation_scenario_carries_a_governed_projection() -> None:
    """The one scenario with a calculation shows exactly the permitted fields.

    Frozen here because H2A-3B3 moves the calculation payload through the
    compiler too, and "the calculation disclosure did not regress" is one of its
    stated gates.  The projection is three fields, and the id is the result's --
    not a second copy of it.
    """

    frozen = BASELINE_V3["calculation_with_explanation"]

    assert frozen["projected_calculation"] == {
        "operation": "difference",
        "unit": "USD",
        "value": "8",
    }
    assert frozen["calculation_ids"] == ["C1-1965a2ad00ce9277"]

    rendered = frozen["prompt"]
    assert "[VERIFIED CALCULATION]" in rendered
    assert "Operation: difference" in rendered
    assert "Value: 8 USD" in rendered


# --- 4. the superseded record ----------------------------------------------------


def test_v1_still_records_the_defect_it_is_kept_to_document() -> None:
    """History is only evidence while it still shows what happened.

    v1 is the only artifact that testifies F10 was real -- that the shipped
    renderer stated a scope, a scale and a metric nobody had established.  If
    someone edits it to agree with v2, that claim stops being checkable, and the
    change reads as tidying rather than as a correction.
    """

    page_variants = BASELINE_V1["page_variants"]["prompt"]
    assert "Scope: Revenue" in page_variants
    assert "Scope: Cost of revenue" in page_variants
    assert "Scope: Operating income" in page_variants
    assert "Scale: 1" in page_variants
    assert "Unit: not specified" in page_variants

    # The same prompt, post-F10: the honest marker is still there, and the two
    # fabrications beside it are gone.
    fixed = BASELINE_V3["page_variants"]["prompt"]
    assert "Unit: not specified" in fixed
    assert "Scope: Revenue" not in fixed
    assert "Scale: 1" not in fixed


def test_the_two_records_differ_only_where_a_default_was_fabricated() -> None:
    """The blast radius of the F10 fix, asserted rather than described.

    Four scenarios moved and one did not.  ``multi_fact`` is the scenario whose
    evidence carries every field, so no fallback ever fired for it -- which is
    the sharpest available statement that the change was the *removal of
    defaults* and not a rewrite of the renderer.  A diff anywhere else, or a
    diff in ``multi_fact``, means something else moved.
    """

    untouched = {"multi_fact"}
    assert untouched <= set(SCENARIO_NAMES), "the control scenario is missing"

    for scenario in SCENARIO_NAMES:
        v1, v2 = BASELINE_V1[scenario], BASELINE_V2_F10[scenario]
        assert set(v1) == set(v2), scenario

        changed = {key for key in v1 if v1[key] != v2[key]}
        expected = set() if scenario in untouched else {"prompt", "prompt_sha256"}

        assert changed == expected, (scenario, sorted(changed))

    # Every recorded digest is the digest of the prompt beside it, in both
    # records -- so a hand-edited prompt cannot leave a stale digest behind.
    for record in (BASELINE_V1, BASELINE_V2_F10):
        for scenario, frozen in record.items():
            assert (
                hashlib.sha256(frozen["prompt"].encode("utf-8")).hexdigest()
                == frozen["prompt_sha256"]
            ), scenario
