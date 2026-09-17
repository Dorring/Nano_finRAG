"""H2A-3B0: the B3 legacy context baseline, held from the pre-compiler commit.

Three different kinds of assertion live here, and they are deliberately not the
same kind:

1. **Equivalence** -- the live path reproduces the frozen values.  This is a
   change detector.  It says "nothing moved", not "this is correct".
2. **Contract properties, checked against the frozen record itself** -- the
   recorded prompts must carry the true page, must not carry the fabricated one,
   and must not carry a forbidden field.  These are authored expectations, so
   they still hold if the code and the snapshot were regenerated together --
   the one failure mode a pure snapshot comparison cannot catch.
3. **Reachability** -- every scenario must still reach the specialist.  Without
   this, a routing change that sent a scenario down the deterministic renderer
   would be recorded as a new shape rather than reported as a moved boundary.

Kind 2 is the half that keeps this file from being the tautology H2A-2's
anti-circularity rule forbids.  A differential that only compared live output to
recorded output would still pass if someone re-captured the baseline after
breaking the page handling.
"""

from __future__ import annotations

import hashlib

import pytest

from tests.harness.b3_legacy_context_baseline import (
    BASELINE,
    NOT_DETERMINED,
    SCENARIOS,
    observe,
)

SCENARIO_NAMES = sorted(SCENARIOS)

#: Fields that must never appear on the specialist's model-facing surface.
FORBIDDEN_IN_MODEL_INPUT = ("source_text", "metadata", "pdf_page", "content")

#: The sources rendered by the ``page_variants`` scenario, in bound order.  The
#: middle one is page ``0`` and the last has no page at all -- the pair the old
#: ``ev.get("page") or 1`` conflated.
PAGE_VARIANT_SOURCES = [
    "Source: doc-1:7",
    "Source: doc-1:0",
    "Source: doc-1:not specified",
]


def test_every_scenario_is_frozen() -> None:
    """A scenario added without a capture would silently assert nothing."""

    assert SCENARIO_NAMES == sorted(BASELINE)
    assert SCENARIO_NAMES, "the baseline must not be empty"


# --- 1. equivalence --------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_the_live_context_path_has_not_moved(scenario: str) -> None:
    """H2A-3B2's differential: the compiler must reproduce this exactly.

    Reported per key, and the prompt by digest, so a failure names what moved
    instead of printing two dicts.
    """

    live = observe(scenario)
    frozen = BASELINE[scenario]

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

    assert BASELINE[scenario]["route_target"] == "LOCAL_SPECIALIST"


# --- 2. contract properties, held against the frozen artifact --------------------


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_no_forbidden_field_reaches_the_frozen_model_input(scenario: str) -> None:
    """Deny by default, asserted against the recorded model input.

    Checked on the frozen record rather than the live one on purpose: it has to
    hold for the artifact H2A-3B2 compares against, not only for today's run.
    """

    frozen = BASELINE[scenario]

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
def test_the_frozen_model_input_states_the_page_the_input_recorded(
    scenario: str,
) -> None:
    """Every source line's page comes from that item's authored page.

    The F5 regression asserted the way it should have been asserted all along:
    by deriving the expected source lines from the *inputs*, not from a known-bad
    constant.  Under the old ``or 1`` every line whose evidence recorded page
    ``0`` or no page read ``:1`` -- and a ``!= "Source: doc-1:1"`` check would
    only have caught that for documents actually named ``doc-1``, failing on the
    sample rather than on the property.

    Deriving from the inputs also makes this the one assertion in the module that
    is independent of both the implementation and the capture: it says what the
    model input *should* contain given what went in.
    """

    expected = [
        "Source: {}:{}".format(
            item.get("document_id") or "filing",
            "not specified" if item.get("page") is None else item["page"],
        )
        for item in SCENARIOS[scenario]["evidence"]
    ]

    actual = [
        line
        for line in BASELINE[scenario]["prompt"].splitlines()
        if line.startswith("Source:")
    ]

    assert actual == expected


def test_the_page_variants_survive_in_the_frozen_record() -> None:
    """Page 7, page 0 and no page are three different things in the model input.

    This is the authored expectation for the shape H2A-3B0 fixed.  ``or 1``
    collapsed all three onto ``1``; a capture cannot be re-taken that way and
    still satisfy this.
    """

    source_lines = [
        line
        for line in BASELINE["page_variants"]["prompt"].splitlines()
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
        assert BASELINE[scenario]["input_tokens"] == NOT_DETERMINED, scenario


def test_the_calculation_scenario_carries_a_governed_projection() -> None:
    """The one scenario with a calculation shows exactly the permitted fields.

    Frozen here because H2A-3B2 moves the calculation payload through the
    compiler too, and "the calculation disclosure did not regress" is one of its
    stated gates.  The projection is three fields, and the id is the result's --
    not a second copy of it.
    """

    frozen = BASELINE["calculation_with_explanation"]

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
