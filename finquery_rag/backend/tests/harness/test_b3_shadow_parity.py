"""H2A-3B1-B/A-3B3: what the compiled pack agrees with, and that B3 now uses it.

The compiler was built in shadow and has since taken over the boundary.  This
file still answers the question it was written for -- does the compiled pack
carry what the legacy path carried? -- because that is the equivalence half of
H2A-3B3's differential.  What changed is who is being compared: the pack is now
compiled by the *production* path rather than beside it, and the structural
guards below were inverted accordingly.

Three kinds of assertion, kept apart on purpose:

1. **Fixture-derived expectations.**  What the compiled pack must contain is
   derived from the authored scenario inputs -- the evidence ids, the pages, the
   calculation -- never from the compiler.  This is the assertion that would
   survive a re-capture of every recorded artifact.
2. **Parity with the frozen legacy projection.**  The baseline's
   ``projected_evidence`` and ``projected_calculation`` are what the legacy path
   actually handed the boundary; the compiled pack must carry the same.  This is
   equivalence, and equivalence is not correctness.
3. **Structural guarantees.**  The production path runs on the compiler, and no
   whole RunState reaches a pack.

The *correctness* half lives in ``test_specialist_truthfulness``,
``test_b3_support_topology`` and ``test_b3_compiler_migration``.  A differential
against a captured baseline cannot supply it: the capture and the implementation
could be regenerated from each other and this file would still be green.

F10 was fixed in H2A-3B2, and note where it lived: the fabricated ``scale``,
``document_id``, ``metric``, ``period`` and ``scope`` defaults were in the
**renderer**, and the compiler does not touch the renderer.  Both the legacy
projection and the compiled pack carried absence as absence throughout, so the
fix moved no pack field -- which is why the parity assertions below are
unchanged by it and why the baseline this file compares against is ``V2``.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from rag_v2.context import (
    AgentContextPackV1,
    ContextCompilerV1,
    ContextRoleV1,
    SpecialistContextPolicyV1,
    specialist_context_request,
)
from rag_v2.evidence.disclosure import (
    EvidenceDisclosureProfile,
    allowed_fields,
)
from tests.harness.b3_legacy_context_baseline import (
    BASELINE_V3,
    SCENARIOS,
    build_state,
    observe,
)

SCENARIO_NAMES = sorted(SCENARIOS)

FORBIDDEN_IN_PACK = ("source_text", "content", "metadata", "temporal")

PROFILE_FIELDS = set(allowed_fields(EvidenceDisclosureProfile.SPECIALIST))

CALCULATION_FIELDS = {"operation", "unit", "value"}


def _compiled(scenario: str) -> AgentContextPackV1:
    """Compile a pack in shadow mode.  Never handed to a model."""

    compiler = ContextCompilerV1(SpecialistContextPolicyV1())
    return compiler.compile(specialist_context_request(build_state(scenario)))


# --- 1. fixture-derived expectations ----------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_the_pack_carries_exactly_the_authored_evidence(scenario: str) -> None:
    """What goes in is what was authored -- not what the compiler returned."""

    authored = SCENARIOS[scenario]["evidence"]
    pack = _compiled(scenario)

    assert pack.evidence_ids == tuple(item["evidence_id"] for item in authored)
    assert pack.budget.evidence_considered == len(authored)
    assert pack.budget.evidence_selected == len(authored)


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_every_disclosed_field_matches_the_authored_input(scenario: str) -> None:
    """Item by item, against the fixture rather than against a snapshot.

    Every authored field on this profile must arrive unchanged, and the
    authoritative page must be the authored one -- 7 stays 7, 0 stays 0, and an
    absent page stays absent rather than becoming any default at all.
    """

    authored = SCENARIOS[scenario]["evidence"]
    pack = _compiled(scenario)

    assert len(pack.evidence) == len(authored)
    for expected, actual in zip(authored, pack.evidence):
        for field in ("evidence_id", "metric", "value", "period", "document_id"):
            if field in expected:
                assert actual.get(field) == expected[field], (scenario, field)
        if expected.get("page") is None:
            assert "page" not in actual, (scenario, expected["evidence_id"])
        else:
            assert actual["page"] == expected["page"], (scenario, expected["evidence_id"])


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_the_pack_holds_nothing_outside_the_specialist_profile(scenario: str) -> None:
    """Deny by default, asserted on the compiled artifact."""

    pack = _compiled(scenario)

    for item in pack.evidence:
        assert set(item) <= PROFILE_FIELDS, (scenario, sorted(set(item) - PROFILE_FIELDS))
        for forbidden in FORBIDDEN_IN_PACK:
            assert forbidden not in item, (scenario, forbidden)


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_the_model_visible_handles_are_the_ones_the_renderer_emits(
    scenario: str,
) -> None:
    """``E1..En`` positionally, plus ``C1`` when there is a calculation.

    Derived from the authored evidence count, not read back from the compiler.
    These handles are what the model cites and what a validator resolves, so
    they have to agree with the prompt that will carry the pack.
    """

    authored = SCENARIOS[scenario]["evidence"]
    pack = _compiled(scenario)

    expected_handles = [f"E{index}" for index in range(1, len(authored) + 1)]
    if SCENARIOS[scenario]["calculation"] is not None:
        expected_handles.append("C1")

    assert [reference.handle for reference in pack.references.handles] == expected_handles
    assert [r.evidence_id for r in pack.references.handles][: len(authored)] == [
        item["evidence_id"] for item in authored
    ]


def test_the_calculation_scenario_carries_only_the_permitted_fields() -> None:
    """Derived from the authored calculation, and bounded by the contract."""

    authored = SCENARIOS["calculation_with_explanation"]["calculation"]
    assert authored is not None
    pack = _compiled("calculation_with_explanation")

    assert pack.calculation is not None
    assert set(pack.calculation) == CALCULATION_FIELDS
    assert pack.calculation["operation"] == authored["operation"]
    assert pack.calculation["value"] == authored["value"]
    assert pack.calculation["unit"] == authored["unit"]


def test_a_scenario_without_a_calculation_carries_none() -> None:
    for scenario in SCENARIO_NAMES:
        if SCENARIOS[scenario]["calculation"] is None:
            assert _compiled(scenario).calculation is None, scenario


# --- 2. parity with the frozen legacy projection ---------------------------------------


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_the_compiled_pack_matches_the_legacy_projection(scenario: str) -> None:
    """Class A parity: same evidence, same fields, same values, same order.

    The legacy side is the frozen record of what the boundary was actually
    handed.  This is an equivalence claim -- it says the compiler introduced no
    change, not that either side is right.
    """

    legacy = BASELINE_V3[scenario]
    pack = _compiled(scenario)

    assert [dict(item) for item in pack.evidence] == [
        dict(item) for item in legacy["projected_evidence"]
    ]
    assert (
        None if pack.calculation is None else dict(pack.calculation)
    ) == legacy["projected_calculation"]
    assert list(pack.evidence_ids) == list(legacy["bound_evidence_ids"])


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_the_disclosed_field_set_agrees_with_the_legacy_trace(scenario: str) -> None:
    """The same fields crossed, in the same namespaces.

    Compared as sets: the legacy trace lists evidence fields then calculation
    fields, which is a reporting order, not a claim about the pack.
    """

    legacy = set(BASELINE_V3[scenario]["disclosed_fields"])
    pack = _compiled(scenario)

    compiled = {
        f"evidence.{field}" for item in pack.evidence for field in item
    } | (
        set()
        if pack.calculation is None
        else {f"calculation.{field}" for field in pack.calculation}
    )

    assert compiled == legacy, (scenario, sorted(compiled ^ legacy))


def test_the_shadow_path_did_not_move_the_legacy_path() -> None:
    """The baseline still reproduces -- building the framework changed nothing.

    This is the question the frozen baseline exists to answer, and it is asked
    *after* the compiler was written, which is the only time it is informative.
    """

    for scenario in SCENARIO_NAMES:
        assert observe(scenario) == BASELINE_V3[scenario], scenario


# --- 3. structural guarantees -----------------------------------------------------------


def test_the_production_generation_path_now_uses_the_compiler() -> None:
    """H2A-3B1's anti-migration guard, inverted by H2A-3B3.

    This file used to assert that ``src/runtime/trusted_v2_generation.py``
    contained no reference to the compiler, so that opting B3 in early would go
    red.  B3 has now been opted in deliberately, and a guard that still asserted
    the old state would be a guard against the migration.

    It is not simply deleted, because the *reason* it existed has not gone away:
    the production path must run on the compiler, and a future edit that quietly
    reverted it should be visible.  The behavioural form of that claim -- one
    compile, one render, the renderer handed the object the compiler returned --
    lives in ``test_b3_compiler_migration``.  What is left here is the shape of
    the wiring, which is the thing that changed.
    """

    source = pathlib.Path(
        "src/runtime/trusted_v2_generation.py"
    ).read_text(encoding="utf-8")

    assert "rag_v2.context" in source
    assert "ContextCompilerV1" in source
    assert "specialist_context_request" in source
    assert "AgentContextPack" in source


def test_no_legacy_context_construction_remains_on_the_production_path() -> None:
    """The other half of the inversion: the old assembly is gone, not dormant.

    A migration that left the legacy ``select -> project -> assemble`` sequence
    in place beside the compiler would have two live context paths, and the
    differential would be proving something about whichever one happened to run.
    ``_routing_evidence_items`` survives, because the routing policy still needs admitted
    items with their authoritative fields, but it no longer selects -- it
    delegates to the adapter's one implementation.
    """

    from src.runtime.trusted_v2_generation import _routing_evidence_items

    import inspect

    source = inspect.getsource(_routing_evidence_items)
    assert "admitted_specialist_evidence" in source, (
        "_routing_evidence_items must delegate to the adapter's selection rather than "
        "reimplementing it"
    )
    assert "for raw in" not in source, "a second selection loop is back"


def test_a_pack_holds_no_run_state_and_no_undeclared_field() -> None:
    """A pack is a fixed, small shape -- not a narrowed RunState.

    Asserted as an exact field set: a future contributor adding
    "just one more" field has to change this test and say so.
    """

    assert set(AgentContextPackV1.__dataclass_fields__) == {
        "role",
        "invocation_id",
        "query",
        "evidence",
        "calculation",
        "references",
        "budget",
        "selection",
    }

    pack = _compiled("multi_fact")
    rendered = repr(pack)
    for absent in (
        "AdaptiveRAGStateV1",
        "evidence_packets",
        "tool_history",
        "replan_rounds",
        "transitions",
    ):
        assert absent not in rendered, absent


def test_a_pack_is_never_durable_state() -> None:
    """Two compilations of one state are equal but independent objects."""

    first = _compiled("multi_fact")
    second = _compiled("multi_fact")

    assert first == second
    assert first is not second
    # Equality is value equality over frozen projections, not shared identity.
    assert first.evidence[0] is not second.evidence[0]


# --- the one shape that must not reach B3 ------------------------------------------------


def test_multi_support_evidence_is_routed_away_from_the_specialist() -> None:
    """Two witnesses of one figure are one claim, and B3 must not restate it.

    The H2A-2C binding keeps every independent support of a canonical value, and
    the rendering rule it established is that the value is stated once with both
    citations.  A free-form generator cannot be relied on to do that, so the
    routing policy sends this shape to the deterministic renderer -- verified
    here against the real capability, not asserted from the policy's source.

    The consequence for the compiler is that B3 is never asked to compile this
    shape in production -- which is why the routing policy is re-verified here
    rather than assumed.  The pack can now represent the shape correctly if it
    ever is asked (asserted below), so the diversion is a routing decision and
    not a workaround for a representation the context layer could not express.
    """

    from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1
    from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability

    def _packet(evidence_id: str, page: int) -> dict[str, Any]:
        return {
            "evidence_id": evidence_id,
            "fact_id": evidence_id,
            "metric": "Revenue",
            "value": "391",
            "period": "FY2024",
            "scope": "consolidated",
            "unit": "USD",
            "document_id": "doc-1",
            "page": page,
        }

    state = AdaptiveRAGStateV1.new("r", "What was revenue?")
    state.add_evidence(
        [
            EvidencePacketV1.from_mapping(_packet("X1", 7)),
            EvidencePacketV1.from_mapping(_packet("X2", 9)),
        ]
    )
    state.bound_evidence_ids = ["X1", "X2"]

    capability = TrustedV2GenerationCapability(model_backend=_RecordingSpecialist())
    capability.generate(state)
    snapshot = capability.trace_snapshot()

    assert snapshot["renderer_invoked"] is True
    assert snapshot["specialist_invoked"] is False

    # B3's own trace says the same thing: no specialist call, no model context.
    assert snapshot["generation_route"] == "MULTI"


def test_the_pack_now_represents_one_claim_with_many_supports() -> None:
    """H2A-3B1 recorded this gap; H2A-3B2 closes it -- without a ninth field.

    Before this phase, a multi-support state compiled to two ordinary evidence
    items, and nothing in the pack distinguished "two witnesses of one figure"
    from "two different figures".  The topology now says which, and it says it
    where H2A-3B1 predicted: inside ``references``, because the pack's field set
    is part of its contract and growing it was not the fix.

    The assertion is driven through the *adapter*, so what is proved is that a
    message the runtime really carries -- ``bound_slot_bindings`` -- reaches the
    pack, not that a hand-built request would.
    """

    from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1

    state = AdaptiveRAGStateV1.new("r", "What was revenue?")
    state.add_evidence(
        [
            EvidencePacketV1.from_mapping(
                {"evidence_id": "X1", "metric": "Revenue", "value": "391",
                 "period": "FY2024", "document_id": "doc-1", "page": 7}
            ),
            EvidencePacketV1.from_mapping(
                {"evidence_id": "X2", "metric": "Revenue", "value": "391",
                 "period": "FY2024", "document_id": "doc-2", "page": 9}
            ),
        ]
    )
    state.bound_evidence_ids = ["X1", "X2"]
    state.bound_slot_bindings = {"revenue/FY2024": ["X1", "X2"]}

    pack = ContextCompilerV1(SpecialistContextPolicyV1()).compile(
        specialist_context_request(state)
    )

    # Both supports still cross as evidence, in order.
    assert pack.evidence_ids == ("X1", "X2")
    assert [item["value"] for item in pack.evidence] == ["391", "391"]

    # And the pack now says they are one claim rather than leaving it to a guess.
    assert len(pack.references.support_groups) == 1
    group = pack.references.support_groups[0]
    assert (group.handle, group.canonical_slot, group.supports) == (
        "G1", "revenue/FY2024", ("E1", "E2"),
    )

    # Still eight top-level fields: the topology went into ``references``.
    assert set(AgentContextPackV1.__dataclass_fields__) == {
        "role",
        "invocation_id",
        "query",
        "evidence",
        "calculation",
        "references",
        "budget",
        "selection",
    }


class _RecordingSpecialist:
    """Never reached in the assertion above; present so the capability is real."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return "recorded"
