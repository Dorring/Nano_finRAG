"""H2A-3B3: the production B3 specialist path runs on the Context Runtime.

Before this phase the compiler was a shadow: H2A-3B1 built it and proved it
agreeable, and a test in ``test_b3_shadow_parity`` existed to fail the moment
anyone wired it into production early.  This file is that test's successor, and
it asserts the opposite thing:

    authoritative state -> specialist adapter -> ContextRequestV1
      -> ContextCompilerV1 -> AgentContextPackV1
      -> pack-based renderer -> the model boundary

Three kinds of assertion, and they are not interchangeable:

1. **Wiring, asserted by substitution.**  The compiler's ``compile`` and the
   renderer are replaced with recording wrappers, so what is proved is that the
   production call path really calls them -- once each, in that order, with the
   renderer receiving the very object the compiler returned.  A behavioural
   assertion of this kind is what §9 asks for in place of a source search: a
   test that greps for an import string passes when the import is unused.
2. **Topology from the authority.**  The four shapes the phase names, compiled
   through the production adapter, with the groups required to come from
   ``bound_slot_bindings`` rather than from anything the compiler worked out.
3. **The migration changed no prompt.**  The compiled context is byte-identical
   to the post-F10 baseline v2, and varying the topology alone does not move it.

The separate question -- is the *behaviour* right, independent of whether it
moved -- is answered by the fixture-derived tests in
``test_specialist_truthfulness`` and ``test_b3_support_topology``.  A differential
against a captured baseline cannot answer it, and would pass if the baseline and
the implementation were regenerated from each other.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest

from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1
from rag_v2.context import (
    AgentContextPackV1,
    ContextCompilerV1,
    ContextRequestV1,
    ContextRoleV1,
    SpecialistContextPolicyV1,
    specialist_context_request,
)
from tests.harness.b3_legacy_context_baseline import (
    BASELINE_V2,
    SCENARIOS,
    build_state,
)
from src.runtime import trusted_v2_generation as generation_module
from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability

#: Reaches B3 without the router turning it away: ``"QUALITATIVE" in route_hint``
#: is the one branch that sends a *single* admitted item to the specialist.
QUALITATIVE = "QUALITATIVE"

#: Reaches B3 with several items.  A query naming a temporal comparison takes the
#: TEMPORAL branch, which is checked before the MULTI branch -- so a state whose
#: items collapse to one canonical fact still reaches the specialist, which is
#: the only way the one-slot-two-supports shape is reachable at all.
TEMPORAL = "Compare revenue year-over-year."


def _packet(evidence_id: str, **fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "evidence_id": evidence_id,
        "citation_id": f"citation:{evidence_id}",
        "metric": "Revenue",
        "value": "391",
        "period": "FY2024",
        "scope": "consolidated",
        "unit": "USD",
        "document_id": "doc-1",
        "page": 7,
    }
    base.update(fields)
    return base


def _state(
    packets: list[dict[str, Any]],
    bindings: dict[str, list[str]],
    *,
    question: str = "Summarise the reported figures.",
    intent: str = "DIRECT_FACT",
) -> AdaptiveRAGStateV1:
    state = AdaptiveRAGStateV1.new("r", question, intent=intent)
    state.add_evidence([EvidencePacketV1.from_mapping(item) for item in packets])
    state.bound_evidence_ids = [item["evidence_id"] for item in packets]
    state.bound_slot_bindings = {slot: list(ids) for slot, ids in bindings.items()}
    return state


class _Backend:
    """Records the prompt and answers; the boundary is what is under test."""

    def __init__(self, answer: str = "recorded") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def _run(state: AdaptiveRAGStateV1, backend: _Backend | None = None):
    backend = backend or _Backend()
    capability = TrustedV2GenerationCapability(model_backend=backend)
    result = capability.generate(state)
    return capability, backend, result


def _groups(pack: AgentContextPackV1) -> list[tuple[str, str, tuple[str, ...]]]:
    return [
        (group.handle, group.canonical_slot, group.supports)
        for group in pack.references.support_groups
    ]


# --- 1. the wiring, asserted by substitution --------------------------------------------


def test_production_compiles_once_and_renders_once_per_specialist_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly one compile and one render, and the renderer gets the pack.

    The identity assertion is the load-bearing one.  A renderer that received a
    pack the compiler did not produce -- or a copy of one -- would still satisfy
    "a pack was passed", and the whole claim of the migration is that the text a
    model reads is the compiler's output and nothing else.
    """

    compiled: list[ContextRequestV1] = []
    rendered: list[Any] = []

    real_compile = ContextCompilerV1.compile
    real_render = generation_module.render_specialist_prompt

    def compile_spy(self: ContextCompilerV1, request: ContextRequestV1):
        compiled.append(request)
        return real_compile(self, request)

    def render_spy(pack: Any) -> str:
        rendered.append(pack)
        return real_render(pack)

    monkeypatch.setattr(ContextCompilerV1, "compile", compile_spy)
    monkeypatch.setattr(generation_module, "render_specialist_prompt", render_spy)

    capability, backend, _ = _run(_state([_packet("e1")], {"revenue/FY2024": ["e1"]},
                                         intent=QUALITATIVE))

    assert len(compiled) == 1, "the compiler ran a number of times other than once"
    assert len(rendered) == 1, "the renderer ran a number of times other than once"
    assert len(backend.prompts) == 1, "the model was called more than once"

    # The renderer is handed an AgentContextPackV1, and it is the one the
    # compiler returned -- not a reconstruction of it.
    assert isinstance(rendered[0], AgentContextPackV1)
    assert rendered[0] is capability.last_context_pack

    assert capability.trace_snapshot()["context_compile_count"] == 1
    assert capability.trace_snapshot()["context_render_count"] == 1


def test_the_request_comes_from_the_adapter_not_from_this_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compiler's input is what the adapter builds from the runtime state.

    Checked against the adapter's own output on the same state rather than
    against a hand-written expectation, because rebuilding it here would be a
    second adapter -- which is the thing this migration removed.
    """

    seen: list[ContextRequestV1] = []
    real_compile = ContextCompilerV1.compile

    def compile_spy(self: ContextCompilerV1, request: ContextRequestV1):
        seen.append(request)
        return real_compile(self, request)

    monkeypatch.setattr(ContextCompilerV1, "compile", compile_spy)

    state = _state(
        [_packet("e1"), _packet("e2", period="FY2023", page=0)],
        {"revenue/FY2024": ["e1"], "revenue/FY2023": ["e2"]},
        question=TEMPORAL,
    )

    _run(state)

    assert seen == [specialist_context_request(state)]


def test_the_model_boundary_receives_a_string_and_nothing_else() -> None:
    """The narrowed backend contract, asserted rather than described.

    This is the migration's disclosure guarantee in its strongest form: the
    backend is handed one string, so there is no argument through which anyone
    could pass evidence that bypassed the compiler.  The signature is checked
    rather than trusted because a future parameter would quietly restore the
    second route.
    """

    import inspect

    parameters = list(
        inspect.signature(_Backend.generate).parameters.values()
    )
    assert [parameter.name for parameter in parameters] == ["self", "prompt"]

    capability, backend, _ = _run(
        _state([_packet("e1")], {"revenue/FY2024": ["e1"]}, intent=QUALITATIVE)
    )

    assert isinstance(backend.prompts[0], str)
    assert "Metric: Revenue" in backend.prompts[0]


def test_a_deterministic_route_compiles_nothing() -> None:
    """One canonical fact with one source renders deterministically.

    No pack, no compile, no render: the compiler is the specialist boundary's
    context path, not a decoration every invocation passes through.
    """

    capability, backend, _ = _run(_state([_packet("e1")], {"revenue/FY2024": ["e1"]}))

    assert capability.trace_snapshot()["renderer_invoked"] is True
    assert capability.trace_snapshot()["specialist_invoked"] is False
    assert capability.last_context_pack is None
    assert capability.trace_snapshot()["context_compile_count"] == 0
    assert capability.trace_snapshot()["context_render_count"] == 0
    assert backend.prompts == []


def test_a_stale_pack_does_not_survive_into_the_next_invocation() -> None:
    """Lifetime is one invocation, enforced on the capability that holds it."""

    capability = TrustedV2GenerationCapability(model_backend=_Backend())
    capability.generate(_state([_packet("e1")], {"revenue/FY2024": ["e1"]},
                               intent=QUALITATIVE))
    assert capability.last_context_pack is not None

    capability.generate(_state([_packet("e1")], {"revenue/FY2024": ["e1"]}))
    assert capability.last_context_pack is None


# --- 2. topology, through the production adapter ------------------------------------------


def _adapter_groups(state: AdaptiveRAGStateV1) -> list[tuple[str, str, tuple[str, ...]]]:
    """Compile through the production adapter, without the capability.

    Used where the routing policy sends a shape to the deterministic renderer --
    which is a decision about which *generator* writes the answer, not about
    whether the context layer can represent the state.  §12 requires the
    topology proved for the shape either way, so the test says which route the
    shape takes rather than pretending it reaches the specialist.
    """

    pack = ContextCompilerV1(SpecialistContextPolicyV1()).compile(
        specialist_context_request(state)
    )
    return _groups(pack)


def test_one_slot_one_support_reaches_the_specialist_as_g1_e1() -> None:
    state = _state([_packet("e1")], {"revenue/FY2024": ["e1"]}, intent=QUALITATIVE)
    capability, _, _ = _run(state)

    assert capability.trace_snapshot()["specialist_invoked"] is True
    assert _groups(capability.last_context_pack) == [("G1", "revenue/FY2024", ("E1",))]


def test_one_slot_two_supports_is_reachable_and_carries_g1_e1_e2() -> None:
    """Two witnesses of one figure, compiled through the production adapter.

    The router sends this shape to the deterministic renderer when it is the
    whole state -- a free-form generator must not restate one value -- so the
    temporal query here is what makes it reach the specialist at all.  Either
    way the *topology* is the same, and the capability path is asserted first
    because it is the stronger claim.
    """

    packets = [_packet("e1"), _packet("e2", document_id="doc-2", page=9)]
    bindings = {"revenue/FY2024": ["e1", "e2"]}

    state = _state(packets, bindings, question=TEMPORAL)
    capability, _, _ = _run(state)

    assert capability.trace_snapshot()["specialist_invoked"] is True
    assert _groups(capability.last_context_pack) == [
        ("G1", "revenue/FY2024", ("E1", "E2"))
    ]

    # And the same state through the adapter alone puts the shape where the
    # router would send it, with identical topology.
    assert _adapter_groups(state) == [("G1", "revenue/FY2024", ("E1", "E2"))]


def test_two_slots_reach_the_specialist_as_g1_e1_and_g2_e2() -> None:
    state = _state(
        [_packet("e1"), _packet("e2", period="FY2023", page=0)],
        {"revenue/FY2024": ["e1"], "revenue/FY2023": ["e2"]},
    )
    capability, _, _ = _run(state)

    assert capability.trace_snapshot()["specialist_invoked"] is True
    assert _groups(capability.last_context_pack) == [
        ("G1", "revenue/FY2024", ("E1",)),
        ("G2", "revenue/FY2023", ("E2",)),
    ]


def test_a_mixed_state_reaches_the_specialist_as_g1_e1_e2_and_g2_e3() -> None:
    state = _state(
        [
            _packet("e1"),
            _packet("e2", document_id="doc-2", page=9),
            _packet("e3", metric="Cost of revenue", value="214", page=0),
        ],
        {"revenue/FY2024": ["e1", "e2"], "cost_of_revenue/FY2024": ["e3"]},
    )
    capability, _, _ = _run(state)

    assert capability.trace_snapshot()["specialist_invoked"] is True
    assert _groups(capability.last_context_pack) == [
        ("G1", "revenue/FY2024", ("E1", "E2")),
        ("G2", "cost_of_revenue/FY2024", ("E3",)),
    ]


def test_the_topology_comes_from_the_binding_and_is_not_inferred() -> None:
    """The same evidence, bound two ways, produces two different topologies.

    Nothing about the evidence differs -- same metric, same period, same value,
    same documents.  If any part of the production path derived groups from
    metric/period/value rather than reading the authority, these two would come
    out the same, and this is the cheapest way to notice that they did not.
    """

    packets = [_packet("e1"), _packet("e2")]
    query = TEMPORAL

    apart = _run(_state(packets, {"a": ["e1"], "b": ["e2"]}, question=query))[0]
    together = _run(_state(packets, {"a": ["e1", "e2"]}, question=query))[0]

    assert _groups(apart.last_context_pack) == [
        ("G1", "a", ("E1",)),
        ("G2", "b", ("E2",)),
    ]
    assert _groups(together.last_context_pack) == [("G1", "a", ("E1", "E2"))]

    # And the evidence crossing is identical in both, so the difference really
    # is the topology and not something else about the state.
    assert [dict(item) for item in apart.last_context_pack.evidence] == [
        dict(item) for item in together.last_context_pack.evidence
    ]


def test_no_binding_means_a_specialist_call_with_no_topology() -> None:
    """The migrated path reports silence honestly rather than inventing groups."""

    state = _state(
        [_packet("e1"), _packet("e2", period="FY2023", page=0)],
        {},
    )
    capability, _, _ = _run(state)

    assert capability.trace_snapshot()["specialist_invoked"] is True
    assert capability.last_context_pack.references.support_groups == ()
    assert capability.trace_snapshot()["context_support_group_count"] == 0


# --- 3. the prompt did not move ---------------------------------------------------------


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_migrated_path_reproduces_baseline_v2_exactly(scenario: str) -> None:
    """The change detector, run against the post-F10 capture.

    The expected values were captured from the hand-written legacy path before
    the compiler took over, so reproducing them here is a statement that the
    migration introduced no change.  It is *not* a statement that either side is
    correct -- ``test_specialist_truthfulness`` and ``test_b3_support_topology``
    carry that half, and neither substitutes for the other.
    """

    import hashlib

    from tests.harness.b3_legacy_context_baseline import RecordingSpecialist

    state = build_state(scenario)
    specialist = RecordingSpecialist()
    capability = TrustedV2GenerationCapability(model_backend=specialist)
    result = capability.generate(state)
    frozen = BASELINE_V2[scenario]

    assert result.route == frozen["route"]
    assert list(result.bound_evidence_ids) == frozen["bound_evidence_ids"]
    assert list(result.citation_ids) == frozen["citation_ids"]
    assert list(capability.trace_snapshot()["disclosed_fields"]) == frozen["disclosed_fields"]

    pack = capability.last_context_pack
    assert [dict(item) for item in pack.evidence] == frozen["projected_evidence"]
    assert (
        None if pack.calculation is None else dict(pack.calculation)
    ) == frozen["projected_calculation"]

    assert specialist.prompts[0] == frozen["prompt"]
    assert (
        hashlib.sha256(specialist.prompts[0].encode("utf-8")).hexdigest()
        == frozen["prompt_sha256"]
    )


def test_changing_only_the_topology_does_not_move_the_prompt() -> None:
    """§6, pinned as its own regression.

    The pack carries support groups and the current B3 renderer does not render
    them, so two packs differing *only* in topology must produce the same text.
    That is what keeps the separation between what the context layer represents
    and what this renderer chooses to present, and it is what will make a future
    renderer that does present it a deliberate, reviewable change rather than an
    accidental one.

    The slot keys are deliberately different, so a renderer that leaked the
    group identity into the prompt would fail here rather than pass by accident.
    """

    packets = [_packet("e1"), _packet("e2", period="FY2023", page=0)]

    apart = _run(_state(packets, {"revenue/FY2024": ["e1"], "revenue/FY2023": ["e2"]},
                        question=TEMPORAL))
    together = _run(_state(packets, {"everything/one-claim": ["e1", "e2"]},
                           question=TEMPORAL))

    assert apart[1].prompts[0] == together[1].prompts[0]

    # The two packs really do differ, so the equality above is not vacuous.
    assert _groups(apart[0].last_context_pack) != _groups(together[0].last_context_pack)

    # And nothing from the topology reached the text: not the group handles,
    # not the slot keys, and not the pack's own vocabulary.
    prompt = together[1].prompts[0]
    for absent in ("G1", "G2", "everything/one-claim", "support_group", "canonical_slot"):
        assert absent not in prompt, absent


def test_the_topology_is_carried_even_though_the_prompt_ignores_it() -> None:
    """§1: available to a future renderer without recompilation, and not removed."""

    state = _state(
        [_packet("e1"), _packet("e2", period="FY2023", page=0)],
        {"revenue/FY2024": ["e1"], "revenue/FY2023": ["e2"]},
    )
    capability, backend, _ = _run(state)

    pack = capability.last_context_pack
    assert [group.canonical_slot for group in pack.references.support_groups] == [
        "revenue/FY2024",
        "revenue/FY2023",
    ]
    assert "revenue/FY2024" not in backend.prompts[0]


# --- §13: no synthetic artifact, no restated value ------------------------------------------


def test_two_supports_do_not_become_one_synthetic_evidence_item() -> None:
    """The pack expresses instances and a relation between them, nothing more.

    A group is not evidence: it carries no value, no metric and no citation, and
    the two witnesses stay two items in the order the authority admits them.  A
    representative item synthesised from the group would be a second evidence
    artifact -- and a renderer reading it could not tell it from a real one.
    """

    state = _state(
        [_packet("e1"), _packet("e2", document_id="doc-2", page=9)],
        {"revenue/FY2024": ["e1", "e2"]},
        question=TEMPORAL,
    )
    capability, backend, _ = _run(state)
    pack = capability.last_context_pack

    assert pack.evidence_ids == ("e1", "e2")
    assert [item["value"] for item in pack.evidence] == ["391", "391"]

    group = pack.references.support_groups[0]
    assert set(group.__dataclass_fields__) == {"handle", "canonical_slot", "supports"}
    assert group.supports == ("E1", "E2")

    # The value is stated once per witness -- twice, because there are two
    # witnesses -- and the topology adds no third copy of it.
    assert backend.prompts[0].count("Value: 391") == 2


# --- §16: the trace records counts, never contents -------------------------------------------


def test_the_trace_records_counts_and_never_evidence_content() -> None:
    state = _state(
        [
            _packet("e1", value="391"),
            _packet("e2", metric="Cost of revenue", value="214", page=0),
        ],
        {"revenue/FY2024": ["e1"], "cost_of_revenue/FY2024": ["e2"]},
    )
    capability, _, _ = _run(state)
    trace = capability.trace_snapshot()

    assert trace["context_compile_count"] == 1
    assert trace["context_evidence_selected"] == 2
    assert trace["context_evidence_dropped"] == 0
    assert trace["context_support_group_count"] == 2

    rendered = repr(trace)
    for value in ("391", "214", "doc-1", "Cost of revenue"):
        assert value not in rendered, value


# --- §17: the failure modes the migration must not change -------------------------------------


def test_a_missing_binding_shape_is_handled_by_the_pack_contract() -> None:
    """A malformed binding raises at the adapter, not silently as "no topology".

    ``admitted_specialist_evidence`` already refuses a non-mapping evidence
    packet; the binding reader does the same, because a binding that cannot be
    read is a runtime defect and reporting "no topology" would hide it behind a
    shape the pack treats as normal.
    """

    state = _state([_packet("e1")], {"revenue/FY2024": ["e1"]}, intent=QUALITATIVE)
    state.bound_slot_bindings = "not a mapping"  # type: ignore[assignment]

    capability = TrustedV2GenerationCapability(model_backend=_Backend())
    with pytest.raises(TypeError):
        capability.generate(state)

    # The failure is before the model, and no half-compiled pack is left behind.
    assert capability.last_context_pack is None


def test_a_page_zero_and_a_missing_page_still_differ_after_migration() -> None:
    """The two shapes F5 was about, re-asserted end to end."""

    state = _state(
        [
            _packet("e1", page=0),
            _packet("e2", period="FY2023", page=None),
        ],
        {"revenue/FY2024": ["e1"], "revenue/FY2023": ["e2"]},
    )
    _, backend, _ = _run(state)
    prompt = backend.prompts[0]

    assert "Source: doc-1:0" in prompt
    assert "Source: doc-1:not specified" in prompt
    assert "Source: doc-1:1" not in prompt


def test_an_admissible_calculation_crosses_and_an_absent_one_does_not() -> None:
    """The calculation projection, through the migrated path."""

    from decimal import Decimal

    from src.domain.calculation import (
        CalculationOperation,
        CalculationResult,
        CalculationStatus,
    )

    packets = [_packet("e1"), _packet("e2", period="FY2023", page=0)]
    bindings = {"revenue/FY2024": ["e1"], "revenue/FY2023": ["e2"]}

    state = _state(packets, bindings, question="Why did revenue change?")
    state.intent = "MULTI_EVIDENCE"
    state._calculation_result_obj = CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation("difference"),
        value=Decimal("8"),
        unit="USD",
    )

    capability, backend, _ = _run(state)

    assert capability.last_context_pack.calculation is not None
    assert dict(capability.last_context_pack.calculation) == {
        "operation": "difference",
        "unit": "USD",
        "value": "8",
    }
    assert "[VERIFIED CALCULATION]" in backend.prompts[0]

    without = _run(_state(packets, bindings, question="Summarise the reported figures."))
    assert without[0].last_context_pack.calculation is None
    assert "[VERIFIED CALCULATION]" not in without[1].prompts[0]


def test_a_budget_truncation_is_visible_in_the_trace_and_in_the_groups() -> None:
    """Evidence-count budgeting still behaves deterministically after migration.

    No token bound is added anywhere by this phase: B3's tokenizer is the
    checkpoint's, and a bound is either measured exactly or not configured.
    """

    from rag_v2.context import ContextBudgetV1

    state = _state(
        [
            _packet("e1"),
            _packet("e2", document_id="doc-2", page=9),
            _packet("e3", metric="Cost of revenue", value="214", page=0),
        ],
        {"revenue/FY2024": ["e1", "e2"], "cost_of_revenue/FY2024": ["e3"]},
    )
    request = specialist_context_request(state)
    pack = ContextCompilerV1(
        SpecialistContextPolicyV1(),
        budget=ContextBudgetV1(max_evidence_items=2),
    ).compile(request)

    assert pack.evidence_ids == ("e1", "e2")
    assert pack.budget.evidence_dropped == 1
    # The third item's group went with it: a group left without supports is not
    # emitted, so the shed item cannot be reported as a corroboration.
    assert _groups(pack) == [("G1", "revenue/FY2024", ("E1", "E2"))]
    assert pack.budget.max_input_tokens is None


# --- §8/§9: the anti-migration guards are gone, replaced by their opposite --------------------


def test_the_production_module_no_longer_builds_its_own_context() -> None:
    """The one structural assertion kept, and why it is worth keeping.

    §9 asks for a behavioural guard where possible, and the substitutions above
    provide it.  This is different in kind: it asserts that a *disclosure*
    decision and a *selection* no longer happen in this module at all.  If a
    future edit added a projection back beside the compiler, the model-facing
    surface would have two policies again -- the asymmetry H2A-1 removed -- and
    no behavioural test of the compiler would notice.

    Checked on the module's syntax tree rather than its text, so the prose above
    that *names* what was removed does not satisfy it.  The first version of
    this test grepped the source and failed on its own comments, which is the
    right failure: a check that a comment can pass is not checking the code.
    """

    source = pathlib.Path("src/runtime/trusted_v2_generation.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    assert not {name for name in imported if name.startswith("rag_v2.evidence")}, (
        "the production generation path must not import the Disclosure Authority "
        "directly; the compiler is the only path from artifacts to context"
    )

    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    referenced |= {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    for absent in ("project", "project_calculation", "evidence_packets"):
        assert absent not in referenced, absent
