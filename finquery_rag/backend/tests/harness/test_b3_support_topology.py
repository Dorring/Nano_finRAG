"""H2A-3B2: the pack carries one claim with N supports, as a projection.

H2A-3B1 recorded the gap this file closes.  A pack compiled for a multi-support
state carried two ordinary evidence items, and nothing in it distinguished "two
witnesses of one figure" from "two different figures" -- so a consumer reading
the pack had no way to avoid stating the value twice, and the only thing standing
between that and a model was a routing policy upstream.

The relation now crosses, and where it comes from is the whole design:

    EvidenceBinding.slot_bindings        (the authority)
          -> the runtime's state
          -> ContextCompilerV1           (a projection, not a judgment)
          -> AgentContextPackV1

and *not*:

    metric + period + value
          -> the compiler decides which items are one claim

The second shape is the failure this phase is built to avoid.  A compiler that
could group evidence would be a second binding authority, and the two would
disagree the first time either changed -- which is precisely what H2A-2 spent a
phase establishing and what ``EvidenceBinding`` exists to be the single answer
to.  ``rag_v2`` also cannot reach the code that computes it: the canonicalisation
lives in ``src/runtime/trusted_v2_binder.py`` and ``src/generation``, and the
dependency runs the other way.  That is a structural reason, and the tests below
add a behavioural one -- a binding that groups two *different* values is carried
as given, because a compiler that second-guessed it would be recomputing.

The scenarios are the ones the phase specifies: a single evidence item; one
canonical fact with two independent supports; two distinct facts; a multi-support
claim beside a single-support one; a conflicting candidate; and a physical-source
duplicate.  The last two are the ones that would go wrong if the compiler were
deciding, and neither is hypothetical -- both are shapes the authority produces
or refuses.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_v2.context import (
    AgentContextPackV1,
    ContextBudgetV1,
    ContextCompilerV1,
    ContextReferencesV1,
    ContextRoleV1,
    ContextSupportGroupV1,
    PackIntegrityError,
    SpecialistContextPolicyV1,
    specialist_context_request,
)


RETURNED = "not specified"


def _packet(evidence_id: str, **fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "evidence_id": evidence_id,
        "citation_id": f"citation:{evidence_id}",
        "metric": "Revenue",
        "value": "391",
        "period": "FY2024",
        "scope": "consolidated",
        "document_id": "doc-1",
        "page": 7,
    }
    base.update(fields)
    return base


def _compile(
    packets: list[dict[str, Any]],
    bindings: dict[str, list[str]],
    *,
    bound: list[str] | None = None,
    budget: ContextBudgetV1 | None = None,
) -> AgentContextPackV1:
    """Compile a pack from a real state, driven through the adapter."""

    from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1

    state = AdaptiveRAGStateV1.new("r", "What was revenue?")
    state.add_evidence([EvidencePacketV1.from_mapping(item) for item in packets])
    state.bound_evidence_ids = list(
        bound if bound is not None else [item["evidence_id"] for item in packets]
    )
    state.bound_slot_bindings = {
        slot: list(ids) for slot, ids in bindings.items()
    }

    compiler = ContextCompilerV1(
        SpecialistContextPolicyV1(),
        budget=budget if budget is not None else ContextBudgetV1(),
    )
    return compiler.compile(specialist_context_request(state))


def _groups(pack: AgentContextPackV1) -> list[tuple[str, str, tuple[str, ...]]]:
    return [
        (group.handle, group.canonical_slot, group.supports)
        for group in pack.references.support_groups
    ]


# --- the scenarios the phase specifies ------------------------------------------------


def test_a_single_evidence_item_is_one_claim_with_one_support() -> None:
    pack = _compile([_packet("e1")], {"revenue/FY2024": ["e1"]})

    assert pack.evidence_ids == ("e1",)
    assert _groups(pack) == [("G1", "revenue/FY2024", ("E1",))]


def test_two_supports_of_one_canonical_fact_are_one_claim() -> None:
    """The shape that motivated the phase: two witnesses, one figure.

    Two documents, the same metric, the same period, the same value.  A renderer
    reading only the evidence list cannot tell this from two separate figures and
    would state the value twice.
    """

    pack = _compile(
        [_packet("e1"), _packet("e2", document_id="doc-2", page=9)],
        {"revenue/FY2024": ["e1", "e2"]},
    )

    assert pack.evidence_ids == ("e1", "e2")
    assert _groups(pack) == [("G1", "revenue/FY2024", ("E1", "E2"))]


def test_two_distinct_facts_are_two_claims() -> None:
    """The other direction, and the reason singletons are emitted at all.

    If only multi-support slots produced a group, a reader could not distinguish
    "these two are one claim" from "the topology was not established" -- and
    "these are two separate claims" would be the one thing the pack could not
    say.
    """

    pack = _compile(
        [_packet("e1"), _packet("e2", period="FY2023", page=0)],
        {"revenue/FY2024": ["e1"], "revenue/FY2023": ["e2"]},
    )

    assert _groups(pack) == [
        ("G1", "revenue/FY2024", ("E1",)),
        ("G2", "revenue/FY2023", ("E2",)),
    ]


def test_a_multi_support_claim_beside_a_single_support_claim() -> None:
    """The mixed shape, where a positional or count-based rule would go wrong."""

    pack = _compile(
        [
            _packet("e1"),
            _packet("e2", document_id="doc-2", page=9),
            _packet("e3", metric="Cost of revenue", value="214", page=0),
        ],
        {"revenue/FY2024": ["e1", "e2"], "cost_of_revenue/FY2024": ["e3"]},
    )

    assert _groups(pack) == [
        ("G1", "revenue/FY2024", ("E1", "E2")),
        ("G2", "cost_of_revenue/FY2024", ("E3",)),
    ]


def test_a_conflicting_candidate_enters_no_admitted_support_group() -> None:
    """An id the boundary did not admit cannot appear in a group.

    The runtime refuses a conflict one layer up -- ``_unresolved_conflict_slots``
    clears the admitted set *and* the slot bindings, so a conflicting candidate
    never reaches the compiler at all -- and that is the primary mechanism.  This
    asserts the compiler's own half of it, which is the half that survives a
    stale or partial binding: the pack's evidence is the admitted set, and a
    group may not name a handle the pack does not carry.

    Both halves are checked, because either alone would leave the property
    resting on the other.
    """

    packets = [
        _packet("e1"),
        _packet("e2", value="999", document_id="doc-2", page=9),  # the rival
        _packet("e3", metric="Cost of revenue", value="214", page=0),
    ]

    # (a) The real conflict path: not admitted, and not bound either.
    pack = _compile(
        packets,
        {"revenue/FY2024": ["e1"], "cost_of_revenue/FY2024": ["e3"]},
        bound=["e1", "e3"],
    )
    assert pack.evidence_ids == ("e1", "e3")
    assert all("e2" not in group[2] for group in _groups(pack))
    assert "e2" not in repr(pack)

    # (b) A binding that still names it.  The compiler must not carry a
    #     corroboration for evidence the boundary did not release.
    stale = _compile(
        packets,
        {"revenue/FY2024": ["e1", "e2"]},
        bound=["e1"],
    )
    assert stale.evidence_ids == ("e1",)
    assert _groups(stale) == [("G1", "revenue/FY2024", ("E1",))]


def test_a_physical_source_duplicate_is_not_promoted_to_a_second_support() -> None:
    """Two extraction rows of one source are one source, and the compiler agrees.

    The Binder deduplicates witnesses by ``physical_source_id`` before binding,
    so the authority's slot holds one id.  What this asserts is that the compiler
    does not re-derive a support set of its own: the second packet is *admitted*
    evidence and would be trivial to add, and adding it would manufacture a
    corroboration that the authority explicitly refused to count.

    The mirror case is checked too -- the compiler is not deduplicating either.
    If the authority did put both ids in one slot, both are reported, because
    collapsing them would be the compiler overruling a binding it did not make.
    """

    duplicate = _packet("e2", physical_source_id="src-1")

    # The Binder's own verdict: one witness for this slot.
    resolved = _compile(
        [_packet("e1", physical_source_id="src-1"), duplicate],
        {"revenue/FY2024": ["e1"]},
    )
    assert resolved.evidence_ids == ("e1", "e2")
    assert _groups(resolved) == [("G1", "revenue/FY2024", ("E1",))]

    # And if the authority had counted both, the compiler reports both.
    as_bound = _compile(
        [_packet("e1", physical_source_id="src-1"), duplicate],
        {"revenue/FY2024": ["e1", "e2"]},
    )
    assert _groups(as_bound) == [("G1", "revenue/FY2024", ("E1", "E2"))]


# --- it is a projection, not a judgment ------------------------------------------------


def test_the_compiler_carries_a_grouping_it_would_not_have_chosen_itself() -> None:
    """The behavioural proof that no claim identity is computed here.

    The two items below state *different values* in different periods.  Any
    canonicaliser would call them two facts; the authority calls them one claim.
    Both answers are wrong to second-guess at this layer -- a compiler that
    "corrected" this binding would be recomputing the relation, and would then
    disagree with the Binder the first time either changed.

    So the pack reports what it was told.  A future change that made the
    compiler group evidence by metric/period/value would fail here, which is the
    point: that change should be visible, not silent.
    """

    pack = _compile(
        [
            _packet("e1", value="391", period="FY2024"),
            _packet("e2", value="42", period="FY2019", page=0),
        ],
        {"one_claim_per_the_authority": ["e1", "e2"]},
    )

    assert [item["value"] for item in pack.evidence] == ["391", "42"]
    assert _groups(pack) == [
        ("G1", "one_claim_per_the_authority", ("E1", "E2")),
    ]


def test_the_slot_key_is_carried_verbatim() -> None:
    """Not recomposed, not normalised, not prettified.

    Whatever the supervisor called the slot is the identity a reader resolves
    against, and a compiler that reformatted it would be inventing a second
    naming for one thing.
    """

    slot = "Revenue/Q4-2024::consolidated (restated)"
    pack = _compile([_packet("e1")], {slot: ["e1"]})

    assert _groups(pack) == [("G1", slot, ("E1",))]


def test_the_group_order_is_the_authority_order() -> None:
    """Insertion order, not sorted -- the authority's order is information."""

    bindings = {"z/last": ["e2"], "a/first": ["e1"]}
    pack = _compile([_packet("e1"), _packet("e2", page=0)], bindings)

    assert [(handle, slot) for handle, slot, _ in _groups(pack)] == [
        ("G1", "z/last"),
        ("G2", "a/first"),
    ]


def test_the_supports_within_a_group_keep_the_authority_order() -> None:
    pack = _compile(
        [_packet("e1"), _packet("e2", page=0), _packet("e3", page=1)],
        {"revenue/FY2024": ["e3", "e1", "e2"]},
    )

    assert _groups(pack) == [("G1", "revenue/FY2024", ("E3", "E1", "E2"))]
    assert pack.evidence_ids == ("e1", "e2", "e3"), "evidence order is untouched"


def test_a_repeated_support_is_not_a_second_witness() -> None:
    """A duplicated id is collapsed, and the reason is truthfulness, not hygiene.

    The authority deduplicates slot bindings where they are built, so this shape
    does not arrive in production.  If it ever did, reporting the id twice would
    tell the model that two independent sources agree when one does.
    """

    pack = _compile([_packet("e1")], {"revenue/FY2024": ["e1", "e1"]})

    assert _groups(pack) == [("G1", "revenue/FY2024", ("E1",))]


# --- when the authority is silent -------------------------------------------------------


def test_no_binding_means_no_topology_rather_than_a_guessed_one() -> None:
    """The compiler does not partition evidence it was not told about.

    "The authority did not place this item" and "this item is its own claim" are
    different statements, and only the first is one a projection may make.  The
    consequence is that an older or partial state compiles to a pack with no
    groups -- which is honest, and visibly different from a pack whose authority
    said "these are three separate claims".
    """

    pack = _compile(
        [_packet("e1"), _packet("e2", page=0), _packet("e3", page=1)],
        {},
    )

    assert pack.evidence_ids == ("e1", "e2", "e3")
    assert pack.references.support_groups == ()


def test_a_state_that_never_bound_anything_still_compiles() -> None:
    """``bound_slot_bindings`` absent is the same answer as empty, not a crash."""

    from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1

    state = AdaptiveRAGStateV1.new("r", "What was revenue?")
    state.add_evidence([EvidencePacketV1.from_mapping(_packet("e1"))])
    state.bound_evidence_ids = ["e1"]
    del state.bound_slot_bindings  # a state shape that predates the field

    pack = ContextCompilerV1(SpecialistContextPolicyV1()).compile(
        specialist_context_request(state)
    )

    assert pack.references.support_groups == ()
    assert pack.evidence_ids == ("e1",)


# --- the handles are the ones the pack carries -------------------------------------------


def test_every_group_names_handles_the_pack_actually_carries() -> None:
    """The invariant, over every scenario in this file."""

    pack = _compile(
        [
            _packet("e1"),
            _packet("e2", page=0),
            _packet("e3", page=1),
        ],
        {"revenue/FY2024": ["e1", "e2"], "cost/FY2024": ["e3"]},
    )

    carried = set(pack.references.evidence_handles)
    assert carried == {"E1", "E2", "E3"}
    for group in pack.references.support_groups:
        assert set(group.supports) <= carried
        assert group.handle not in carried, "a group is not an evidence handle"


def test_the_calculation_handle_is_not_an_evidence_handle() -> None:
    """``C1`` is a handle, and a group may not name it.

    The two are different namespaces for the same reason they are different
    reference kinds: a support is evidence, and a calculation has operands
    rather than supports.
    """

    from decimal import Decimal

    from rag_v2.adaptive.adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1
    from src.domain.calculation import (
        CalculationOperation,
        CalculationResult,
        CalculationStatus,
    )

    state = AdaptiveRAGStateV1.new("r", "Why did revenue change?")
    state.add_evidence([EvidencePacketV1.from_mapping(_packet("e1"))])
    state.bound_evidence_ids = ["e1"]
    state.bound_slot_bindings = {"revenue/FY2024": ["e1"]}
    state._calculation_result_obj = CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation("difference"),
        value=Decimal("8"),
        unit="USD",
    )

    pack = ContextCompilerV1(SpecialistContextPolicyV1()).compile(
        specialist_context_request(state)
    )

    assert "C1" in [reference.handle for reference in pack.references.handles]
    assert pack.references.evidence_handles == ("E1",)
    assert _groups(pack) == [("G1", "revenue/FY2024", ("E1",))]


def test_a_pack_refuses_a_group_naming_evidence_it_does_not_carry() -> None:
    """The structural guard, asserted on a hand-built pack.

    The compiler cannot produce this -- the projection restricts to handles it
    just assigned -- but a guard that only holds when its one caller behaves is
    not a guard.  The pack is where "a group names only what the model was shown"
    becomes a property, in the same way that refusing nested mappings is where
    "no raw authority object" does.
    """

    pack = _compile([_packet("e1")], {"revenue/FY2024": ["e1"]})

    with pytest.raises(PackIntegrityError):
        AgentContextPackV1(
            role=ContextRoleV1.SPECIALIST,
            invocation_id="inv",
            query="q",
            evidence=pack.evidence,
            calculation=None,
            references=ContextReferencesV1(
                handles=pack.references.handles,
                support_groups=(
                    ContextSupportGroupV1("G1", "revenue/FY2024", ("E1", "E2")),
                ),
            ),
            budget=pack.budget,
            selection=pack.selection,
        )


def test_a_pack_refuses_a_group_with_nothing_in_it() -> None:
    """An empty group is a claim with no evidence behind it."""

    pack = _compile([_packet("e1")], {"revenue/FY2024": ["e1"]})

    with pytest.raises(PackIntegrityError):
        AgentContextPackV1(
            role=ContextRoleV1.SPECIALIST,
            invocation_id="inv",
            query="q",
            evidence=pack.evidence,
            calculation=None,
            references=ContextReferencesV1(
                handles=pack.references.handles,
                support_groups=(ContextSupportGroupV1("G1", "revenue/FY2024", ()),),
            ),
            budget=pack.budget,
            selection=pack.selection,
        )


# --- determinism and the budget -----------------------------------------------------------


def test_the_same_input_produces_the_same_topology() -> None:
    packets = [_packet("e1"), _packet("e2", page=0)]
    bindings = {"revenue/FY2024": ["e1", "e2"]}

    assert _compile(packets, bindings) == _compile(packets, bindings)


def test_shedding_evidence_shrinks_its_group_and_never_orphans_a_handle() -> None:
    """The topology follows the selection, and the pack refuses to lie about it.

    With an evidence cap of one, the second support is dropped -- so it is no
    longer a witness this invocation can see, and the group must not keep naming
    it.  That is not a judgment about the evidence; it is the same rule as "a
    group names only what the pack carries", applied to a selection that removed
    something after the groups were first built.
    """

    packets = [_packet("e1"), _packet("e2", page=0)]
    bindings = {"revenue/FY2024": ["e1", "e2"]}

    full = _compile(packets, bindings)
    assert _groups(full) == [("G1", "revenue/FY2024", ("E1", "E2"))]

    shed = _compile(packets, bindings, budget=ContextBudgetV1(max_evidence_items=1))
    assert shed.evidence_ids == ("e1",)
    assert _groups(shed) == [("G1", "revenue/FY2024", ("E1",))]
    assert shed.budget.evidence_dropped == 1


def test_a_group_whose_supports_are_all_unadmitted_is_not_emitted() -> None:
    """The empty-group path, reached the way it actually happens.

    An id the Binder bound but the boundary did not release has no handle to name
    -- and a group left with nothing behind it is the same claim with no
    evidence, which the pack refuses outright.  So the group is not emitted, and
    the pack says nothing rather than saying something false.
    """

    pack = _compile([_packet("e1")], {"revenue/FY2024": ["e1"]}, bound=[])

    assert pack.evidence_ids == ()
    assert pack.references.handles == ()
    assert pack.references.support_groups == ()
