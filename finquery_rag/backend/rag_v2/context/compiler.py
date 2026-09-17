"""The Context Compiler kernel: deterministic selection, then disclosure.

Fixed order, and the order is the design:

    already-admitted authoritative artifacts
        -> deterministic model-visible selection     (on authoritative fields)
        -> Disclosure Authority projection           (deny by default)
        -> ContextBudget enforcement                 (on the disclosed payload)
        -> AgentContextPackV1                        (disclosed projections only)

Why selection comes first.  Selection has to read *authoritative* semantic
fields -- which metric, which period, which scope -- to decide relevance at all.
Projecting first would mean selecting on a narrowed view, and for V1 would mean
admitting a container the profile deliberately governs key-by-key.  So selection
inspects authority and the pack never holds it; the two are separated by the
projection step rather than by trusting callers to remember.

The kernel is intentionally small.  It is not an authority and owns no truth:
it does not retrieve, admit, bind, resolve conflicts, calculate, verify claims,
replan, or repair prompts.  Every one of those happens strictly upstream, and
the compiler's whole input is their result.  A compiler that could admit
evidence would be a second Binder; a compiler that could decide which value is
canonical would be a second binding authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rag_v2.evidence.disclosure import (
    EvidenceDisclosureProfile,
    project,
    project_calculation,
)

from .contracts import (
    AgentContextPackV1,
    ContextBudgetAccountingV1,
    ContextBudgetUnsupported,
    ContextBudgetV1,
    ContextReferenceV1,
    ContextReferencesV1,
    ContextRoleV1,
    ContextSelectionEntryV1,
    ContextSelectionReasonV1,
    ContextSelectionTraceV1,
    ContextSupportGroupV1,
    ExactTokenCounterV1,
    PackIntegrityError,
    UnknownContextRole,
)

__all__ = [
    "ContextCompilerV1",
    "ContextRequestV1",
    "EvidenceSelectionPolicyV1",
    "SelectionResultV1",
]


@dataclass(frozen=True)
class ContextRequestV1:
    """The compiler's entire input: artifacts the runtime has already admitted.

    ``admitted_evidence`` are authoritative objects -- the same shapes the
    runtime already carries -- in a stable order.  They are *admitted*: whether
    they are trusted, corroborated, or conflict-free was settled upstream.  The
    compiler may reorder and subset them; it may not promote or demote one.

    ``calculation`` is the authoritative calculation payload, or ``None``.  It
    is not recomputed here, and a pack never carries a calculation the runtime
    did not already produce.

    ``slot_bindings`` is the **authoritative claim/support relation**, carried
    as ordered pairs so that the compiler inherits the authority's ordering
    rather than imposing one: each pair is a Binder slot key and the evidence
    ids admitted as that slot's supports.  One id is one claim with one source;
    several are one claim with several independent sources.  The compiler does
    not group them, does not decide when two items are the same claim, and has
    no code that could -- it projects the relation it was handed.  An empty
    tuple means the runtime established no topology, and the pack then reports
    none rather than partitioning the evidence itself.
    """

    role: ContextRoleV1
    invocation_id: str
    query: str
    admitted_evidence: tuple[Mapping[str, Any], ...] = ()
    calculation: Mapping[str, Any] | None = None
    slot_bindings: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class SelectionResultV1:
    """What a role policy chose, and why -- in admitted order, complete."""

    selected: tuple[Mapping[str, Any], ...]
    entries: tuple[ContextSelectionEntryV1, ...]


class EvidenceSelectionPolicyV1(Protocol):
    """A role's deterministic answer to "which of these may this model see?".

    The policy is where role knowledge lives -- which artifacts a boundary needs
    and in what order -- and it is the only place a role is named.  It reads
    authoritative fields to choose; it never decides what is true, and it never
    projects.  Projection belongs to the kernel, so a policy cannot widen what
    crosses by returning a shape it built itself.
    """

    @property
    def role(self) -> ContextRoleV1: ...

    @property
    def evidence_profile(self) -> EvidenceDisclosureProfile: ...

    @property
    def calculation_profile(self) -> EvidenceDisclosureProfile | None: ...

    def select(
        self,
        admitted: Sequence[Mapping[str, Any]],
        budget: ContextBudgetV1,
    ) -> SelectionResultV1: ...


def _payload_text(
    evidence_views: Sequence[Mapping[str, Any]],
    calculation_view: Mapping[str, Any] | None,
    references: ContextReferencesV1,
) -> str:
    """Canonical serialization of the compiler's own dynamic payload.

    Measured only when an exact counter is supplied.  The text is a *stable*
    rendering of the disclosed projections and the model-visible topology, and
    nothing else -- no instructions, no formatting, no renderer output -- so a
    count taken here describes the context the compiler selected, which is not
    the same number as the model's prompt length.  ``selected_context_tokens``
    is named for exactly that reason.

    The topology belongs in the count because it is context: a support group is
    something the invocation sees, so a bound measured without it would
    under-count the thing it exists to bound.  The ``E1..En`` handles are not
    listed separately -- they are a positional function of the evidence order,
    so they carry no information this payload does not already have.
    """

    return json.dumps(
        {
            "evidence": [dict(view) for view in evidence_views],
            "calculation": None if calculation_view is None else dict(calculation_view),
            "support_groups": [
                {
                    "handle": group.handle,
                    "canonical_slot": group.canonical_slot,
                    "supports": list(group.supports),
                }
                for group in references.support_groups
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


class ContextCompilerV1:
    """Compile one governed, temporary context pack for one model invocation."""

    def __init__(
        self,
        policy: EvidenceSelectionPolicyV1,
        *,
        budget: ContextBudgetV1 | None = None,
        token_counter: ExactTokenCounterV1 | None = None,
    ) -> None:
        role = getattr(policy, "role", None)
        if not isinstance(role, ContextRoleV1):
            raise UnknownContextRole(
                f"policy role must be a ContextRoleV1, got {role!r}"
            )
        self.policy = policy
        self.budget = budget if budget is not None else ContextBudgetV1()
        self.token_counter = token_counter
        # Fail at construction, not at the first call: a boundary configured
        # with a token bound it cannot measure is a deployment error, and it
        # should surface where it is configured rather than where it is used.
        if self.budget.requires_exact_counter and token_counter is None:
            raise ContextBudgetUnsupported(
                "max_input_tokens is configured but no exact token counter was "
                "supplied for this boundary. A word count, a character count, a "
                "tokenizer for a different model, or any other estimate is not "
                "an acceptable substitute: leave the bound unset instead of "
                "measuring it approximately."
            )

    def _measure(self, text: str) -> int:
        counter = self.token_counter
        if counter is None:  # guarded by __init__; kept explicit for clarity
            raise ContextBudgetUnsupported("no exact token counter available")
        return int(counter.count(text))

    def compile(self, request: ContextRequestV1) -> AgentContextPackV1:
        if request.role is not self.policy.role:
            # ``!r`` rather than ``.value``: the guard has to hold when the role
            # is not an enum at all, and an ordinary mistake -- a caller passing
            # the string "SPECIALIST" instead of the member -- should report
            # which role was wrong, not raise an AttributeError from the message
            # that was supposed to explain it.
            raise UnknownContextRole(
                f"request role {request.role!r} does not match policy role "
                f"{self.policy.role!r}"
            )

        # 1. Deterministic selection, on authoritative fields.
        selection = self.policy.select(request.admitted_evidence, self.budget)
        # A policy that manufactures artifacts has stopped selecting, and a
        # pack built from invented evidence would be indistinguishable
        # downstream from one built from admitted evidence.
        if len(selection.selected) > len(request.admitted_evidence):
            raise PackIntegrityError(
                "selection returned more evidence than was admitted"
            )

        # 2. Disclosure Authority projection.  Only this crosses.
        evidence_views = [
            project(item, profile=self.policy.evidence_profile)
            for item in selection.selected
        ]
        calculation_view = (
            None
            if request.calculation is None
            else project_calculation(
                request.calculation, profile=self.policy.calculation_profile
            )
        )

        # 3. Token budget, enforced on the disclosed payload.  Evidence is shed
        #    from the end, which is the least-preferred position by the role
        #    policy's own ordering -- never in the middle, which would make the
        #    surviving order depend on how many were dropped.
        entries = list(selection.entries)
        references = _references(
            evidence_views, calculation_view, request.slot_bindings
        )
        if self.budget.max_input_tokens is not None:
            limit = self.budget.max_input_tokens
            while evidence_views and self._measure(
                _payload_text(evidence_views, calculation_view, references)
            ) > limit:
                evidence_views.pop()
                entries = _mark_last_kept_as_dropped(entries)
                # The topology shrinks with the evidence.  A shed support is no
                # longer a witness this invocation can see, and the pack refuses
                # a group naming a handle it does not carry -- so the groups are
                # rebuilt rather than left describing evidence that is gone.
                references = _references(
                    evidence_views, calculation_view, request.slot_bindings
                )

        # 4. Budget accounting and the pack.
        payload_tokens = (
            None
            if self.token_counter is None
            else self._measure(
                _payload_text(evidence_views, calculation_view, references)
            )
        )

        considered = len(request.admitted_evidence)
        selected_count = len(evidence_views)

        return AgentContextPackV1(
            role=request.role,
            invocation_id=request.invocation_id,
            query=request.query,
            evidence=tuple(evidence_views),
            calculation=calculation_view,
            references=references,
            budget=ContextBudgetAccountingV1(
                max_input_tokens=self.budget.max_input_tokens,
                max_evidence_items=self.budget.max_evidence_items,
                reserved_output_tokens=self.budget.reserved_output_tokens,
                evidence_considered=considered,
                evidence_selected=selected_count,
                evidence_dropped=considered - selected_count,
                selected_context_tokens=payload_tokens,
                token_counter_id=(
                    None if self.token_counter is None
                    else str(self.token_counter.counter_id)
                ),
            ),
            selection=ContextSelectionTraceV1(tuple(entries)),
        )


def _mark_last_kept_as_dropped(
    entries: list[ContextSelectionEntryV1],
) -> list[ContextSelectionEntryV1]:
    """Reclassify the last surviving artifact as dropped by the token budget.

    The trace is a complete record of the decision, so shedding an artifact has
    to be visible in it as a *reason* rather than as an absence -- otherwise the
    trace would say an artifact was admitted while the pack did not hold it.
    """

    for index in range(len(entries) - 1, -1, -1):
        if entries[index].reason is ContextSelectionReasonV1.ADMITTED:
            entries[index] = ContextSelectionEntryV1(
                evidence_id=entries[index].evidence_id,
                reason=ContextSelectionReasonV1.DROPPED_BY_TOKEN_BUDGET,
                rank=entries[index].rank,
            )
            return entries
    return entries


def _references(
    evidence_views: Sequence[Mapping[str, Any]],
    calculation_view: Mapping[str, Any] | None,
    slot_bindings: Sequence[tuple[str, Sequence[str]]] = (),
) -> ContextReferencesV1:
    """Model-visible handles, assigned by selection order.

    These are the same handles the B3 renderer emits -- ``E1..En`` positionally
    and ``C1`` for the calculation -- so a citation a model writes can be
    resolved back to an authoritative identity without the pack carrying one.
    """

    handles = [
        ContextReferenceV1(
            handle=f"E{index}",
            kind="evidence",
            evidence_id=(
                None if view.get("evidence_id") is None else str(view["evidence_id"])
            ),
            citation_id=(
                None if view.get("citation_id") is None else str(view["citation_id"])
            ),
        )
        for index, view in enumerate(evidence_views, start=1)
    ]
    if calculation_view is not None:
        handles.append(ContextReferenceV1(handle="C1", kind="calculation"))
    return ContextReferencesV1(
        handles=tuple(handles),
        support_groups=_support_groups(handles, slot_bindings),
    )


def _support_groups(
    handles: Sequence[ContextReferenceV1],
    slot_bindings: Sequence[tuple[str, Sequence[str]]],
) -> tuple[ContextSupportGroupV1, ...]:
    """Project the authority's claim/support relation onto the handles it maps to.

    This is a rename and a filter, and deliberately nothing more.  Each slot the
    Binder bound becomes one group, named ``G1..Gm`` in the authority's own
    order, carrying the slots' supports translated from evidence ids into the
    handles the model will actually cite.  No slot key is composed here, no two
    items are judged to be the same claim here, and there is no code in this
    module that could do either -- that judgment is slot binding's, and it
    happened upstream.

    Two restrictions, neither of which is a judgment about the evidence:

    * an id the pack does not carry is left out.  The boundary did not release
      it, so the model cannot check a corroboration it cannot see, and a group
      naming it would be a claim about evidence that never crossed;
    * a slot whose every support was left out is not emitted at all, because an
      empty group is that same claim with nothing behind it -- and the pack
      refuses to hold one.

    A repeated id collapses.  Slot bindings are deduplicated where they are
    built, so the authority does not produce one; the guard is here because if
    one ever did, reporting it twice would state a corroboration that does not
    exist, which is the failure this phase removes rather than reproduces.
    """

    handle_by_id: dict[str, str] = {}
    for reference in handles:
        if reference.kind != "evidence" or reference.evidence_id is None:
            continue
        # First occurrence wins, matching the positional assignment above: a
        # duplicate identity would have produced a second handle, and the group
        # must point at the one the renderer emits first.
        handle_by_id.setdefault(reference.evidence_id, reference.handle)

    groups: list[ContextSupportGroupV1] = []
    for slot_id, evidence_ids in slot_bindings:
        supports: list[str] = []
        for evidence_id in evidence_ids:
            handle = handle_by_id.get(str(evidence_id))
            if handle is not None and handle not in supports:
                supports.append(handle)
        if not supports:
            continue
        groups.append(
            ContextSupportGroupV1(
                handle=f"G{len(groups) + 1}",
                canonical_slot=str(slot_id),
                supports=tuple(supports),
            )
        )
    return tuple(groups)
