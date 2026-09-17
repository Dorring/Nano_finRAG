"""B3's role policy, and the shadow path that compiles a pack from a RunState.

Two things live here, and the split between them is the design:

* ``SpecialistContextPolicyV1`` is **role knowledge** -- which of the already
  admitted artifacts the financial specialist's model invocation may see, in
  what order.  It is pure and takes authoritative objects, never a RunState.
* ``specialist_context_request`` is the **adapter** -- it reads a RunState and
  produces the compiler's input.  It is the only thing here that knows what a
  RunState is, and it hands the kernel plain artifacts.

The adapter also carries the **claim/support relation** across: the state's
``bound_slot_bindings`` becomes the request's ``slot_bindings``.  That relation
is read, never derived -- the compiler does not decide when two evidence items
are one claim, and grouping is deliberately not part of the policy either, since
it is not a role's judgment about evidence.  ``rag_v2`` could not compute it in
any case: the canonicalisation lives in ``src/runtime`` and ``src/generation``,
and the dependency runs the other way.

The adapter is a faithful re-implementation of
``trusted_v2_generation._bound_items``, deliberately: H2A-3B1 proves the
framework without moving the production path, so the production path keeps its
own code and this one runs beside it.  That duplication is temporary and is
exactly what H2A-3B3's migration exists to resolve -- but it is duplication of
*selection*, not of admission.  Which evidence is trusted stays with the Binder;
this only decides which admitted items a model is shown.

Nothing in ``src/`` imports this module.  That is not an accident of wiring, it
is H2A-3B1's central constraint: during 3B1 the compiled pack must never reach a
model call, and that remains true until 3B3 moves the boundary deliberately.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag_v2.evidence.disclosure import EvidenceDisclosureProfile

from .compiler import ContextRequestV1, SelectionResultV1
from .contracts import (
    ContextBudgetV1,
    ContextRoleV1,
    ContextSelectionEntryV1,
    ContextSelectionReasonV1,
)

__all__ = [
    "SPECIALIST_INVOCATION",
    "SpecialistContextPolicyV1",
    "admitted_specialist_evidence",
    "evidence_identity",
    "specialist_context_request",
]

#: Names the model call this pack is for.  A request id alone would not be an
#: *invocation* identity if a boundary ever made more than one call.
SPECIALIST_INVOCATION = "candidate-generation"


def _stable_unique(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def evidence_identity(item: Mapping[str, Any]) -> str:
    """The field ladder an admitted evidence object uses for its identity.

    ``fact_id`` first because the live fact store emits it, then ``evidence_id``.
    This mirrors the production resolver rather than inventing a fourth ladder --
    a compiler that resolved identity differently from everything around it
    would silently change which items are considered duplicates.
    """

    value = item.get("fact_id") or item.get("evidence_id") or item.get("candidate_id")
    return "" if value is None else str(value).strip()


def admitted_specialist_evidence(
    state: Any,
) -> tuple[Mapping[str, Any], ...]:
    """The already-admitted evidence for this request, in stable order.

    Takes a RunState because the *adapter* is allowed to; the policy and the
    kernel are not.  Order is the state's own packet order, deduplicated by
    evidence identity with the first occurrence winning, which is what the
    production path does -- so a compiled pack and the legacy context see the
    same items in the same sequence.
    """

    allowed = set(_stable_unique(getattr(state, "bound_evidence_ids", ())))
    if not allowed:
        return ()
    admitted: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for raw in getattr(state, "evidence_packets", ()):
        if not isinstance(raw, Mapping):
            raise TypeError("admitted evidence must be a mapping")
        identity = evidence_identity(raw)
        if identity in allowed and identity not in seen:
            seen.add(identity)
            admitted.append(dict(raw))
    return tuple(admitted)


def _calculation_payload(state: Any) -> Mapping[str, Any] | None:
    """The authoritative calculation payload, or ``None``.

    Duck-typed on ``to_dict`` rather than isinstance-checked against
    ``CalculationResult``.  That type lives in ``src/domain``, and ``rag_v2``
    imports ``src`` zero times -- pointing this package at the application layer
    to satisfy a type check would invert the dependency the whole shadow
    architecture rests on.
    """

    result = getattr(state, "_calculation_result_obj", None)
    if result is None:
        return None
    to_dict = getattr(result, "to_dict", None)
    if not callable(to_dict):
        return None
    payload = to_dict()
    return payload if isinstance(payload, Mapping) else None


def _slot_bindings(state: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The Binder's claim/support relation, as the state carries it.

    ``AdaptiveRAGStateV1.bound_slot_bindings`` is the runtime's projection of
    ``EvidenceBinding.slot_bindings``: one entry per slot the Binder bound,
    naming the evidence ids it admitted as that slot's supports.  A slot with
    two ids is one claim with two witnesses, which is the whole of the relation
    the compiler needs -- and it is *read here*, not computed, because the
    compiler must not be the second place that decides when two evidence items
    are the same claim.

    Absent bindings produce no entries, and the pack then reports no topology.
    That is the honest result rather than a gap to fill: "the authority did not
    place this item" and "this item is its own claim" are different statements,
    and inferring the second from the first would be the compiler inventing a
    relation it was not given.

    A malformed value raises, matching ``admitted_specialist_evidence``.  A
    binding that cannot be read is a runtime defect, and silently reporting "no
    topology" would hide it behind a shape the pack treats as normal.
    """

    raw = getattr(state, "bound_slot_bindings", None)
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise TypeError("bound slot bindings must be a mapping")

    bindings: list[tuple[str, tuple[str, ...]]] = []
    for slot_id, evidence_ids in raw.items():
        if isinstance(evidence_ids, (str, bytes)) or not isinstance(
            evidence_ids, Iterable
        ):
            raise TypeError("a slot's supports must be an iterable of ids")
        bindings.append((str(slot_id), tuple(str(item) for item in evidence_ids)))
    return tuple(bindings)


def specialist_context_request(
    state: Any,
    *,
    invocation: str = SPECIALIST_INVOCATION,
) -> ContextRequestV1:
    """Turn a RunState into the compiler's input for the B3 boundary."""

    return ContextRequestV1(
        role=ContextRoleV1.SPECIALIST,
        invocation_id=f"{getattr(state, 'request_id', '')}:{invocation}",
        query=str(getattr(state, "normalized_query", "")),
        admitted_evidence=admitted_specialist_evidence(state),
        calculation=_calculation_payload(state),
        slot_bindings=_slot_bindings(state),
    )


@dataclass(frozen=True)
class SpecialistContextPolicyV1:
    """The financial specialist's model-visible selection.

    The policy is deliberately thin, because B3's current semantics are: show the
    model every item the Binder admitted, in the order the state carries them.
    Narrowing that here would be a behaviour change smuggled into a framework
    phase, and H2A-3B2's differential would then be comparing two changes at
    once.  What the policy *does* own is the ordering contract and the evidence
    budget, which is where a later role adds relevance ranking if one is needed.

    Relevance ranking is explicitly absent.  The contract for that work is that
    it must be deterministic and inspectable; an LLM context selector would put
    a second model between the evidence and the model that answers from it.
    """

    @property
    def role(self) -> ContextRoleV1:
        return ContextRoleV1.SPECIALIST

    @property
    def evidence_profile(self) -> EvidenceDisclosureProfile:
        return EvidenceDisclosureProfile.SPECIALIST

    @property
    def calculation_profile(self) -> EvidenceDisclosureProfile:
        return EvidenceDisclosureProfile.SPECIALIST

    def select(
        self,
        admitted: Sequence[Mapping[str, Any]],
        budget: ContextBudgetV1,
    ) -> SelectionResultV1:
        limit = budget.max_evidence_items
        selected: list[Mapping[str, Any]] = []
        entries: list[ContextSelectionEntryV1] = []

        for rank, item in enumerate(admitted):
            identity = evidence_identity(item)
            if limit is not None and len(selected) >= limit:
                entries.append(
                    ContextSelectionEntryV1(
                        evidence_id=identity,
                        reason=ContextSelectionReasonV1.DROPPED_BY_EVIDENCE_BUDGET,
                        rank=rank,
                    )
                )
                continue
            selected.append(item)
            entries.append(
                ContextSelectionEntryV1(
                    evidence_id=identity,
                    reason=ContextSelectionReasonV1.ADMITTED,
                    rank=rank,
                )
            )

        return SelectionResultV1(tuple(selected), tuple(entries))
