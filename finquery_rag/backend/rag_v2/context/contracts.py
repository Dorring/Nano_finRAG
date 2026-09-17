"""The Context Compiler's contracts: what a pack is, and what it may hold.

The compiler answers one question:

    which already-admitted information may *this* model invocation see?

and nothing else.  It does not decide what is true, what is trusted, what is
admitted, or what the answer is.  `ContextCompilerV1` in ``compiler.py`` is the
kernel; this module is the vocabulary it speaks.

Three properties are enforced structurally here rather than by convention,
because each one is a safety property that a future contributor could otherwise
undo without noticing:

* **A pack holds disclosed projections only.**  ``AgentContextPackV1`` refuses
  to be constructed with a nested mapping anywhere in its evidence, which is the
  shape a raw evidence packet has (``metadata``, ``temporal``).  The pack is not
  a container that can carry an authoritative object "for convenience".
* **A pack cannot claim a token bound it did not measure.**  Accounting records
  the *identity* of the counter that produced its number, so a test double can
  never be mistaken for a production measurement.
* **A configured token bound without an exact counter is a configuration
  error**, not something to approximate.  See ``ContextBudgetUnsupported``.
* **A support group cannot name evidence the pack does not carry.**  The
  topology between evidence items is a projection of the Binder's slot binding,
  and the pack refuses to hold a group whose members it was not given.  See
  ``ContextSupportGroupV1``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AgentContextPackV1",
    "ContextBudgetAccountingV1",
    "ContextBudgetUnsupported",
    "ContextBudgetV1",
    "ContextReferenceV1",
    "ContextReferencesV1",
    "ContextRoleV1",
    "ContextSelectionEntryV1",
    "ContextSelectionReasonV1",
    "ContextSelectionTraceV1",
    "ContextSupportGroupV1",
    "ExactTokenCounterV1",
    "PackIntegrityError",
    "UnknownContextRole",
]


class ContextRoleV1(str, Enum):
    """The model boundaries a pack may be compiled for.

    Only roles that exist are listed, for the same reason
    ``EvidenceDisclosureProfile`` lists only real boundaries: a role added for
    symmetry would be a field list nobody validates against a real prompt.  B5,
    B6 and V1 add themselves when their requirements are known, not before.
    """

    SPECIALIST = "SPECIALIST"


#: Roles whose disclosure profile is established.  A role absent from this map
#: cannot be compiled -- an unknown boundary fails rather than defaulting to the
#: most permissive profile.
ROLE_EVIDENCE_PROFILE: Mapping[ContextRoleV1, str] = MappingProxyType(
    {ContextRoleV1.SPECIALIST: "SPECIALIST"}
)


class UnknownContextRole(ValueError):
    """Raised when a caller asks to compile for a boundary that is not defined."""


class ContextBudgetUnsupported(RuntimeError):
    """Raised when a token bound is configured but cannot be measured exactly.

    The rule this exception exists to enforce: a token bound is either measured
    with the real tokenizer for that model, or it is not configured.  Word
    counts, character counts, `tiktoken` on a model that is not a `tiktoken`
    model, and any other substitution are all forbidden -- an estimate presented
    as a measurement is worse than no bound, because it silently truncates
    context that the boundary believed it had room for.
    """


class PackIntegrityError(ValueError):
    """Raised when a pack would be constructed with a shape it must not hold."""


# --- token counting --------------------------------------------------------------


@runtime_checkable
class ExactTokenCounterV1(Protocol):
    """An exact tokenizer for one specific model boundary.

    ``counter_id`` is not decoration.  It is recorded into every pack's budget
    accounting so that a reader can tell which counter produced a number, and so
    that a test double can never be quoted as a production measurement.
    """

    @property
    def counter_id(self) -> str: ...

    def count(self, text: str) -> int: ...


# --- budget ------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextBudgetV1:
    """How much information one model invocation may see.

    This is deliberately *not* ``AdaptiveRAGBudgetV1``.  That type bounds how
    much **work** the runtime may perform -- replans, tool calls, retries -- and
    the two answer different questions.  Merging them would mean one knob
    changing both, which is how a context change becomes a latency change.

    ``max_evidence_items`` is enforced by the compiler.  ``max_input_tokens`` is
    enforced only when an exact counter is supplied, and configuring it without
    one raises ``ContextBudgetUnsupported`` rather than approximating.  Both
    bounds are optional; a boundary with neither configured is bounded by
    whatever already limited its inputs upstream, which is the state B3 is in
    today.

    ``reserved_output_tokens`` is carried, not enforced.  It is the caller's
    declaration that the boundary's window must also hold the answer; the
    compiler enforces only what it can measure, and it does not subtract this
    from ``max_input_tokens``, because doing so would silently reinterpret a
    number the caller already chose.
    """

    max_input_tokens: int | None = None
    max_evidence_items: int | None = None
    reserved_output_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_input_tokens",
            "max_evidence_items",
            "reserved_output_tokens",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer or None")
            if value < 0:
                raise ValueError(f"{name} must not be negative")

        # A boundary whose model-visible evidence cap is zero is not a
        # configuration, it is a mistake: the pack would be empty and the call
        # pointless.  Genuinely-absent evidence is the runtime's fail-closed
        # path, not a budget's, so the two are kept apart.
        if self.max_evidence_items is not None and self.max_evidence_items < 1:
            raise ValueError("max_evidence_items must be at least 1 when set")

    @property
    def requires_exact_counter(self) -> bool:
        return self.max_input_tokens is not None


# --- selection -----------------------------------------------------------------------


class ContextSelectionReasonV1(str, Enum):
    """Why one admitted artifact is, or is not, in the pack.

    Values carry no content.  A trace built from these can be printed, stored
    and compared without any evidence text crossing into it.
    """

    ADMITTED = "ADMITTED"
    DROPPED_BY_EVIDENCE_BUDGET = "DROPPED_BY_EVIDENCE_BUDGET"
    DROPPED_BY_TOKEN_BUDGET = "DROPPED_BY_TOKEN_BUDGET"


@dataclass(frozen=True)
class ContextSelectionEntryV1:
    """One considered artifact: its identity, where it sat, and what happened."""

    evidence_id: str
    reason: ContextSelectionReasonV1
    #: Position in the admitted order.  Stable, content-free, and the only
    #: ordering fact a reader needs to reconstruct the selection.
    rank: int


@dataclass(frozen=True)
class ContextSelectionTraceV1:
    """The minimal deterministic account of a selection.

    Every considered artifact appears exactly once, in admitted order, so the
    trace is a complete record rather than only the survivors.  It never carries
    text: a trace is the part of a context decision that is safe to keep forever.
    """

    entries: tuple[ContextSelectionEntryV1, ...] = ()

    @property
    def selected_evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.evidence_id
            for entry in self.entries
            if entry.reason is ContextSelectionReasonV1.ADMITTED
        )

    @property
    def dropped_evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.evidence_id
            for entry in self.entries
            if entry.reason is not ContextSelectionReasonV1.ADMITTED
        )


# --- accounting ----------------------------------------------------------------------


@dataclass(frozen=True)
class ContextBudgetAccountingV1:
    """What budget governed the invocation, and what it did.

    ``selected_context_tokens`` measures **the compiler's own payload** -- the
    serialized disclosed projections -- with the counter named by
    ``token_counter_id``.  It is *not* the model's prompt token count: the
    prompt also carries static instructions the boundary owns.  Anyone quoting a
    token number from a pack must quote it with that distinction, which is why
    the counter's identity travels beside the number.
    """

    max_input_tokens: int | None
    max_evidence_items: int | None
    reserved_output_tokens: int | None
    evidence_considered: int
    evidence_selected: int
    evidence_dropped: int
    selected_context_tokens: int | None
    token_counter_id: str | None

    @property
    def token_bound_was_measured(self) -> bool:
        return self.selected_context_tokens is not None


# --- references ------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextReferenceV1:
    """A model-visible handle and the authoritative identity it stands for.

    The handle is what the model cites (``[E1]``); the identities are what a
    validator resolves that citation back to.  Carrying the pair is what makes
    output validation possible without the pack holding a raw artifact.
    """

    handle: str
    kind: str
    evidence_id: str | None = None
    citation_id: str | None = None


@dataclass(frozen=True)
class ContextSupportGroupV1:
    """One canonical claim, and the model-visible handles that support it.

    ``canonical_slot`` is the Binder's own slot key, carried **verbatim**.  The
    compiler does not compose it, does not canonicalise it, and has no function
    that could: the identity of a claim is settled by slot binding upstream, and
    a context compiler that recomputed it would be a second binding authority
    wearing a different hat.

    ``supports`` holds handles, not identities, and every one of them is a
    handle the pack already carries.  That is enforced by the pack itself, so a
    group can never name evidence the model was not shown.

    One group is emitted per bound slot, including slots with a single support:
    "these three items are three separate claims" is worth as much to a reader
    as "these two are one", and a representation that only marked the
    interesting case would leave the ordinary one ambiguous.
    """

    handle: str
    canonical_slot: str
    supports: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextReferencesV1:
    """The pack's model-visible handles, and the topology between them.

    Two different questions, kept as two fields rather than mixed into one
    sequence: ``handles`` is the citation namespace the model writes in and a
    validator resolves against, and ``support_groups`` is *which of those
    handles say the same thing*.  A consumer that resolves a citation must not
    have to filter group entries out to do it.

    ``support_groups`` is a **projection of an authority, never an inference**.
    When the runtime carries no binding, this is empty and the pack says nothing
    rather than partitioning the evidence itself -- "the authority did not place
    this item" and "this item is its own claim" are different statements, and
    only the first is one a compiler is entitled to make.
    """

    handles: tuple[ContextReferenceV1, ...] = ()
    support_groups: tuple[ContextSupportGroupV1, ...] = ()

    @property
    def evidence_handles(self) -> tuple[str, ...]:
        """The evidence handles only, in order -- ``E1..En``, no ``C1``."""

        return tuple(
            reference.handle
            for reference in self.handles
            if reference.kind == "evidence"
        )


# --- the pack ----------------------------------------------------------------------------


def _frozen_view(value: Mapping[str, Any], *, where: str) -> Mapping[str, Any]:
    """A read-only copy of a disclosed projection.

    Nested mappings are refused outright.  That is not defensive style: a raw
    evidence packet is *distinguished* by carrying containers (``metadata``,
    ``temporal``), so refusing containers is the structural form of "no
    authoritative object was smuggled in".  A disclosed projection on the
    SPECIALIST profile is flat by construction, so nothing legitimate is lost.
    """

    view = dict(value)
    for name, item in view.items():
        if isinstance(item, Mapping):
            raise PackIntegrityError(
                f"{where} carries a nested mapping at {name!r}; a pack holds "
                f"disclosed projections only, and a container here is the shape "
                f"of a raw authority object"
            )
    return MappingProxyType(view)


@dataclass(frozen=True)
class AgentContextPackV1:
    """Governed, temporary, per-invocation context for one model call.

    Lifetime: it is built immediately before an invocation and is discarded once
    that invocation and its trace are complete.  It is not state, not a store,
    not a cache, and not a source of truth.  Nothing may read a pack to learn
    what the runtime believes -- only what one model was shown.

    What it is *not*, frozen as negative requirements:

    * not persistent runtime state, and not an ``ArtifactStore``;
    * not a replacement for ``EvidenceBinding``, ``ClaimProvenance`` or
      ``CalculationResult`` -- those remain the authorities a pack is derived
      from, and a pack never arbitrates between them;
    * not serialized ``RunState``;
    * not a model-generated plan;
    * not a container that lets a raw evidence or calculation object bypass
      Disclosure Authority.

    Static instructions are deliberately absent.  The prompt renderer owns
    system text, formatting and output-schema wording; a pack supplies governed
    *dynamic* context.  Putting instructions here would turn a context compiler
    into a universal prompt compiler, which is a different and much larger thing.
    """

    role: ContextRoleV1
    invocation_id: str
    query: str
    evidence: tuple[Mapping[str, Any], ...]
    calculation: Mapping[str, Any] | None
    references: ContextReferencesV1
    budget: ContextBudgetAccountingV1
    selection: ContextSelectionTraceV1

    def __post_init__(self) -> None:
        if not isinstance(self.role, ContextRoleV1):
            raise PackIntegrityError("role must be a ContextRoleV1")
        if not str(self.invocation_id).strip():
            raise PackIntegrityError("invocation_id must be non-empty")
        if not isinstance(self.query, str):
            raise PackIntegrityError("query must be a string")
        if not isinstance(self.references, ContextReferencesV1):
            raise PackIntegrityError("references must be a ContextReferencesV1")

        object.__setattr__(
            self,
            "evidence",
            tuple(
                _frozen_view(item, where=f"evidence[{index}]")
                for index, item in enumerate(self.evidence)
            ),
        )
        if self.calculation is not None:
            object.__setattr__(
                self,
                "calculation",
                _frozen_view(self.calculation, where="calculation"),
            )
        self._check_references()

    def _check_references(self) -> None:
        """The handles and the evidence they stand for must correspond.

        Two properties, both structural rather than promised.

        **One evidence handle per evidence item.**  A pack's handles are the
        citation namespace: a model writes ``[E1]`` and a validator resolves it,
        and the renderer is handed the pack rather than minting its own.  A pack
        whose handle list did not correspond to its evidence would render a
        citation that resolves to the wrong item or to nothing -- so the
        correspondence is checked where the handles are, not left to whichever
        consumer pairs them up.

        **A support group may only name handles the model is actually being
        shown.**  A group naming ``E3`` in a pack that carries two items would
        tell the model that a corroboration exists which the boundary did not
        release -- the claim and its evidence crossing separately, which is how
        a governed context starts asserting things the boundary never
        established.  Also refused: an empty group, and a group naming a handle
        twice.  A repeated handle is not a second witness, so reporting it as
        one would state a corroboration that does not exist -- the same failure
        H2A-3B2 removed from the renderer, one layer up.
        """

        evidence_handles = self.references.evidence_handles
        if len(evidence_handles) != len(self.evidence):
            raise PackIntegrityError(
                f"the pack carries {len(self.evidence)} evidence items and "
                f"{len(evidence_handles)} evidence handles; the handles are the "
                f"citation namespace and must correspond to the evidence"
            )

        cited = set(evidence_handles)
        for group in self.references.support_groups:
            if not group.supports:
                raise PackIntegrityError(
                    f"support group {group.handle!r} names no supports; a group "
                    f"with nothing in it states a claim the model cannot check"
                )
            unknown = [
                handle
                for handle in group.supports
                if handle not in cited
            ]
            if unknown:
                raise PackIntegrityError(
                    f"support group {group.handle!r} names {unknown}, which the "
                    f"pack does not carry as evidence"
                )
            if len(set(group.supports)) != len(group.supports):
                raise PackIntegrityError(
                    f"support group {group.handle!r} names a support twice; a "
                    f"repeated handle is not a second witness"
                )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """The authoritative identities of the selected evidence, in order.

        Read from the disclosed projections, which carry ``evidence_id`` on the
        SPECIALIST profile.  This is a convenience for traceability; the pack is
        not the authority for admission, and this list is not one either.
        """

        return tuple(
            str(item["evidence_id"])
            for item in self.evidence
            if item.get("evidence_id") is not None
        )
