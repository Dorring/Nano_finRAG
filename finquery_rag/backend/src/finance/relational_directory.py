"""Resolving a relational result into something a reader or a check can use.

A relational result answers with `slot_id`s::

    ordering_groups = (("s2",), ("s1",))

That is the right authority -- it is semantic, it survives a different evidence
support being bound, and it cannot be disturbed by the order of `operands`.  But
it is not what a reader needs, and not what a validator can compare an answer
against.  Both want the same thing:

    (("Visa",), ("Apple",))     ->  "Visa > Apple"

`RelationalOperandDirectory` is that mapping, built once from
`SupervisorPlan.required_slots` and passed to both consumers.  Neither the
renderer nor the validator reads the plan, so neither can invent a second
`slot_id -> entity` lookup that disagrees with the other -- which is the failure
this split exists to prevent, and the same reason `SlotRetrievalRequestV1` is a
separate type rather than a field on the retrieval plan.

**Nothing here fetches anything.**  The directory carries only the coordinates a
relation is *stated* with -- entity, metric, period, role -- and deliberately no
evidence id, chunk, retrieval score or source text.  A resolution that needed
evidence would be a second bind, and binding already happened.

Stdlib and the calculation domain only: the renderer lives in the finance layer
and must be able to hold these types without reaching for the supervisor.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.calculation import (
    RELATIONAL_OPERATIONS,
    CalculationOperation,
    CalculationResult,
    ComparisonRelation,
)


@dataclass(frozen=True)
class RelationalOperandRef:
    """The coordinates a relational result is stated in.

    ``entity`` is the open-world mention and ``entity_id`` the ontology's
    identity for it, with the same one-way relationship as everywhere else in
    this codebase.  Either may be ``None``: a slot the plan left unqualified is
    still an operand, and refusing to resolve it would be inventing a
    requirement the plan did not state.
    """

    slot_id: str
    entity: str | None = None
    entity_id: str | None = None
    metric: str | None = None
    period: str | None = None
    role: str | None = None

    @property
    def label(self) -> str:
        """How this operand is named when the relation is stated.

        The entity if there is one, otherwise the metric, otherwise the slot id.
        The fallback order is fixed rather than clever: an answer that silently
        used a metric where the reader expected a company would be wrong in a
        way no test could see coming.  A slot with none of the three still
        resolves -- to its own id, which is at least traceable.
        """

        return self.entity or self.metric or self.slot_id


@dataclass(frozen=True)
class RelationalOperandDirectory:
    """`slot_id -> RelationalOperandRef`, built once from the plan's slots.

    A tuple rather than a mapping, so two directories built from the same slots
    compare equal and the order is a property of the plan rather than of how the
    dictionary was populated.
    """

    refs: tuple[RelationalOperandRef, ...] = ()

    def get(self, slot_id: str) -> RelationalOperandRef | None:
        for ref in self.refs:
            if ref.slot_id == slot_id:
                return ref
        return None

    def __contains__(self, slot_id: object) -> bool:
        return isinstance(slot_id, str) and self.get(slot_id) is not None


@dataclass(frozen=True)
class ResolvedRelationalResult:
    """A relational result whose refs have been named.

    ``calculation_id`` is carried through so a caller can tie the rendered or
    validated statement back to the result it came from; the resolver adds no
    identity of its own.
    """

    operation: CalculationOperation
    ordering_groups: tuple[tuple[RelationalOperandRef, ...], ...]
    calculation_id: str | None = None

    @property
    def relation(self) -> ComparisonRelation | None:
        """Derived, exactly as it is on the result -- same rule, same answer."""

        groups = self.ordering_groups
        if not groups:
            return None
        if sum(len(group) for group in groups) != 2:
            return None
        if len(groups) == 1:
            return ComparisonRelation.EQUAL
        if len(groups) == 2:
            return ComparisonRelation.FIRST_GT_SECOND
        return None

    def labels(self) -> tuple[tuple[str, ...], ...]:
        """The ordering as names, in the same shape and the same order."""

        return tuple(tuple(ref.label for ref in group) for group in self.ordering_groups)


def resolve_relational_result(
    result: CalculationResult,
    directory: RelationalOperandDirectory,
) -> ResolvedRelationalResult | None:
    """Name every ref in a relational result, or return ``None``.

    ``None`` means *this result cannot be stated*, and every reason for it is a
    reason to refuse rather than to fall back:

    * the result is not an admissible relational one;
    * a ref names a slot the directory does not have -- there is no label to
      give it, and inventing one from the slot id would state a relation in
      terms the question never used;
    * the ordering is empty, or has an empty group.

    The structural rules are `CalculationResult.relational_result_is_well_formed`'s
    and are not restated here; this adds the one thing that check cannot see,
    which is whether the *plan* can name what the ordering refers to.
    """

    if result.operation not in RELATIONAL_OPERATIONS:
        return None
    if not result.relational_result_is_well_formed:
        return None

    groups = result.ordering_groups or ()
    resolved: list[tuple[RelationalOperandRef, ...]] = []
    for group in groups:
        refs: list[RelationalOperandRef] = []
        for slot_id in group:
            ref = directory.get(slot_id)
            if ref is None:
                return None
            refs.append(ref)
        resolved.append(tuple(refs))

    return ResolvedRelationalResult(
        operation=result.operation,
        ordering_groups=tuple(resolved),
        calculation_id=result.calculation_id,
    )


def render_relational_result(resolved: ResolvedRelationalResult) -> str:
    """State a resolved ordering as text.  No computation, no lookup.

    A renderer that re-sorted by value, re-read the plan, or inferred a relation
    from anything but the groups it was handed would be a second executor.  This
    one only writes down what it was given:

        (("Visa",), ("Apple",))  ->  "Visa > Apple"
        (("A", "B"), ("C",))     ->  "A = B > C"

    ``=`` binds tighter than ``>`` in reading and in the grouping, which is the
    convention the groups already encode.
    """

    if not resolved.ordering_groups:
        return ""
    return " > ".join(" = ".join(group) for group in resolved.labels())
