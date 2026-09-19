"""The retrieval demand: one lane per required slot, and nothing else decides it.

`SupervisorPlan.required_slots` is the Harness's statement of *what facts the
question needs*.  The retrieval layer used to re-derive that count from the
question text -- `route_question` -> `profile.task_type` -> `build_operand_slots`
-- and got it wrong in a way that was invisible from either side: every
cross-entity comparison and ranking classified as `general_single_fact` or
`table_single_fact`, which emit **exactly one** `OperandSlot`, so a four-company
ranking shared one retrieval pool and one top-40.

Measured on the twenty cross-entity fixtures: `operand_slots == 1` for all
twenty, `is_multi_slot` therefore False, `build_slot_pool` never called, and
0/21 gold facts reaching the packet while 157/157 sat in the index.

`OperandSlot` is also structurally unable to express the demand: it has no
`entity` field, so the entity is re-attached plan-wide by `_entity_terms`, which
is why one slot's query reads

    "and JPMorganChase larger | FY2025 | tsla | jpmorganchase"

-- both companies in every slot's query.

This module is the explicit form of the demand.  It is deliberately a separate
type from `OperandSlot` rather than an extra field on it: `OperandSlot` answers
*how to search*, this answers *what is wanted*, and the refactor exists to stop
the first from being able to change the second's cardinality.

**The invariant.**  A sequence of these has exactly one entry per
`RequiredSlot`, in the same order.  Nothing downstream may add, drop, merge or
reorder a lane; a component that wants to is expressing a retrieval policy
question, not a demand.

Stdlib only, and no `rag_v2` import, so both the retrieval package and the
runtime layer that composes it can hold the type.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlotRetrievalRequestV1:
    """One slot's retrieval demand, copied from the `RequiredSlot` it came from.

    The coordinates are carried verbatim rather than re-derived.  ``entity`` is
    the open-world mention and ``entity_id`` the ontology's identity for it, with
    the same one-way relationship and the same meaning as on `RequiredSlot`:
    ``entity_id is None`` says *the mention is constrained and the ontology
    cannot name it*, never *there is no entity constraint*.  A retrieval lane
    that read the second as the first would match every company that reports the
    metric, which is the defect the entity field was added to remove.
    """

    slot_id: str
    metric: str
    period: str
    role: str
    value_type: str
    unit: str | None = None
    entity: str | None = None
    entity_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("slot_id", "metric", "period", "role", "value_type"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("unit", "entity", "entity_id"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be None or a string")

    @property
    def query(self) -> str:
        """The deterministic retrieval query for this slot.

        ``<entity> <metric> <period>`` and nothing else.  Deliberately not the
        question, not a paraphrase of it, and not any other slot's terms: the
        measured failure was exactly that cross-slot contamination, and a
        per-slot lane whose query still names the other slots is a per-slot lane
        in name only.

        The entity mention is used rather than ``entity_id`` because that is
        what was measured -- an entity-scoped query built this way found 47/47
        gold facts at a median rank of 20 against the production query's 39/47
        at a median of 129.  Swapping in the canonical id is an untested
        variation on that result, not a neutral tidy-up.

        A slot may legitimately carry no entity (a single-company question), in
        which case the query is the metric and period alone.
        """

        parts = (self.entity, self.metric, self.period)
        return " ".join(part.strip() for part in parts if part and part.strip())
