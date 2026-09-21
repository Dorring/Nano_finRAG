from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding
from rag_v2.contracts.financial_semantics import quantity_identity
from rag_v2.contracts.plan import SupervisorPlan


def fact_quantity_key(fact: Mapping[str, Any]) -> str:
    """The canonical quantity a bound fact states, as the shared semantics read it.

    Not a comparison of this module's own: ``1 million`` and ``1000 thousand``
    have to come out equal here for the same reason they do in the conflict
    gate and the content fingerprint, and a second notion of "same quantity"
    is how those three drifted apart the first time.
    """

    value = fact.get("parsed_numeric_value")
    if value is None:
        value = fact.get("value")
    if value is None:
        value = fact.get("raw_value")
    return quantity_identity(
        value,
        scale=fact.get("scale"),
        unit=fact.get("unit"),
        currency=fact.get("currency"),
    )


def support_source(fact: Mapping[str, Any], fact_id: str) -> str:
    """Which physical witness a bound fact speaks for.

    The same ladder the consensus uses when it decides whether a row has already
    been counted, so "independent" means one thing in both places.  A fact with
    no stated source is its own witness, which is the conservative reading: it
    cannot silently merge with another unsourced row.
    """

    return str(
        fact.get("physical_source_id") or fact.get("source_id") or fact_id
    ).strip()


@dataclass(frozen=True)
class BindingValidationResult:
    passed: bool
    final_status: str
    reasons: tuple[str, ...]
    selected_fact_ids: tuple[str, ...]
    bound_slot_count: int
    missing_slots: tuple[str, ...]
    ambiguous_slots: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "final_status": self.final_status,
            "reasons": list(self.reasons),
            "selected_fact_ids": list(self.selected_fact_ids),
            "bound_slot_count": self.bound_slot_count,
            "missing_slots": list(self.missing_slots),
            "ambiguous_slots": list(self.ambiguous_slots),
        }


def _fact_ids(facts: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(fact.get("fact_id")): fact for fact in facts if fact.get("fact_id")}


def validate_binding(
    binding: EvidenceBinding,
    plan: SupervisorPlan,
    facts: Iterable[Mapping[str, Any]],
) -> BindingValidationResult:
    """Validate structural safety without reintroducing metric equality."""

    allowed_slots = {slot.slot_id for slot in plan.required_slots}
    fact_map = _fact_ids(facts)
    reasons: list[str] = []
    selected: list[str] = []
    for slot_id, ids in binding.slot_bindings.items():
        if slot_id not in allowed_slots:
            reasons.append(f"unknown_slot:{slot_id}")
        if not ids:
            reasons.append(f"empty_fact_binding:{slot_id}")
        for fact_id in ids:
            selected.append(fact_id)
            fact = fact_map.get(fact_id)
            if fact is None:
                reasons.append(f"unknown_fact:{fact_id}")
            elif fact.get("provenance_complete") is not True:
                reasons.append(f"incomplete_provenance:{fact_id}")
    if len(selected) != len(set(selected)):
        reasons.append("duplicate_fact_across_slots")
    if any(slot_id not in allowed_slots for slot_id in binding.missing_slots):
        reasons.append("unknown_missing_slot")
    if any(slot_id not in allowed_slots for slot_id in binding.ambiguous_slots):
        reasons.append("unknown_ambiguous_slot")
    if binding.status == BindingStatus.BOUND:
        if set(binding.slot_bindings) != allowed_slots:
            reasons.append("bound_slot_cardinality_mismatch")
        # H2A-2C-2: a slot may carry several *independent supports of one
        # canonical fact*.  Empty is still wrong, and so is a set whose members
        # disagree about the quantity they state -- permitting arrays without
        # this would let a provider bind `1 billion` and `1.2 billion` to one
        # slot and call it corroboration.
        #
        # This is deliberately not the consensus rule.  Which candidate set wins,
        # and how ties fail closed, is arbitration and stays in the binding
        # layer; what is checked here is the *contract* the arbitration output
        # has to satisfy, using the same canonical quantity semantics every
        # other comparison in the repository uses.
        if any(len(ids) < 1 for ids in binding.slot_bindings.values()):
            reasons.append("bound_fact_cardinality_mismatch")
        for slot_id, ids in binding.slot_bindings.items():
            in_slot = [(fact_id, fact_map[fact_id]) for fact_id in ids if fact_id in fact_map]
            stated = {fact_quantity_key(fact) for _fact_id, fact in in_slot}
            if len(stated) > 1:
                reasons.append(f"slot_supports_disagree:{slot_id}")
            # A support set is independent *witnesses*, not rows.  Two rows from
            # one physical source agreeing with each other is one source, and
            # three rows from it must not read as a corroborated fact.  This is
            # the same policy `_consensus_fact_for_slot` applies when it builds
            # the set -- checked here as a contract on whatever a caller
            # supplies directly, without re-deriving which set should have won.
            sources = [
                support_source(fact, fact_id) for fact_id, fact in in_slot
            ]
            if len(set(sources)) != len(sources):
                reasons.append(f"slot_supports_share_source:{slot_id}")
        if binding.missing_slots or binding.ambiguous_slots or binding.invalid_reasons:
            reasons.append("bound_has_error_fields")
    elif binding.status == BindingStatus.MISSING:
        if not binding.missing_slots:
            reasons.append("missing_without_slots")
    elif binding.status == BindingStatus.AMBIGUOUS:
        if not binding.ambiguous_slots:
            reasons.append("ambiguous_without_slots")
    elif binding.status == BindingStatus.INVALID and not binding.invalid_reasons:
        reasons.append("invalid_without_reasons")
    return BindingValidationResult(
        passed=not reasons,
        final_status=binding.status if not reasons else BindingStatus.INVALID.value,
        reasons=tuple(reasons),
        selected_fact_ids=tuple(dict.fromkeys(selected)),
        bound_slot_count=len(binding.slot_bindings),
        missing_slots=tuple(binding.missing_slots),
        ambiguous_slots=tuple(binding.ambiguous_slots),
    )
