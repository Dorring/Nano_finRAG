from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding
from rag_v2.contracts.plan import SupervisorPlan


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
        if any(len(ids) != 1 for ids in binding.slot_bindings.values()):
            reasons.append("bound_fact_cardinality_mismatch")
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
