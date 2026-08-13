"""Fail-closed selective admission for the final NF-V2-03 Binder architecture.

This module is deliberately semantic-agnostic.  It only releases an already
selected fact when every frozen structural and safety precondition is true;
otherwise it emits a safe ``MISSING`` or ``AMBIGUOUS`` binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding


SELECTIVE_ADMISSION_CONTRACT = "SelectiveBindingAdmissionV1"


@dataclass(frozen=True)
class SelectiveAdmissionResult:
    """Audit record for one fail-closed admission decision."""

    binding: EvidenceBinding
    released: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": SELECTIVE_ADMISSION_CONTRACT,
            "released": self.released,
            "reasons": list(self.reasons),
            "binding": self.binding.to_dict(),
        }


def admit_selective_binding(
    *,
    slot_ids: Iterable[str],
    slot_bindings: Mapping[str, Iterable[str]],
    packet_facts: Iterable[Mapping[str, Any]],
    binding_validator_pass: bool,
    unique_admissible_selection: bool,
    source_relation_valid: bool = True,
    cardinality_valid: bool = True,
    safety_condition_pass: bool = True,
    fallback_status: str = BindingStatus.MISSING.value,
) -> SelectiveAdmissionResult:
    """Release BOUND only when all deterministic safety gates pass.

    No metric, period, role, Gold, or semantic comparison is performed here.
    The caller supplies the already-produced selection and structural audit
    flags.  Any failed condition is converted to a fail-closed status.
    """

    expected_slots = tuple(str(slot_id) for slot_id in slot_ids)
    allowed_slots = set(expected_slots)
    facts = {str(fact.get("fact_id")): fact for fact in packet_facts if fact.get("fact_id")}
    normalized: dict[str, tuple[str, ...]] = {
        str(slot_id): tuple(str(fact_id) for fact_id in fact_ids)
        for slot_id, fact_ids in slot_bindings.items()
    }
    reasons: list[str] = []
    if set(normalized) != allowed_slots:
        reasons.append("selected_slot_set_mismatch")
    if any(len(ids) != 1 for ids in normalized.values()):
        reasons.append("cardinality_violation")
    selected = [fact_id for ids in normalized.values() for fact_id in ids]
    if len(selected) != len(set(selected)):
        reasons.append("duplicate_fact_selection")
    for fact_id in selected:
        fact = facts.get(fact_id)
        if fact is None:
            reasons.append(f"unknown_fact:{fact_id}")
        elif fact.get("provenance_complete") is not True:
            reasons.append(f"incomplete_provenance:{fact_id}")
    if not source_relation_valid:
        reasons.append("source_relation_invalid")
    if not cardinality_valid:
        reasons.append("structural_cardinality_invalid")
    if not binding_validator_pass:
        reasons.append("binding_validator_failed")
    if not unique_admissible_selection:
        reasons.append("selection_not_unique")
    if not safety_condition_pass:
        reasons.append("safety_condition_failed")

    if not reasons:
        binding = EvidenceBinding(status=BindingStatus.BOUND.value, slot_bindings=normalized)
        return SelectiveAdmissionResult(binding=binding, released=True, reasons=())

    if fallback_status == BindingStatus.AMBIGUOUS.value:
        binding = EvidenceBinding(
            status=BindingStatus.AMBIGUOUS.value,
            slot_bindings={},
            ambiguous_slots=expected_slots,
        )
    else:
        binding = EvidenceBinding(
            status=BindingStatus.MISSING.value,
            slot_bindings={},
            missing_slots=expected_slots,
        )
    return SelectiveAdmissionResult(binding=binding, released=False, reasons=tuple(reasons))

