"""Provider-facing constrained binding DTO for NF-V2-03 R1B.

This module deliberately sits outside the frozen EvidenceBinding contract.  It
uses query-local handles at the model boundary and deterministically adapts a
validated selection back to the existing internal fact IDs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding
from rag_v2.contracts.plan import SupervisorPlan

from .binder_service import BinderRequest
from .prompt import BINDER_SYSTEM_PROMPT_V1


SELECTION_STATUSES = (BindingStatus.BOUND.value, BindingStatus.MISSING.value, BindingStatus.AMBIGUOUS.value)


@dataclass(frozen=True)
class BinderSelectionSlot:
    status: str
    fact_handles: tuple[str, ...]


@dataclass(frozen=True)
class BinderSelectionDTOv1:
    slots: Mapping[str, BinderSelectionSlot]


def fact_handle_map(request: BinderRequest) -> dict[str, str]:
    """Assign F01..Fn in frozen packet order, without filtering or ranking."""
    return {f"F{index:02d}": str(fact["fact_id"]) for index, fact in enumerate(request.facts, 1)}


def reverse_fact_handle_map(request: BinderRequest) -> dict[str, str]:
    return {fact_id: handle for handle, fact_id in fact_handle_map(request).items()}


def selection_schema(plan: SupervisorPlan, handles: Mapping[str, str]) -> dict[str, Any]:
    slot_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "fact_handles"],
        "properties": {
            "status": {"type": "string", "enum": list(SELECTION_STATUSES)},
            "fact_handles": {
                "type": "array",
                "items": {"type": "string", "enum": list(handles)},
            },
        },
    }
    slot_ids = [slot.slot_id for slot in plan.required_slots]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["slots"],
        "properties": {
            "slots": {
                "type": "object",
                "additionalProperties": False,
                "required": slot_ids,
                "properties": {slot_id: slot_schema for slot_id in slot_ids},
            },
        },
    }


def provider_request(request: BinderRequest) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    handles = fact_handle_map(request)
    public_facts: list[dict[str, Any]] = []
    for handle, fact in handles.items():
        original = next(item for item in request.facts if str(item["fact_id"]) == fact)
        projection = {key: value for key, value in dict(original).items() if key not in {"fact_id", "candidate_ids"}}
        projection["fact_handle"] = handle
        public_facts.append(projection)
    payload = {
        "question": request.question,
        "intent": request.plan.intent.value,
        "operation": request.plan.operation,
        "required_slots": [slot.to_dict() for slot in request.plan.required_slots],
        "fact_handles": public_facts,
    }
    return payload, handles, selection_schema(request.plan, handles)


def build_selection_messages(request: BinderRequest, payload: Mapping[str, Any]) -> list[dict[str, str]]:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    system = BINDER_SYSTEM_PROMPT_V1.replace(
        "Return only the strict\nEvidenceBinding JSON schema.",
        "Return only the strict BinderSelectionDTOv1 JSON schema.",
    ).replace(
        "The output must contain exactly the schema fields and only IDs from the packet.",
        "The output must contain exactly the DTO fields and only supplied F-handles from the packet.",
    ) + (
        "\nThe DTO top-level object contains only `slots`; use exactly the RequiredSlot IDs as slot properties "
        "and only the supplied F-handles. Do not emit an overall query status, reasoning, answer, value, or calculation."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Constrained BinderRequest JSON:\n{body}"},
    ]


def parse_selection(payload: Any, plan: SupervisorPlan, handles: Mapping[str, str]) -> BinderSelectionDTOv1:
    if not isinstance(payload, dict) or set(payload) != {"slots"} or not isinstance(payload["slots"], dict):
        raise ValueError("BinderSelectionDTOv1 must contain only slots")
    expected = {slot.slot_id for slot in plan.required_slots}
    if set(payload["slots"]) != expected:
        raise ValueError("BinderSelectionDTOv1 slot properties must exactly match RequiredSlots")
    values: dict[str, BinderSelectionSlot] = {}
    for slot_id in expected:
        value = payload["slots"][slot_id]
        if not isinstance(value, dict) or set(value) != {"status", "fact_handles"}:
            raise ValueError(f"invalid selection fields for {slot_id}")
        status = value["status"]
        fact_values = value["fact_handles"]
        if status not in SELECTION_STATUSES or not isinstance(fact_values, list) or any(item not in handles for item in fact_values):
            raise ValueError(f"invalid selection value for {slot_id}")
        if len(fact_values) != len(set(fact_values)):
            raise ValueError(f"duplicate fact handle for {slot_id}")
        if status == BindingStatus.BOUND.value and len(fact_values) != 1:
            raise ValueError(f"BOUND cardinality violation for {slot_id}")
        if status == BindingStatus.MISSING.value and fact_values:
            raise ValueError(f"MISSING cardinality violation for {slot_id}")
        if status == BindingStatus.AMBIGUOUS.value and len(fact_values) < 2:
            raise ValueError(f"AMBIGUOUS cardinality violation for {slot_id}")
        values[slot_id] = BinderSelectionSlot(status, tuple(fact_values))
    return BinderSelectionDTOv1(values)


def selection_to_binding(dto: BinderSelectionDTOv1, request: BinderRequest, handles: Mapping[str, str]) -> EvidenceBinding:
    expected = {slot.slot_id for slot in request.plan.required_slots}
    if set(dto.slots) != expected:
        raise ValueError("adapter received a non-exact slot set")
    slot_bindings: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    for slot_id in (slot.slot_id for slot in request.plan.required_slots):
        selected = dto.slots[slot_id]
        fact_ids = tuple(handles[handle] for handle in selected.fact_handles)
        if selected.status == BindingStatus.BOUND.value:
            slot_bindings[slot_id] = fact_ids
        elif selected.status == BindingStatus.MISSING.value:
            missing.append(slot_id)
        else:
            slot_bindings[slot_id] = fact_ids
            ambiguous.append(slot_id)
    if all(dto.slots[slot_id].status == BindingStatus.BOUND.value for slot_id in expected):
        status = BindingStatus.BOUND.value
    elif ambiguous:
        status = BindingStatus.AMBIGUOUS.value
    else:
        status = BindingStatus.MISSING.value
    return EvidenceBinding(status=status, slot_bindings=slot_bindings, missing_slots=tuple(missing), ambiguous_slots=tuple(ambiguous))
