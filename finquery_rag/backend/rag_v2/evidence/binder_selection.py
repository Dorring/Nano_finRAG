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
from .binder_fact_view import build_binder_fact_views, build_binder_fact_views_v2
from .prompt import BINDER_SYSTEM_PROMPT_V1


@dataclass(frozen=True)
class BinderSelectionDTOv1:
    """Selection-only provider DTO; status is never model-authored."""

    slots: Mapping[str, tuple[str, ...]]


class DuplicateFactHandleError(ValueError):
    """A parsed selection repeated a handle within one slot."""

    classification = "duplicate_fact_handle"


def fact_handle_map(request: BinderRequest) -> dict[str, str]:
    """Assign F01..Fn in frozen packet order, without filtering or ranking."""
    return {f"F{index:02d}": str(fact["fact_id"]) for index, fact in enumerate(request.facts, 1)}


def reverse_fact_handle_map(request: BinderRequest) -> dict[str, str]:
    return {fact_id: handle for handle, fact_id in fact_handle_map(request).items()}


def selection_schema(plan: SupervisorPlan, handles: Mapping[str, str]) -> dict[str, Any]:
    slot_schema = {
        "type": "array",
        "items": {"type": "string", "enum": list(handles)},
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


def provider_request(
    request: BinderRequest,
    *,
    fact_view_version: str = "v1",
    source_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    handles = fact_handle_map(request)
    if fact_view_version == "v1":
        public_facts = build_binder_fact_views(list(request.facts))
    elif fact_view_version == "v2":
        public_facts = build_binder_fact_views_v2(list(request.facts), source_by_candidate)
    else:
        raise ValueError(f"unsupported BinderFactView version: {fact_view_version}")
    payload = {
        "question": request.question,
        "intent": request.plan.intent.value,
        "operation": request.plan.operation,
        "required_slots": [slot.to_dict() for slot in request.plan.required_slots],
        "binder_fact_views": public_facts,
    }
    return payload, handles, selection_schema(request.plan, handles)


def build_selection_messages(
    request: BinderRequest,
    payload: Mapping[str, Any],
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    system = (system_prompt or BINDER_SYSTEM_PROMPT_V1).replace(
        "Return only the strict\nEvidenceBinding JSON schema.",
        "Return only the strict BinderSelectionDTOv1 JSON schema.",
    ).replace(
        "The output must contain exactly the schema fields and only IDs from the packet.",
        "The output must contain exactly the DTO fields and only supplied F-handles from the packet.",
    ) + (
        "\nThe DTO top-level object contains only `slots`. Each exact RequiredSlot ID maps directly to an array "
        "of zero or more supplied F-handles. Do not emit statuses, an overall query status, reasoning, answer, "
        "value, or calculation."
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
    values: dict[str, tuple[str, ...]] = {}
    for slot_id in expected:
        fact_values = payload["slots"][slot_id]
        if not isinstance(fact_values, list) or any(item not in handles for item in fact_values):
            raise ValueError(f"invalid selection value for {slot_id}")
        values[slot_id] = tuple(fact_values)
    return BinderSelectionDTOv1(values)


def selection_to_binding(dto: BinderSelectionDTOv1, request: BinderRequest, handles: Mapping[str, str]) -> EvidenceBinding:
    expected = {slot.slot_id for slot in request.plan.required_slots}
    if set(dto.slots) != expected:
        raise ValueError("adapter received a non-exact slot set")
    slot_bindings: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    for slot_id in (slot.slot_id for slot in request.plan.required_slots):
        selected_handles = dto.slots[slot_id]
        if len(selected_handles) != len(set(selected_handles)):
            raise DuplicateFactHandleError(f"duplicate_fact_handle:{slot_id}")
        fact_ids = tuple(handles[handle] for handle in selected_handles)
        if len(fact_ids) == 1:
            slot_bindings[slot_id] = fact_ids
        elif not fact_ids:
            missing.append(slot_id)
        else:
            slot_bindings[slot_id] = fact_ids
            ambiguous.append(slot_id)
    if all(len(dto.slots[slot_id]) == 1 for slot_id in expected):
        status = BindingStatus.BOUND.value
    elif ambiguous:
        status = BindingStatus.AMBIGUOUS.value
    else:
        status = BindingStatus.MISSING.value
    return EvidenceBinding(status=status, slot_bindings=slot_bindings, missing_slots=tuple(missing), ambiguous_slots=tuple(ambiguous))
