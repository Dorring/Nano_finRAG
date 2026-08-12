"""Slot-wise discriminative provider boundary for NF-V2-03 R5.

This module changes only the provider-facing task formulation.  The internal
EvidenceBinding contract and deterministic validator remain the same as R1B.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding
from rag_v2.contracts.plan import SupervisorPlan

from .binder_fact_view import build_binder_fact_views, build_binder_fact_views_v2
from .binder_provider import (
    BinderProviderError,
    BinderProviderResult,
    BailianBinderProvider,
    _exception_chain,
    _exception_http_status,
    _safe_message,
)
from .binder_service import BinderRequest
from .binder_selection import DuplicateFactHandleError, fact_handle_map


SLOTWISE_FORMULATION = "slotwise_discriminative_v1"

SLOTWISE_SYSTEM_PROMPT = """You are a slot-wise evidence selection controller for a financial RAG system.

Each task is an independent evidence-selection problem. Solve every RequiredSlot independently:
1. identify the exact requested metric;
2. enforce the requested reporting period;
3. enforce scope or segment;
4. use row, header, table, statement, and section context;
5. return exactly one handle when one candidate uniquely satisfies all constraints;
6. return an empty list when no candidate satisfies all material constraints;
7. return more than one handle only when candidates remain genuinely indistinguishable.

Do not let one slot's decision change another slot's decision. Do not globally allocate facts.
A fact may support multiple slots when it independently satisfies each slot.

For calculation tasks, evaluate every operand slot independently and respect its operation and
operand role (for example current/prior, numerator/denominator, component/total, or
minuend/subtrahend). Do not calculate, answer, explain, or emit a numeric result.

Use only supplied handles. Do not invent facts, slots, sources, answers, or reasoning.
Return only the strict JSON object with a `tasks` object whose exact slot IDs map to arrays of
query-local fact handles.
"""


@dataclass(frozen=True)
class SlotBindingTaskV1:
    slot_id: str
    requirement: Mapping[str, Any]
    candidate_handles: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "requirement": dict(self.requirement),
            "candidate_handles": list(self.candidate_handles),
        }


@dataclass(frozen=True)
class SlotwiseSelectionDTOv1:
    tasks: Mapping[str, tuple[str, ...]]


def slotwise_provider_request(
    request: BinderRequest,
    *,
    fact_view_version: str = "v2",
    source_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    handles = fact_handle_map(request)
    if fact_view_version == "v1":
        views = build_binder_fact_views(list(request.facts))
    elif fact_view_version == "v2":
        views = build_binder_fact_views_v2(list(request.facts), source_by_candidate)
    else:
        raise ValueError(f"unsupported BinderFactView version: {fact_view_version}")
    facts = {view["fact_handle"]: view for view in views}
    tasks: dict[str, dict[str, Any]] = {}
    slot_ids: list[str] = []
    for slot in request.plan.required_slots:
        slot_ids.append(slot.slot_id)
        requirement = slot.to_dict()
        requirement["scope"] = getattr(slot, "scope", None)
        requirement["operation"] = request.plan.operation
        requirement["operand_role"] = slot.role
        tasks[slot.slot_id] = {
            "requirement": requirement,
            "candidate_handles": list(handles),
        }
    payload = {
        "question": request.question,
        "intent": request.plan.intent.value,
        "operation": request.plan.operation,
        "facts": facts,
        "tasks": tasks,
    }
    slot_schema = {
        "type": "array",
        "items": {"type": "string", "enum": list(handles)},
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["tasks"],
        "properties": {
            "tasks": {
                "type": "object",
                "additionalProperties": False,
                "required": slot_ids,
                "properties": {slot_id: slot_schema for slot_id in slot_ids},
            },
        },
    }
    return payload, handles, schema


def build_slotwise_selection_messages(
    request: BinderRequest,
    payload: Mapping[str, Any],
    *,
    system_prompt: str = SLOTWISE_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"SlotBindingTaskV1 request JSON:\n{body}"},
    ]


def parse_slotwise_selection(payload: Any, plan: SupervisorPlan, handles: Mapping[str, str]) -> SlotwiseSelectionDTOv1:
    if not isinstance(payload, dict) or set(payload) != {"tasks"} or not isinstance(payload["tasks"], dict):
        raise ValueError("SlotwiseSelectionDTOv1 must contain only tasks")
    expected = {slot.slot_id for slot in plan.required_slots}
    if set(payload["tasks"]) != expected:
        raise ValueError("slot-wise task properties must exactly match RequiredSlots")
    values: dict[str, tuple[str, ...]] = {}
    for slot_id in expected:
        selected = payload["tasks"][slot_id]
        if not isinstance(selected, list) or any(item not in handles for item in selected):
            raise ValueError(f"invalid slot-wise selection value for {slot_id}")
        if len(selected) != len(set(selected)):
            raise DuplicateFactHandleError(f"duplicate_fact_handle:{slot_id}")
        values[slot_id] = tuple(selected)
    return SlotwiseSelectionDTOv1(values)


def slotwise_selection_to_binding(dto: SlotwiseSelectionDTOv1, request: BinderRequest, handles: Mapping[str, str]) -> EvidenceBinding:
    expected = [slot.slot_id for slot in request.plan.required_slots]
    if set(dto.tasks) != set(expected):
        raise ValueError("slot-wise adapter received a non-exact slot set")
    slot_bindings: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    for slot_id in expected:
        fact_ids = tuple(handles[handle] for handle in dto.tasks[slot_id])
        if len(fact_ids) == 0:
            missing.append(slot_id)
        elif len(fact_ids) == 1:
            slot_bindings[slot_id] = fact_ids
        else:
            slot_bindings[slot_id] = fact_ids
            ambiguous.append(slot_id)
    if all(len(dto.tasks[slot_id]) == 1 for slot_id in expected):
        status = BindingStatus.BOUND.value
    elif ambiguous:
        status = BindingStatus.AMBIGUOUS.value
    else:
        status = BindingStatus.MISSING.value
    return EvidenceBinding(
        status=status,
        slot_bindings=slot_bindings,
        missing_slots=tuple(missing),
        ambiguous_slots=tuple(ambiguous),
    )


class SlotwiseBinderAdapterError(BinderProviderError):
    schema_valid = True
    adapter_failure = True
    classification = "slotwise_adapter_failure"


class BailianSlotwiseBinderProvider(BailianBinderProvider):
    """One-call Bailian provider for independent slot selection."""

    provider_role = "evidence_binder"
    formulation = SLOTWISE_FORMULATION

    def __init__(
        self,
        *args: Any,
        system_prompt: str | None = None,
        fact_view_version: str = "v2",
        source_metadata_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = system_prompt or SLOTWISE_SYSTEM_PROMPT
        self.fact_view_version = fact_view_version
        self.source_metadata_by_candidate = dict(source_metadata_by_candidate or {})

    def bind(self, request: Any) -> BinderProviderResult:
        if not isinstance(request, BinderRequest):
            request = BinderRequest(
                question_id=str(request["question_id"]),
                question=str(request["question"]),
                plan=SupervisorPlan.from_dict({
                    "intent": request["intent"],
                    "required_slots": request["required_slots"],
                    "operation": request["operation"],
                    "next_action": "RETRIEVE",
                }),
                facts=tuple(request.get("financial_facts") or ()),
            )
        started = time.perf_counter()
        self.last_raw_response = None
        response: Any | None = None
        payload, handles, schema = slotwise_provider_request(
            request,
            fact_view_version=self.fact_view_version,
            source_by_candidate=self.source_metadata_by_candidate,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=build_slotwise_selection_messages(request, payload, system_prompt=self.system_prompt),
                temperature=self.temperature,
                response_format={"type": "json_schema", "json_schema": {"name": "SlotwiseSelectionDTOv1", "strict": True, "schema": schema}},
                extra_body={"enable_thinking": self.enable_thinking},
            )
            message = response.choices[0].message if response.choices else None
            raw = getattr(message, "content", None) if message is not None else None
            self.last_raw_response = raw if isinstance(raw, str) else None
            if not self.last_raw_response or not self.last_raw_response.strip():
                raise BinderProviderError("Bailian returned an empty SlotwiseSelectionDTOv1 response")
            try:
                parsed = json.loads(self.last_raw_response.strip())
            except json.JSONDecodeError as exc:
                raise BinderProviderError("Bailian SlotwiseSelectionDTOv1 response was not strict JSON") from exc
            dto = parse_slotwise_selection(parsed, request.plan, handles)
            binding = slotwise_selection_to_binding(dto, request, handles)
            metadata = self._metadata(response, started, structured=True, raw_content_length=len(self.last_raw_response), request_id=getattr(response, "id", None))
            self.last_call = metadata
            return BinderProviderResult(binding=binding, metadata=metadata, raw_response=self.last_raw_response)
        except DuplicateFactHandleError as exc:
            metadata = self._metadata(response, started, structured=True, error=str(exc), provider_success=response is not None, exception_type=type(exc).__name__, raw_content_length=len(self.last_raw_response or ""), request_id=getattr(response, "id", None), http_status=_exception_http_status(exc), exception_chain=_exception_chain(exc))
            self.last_call = metadata
            raise SlotwiseBinderAdapterError(str(exc)) from exc
        except BinderProviderError as exc:
            cause = exc.__cause__ or exc.__context__
            metadata = self._metadata(response, started, structured=False, error=str(exc), provider_success=response is not None, exception_type=type(exc).__name__, exception_cause_type=type(cause).__name__ if cause is not None else None, exception_cause_message=_safe_message(cause) if cause is not None else None, raw_content_length=len(self.last_raw_response or ""), request_id=getattr(response, "id", None), http_status=_exception_http_status(exc), exception_chain=_exception_chain(exc))
            self.last_call = metadata
            raise
        except Exception as exc:
            cause = exc.__cause__ or exc.__context__
            metadata = self._metadata(response, started, structured=False, error=_safe_message(exc), provider_success=False, exception_type=type(exc).__name__, exception_cause_type=type(cause).__name__ if cause is not None else None, exception_cause_message=_safe_message(cause) if cause is not None else None, errno=getattr(exc, "errno", None), raw_content_length=len(self.last_raw_response or ""), request_id=getattr(response, "id", None), http_status=_exception_http_status(exc), exception_chain=_exception_chain(exc))
            self.last_call = metadata
            raise BinderProviderError(f"Bailian slot-wise binder API call failed: {_safe_message(exc)}") from exc
