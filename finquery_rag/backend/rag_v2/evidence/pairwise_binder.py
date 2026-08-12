"""Batched pairwise compatibility provider boundary for NF-V2-03 R5.1."""

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


PAIRWISE_FORMULATION = "batched_pairwise_compatibility_v1"
PAIRWISE_LABELS = ("MATCH", "REJECT", "INDETERMINATE")

PAIRWISE_SYSTEM_PROMPT = """You are a pairwise financial evidence compatibility classifier.

Each RequiredSlot is independent. For every supplied candidate handle under every slot,
classify only whether that candidate is compatible with the complete requested slot.

MATCH means the visible evidence satisfies every material constraint: exact metric and
scope, requested period, statement or table context, and operand role when applicable.
REJECT means visible evidence proves a material conflict such as a wrong metric, period,
scope, statement, or operand identity. INDETERMINATE means the candidate may be relevant,
but the supplied evidence does not prove a complete match or a material conflict.

Do not treat lexical similarity as MATCH. Do not use a parent metric for a narrower metric,
another period, another segment, or another statement. Do not let one slot affect another.
For calculation slots, respect the explicit operand role independently; do not calculate,
answer, explain, or emit numbers.

Classify every listed handle exactly once with one of MATCH, REJECT, or INDETERMINATE.
Use only supplied slot IDs and fact handles. Return only the strict JSON object with a
`tasks` object. Do not return a selection, status, source ID, answer, or reasoning.
"""


@dataclass(frozen=True)
class CandidateCompatibilityDTOv1:
    tasks: Mapping[str, Mapping[str, str]]


def pairwise_provider_request(
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
    label_schema = {"type": "string", "enum": list(PAIRWISE_LABELS)}
    task_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(handles),
        "properties": {handle: label_schema for handle in handles},
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
                "properties": {slot_id: task_schema for slot_id in slot_ids},
            },
        },
    }
    return payload, handles, schema


def build_pairwise_messages(
    request: BinderRequest,
    payload: Mapping[str, Any],
    *,
    system_prompt: str = PAIRWISE_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CandidateCompatibilityDTOv1 request JSON:\n{body}"},
    ]


def parse_pairwise_compatibility(
    payload: Any,
    plan: SupervisorPlan,
    handles: Mapping[str, str],
) -> CandidateCompatibilityDTOv1:
    if not isinstance(payload, dict) or set(payload) != {"tasks"} or not isinstance(payload["tasks"], dict):
        raise ValueError("CandidateCompatibilityDTOv1 must contain only tasks")
    expected_slots = {slot.slot_id for slot in plan.required_slots}
    if set(payload["tasks"]) != expected_slots:
        raise ValueError("pairwise task properties must exactly match RequiredSlots")
    expected_handles = set(handles)
    values: dict[str, dict[str, str]] = {}
    for slot_id in expected_slots:
        classifications = payload["tasks"][slot_id]
        if not isinstance(classifications, dict) or set(classifications) != expected_handles:
            raise ValueError(f"pairwise handles must exactly match the query packet for {slot_id}")
        normalized: dict[str, str] = {}
        for handle, label in classifications.items():
            if handle not in handles or label not in PAIRWISE_LABELS:
                raise ValueError(f"invalid pairwise classification for {slot_id}:{handle}")
            normalized[handle] = label
        values[slot_id] = normalized
    return CandidateCompatibilityDTOv1(values)


def pairwise_compatibility_to_binding(
    dto: CandidateCompatibilityDTOv1,
    request: BinderRequest,
    handles: Mapping[str, str],
) -> tuple[EvidenceBinding, dict[str, dict[str, str]]]:
    expected_slots = [slot.slot_id for slot in request.plan.required_slots]
    if set(dto.tasks) != set(expected_slots):
        raise ValueError("pairwise adapter received a non-exact slot set")
    slot_bindings: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    outcomes: dict[str, dict[str, str]] = {}
    for slot_id in expected_slots:
        classifications = dict(dto.tasks[slot_id])
        outcomes[slot_id] = classifications
        matches = [handle for handle, label in classifications.items() if label == "MATCH"]
        indeterminate = [handle for handle, label in classifications.items() if label == "INDETERMINATE"]
        if len(matches) == 1:
            slot_bindings[slot_id] = (handles[matches[0]],)
        elif len(matches) >= 2:
            slot_bindings[slot_id] = tuple(handles[handle] for handle in matches)
            ambiguous.append(slot_id)
        elif indeterminate:
            ambiguous.append(slot_id)
        else:
            missing.append(slot_id)
    if all(
        len([label for label in dto.tasks[slot_id].values() if label == "MATCH"]) == 1
        for slot_id in expected_slots
    ):
        status = BindingStatus.BOUND.value
    elif ambiguous:
        status = BindingStatus.AMBIGUOUS.value
    else:
        status = BindingStatus.MISSING.value
    binding = EvidenceBinding(
        status=status,
        slot_bindings=slot_bindings,
        missing_slots=tuple(missing),
        ambiguous_slots=tuple(ambiguous),
    )
    return binding, outcomes


class PairwiseBinderAdapterError(BinderProviderError):
    schema_valid = True
    adapter_failure = True
    classification = "pairwise_adapter_failure"


class BailianPairwiseBinderProvider(BailianBinderProvider):
    """One-call Bailian provider for batched pairwise compatibility labels."""

    provider_role = "evidence_binder"
    formulation = PAIRWISE_FORMULATION

    def __init__(
        self,
        *args: Any,
        system_prompt: str | None = None,
        fact_view_version: str = "v2",
        source_metadata_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = system_prompt or PAIRWISE_SYSTEM_PROMPT
        self.fact_view_version = fact_view_version
        self.source_metadata_by_candidate = dict(source_metadata_by_candidate or {})
        self.last_pairwise_outcomes: dict[str, dict[str, str]] | None = None

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
        self.last_pairwise_outcomes = None
        response: Any | None = None
        payload, handles, schema = pairwise_provider_request(
            request,
            fact_view_version=self.fact_view_version,
            source_by_candidate=self.source_metadata_by_candidate,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=build_pairwise_messages(request, payload, system_prompt=self.system_prompt),
                temperature=self.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "CandidateCompatibilityDTOv1",
                        "strict": True,
                        "schema": schema,
                    },
                },
                extra_body={"enable_thinking": self.enable_thinking},
            )
            message = response.choices[0].message if response.choices else None
            raw = getattr(message, "content", None) if message is not None else None
            self.last_raw_response = raw if isinstance(raw, str) else None
            if not self.last_raw_response or not self.last_raw_response.strip():
                raise BinderProviderError("Bailian returned an empty CandidateCompatibilityDTOv1 response")
            try:
                parsed = json.loads(self.last_raw_response.strip())
            except json.JSONDecodeError as exc:
                raise BinderProviderError("Bailian CandidateCompatibilityDTOv1 response was not strict JSON") from exc
            dto = parse_pairwise_compatibility(parsed, request.plan, handles)
            binding, outcomes = pairwise_compatibility_to_binding(dto, request, handles)
            self.last_pairwise_outcomes = outcomes
            metadata = self._metadata(
                response,
                started,
                structured=True,
                raw_content_length=len(self.last_raw_response),
                request_id=getattr(response, "id", None),
            )
            self.last_call = metadata
            return BinderProviderResult(binding=binding, metadata=metadata, raw_response=self.last_raw_response)
        except DuplicateFactHandleError as exc:
            metadata = self._metadata(
                response,
                started,
                structured=True,
                error=str(exc),
                provider_success=response is not None,
                exception_type=type(exc).__name__,
                raw_content_length=len(self.last_raw_response or ""),
                request_id=getattr(response, "id", None),
                http_status=_exception_http_status(exc),
                exception_chain=_exception_chain(exc),
            )
            self.last_call = metadata
            raise PairwiseBinderAdapterError(str(exc)) from exc
        except BinderProviderError as exc:
            cause = exc.__cause__ or exc.__context__
            metadata = self._metadata(
                response,
                started,
                structured=False,
                error=str(exc),
                provider_success=response is not None,
                exception_type=type(exc).__name__,
                exception_cause_type=type(cause).__name__ if cause is not None else None,
                exception_cause_message=_safe_message(cause) if cause is not None else None,
                raw_content_length=len(self.last_raw_response or ""),
                request_id=getattr(response, "id", None),
                http_status=_exception_http_status(exc),
                exception_chain=_exception_chain(exc),
            )
            self.last_call = metadata
            raise
        except Exception as exc:
            cause = exc.__cause__ or exc.__context__
            metadata = self._metadata(
                response,
                started,
                structured=False,
                error=_safe_message(exc),
                provider_success=False,
                exception_type=type(exc).__name__,
                exception_cause_type=type(cause).__name__ if cause is not None else None,
                exception_cause_message=_safe_message(cause) if cause is not None else None,
                errno=getattr(exc, "errno", None),
                raw_content_length=len(self.last_raw_response or ""),
                request_id=getattr(response, "id", None),
                http_status=_exception_http_status(exc),
                exception_chain=_exception_chain(exc),
            )
            self.last_call = metadata
            raise BinderProviderError(f"Bailian pairwise binder API call failed: {_safe_message(exc)}") from exc
