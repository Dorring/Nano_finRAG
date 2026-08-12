"""Deterministic shortlist plus comparative Binder provider for NF-V2-03 R6."""

from __future__ import annotations

import json
import re
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
from .binder_selection import fact_handle_map
from .binder_service import BinderRequest


SHORTLIST_FORMULATION = "deterministic_shortlist_comparative_v1"
SHORTLIST_MAX = 5
SHORTLIST_WEIGHTS = {
    "normalized_metric_overlap": 5.0,
    "raw_metric_overlap": 3.0,
    "row_header_overlap": 3.0,
    "period_exactness": 4.0,
    "scope_overlap": 3.0,
    "statement_table_overlap": 2.0,
    "section_overlap": 1.0,
}
COMPARATIVE_DECISIONS = ("SELECT", "NONE", "AMBIGUOUS")

COMPARATIVE_SYSTEM_PROMPT = """You are a comparative financial evidence verifier.

Compare all shortlisted candidates jointly for each independent RequiredSlot. SELECT exactly
one candidate only when it satisfies the metric, requested period, scope, statement context,
row/header structure, and applicable operand role materially better than every other candidate.

Return NONE when no candidate fully satisfies the requirement. Return AMBIGUOUS when two or
more candidates remain genuinely indistinguishable after using all visible context. Candidate
presence never implies a valid match. Do not select a closest lexical match, parent metric,
wrong period, wrong segment, or wrong statement. For calculation operands, respect the
operand role independently and never calculate or emit a result.

Return only the strict JSON object. Each exact slot ID must contain decision SELECT, NONE, or
AMBIGUOUS and selected_handle, which is one supplied handle only for SELECT and null otherwise.
Do not return reasoning, answers, status fields, or source IDs.
"""


@dataclass(frozen=True)
class CandidateShortlist:
    slot_id: str
    candidates: tuple[dict[str, Any], ...]
    hard_rejected: tuple[dict[str, Any], ...]

    @property
    def handles(self) -> tuple[str, ...]:
        return tuple(str(item["handle"]) for item in self.candidates)


@dataclass(frozen=True)
class ComparativeDecisionDTOv1:
    tasks: Mapping[str, Mapping[str, Any]]


def _tokens(value: Any) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(value or "").casefold()) if token}


def _field_tokens(view: Mapping[str, Any], *fields: str) -> set[str]:
    tokens: set[str] = set()
    for field in fields:
        value = view.get(field)
        if isinstance(value, (list, tuple)):
            for item in value:
                tokens |= _tokens(item)
        else:
            tokens |= _tokens(value)
    return tokens


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), len(right))


def _period_key(value: Any) -> str:
    text = " ".join(str(value or "").casefold().split())
    match = re.search(r"fy\s*(\d{4})", text)
    return f"fy{match.group(1)}" if match else text


def _hard_rejection(slot: Any, fact: Mapping[str, Any]) -> str | None:
    if fact.get("provenance_complete") is not True:
        return "provenance_invalid"
    slot_period = _period_key(slot.period)
    fact_period = _period_key(fact.get("normalized_period") or fact.get("raw_period"))
    if slot_period and fact_period and slot_period != fact_period:
        return "explicit_period_conflict"
    slot_unit = str(slot.unit or "").casefold().strip()
    fact_unit = str(fact.get("unit") or "").casefold().strip()
    if slot_unit and fact_unit and slot_unit != fact_unit:
        return "explicit_unit_conflict"
    return None


def _score_candidate(slot: Any, view: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    metric = _tokens(slot.metric)
    raw_metric = _tokens(view.get("raw_metric"))
    normalized_metric = _tokens(view.get("normalized_metric"))
    row_headers = _field_tokens(view, "row_label", "row_path", "row_hierarchy", "column_header", "column_header_path", "multi_level_column_headers")
    scope = _tokens(getattr(slot, "scope", None))
    statement = _field_tokens(view, "table_title", "statement_title", "statement_type")
    section = _field_tokens(view, "section_heading", "section_title", "section_path")
    period = _period_key(slot.period)
    fact_period = _period_key(view.get("normalized_period") or view.get("raw_period"))
    signals = {
        "normalized_metric_overlap": _overlap(metric, normalized_metric),
        "raw_metric_overlap": _overlap(metric, raw_metric),
        "row_header_overlap": _overlap(metric, row_headers),
        "period_exactness": 1.0 if period and fact_period and period == fact_period else 0.0,
        "scope_overlap": _overlap(scope, row_headers | statement | section),
        "statement_table_overlap": _overlap(metric, statement),
        "section_overlap": _overlap(metric, section),
    }
    score = sum(SHORTLIST_WEIGHTS[name] * value for name, value in signals.items())
    return score, signals


def build_shortlists(
    request: BinderRequest,
    *,
    fact_view_version: str = "v2",
    source_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, CandidateShortlist], dict[str, dict[str, Any]], dict[str, str]]:
    handles_to_ids = fact_handle_map(request)
    if fact_view_version == "v1":
        views = build_binder_fact_views(list(request.facts))
    elif fact_view_version == "v2":
        views = build_binder_fact_views_v2(list(request.facts), source_by_candidate)
    else:
        raise ValueError(f"unsupported BinderFactView version: {fact_view_version}")
    views_by_handle = {str(view["fact_handle"]): view for view in views}
    facts_by_handle = {
        handle: fact
        for handle, fact in zip(handles_to_ids, request.facts, strict=True)
    }
    result: dict[str, CandidateShortlist] = {}
    for slot in request.plan.required_slots:
        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for handle in handles_to_ids:
            fact = facts_by_handle[handle]
            reason = _hard_rejection(slot, fact)
            if reason is not None:
                rejected.append({"handle": handle, "fact_id": handles_to_ids[handle], "reason": reason})
                continue
            score, signals = _score_candidate(slot, views_by_handle[handle])
            candidates.append({"handle": handle, "fact_id": handles_to_ids[handle], "score": round(score, 8), "signals": signals, "fact_view": views_by_handle[handle]})
        candidates.sort(key=lambda item: (-float(item["score"]), str(item["handle"])))
        result[slot.slot_id] = CandidateShortlist(slot.slot_id, tuple(candidates[:SHORTLIST_MAX]), tuple(rejected))
    union_handles = {handle for item in result.values() for handle in item.handles}
    facts = {handle: views_by_handle[handle] for handle in sorted(union_handles)}
    return result, facts, handles_to_ids


def shortlist_provider_request(
    request: BinderRequest,
    *,
    fact_view_version: str = "v2",
    source_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, CandidateShortlist], dict[str, str], dict[str, Any]]:
    shortlists, facts, handles_to_ids = build_shortlists(request, fact_view_version=fact_view_version, source_by_candidate=source_by_candidate)
    tasks: dict[str, dict[str, Any]] = {}
    slot_ids: list[str] = []
    for slot in request.plan.required_slots:
        slot_ids.append(slot.slot_id)
        shortlist = shortlists[slot.slot_id]
        requirement = slot.to_dict()
        requirement["scope"] = getattr(slot, "scope", None)
        requirement["operation"] = request.plan.operation
        requirement["operand_role"] = slot.role
        tasks[slot.slot_id] = {"requirement": requirement, "candidate_handles": list(shortlist.handles), "pre_status": "NO_ELIGIBLE_CANDIDATE" if not shortlist.candidates else None}
    payload = {"question": request.question, "intent": request.plan.intent.value, "operation": request.plan.operation, "facts": facts, "tasks": tasks}
    schema: dict[str, Any] = {"type": "object", "additionalProperties": False, "required": ["tasks"], "properties": {"tasks": {"type": "object", "additionalProperties": False, "required": slot_ids, "properties": {}}}}
    for slot_id in slot_ids:
        handles = list(shortlists[slot_id].handles)
        schema["properties"]["tasks"]["properties"][slot_id] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["decision", "selected_handle"],
            "properties": {
                "decision": {"type": "string", "enum": list(COMPARATIVE_DECISIONS)},
                "selected_handle": {"type": ["string", "null"], "enum": handles + [None]},
            },
        }
    return payload, shortlists, handles_to_ids, schema


def build_comparative_messages(
    request: BinderRequest,
    payload: Mapping[str, Any],
    *,
    system_prompt: str = COMPARATIVE_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"BinderCandidateShortlistV1 request JSON:\n{body}"}]


def parse_comparative_decisions(payload: Any, plan: SupervisorPlan, shortlists: Mapping[str, CandidateShortlist]) -> ComparativeDecisionDTOv1:
    if not isinstance(payload, dict) or set(payload) != {"tasks"} or not isinstance(payload["tasks"], dict):
        raise ValueError("ComparativeDecisionDTOv1 must contain only tasks")
    expected_slots = {slot.slot_id for slot in plan.required_slots}
    if set(payload["tasks"]) != expected_slots:
        raise ValueError("comparative task properties must exactly match RequiredSlots")
    values: dict[str, dict[str, Any]] = {}
    for slot_id in expected_slots:
        decision = payload["tasks"][slot_id]
        if not isinstance(decision, dict) or set(decision) != {"decision", "selected_handle"}:
            raise ValueError(f"invalid comparative decision shape for {slot_id}")
        if decision["decision"] not in COMPARATIVE_DECISIONS:
            raise ValueError(f"invalid comparative decision for {slot_id}")
        handle = decision["selected_handle"]
        if handle is not None and handle not in shortlists[slot_id].handles:
            raise ValueError(f"selected handle outside shortlist for {slot_id}")
        if decision["decision"] == "SELECT" and handle is None:
            raise ValueError(f"SELECT requires a shortlisted handle for {slot_id}")
        if decision["decision"] != "SELECT" and handle is not None:
            raise ValueError(f"{decision['decision']} must not carry a selected handle for {slot_id}")
        values[slot_id] = {"decision": decision["decision"], "selected_handle": handle}
    return ComparativeDecisionDTOv1(values)


def comparative_decisions_to_binding(
    dto: ComparativeDecisionDTOv1,
    request: BinderRequest,
    shortlists: Mapping[str, CandidateShortlist],
    handles_to_ids: Mapping[str, str],
) -> tuple[EvidenceBinding, dict[str, dict[str, Any]]]:
    expected_slots = [slot.slot_id for slot in request.plan.required_slots]
    if set(dto.tasks) != set(expected_slots):
        raise ValueError("comparative adapter received a non-exact slot set")
    slot_bindings: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    invalid: list[str] = []
    outcomes: dict[str, dict[str, Any]] = {}
    for slot_id in expected_slots:
        decision = str(dto.tasks[slot_id]["decision"])
        handle = dto.tasks[slot_id]["selected_handle"]
        eligible = set(shortlists[slot_id].handles)
        outcomes[slot_id] = {"decision": decision, "selected_handle": handle}
        if not eligible:
            if decision != "NONE" or handle is not None:
                invalid.append(f"no_eligible_candidate_selected:{slot_id}")
            else:
                missing.append(slot_id)
        elif decision == "SELECT":
            if handle is None or handle not in eligible:
                invalid.append(f"selected_handle_not_eligible:{slot_id}")
            else:
                slot_bindings[slot_id] = (handles_to_ids[handle],)
        elif decision == "NONE":
            if handle is not None:
                invalid.append(f"none_with_selected_handle:{slot_id}")
            else:
                missing.append(slot_id)
        elif decision == "AMBIGUOUS":
            if handle is not None:
                invalid.append(f"ambiguous_with_selected_handle:{slot_id}")
            else:
                ambiguous.append(slot_id)
    if invalid:
        binding = EvidenceBinding(status=BindingStatus.INVALID.value, slot_bindings=slot_bindings, missing_slots=tuple(missing), ambiguous_slots=tuple(ambiguous), invalid_reasons=tuple(invalid))
    elif all(dto.tasks[slot_id]["decision"] == "SELECT" for slot_id in expected_slots):
        binding = EvidenceBinding(status=BindingStatus.BOUND.value, slot_bindings=slot_bindings)
    elif ambiguous:
        binding = EvidenceBinding(status=BindingStatus.AMBIGUOUS.value, slot_bindings=slot_bindings, missing_slots=tuple(missing), ambiguous_slots=tuple(ambiguous))
    else:
        binding = EvidenceBinding(status=BindingStatus.MISSING.value, slot_bindings=slot_bindings, missing_slots=tuple(missing))
    return binding, outcomes


class ComparativeBinderAdapterError(BinderProviderError):
    schema_valid = True
    adapter_failure = True
    classification = "comparative_adapter_failure"


class BailianShortlistComparativeBinderProvider(BailianBinderProvider):
    """One-call Bailian provider over a deterministic high-recall shortlist."""

    provider_role = "evidence_binder"
    formulation = SHORTLIST_FORMULATION

    def __init__(self, *args: Any, system_prompt: str | None = None, fact_view_version: str = "v2", source_metadata_by_candidate: Mapping[str, Mapping[str, Any]] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = system_prompt or COMPARATIVE_SYSTEM_PROMPT
        self.fact_view_version = fact_view_version
        self.source_metadata_by_candidate = dict(source_metadata_by_candidate or {})
        self.last_comparative_outcomes: dict[str, dict[str, Any]] | None = None
        self.last_shortlists: dict[str, CandidateShortlist] = {}

    def bind(self, request: Any) -> BinderProviderResult:
        if not isinstance(request, BinderRequest):
            request = BinderRequest(question_id=str(request["question_id"]), question=str(request["question"]), plan=SupervisorPlan.from_dict({"intent": request["intent"], "required_slots": request["required_slots"], "operation": request["operation"], "next_action": "RETRIEVE"}), facts=tuple(request.get("financial_facts") or ()))
        started = time.perf_counter()
        self.last_raw_response = None
        self.last_comparative_outcomes = None
        payload, shortlists, handles_to_ids, schema = shortlist_provider_request(request, fact_view_version=self.fact_view_version, source_by_candidate=self.source_metadata_by_candidate)
        self.last_shortlists = shortlists
        try:
            response = self.client.chat.completions.create(model=self.model_name, messages=build_comparative_messages(request, payload, system_prompt=self.system_prompt), temperature=self.temperature, response_format={"type": "json_schema", "json_schema": {"name": "ComparativeDecisionDTOv1", "strict": True, "schema": schema}}, extra_body={"enable_thinking": self.enable_thinking})
            message = response.choices[0].message if response.choices else None
            raw = getattr(message, "content", None) if message is not None else None
            self.last_raw_response = raw if isinstance(raw, str) else None
            if not self.last_raw_response or not self.last_raw_response.strip():
                raise BinderProviderError("Bailian returned an empty ComparativeDecisionDTOv1 response")
            try:
                parsed = json.loads(self.last_raw_response.strip())
            except json.JSONDecodeError as exc:
                raise BinderProviderError("Bailian ComparativeDecisionDTOv1 response was not strict JSON") from exc
            dto = parse_comparative_decisions(parsed, request.plan, shortlists)
            binding, outcomes = comparative_decisions_to_binding(dto, request, shortlists, handles_to_ids)
            self.last_comparative_outcomes = outcomes
            metadata = self._metadata(response, started, structured=True, raw_content_length=len(self.last_raw_response), request_id=getattr(response, "id", None))
            self.last_call = metadata
            return BinderProviderResult(binding=binding, metadata=metadata, raw_response=self.last_raw_response)
        except BinderProviderError as exc:
            cause = exc.__cause__ or exc.__context__
            metadata = self._metadata(response if "response" in locals() else None, started, structured=False, error=str(exc), provider_success="response" in locals(), exception_type=type(exc).__name__, exception_cause_type=type(cause).__name__ if cause is not None else None, exception_cause_message=_safe_message(cause) if cause is not None else None, raw_content_length=len(self.last_raw_response or ""), request_id=getattr(response, "id", None) if "response" in locals() else None, http_status=_exception_http_status(exc), exception_chain=_exception_chain(exc))
            self.last_call = metadata
            raise
        except Exception as exc:
            cause = exc.__cause__ or exc.__context__
            metadata = self._metadata(response if "response" in locals() else None, started, structured=False, error=_safe_message(exc), provider_success=False, exception_type=type(exc).__name__, exception_cause_type=type(cause).__name__ if cause is not None else None, exception_cause_message=_safe_message(cause) if cause is not None else None, errno=getattr(exc, "errno", None), raw_content_length=len(self.last_raw_response or ""), request_id=getattr(response, "id", None) if "response" in locals() else None, http_status=_exception_http_status(exc), exception_chain=_exception_chain(exc))
            self.last_call = metadata
            raise BinderProviderError(f"Bailian shortlist comparative API call failed: {_safe_message(exc)}") from exc
