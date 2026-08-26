"""TV2-04 candidate generation and routing capabilities.

The adapters reuse the frozen GeneratorRoutingPolicy and deterministic
calculation renderer.  They create a candidate only; Validator/Release is a
separate TV2-05 boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from rag_v2.adaptive import AdaptiveRAGStateV1
from src.generation.generator_routing_policy import (
    GeneratorRouteDecision,
    GeneratorRoutingPolicy,
    GeneratorTarget,
    RouteName,
)

from src.domain.calculation import CalculationResult, CalculationStatus
from src.finance.calculation_renderer import render_calculation_result


class CandidateGenerationCapabilityError(RuntimeError):
    """Raised when a candidate-generation contract cannot be satisfied."""


@dataclass(frozen=True)
class CandidateExecutionResult:
    """Structured candidate result; it is never a release decision."""

    candidate_answer: str
    route: str
    route_reason: str
    bound_evidence_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    calculation_ids: tuple[str, ...] = ()
    candidate_status: str = "CANDIDATE_READY_FOR_VALIDATION"
    generation_metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_generation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_answer, str) or not self.candidate_answer.strip():
            raise ValueError("candidate_answer must be non-empty")
        if not isinstance(self.route, str) or not self.route.strip():
            raise ValueError("route must be non-empty")
        if self.candidate_status != "CANDIDATE_READY_FOR_VALIDATION":
            raise ValueError("candidate_status must remain validation-pending")
        object.__setattr__(self, "bound_evidence_ids", _stable_unique(self.bound_evidence_ids))
        object.__setattr__(self, "citation_ids", _stable_unique(self.citation_ids))
        object.__setattr__(self, "calculation_ids", _stable_unique(self.calculation_ids))
        object.__setattr__(self, "generation_metadata", dict(self.generation_metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_answer": self.candidate_answer,
            "route": self.route,
            "route_reason": self.route_reason,
            "bound_evidence_ids": list(self.bound_evidence_ids),
            "citation_ids": list(self.citation_ids),
            "calculation_ids": list(self.calculation_ids),
            "candidate_status": self.candidate_status,
            "generation_metadata": dict(self.generation_metadata),
            "candidate_generation_id": self.candidate_generation_id,
        }


def _stable_unique(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("fact_id") or candidate.get("evidence_id") or candidate.get("candidate_id")
    return str(value).strip() if value is not None else ""


def _bound_items(state: AdaptiveRAGStateV1) -> list[dict[str, Any]]:
    allowed = set(_stable_unique(getattr(state, "bound_evidence_ids", ())))
    if not allowed:
        return []
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in state.evidence_packets:
        if not isinstance(raw, Mapping):
            raise CandidateGenerationCapabilityError("candidate_evidence_must_be_mapping")
        item = dict(raw)
        identity = _candidate_identity(item)
        if identity in allowed and identity not in seen:
            seen.add(identity)
            items.append(item)
    return items


def _bound_citation_ids(items: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return _stable_unique(
        str(item.get("citation_id"))
        for item in items
        if item.get("citation_id")
    )


def _calculation_id(state: AdaptiveRAGStateV1) -> tuple[str, ...]:
    value = getattr(state, "calculation_result_id", None)
    return (str(value),) if value else ()


def _candidate_generation_id(
    answer: str,
    route: str,
    evidence_ids: Iterable[str],
    calculation_ids: Iterable[str],
) -> str:
    payload = {
        "answer": answer,
        "route": route,
        "evidence_ids": list(evidence_ids),
        "calculation_ids": list(calculation_ids),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"G1-{digest}"


class DeterministicFactRenderer:
    """Small structured renderer for an admitted single fact.

    The existing calculation renderer remains the canonical calculation
    renderer.  This renderer handles the separate direct-fact candidate shape
    without searching or parsing arbitrary answer text.
    """

    renderer_id = "deterministic_fact_renderer_tv2_04"

    def render(self, item: Mapping[str, Any], *, question: str = "") -> str:
        metric = item.get("metric") or item.get("normalized_metric") or "Result"
        period = item.get("period") or item.get("normalized_period")
        value = item.get("value")
        if value is None:
            value = item.get("parsed_numeric_value")
        unit = item.get("unit") or item.get("currency")
        scale = item.get("scale")
        citation = item.get("citation_id")
        label = str(metric)
        if period:
            label = f"{label} ({period})"
        rendered_value = "N/A" if value is None else str(value)
        suffix = " ".join(str(value) for value in (unit, scale) if value)
        answer = f"{label}: {rendered_value}"
        if suffix:
            answer += f" {suffix}"
        if citation:
            answer += f" [{citation}]"
        return answer


class LocalSpecialistGenerationAdapter:
    """Adapt the existing LocalSpecialistGenerator contract.

    The backend is injected so CPU-safe tests can use a deterministic provider;
    a production/canonical smoke may inject LocalSpecialistGenerator itself.
    """

    def __init__(self, backend: Any) -> None:
        if not callable(getattr(backend, "generate", None)):
            raise TypeError("specialist backend must expose generate")
        self.backend = backend
        self.calls = 0

    def generate(
        self,
        question: str,
        evidence_items: list[dict[str, Any]],
        calculation_result: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | str:
        self.calls += 1
        return self.backend.generate(question, evidence_items, calculation_result)


class TrustedV2GenerationCapability:
    """Reuse routing policy and call one candidate generator target."""

    candidate_mode = True

    def __init__(
        self,
        *,
        routing_policy: Any | None = None,
        renderer: Any | None = None,
        specialist: Any | None = None,
    ) -> None:
        self.routing_policy = routing_policy or GeneratorRoutingPolicy()
        self.renderer = renderer or DeterministicFactRenderer()
        self.specialist = specialist
        self.route_calls = 0
        self.renderer_calls = 0
        self.specialist_calls = 0
        self.last_decision: GeneratorRouteDecision | None = None
        self.last_result: CandidateExecutionResult | None = None
        self._unknown_citation_count = 0

    @staticmethod
    def _calculation_object(state: AdaptiveRAGStateV1) -> CalculationResult | None:
        value = getattr(state, "_calculation_result_obj", None)
        return value if isinstance(value, CalculationResult) else None

    def _route(
        self,
        state: AdaptiveRAGStateV1,
        items: list[dict[str, Any]],
        calculation: CalculationResult | None,
    ) -> GeneratorRouteDecision:
        calculation_payload = calculation.to_dict() if calculation else None
        route_hint = str(getattr(state, "intent", "") or "")
        try:
            decision = self.routing_policy.route(
                state.normalized_query,
                items,
                calculation_result=calculation_payload,
                route_hint=route_hint,
            )
        except Exception as exc:
            raise CandidateGenerationCapabilityError("generator_routing_failed") from exc
        # A calculation plan has already been semantically selected by the
        # Supervisor.  Do not allow evidence cardinality or wording heuristics
        # to hand numeric authority to the Specialist.
        if str(state.intent).upper() == "CALCULATION":
            decision = GeneratorRouteDecision(
                route_name=RouteName.CALCULATION_SIMPLE,
                target=GeneratorTarget.DETERMINISTIC_CALCULATOR,
                reason="PLAN_REQUIRES_DETERMINISTIC_CALCULATION",
                requires_c1=True,
            )
        self.route_calls += 1
        self.last_decision = decision
        return decision

    @staticmethod
    def _call_renderer(
        renderer: Any,
        *,
        item: Mapping[str, Any] | None,
        question: str,
        calculation: CalculationResult | None,
    ) -> str:
        if calculation is not None:
            # The existing calculation_renderer is the authoritative path.
            return render_calculation_result(calculation)
        if item is None:
            raise CandidateGenerationCapabilityError("renderer_missing_bound_fact")
        method = getattr(renderer, "render", None)
        if callable(method):
            try:
                value = method(item, question=question)
            except TypeError:
                value = method(item)
        elif callable(renderer):
            value = renderer(item)
        else:
            raise CandidateGenerationCapabilityError("renderer_not_callable")
        if not isinstance(value, str) or not value.strip():
            raise CandidateGenerationCapabilityError("renderer_returned_empty_candidate")
        return value.strip()

    def _call_specialist(
        self,
        state: AdaptiveRAGStateV1,
        items: list[dict[str, Any]],
        calculation: CalculationResult | None,
        allowed_citations: tuple[str, ...],
    ) -> tuple[str, tuple[str, ...], dict[str, Any]]:
        if self.specialist is None:
            raise CandidateGenerationCapabilityError("financial_specialist_not_configured")
        method = getattr(self.specialist, "generate", None)
        if not callable(method):
            raise CandidateGenerationCapabilityError("financial_specialist_not_callable")
        calculation_payload = calculation.to_dict() if calculation else None
        self.specialist_calls += 1
        raw = method(state.normalized_query, items, calculation_payload)
        metadata: dict[str, Any] = {}
        if isinstance(raw, Mapping):
            answer = raw.get("answer_text") or raw.get("answer") or raw.get("raw_output")
            raw_citations = raw.get("citation_ids", ())
            if isinstance(raw.get("metadata"), Mapping):
                metadata.update(dict(raw["metadata"]))
        else:
            answer = raw
            raw_citations = ()
        if not isinstance(answer, str) or not answer.strip():
            raise CandidateGenerationCapabilityError("financial_specialist_empty_candidate")
        raw_citations = raw_citations if isinstance(raw_citations, (list, tuple, set)) else ()
        unknown = [str(item) for item in raw_citations if str(item) not in allowed_citations]
        self._unknown_citation_count += len(unknown)
        metadata.update(
            {
                "provider_citation_ids": [str(item) for item in raw_citations],
                "unknown_generated_citation_ids": unknown,
            }
        )
        citations = tuple(
            str(item) for item in raw_citations if str(item) in allowed_citations
        )
        if not citations:
            citations = allowed_citations
        return answer.strip(), _stable_unique(citations), metadata

    def generate(self, state: AdaptiveRAGStateV1) -> CandidateExecutionResult:
        items = _bound_items(state)
        if not items:
            raise CandidateGenerationCapabilityError("generation_requires_bound_evidence")
        calculation = self._calculation_object(state)
        decision = self._route(state, items, calculation)
        evidence_ids = _stable_unique(getattr(state, "bound_evidence_ids", ()))
        allowed_citations = _bound_citation_ids(items)
        calculation_ids = _calculation_id(state)
        metadata: dict[str, Any] = {
            "renderer_id": getattr(self.renderer, "renderer_id", None),
            "route_name": decision.route_name.value,
            "route_target": decision.target.value,
            "route_reason": decision.reason,
            "calculator_invoked": calculation is not None,
        }

        if decision.target is GeneratorTarget.FAIL_CLOSED_PRE_GEN:
            raise CandidateGenerationCapabilityError("generator_route_fail_closed")
        if decision.target is GeneratorTarget.DETERMINISTIC_CALCULATOR:
            if calculation is None or calculation.status is not CalculationStatus.EXECUTED:
                raise CandidateGenerationCapabilityError("calculation_result_not_ready")
            answer = render_calculation_result(calculation)
            self.renderer_calls += 1
        elif decision.target is GeneratorTarget.DETERMINISTIC_RENDERER:
            item = items[0] if items else None
            answer = self._call_renderer(
                self.renderer,
                item=item,
                question=state.normalized_query,
                calculation=calculation,
            )
            self.renderer_calls += 1
        elif decision.target is GeneratorTarget.LOCAL_SPECIALIST:
            answer, citations, specialist_metadata = self._call_specialist(
                state, items, calculation, allowed_citations
            )
            metadata.update(specialist_metadata)
            allowed_citations = citations
        else:
            raise CandidateGenerationCapabilityError("unsupported_generator_target")

        result = CandidateExecutionResult(
            candidate_answer=answer,
            route=decision.route_name.value,
            route_reason=decision.reason,
            bound_evidence_ids=evidence_ids,
            citation_ids=allowed_citations,
            calculation_ids=calculation_ids,
            generation_metadata=metadata,
            candidate_generation_id=_candidate_generation_id(
                answer, decision.route_name.value, evidence_ids, calculation_ids
            ),
        )
        self.last_result = result
        state.generation_route = decision.route_name.value
        state.route_reason = decision.reason
        state.candidate_answer = result.candidate_answer
        state.candidate_generation_id = result.candidate_generation_id
        state.candidate_status = result.candidate_status
        state.validation_pending = True
        return result

    def trace_snapshot(self) -> dict[str, Any]:
        result = self.last_result
        return {
            "generation_route": result.route if result else None,
            "route_reason": result.route_reason if result else None,
            "route_calls": self.route_calls,
            "renderer_invoked": self.renderer_calls > 0,
            "renderer_call_count": self.renderer_calls,
            "specialist_invoked": self.specialist_calls > 0,
            "specialist_call_count": self.specialist_calls,
            "candidate_ready": result is not None,
            "validation_pending": result is not None,
            "candidate_generation_id": result.candidate_generation_id if result else None,
            "unknown_generated_citation_count": self._unknown_citation_count,
        }


__all__ = [
    "CandidateExecutionResult",
    "CandidateGenerationCapabilityError",
    "DeterministicFactRenderer",
    "LocalSpecialistGenerationAdapter",
    "TrustedV2GenerationCapability",
]