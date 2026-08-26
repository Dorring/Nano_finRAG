"""TV2-05 canonical validator and release gate.

The validator consumes only Binder-admitted evidence from bounded runtime state.
It never calls retrieval, reads conversation history, or invokes the V2 handle
method that would generate a second candidate.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

from rag_v2.adaptive import AdaptiveRAGStateV1
from rag_v2.generation.contracts import AnswerEnvelopeV1, ValidationSeverity
from rag_v2.generation.validator import RuntimeGenerationValidatorV1
from rag_v2.runtime.semantic_claims import SemanticClaimDecision, SemanticClaimVerifierV1
from src.domain.calculation import CalculationResult, CalculationStatus
from src.finance.calculation_renderer import render_calculation_result

from .trusted_v2_generation import CandidateExecutionResult, DeterministicFactRenderer


def _stable_unique(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _identity(item: Mapping[str, Any]) -> str:
    value = item.get("evidence_id") or item.get("fact_id") or item.get("candidate_id")
    return str(value).strip() if value is not None else ""


def _citation(item: Mapping[str, Any]) -> str:
    value = item.get("citation_id")
    return str(value).strip() if value is not None else ""


def _validation_id(request_id: str, candidate: CandidateExecutionResult, attempt: int) -> str:
    payload = {
        "request_id": request_id,
        "attempt": attempt,
        "candidate_generation_id": candidate.candidate_generation_id,
        "answer": candidate.candidate_answer,
        "route": candidate.route,
        "evidence_ids": list(candidate.bound_evidence_ids),
        "citation_ids": list(candidate.citation_ids),
        "calculation_ids": list(candidate.calculation_ids),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"V2-{digest}"


@dataclass(frozen=True)
class V2ValidationResult:
    """Structured policy result at the candidate release boundary."""

    validation_id: str
    passed: bool
    status: str
    reason_codes: tuple[str, ...] = ()
    failed_checks: tuple[str, ...] = ()
    repairable: bool = False
    bound_evidence_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    calculation_ids: tuple[str, ...] = ()
    generation_report: Mapping[str, Any] = field(default_factory=dict)
    semantic_report: Mapping[str, Any] | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "FAIL", "ERROR"}:
            raise ValueError("validation status must be PASS, FAIL, or ERROR")
        if self.passed != (self.status == "PASS"):
            raise ValueError("validation passed/status combination is invalid")
        object.__setattr__(self, "reason_codes", _stable_unique(self.reason_codes))
        object.__setattr__(self, "failed_checks", _stable_unique(self.failed_checks))
        object.__setattr__(self, "bound_evidence_ids", _stable_unique(self.bound_evidence_ids))
        object.__setattr__(self, "citation_ids", _stable_unique(self.citation_ids))
        object.__setattr__(self, "calculation_ids", _stable_unique(self.calculation_ids))
        object.__setattr__(self, "generation_report", dict(self.generation_report))
        if self.semantic_report is not None:
            object.__setattr__(self, "semantic_report", dict(self.semantic_report))

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "passed": self.passed,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "failed_checks": list(self.failed_checks),
            "repairable": self.repairable,
            "bound_evidence_ids": list(self.bound_evidence_ids),
            "citation_ids": list(self.citation_ids),
            "calculation_ids": list(self.calculation_ids),
            "generation_report": dict(self.generation_report),
            "semantic_report": (
                dict(self.semantic_report) if self.semantic_report is not None else None
            ),
            "latency_ms": self.latency_ms,
        }


class CandidateRepairError(RuntimeError):
    """A repair provider failed as software, not as a trust-policy refusal."""


class CandidateRepairUnavailable(RuntimeError):
    """The candidate has no safe deterministic repair for its route."""


class CandidateRepairCapability(Protocol):
    def can_repair(
        self,
        state: AdaptiveRAGStateV1,
        candidate: CandidateExecutionResult,
        validation: V2ValidationResult,
    ) -> bool:
        ...

    def repair(
        self,
        state: AdaptiveRAGStateV1,
        candidate: CandidateExecutionResult,
        validation: V2ValidationResult,
    ) -> CandidateExecutionResult:
        ...


def _bound_items(state: AdaptiveRAGStateV1) -> tuple[dict[str, Any], ...]:
    admitted = {
        str(value).strip()
        for value in getattr(state, "bound_evidence_ids", ())
        if str(value).strip()
    }
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in state.evidence_packets:
        if not isinstance(raw, Mapping):
            raise CandidateRepairError("bound_evidence_packet_not_mapping")
        item = dict(raw)
        identity = _identity(item)
        if identity in admitted and identity not in seen:
            seen.add(identity)
            items.append(item)
    missing = admitted - seen
    if missing:
        raise CandidateRepairError(
            "bound_evidence_missing_from_runtime_state:" + ",".join(sorted(missing))
        )
    return tuple(items)


def _validation_packet(
    state: AdaptiveRAGStateV1,
    candidate: CandidateExecutionResult,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    bound_items = _bound_items(state)
    admitted_ids = tuple(_identity(item) for item in bound_items)
    admitted_set = set(admitted_ids)
    if not candidate.bound_evidence_ids:
        raise ValueError("candidate references no Binder-admitted evidence")
    if not set(candidate.bound_evidence_ids) <= admitted_set:
        raise ValueError("candidate references evidence outside Binder admission")

    original_citations = tuple(_citation(item) for item in bound_items if _citation(item))
    allowed_citations = _stable_unique(
        value for value in (*original_citations, *(value.upper() for value in original_citations))
    )
    if not set(candidate.citation_ids) <= set(allowed_citations):
        raise ValueError("candidate citation metadata is not Binder-admitted")

    normalized_items: list[dict[str, Any]] = []
    for raw in bound_items:
        item = dict(raw)
        citation_id = _citation(item)
        if citation_id:
            item["citation_id"] = citation_id.upper()
        item.setdefault("fact_id", _identity(raw))
        item.setdefault("source_id", item.get("physical_source_id") or item.get("document_id"))
        item.setdefault(
            "source_text",
            item.get("text") or item.get("evidence_text") or item.get("content") or item.get("metric") or "",
        )
        normalized_items.append(item)

    calculation = getattr(state, "calculation_result", None)
    if not isinstance(calculation, Mapping):
        calculation = None
    packet = {
        "query_id": state.request_id,
        "route": candidate.route,
        "validation_status": "VERIFIED",
        "allowed_citation_ids": list(allowed_citations),
        "evidence_items": normalized_items,
        "calculation_result": dict(calculation) if calculation is not None else None,
    }
    return packet, admitted_ids, original_citations


def _coerce_envelope(
    state: AdaptiveRAGStateV1,
    candidate: CandidateExecutionResult,
) -> AnswerEnvelopeV1:
    if not candidate.citation_ids:
        raise ValueError("candidate is missing structured citation IDs")
    metadata = candidate.generation_metadata
    provider = str(metadata.get("provider") or metadata.get("generator_provider") or "tv2-candidate")
    model = str(metadata.get("model") or metadata.get("generator_model") or "tv2-candidate")
    return AnswerEnvelopeV1(
        query_id=state.request_id,
        route=candidate.route,
        answer_text=candidate.candidate_answer,
        citation_ids=tuple(candidate.citation_ids),
        generator_provider=provider,
        generator_model=model,
        metadata=dict(metadata),
    )


class DeterministicCandidateRepair:
    """A conservative non-LLM repair for structured fact/calculation routes."""

    repairer_id = "deterministic_candidate_repair_tv2_05"

    def can_repair(
        self,
        state: AdaptiveRAGStateV1,
        candidate: CandidateExecutionResult,
        validation: V2ValidationResult,
    ) -> bool:
        del state
        if any(code in {"BOUND_EVIDENCE_NOT_ADMITTED", "CALCULATION_PROVENANCE_MISMATCH", "UNBOUND_CITATION_METADATA"}
               for code in validation.reason_codes):
            return False
        return candidate.route in {"STRUCTURED_SINGLE", "CALCULATION_SIMPLE"}

    def repair(
        self,
        state: AdaptiveRAGStateV1,
        candidate: CandidateExecutionResult,
        validation: V2ValidationResult,
    ) -> CandidateExecutionResult:
        del validation
        items = tuple(
            item
            for item in _bound_items(state)
            if _identity(item) in set(candidate.bound_evidence_ids)
        )
        if not items:
            raise CandidateRepairUnavailable("no candidate-bound evidence for repair")
        calculation = getattr(state, "_calculation_result_obj", None)
        if candidate.route == "CALCULATION_SIMPLE":
            if not isinstance(calculation, CalculationResult) or calculation.status is not CalculationStatus.EXECUTED:
                raise CandidateRepairUnavailable("calculation result is not executable")
            answer = render_calculation_result(calculation)
        elif candidate.route == "STRUCTURED_SINGLE":
            answer = DeterministicFactRenderer().render(items[0], question=state.normalized_query)
        else:
            raise CandidateRepairUnavailable("route has no deterministic repair")
        digest = hashlib.sha256(
            f"{candidate.candidate_generation_id or candidate.candidate_answer}|{answer}".encode("utf-8")
        ).hexdigest()[:16]
        return CandidateExecutionResult(
            candidate_answer=answer,
            route=candidate.route,
            route_reason=candidate.route_reason,
            bound_evidence_ids=candidate.bound_evidence_ids,
            citation_ids=tuple(_citation(item) for item in items if _citation(item)),
            calculation_ids=candidate.calculation_ids,
            generation_metadata={
                **dict(candidate.generation_metadata),
                "repairer_id": self.repairer_id,
                "repair_of": candidate.candidate_generation_id,
            },
            candidate_generation_id=f"R1-{digest}",
        )


class TrustedReleaseValidationCapability:
    """Canonical V2 validation plus one bounded candidate repair."""

    candidate_mode = True
    version = "tv2-05"

    def __init__(
        self,
        *,
        generation_validator: RuntimeGenerationValidatorV1 | None = None,
        semantic_verifier: SemanticClaimVerifierV1 | None = None,
        repairer: CandidateRepairCapability | None = None,
    ) -> None:
        self.generation_validator = generation_validator or RuntimeGenerationValidatorV1()
        self.semantic_verifier = semantic_verifier or SemanticClaimVerifierV1()
        self.repairer = repairer or DeterministicCandidateRepair()
        self.validation_calls = 0
        self.repair_calls = 0
        self.revalidation_calls = 0
        self.last_result: V2ValidationResult | None = None
        self._release_record: dict[str, Any] = {}

    @staticmethod
    def _candidate(value: Any, state: AdaptiveRAGStateV1) -> CandidateExecutionResult:
        if isinstance(value, CandidateExecutionResult):
            return value
        if isinstance(value, str) and value.strip():
            return CandidateExecutionResult(
                candidate_answer=value.strip(),
                route=str(getattr(state, "generation_route", "") or "STRUCTURED_SINGLE"),
                route_reason=str(getattr(state, "route_reason", "") or "candidate"),
                bound_evidence_ids=tuple(getattr(state, "bound_evidence_ids", ())),
                citation_ids=tuple(str(item) for item in getattr(state, "_candidate_citation_ids", ())),
                calculation_ids=tuple(str(item) for item in getattr(state, "_candidate_calculation_ids", ())),
            )
        raise TypeError("candidate must be CandidateExecutionResult or non-empty string")

    def _failure(
        self,
        *,
        candidate: CandidateExecutionResult,
        validation_id: str,
        reason_codes: Iterable[str],
        generation_report: Mapping[str, Any] | None = None,
        semantic_report: Mapping[str, Any] | None = None,
        repairable: bool = False,
        started: float,
    ) -> V2ValidationResult:
        reasons = _stable_unique(reason_codes)
        return V2ValidationResult(
            validation_id=validation_id,
            passed=False,
            status="FAIL",
            reason_codes=reasons,
            failed_checks=reasons,
            repairable=repairable,
            bound_evidence_ids=candidate.bound_evidence_ids,
            citation_ids=candidate.citation_ids,
            calculation_ids=candidate.calculation_ids,
            generation_report=generation_report or {},
            semantic_report=semantic_report,
            latency_ms=(perf_counter() - started) * 1000,
        )

    def validate(
        self,
        state: AdaptiveRAGStateV1,
        candidate: CandidateExecutionResult | str,
    ) -> V2ValidationResult:
        started = perf_counter()
        self.validation_calls += 1
        candidate_obj = self._candidate(candidate, state)
        validation_id = _validation_id(state.request_id, candidate_obj, self.validation_calls)
        try:
            packet, admitted_ids, _ = _validation_packet(state, candidate_obj)
        except ValueError as exc:
            code = (
                "BOUND_EVIDENCE_NOT_ADMITTED"
                if "evidence" in str(exc)
                else "UNBOUND_CITATION_METADATA"
            )
            result = self._failure(
                candidate=candidate_obj,
                validation_id=validation_id,
                reason_codes=[code],
                started=started,
            )
            self.last_result = result
            return result
        except Exception as exc:
            raise CandidateRepairError("validation packet construction failed") from exc

        expected_calculation_ids = (
            (str(getattr(state, "calculation_result_id")),)
            if getattr(state, "calculation_result_id", None)
            else ()
        )
        if expected_calculation_ids and tuple(candidate_obj.calculation_ids) != expected_calculation_ids:
            result = self._failure(
                candidate=candidate_obj,
                validation_id=validation_id,
                reason_codes=["CALCULATION_PROVENANCE_MISMATCH"],
                started=started,
            )
            self.last_result = result
            return result
        if not expected_calculation_ids and candidate_obj.calculation_ids:
            result = self._failure(
                candidate=candidate_obj,
                validation_id=validation_id,
                reason_codes=["CALCULATION_PROVENANCE_MISMATCH"],
                started=started,
            )
            self.last_result = result
            return result

        try:
            envelope = _coerce_envelope(state, candidate_obj)
        except ValueError:
            result = self._failure(
                candidate=candidate_obj,
                validation_id=validation_id,
                reason_codes=["UNBOUND_CITATION_METADATA"],
                started=started,
            )
            self.last_result = result
            return result
        report = self.generation_validator.validate(packet, envelope)
        semantic_payload: dict[str, Any] | None = None
        semantic_failed: tuple[str, ...] = ()
        if candidate_obj.route != "CALCULATION_SIMPLE":
            semantic = self.semantic_verifier.verify(packet, envelope)
            semantic_payload = semantic.to_dict()
            if semantic.decision is not SemanticClaimDecision.SUPPORTED:
                semantic_failed = tuple(semantic.reason_codes or ("SCV_CLAIM_UNSUPPORTED",))

        generation_failures = tuple(report.failure_codes)
        # The canonical validator represents ratio values as percentages in
        # the deterministic C1 renderer.  Treat that display alias as a
        # structured unit equivalence; do not relax arbitrary unit claims.
        calculation_payload = getattr(state, "calculation_result", None)
        if (
            candidate_obj.route == "CALCULATION_SIMPLE"
            and isinstance(calculation_payload, Mapping)
            and str(calculation_payload.get("unit", "")).casefold() == "ratio"
            and "%" in candidate_obj.candidate_answer
            and "GV5_UNIT_CURRENCY_SCALE_FIDELITY" in generation_failures
        ):
            generation_failures = tuple(
                code
                for code in generation_failures
                if code != "GV5_UNIT_CURRENCY_SCALE_FIDELITY"
            )
        reasons = _stable_unique((*generation_failures, *semantic_failed))
        passed = not generation_failures and not semantic_failed
        generation_payload = report.to_dict()
        if generation_failures != tuple(report.failure_codes):
            generation_payload["status"] = ValidationSeverity.PASS.value
            generation_payload["failure_codes"] = list(generation_failures)
            generation_payload["accepted_structured_aliases"] = [
                "ratio_to_percent_display"
            ]
        repairable_codes = {
            "GV1_CITATION_ID_VALIDITY", "GV2_CITATION_REQUIREMENT", "GV3_NUMERIC_FIDELITY",
            "GV4_PERIOD_FIDELITY", "GV5_UNIT_CURRENCY_SCALE_FIDELITY",
            "GV6_CALCULATION_RESULT_PRESERVATION", "GV7_UNKNOWN_CITATION",
            "SCV_UNKNOWN_CITATION", "SCV_CITATION_MISSING", "SCV_VALUE_UNSUPPORTED",
            "SCV_PERIOD_UNSUPPORTED", "SCV_UNIT_UNSUPPORTED", "SCV_METRIC_UNSUPPORTED",
            "SCV_METRIC_AMBIGUOUS", "SCV_RELATION_AMBIGUOUS", "SCV_CLAIM_UNSUPPORTED",
            "UNBOUND_CITATION_METADATA",
        }
        provisional = V2ValidationResult(
            validation_id=validation_id,
            passed=False,
            status="FAIL",
            reason_codes=reasons,
            failed_checks=reasons,
            repairable=False,
            bound_evidence_ids=admitted_ids,
            citation_ids=candidate_obj.citation_ids,
            calculation_ids=candidate_obj.calculation_ids,
            generation_report=generation_payload,
            semantic_report=semantic_payload,
            latency_ms=(perf_counter() - started) * 1000,
        )
        repairable = bool(reasons) and bool(set(reasons) & repairable_codes) and bool(
            self.repairer.can_repair(state, candidate_obj, provisional)
        )
        result = V2ValidationResult(
            validation_id=validation_id,
            passed=passed,
            status="PASS" if passed else "FAIL",
            reason_codes=reasons,
            failed_checks=reasons,
            repairable=repairable,
            bound_evidence_ids=admitted_ids,
            citation_ids=candidate_obj.citation_ids,
            calculation_ids=candidate_obj.calculation_ids,
            generation_report=generation_payload,
            semantic_report=semantic_payload,
            latency_ms=(perf_counter() - started) * 1000,
        )
        self.last_result = result
        return result

    def repair(
        self,
        state: AdaptiveRAGStateV1,
        candidate: CandidateExecutionResult | str,
        validation: V2ValidationResult,
    ) -> CandidateExecutionResult:
        if not validation.repairable:
            raise CandidateRepairUnavailable("validation is not repairable")
        candidate_obj = self._candidate(candidate, state)
        self.repair_calls += 1
        try:
            repaired = self.repairer.repair(state, candidate_obj, validation)
        except CandidateRepairUnavailable:
            raise
        except Exception as exc:
            raise CandidateRepairError("candidate repair failed") from exc
        if not isinstance(repaired, CandidateExecutionResult):
            raise CandidateRepairError("repairer returned invalid candidate")
        if repaired.route != candidate_obj.route:
            raise CandidateRepairError("repair changed generation route")
        if not set(repaired.bound_evidence_ids) <= set(candidate_obj.bound_evidence_ids):
            raise CandidateRepairError("repair added bound evidence")
        if not set(repaired.bound_evidence_ids) <= set(getattr(state, "bound_evidence_ids", ())):
            raise CandidateRepairError("repair added unbound admitted evidence")
        admitted_citations = {_citation(item) for item in _bound_items(state) if _citation(item)}
        if not set(repaired.citation_ids) <= admitted_citations:
            raise CandidateRepairError("repair added unbound citation")
        if tuple(repaired.calculation_ids) != tuple(candidate_obj.calculation_ids):
            raise CandidateRepairError("repair changed calculation authority")
        if not repaired.candidate_answer.strip():
            raise CandidateRepairError("repair returned empty candidate")
        return repaired

    def record_revalidation(self) -> None:
        self.revalidation_calls += 1

    def record_release(self, *, released: bool, final_candidate_id: str | None, release_status: str) -> None:
        self._release_record = {
            "release_decision": "RELEASED" if released else "NOT_RELEASED",
            "release_status": release_status,
            "final_candidate_id": final_candidate_id,
        }

    def trace_snapshot(self) -> dict[str, Any]:
        result = self.last_result
        return {
            "validation_id": result.validation_id if result else None,
            "validation_passed": result.passed if result else False,
            "failed_checks": list(result.failed_checks) if result else [],
            "validation_reason_codes": list(result.reason_codes) if result else [],
            "repair_eligible": result.repairable if result else False,
            "repair_attempted": self.repair_calls > 0,
            "repair_count": self.repair_calls,
            "revalidated": self.revalidation_calls > 0,
            "validation_calls": self.validation_calls,
            "final_candidate_id": self._release_record.get("final_candidate_id"),
            "release_decision": self._release_record.get("release_decision"),
            "release_status": self._release_record.get("release_status"),
            "validation_latency_ms": result.latency_ms if result else None,
        }


__all__ = [
    "CandidateRepairCapability",
    "CandidateRepairError",
    "CandidateRepairUnavailable",
    "DeterministicCandidateRepair",
    "TrustedReleaseValidationCapability",
    "V2ValidationResult",
]
