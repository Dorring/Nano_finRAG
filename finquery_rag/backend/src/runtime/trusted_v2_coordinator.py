"""TV2-02/03 Supervisor plus bounded-runtime coordinator.

This module wires the existing one-call SupervisorService to the existing
PLAN -> ACT -> OBSERVE -> EVALUATE bounded adaptive controller.  TV2-03 may
inject the real R4 retrieval and Semantic Binder ports, while Calculator,
Generator, Validator, and TrustedRAGRuntimeV2 remain later-stage components.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from rag_v2.adaptive import (
    AdaptiveRAGBudgetV1,
    AdaptiveRAGStateV1,
    BoundedAdaptiveRAGV1,
    EvidenceEvaluationV1,
    EvidenceStateEvaluatorV1,
    ReasonCode,
    ReplanActionV1,
    ToolCapability,
)
from rag_v2.contracts.plan import Action, Intent, SupervisorPlan
from rag_v2.supervisor import (
    SemanticAlignmentStatus,
    SupervisorService,
    UnknownSemanticPolicy,
    align_query_to_plan,
    coerce_unknown_semantic_policy,
    validate_plan_v2_01,
)

from .runtime_contract import ReleaseStatus
from .trusted_v2_capabilities import TrustedV2CapabilityPorts
from .trusted_v2_generation import CandidateExecutionResult
from .trusted_v2_validation import (
    CandidateRepairError,
    CandidateRepairUnavailable,
    V2ValidationResult,
)
from .trusted_v2_contracts import (
    TrustedV2ExecutionCoordinator,
    V2ExecutionOutcome,
    V2ExecutionRequest,
    V2ExecutionStatus,
)


def _stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _plan_id(request: V2ExecutionRequest, plan: SupervisorPlan) -> str:
    import hashlib
    import json

    payload = {
        "request_id": request.request_id,
        "standalone_query": request.standalone_query,
        "plan": plan.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class V2ExecutionTrace:
    """Small structured trace; never stores provider reasoning or CoT."""

    request_id: str
    plan_id: str | None
    transitions: tuple[dict[str, Any], ...]
    tool_history: tuple[dict[str, Any], ...]
    reason_codes: tuple[str, ...]
    replan_count: int
    tool_call_count: int
    same_tool_retry_count: int
    no_progress_count: int
    terminal_state: str
    retrieval_rounds: tuple[dict[str, Any], ...] = ()
    candidate_count_per_round: tuple[int, ...] = ()
    candidate_ids_per_round: tuple[tuple[str, ...], ...] = ()
    targeted_slot_ids: tuple[str, ...] = ()
    binder_status_per_round: tuple[str, ...] = ()
    bound_slot_ids: tuple[str, ...] = ()
    missing_slot_ids: tuple[str, ...] = ()
    wrong_period_slots: tuple[str, ...] = ()
    missing_operand_slots: tuple[str, ...] = ()
    conflict_ids: tuple[str, ...] = ()
    bound_evidence_ids: tuple[str, ...] = ()
    generation_route: str | None = None
    route_reason: str | None = None
    calculator_invoked: bool = False
    renderer_invoked: bool = False
    specialist_invoked: bool = False
    calculation_result_id: str | None = None
    candidate_generation_id: str | None = None
    candidate_ready: bool = False
    validation_pending: bool = False
    validation_id: str | None = None
    validation_passed: bool = False
    failed_checks: tuple[str, ...] = ()
    validation_reason_codes: tuple[str, ...] = ()
    repair_eligible: bool = False
    repair_attempted: bool = False
    repair_count: int = 0
    revalidated: bool = False
    final_candidate_id: str | None = None
    release_decision: str | None = None
    release_status: str | None = None
    semantic_alignment: dict[str, Any] | None = None

    @classmethod
    def from_state(
        cls,
        *,
        request_id: str,
        plan_id: str | None,
        state: AdaptiveRAGStateV1,
        reason_codes: Iterable[str],
        capability_trace: Mapping[str, Any] | None = None,
        semantic_alignment: Mapping[str, Any] | None = None,
    ) -> "V2ExecutionTrace":
        no_progress_count = int(state.stop_reason == ReasonCode.NO_PROGRESS.value)
        capability_trace = capability_trace or {}
        retrieval = capability_trace.get("retrieval", {})
        binder = capability_trace.get("binder", {})
        calculation = capability_trace.get("calculation", {})
        generation = capability_trace.get("generation", {})
        validation = capability_trace.get("validation", {})
        if semantic_alignment is None and isinstance(state.plan, Mapping):
            state_alignment = state.plan.get("semantic_alignment")
            if isinstance(state_alignment, Mapping):
                semantic_alignment = state_alignment
        trace_reason_codes = list(reason_codes)
        for record in binder.get("binder_rounds", ()):
            if isinstance(record, Mapping):
                trace_reason_codes.extend(
                    str(item) for item in record.get("reason_codes", ())
                )
        retrieval_rounds = tuple(
            copy.deepcopy(item)
            for item in retrieval.get("retrieval_rounds", ())
            if isinstance(item, Mapping)
        )
        candidate_counts = tuple(
            int(item) for item in retrieval.get("candidate_count_per_round", ())
        )
        candidate_ids = tuple(
            tuple(str(value) for value in item)
            for item in retrieval.get("candidate_ids_per_round", ())
        )
        return cls(
            request_id=request_id,
            plan_id=plan_id,
            transitions=tuple(copy.deepcopy(state.transitions)),
            tool_history=tuple(copy.deepcopy(state.tool_history)),
            reason_codes=tuple(_stable_unique(trace_reason_codes)),
            replan_count=state.replan_rounds,
            tool_call_count=state.tool_calls,
            same_tool_retry_count=sum(state.same_tool_retries.values()),
            no_progress_count=no_progress_count,
            terminal_state=state.status,
            retrieval_rounds=retrieval_rounds,
            candidate_count_per_round=candidate_counts,
            candidate_ids_per_round=candidate_ids,
            targeted_slot_ids=tuple(
                str(item) for item in retrieval.get("targeted_slot_ids", ())
            ),
            binder_status_per_round=tuple(
                str(item) for item in binder.get("binder_status_per_round", ())
            ),
            bound_slot_ids=tuple(
                str(item) for item in binder.get("bound_slot_ids", ())
            ),
            missing_slot_ids=tuple(
                str(item) for item in binder.get("missing_slot_ids", ())
            ),
            wrong_period_slots=tuple(
                str(item) for item in binder.get("wrong_period_slots", ())
            ),
            missing_operand_slots=tuple(
                str(item) for item in binder.get("missing_operand_slots", ())
            ),
            conflict_ids=tuple(str(item) for item in binder.get("conflict_ids", ())),
            bound_evidence_ids=tuple(
                str(item) for item in binder.get("bound_evidence_ids", ())
            ),
            generation_route=(
                str(generation["generation_route"])
                if generation.get("generation_route")
                else None
            ),
            route_reason=(
                str(generation["route_reason"])
                if generation.get("route_reason")
                else None
            ),
            calculator_invoked=bool(calculation.get("calculator_invoked", False)),
            renderer_invoked=bool(generation.get("renderer_invoked", False)),
            specialist_invoked=bool(generation.get("specialist_invoked", False)),
            calculation_result_id=(
                str(calculation["calculation_result_id"])
                if calculation.get("calculation_result_id")
                else None
            ),
            candidate_generation_id=(
                str(generation["candidate_generation_id"])
                if generation.get("candidate_generation_id")
                else None
            ),
            candidate_ready=bool(generation.get("candidate_ready", False)),
            validation_pending=bool(generation.get("validation_pending", False)),
            validation_id=(
                str(validation["validation_id"])
                if validation.get("validation_id")
                else None
            ),
            validation_passed=bool(validation.get("validation_passed", False)),
            failed_checks=tuple(str(item) for item in validation.get("failed_checks", ())),
            validation_reason_codes=tuple(
                str(item) for item in validation.get("validation_reason_codes", ())
            ),
            repair_eligible=bool(validation.get("repair_eligible", False)),
            repair_attempted=bool(validation.get("repair_attempted", False)),
            repair_count=int(validation.get("repair_count", 0)),
            revalidated=bool(validation.get("revalidated", False)),
            final_candidate_id=(
                str(validation["final_candidate_id"])
                if validation.get("final_candidate_id")
                else None
            ),
            release_decision=(
                str(validation["release_decision"])
                if validation.get("release_decision")
                else None
            ),
            release_status=(
                str(validation["release_status"])
                if validation.get("release_status")
                else None
            ),
            semantic_alignment=(
                copy.deepcopy(dict(semantic_alignment))
                if isinstance(semantic_alignment, Mapping)
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "plan_id": self.plan_id,
            "transitions": copy.deepcopy(list(self.transitions)),
            "tool_history": copy.deepcopy(list(self.tool_history)),
            "reason_codes": list(self.reason_codes),
            "replan_count": self.replan_count,
            "tool_call_count": self.tool_call_count,
            "same_tool_retry_count": self.same_tool_retry_count,
            "no_progress_count": self.no_progress_count,
            "terminal_state": self.terminal_state,
            "retrieval_rounds": copy.deepcopy(list(self.retrieval_rounds)),
            "candidate_count_per_round": list(self.candidate_count_per_round),
            "candidate_ids_per_round": [
                list(item) for item in self.candidate_ids_per_round
            ],
            "targeted_slot_ids": list(self.targeted_slot_ids),
            "binder_status_per_round": list(self.binder_status_per_round),
            "bound_slot_ids": list(self.bound_slot_ids),
            "missing_slot_ids": list(self.missing_slot_ids),
            "wrong_period_slots": list(self.wrong_period_slots),
            "missing_operand_slots": list(self.missing_operand_slots),
            "conflict_ids": list(self.conflict_ids),
            "bound_evidence_ids": list(self.bound_evidence_ids),
            "generation_route": self.generation_route,
            "route_reason": self.route_reason,
            "calculator_invoked": self.calculator_invoked,
            "renderer_invoked": self.renderer_invoked,
            "specialist_invoked": self.specialist_invoked,
            "calculation_result_id": self.calculation_result_id,
            "candidate_generation_id": self.candidate_generation_id,
            "candidate_ready": self.candidate_ready,
            "validation_pending": self.validation_pending,
            "validation_id": self.validation_id,
            "validation_passed": self.validation_passed,
            "failed_checks": list(self.failed_checks),
            "validation_reason_codes": list(self.validation_reason_codes),
            "repair_eligible": self.repair_eligible,
            "repair_attempted": self.repair_attempted,
            "repair_count": self.repair_count,
            "revalidated": self.revalidated,
            "final_candidate_id": self.final_candidate_id,
            "release_decision": self.release_decision,
            "release_status": self.release_status,
            "semantic_alignment": (
                copy.deepcopy(self.semantic_alignment)
                if self.semantic_alignment is not None
                else None
            ),
        }


class _EvaluatorAdapter:
    """Adapt an injected evaluator while prioritizing missing operands."""

    def __init__(
        self,
        delegate: Any,
        *,
        intent: Intent,
        errors: list[Exception] | None = None,
    ) -> None:
        self.delegate = delegate
        self.intent = intent
        self.errors = errors if errors is not None else []

    @property
    def bound_evidence_ids(self) -> tuple[str, ...]:
        values = getattr(self.delegate, "last_bound_evidence_ids", ())
        return tuple(str(item) for item in values)

    @property
    def citation_ids(self) -> tuple[str, ...]:
        values = getattr(self.delegate, "last_citation_ids", ())
        return tuple(str(item) for item in values)

    @property
    def bound_slot_bindings(self) -> dict[str, tuple[str, ...]]:
        values = getattr(self.delegate, "last_bound_slot_bindings", {})
        if not isinstance(values, Mapping):
            return {}
        return {
            str(key): tuple(str(item) for item in value)
            for key, value in values.items()
            if isinstance(value, (list, tuple, set))
        }

    def trace_snapshot(self) -> Mapping[str, Any]:
        getter = getattr(self.delegate, "trace_snapshot", None)
        if callable(getter):
            snapshot = getter()
            if isinstance(snapshot, Mapping):
                return snapshot
        return {}

    def evaluate(self, state: AdaptiveRAGStateV1) -> EvidenceEvaluationV1:
        try:
            result = self.delegate.evaluate(state)
        except Exception as exc:
            self.errors.append(exc)
            raise
        if not isinstance(result, EvidenceEvaluationV1):
            raise TypeError("evidence evaluator must return EvidenceEvaluationV1")
        state.bound_evidence_ids = list(self.bound_evidence_ids)
        state.bound_slot_bindings = {
            key: list(value) for key, value in self.bound_slot_bindings.items()
        }
        reasons = list(result.reason_codes)
        if (
            self.intent is Intent.CALCULATION
            and ReasonCode.MISSING_OPERAND in reasons
        ):
            reasons = [
                ReasonCode.MISSING_OPERAND,
                *[reason for reason in reasons if reason is not ReasonCode.MISSING_OPERAND],
            ]
            return EvidenceEvaluationV1(
                decision=result.decision,
                reason_codes=tuple(reasons),
                requested_slots=result.requested_slots,
                supported_slots=result.supported_slots,
                missing_slots=result.missing_slots,
                supporting_evidence_ids=result.supporting_evidence_ids,
                temporal_status=result.temporal_status,
                conflicts=result.conflicts,
                calculation_ready=result.calculation_ready,
                recommended_action=result.recommended_action,
            )
        return result


class _RetrievalTool:
    """Bridge one adaptive tool callback to the structured retrieval port."""

    def __init__(
        self,
        capability: ToolCapability,
        port: Any,
        errors: list[Exception],
    ) -> None:
        self.capability = capability
        self.port = port
        self.errors = errors

    def __call__(
        self,
        query: str,
        state: AdaptiveRAGStateV1,
    ) -> Iterable[Mapping[str, Any]]:
        payload = state.last_action or {}
        try:
            reason = ReasonCode(
                payload.get("reason_code", ReasonCode.MISSING_SLOT.value),
            )
        except (TypeError, ValueError):
            reason = ReasonCode.MISSING_SLOT
        action = ReplanActionV1(
            capability=self.capability,
            query=query,
            reason_code=reason,
            target_slots=tuple(
                str(item) for item in payload.get("target_slots", ())
            ),
            constraints=payload.get("constraints", {}),
        )
        try:
            return self.port.retrieve(action, state)
        except Exception as exc:
            self.errors.append(exc)
            raise


class BoundedTrustedV2Coordinator(TrustedV2ExecutionCoordinator):
    """Compose the existing Supervisor and bounded adaptive controller.

    A real production instance is intentionally not created by TV2-02.  The
    optional downstream ports can produce a released result only when the
    caller explicitly enables allow_test_release for deterministic tests.
    """

    def __init__(
        self,
        supervisor: SupervisorService,
        *,
        capabilities: TrustedV2CapabilityPorts,
        budget: AdaptiveRAGBudgetV1 | None = None,
        allow_test_release: bool = False,
        unknown_semantic_policy: UnknownSemanticPolicy | str = (
            UnknownSemanticPolicy.COMPATIBILITY
        ),
    ) -> None:
        if not isinstance(supervisor, SupervisorService):
            raise TypeError("supervisor must be SupervisorService")
        self.supervisor = supervisor
        self.capabilities = capabilities
        self.budget = budget or AdaptiveRAGBudgetV1()
        self.allow_test_release = bool(allow_test_release)
        self.unknown_semantic_policy = coerce_unknown_semantic_policy(
            unknown_semantic_policy,
        )

    @staticmethod
    def _slot_dicts(plan: SupervisorPlan) -> list[dict[str, Any]]:
        slots: list[dict[str, Any]] = []
        for slot in plan.required_slots:
            payload = slot.to_dict()
            payload["value_required"] = True
            slots.append(payload)
        return slots

    @staticmethod
    def _calculation_requirements(
        plan: SupervisorPlan,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if plan.intent is not Intent.CALCULATION:
            return {}
        requirements: dict[str, Any] = {
            "operation": plan.operation,
            "operand_slots": [slot.slot_id for slot in plan.required_slots],
        }
        metadata = request_metadata or {}
        calculation_metadata = metadata.get("calculation")
        if isinstance(calculation_metadata, Mapping):
            metadata = {**dict(metadata), **dict(calculation_metadata)}
        for key in (
            "source_scale",
            "target_scale",
            "precision",
            "target_metric",
            "label",
        ):
            if key in metadata and metadata[key] is not None:
                requirements[key] = metadata[key]
        return requirements

    def _capability_trace(self) -> dict[str, Any]:
        trace: dict[str, Any] = {}
        for name, port in (
            ("retrieval", self.capabilities.retrieval),
            ("binder", self.capabilities.evidence_evaluator),
            ("calculation", self.capabilities.calculation),
            ("generation", self.capabilities.generation),
            ("validation", self.capabilities.release_validator),
        ):
            getter = getattr(port, "trace_snapshot", None)
            if callable(getter):
                try:
                    snapshot = getter()
                except Exception:
                    snapshot = {}
                if isinstance(snapshot, Mapping):
                    trace[name] = snapshot
        return trace

    def _trace(
        self,
        request: V2ExecutionRequest,
        plan_id: str | None,
        state: AdaptiveRAGStateV1 | None,
        reason_codes: Iterable[str],
        terminal_state: str,
        semantic_alignment: Mapping[str, Any] | None = None,
    ) -> V2ExecutionTrace:
        capability_trace = self._capability_trace()
        if state is not None:
            return V2ExecutionTrace.from_state(
                request_id=request.request_id,
                plan_id=plan_id,
                state=state,
                reason_codes=reason_codes,
                capability_trace=capability_trace,
                semantic_alignment=semantic_alignment,
            )
        return V2ExecutionTrace(
            request_id=request.request_id,
            plan_id=plan_id,
            transitions=(),
            tool_history=(),
            reason_codes=tuple(_stable_unique(reason_codes)),
            replan_count=0,
            tool_call_count=0,
            same_tool_retry_count=0,
            no_progress_count=0,
            terminal_state=terminal_state,
            bound_evidence_ids=tuple(
                str(item)
                for item in capability_trace.get("binder", {}).get(
                    "bound_evidence_ids", ()
                )
            ),
            semantic_alignment=(
                copy.deepcopy(dict(semantic_alignment))
                if isinstance(semantic_alignment, Mapping)
                else None
            ),
        )

    @staticmethod
    def _metadata(
        *,
        plan_id: str | None,
        terminal_state: str,
        downstream_execution_wired: bool,
        calculation_port_configured: bool,
        candidate_generation_wired: bool,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "coordinator": "bounded_trusted_v2",
            "config_version": "tv2-04" if candidate_generation_wired else "tv2-03",
            "production_routing": False,
            "downstream_execution_wired": downstream_execution_wired,
            "calculation_port_configured": calculation_port_configured,
            "candidate_generation_wired": candidate_generation_wired,
            "plan_id": plan_id,
            "terminal_state": terminal_state,
        }
        if extra:
            payload.update(dict(extra))
        return payload

    def _candidate_generation_enabled(self) -> bool:
        return bool(
            getattr(self.capabilities.calculation, "candidate_mode", False)
            or getattr(self.capabilities.generation, "candidate_mode", False)
        )

    def _candidate_stage(
        self,
        *,
        request: V2ExecutionRequest,
        plan: SupervisorPlan,
        plan_id: str,
        state: AdaptiveRAGStateV1,
        evaluator_adapter: _EvaluatorAdapter,
    ) -> V2ExecutionOutcome:
        """Prepare one Candidate and, when wired, cross the TV2-05 gate."""

        calculation_ids: tuple[str, ...] = ()
        candidate_answer: str | None = None
        extra: dict[str, Any] = {}
        if plan.intent is Intent.CALCULATION:
            capability = self.capabilities.calculation
            if capability is None:
                return self._outcome(
                    request=request, plan=plan, plan_id=plan_id, state=state,
                    reason_codes=["CALCULATOR_NOT_WIRED"],
                    status=V2ExecutionStatus.FAIL_CLOSED,
                    terminal_state="EVIDENCE_READY",
                    evidence_ids=evaluator_adapter.bound_evidence_ids,
                    citation_ids=evaluator_adapter.citation_ids,
                )
            try:
                result = capability.calculate(state)
            except Exception:
                return self._outcome(
                    request=request, plan=plan, plan_id=plan_id, state=state,
                    reason_codes=["CALCULATOR_EXCEPTION"],
                    status=V2ExecutionStatus.EXECUTION_ERROR,
                    terminal_state="CALCULATE",
                    evidence_ids=evaluator_adapter.bound_evidence_ids,
                    citation_ids=evaluator_adapter.citation_ids,
                )
            from src.domain.calculation import CalculationResult, CalculationStatus

            if not isinstance(result, CalculationResult):
                return self._outcome(
                    request=request, plan=plan, plan_id=plan_id, state=state,
                    reason_codes=["CALCULATOR_CONTRACT_INVALID"],
                    status=V2ExecutionStatus.EXECUTION_ERROR,
                    terminal_state="CALCULATE",
                    evidence_ids=evaluator_adapter.bound_evidence_ids,
                    citation_ids=evaluator_adapter.citation_ids,
                )
            if result.status is not CalculationStatus.EXECUTED:
                code = (
                    "CALCULATION_FAILED"
                    if result.status is CalculationStatus.FAILED
                    and result.error_code == "PRIMITIVE_EXCEPTION"
                    else "CALCULATION_INVALID"
                )
                status = (
                    V2ExecutionStatus.EXECUTION_ERROR
                    if code == "CALCULATION_FAILED"
                    else V2ExecutionStatus.FAIL_CLOSED
                )
                return self._outcome(
                    request=request, plan=plan, plan_id=plan_id, state=state,
                    reason_codes=[code, result.error_code or "CALCULATION_NOT_EXECUTED"],
                    status=status, terminal_state="CALCULATE",
                    evidence_ids=evaluator_adapter.bound_evidence_ids,
                    citation_ids=evaluator_adapter.citation_ids,
                )
            calculation_ids = (
                (str(getattr(capability, "last_calculation_id")),)
                if getattr(capability, "last_calculation_id", None)
                else ()
            )
            state.calculation_result_id = calculation_ids[0] if calculation_ids else None
            extra["calculation_status"] = result.status.value
            extra["calculation_result_id"] = state.calculation_result_id

        generation = self.capabilities.generation
        if generation is None:
            return self._outcome(
                request=request, plan=plan, plan_id=plan_id, state=state,
                reason_codes=["GENERATOR_NOT_WIRED"],
                status=V2ExecutionStatus.FAIL_CLOSED,
                terminal_state="EVIDENCE_READY",
                evidence_ids=evaluator_adapter.bound_evidence_ids,
                citation_ids=evaluator_adapter.citation_ids,
                calculation_ids=calculation_ids,
            )
        try:
            raw_candidate = generation.generate(state)
        except Exception:
            return self._outcome(
                request=request, plan=plan, plan_id=plan_id, state=state,
                reason_codes=["GENERATION_EXCEPTION"],
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="GENERATE",
                evidence_ids=evaluator_adapter.bound_evidence_ids,
                citation_ids=evaluator_adapter.citation_ids,
                calculation_ids=calculation_ids,
            )

        if isinstance(raw_candidate, CandidateExecutionResult):
            candidate = raw_candidate
            candidate_answer = candidate.candidate_answer
            extra.update(dict(candidate.generation_metadata))
            extra["candidate_status"] = candidate.candidate_status
            extra["candidate_generation_id"] = candidate.candidate_generation_id
            calculation_ids = tuple(candidate.calculation_ids) or calculation_ids
        elif isinstance(raw_candidate, str) and raw_candidate.strip():
            candidate_answer = raw_candidate.strip()
            candidate = CandidateExecutionResult(
                candidate_answer=candidate_answer,
                route=str(getattr(state, "generation_route", "") or "STRUCTURED_SINGLE"),
                route_reason=str(getattr(state, "route_reason", "") or "candidate"),
                bound_evidence_ids=evaluator_adapter.bound_evidence_ids,
                citation_ids=evaluator_adapter.citation_ids,
                calculation_ids=calculation_ids,
            )
        else:
            return self._outcome(
                request=request, plan=plan, plan_id=plan_id, state=state,
                reason_codes=["CANDIDATE_CONTRACT_INVALID"],
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="GENERATE",
                evidence_ids=evaluator_adapter.bound_evidence_ids,
                citation_ids=evaluator_adapter.citation_ids,
                calculation_ids=calculation_ids,
            )

        extra["candidate_ready"] = True
        extra["validation_pending"] = True
        validator = self.capabilities.release_validator
        if getattr(validator, "candidate_mode", False):
            return self._validated_candidate_stage(
                request=request,
                plan=plan,
                plan_id=plan_id,
                state=state,
                candidate=candidate,
                extra_metadata=extra,
            )

        return self._outcome(
            request=request, plan=plan, plan_id=plan_id, state=state,
            reason_codes=["FINAL_VALIDATION_NOT_WIRED"],
            status=V2ExecutionStatus.FAIL_CLOSED,
            answer=candidate.candidate_answer,
            route=candidate.route,
            terminal_state="CANDIDATE_READY_FOR_VALIDATION",
            evidence_ids=candidate.bound_evidence_ids or evaluator_adapter.bound_evidence_ids,
            citation_ids=candidate.citation_ids or evaluator_adapter.citation_ids,
            calculation_ids=candidate.calculation_ids,
            extra_metadata=extra,
        )

    @staticmethod
    def _record_validator_release(
        validator: Any,
        *,
        released: bool,
        candidate: CandidateExecutionResult,
    ) -> None:
        recorder = getattr(validator, "record_release", None)
        if callable(recorder):
            recorder(
                released=released,
                final_candidate_id=candidate.candidate_generation_id,
                release_status=(
                    ReleaseStatus.RELEASED.value
                    if released
                    else ReleaseStatus.NOT_RELEASED.value
                ),
            )

    @staticmethod
    def _validation_metadata(
        extra: Mapping[str, Any],
        *,
        validation: V2ValidationResult | None,
        candidate: CandidateExecutionResult,
        repair_attempted: bool,
        repair_count: int,
        revalidated: bool,
        release_decision: str,
    ) -> dict[str, Any]:
        payload = dict(extra)
        payload.update(
            {
                "config_version": "tv2-05",
                "candidate_generation_id": candidate.candidate_generation_id,
                "final_candidate_id": candidate.candidate_generation_id,
                "validation_status": validation.status if validation else "ERROR",
                "validation_id": validation.validation_id if validation else None,
                "validation_passed": bool(validation and validation.passed),
                "failed_checks": list(validation.failed_checks) if validation else [],
                "validation_reason_codes": list(validation.reason_codes) if validation else [],
                "repair_eligible": bool(validation and validation.repairable),
                "repair_attempted": repair_attempted,
                "repair_count": repair_count,
                "revalidated": revalidated,
                "release_decision": release_decision,
                "release_status": (
                    ReleaseStatus.RELEASED.value
                    if release_decision == "RELEASED"
                    else ReleaseStatus.NOT_RELEASED.value
                ),
            }
        )
        if validation is not None:
            payload["validation"] = validation.to_dict()
        return payload

    def _validated_candidate_stage(
        self,
        *,
        request: V2ExecutionRequest,
        plan: SupervisorPlan,
        plan_id: str,
        state: AdaptiveRAGStateV1,
        candidate: CandidateExecutionResult,
        extra_metadata: Mapping[str, Any],
    ) -> V2ExecutionOutcome:
        validator = self.capabilities.release_validator
        if validator is None:
            raise RuntimeError("candidate validation stage requires a validator")

        def failed_outcome(
            current: CandidateExecutionResult,
            validation: V2ValidationResult | None,
            reasons: Iterable[str],
            *,
            status: V2ExecutionStatus = V2ExecutionStatus.FAIL_CLOSED,
            terminal_state: str = "FINAL_VALIDATION",
            repair_attempted: bool = False,
            repair_count: int = 0,
            revalidated: bool = False,
        ) -> V2ExecutionOutcome:
            self._record_validator_release(validator, released=False, candidate=current)
            metadata = self._validation_metadata(
                extra_metadata,
                validation=validation,
                candidate=current,
                repair_attempted=repair_attempted,
                repair_count=repair_count,
                revalidated=revalidated,
                release_decision="NOT_RELEASED",
            )
            return self._outcome(
                request=request, plan=plan, plan_id=plan_id, state=state,
                reason_codes=reasons,
                status=status,
                answer=current.candidate_answer,
                route=current.route,
                terminal_state=terminal_state,
                evidence_ids=current.bound_evidence_ids,
                citation_ids=current.citation_ids,
                calculation_ids=current.calculation_ids,
                extra_metadata=metadata,
                validator_status=validation.status if validation else "ERROR",
            )

        try:
            validation = validator.validate(state, candidate)
        except Exception:
            return failed_outcome(
                candidate,
                None,
                ["VALIDATOR_EXCEPTION"],
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="FINAL_VALIDATION_ERROR",
            )
        if not isinstance(validation, V2ValidationResult):
            return failed_outcome(
                candidate,
                None,
                ["VALIDATOR_CONTRACT_INVALID"],
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="FINAL_VALIDATION_ERROR",
            )

        if validation.passed:
            self._record_validator_release(validator, released=True, candidate=candidate)
            metadata = self._validation_metadata(
                extra_metadata,
                validation=validation,
                candidate=candidate,
                repair_attempted=False,
                repair_count=0,
                revalidated=False,
                release_decision="RELEASED",
            )
            return self._outcome(
                request=request, plan=plan, plan_id=plan_id, state=state,
                reason_codes=["VALIDATED_RELEASE"],
                status=V2ExecutionStatus.READY_FOR_RELEASE,
                answer=candidate.candidate_answer,
                route=candidate.route,
                terminal_state="RELEASED",
                evidence_ids=candidate.bound_evidence_ids,
                citation_ids=candidate.citation_ids,
                calculation_ids=candidate.calculation_ids,
                extra_metadata=metadata,
                validator_status=validation.status,
            )

        if not validation.repairable:
            return failed_outcome(candidate, validation, validation.reason_codes)

        try:
            repaired = validator.repair(state, candidate, validation)
        except CandidateRepairUnavailable:
            return failed_outcome(
                candidate,
                validation,
                [*validation.reason_codes, "REPAIR_UNAVAILABLE"],
                repair_attempted=True,
                repair_count=1,
            )
        except CandidateRepairError:
            return failed_outcome(
                candidate,
                validation,
                [*validation.reason_codes, "REPAIR_EXCEPTION"],
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="REPAIR_ERROR",
                repair_attempted=True,
                repair_count=1,
            )
        except Exception:
            return failed_outcome(
                candidate,
                validation,
                [*validation.reason_codes, "REPAIR_EXCEPTION"],
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="REPAIR_ERROR",
                repair_attempted=True,
                repair_count=1,
            )
        if not isinstance(repaired, CandidateExecutionResult):
            return failed_outcome(
                candidate,
                validation,
                [*validation.reason_codes, "REPAIR_CONTRACT_INVALID"],
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="REPAIR_ERROR",
                repair_attempted=True,
                repair_count=1,
            )

        recorder = getattr(validator, "record_revalidation", None)
        if callable(recorder):
            recorder()
        try:
            second = validator.validate(state, repaired)
        except Exception:
            return failed_outcome(
                repaired,
                validation,
                [*validation.reason_codes, "REVALIDATION_EXCEPTION"],
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="REVALIDATION_ERROR",
                repair_attempted=True,
                repair_count=1,
                revalidated=True,
            )
        if not isinstance(second, V2ValidationResult):
            return failed_outcome(
                repaired,
                validation,
                [*validation.reason_codes, "REVALIDATION_CONTRACT_INVALID"],
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="REVALIDATION_ERROR",
                repair_attempted=True,
                repair_count=1,
                revalidated=True,
            )
        if second.passed:
            self._record_validator_release(validator, released=True, candidate=repaired)
            metadata = self._validation_metadata(
                extra_metadata,
                validation=second,
                candidate=repaired,
                repair_attempted=True,
                repair_count=1,
                revalidated=True,
                release_decision="RELEASED",
            )
            metadata["repaired_from_validation_id"] = validation.validation_id
            return self._outcome(
                request=request, plan=plan, plan_id=plan_id, state=state,
                reason_codes=["REPAIRED_ONCE", "VALIDATED_RELEASE"],
                status=V2ExecutionStatus.READY_FOR_RELEASE,
                answer=repaired.candidate_answer,
                route=repaired.route,
                terminal_state="RELEASED_AFTER_REPAIR",
                evidence_ids=repaired.bound_evidence_ids,
                citation_ids=repaired.citation_ids,
                calculation_ids=repaired.calculation_ids,
                extra_metadata=metadata,
                validator_status=second.status,
            )
        return failed_outcome(
            repaired,
            second,
            [
                *validation.reason_codes,
                "REPAIR_REVALIDATION_FAILED",
                *second.reason_codes,
            ],
            repair_attempted=True,
            repair_count=1,
            revalidated=True,
        )

    def _outcome(
        self,
        *,
        request: V2ExecutionRequest,
        plan: SupervisorPlan | None,
        plan_id: str | None,
        state: AdaptiveRAGStateV1 | None,
        reason_codes: Iterable[str],
        status: V2ExecutionStatus,
        answer: str | None = None,
        evidence_ids: Iterable[str] = (),
        citation_ids: Iterable[str] = (),
        calculation_ids: Iterable[str] = (),
        route: str | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
        semantic_alignment: Mapping[str, Any] | None = None,
        validator_status: str | None = None,
        terminal_state: str,
    ) -> V2ExecutionOutcome:
        reason_list = _stable_unique(reason_codes)
        calculation_id_list = _stable_unique(calculation_ids)
        alignment_metadata: Mapping[str, Any] | None = semantic_alignment
        if alignment_metadata is None and state is not None and isinstance(
            state.plan,
            Mapping,
        ):
            state_alignment = state.plan.get("semantic_alignment")
            if isinstance(state_alignment, Mapping):
                alignment_metadata = state_alignment
        metadata_extra = dict(extra_metadata or {})
        if alignment_metadata is not None:
            metadata_extra.setdefault(
                "semantic_alignment",
                copy.deepcopy(dict(alignment_metadata)),
            )
        trace = self._trace(
            request,
            plan_id,
            state,
            reason_list,
            terminal_state,
            semantic_alignment=alignment_metadata,
        )
        release_status = (
            ReleaseStatus.RELEASED
            if status is V2ExecutionStatus.READY_FOR_RELEASE
            else ReleaseStatus.NOT_RELEASED
        )
        return V2ExecutionOutcome(
            status=status,
            answer=answer,
            evidence_ids=_stable_unique(evidence_ids),
            citation_ids=_stable_unique(citation_ids),
            reason_codes=reason_list,
            release_status=release_status,
            route=route or (plan.intent.value if plan is not None else None),
            calculation_ids=calculation_id_list,
            calculation_result_id=(
                calculation_id_list[0] if calculation_id_list else None
            ),
            validator_status=validator_status,
            runtime_metadata=self._metadata(
                plan_id=plan_id,
                terminal_state=terminal_state,
                downstream_execution_wired=bool(
                    (
                        self.allow_test_release
                        and self.capabilities.generation is not None
                        and self.capabilities.release_validator is not None
                    )
                    or (
                        self._candidate_generation_enabled()
                        and bool(
                            getattr(
                                self.capabilities.release_validator,
                                "candidate_mode",
                                False,
                            )
                        )
                    )
                ),
                calculation_port_configured=self.capabilities.calculation is not None,
                candidate_generation_wired=self._candidate_generation_enabled(),
                extra=metadata_extra,
            ),
            debug_metadata={"trace": trace.to_dict()},
            plan_id=plan_id,
        )

    async def execute(
        self,
        request: V2ExecutionRequest,
    ) -> V2ExecutionOutcome:
        if not isinstance(request, V2ExecutionRequest):
            raise TypeError("request must be a V2ExecutionRequest")

        try:
            supervisor_run = self.supervisor.plan(request.standalone_query)
        except Exception:
            return self._outcome(
                request=request,
                plan=None,
                plan_id=None,
                state=None,
                reason_codes=["SUPERVISOR_EXCEPTION"],
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="SUPERVISOR",
            )

        if not supervisor_run.plan_valid:
            if supervisor_run.plan is None:
                return self._outcome(
                    request=request,
                    plan=None,
                    plan_id=None,
                    state=None,
                    reason_codes=["SUPERVISOR_ERROR"],
                    status=V2ExecutionStatus.EXECUTION_ERROR,
                    terminal_state="SUPERVISOR",
                )
            plan_id = _plan_id(request, supervisor_run.plan)
            return self._outcome(
                request=request,
                plan=supervisor_run.plan,
                plan_id=plan_id,
                state=None,
                reason_codes=["INVALID_PLAN"],
                status=V2ExecutionStatus.FAIL_CLOSED,
                terminal_state="PLAN",
            )

        plan = supervisor_run.plan
        if plan is None:
            return self._outcome(
                request=request,
                plan=None,
                plan_id=None,
                state=None,
                reason_codes=["SUPERVISOR_ERROR"],
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="SUPERVISOR",
            )
        plan_id = _plan_id(request, plan)
        try:
            validate_plan_v2_01(plan)
        except Exception:
            return self._outcome(
                request=request,
                plan=plan,
                plan_id=plan_id,
                state=None,
                reason_codes=["INVALID_PLAN"],
                status=V2ExecutionStatus.FAIL_CLOSED,
                terminal_state="PLAN",
            )
        semantic_alignment = align_query_to_plan(
            request.standalone_query,
            plan,
            unknown_policy=self.unknown_semantic_policy,
            semantic_context=(
                request.conversation_metadata.get("semantic_expectations")
                if isinstance(request.conversation_metadata, Mapping)
                and isinstance(
                    request.conversation_metadata.get("semantic_expectations"),
                    Mapping,
                )
                else None
            ),
        )
        if not semantic_alignment.allowed:
            reason_code_by_status = {
                SemanticAlignmentStatus.MISMATCH: ReasonCode.QUERY_PLAN_SEMANTIC_MISMATCH,
                SemanticAlignmentStatus.AMBIGUOUS: ReasonCode.QUERY_PLAN_SEMANTIC_AMBIGUOUS,
                SemanticAlignmentStatus.UNKNOWN: ReasonCode.QUERY_PLAN_SEMANTIC_UNKNOWN,
            }
            reason_code = reason_code_by_status.get(
                semantic_alignment.status,
                ReasonCode.QUERY_PLAN_SEMANTIC_MISMATCH,
            ).value
            return self._outcome(
                request=request,
                plan=plan,
                plan_id=plan_id,
                state=None,
                reason_codes=[reason_code],
                status=V2ExecutionStatus.FAIL_CLOSED,
                semantic_alignment=semantic_alignment.to_dict(),
                terminal_state="PLAN",
            )
        if plan.next_action is Action.ABSTAIN:
            return self._outcome(
                request=request,
                plan=plan,
                plan_id=plan_id,
                state=None,
                reason_codes=["SUPERVISOR_ABSTAIN"],
                status=V2ExecutionStatus.FAIL_CLOSED,
                semantic_alignment=semantic_alignment.to_dict(),
                terminal_state="PLAN",
            )

        state = AdaptiveRAGStateV1.new(
            request_id=request.request_id,
            query=request.standalone_query,
            intent=plan.intent.value,
            task_type=plan.intent.value,
            required_slots=self._slot_dicts(plan),
            plan={
                "supervisor_plan": plan.to_dict(),
                "plan_id": plan_id,
                "semantic_alignment": semantic_alignment.to_dict(),
            },
            calculation_requirements=self._calculation_requirements(
                plan, request.request_metadata
            ),
        )
        capability_errors: list[Exception] = []
        evaluator = self.capabilities.evidence_evaluator or EvidenceStateEvaluatorV1()
        evaluator_adapter = _EvaluatorAdapter(
            evaluator,
            intent=plan.intent,
            errors=capability_errors,
        )
        tools: dict[str, Any] = {}
        if self.capabilities.retrieval is not None:
            for capability in ToolCapability:
                tools[capability.value] = _RetrievalTool(
                    capability,
                    self.capabilities.retrieval,
                    capability_errors,
                )
        initial_action = ReplanActionV1(
            capability=ToolCapability.SEMANTIC_RETRIEVAL,
            query=request.standalone_query,
            reason_code=ReasonCode.MISSING_SLOT,
            target_slots=tuple(slot.slot_id for slot in plan.required_slots),
        )
        generator = None
        verifier = None
        if (
            self.allow_test_release
            and self.capabilities.generation is not None
            and self.capabilities.release_validator is not None
        ):
            generator = self.capabilities.generation.generate
            verifier = self.capabilities.release_validator.validate

        try:
            bounded_result = BoundedAdaptiveRAGV1(
                evaluator=evaluator_adapter,
                budget=self.budget,
            ).run(
                state,
                tools,
                initial_action=initial_action,
                generator=generator,
                verifier=verifier,
            )
        except Exception:
            return self._outcome(
                request=request,
                plan=plan,
                plan_id=plan_id,
                state=state,
                reason_codes=(
                    ["CAPABILITY_EXCEPTION"]
                    if capability_errors
                    else ["COORDINATOR_EXCEPTION"]
                ),
                evidence_ids=evaluator_adapter.bound_evidence_ids,
                citation_ids=evaluator_adapter.citation_ids,
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="EXECUTION_ERROR",
            )

        if capability_errors:
            return self._outcome(
                request=request,
                plan=plan,
                plan_id=plan_id,
                state=state,
                reason_codes=["CAPABILITY_EXCEPTION"],
                evidence_ids=evaluator_adapter.bound_evidence_ids,
                citation_ids=evaluator_adapter.citation_ids,
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="EXECUTION_ERROR",
            )

        final_state = bounded_result.state.status
        if final_state == "READY_TO_GENERATE" and self._candidate_generation_enabled():
            return self._candidate_stage(
                request=request,
                plan=plan,
                plan_id=plan_id,
                state=state,
                evaluator_adapter=evaluator_adapter,
            )
        evaluation_reasons = (
            item.value
            for item in (
                bounded_result.evaluation.reason_codes
                if bounded_result.evaluation is not None
                else ()
            )
        )
        reasons = [*evaluation_reasons]
        if state.stop_reason:
            reasons.append(state.stop_reason)
        if final_state == "READY_TO_GENERATE":
            reasons.append("DOWNSTREAM_EXECUTION_NOT_WIRED")
        if final_state == "RELEASE":
            if not self.allow_test_release:
                reasons.append("DOWNSTREAM_EXECUTION_NOT_WIRED")
            else:
                reasons.append("READY_FOR_RELEASE")
        if final_state == "FAIL_CLOSED" and not reasons:
            reasons.append("FAIL_CLOSED")

        bound_evidence_ids = evaluator_adapter.bound_evidence_ids
        citation_ids = evaluator_adapter.citation_ids

        if final_state == "RELEASE" and self.allow_test_release:
            output = bounded_result.output
            if not isinstance(output, str) or not output.strip():
                return self._outcome(
                    request=request,
                    plan=plan,
                    plan_id=plan_id,
                    state=state,
                    reason_codes=["GENERATION_CONTRACT_INVALID"],
                    evidence_ids=bound_evidence_ids,
                    citation_ids=citation_ids,
                    status=V2ExecutionStatus.EXECUTION_ERROR,
                    terminal_state="GENERATE",
                )
            return self._outcome(
                request=request,
                plan=plan,
                plan_id=plan_id,
                state=state,
                reason_codes=reasons,
                evidence_ids=bound_evidence_ids,
                citation_ids=citation_ids,
                status=V2ExecutionStatus.READY_FOR_RELEASE,
                answer=output,
                terminal_state=final_state,
            )

        return self._outcome(
            request=request,
            plan=plan,
            plan_id=plan_id,
            state=state,
            reason_codes=reasons,
            evidence_ids=bound_evidence_ids,
            citation_ids=citation_ids,
            status=V2ExecutionStatus.FAIL_CLOSED,
            terminal_state=final_state,
        )


__all__ = ["BoundedTrustedV2Coordinator", "V2ExecutionTrace"]
