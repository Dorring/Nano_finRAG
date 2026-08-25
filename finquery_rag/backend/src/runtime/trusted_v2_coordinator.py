"""TV2-02 Supervisor plus bounded-runtime coordinator.

This module wires the existing one-call SupervisorService to the existing
PLAN -> ACT -> OBSERVE -> EVALUATE bounded adaptive controller.  It does not
wire real R4 retrieval, Binder, Calculator, Generator, Validator, or
TrustedRAGRuntimeV2.  Those components arrive in later TV2 gates.
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
from rag_v2.supervisor import SupervisorService, validate_plan_v2_01

from .runtime_contract import ReleaseStatus
from .trusted_v2_capabilities import TrustedV2CapabilityPorts
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

    @classmethod
    def from_state(
        cls,
        *,
        request_id: str,
        plan_id: str | None,
        state: AdaptiveRAGStateV1,
        reason_codes: Iterable[str],
    ) -> "V2ExecutionTrace":
        no_progress_count = int(state.stop_reason == ReasonCode.NO_PROGRESS.value)
        return cls(
            request_id=request_id,
            plan_id=plan_id,
            transitions=tuple(copy.deepcopy(state.transitions)),
            tool_history=tuple(copy.deepcopy(state.tool_history)),
            reason_codes=tuple(_stable_unique(reason_codes)),
            replan_count=state.replan_rounds,
            tool_call_count=state.tool_calls,
            same_tool_retry_count=sum(state.same_tool_retries.values()),
            no_progress_count=no_progress_count,
            terminal_state=state.status,
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
        }


class _EvaluatorAdapter:
    """Adapt an injected evaluator while prioritizing missing operands."""

    def __init__(
        self,
        delegate: Any,
        *,
        intent: Intent,
    ) -> None:
        self.delegate = delegate
        self.intent = intent

    def evaluate(self, state: AdaptiveRAGStateV1) -> EvidenceEvaluationV1:
        result = self.delegate.evaluate(state)
        if not isinstance(result, EvidenceEvaluationV1):
            raise TypeError("evidence evaluator must return EvidenceEvaluationV1")
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
    ) -> None:
        if not isinstance(supervisor, SupervisorService):
            raise TypeError("supervisor must be SupervisorService")
        self.supervisor = supervisor
        self.capabilities = capabilities
        self.budget = budget or AdaptiveRAGBudgetV1()
        self.allow_test_release = bool(allow_test_release)

    @staticmethod
    def _slot_dicts(plan: SupervisorPlan) -> list[dict[str, Any]]:
        slots: list[dict[str, Any]] = []
        for slot in plan.required_slots:
            payload = slot.to_dict()
            payload["value_required"] = True
            slots.append(payload)
        return slots

    @staticmethod
    def _calculation_requirements(plan: SupervisorPlan) -> dict[str, Any]:
        if plan.intent is not Intent.CALCULATION:
            return {}
        return {
            "operation": plan.operation,
            "operand_slots": [slot.slot_id for slot in plan.required_slots],
        }

    @staticmethod
    def _trace(
        request: V2ExecutionRequest,
        plan_id: str | None,
        state: AdaptiveRAGStateV1 | None,
        reason_codes: Iterable[str],
        terminal_state: str,
    ) -> V2ExecutionTrace:
        if state is not None:
            return V2ExecutionTrace.from_state(
                request_id=request.request_id,
                plan_id=plan_id,
                state=state,
                reason_codes=reason_codes,
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
        )

    @staticmethod
    def _metadata(
        *,
        plan_id: str | None,
        terminal_state: str,
        downstream_execution_wired: bool,
        calculation_port_configured: bool,
    ) -> dict[str, Any]:
        return {
            "coordinator": "bounded_trusted_v2",
            "config_version": "tv2-02",
            "production_routing": False,
            "downstream_execution_wired": downstream_execution_wired,
            "calculation_port_configured": calculation_port_configured,
            "plan_id": plan_id,
            "terminal_state": terminal_state,
        }

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
        terminal_state: str,
    ) -> V2ExecutionOutcome:
        reason_list = _stable_unique(reason_codes)
        trace = self._trace(
            request,
            plan_id,
            state,
            reason_list,
            terminal_state,
        )
        release_status = (
            ReleaseStatus.RELEASED
            if status is V2ExecutionStatus.READY_FOR_RELEASE
            else ReleaseStatus.NOT_RELEASED
        )
        return V2ExecutionOutcome(
            status=status,
            answer=answer,
            reason_codes=reason_list,
            release_status=release_status,
            route=plan.intent.value if plan is not None else None,
            runtime_metadata=self._metadata(
                plan_id=plan_id,
                terminal_state=terminal_state,
                downstream_execution_wired=bool(
                    self.allow_test_release
                    and self.capabilities.generation is not None
                    and self.capabilities.release_validator is not None
                ),
                calculation_port_configured=self.capabilities.calculation is not None,
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
        if plan.next_action is Action.ABSTAIN:
            return self._outcome(
                request=request,
                plan=plan,
                plan_id=plan_id,
                state=None,
                reason_codes=["SUPERVISOR_ABSTAIN"],
                status=V2ExecutionStatus.FAIL_CLOSED,
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
            },
            calculation_requirements=self._calculation_requirements(plan),
        )
        capability_errors: list[Exception] = []
        evaluator = self.capabilities.evidence_evaluator or EvidenceStateEvaluatorV1()
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
                evaluator=_EvaluatorAdapter(evaluator, intent=plan.intent),
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
                reason_codes=["COORDINATOR_EXCEPTION"],
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
                status=V2ExecutionStatus.EXECUTION_ERROR,
                terminal_state="EXECUTION_ERROR",
            )

        final_state = bounded_result.state.status
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

        if final_state == "RELEASE" and self.allow_test_release:
            output = bounded_result.output
            if not isinstance(output, str) or not output.strip():
                return self._outcome(
                    request=request,
                    plan=plan,
                    plan_id=plan_id,
                    state=state,
                    reason_codes=["GENERATION_CONTRACT_INVALID"],
                    status=V2ExecutionStatus.EXECUTION_ERROR,
                    terminal_state="GENERATE",
                )
            return self._outcome(
                request=request,
                plan=plan,
                plan_id=plan_id,
                state=state,
                reason_codes=reasons,
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
            status=V2ExecutionStatus.FAIL_CLOSED,
            terminal_state=final_state,
        )


__all__ = ["BoundedTrustedV2Coordinator", "V2ExecutionTrace"]
