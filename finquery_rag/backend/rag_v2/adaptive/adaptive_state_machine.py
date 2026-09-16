"""Bounded PLAN→ACT→OBSERVE→EVALUATE adaptive controller.

The controller is an experimental control-plane wrapper.  It does not
replace the existing trusted generator/validator path; generation is exposed
only through an explicit callback after the READY_TO_GENERATE transition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .adaptive_budget import AdaptiveRAGBudgetV1
from .adaptive_contracts import (
    AdaptivePhase,
    AdaptiveRAGStateV1,
    EvidenceDecision,
    EvidenceEvaluationV1,
    EvidencePacketV1,
    ReasonCode,
    ReplanActionV1,
    ToolCapability,
)
from .adaptive_evaluator import EvidenceStateEvaluatorV1
from .adaptive_policy import AdaptiveActionPolicyV1
from .adaptive_progress import ProgressDetectorV1
from .adaptive_replanner import BoundedReplannerV1


ToolFn = Callable[[str, AdaptiveRAGStateV1], Iterable[Mapping[str, Any]]]
CalculatorFn = Callable[[AdaptiveRAGStateV1], Any]
GeneratorFn = Callable[[AdaptiveRAGStateV1], Any]
VerifierFn = Callable[[AdaptiveRAGStateV1, Any], bool]


@dataclass(frozen=True)
class AdaptiveRunResultV1:
    state: AdaptiveRAGStateV1
    evaluation: EvidenceEvaluationV1 | None
    output: Any = None

    @property
    def released(self) -> bool:
        return self.state.status == AdaptivePhase.RELEASE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "released": self.released,
        }


class BoundedAdaptiveRAGV1:
    """Run an observation-driven loop with explicit hard bounds."""

    def __init__(
        self,
        *,
        evaluator: EvidenceStateEvaluatorV1 | None = None,
        replanner: BoundedReplannerV1 | None = None,
        progress_detector: ProgressDetectorV1 | None = None,
        budget: AdaptiveRAGBudgetV1 | None = None,
        policy: AdaptiveActionPolicyV1 | None = None,
    ) -> None:
        self.budget = budget or AdaptiveRAGBudgetV1()
        self.evaluator = evaluator or EvidenceStateEvaluatorV1()
        self.replanner = replanner or BoundedReplannerV1(self.budget)
        self.progress = progress_detector or ProgressDetectorV1()
        # The replanner proposes; the policy permits.  See adaptive_policy.py.
        self.policy = policy or AdaptiveActionPolicyV1(self.budget)

    @staticmethod
    def _capability(value: Any) -> ToolCapability:
        if isinstance(value, ToolCapability):
            return value
        return ToolCapability(str(value))

    @staticmethod
    def _fail(state: AdaptiveRAGStateV1, reason: ReasonCode) -> None:
        state.stop_reason = reason.value
        state.transition(AdaptivePhase.FAIL_CLOSED, reason.value)

    @staticmethod
    def _needs_calculation(
        state: AdaptiveRAGStateV1,
        calculator: CalculatorFn | None,
    ) -> bool:
        """Whether this run should run the calculator inside the loop.

        Two conditions, both required.  The plan must carry calculation
        requirements, and a calculator must have been supplied.  Without a
        calculator the harness is not the owner of calculation: the caller
        keeps that step, which is the legacy contract.
        """

        if calculator is None:
            return False
        return bool(state.calculation_requirements) and not state.calculation_attempted

    def run(
        self,
        state: AdaptiveRAGStateV1,
        tools: Mapping[ToolCapability | str, ToolFn],
        *,
        initial_action: ReplanActionV1 | None = None,
        calculator: CalculatorFn | None = None,
        generator: GeneratorFn | None = None,
        verifier: VerifierFn | None = None,
    ) -> AdaptiveRunResultV1:
        pending = initial_action or ReplanActionV1(
            ToolCapability.SEMANTIC_RETRIEVAL, state.normalized_query,
            ReasonCode.MISSING_SLOT, tuple(state.missing_slots), {},
        )
        evaluation: EvidenceEvaluationV1 | None = None
        output: Any = None
        no_progress = False
        guard = 0
        # One spin per phase transition, not per tool call.  The margin covers
        # PLAN/ACT/OBSERVE/EVALUATE/REPLAN/CALCULATE/READY_TO_GENERATE plus the
        # GENERATE/VERIFY/RELEASE tail for every round.
        while guard < self.budget.max_total_tool_calls * 6 + 20:
            guard += 1
            phase = AdaptivePhase(state.status)
            if phase is AdaptivePhase.PLAN:
                state.transition(AdaptivePhase.ACT, "initial plan accepted")
                continue
            if phase is AdaptivePhase.ACT:
                denied = self.policy.check_tool_call(state)
                if denied is not None:
                    self._fail(state, denied)
                    break
                capability = pending.capability
                try:
                    tool = tools.get(capability) or tools.get(capability.value)
                except (TypeError, AttributeError):
                    tool = None
                if tool is None:
                    self._fail(state, ReasonCode.UNSUPPORTED_TOOL_ROUTE)
                    break
                key = capability.value
                prior = state.same_tool_retries.get(key, 0)
                denied = self.policy.check_tool_retry(state, capability)
                if denied is not None:
                    self._fail(state, denied)
                    break
                state.same_tool_retries[key] = prior + 1
                state.tool_calls += 1
                state.iteration += 1
                state.last_action = pending.to_dict()
                state.record_turn(key, pending.reason_code.value, query=pending.query)
                state.tool_history.append({"capability": key, "query": pending.query, "iteration": state.iteration})
                state.query_history.append(pending.query)
                try:
                    raw_packets = list(tool(pending.query, state))
                except Exception as exc:  # deterministic fail-closed; expose only type
                    state.last_observation = {"error": type(exc).__name__, "packet_count": 0}
                    state.observe_turn(state.last_observation)
                    state.stop_reason = ReasonCode.TOOL_ERROR.value
                    state.transition(AdaptivePhase.OBSERVE, ReasonCode.TOOL_ERROR.value)
                    continue
                packets = [EvidencePacketV1.from_mapping(item) for item in raw_packets]
                state.add_evidence(packets)
                state.last_observation = {"packet_count": len(packets), "evidence_ids": [item.evidence_id for item in packets]}
                state.observe_turn(state.last_observation)
                state.transition(AdaptivePhase.OBSERVE, "tool observation captured")
                continue
            if phase is AdaptivePhase.OBSERVE:
                packets = [EvidencePacketV1.from_mapping(item) for item in state.evidence_packets]
                signature = self.progress.signature(
                    query=pending.query, capability=pending.capability.value,
                    packets=packets, filled_slots=state.filled_slots,
                    missing_slots=state.missing_slots, conflicts=state.conflicts,
                    calculation_ready=state.calculation_ready,
                )
                no_progress = not self.progress.observe(state, signature)
                state.last_observation = {**(state.last_observation or {}), "progress_signature": signature, "no_progress": no_progress}
                state.transition(AdaptivePhase.EVALUATE, "observation evaluated")
                continue
            if phase is AdaptivePhase.EVALUATE:
                if state.stop_reason == ReasonCode.TOOL_ERROR.value:
                    evaluation = EvidenceEvaluationV1(
                        EvidenceDecision.REPAIRABLE, (ReasonCode.TOOL_ERROR,),
                        tuple(slot.get("slot_id", "") for slot in state.required_slots), (),
                        tuple(state.missing_slots), (), "UNKNOWN", (), False,
                    )
                else:
                    evaluation = self.evaluator.evaluate(state)
                if no_progress:
                    evaluation = EvidenceEvaluationV1(
                        EvidenceDecision.TERMINAL_INSUFFICIENT,
                        tuple(dict.fromkeys((*evaluation.reason_codes, ReasonCode.NO_PROGRESS))),
                        evaluation.requested_slots, evaluation.supported_slots,
                        evaluation.missing_slots, evaluation.supporting_evidence_ids,
                        evaluation.temporal_status, evaluation.conflicts,
                        evaluation.calculation_ready,
                    )
                state.missing_slots = list(evaluation.missing_slots)
                state.calculation_ready = evaluation.calculation_ready
                state.conflicts = [dict(item) for item in evaluation.conflicts]
                state.filled_slots = {slot: [] for slot in evaluation.supported_slots}
                if evaluation.decision is EvidenceDecision.SUFFICIENT:
                    if self._needs_calculation(state, calculator):
                        state.transition(AdaptivePhase.CALCULATE, "evidence sufficient; calculation required")
                    else:
                        state.transition(AdaptivePhase.READY_TO_GENERATE, "evidence sufficient")
                elif evaluation.decision is EvidenceDecision.REPAIRABLE:
                    if self.policy.check_replan(state) is not None:
                        self._fail(state, ReasonCode.BUDGET_EXHAUSTED)
                    else:
                        state.transition(AdaptivePhase.REPLAN, "concrete information gap")
                else:
                    reason = ReasonCode.NO_PROGRESS if ReasonCode.NO_PROGRESS in evaluation.reason_codes else next(iter(evaluation.reason_codes), ReasonCode.STRUCTURAL_NOT_READY)
                    self._fail(state, reason)
                continue
            if phase is AdaptivePhase.REPLAN:
                pending = self.replanner.replan(state, evaluation or EvidenceEvaluationV1(EvidenceDecision.TERMINAL_INSUFFICIENT))
                if pending is None:
                    self._fail(state, ReasonCode.NO_PROGRESS if no_progress else ReasonCode.BUDGET_EXHAUSTED)
                    break
                state.replan_rounds += 1
                state.stop_reason = None
                no_progress = False
                state.transition(AdaptivePhase.ACT, f"replan:{pending.reason_code.value}")
                continue
            if phase is AdaptivePhase.CALCULATE:
                # Deterministic calculation is a harness step, not a downstream
                # afterthought: it runs inside the loop so the run trace and the
                # budget cover it.  Entered only when a calculator was supplied.
                # Reachable when the loop is entered with a state already in
                # this phase (a resumed or externally supplied run), even though
                # _needs_calculation gates the normal transition into it.
                if calculator is None:
                    self._fail(state, ReasonCode.CALCULATOR_NOT_WIRED)
                    break
                state.record_turn(AdaptivePhase.CALCULATE.value)
                try:
                    result = calculator(state)
                except Exception as exc:  # deterministic fail-closed; expose only type
                    state.calculation_attempted = True
                    state.last_observation = {"error": type(exc).__name__}
                    state.observe_turn(state.last_observation)
                    self._fail(state, ReasonCode.CALCULATION_ERROR)
                    break
                state.calculation_attempted = True
                calculation_status = getattr(result, "status", None)
                state.last_observation = {
                    "calculation_status": getattr(calculation_status, "value", str(calculation_status)),
                }
                state.observe_turn(state.last_observation)
                state.transition(AdaptivePhase.READY_TO_GENERATE, "calculation complete")
                continue
            if phase is AdaptivePhase.READY_TO_GENERATE:
                if generator is None:
                    return AdaptiveRunResultV1(state, evaluation, output)
                state.transition(AdaptivePhase.GENERATE, "trusted evidence ready")
                continue
            if phase is AdaptivePhase.GENERATE:
                # Reachable when the loop is entered with a state already in
                # this phase (a resumed or externally supplied run).
                if generator is None:
                    self._fail(state, ReasonCode.GENERATOR_NOT_WIRED)
                    break
                state.record_turn(AdaptivePhase.GENERATE.value)
                try:
                    output = generator(state)
                except Exception as exc:  # deterministic fail-closed; expose only type
                    state.last_observation = {"error": type(exc).__name__}
                    state.observe_turn(state.last_observation)
                    self._fail(state, ReasonCode.GENERATION_ERROR)
                    break
                state.last_observation = {"generated": True, "output_type": type(output).__name__}
                state.observe_turn(state.last_observation)
                state.transition(AdaptivePhase.VERIFY, "generator output produced")
                continue
            if phase is AdaptivePhase.VERIFY:
                # A missing verifier is a wiring defect, not a release licence.
                # Releasing on `verifier is None` would let an unvalidated answer
                # through, so this fails closed instead.
                if verifier is None:
                    self._fail(state, ReasonCode.VERIFICATION_NOT_WIRED)
                    break
                try:
                    passed = bool(verifier(state, output))
                except Exception as exc:  # deterministic fail-closed; expose only type
                    state.last_observation = {"error": type(exc).__name__}
                    self._fail(state, ReasonCode.VERIFICATION_ERROR)
                    break
                if passed:
                    state.transition(AdaptivePhase.RELEASE, "existing validator path passed")
                else:
                    state.transition(AdaptivePhase.REPAIR, "existing validator requested repair")
                continue
            if phase is AdaptivePhase.REPAIR:
                self._fail(state, ReasonCode.STRUCTURAL_NOT_READY)
                break
            if phase in {AdaptivePhase.RELEASE, AdaptivePhase.FAIL_CLOSED}:
                break
        if state.status not in {AdaptivePhase.RELEASE.value, AdaptivePhase.FAIL_CLOSED.value}:
            self._fail(state, ReasonCode.BUDGET_EXHAUSTED)
        return AdaptiveRunResultV1(state, evaluation, output)
