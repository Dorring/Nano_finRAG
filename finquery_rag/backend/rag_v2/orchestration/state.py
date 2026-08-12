from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from rag_v2.contracts.calculation import CalculationResultPacket
from rag_v2.contracts.errors import StateTransitionError
from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding
from rag_v2.contracts.plan import Action, Intent, SupervisorPlan
from rag_v2.contracts.validation import ValidationDecision, ValidationResult
from rag_v2.supervisor.plan_validator import validate_plan

from .budgets import RepairBudget


class State(str, Enum):
    RECEIVED = "RECEIVED"
    PLANNED = "PLANNED"
    RETRIEVED = "RETRIEVED"
    MATERIALIZED = "MATERIALIZED"
    BOUND = "BOUND"
    CALCULATED = "CALCULATED"
    GENERATED = "GENERATED"
    VALIDATED = "VALIDATED"
    REPAIRING = "REPAIRING"
    RELEASED = "RELEASED"
    ABSTAINED = "ABSTAINED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class TransitionRecord:
    from_state: State
    action: str
    to_state: State
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_state": self.from_state.value,
            "action": self.action,
            "to_state": self.to_state.value,
            "reason": self.reason,
        }


class StateMachine:
    """Small deterministic V2 control-plane state machine.

    Providers may propose plans or bindings, but this object alone decides
    whether a transition is executable.  It has no model, retrieval, or
    production dependencies.
    """

    def __init__(self, budget: RepairBudget | None = None) -> None:
        self.budget = budget or RepairBudget()
        self.state = State.RECEIVED
        self.plan: SupervisorPlan | None = None
        self.binding: EvidenceBinding | None = None
        self.calculation_result: CalculationResultPacket | None = None
        self.validation_result: ValidationResult | None = None
        self.retrieval_repairs = 0
        self.generation_repairs = 0
        self.tool_steps = 0
        self.repair_kind: str | None = None
        self.history: list[TransitionRecord] = []

    @property
    def terminal(self) -> bool:
        return self.state in {State.RELEASED, State.ABSTAINED, State.FAILED}

    @property
    def can_generate(self) -> bool:
        return self.state in {State.BOUND, State.CALCULATED} and self.binding is not None and self.binding.is_bound

    def accept_plan(self, plan: SupervisorPlan) -> None:
        if self.state is not State.RECEIVED:
            raise StateTransitionError("a plan can only be accepted from RECEIVED")
        validate_plan(plan)
        self.plan = plan
        self._move(State.PLANNED, "PLAN", "validated supervisor plan")

    def record_binding(self, binding: EvidenceBinding) -> None:
        if self.state not in {State.RETRIEVED, State.MATERIALIZED}:
            raise StateTransitionError("binding can only be recorded after retrieval/materialization")
        self.binding = binding

    def record_materialized(self) -> None:
        """Record completion of query-independent FinancialFact materialization."""

        self._require(State.RETRIEVED, Action.BIND)
        self._move(State.MATERIALIZED, "MATERIALIZE", "typed evidence materialized")

    def record_calculation(self, result: CalculationResultPacket) -> None:
        if self.state is not State.CALCULATED:
            raise StateTransitionError("calculation result requires CALCULATED state")
        self.calculation_result = result

    def record_validation(self, result: ValidationResult) -> None:
        if self.state is not State.GENERATED:
            raise StateTransitionError("validation requires GENERATED state")
        self.validation_result = result
        self._move(State.VALIDATED, "VALIDATE", result.decision.value)

    def begin_repair(self, kind: str) -> None:
        if kind not in {"retrieval", "generation"}:
            raise StateTransitionError("repair kind must be retrieval or generation")
        if kind == "retrieval":
            if self.state not in {State.RETRIEVED, State.MATERIALIZED} or self.binding is None or self.binding.status not in {BindingStatus.MISSING, BindingStatus.AMBIGUOUS}:
                raise StateTransitionError("retrieval repair requires missing/ambiguous binding")
            if self.retrieval_repairs >= self.budget.retrieval_repair_max:
                self._terminal(State.ABSTAINED, "retrieval repair budget exhausted")
                return
        else:
            if self.state is not State.VALIDATED or self.validation_result is None or self.validation_result.decision is ValidationDecision.PASS:
                raise StateTransitionError("generation repair requires failed validation")
            if self.generation_repairs >= self.budget.generation_repair_max:
                self._terminal(State.ABSTAINED, "generation repair budget exhausted")
                return
        self.repair_kind = kind
        self._move(State.REPAIRING, f"REPAIR_{kind.upper()}", "repair budget available")

    def execute(self, action: Action) -> None:
        if self.terminal:
            raise StateTransitionError(f"cannot execute {action.value} from terminal state {self.state.value}")
        if action is Action.ABSTAIN:
            self._terminal(State.ABSTAINED, "explicit abstain")
            return
        if action is Action.STOP:
            self._terminal(State.FAILED, "explicit stop")
            return
        if action is Action.RETRIEVE:
            self._require(State.PLANNED, action)
            self._consume_tool_step()
            self._move(State.RETRIEVED, action.value, "retrieval action accepted")
        elif action is Action.BIND:
            if self.state not in {State.RETRIEVED, State.MATERIALIZED}:
                raise StateTransitionError(f"BIND requires RETRIEVED/MATERIALIZED, got {self.state.value}")
            if self.binding is None or not self.binding.is_bound:
                raise StateTransitionError("BIND requires a complete BOUND EvidenceBinding")
            self._consume_tool_step()
            self._move(State.BOUND, action.value, "binding contract satisfied")
        elif action is Action.REPAIR_RETRIEVAL:
            self._require(State.REPAIRING, action)
            if self.repair_kind != "retrieval":
                raise StateTransitionError("REPAIR_RETRIEVAL requires retrieval repair mode")
            self._consume_tool_step()
            self.retrieval_repairs += 1
            self._move(State.RETRIEVED, action.value, "retrieval repair completed")
            self.repair_kind = None
        elif action is Action.CALCULATE:
            self._require(State.BOUND, action)
            if self.plan is None or self.plan.intent is not Intent.CALCULATION:
                raise StateTransitionError("CALCULATE requires CALCULATION intent")
            self._consume_tool_step()
            self._move(State.CALCULATED, action.value, "calculation action accepted")
        elif action is Action.GENERATE:
            if not self.can_generate:
                raise StateTransitionError("GENERATE requires BOUND evidence and complete binding")
            self._consume_tool_step()
            self._move(State.GENERATED, action.value, "generation action accepted")
        elif action is Action.REPAIR_GENERATION:
            self._require(State.REPAIRING, action)
            if self.repair_kind != "generation":
                raise StateTransitionError("REPAIR_GENERATION requires generation repair mode")
            self._consume_tool_step()
            self.generation_repairs += 1
            self._move(State.GENERATED, action.value, "generation repair completed")
            self.repair_kind = None
        else:
            raise StateTransitionError(f"unsupported action: {action.value}")

    def release(self) -> None:
        if self.state is not State.VALIDATED or self.validation_result is None:
            raise StateTransitionError("release requires VALIDATED state")
        if self.validation_result.decision is not ValidationDecision.PASS:
            raise StateTransitionError("release requires ValidationDecision.PASS")
        self._move(State.RELEASED, "RELEASE", "validation passed")

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "plan": self.plan.to_dict() if self.plan else None,
            "binding": self.binding.to_dict() if self.binding else None,
            "calculation_result": self.calculation_result.to_dict() if self.calculation_result else None,
            "validation": self.validation_result.to_dict() if self.validation_result else None,
            "retrieval_repairs": self.retrieval_repairs,
            "generation_repairs": self.generation_repairs,
            "tool_steps": self.tool_steps,
            "repair_kind": self.repair_kind,
            "history": [record.to_dict() for record in self.history],
        }

    def _consume_tool_step(self) -> None:
        if self.tool_steps >= self.budget.total_tool_steps_max:
            self._terminal(State.ABSTAINED, "total tool-step budget exhausted")
            raise StateTransitionError("total tool-step budget exhausted; state is ABSTAINED")
        self.tool_steps += 1

    def _require(self, expected: State, action: Action) -> None:
        if self.state is not expected:
            raise StateTransitionError(f"{action.value} requires {expected.value}, got {self.state.value}")

    def _move(self, state: State, action: str, reason: str | None) -> None:
        self.history.append(TransitionRecord(self.state, action, state, reason))
        self.state = state

    def _terminal(self, state: State, reason: str) -> None:
        if self.terminal:
            return
        self._move(state, state.value, reason)
