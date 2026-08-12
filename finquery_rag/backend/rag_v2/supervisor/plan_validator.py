from __future__ import annotations

from rag_v2.contracts.errors import PlanValidationError
from rag_v2.contracts.plan import Action, Intent, SupervisorPlan


def validate_plan(plan: SupervisorPlan) -> SupervisorPlan:
    """Validate a plan without repairing or inferring any fields."""

    if not isinstance(plan, SupervisorPlan):
        raise PlanValidationError("expected SupervisorPlan")
    if plan.intent == Intent.CALCULATION and not plan.operation:
        raise PlanValidationError("CALCULATION plans require an operation")
    if plan.intent != Intent.CALCULATION and plan.operation is not None:
        raise PlanValidationError("non-calculation plans cannot carry an operation")
    if plan.next_action == Action.CALCULATE and plan.intent != Intent.CALCULATION:
        raise PlanValidationError("CALCULATE is only valid for CALCULATION intent")
    if plan.next_action == Action.GENERATE and plan.intent == Intent.CALCULATION:
        raise PlanValidationError("CALCULATION must pass through CALCULATE before GENERATE")
    if plan.next_action == Action.REPAIR_GENERATION and plan.intent == Intent.CALCULATION:
        raise PlanValidationError("calculation generation repair is not a planning shortcut")
    return plan
