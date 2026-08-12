from __future__ import annotations

from rag_v2.contracts.errors import PlanValidationError
from rag_v2.contracts.plan import Action, Intent, SupervisorPlan


_ALLOWED_ROLES = frozenset(
    {
        "value",
        "operand",
        "current",
        "prior",
        "current_period",
        "base_period",
        "numerator",
        "denominator",
        "minuend",
        "subtrahend",
        "gross_profit",
        "revenue",
        "net_income",
        "debt",
        "assets",
    }
)

_OPERATION_ROLES = {
    "growth_rate": frozenset({"current", "prior", "current_period", "base_period"}),
    "percentage_share": frozenset({"numerator", "denominator"}),
    "difference": frozenset({"minuend", "subtrahend"}),
    "sum": frozenset({"operand"}),
    "average": frozenset({"operand"}),
    "gross_margin": frozenset({"gross_profit", "revenue"}),
    "net_margin": frozenset({"net_income", "revenue"}),
    "debt_ratio": frozenset({"debt", "assets", "liabilities"}),
    "scale_conversion": frozenset({"value", "operand"}),
}


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


def validate_plan_v2_01(plan: SupervisorPlan) -> SupervisorPlan:
    """NF-V2-01 strict planning checks layered over the frozen V2-00 rules."""

    validate_plan(plan)
    if plan.next_action not in {Action.RETRIEVE, Action.ABSTAIN}:
        raise PlanValidationError("initial SupervisorPlan may only propose RETRIEVE or ABSTAIN")
    for slot in plan.required_slots:
        if slot.role.strip().lower() not in _ALLOWED_ROLES:
            raise PlanValidationError(f"invalid operand role: {slot.role}")
    if plan.intent is Intent.CALCULATION:
        if plan.operation in {"growth_rate", "percentage_share", "difference", "gross_margin", "net_margin", "debt_ratio"} and len(plan.required_slots) < 2:
            raise PlanValidationError(f"{plan.operation} requires at least two operand slots")
        allowed = _OPERATION_ROLES.get(plan.operation or "", frozenset())
        if allowed and any(slot.role.strip().lower() not in allowed for slot in plan.required_slots):
            raise PlanValidationError(f"slot role does not satisfy operation {plan.operation}")
    return plan
