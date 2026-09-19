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
    # Relational operations take the multi-evidence role.  Their slots come
    # from the one-slot-per-gold-fact shape, which names every side ``value``:
    # a ranking has no "minuend" and a comparison has no "current", because the
    # operands are a set until the ordering ranks them.
    "comparison": frozenset({"value", "operand"}),
    "ranking": frozenset({"value", "operand"}),
}

#: Operations whose answer needs at least two operands.  Restated here rather
#: than read from the executable registry, which lives in `src` and may not be
#: imported from the contract layer -- the two are kept in agreement by the
#: tests that assert every registered operation is expressible here.
_AT_LEAST_TWO_OPERANDS = frozenset(
    {
        "growth_rate",
        "percentage_share",
        "difference",
        "gross_margin",
        "net_margin",
        "debt_ratio",
        "comparison",
        "ranking",
    }
)

#: Operations whose answer is an ordering over *exactly* two operands.  A
#: comparison with three would be a ranking wearing the wrong operation.
_EXACTLY_TWO_OPERANDS = frozenset({"comparison"})


def validate_plan(plan: SupervisorPlan) -> SupervisorPlan:
    """Validate a plan without repairing or inferring any fields."""

    if not isinstance(plan, SupervisorPlan):
        raise PlanValidationError("expected SupervisorPlan")
    if plan.intent == Intent.CALCULATION and not plan.operation:
        raise PlanValidationError("CALCULATION plans require an operation")
    # There used to be a rule here reading "non-calculation plans cannot carry
    # an operation", and it is the reason a whole stratum could not be executed.
    # Whether something may be executed deterministically is a property of the
    # *operation*, not of the plan's intent: cross-entity comparison and ranking
    # are MULTI_EVIDENCE questions -- they name several sides -- and forbidding
    # them an operation left their answers as prose no validator could check.
    # The rule is gone from the calculator and the coordinator too; those were
    # fixed first and this one was missed, which the fixtures then proved.
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
    # Keyed on the operation rather than the intent, for the same reason the
    # rule above was removed: a plan that names an operation must satisfy that
    # operation's contract whatever intent it was planned under.
    if plan.operation:
        if plan.operation in _AT_LEAST_TWO_OPERANDS and len(plan.required_slots) < 2:
            raise PlanValidationError(f"{plan.operation} requires at least two operand slots")
        if plan.operation in _EXACTLY_TWO_OPERANDS and len(plan.required_slots) != 2:
            raise PlanValidationError(f"{plan.operation} ranks exactly two operands")
        allowed = _OPERATION_ROLES.get(plan.operation, frozenset())
        if allowed and any(slot.role.strip().lower() not in allowed for slot in plan.required_slots):
            raise PlanValidationError(f"slot role does not satisfy operation {plan.operation}")
    return plan
