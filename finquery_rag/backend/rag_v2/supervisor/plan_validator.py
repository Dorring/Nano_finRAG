from __future__ import annotations

from rag_v2.contracts.derived_metric import get_derived_metric_definition
from rag_v2.contracts.errors import PlanValidationError
from rag_v2.contracts.plan import Action, ExecutionDAG, Intent, SupervisorPlan


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
        "part",
        "total",
    }
)

_OPERATION_ROLES = {
    "growth_rate": frozenset({"current", "prior", "current_period", "base_period"}),
    "percentage_share": frozenset({"numerator", "denominator", "part", "total"}),
    "difference": frozenset({"minuend", "subtrahend", "current", "previous", "prior"}),
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

MAX_DAG_STEPS = 10
MAX_DAG_DEPTH = 5
MAX_STEP_OPERANDS = 10


def validate_plan(plan: SupervisorPlan) -> SupervisorPlan:
    """Validate a plan without repairing or inferring any fields."""

    if not isinstance(plan, SupervisorPlan):
        raise PlanValidationError("expected SupervisorPlan")
    if plan.intent == Intent.CALCULATION:
        if plan.operation is not None and plan.execution_dag is not None:
            raise PlanValidationError("CALCULATION plans must specify either operation or execution_dag, not both")
        if plan.operation is None and plan.execution_dag is None:
            raise PlanValidationError("CALCULATION plans require an operation or execution_dag")
    else:
        if plan.execution_dag is not None:
            raise PlanValidationError("execution_dag is only valid for CALCULATION intent")
    if plan.next_action == Action.CALCULATE and plan.intent != Intent.CALCULATION:
        raise PlanValidationError("CALCULATE is only valid for CALCULATION intent")
    if plan.next_action == Action.GENERATE and plan.intent == Intent.CALCULATION:
        raise PlanValidationError("CALCULATION must pass through CALCULATE before GENERATE")
    if plan.next_action == Action.REPAIR_GENERATION and plan.intent == Intent.CALCULATION:
        raise PlanValidationError("calculation generation repair is not a planning shortcut")
    return plan


def validate_execution_dag(plan: SupervisorPlan, dag: ExecutionDAG) -> None:
    """Validate the acyclicity, structural integrity, and semantic conformance of a DAG."""

    if len(dag.steps) > MAX_DAG_STEPS:
        raise PlanValidationError(
            f"execution_dag steps exceed limit ({len(dag.steps)} > {MAX_DAG_STEPS})"
        )

    slot_ids = {slot.slot_id for slot in plan.required_slots}
    step_ids = [step.step_id for step in dag.steps]
    step_id_set = set(step_ids)
    if len(step_id_set) != len(step_ids):
        raise PlanValidationError("execution_dag steps must have unique step_ids")

    if dag.output_step_id not in step_id_set:
        raise PlanValidationError(f"output_step_id {dag.output_step_id!r} not found in steps")

    # Step dependencies
    adjacency: dict[str, list[str]] = {s: [] for s in step_ids}
    in_degree: dict[str, int] = {s: 0 for s in step_ids}
    step_by_id = {s.step_id: s for s in dag.steps}
    referenced_slots: set[str] = set()

    for step in dag.steps:
        if len(step.operands) > MAX_STEP_OPERANDS:
            raise PlanValidationError(
                f"step {step.step_id} operands exceed limit ({len(step.operands)} > {MAX_STEP_OPERANDS})"
            )
        for operand in step.operands:
            if operand.is_slot:
                if operand.ref_id not in slot_ids:
                    raise PlanValidationError(
                        f"dangling slot reference: {operand.ref_id} in step {step.step_id}"
                    )
                referenced_slots.add(operand.ref_id)
            elif operand.is_step:
                if operand.ref_id not in step_id_set:
                    raise PlanValidationError(
                        f"dangling step reference: {operand.ref_id} in step {step.step_id}"
                    )
                adjacency[operand.ref_id].append(step.step_id)
                in_degree[step.step_id] += 1
            else:
                raise PlanValidationError(f"invalid operand ref_type: {operand.ref_type}")

    # Topological sort / cycle detection
    queue = [s for s in step_ids if in_degree[s] == 0]
    visited_count = 0
    step_depth: dict[str, int] = {s: 1 for s in step_ids}

    while queue:
        curr = queue.pop(0)
        visited_count += 1
        curr_depth = step_depth[curr]
        if curr_depth > MAX_DAG_DEPTH:
            raise PlanValidationError(
                f"execution_dag depth exceeds limit ({curr_depth} > {MAX_DAG_DEPTH})"
            )
        for nxt in adjacency[curr]:
            step_depth[nxt] = max(step_depth[nxt], curr_depth + 1)
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if visited_count != len(step_ids):
        raise PlanValidationError("execution_dag contains cycles")

    # Reachability: verify all steps contribute to output_step_id (backward reachability)
    reverse_adj: dict[str, list[str]] = {s: [] for s in step_ids}
    for step in dag.steps:
        for operand in step.operands:
            if operand.is_step:
                reverse_adj[step.step_id].append(operand.ref_id)

    reached: set[str] = set()
    b_queue = [dag.output_step_id]
    reached.add(dag.output_step_id)
    while b_queue:
        curr = b_queue.pop(0)
        for pred in reverse_adj.get(curr, []):
            if pred not in reached:
                reached.add(pred)
                b_queue.append(pred)

    if reached != step_id_set:
        unreached = step_id_set - reached
        raise PlanValidationError(
            f"execution_dag contains isolated or non-contributing steps: {sorted(unreached)}"
        )

    # Step operation constraints
    for step in dag.steps:
        if step.operation in _AT_LEAST_TWO_OPERANDS and len(step.operands) < 2:
            raise PlanValidationError(
                f"step {step.step_id} operation {step.operation} requires at least two operands"
            )
        if step.operation in _EXACTLY_TWO_OPERANDS and len(step.operands) != 2:
            raise PlanValidationError(
                f"step {step.step_id} operation {step.operation} requires exactly two operands"
            )
        allowed_roles = _OPERATION_ROLES.get(step.operation)
        if allowed_roles:
            for operand in step.operands:
                if operand.role and operand.role.strip().lower() not in allowed_roles:
                    raise PlanValidationError(
                        f"step {step.step_id} operand role {operand.role} does not satisfy {step.operation}"
                    )

    # Derived metric conformance check
    target_metric = dag.target_metric
    if not target_metric:
        out_step = step_by_id[dag.output_step_id]
        target_metric = out_step.target_metric or out_step.step_id

    # Restricted cash check
    if target_metric == "net_debt":
        for slot in plan.required_slots:
            text_to_check = f"{slot.metric} {slot.role} {slot.scope or ''}".lower()
            if "restricted" in text_to_check:
                raise PlanValidationError("restricted_cash_ambiguity: net_debt requires unrestricted cash disclosure")

    definition = get_derived_metric_definition(dag.definition_version or target_metric)
    if definition is not None:
        planned_slot_metrics = {s.metric.strip().lower() for s in plan.required_slots}
        for req_m in definition.required_leaf_metrics:
            if req_m.lower() not in planned_slot_metrics:
                raise PlanValidationError(
                    f"derived metric {definition.metric} requires leaf slot metric {req_m}"
                )
    elif dag.definition_version is not None:
        raise PlanValidationError(
            f"unsupported derived metric definition_version: {dag.definition_version}"
        )

    # Coordinate consistency and disclosure conditions
    entities = {s.entity for s in plan.required_slots if s.entity}
    if len(entities) > 1:
        raise PlanValidationError("multi-step calculation cannot mix distinct entities across slots")
    periods = {s.period for s in plan.required_slots if s.period}
    if len(periods) > 1:
        if target_metric in {"gross_margin", "net_debt"}:
            raise PlanValidationError(f"composite metric {target_metric} requires matching period across slots")


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
    elif plan.execution_dag is not None:
        validate_execution_dag(plan, plan.execution_dag)
    return plan
