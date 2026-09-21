"""P1.5-S2-C: what decides whether a plan may be executed deterministically.

The calculator used to answer that from the plan's **intent**:

    if plan.intent is not Intent.CALCULATION:
        raise ...("calculator_called_for_non_calculation_plan")

and the coordinator only offered it plans with that intent.  Two gates reading
the same wrong field.

The cost was an entire stratum.  Cross-entity comparison and ranking are planned
as `MULTI_EVIDENCE`, so however clearly such a plan named an operation the
calculator refused it -- and their answers stayed prose that no validator had a
structured result to check, which is how `compare-002` released an answer that
never performed its comparison.

The question "can this be executed deterministically" is a property of the
*operation*, answered by the registry.  Intent answers a different question --
what the question wanted, which decides routing -- and it keeps that job.

These tests assert the split in both directions, because only one of them is the
interesting one: a registered operation must not need the intent, **and** a plan
with the intent and no operation must not acquire execution it does not have.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rag_v2.adaptive import AdaptiveRAGStateV1
from rag_v2.contracts.errors import ContractError
from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan
from src.domain.calculation import (
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
)
from src.finance.calculation_registry import CALCULATION_REGISTRY, supports
from src.runtime.trusted_v2_calculation import DeterministicCalculationCapability

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CALCULATION_MODULE = BACKEND_ROOT / "src" / "runtime" / "trusted_v2_calculation.py"


def _state(*, intent: Intent, operation: str | None) -> AdaptiveRAGStateV1:
    slots = (
        RequiredSlot(
            slot_id="slot-current",
            metric="Revenue",
            period="FY2024",
            role="current",
            value_type="numeric",
            unit=None,
        ),
        RequiredSlot(
            slot_id="slot-previous",
            metric="Revenue",
            period="FY2023",
            role="previous",
            value_type="numeric",
            unit=None,
        ),
    )
    plan = SupervisorPlan(
        intent=intent,
        required_slots=slots,
        operation=operation,
        next_action=Action.RETRIEVE,
    )
    return AdaptiveRAGStateV1.new(
        request_id="capability-authority",
        query="whatever",
        intent=intent.value,
        task_type=intent.value,
        required_slots=[slot.to_dict() for slot in slots],
        plan={"supervisor_plan": plan.to_dict()},
        calculation_requirements={
            "operation": operation,
            "operand_slots": [slot.slot_id for slot in slots],
        },
    )


# --- the registry answers "is this executable" --------------------------------


def test_a_registered_operation_is_executable() -> None:
    for operation in CalculationOperation:
        assert supports(operation) is True, operation


def test_an_unregistered_operation_is_not_executable() -> None:
    assert supports("not_an_operation") is False


def test_no_operation_is_not_executable() -> None:
    """The `None` case is distinct from unsupported -- nothing to execute,
    rather than something that cannot be."""

    assert supports(None) is False


def test_the_registry_and_its_projection_agree() -> None:
    """`supports` and the registry cannot disagree: same mapping, read twice."""

    assert {op for op in CalculationOperation if supports(op)} == set(CALCULATION_REGISTRY)


# --- eligibility no longer comes from the intent ------------------------------


@pytest.mark.parametrize(
    "intent",
    [Intent.MULTI_EVIDENCE, Intent.DIRECT_FACT, Intent.CALCULATION],
    ids=lambda intent: intent.value,
)
def test_a_registered_operation_reaches_the_executor_from_any_intent(
    intent: Intent,
) -> None:
    """The point of the whole change.

    `MULTI_EVIDENCE` is what cross-entity comparison and ranking are planned as.
    Before this, that intent alone was enough for the calculator to refuse, and
    the operation the plan named was never consulted.
    """

    result = DeterministicCalculationCapability().calculate(
        _state(intent=intent, operation="difference")
    )

    # No bound evidence in this fixture, so the honest outcome is BLOCKED --
    # what matters is that it is a *result*, reached by executing the plan,
    # rather than the intent refusal that used to be raised here.
    assert isinstance(result, CalculationResult)
    assert result.status is CalculationStatus.BLOCKED
    assert result.error_code == "INSUFFICIENT_OPERANDS"


def test_the_intent_refusal_is_gone() -> None:
    """Guards the test above from passing for the wrong reason.

    If `calculate` still raised for a non-CALCULATION intent, the call would
    raise rather than return, and the assertion above would error instead of
    pass -- so this states the absence directly.
    """

    state = _state(intent=Intent.MULTI_EVIDENCE, operation="comparison")

    result = DeterministicCalculationCapability().calculate(state)

    assert isinstance(result, CalculationResult)


# --- and the reverse direction ------------------------------------------------


def test_calculation_intent_without_an_operation_does_not_execute() -> None:
    """A plan that asks for a calculation and names none is a planning fault.

    It must not acquire execution it does not have -- and it must not quietly
    become prose either, which is why `None` raises a capability error naming
    the fault rather than returning a not-applicable result.
    """

    with pytest.raises(Exception) as caught:
        DeterministicCalculationCapability().calculate(
            _state(intent=Intent.CALCULATION, operation=None)
        )

    assert "operation" in str(caught.value)


def test_an_unsupported_operation_cannot_be_expressed_as_a_plan() -> None:
    """The plan contract refuses it long before the calculator could.

    Stronger than the calculator failing closed, and worth stating separately:
    an operation outside `_OPERATION_VALUES` cannot be written into a plan at
    all, so there is no path by which an unexecutable operation reaches
    execution.  The calculator's own `unsupported_calculation_operation` guard
    therefore covers the case where the *vocabulary* and the *registry* have
    drifted -- a name that is expressible but not executable -- rather than a
    malformed plan.
    """

    with pytest.raises(ContractError):
        _state(intent=Intent.CALCULATION, operation="not_an_operation")


def test_the_vocabulary_and_the_registry_do_not_drift() -> None:
    """Every name a plan may carry is executable, and the reverse.

    This is the invariant that keeps the guard above from ever being the thing
    that catches a real fault: if the two sets can differ, one of them is
    wrong and nothing says which.
    """

    from rag_v2.contracts.plan import _OPERATION_VALUES

    assert _OPERATION_VALUES == {operation.value for operation in CALCULATION_REGISTRY}


# --- the tripwire -------------------------------------------------------------


def _referenced_names(path: Path, function: str) -> set[str]:
    """Every attribute/name identifier read inside one function.

    Syntax tree, not source text: a comment explaining why the intent gate was
    removed must not be able to satisfy this, and must not fail it either.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            found: set[str] = set()
            for inner in ast.walk(node):
                if isinstance(inner, ast.Attribute):
                    found.add(inner.attr)
                elif isinstance(inner, ast.Name):
                    found.add(inner.id)
            return found
    raise AssertionError(f"{function} not found in {path}")


def test_the_scan_finds_an_intent_read(tmp_path: Path) -> None:
    """Vacuity guard: this tripwire must be able to fail."""

    offending = tmp_path / "offending.py"
    offending.write_text(
        "def calculate(self, state):\n"
        "    return self._plan(state).intent is Intent.CALCULATION\n",
        encoding="utf-8",
    )

    assert "intent" in _referenced_names(offending, "calculate")


def test_calculate_does_not_read_the_plan_intent() -> None:
    """The gate is gone from the function, not merely reordered within it."""

    referenced = _referenced_names(CALCULATION_MODULE, "calculate")

    assert referenced, "the scan yielded nothing to read"
    assert "intent" not in referenced


# --- the third place the coupling lived ---------------------------------------


def test_the_plan_validator_allows_an_operation_on_a_multi_evidence_plan() -> None:
    """`non-calculation plans cannot carry an operation` was the whole blocker.

    The coupling was removed from the calculator and the coordinator first, and
    this rule was missed.  The S2E migration then failed every case with
    INVALID_PLAN -- which is how it was found, and why this test exists rather
    than a second round of reading the same three files.
    """

    from rag_v2.supervisor.plan_validator import validate_plan_v2_01

    plan = SupervisorPlan(
        intent=Intent.MULTI_EVIDENCE,
        required_slots=tuple(
            RequiredSlot(
                slot_id=f"s{index}",
                metric="General and administrative",
                period="FY2025",
                role="value",
                value_type="numeric",
                entity=entity,
            )
            for index, entity in enumerate(("Apple", "Visa"), 1)
        ),
        operation="comparison",
        next_action=Action.RETRIEVE,
    )

    assert validate_plan_v2_01(plan) is plan


def test_the_validator_checks_the_operation_whatever_the_intent() -> None:
    """A named operation must satisfy its own arity contract either way."""

    from rag_v2.contracts.errors import PlanValidationError
    from rag_v2.supervisor.plan_validator import validate_plan_v2_01

    def plan(intent: Intent, operation: str, slots: int) -> SupervisorPlan:
        return SupervisorPlan(
            intent=intent,
            required_slots=tuple(
                RequiredSlot(
                    slot_id=f"s{index}",
                    metric="Revenue",
                    period="FY2025",
                    role="value",
                    value_type="numeric",
                )
                for index in range(slots)
            ),
            operation=operation,
            next_action=Action.RETRIEVE,
        )

    # A comparison ranks exactly two, under either intent.
    for intent in (Intent.MULTI_EVIDENCE, Intent.CALCULATION):
        with pytest.raises(PlanValidationError, match="exactly two"):
            validate_plan_v2_01(plan(intent, "comparison", 3))


def test_a_calculation_without_an_operation_is_still_a_planning_fault() -> None:
    """The rule that was *not* removed, guarded against over-deletion."""

    from rag_v2.contracts.errors import PlanValidationError
    from rag_v2.supervisor.plan_validator import validate_plan_v2_01

    with pytest.raises(PlanValidationError, match="require an operation"):
        validate_plan_v2_01(
            SupervisorPlan(
                intent=Intent.CALCULATION,
                required_slots=(
                    RequiredSlot(
                        slot_id="s1",
                        metric="Revenue",
                        period="FY2024",
                        role="value",
                        value_type="numeric",
                    ),
                ),
                operation=None,
                next_action=Action.RETRIEVE,
            )
        )


def test_a_plan_with_no_operation_is_unaffected() -> None:
    """Every factual and abstention plan, which must not start being rejected."""

    from rag_v2.supervisor.plan_validator import validate_plan_v2_01

    plan = SupervisorPlan(
        intent=Intent.DIRECT_FACT,
        required_slots=(
            RequiredSlot(
                slot_id="s1",
                metric="Revenue",
                period="FY2025",
                role="value",
                value_type="numeric",
            ),
        ),
        operation=None,
        next_action=Action.RETRIEVE,
    )

    assert validate_plan_v2_01(plan) is plan
