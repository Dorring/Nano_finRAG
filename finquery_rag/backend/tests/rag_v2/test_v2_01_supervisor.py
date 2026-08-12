from __future__ import annotations

import pytest

from rag_v2.contracts import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.contracts.errors import ContractError, PlanValidationError, StateTransitionError
from rag_v2.orchestration import State, StateMachine
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService, validate_plan_v2_01


def slot(slot_id: str = "slot_1", role: str = "value") -> RequiredSlot:
    return RequiredSlot(slot_id, "total net sales", "FY2025", role, "numeric", None)


def direct(action: Action = Action.RETRIEVE) -> SupervisorPlan:
    return SupervisorPlan(Intent.DIRECT_FACT, (slot(),), None, action)


def growth() -> SupervisorPlan:
    return SupervisorPlan(
        Intent.CALCULATION,
        (slot("current", "current_period"), slot("base", "base_period")),
        "growth_rate",
        Action.RETRIEVE,
    )


def test_provider_abstraction_and_one_call_service() -> None:
    provider = DeterministicFallbackProvider({"q": direct()})
    run = SupervisorService(provider).plan("q")
    assert run.plan_valid is True
    assert run.plan == direct()
    assert provider.last_call is not None


def test_invalid_intent_is_rejected_by_frozen_schema() -> None:
    with pytest.raises(ContractError):
        SupervisorPlan.from_dict({"intent": "NO_ANSWER", "required_slots": [], "operation": None, "next_action": "RETRIEVE"})


def test_calculation_requires_operation_and_roles() -> None:
    with pytest.raises(PlanValidationError):
        validate_plan_v2_01(SupervisorPlan(Intent.CALCULATION, (slot(), slot("b")), None, Action.RETRIEVE))
    assert validate_plan_v2_01(growth()).operation == "growth_rate"


def test_invalid_operand_role_is_rejected() -> None:
    with pytest.raises(PlanValidationError):
        validate_plan_v2_01(SupervisorPlan(Intent.CALCULATION, (slot("a", "made_up"), slot("b", "base_period")), "growth_rate", Action.RETRIEVE))


def test_duplicate_slot_ids_are_rejected() -> None:
    with pytest.raises(ContractError):
        SupervisorPlan(Intent.DIRECT_FACT, (slot(), slot()), None, Action.RETRIEVE)


@pytest.mark.parametrize("action", [Action.CALCULATE, Action.GENERATE])
def test_premature_downstream_actions_are_rejected(action: Action) -> None:
    with pytest.raises(PlanValidationError):
        validate_plan_v2_01(direct(action))


def test_supervisor_cannot_mutate_state_directly() -> None:
    machine = StateMachine()
    machine.accept_plan(direct())
    assert machine.state is State.PLANNED
    with pytest.raises(StateTransitionError):
        machine.execute(Action.CALCULATE)


def test_service_does_not_retry_provider_failure() -> None:
    class FailingProvider:
        provider_name = "test"
        model_name = "test"
        last_call = None
        calls = 0

        def plan(self, question: str):
            self.calls += 1
            raise RuntimeError("parse failure")

    provider = FailingProvider()
    run = SupervisorService(provider).plan("q")
    assert run.plan_valid is False
    assert provider.calls == 1


def test_no_downstream_calls_are_part_of_provider_contract() -> None:
    provider = DeterministicFallbackProvider({"q": direct()})
    run = SupervisorService(provider).plan("q")
    assert run.metadata.provider == "deterministic_fallback"
    assert not hasattr(provider, "retrieve")
    assert not hasattr(provider, "calculate")
