"""TV2-02 Supervisor and bounded coordinator tests."""

from __future__ import annotations

import asyncio
from typing import Any

from rag_v2.adaptive import AdaptiveRAGBudgetV1, AdaptiveRAGStateV1
from rag_v2.adaptive import ReasonCode, ReplanActionV1
from rag_v2.contracts import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from src.runtime import V2ExecutionRequest, V2ExecutionStatus
from src.runtime.trusted_v2_capabilities import TrustedV2CapabilityPorts
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator


def _slot(
    slot_id: str,
    metric: str = "Revenue",
    period: str = "FY2024",
    role: str = "value",
) -> RequiredSlot:
    return RequiredSlot(slot_id, metric, period, role, "numeric", None)


def _plan(
    *slots: RequiredSlot,
    intent: Intent = Intent.DIRECT_FACT,
    operation: str | None = None,
) -> SupervisorPlan:
    return SupervisorPlan(intent, tuple(slots), operation, Action.RETRIEVE)


def _request(query: str = "What was revenue?") -> V2ExecutionRequest:
    return V2ExecutionRequest(
        request_id="tv2-02-request",
        user_id="user-7",
        session_id="session-1",
        original_query=query,
        standalone_query=query,
    )


class ScriptedRetrieval:
    def __init__(self, batches: list[list[dict[str, Any]]]) -> None:
        self.batches = list(batches)
        self.actions: list[ReplanActionV1] = []

    def retrieve(
        self,
        action: ReplanActionV1,
        state: AdaptiveRAGStateV1,
    ) -> list[dict[str, Any]]:
        self.actions.append(action)
        if self.batches:
            return self.batches.pop(0)
        return []


class RaisingRetrieval:
    def retrieve(
        self,
        action: ReplanActionV1,
        state: AdaptiveRAGStateV1,
    ) -> list[dict[str, Any]]:
        raise RuntimeError("retrieval secret")


class FakeCalculation:
    def __init__(self) -> None:
        self.calls = 0

    def calculate(self, state: AdaptiveRAGStateV1) -> str:
        self.calls += 1
        return "should not run in TV2-02"


class FakeGeneration:
    def generate(self, state: AdaptiveRAGStateV1) -> str:
        return "released only by explicit test wiring"


class FakeValidator:
    def __init__(self, decision: bool = True) -> None:
        self.decision = decision
        self.calls = 0

    def validate(self, state: AdaptiveRAGStateV1, candidate: str) -> bool:
        self.calls += 1
        return self.decision


def _coordinator(
    plan: SupervisorPlan,
    retrieval: Any,
    *,
    budget: AdaptiveRAGBudgetV1 | None = None,
    calculation: Any = None,
    generation: Any = None,
    validator: Any = None,
    allow_test_release: bool = False,
) -> BoundedTrustedV2Coordinator:
    provider = DeterministicFallbackProvider(
        {"What was revenue?": plan, "Compare years": plan},
    )
    return BoundedTrustedV2Coordinator(
        SupervisorService(provider),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            calculation=calculation,
            generation=generation,
            release_validator=validator,
        ),
        budget=budget,
        allow_test_release=allow_test_release,
    )


def _packet(
    evidence_id: str,
    metric: str = "Revenue",
    period: str = "FY2024",
    *,
    slots: list[str] | None = None,
    value: str = "100",
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "metric": metric,
        "value": value,
        "period": period,
        "entity": "Acme",
        "scope": "consolidated",
        "source": "fixture",
        "document_id": evidence_id,
        "slots": slots or [],
    }


def test_one_shot_plan_and_bounded_loop_do_not_release_without_downstream() -> None:
    retrieval = ScriptedRetrieval([[_packet("e1", slots=["revenue"])]])
    coordinator = _coordinator(_plan(_slot("revenue")), retrieval)
    outcome = asyncio.run(coordinator.execute(_request()))
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert outcome.release_status.value == "NOT_RELEASED"
    assert "DOWNSTREAM_EXECUTION_NOT_WIRED" in outcome.reason_codes
    trace = outcome.debug_metadata["trace"]
    assert [item["to"] for item in trace["transitions"]] == [
        "ACT",
        "OBSERVE",
        "EVALUATE",
        "READY_TO_GENERATE",
    ]
    assert trace["replan_count"] == 0
    assert trace["tool_call_count"] == 1


def test_missing_slot_drives_one_targeted_recovery() -> None:
    retrieval = ScriptedRetrieval([[], [_packet("e2", slots=["revenue"])]])
    coordinator = _coordinator(_plan(_slot("revenue")), retrieval)
    outcome = asyncio.run(coordinator.execute(_request()))
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert len(retrieval.actions) == 2
    assert retrieval.actions[0].reason_code is ReasonCode.MISSING_SLOT
    assert retrieval.actions[1].reason_code is ReasonCode.MISSING_SLOT
    assert retrieval.actions[1].target_slots == ("revenue",)
    assert outcome.debug_metadata["trace"]["replan_count"] == 1


def test_wrong_period_selects_structured_period_recovery() -> None:
    retrieval = ScriptedRetrieval([
        [_packet("wrong", period="FY2023", slots=["revenue"])],
        [_packet("right", period="FY2024", slots=["revenue"])],
    ])
    coordinator = _coordinator(_plan(_slot("revenue")), retrieval)
    outcome = asyncio.run(coordinator.execute(_request()))
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert [action.capability.value for action in retrieval.actions] == [
        "SEMANTIC_RETRIEVAL",
        "STRUCTURED_FINANCIAL_LOOKUP",
    ]
    assert retrieval.actions[1].reason_code is ReasonCode.WRONG_PERIOD


def test_missing_operand_selects_operand_recovery_and_never_calls_calculator() -> None:
    plan = _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("prior", period="FY2023", role="prior"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )
    retrieval = ScriptedRetrieval([
        [_packet("current", period="FY2024", slots=["current"])],
        [_packet("prior", period="FY2023", slots=["prior"])],
    ])
    calculation = FakeCalculation()
    coordinator = _coordinator(plan, retrieval, calculation=calculation)
    outcome = asyncio.run(coordinator.execute(_request("Compare years")))
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert retrieval.actions[1].reason_code is ReasonCode.MISSING_OPERAND
    assert retrieval.actions[1].capability.value == "STRUCTURED_FINANCIAL_LOOKUP"
    assert calculation.calls == 0


def test_no_progress_terminates_without_unbounded_loop() -> None:
    retrieval = ScriptedRetrieval([
        [_packet("same", metric="Cost", slots=[])],
        [_packet("same", metric="Cost", slots=[])],
    ])
    coordinator = _coordinator(_plan(_slot("revenue")), retrieval)
    outcome = asyncio.run(coordinator.execute(_request()))
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "NO_PROGRESS" in outcome.reason_codes
    assert outcome.debug_metadata["trace"]["no_progress_count"] == 1
    assert len(retrieval.actions) <= 2


def test_tool_budget_stops_before_extra_capability_call() -> None:
    retrieval = ScriptedRetrieval([[], [_packet("late", slots=["revenue"])]])
    coordinator = _coordinator(
        _plan(_slot("revenue")),
        retrieval,
        budget=AdaptiveRAGBudgetV1(
            max_replan_rounds=3,
            max_total_tool_calls=1,
            max_same_tool_retry=1,
        ),
    )
    outcome = asyncio.run(coordinator.execute(_request()))
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "BUDGET_EXHAUSTED" in outcome.reason_codes
    assert len(retrieval.actions) == 1


def test_replan_budget_stops_after_initial_action() -> None:
    retrieval = ScriptedRetrieval([[], [_packet("late", slots=["revenue"])]])
    coordinator = _coordinator(
        _plan(_slot("revenue")),
        retrieval,
        budget=AdaptiveRAGBudgetV1(
            max_replan_rounds=0,
            max_total_tool_calls=3,
            max_same_tool_retry=1,
        ),
    )
    outcome = asyncio.run(coordinator.execute(_request()))
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "BUDGET_EXHAUSTED" in outcome.reason_codes
    assert len(retrieval.actions) == 1


def test_capability_crash_is_execution_error_not_policy_refusal() -> None:
    coordinator = _coordinator(_plan(_slot("revenue")), RaisingRetrieval())
    outcome = asyncio.run(coordinator.execute(_request()))
    assert outcome.status is V2ExecutionStatus.EXECUTION_ERROR
    assert "CAPABILITY_EXCEPTION" in outcome.reason_codes
    assert "retrieval secret" not in str(outcome.to_dict())


def test_invalid_plan_has_zero_capability_calls() -> None:
    invalid = object.__new__(SupervisorPlan)
    object.__setattr__(invalid, "intent", Intent.DIRECT_FACT)
    object.__setattr__(
        invalid,
        "required_slots",
        (RequiredSlot("revenue", "Revenue", "FY2024", "made_up", "numeric", None),),
    )
    object.__setattr__(invalid, "operation", None)
    object.__setattr__(invalid, "next_action", Action.RETRIEVE)
    retrieval = ScriptedRetrieval([[_packet("never")]])
    coordinator = _coordinator(invalid, retrieval)
    outcome = asyncio.run(coordinator.execute(_request()))
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "INVALID_PLAN" in outcome.reason_codes
    assert retrieval.actions == []


def test_supervisor_is_called_once_and_plan_is_canonical() -> None:
    plan = _plan(_slot("revenue"))
    provider = DeterministicFallbackProvider({"What was revenue?": plan})
    supervisor = SupervisorService(provider)
    retrieval = ScriptedRetrieval([[_packet("e1", slots=["revenue"])]])
    coordinator = BoundedTrustedV2Coordinator(
        supervisor,
        capabilities=TrustedV2CapabilityPorts(retrieval=retrieval),
    )
    outcome = asyncio.run(coordinator.execute(_request()))
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert provider.last_call is not None
    trace = outcome.debug_metadata["trace"]
    assert trace["plan_id"] == outcome.plan_id


def test_explicit_fake_generation_and_validator_can_release_only_in_test_mode() -> None:
    retrieval = ScriptedRetrieval([[_packet("e1", slots=["revenue"])]])
    validator = FakeValidator(True)
    coordinator = _coordinator(
        _plan(_slot("revenue")),
        retrieval,
        generation=FakeGeneration(),
        validator=validator,
        allow_test_release=True,
    )
    outcome = asyncio.run(coordinator.execute(_request()))
    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert outcome.release_status.value == "RELEASED"
    assert outcome.answer == "released only by explicit test wiring"
    assert validator.calls == 1


def test_fake_generator_is_not_called_when_test_release_is_disabled() -> None:
    class CountingGeneration(FakeGeneration):
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, state: AdaptiveRAGStateV1) -> str:
            self.calls += 1
            return super().generate(state)

    retrieval = ScriptedRetrieval([[_packet("e1", slots=["revenue"])]])
    generation = CountingGeneration()
    coordinator = _coordinator(
        _plan(_slot("revenue")),
        retrieval,
        generation=generation,
        validator=FakeValidator(True),
    )
    outcome = asyncio.run(coordinator.execute(_request()))
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert generation.calls == 0


def test_conflict_is_policy_fail_closed() -> None:
    retrieval = ScriptedRetrieval([
        [
            _packet("a", value="100", slots=["revenue"]),
            _packet("b", value="200", slots=["revenue"]),
        ],
    ])
    coordinator = _coordinator(_plan(_slot("revenue")), retrieval)
    outcome = asyncio.run(coordinator.execute(_request()))
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "EVIDENCE_CONFLICT" in outcome.reason_codes


def test_trace_contains_structured_state_only() -> None:
    retrieval = ScriptedRetrieval([[]])
    coordinator = _coordinator(_plan(_slot("revenue")), retrieval)
    outcome = asyncio.run(coordinator.execute(_request()))
    trace_text = str(outcome.debug_metadata["trace"])
    assert "reasoning" not in trace_text.casefold()
    assert "chain of thought" not in trace_text.casefold()


def test_capability_port_container_is_injectable() -> None:
    ports = TrustedV2CapabilityPorts()
    assert ports.retrieval is None
    assert ports.calculation is None
