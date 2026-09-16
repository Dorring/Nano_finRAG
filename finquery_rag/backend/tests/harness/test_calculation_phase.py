"""NF-V3 H1-D3: deterministic calculation is a harness phase.

In ``legacy`` mode calculation runs after the loop, in the coordinator's
candidate stage.  In ``harness_v3`` it runs inside the loop, between EVALUATE
and READY_TO_GENERATE, so the run trace and the budget cover it.

Both modes must call the calculator exactly once.  A double calculation would
be worse than either placement.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rag_v2.adaptive import (
    AdaptivePhase,
    AdaptiveRAGBudgetV1,
    AdaptiveRAGStateV1,
    BoundedAdaptiveRAGV1,
    ReasonCode,
    ReplanActionV1,
    ToolCapability,
)
from rag_v2.contracts import Intent
from src.runtime import (
    DeterministicCalculationCapability,
    TrustedReleaseValidationCapability,
    TrustedV2CapabilityPorts,
    TrustedV2GenerationCapability,
    V2ExecutionStatus,
)
from src.runtime.harness_runtime_mode import AgentRuntimeMode
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from tests.test_trusted_v2_r4_binder import (
    SelectingBinderProvider,
    _fact,
    _plan,
    _real_capabilities,
    _request,
    _slot,
)

_FACTS = {
    "CURRENT": _fact("CURRENT", period="FY2024", slots=("current",), value="391"),
    "PRIOR": _fact("PRIOR", period="FY2023", slots=("prior",), value="383"),
}

_BUDGET = AdaptiveRAGBudgetV1(
    max_replan_rounds=3, max_total_tool_calls=4, max_same_tool_retry=3
)


def _calculation_plan() -> Any:
    return _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("prior", period="FY2023", role="prior"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )


def _calculation_capability() -> DeterministicCalculationCapability:
    """The real capability; it already counts its own invocations in ``calls``."""

    return DeterministicCalculationCapability()


def _execute(mode: AgentRuntimeMode | None, calculation: Any) -> Any:
    retrieval, binder, _, _, _ = _real_capabilities(
        [["CURRENT", "PRIOR"]], _FACTS, SelectingBinderProvider()
    )
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(DeterministicFallbackProvider({"Compare years": _calculation_plan()})),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=binder,
            calculation=calculation,
            generation=TrustedV2GenerationCapability(),
            release_validator=TrustedReleaseValidationCapability(),
        ),
        budget=_BUDGET,
        runtime_mode=mode,
    )
    return asyncio.run(coordinator.execute(_request("Compare years", "h1-d3")))


def _transitions(outcome: Any) -> list[str]:
    return [item["to"] for item in outcome.debug_metadata["trace"]["transitions"]]


def test_legacy_mode_calculates_outside_the_loop() -> None:
    calculation = _calculation_capability()
    outcome = _execute(AgentRuntimeMode.LEGACY, calculation)

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert calculation.calls == 1
    assert "CALCULATE" not in _transitions(outcome)
    assert "CALCULATE" not in outcome.debug_metadata["trace"]["action_trace"]


def test_harness_v3_calculates_inside_the_loop() -> None:
    calculation = _calculation_capability()
    outcome = _execute(AgentRuntimeMode.HARNESS_V3, calculation)

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    # The full closed loop: retrieval, the calculation phase, then the
    # generation/verification/release tail, all inside the harness.
    assert _transitions(outcome) == [
        "ACT",
        "OBSERVE",
        "EVALUATE",
        "CALCULATE",
        "READY_TO_GENERATE",
        "GENERATE",
        "VERIFY",
        "RELEASE",
    ]
    assert outcome.debug_metadata["trace"]["action_trace"] == [
        "SEMANTIC_RETRIEVAL",
        "CALCULATE",
        "GENERATE",
        "VERIFY",
    ]


def test_calculation_runs_exactly_once_in_either_mode() -> None:
    legacy = _calculation_capability()
    harness = _calculation_capability()

    _execute(AgentRuntimeMode.LEGACY, legacy)
    _execute(AgentRuntimeMode.HARNESS_V3, harness)

    assert legacy.calls == 1
    assert harness.calls == 1


def test_direct_fact_query_never_enters_calculate_in_harness_v3() -> None:
    """A plan without calculation requirements must skip the phase entirely."""

    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
    calculation = _calculation_capability()
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(DeterministicFallbackProvider({"What was revenue?": _plan(_slot("revenue"))})),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=binder,
            calculation=calculation,
            generation=TrustedV2GenerationCapability(),
            release_validator=TrustedReleaseValidationCapability(),
        ),
        budget=_BUDGET,
        runtime_mode=AgentRuntimeMode.HARNESS_V3,
    )
    outcome = asyncio.run(coordinator.execute(_request("What was revenue?", "h1-d3-fact")))

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert "CALCULATE" not in _transitions(outcome)
    assert calculation.calls == 0


# --- harness-level unit tests (no coordinator) -------------------------------


def _harness_state() -> AdaptiveRAGStateV1:
    state = AdaptiveRAGStateV1.new(
        "q1",
        "Compare revenue across years",
        required_slots=[{"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"}],
        calculation_requirements={"operation": "growth_rate", "operand_slots": ["revenue"]},
    )
    # The evidence evaluator admits evidence before the loop may calculate; a
    # bare state has no admitted evidence, so the CALCULATE phase is not entered.
    state.bound_evidence_ids = ["e1"]
    return state


def _revenue_packet() -> dict[str, Any]:
    return {
        "evidence_id": "e1",
        "metric": "Revenue",
        "value": "120",
        "period": "FY2024",
        "entity": "Acme",
        "scope": "consolidated",
        "source": "10-K",
        "document_id": "e1",
    }


class _Result:
    status = "EXECUTED"


def test_calculate_phase_runs_the_calculator_and_records_a_turn() -> None:
    state = _harness_state()
    calls: list[int] = []

    def calculator(current: AdaptiveRAGStateV1) -> Any:
        calls.append(1)
        return _Result()

    result = BoundedAdaptiveRAGV1().run(
        state,
        {ToolCapability.SEMANTIC_RETRIEVAL: lambda query, current: [_revenue_packet()]},
        initial_action=ReplanActionV1(
            ToolCapability.SEMANTIC_RETRIEVAL, state.normalized_query, ReasonCode.MISSING_SLOT
        ),
        calculator=calculator,
    )

    assert result.state.status == AdaptivePhase.READY_TO_GENERATE.value
    assert calls == [1]
    assert result.state.calculation_attempted is True
    assert result.state.action_trace == ["SEMANTIC_RETRIEVAL", "CALCULATE"]
    assert result.state.turns[1]["outcome"]["calculation_status"] == "EXECUTED"


def test_required_calculation_without_a_calculator_stays_with_the_caller() -> None:
    """No calculator means the harness is not the owner of calculation.

    This is the legacy contract: the run proceeds to READY_TO_GENERATE and the
    caller performs the calculation.  It is not a harness failure.
    """

    state = _harness_state()

    result = BoundedAdaptiveRAGV1().run(
        state,
        {ToolCapability.SEMANTIC_RETRIEVAL: lambda query, current: [_revenue_packet()]},
        initial_action=ReplanActionV1(
            ToolCapability.SEMANTIC_RETRIEVAL, state.normalized_query, ReasonCode.MISSING_SLOT
        ),
    )

    assert result.state.status == AdaptivePhase.READY_TO_GENERATE.value
    assert "CALCULATE" not in result.state.action_trace
    assert result.state.calculation_attempted is False


def test_calculator_exception_fails_closed_without_retry() -> None:
    state = _harness_state()
    attempts: list[int] = []

    def calculator(current: AdaptiveRAGStateV1) -> Any:
        attempts.append(1)
        raise RuntimeError("calculator secret")

    result = BoundedAdaptiveRAGV1().run(
        state,
        {ToolCapability.SEMANTIC_RETRIEVAL: lambda query, current: [_revenue_packet()]},
        initial_action=ReplanActionV1(
            ToolCapability.SEMANTIC_RETRIEVAL, state.normalized_query, ReasonCode.MISSING_SLOT
        ),
        calculator=calculator,
    )

    assert result.state.status == AdaptivePhase.FAIL_CLOSED.value
    assert result.state.stop_reason == ReasonCode.CALCULATION_ERROR.value
    assert attempts == [1]
    assert "calculator secret" not in str(result.state.last_observation)
