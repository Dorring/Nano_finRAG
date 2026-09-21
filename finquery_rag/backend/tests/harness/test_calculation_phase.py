"""NF-V3 H1-D3: deterministic calculation is a harness phase.

In ``legacy`` mode calculation runs after the loop, in the coordinator's
candidate stage.  In ``harness_v3`` it runs inside the loop, between EVALUATE
and READY_TO_GENERATE, so the run trace and the budget cover it.

Both modes must call the calculator exactly once.  A double calculation would be
worse than either placement.

Fixtures are shared with the closed-loop suite; see ``harness_support``.
"""

from __future__ import annotations

from typing import Any

from rag_v2.adaptive import AdaptivePhase, AdaptiveRAGStateV1, ReasonCode
from src.runtime import DeterministicCalculationCapability, V2ExecutionStatus
from src.runtime.harness_runtime_mode import AgentRuntimeMode
from tests.harness.harness_support import (
    CALCULATION_FACTS,
    REVENUE_FACTS,
    calculation_loop_state,
    calculation_plan,
    execute,
    run_loop,
    transitions,
)
from tests.test_trusted_v2_r4_binder import _plan, _slot


def test_legacy_mode_calculates_outside_the_loop() -> None:
    calculation = DeterministicCalculationCapability()
    outcome = execute(
        AgentRuntimeMode.LEGACY,
        "Compare years",
        calculation_plan(),
        CALCULATION_FACTS,
        [["CURRENT", "PRIOR"]],
        calculation=calculation,
    )

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert calculation.calls == 1
    assert "CALCULATE" not in transitions(outcome)
    assert "CALCULATE" not in outcome.debug_metadata["trace"]["action_trace"]


def test_harness_v3_calculates_inside_the_loop() -> None:
    calculation = DeterministicCalculationCapability()
    outcome = execute(
        AgentRuntimeMode.HARNESS_V3,
        "Compare years",
        calculation_plan(),
        CALCULATION_FACTS,
        [["CURRENT", "PRIOR"]],
        calculation=calculation,
    )

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    # The full closed loop: retrieval, the calculation phase, then the
    # generation/verification/release tail, all inside the harness.
    assert transitions(outcome) == [
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
    legacy = DeterministicCalculationCapability()
    harness = DeterministicCalculationCapability()

    execute(
        AgentRuntimeMode.LEGACY, "Compare years", calculation_plan(),
        CALCULATION_FACTS, [["CURRENT", "PRIOR"]], calculation=legacy,
    )
    execute(
        AgentRuntimeMode.HARNESS_V3, "Compare years", calculation_plan(),
        CALCULATION_FACTS, [["CURRENT", "PRIOR"]], calculation=harness,
    )

    assert legacy.calls == 1
    assert harness.calls == 1


def test_direct_fact_query_never_enters_calculate_in_harness_v3() -> None:
    """A plan without calculation requirements must skip the phase entirely."""

    calculation = DeterministicCalculationCapability()
    outcome = execute(
        AgentRuntimeMode.HARNESS_V3,
        "What was revenue?",
        _plan(_slot("revenue")),
        REVENUE_FACTS,
        [["E1"]],
        calculation=calculation,
    )

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert "CALCULATE" not in transitions(outcome)
    assert calculation.calls == 0


# --- harness-level unit tests (no coordinator) -------------------------------


class _ExecutedResult:
    status = "EXECUTED"


def test_calculate_phase_runs_the_calculator_and_records_a_turn() -> None:
    calls: list[int] = []

    def calculator(current: AdaptiveRAGStateV1) -> Any:
        calls.append(1)
        return _ExecutedResult()

    result = run_loop(calculation_loop_state(), calculator=calculator)

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

    result = run_loop(calculation_loop_state())

    assert result.state.status == AdaptivePhase.READY_TO_GENERATE.value
    assert "CALCULATE" not in result.state.action_trace
    assert result.state.calculation_attempted is False


def test_required_calculation_without_admitted_evidence_never_calculates() -> None:
    """Calculation sits behind evidence admission, not merely behind the plan."""

    state = calculation_loop_state()
    state.bound_evidence_ids = []
    calls: list[int] = []

    def calculator(current: AdaptiveRAGStateV1) -> Any:
        calls.append(1)
        return _ExecutedResult()

    result = run_loop(state, calculator=calculator)

    assert calls == []
    assert result.state.calculation_attempted is False


def test_calculator_exception_fails_closed_without_retry() -> None:
    attempts: list[int] = []

    def calculator(current: AdaptiveRAGStateV1) -> Any:
        attempts.append(1)
        raise RuntimeError("calculator secret")

    result = run_loop(calculation_loop_state(), calculator=calculator)

    assert result.state.status == AdaptivePhase.FAIL_CLOSED.value
    assert result.state.stop_reason == ReasonCode.CALCULATION_ERROR.value
    assert attempts == [1]
    assert "calculator secret" not in str(result.state.last_observation)
