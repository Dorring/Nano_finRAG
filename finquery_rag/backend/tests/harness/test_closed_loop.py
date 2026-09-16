"""NF-V3 H1-D4: the loop closes.

``legacy`` stops the harness at READY_TO_GENERATE and produces the answer in the
coordinator's candidate stage.  ``harness_v3`` runs generation, verification and
release as harness phases, so ``AdaptivePhase.RELEASE`` is a live production
terminal.

The harness sequences; the deterministic validator still decides.  These tests
assert both the closed loop and that the verdict is unchanged between modes.
"""

from __future__ import annotations

import asyncio
import copy
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
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from src.domain.calculation import (
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
)
from src.runtime import (
    DeterministicCalculationCapability,
    TrustedReleaseValidationCapability,
    TrustedV2CapabilityPorts,
    TrustedV2GenerationCapability,
    V2ExecutionStatus,
)
from src.runtime.harness_runtime_mode import AgentRuntimeMode
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator
from tests.test_trusted_v2_r4_binder import (
    SelectingBinderProvider,
    _fact,
    _plan,
    _real_capabilities,
    _request,
    _slot,
)

_BUDGET = AdaptiveRAGBudgetV1(
    max_replan_rounds=3, max_total_tool_calls=4, max_same_tool_retry=3
)


def _execute(
    mode: AgentRuntimeMode,
    query: str,
    plan: Any,
    facts: dict[str, Any],
    keys: list[list[str]],
    *,
    binder_provider: Any = None,
    calculation: Any = None,
) -> Any:
    retrieval, binder, _, _, _ = _real_capabilities(keys, facts, binder_provider)
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(DeterministicFallbackProvider({query: plan})),
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
    return asyncio.run(coordinator.execute(_request(query, "h1-d4")))


def _transitions(outcome: Any) -> list[str]:
    return [item["to"] for item in outcome.debug_metadata["trace"]["transitions"]]


def _semantic_metadata(outcome: Any) -> dict[str, Any]:
    """Runtime metadata with timing removed.

    Every other field — validation id, status, claims, reports, provenance —
    must match exactly between modes.  ``latency_ms`` is a measurement, not a
    decision, so it is the only thing excluded.
    """

    metadata = copy.deepcopy(dict(outcome.runtime_metadata))
    validation = metadata.get("validation")
    if isinstance(validation, dict):
        validation.pop("latency_ms", None)
    return metadata


def test_harness_v3_releases_inside_the_loop() -> None:
    facts = {"E1": _fact("E1", value="100")}

    outcome = _execute(
        AgentRuntimeMode.HARNESS_V3, "What was revenue?", _plan(_slot("revenue")), facts, [["E1"]]
    )

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert outcome.release_status.value == "RELEASED"
    assert _transitions(outcome) == [
        "ACT",
        "OBSERVE",
        "EVALUATE",
        "READY_TO_GENERATE",
        "GENERATE",
        "VERIFY",
        "RELEASE",
    ]
    assert outcome.debug_metadata["trace"]["action_trace"] == [
        "SEMANTIC_RETRIEVAL",
        "GENERATE",
    ]


def test_legacy_never_enters_generate_or_release() -> None:
    facts = {"E1": _fact("E1", value="100")}

    outcome = _execute(
        AgentRuntimeMode.LEGACY, "What was revenue?", _plan(_slot("revenue")), facts, [["E1"]]
    )

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert not {"GENERATE", "VERIFY", "RELEASE"} & set(_transitions(outcome))


def test_both_modes_agree_on_release_decision_and_answer() -> None:
    """The whole point of the flag: same semantics, different execution model."""

    facts = {"E1": _fact("E1", value="100")}
    plan = _plan(_slot("revenue"))

    legacy = _execute(AgentRuntimeMode.LEGACY, "What was revenue?", plan, facts, [["E1"]])
    harness = _execute(AgentRuntimeMode.HARNESS_V3, "What was revenue?", plan, facts, [["E1"]])

    assert legacy.status is harness.status
    assert legacy.release_status == harness.release_status
    assert legacy.answer == harness.answer
    assert legacy.citation_ids == harness.citation_ids
    assert legacy.evidence_ids == harness.evidence_ids
    assert legacy.validator_status == harness.validator_status
    # No calculation on this plan, so the metadata must match exactly.
    assert _semantic_metadata(legacy) == _semantic_metadata(harness)


def test_both_modes_agree_on_a_calculation_plan() -> None:
    facts = {
        "CURRENT": _fact("CURRENT", period="FY2024", slots=("current",), value="391"),
        "PRIOR": _fact("PRIOR", period="FY2023", slots=("prior",), value="383"),
    }
    plan = _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("prior", period="FY2023", role="prior"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )

    def run(mode: AgentRuntimeMode) -> Any:
        return _execute(
            mode,
            "Compare years",
            plan,
            facts,
            [["CURRENT", "PRIOR"]],
            binder_provider=SelectingBinderProvider(),
            calculation=DeterministicCalculationCapability(),
        )

    legacy = run(AgentRuntimeMode.LEGACY)
    harness = run(AgentRuntimeMode.HARNESS_V3)

    assert legacy.status is harness.status
    assert legacy.release_status == harness.release_status
    assert legacy.answer == harness.answer
    assert legacy.calculation_ids == harness.calculation_ids
    assert "CALCULATE" in _transitions(harness)
    assert "CALCULATE" not in _transitions(legacy)

    # Metadata differs only by the marker recording where calculation ran.
    harness_metadata = _semantic_metadata(harness)
    assert harness_metadata.pop("calculation_in_harness", None) is True
    assert _semantic_metadata(legacy) == harness_metadata


class _BlockedCalculation:
    """A wired calculator that deterministically declines to compute."""

    candidate_mode = True
    last_calculation_id = None

    def __init__(self) -> None:
        self.calls = 0
        self.last_result = CalculationResult(
            status=CalculationStatus.BLOCKED,
            operation=CalculationOperation.GROWTH_RATE,
            error_code="OPERAND_MISSING",
        )

    def calculate(self, state: AdaptiveRAGStateV1) -> Any:
        self.calls += 1
        return self.last_result


def test_blocked_calculation_fails_closed_in_both_modes() -> None:
    """A blocked calculation must not reach generation in either mode.

    harness_v3 runs the calculator inside the loop; the candidate stage then
    re-reads rather than re-runs its result.  It must still *validate* that
    result, otherwise a blocked calculation would slip through to generation and
    lose the precise terminal reason the legacy path reports.
    """

    facts = {
        "CURRENT": _fact("CURRENT", period="FY2024", slots=("current",), value="391"),
        "PRIOR": _fact("PRIOR", period="FY2023", slots=("prior",), value="383"),
    }
    plan = _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("prior", period="FY2023", role="prior"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )

    outcomes = {}
    for mode in (AgentRuntimeMode.LEGACY, AgentRuntimeMode.HARNESS_V3):
        calculation = _BlockedCalculation()
        outcome = _execute(
            mode,
            "Compare years",
            plan,
            facts,
            [["CURRENT", "PRIOR"]],
            binder_provider=SelectingBinderProvider(),
            calculation=calculation,
        )
        assert calculation.calls == 1
        outcomes[mode] = outcome

    for mode, outcome in outcomes.items():
        assert outcome.status is not V2ExecutionStatus.READY_FOR_RELEASE, mode
        assert "CALCULATION_INVALID" in outcome.reason_codes, mode
        assert mode.value in ("legacy", "harness_v3")

    legacy = outcomes[AgentRuntimeMode.LEGACY]
    harness = outcomes[AgentRuntimeMode.HARNESS_V3]
    assert legacy.status is harness.status
    assert legacy.reason_codes == harness.reason_codes


# --- harness-level verification guards --------------------------------------

def _loop_state() -> AdaptiveRAGStateV1:
    return AdaptiveRAGStateV1.new(
        "q1",
        "What was revenue?",
        required_slots=[{"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"}],
    )


def _packet() -> dict[str, Any]:
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


def _run_loop(**kwargs: Any) -> Any:
    state = _loop_state()
    return BoundedAdaptiveRAGV1().run(
        state,
        {ToolCapability.SEMANTIC_RETRIEVAL: lambda query, current: [_packet()]},
        initial_action=ReplanActionV1(
            ToolCapability.SEMANTIC_RETRIEVAL, state.normalized_query, ReasonCode.MISSING_SLOT
        ),
        **kwargs,
    )


def test_generator_and_verifier_pass_reaches_release() -> None:
    result = _run_loop(generator=lambda state: "answer", verifier=lambda state, out: True)

    assert result.state.status == AdaptivePhase.RELEASE.value
    assert result.released is True
    assert result.state.action_trace == ["SEMANTIC_RETRIEVAL", "GENERATE"]


def test_a_generator_without_a_verifier_fails_closed() -> None:
    """Releasing on a missing verifier would let an unvalidated answer through."""

    result = _run_loop(generator=lambda state: "answer")

    assert result.state.status == AdaptivePhase.FAIL_CLOSED.value
    assert result.state.stop_reason == ReasonCode.VERIFICATION_NOT_WIRED.value
    assert result.released is False


def test_rejected_verification_does_not_release() -> None:
    result = _run_loop(generator=lambda state: "answer", verifier=lambda state, out: False)

    assert result.state.status == AdaptivePhase.FAIL_CLOSED.value
    assert result.released is False


def test_generator_exception_fails_closed_without_leaking_detail() -> None:
    def generator(state: AdaptiveRAGStateV1) -> str:
        raise RuntimeError("generator secret")

    result = _run_loop(generator=generator, verifier=lambda state, out: True)

    assert result.state.status == AdaptivePhase.FAIL_CLOSED.value
    assert result.state.stop_reason == ReasonCode.GENERATION_ERROR.value
    assert "generator secret" not in str(result.state.last_observation)


def test_verifier_exception_fails_closed() -> None:
    def verifier(state: AdaptiveRAGStateV1, out: Any) -> bool:
        raise RuntimeError("validator secret")

    result = _run_loop(generator=lambda state: "answer", verifier=verifier)

    assert result.state.status == AdaptivePhase.FAIL_CLOSED.value
    assert result.state.stop_reason == ReasonCode.VERIFICATION_ERROR.value
    assert "validator secret" not in str(result.state.last_observation)
