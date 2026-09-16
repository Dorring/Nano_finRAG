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
from typing import Any

from rag_v2.adaptive import (
    AdaptivePhase,
    AdaptiveRAGStateV1,
    BoundedAdaptiveRAGV1,
    ReasonCode,
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
    TrustedV2CapabilityPorts,
    V2ExecutionStatus,
)
from src.runtime.harness_runtime_mode import AgentRuntimeMode
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator
from tests.harness.harness_support import (
    BUDGET,
    execute,
    loop_state,
    packet,
    run_loop,
    semantic_metadata,
    transitions,
)
from tests.test_trusted_v2_r4_binder import (
    SelectingBinderProvider,
    _fact,
    _plan,
    _real_capabilities,
    _request,
    _slot,
)


def test_harness_v3_releases_inside_the_loop() -> None:
    facts = {"E1": _fact("E1", value="100")}

    outcome = execute(
        AgentRuntimeMode.HARNESS_V3, "What was revenue?", _plan(_slot("revenue")), facts, [["E1"]]
    )

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert outcome.release_status.value == "RELEASED"
    assert transitions(outcome) == [
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
        "VERIFY",
    ]


def test_legacy_never_enters_generate_or_release() -> None:
    facts = {"E1": _fact("E1", value="100")}

    outcome = execute(
        AgentRuntimeMode.LEGACY, "What was revenue?", _plan(_slot("revenue")), facts, [["E1"]]
    )

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert not {"GENERATE", "VERIFY", "RELEASE"} & set(transitions(outcome))


def test_both_modes_agree_on_release_decision_and_answer() -> None:
    """The whole point of the flag: same semantics, different execution model."""

    facts = {"E1": _fact("E1", value="100")}
    plan = _plan(_slot("revenue"))

    legacy = execute(AgentRuntimeMode.LEGACY, "What was revenue?", plan, facts, [["E1"]])
    harness = execute(AgentRuntimeMode.HARNESS_V3, "What was revenue?", plan, facts, [["E1"]])

    assert legacy.status is harness.status
    assert legacy.release_status == harness.release_status
    assert legacy.answer == harness.answer
    assert legacy.citation_ids == harness.citation_ids
    assert legacy.evidence_ids == harness.evidence_ids
    assert legacy.validator_status == harness.validator_status
    assert legacy.calculations == harness.calculations
    assert legacy.calculation_result_id == harness.calculation_result_id
    assert legacy.claim_provenance == harness.claim_provenance
    # No calculation on this plan, so the metadata must match exactly.
    assert semantic_metadata(legacy) == semantic_metadata(harness)


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
        return execute(
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
    assert legacy.calculation_result_id == harness.calculation_result_id
    assert legacy.calculations == harness.calculations
    assert legacy.claim_provenance == harness.claim_provenance
    assert "CALCULATE" in transitions(harness)
    assert "CALCULATE" not in transitions(legacy)

    # Metadata differs only by the marker recording where calculation ran.
    harness_metadata = semantic_metadata(harness)
    assert harness_metadata.pop("calculation_in_harness", None) is True
    assert semantic_metadata(legacy) == harness_metadata


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
        outcome = execute(
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


def test_route_matches_between_modes() -> None:
    """The rebuilt harness_v3 outcome must keep the generation route.

    Without an explicit route the rebuild falls back to the plan intent, which
    contradicts the trace's own generation_route on a public field.
    """

    facts = {"E1": _fact("E1", value="100")}
    plan = _plan(_slot("revenue"))

    legacy = execute(AgentRuntimeMode.LEGACY, "What was revenue?", plan, facts, [["E1"]])
    harness = execute(AgentRuntimeMode.HARNESS_V3, "What was revenue?", plan, facts, [["E1"]])

    assert legacy.route == harness.route
    assert harness.route == harness.debug_metadata["trace"]["generation_route"]


def test_reason_codes_are_not_polluted_by_resolved_recovery_rounds() -> None:
    """A clean release must not carry the reason code of a round that succeeded.

    The trace deliberately folds every binder round's reason codes into its own
    list; the released outcome must not inherit that superset.
    """

    facts = {
        "WRONG": _fact("WRONG", period="FY2023", slots=("revenue",), value="90"),
        "RIGHT": _fact("RIGHT", period="FY2024", slots=("revenue",), value="100"),
    }
    plan = _plan(_slot("revenue"))

    def run(mode: AgentRuntimeMode) -> Any:
        return execute(
            mode,
            "What was revenue?",
            plan,
            facts,
            [["WRONG"], ["RIGHT"]],
            binder_provider=SelectingBinderProvider(),
        )

    legacy = run(AgentRuntimeMode.LEGACY)
    harness = run(AgentRuntimeMode.HARNESS_V3)

    assert legacy.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert harness.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert legacy.reason_codes == harness.reason_codes


def test_missing_operand_never_invokes_the_calculator_in_either_mode() -> None:
    """The calculator must stay behind evidence admission.

    Running it inside the loop must not let it run before the gate: a run whose
    operands were never admitted must not calculate at all.
    """

    facts = {"CURRENT": _fact("CURRENT", period="FY2024", slots=("current",), value="391")}
    plan = _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("prior", period="FY2023", role="prior"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )

    for mode in (AgentRuntimeMode.LEGACY, AgentRuntimeMode.HARNESS_V3):
        calculation = DeterministicCalculationCapability()
        outcome = execute(
            mode,
            "Compare years",
            plan,
            facts,
            [["CURRENT"], []],
            binder_provider=SelectingBinderProvider(),
            calculation=calculation,
        )
        assert calculation.calls == 0, mode
        assert outcome.status is not V2ExecutionStatus.READY_FOR_RELEASE, mode


class _RaisingCalculation:
    """A wired calculator that raises, like a contract violation would."""

    candidate_mode = True
    last_calculation_id = None
    last_result = None

    def calculate(self, state: AdaptiveRAGStateV1) -> Any:
        raise RuntimeError("calculator secret")


def test_calculator_exception_has_the_same_terminal_in_both_modes() -> None:
    """The ablation must not change the observable failure class.

    A raising calculator is an execution error, not a policy refusal; both modes
    must say so, and must not leak the exception text.
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
        outcome = execute(
            mode,
            "Compare years",
            plan,
            facts,
            [["CURRENT", "PRIOR"]],
            binder_provider=SelectingBinderProvider(),
            calculation=_RaisingCalculation(),
        )
        assert "calculator secret" not in str(outcome.to_dict()), mode
        outcomes[mode] = outcome

    legacy = outcomes[AgentRuntimeMode.LEGACY]
    harness = outcomes[AgentRuntimeMode.HARNESS_V3]
    assert legacy.status is harness.status
    assert legacy.reason_codes == harness.reason_codes


def test_rejected_validator_never_releases_on_the_test_release_path() -> None:
    """A verdict object is truthy regardless of its verdict.

    The test-release path wires the validator into the harness.  A validator
    that returns a result object rather than a bool must still be able to
    reject: reading truthiness instead of the verdict would release a candidate
    the validator just refused.
    """

    verdicts: list[str] = []

    class Verdict:
        passed = False
        status = "FAIL"
        reason_codes = ("BOUND_EVIDENCE_NOT_ADMITTED",)

    class RejectingValidation:
        def validate(self, state: AdaptiveRAGStateV1, candidate: Any) -> Any:
            verdicts.append("called")
            return Verdict()

    class StringGeneration:
        def generate(self, state: AdaptiveRAGStateV1) -> str:
            return "UNVALIDATED ANSWER"

    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(
            DeterministicFallbackProvider({"What was revenue?": _plan(_slot("revenue"))})
        ),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=binder,
            generation=StringGeneration(),
            release_validator=RejectingValidation(),
        ),
        budget=BUDGET,
        allow_test_release=True,
    )
    outcome = asyncio.run(coordinator.execute(_request("What was revenue?", "h1-verdict")))

    assert verdicts == ["called"], "the validator must actually be consulted"
    assert outcome.release_status.value != "RELEASED"
    assert outcome.answer is None


def test_unknown_status_fails_closed_instead_of_raising() -> None:
    """A resumed state may carry a phase this controller does not own."""

    state = loop_state()
    state.status = "EXECUTION_ERROR"
    result = BoundedAdaptiveRAGV1().run(
        state,
        {ToolCapability.SEMANTIC_RETRIEVAL: lambda query, current: [packet()]},
    )

    assert result.state.status == AdaptivePhase.FAIL_CLOSED.value
    assert result.state.stop_reason == ReasonCode.STRUCTURAL_NOT_READY.value


def test_malformed_tool_packet_fails_closed_like_a_raising_tool() -> None:
    """Normalization is part of the tool contract, not outside it.

    A packet the harness cannot normalize must be captured as a tool error and
    fail the run closed, rather than raising out of ``run()`` and bypassing the
    bounded-result contract.
    """

    result = BoundedAdaptiveRAGV1().run(
        loop_state(),
        {ToolCapability.SEMANTIC_RETRIEVAL: lambda query, current: [{"evidence_id": "e1", "slots": 5}]},
    )

    # The run terminated through the bounded contract, not by raising.
    assert result.state.status == AdaptivePhase.FAIL_CLOSED.value
    # The malformed packet was recorded as a tool error on its turn.
    assert result.state.turns[0]["action"] == "SEMANTIC_RETRIEVAL"
    assert result.state.turns[0]["outcome"]["error"] == "TypeError"
    assert result.state.turns[0]["outcome"]["packet_count"] == 0


def test_verification_is_recorded_as_a_turn() -> None:
    """The trace must show that verification ran, and whether it passed."""

    passed = run_loop(generator=lambda state: "answer", verifier=lambda state, out: True)
    assert passed.state.turns[-1]["action"] == "VERIFY"
    assert passed.state.turns[-1]["outcome"] == {"verification_passed": True}

    rejected = run_loop(generator=lambda state: "answer", verifier=lambda state, out: False)
    assert rejected.state.turns[-1]["action"] == "VERIFY"
    assert rejected.state.turns[-1]["outcome"] == {"verification_passed": False}


# --- harness-level verification guards --------------------------------------


def test_generator_and_verifier_pass_reaches_release() -> None:
    result = run_loop(generator=lambda state: "answer", verifier=lambda state, out: True)

    assert result.state.status == AdaptivePhase.RELEASE.value
    assert result.released is True
    assert result.state.action_trace == ["SEMANTIC_RETRIEVAL", "GENERATE", "VERIFY"]


def test_a_generator_without_a_verifier_fails_closed() -> None:
    """Releasing on a missing verifier would let an unvalidated answer through."""

    result = run_loop(generator=lambda state: "answer")

    assert result.state.status == AdaptivePhase.FAIL_CLOSED.value
    assert result.state.stop_reason == ReasonCode.VERIFICATION_NOT_WIRED.value
    assert result.released is False


def test_rejected_verification_does_not_release() -> None:
    result = run_loop(generator=lambda state: "answer", verifier=lambda state, out: False)

    assert result.state.status == AdaptivePhase.FAIL_CLOSED.value
    assert result.released is False


def test_generator_exception_fails_closed_without_leaking_detail() -> None:
    def generator(state: AdaptiveRAGStateV1) -> str:
        raise RuntimeError("generator secret")

    result = run_loop(generator=generator, verifier=lambda state, out: True)

    assert result.state.status == AdaptivePhase.FAIL_CLOSED.value
    assert result.state.stop_reason == ReasonCode.GENERATION_ERROR.value
    assert "generator secret" not in str(result.state.last_observation)


def test_verifier_exception_fails_closed() -> None:
    def verifier(state: AdaptiveRAGStateV1, out: Any) -> bool:
        raise RuntimeError("validator secret")

    result = run_loop(generator=lambda state: "answer", verifier=verifier)

    assert result.state.status == AdaptivePhase.FAIL_CLOSED.value
    assert result.state.stop_reason == ReasonCode.VERIFICATION_ERROR.value
    assert "validator secret" not in str(result.state.last_observation)
