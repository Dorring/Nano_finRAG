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
from dataclasses import replace
from typing import Any

from rag_v2.adaptive import (
    AdaptivePhase,
    AdaptiveRAGStateV1,
    BoundedAdaptiveRAGV1,
    ReasonCode,
    ToolCapability,
)
from rag_v2.contracts import Intent
from src.domain.calculation import (
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
)
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from src.runtime import (
    DeterministicCalculationCapability,
    TrustedReleaseValidationCapability,
    TrustedV2CapabilityPorts,
    TrustedV2GenerationCapability,
    V2ExecutionStatus,
)
from src.runtime.harness_runtime_mode import AgentRuntimeMode
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator
from tests.harness.equivalence import (
    DECISION_BEARING_FIELDS,
    assert_decision_equivalent,
    decision_differences,
)
from tests.harness.harness_support import (
    BUDGET,
    BlockedCalculation,
    RaisingCalculation,
    execute,
    loop_state,
    packet,
    run_loop,
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
    """The whole point of the flag: same semantics, different execution model.

    Compared through the shared contract, not a list of fields chosen here.  A
    hand-picked list is exactly how ``route`` and ``reason_codes`` diverged
    unnoticed the first time round.
    """

    facts = {"E1": _fact("E1", value="100")}
    plan = _plan(_slot("revenue"))

    legacy = execute(AgentRuntimeMode.LEGACY, "What was revenue?", plan, facts, [["E1"]])
    harness = execute(AgentRuntimeMode.HARNESS_V3, "What was revenue?", plan, facts, [["E1"]])

    assert_decision_equivalent(legacy, harness)
    assert legacy.status is V2ExecutionStatus.READY_FOR_RELEASE


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

    assert_decision_equivalent(legacy, harness)
    assert "CALCULATE" in transitions(harness)
    assert "CALCULATE" not in transitions(legacy)

    # The one field the contract strips for harness_v3 is the marker recording
    # where calculation ran.  Stripping it is only safe while it is actually
    # set, so assert it directly rather than letting the contract excuse it.
    assert harness.runtime_metadata.get("calculation_in_harness") is True
    assert "calculation_in_harness" not in legacy.runtime_metadata


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
        calculation = BlockedCalculation()
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

    assert set(outcomes) == {AgentRuntimeMode.LEGACY, AgentRuntimeMode.HARNESS_V3}
    for mode, outcome in outcomes.items():
        assert outcome.status is not V2ExecutionStatus.READY_FOR_RELEASE, mode
        assert "CALCULATION_INVALID" in outcome.reason_codes, mode

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
    assert_decision_equivalent(legacy, harness)

    # The point of the fixture: a recovery round *did* run, so the trace carries
    # the reason code it resolved.  Without that the test would pass vacuously.
    assert ReasonCode.WRONG_PERIOD.value in harness.debug_metadata["trace"]["reason_codes"]
    assert ReasonCode.WRONG_PERIOD.value not in harness.reason_codes


def test_the_equivalence_contract_covers_the_fields_that_once_diverged() -> None:
    """Guards the contract itself, not the runtime.

    ``route`` and ``reason_codes`` are the two fields the first differential
    assertion set omitted; ``calculations``, ``calculation_result_id`` and
    ``claim_provenance`` are the three the second pass had to add.  Every one of
    them diverged at some point while the old hand-picked lists stayed green, so
    their presence in the contract is the regression test for the process.
    """

    for field in (
        "route",
        "reason_codes",
        "calculations",
        "calculation_result_id",
        "claim_provenance",
    ):
        assert field in DECISION_BEARING_FIELDS, field


def test_the_equivalence_contract_reports_every_field_not_just_the_first() -> None:
    """A one-line assertion would hide the other divergences in the same run."""

    facts = {"E1": _fact("E1", value="100")}
    plan = _plan(_slot("revenue"))
    legacy = execute(AgentRuntimeMode.LEGACY, "What was revenue?", plan, facts, [["E1"]])
    harness = execute(AgentRuntimeMode.HARNESS_V3, "What was revenue?", plan, facts, [["E1"]])
    assert_decision_equivalent(legacy, harness)

    divergent = replace(harness, route="ABSTAIN", answer="a different answer")

    assert decision_differences(legacy, divergent) == [
        {
            "field": "answer",
            "legacy": legacy.answer,
            "harness_v3": "a different answer",
        },
        {"field": "route", "legacy": legacy.route, "harness_v3": "ABSTAIN"},
    ]


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
            calculation=RaisingCalculation(),
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


def test_the_test_release_flag_does_not_split_the_two_modes() -> None:
    """`allow_test_release` with candidate-mode ports used to diverge.

    The flag wires the *string*-returning generator the pre-closure loop was
    built for, and the post-loop release branch rejects anything that is not a
    string.  With candidate-mode ports -- which return a CandidateExecutionResult
    -- legacy therefore ran the raw generator, reached RELEASE, and then failed
    its own contract check, while harness_v3's finalizer ignored the flag and
    released.  The two modes disagreed on nine decision-bearing fields.

    No existing test covered the combination: the flag's own tests use
    generators without `candidate_mode`, so this wiring never engaged.
    """

    facts = {"E1": _fact("E1", value="100")}
    plan = _plan(_slot("revenue"))

    def run(mode: AgentRuntimeMode) -> Any:
        retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
        coordinator = BoundedTrustedV2Coordinator(
            SupervisorService(
                DeterministicFallbackProvider({"What was revenue?": plan})
            ),
            capabilities=TrustedV2CapabilityPorts(
                retrieval=retrieval,
                evidence_evaluator=binder,
                generation=TrustedV2GenerationCapability(),
                release_validator=TrustedReleaseValidationCapability(),
            ),
            budget=BUDGET,
            runtime_mode=mode,
            allow_test_release=True,
        )
        return asyncio.run(coordinator.execute(_request("What was revenue?", "h1-flag")))

    legacy = run(AgentRuntimeMode.LEGACY)
    harness = run(AgentRuntimeMode.HARNESS_V3)

    assert_decision_equivalent(legacy, harness)
    assert legacy.status is V2ExecutionStatus.READY_FOR_RELEASE


def test_the_ablation_does_not_invoke_the_calculator_where_legacy_does_not() -> None:
    """CALCULATE is a harness phase only when the harness owns the downstream.

    With a calculation port that is not candidate-mode and no candidate-mode
    generator, legacy returns at READY_TO_GENERATE and never reaches the
    candidate stage's ``calculate()``.  Running CALCULATE anyway made the
    ablation invoke the calculator where the baseline does not -- contract-blind,
    because the decision surface is identical, and visible only in the port's own
    call count and the state it mutates.
    """

    class PlainCalculation:
        """A calculator port without ``candidate_mode``."""

        last_calculation_id = None

        def __init__(self) -> None:
            self.calls = 0

        def calculate(self, state: Any) -> Any:
            self.calls += 1
            return _calculation_result()

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

    calls = {}
    for mode in (AgentRuntimeMode.LEGACY, AgentRuntimeMode.HARNESS_V3):
        calculation = PlainCalculation()
        retrieval, binder, _, _, _ = _real_capabilities([["CURRENT", "PRIOR"]], facts)
        coordinator = BoundedTrustedV2Coordinator(
            SupervisorService(DeterministicFallbackProvider({"Compare years": plan})),
            capabilities=TrustedV2CapabilityPorts(
                retrieval=retrieval,
                evidence_evaluator=binder,
                calculation=calculation,
            ),
            budget=BUDGET,
            runtime_mode=mode,
        )
        asyncio.run(coordinator.execute(_request("Compare years", "h1-calc-gate")))
        calls[mode] = calculation.calls

    assert calls[AgentRuntimeMode.LEGACY] == calls[AgentRuntimeMode.HARNESS_V3]


def _calculation_result() -> Any:
    return CalculationResult(
        status=CalculationStatus.EXECUTED,
        operation=CalculationOperation.GROWTH_RATE,
    )
