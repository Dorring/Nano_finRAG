"""Characterization baseline for the ``legacy`` execution model.

These tests pin the trusted V2 runtime's behaviour as it stood before NF-V3 H1,
and they still describe the default runtime mode.  They exist so that "nothing
regressed" is a checkable claim rather than an assertion.

Plan of record: `docs/architecture/nf-v3-h1-agent-harness-plan.md`.

NF-V3 H1 added a second mode, ``harness_v3``, which closes the loop: calculation,
generation, verification and release all run as harness phases.  ``legacy`` keeps
the original split, and remains the production default.  The closed-loop
expectations live in the sibling test modules:

- ``test_calculation_phase.py`` — calculation as a harness phase
- ``test_closed_loop.py`` — generation, verification and release in the loop

Two facts this file still freezes, both specific to ``legacy``:

1. The harness loop stops at ``READY_TO_GENERATE``.  Deterministic calculation
   runs *outside* the loop, in the coordinator's candidate stage, so the
   transition trace never contains ``CALCULATE``.
2. ``generator`` / ``verifier`` are wired into the harness only when
   ``allow_test_release`` is true, so the harness's ``GENERATE`` / ``VERIFY`` /
   ``RELEASE`` phases are unreachable on this path.  The answer is still
   released, but by ``_candidate_stage`` after the loop returns.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rag_v2.adaptive import AdaptiveRAGBudgetV1
from rag_v2.contracts import Intent, SupervisorPlan
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from src.runtime import (
    DeterministicCalculationCapability,
    TrustedReleaseValidationCapability,
    TrustedV2CapabilityPorts,
    TrustedV2GenerationCapability,
    V2ExecutionStatus,
)
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator
from tests.test_trusted_v2_r4_binder import (
    SelectingBinderProvider,
    _fact,
    _plan,
    _real_capabilities,
    _request,
    _slot,
)

# The production trace contract, frozen.  Adding keys is a deliberate act that
# must update this set; removing or renaming one breaks evaluation consumers.
LEGACY_TRACE_KEYS = frozenset(
    {
        "binder_status_per_round",
        "bound_evidence_ids",
        "bound_slot_ids",
        "calculation_result_id",
        "calculator_invoked",
        "candidate_count_per_round",
        "candidate_generation_id",
        "candidate_ids_per_round",
        "candidate_ready",
        "claim_provenance",
        "conflict_ids",
        "context_trust_levels",
        "execution_id",
        "failed_checks",
        "final_candidate_id",
        "financial_fact_context_levels",
        "generation_route",
        "missing_operand_slots",
        "missing_slot_ids",
        "no_progress_count",
        "plan_id",
        "reason_codes",
        "release_decision",
        "release_status",
        "renderer_invoked",
        "repair_attempted",
        "repair_count",
        "repair_eligible",
        "replan_count",
        "request_id",
        "retrieval_rounds",
        "revalidated",
        "route_reason",
        "same_tool_retry_count",
        "semantic_alignment",
        "specialist_invoked",
        "targeted_slot_ids",
        "terminal_state",
        "tool_call_count",
        "tool_history",
        "transitions",
        "validation_id",
        "validation_passed",
        "validation_pending",
        "validation_reason_codes",
        "wrong_period_slots",
        # Added by NF-V3 H1 commit 2 (run turn trace).  Deliberate and additive.
        "turns",
        "turn_count",
        "action_trace",
    }
)

_BUDGET = AdaptiveRAGBudgetV1(
    max_replan_rounds=3,
    max_total_tool_calls=4,
    max_same_tool_retry=3,
)

_CALCULATION_FACTS = {
    "CURRENT": _fact("CURRENT", period="FY2024", slots=("current",), value="391"),
    "PRIOR": _fact("PRIOR", period="FY2023", slots=("prior",), value="383"),
}


def _calculation_plan() -> SupervisorPlan:
    return _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("prior", period="FY2023", role="prior"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )


def _run(
    query: str,
    plan: SupervisorPlan,
    retrieval: Any,
    binder: Any,
    *,
    request_id: str,
    calculation: Any = None,
    generation: Any = None,
    validator: Any = None,
) -> Any:
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(DeterministicFallbackProvider({query: plan})),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=binder,
            calculation=calculation,
            generation=generation,
            release_validator=validator,
        ),
        budget=_BUDGET,
    )
    return asyncio.run(
        coordinator.execute(_request(query, request_id))
    )


def _transitions(outcome: Any) -> list[str]:
    return [item["to"] for item in outcome.debug_metadata["trace"]["transitions"]]


def test_legacy_calculation_runs_outside_the_harness_loop() -> None:
    """CALCULATE is a coordinator-stage step, not a harness phase."""

    retrieval, binder, _, _, _ = _real_capabilities(
        [["CURRENT", "PRIOR"]], _CALCULATION_FACTS, SelectingBinderProvider()
    )
    calculation = DeterministicCalculationCapability()
    generation = TrustedV2GenerationCapability()

    outcome = _run(
        "Compare years",
        _calculation_plan(),
        retrieval,
        binder,
        request_id="h1-baseline-calc",
        calculation=calculation,
        generation=generation,
        validator=TrustedReleaseValidationCapability(),
    )

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert outcome.release_status.value == "RELEASED"
    assert calculation.calls == 1

    # The loop stops before calculation; the phase never appears in the trace.
    assert _transitions(outcome) == [
        "ACT",
        "OBSERVE",
        "EVALUATE",
        "READY_TO_GENERATE",
    ]
    assert "CALCULATE" not in _transitions(outcome)


def test_legacy_direct_fact_path_releases_without_calculation() -> None:
    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)

    outcome = _run(
        "What was revenue?",
        _plan(_slot("revenue")),
        retrieval,
        binder,
        request_id="h1-baseline-fact",
        generation=TrustedV2GenerationCapability(),
        validator=TrustedReleaseValidationCapability(),
    )

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert outcome.release_status.value == "RELEASED"
    assert _transitions(outcome) == [
        "ACT",
        "OBSERVE",
        "EVALUATE",
        "READY_TO_GENERATE",
    ]


def test_legacy_trace_key_contract_is_frozen() -> None:
    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)

    outcome = _run(
        "What was revenue?",
        _plan(_slot("revenue")),
        retrieval,
        binder,
        request_id="h1-baseline-keys",
        generation=TrustedV2GenerationCapability(),
        validator=TrustedReleaseValidationCapability(),
    )

    assert set(outcome.debug_metadata["trace"]) == LEGACY_TRACE_KEYS


def test_trace_exposes_the_run_turn_trace() -> None:
    """H1-D1: the run's decision trace is now first class.

    ``tool_call_count`` counts tool invocations; ``turns`` records each action
    the controller took, in order, with why it took it and what came back.
    """

    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)

    outcome = _run(
        "What was revenue?",
        _plan(_slot("revenue")),
        retrieval,
        binder,
        request_id="h1-turns",
        generation=TrustedV2GenerationCapability(),
        validator=TrustedReleaseValidationCapability(),
    )

    trace = outcome.debug_metadata["trace"]
    assert trace["turn_count"] == 1
    assert trace["action_trace"] == ["SEMANTIC_RETRIEVAL"]
    assert trace["tool_call_count"] == 1

    (turn,) = trace["turns"]
    assert turn["turn"] == 1
    assert turn["action"] == "SEMANTIC_RETRIEVAL"
    assert turn["reason_code"] == "MISSING_SLOT"
    assert turn["outcome"]["packet_count"] == 1


def test_turn_trace_records_each_recovery_action_in_order() -> None:
    """A repair round shows both actions, in the order they were taken."""

    facts = {
        "WRONG": _fact("WRONG", period="FY2023", slots=("revenue",), value="90"),
        "RIGHT": _fact("RIGHT", period="FY2024", slots=("revenue",), value="100"),
    }
    retrieval, binder, _, _, _ = _real_capabilities(
        [["WRONG"], ["RIGHT"]], facts, SelectingBinderProvider()
    )

    outcome = _run(
        "What was revenue?",
        _plan(_slot("revenue")),
        retrieval,
        binder,
        request_id="h1-turns-replan",
        generation=TrustedV2GenerationCapability(),
        validator=TrustedReleaseValidationCapability(),
    )

    trace = outcome.debug_metadata["trace"]
    assert trace["action_trace"] == ["SEMANTIC_RETRIEVAL", "STRUCTURED_FINANCIAL_LOOKUP"]
    assert [turn["turn"] for turn in trace["turns"]] == [1, 2]
    assert trace["turns"][1]["reason_code"] == "WRONG_PERIOD"


def test_legacy_harness_release_phase_is_unreachable_without_test_wiring() -> None:
    """The answer is released by the candidate stage, not by the harness.

    ``build_trusted_v2_runtime`` constructs the coordinator with
    ``allow_test_release=False`` (src/runtime/trusted_v2_factory.py:75), so the
    in-loop ``GENERATE`` / ``VERIFY`` / ``RELEASE`` branch never runs.  Release
    still happens, but *outside* the loop, in ``_candidate_stage``.  That split
    is the gap H1-D4 closes.
    """

    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)

    class CountingGeneration(TrustedV2GenerationCapability):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def generate(self, state: Any) -> Any:
            self.calls += 1
            return super().generate(state)

    generation = CountingGeneration()
    outcome = _run(
        "What was revenue?",
        _plan(_slot("revenue")),
        retrieval,
        binder,
        request_id="h1-baseline-nowire",
        generation=generation,
        validator=TrustedReleaseValidationCapability(),
    )

    # The harness never entered its own release phase...
    assert "RELEASE" not in _transitions(outcome)
    assert _transitions(outcome) == [
        "ACT",
        "OBSERVE",
        "EVALUATE",
        "READY_TO_GENERATE",
    ]

    # ...yet a release did happen, produced after the loop returned.
    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert outcome.release_status.value == "RELEASED"
    assert generation.calls == 1


def test_legacy_missing_downstream_is_named_not_silently_dropped() -> None:
    """With no generation port at all, the coordinator names the gap.

    ``DOWNSTREAM_EXECUTION_NOT_WIRED`` is the coordinator's own admission that
    answer production is not part of the harness loop.
    """

    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)

    outcome = _run(
        "What was revenue?",
        _plan(_slot("revenue")),
        retrieval,
        binder,
        request_id="h1-baseline-nodownstream",
    )

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "DOWNSTREAM_EXECUTION_NOT_WIRED" in outcome.reason_codes


def test_legacy_coordinator_defaults_to_no_test_release() -> None:
    """Guard the flag the production factory relies on."""

    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(DeterministicFallbackProvider({})),
        capabilities=TrustedV2CapabilityPorts(),
    )
    assert coordinator.allow_test_release is False


def test_legacy_request_metadata_carries_candidate_mode_ports() -> None:
    """The candidate stage is the production answer path.

    Both real generation and validation capabilities declare ``candidate_mode``,
    which is what makes ``_candidate_generation_enabled()`` true in production.
    """

    assert getattr(TrustedV2GenerationCapability(), "candidate_mode", False) is True
    assert getattr(TrustedReleaseValidationCapability(), "candidate_mode", False) is True
    assert getattr(DeterministicCalculationCapability(), "candidate_mode", False) is True
