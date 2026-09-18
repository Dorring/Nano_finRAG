"""TV2-02 Supervisor and bounded coordinator tests."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

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
    alignment_override: Any = None,
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
        alignment_override=alignment_override,
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


# --- P1.2: the alignment override seam ---------------------------------------------------------
#
# The seam exists so a benchmark can measure the chain *after* the semantic
# alignment gate.  These tests exist because a seam that cannot be told apart
# from normal operation is worse than no seam: its results would be quotable as
# production behaviour.  Three things are asserted -- production never sets it,
# it can carry a run past the gate, and the run says so in its own outcome.


def _mismatching_plan_and_query() -> tuple[SupervisorPlan, str]:
    """A plan whose metric is not in the canonical vocabulary.

    The alignment gate reads an unrecognized plan metric as a mismatch, so this
    pair is rejected with ``QUERY_PLAN_SEMANTIC_MISMATCH`` -- asserted in the
    first test below rather than assumed, because if the gate ever stopped
    rejecting it the rest of this section would be testing nothing.
    """

    query = "What was revenue?"
    plan = _plan(_slot("m", metric="Deferred"))
    return plan, query


def _aligned_override(plan: SupervisorPlan, query: str):
    import dataclasses

    from rag_v2.supervisor import (
        SemanticAlignmentStatus,
        UnknownSemanticPolicy,
        align_query_to_plan,
    )

    computed = align_query_to_plan(
        query, plan, unknown_policy=UnknownSemanticPolicy.STRICT_DIRECT_FACT
    )
    assert computed.allowed is False, (
        "the fixture must actually be rejected, or this section proves nothing"
    )
    return dataclasses.replace(
        computed,
        status=SemanticAlignmentStatus.ALIGNED,
        mismatches=(),
        ambiguous_query_fields=(),
    )


def test_a_coordinator_is_built_without_an_alignment_override() -> None:
    """The production default, asserted rather than documented.

    ``_coordinator`` builds one the way every other call site does, so if the
    override were ever given a non-``None`` default this fails.
    """

    coordinator = _coordinator(_plan(_slot("revenue")), ScriptedRetrieval([[]]))

    assert coordinator.alignment_override is None


def test_an_alignment_override_is_refused_when_it_is_not_one() -> None:
    plan, _ = _mismatching_plan_and_query()

    with pytest.raises(TypeError):
        _coordinator(
            plan,
            ScriptedRetrieval([[]]),
            alignment_override=object(),
        )


def test_the_gate_rejects_the_fixture_without_an_override() -> None:
    """The control.  Without this, the next test would not mean anything."""

    plan, query = _mismatching_plan_and_query()
    coordinator = _coordinator(plan, ScriptedRetrieval([[]]))

    outcome = asyncio.run(coordinator.execute(_request(query)))

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "QUERY_PLAN_SEMANTIC_MISMATCH" in outcome.reason_codes


def test_an_alignment_override_carries_the_run_past_the_gate() -> None:
    """Same query, same plan, one injected verdict -- and a different ending.

    The run does not succeed; it fails later, for a reason that is not the gate.
    That is the whole claim: the override moves the decision, it does not
    manufacture an answer.
    """

    plan, query = _mismatching_plan_and_query()
    coordinator = _coordinator(
        plan,
        ScriptedRetrieval([[]]),
        alignment_override=_aligned_override(plan, query),
    )

    outcome = asyncio.run(coordinator.execute(_request(query)))

    assert "QUERY_PLAN_SEMANTIC_MISMATCH" not in outcome.reason_codes
    assert "QUERY_PLAN_SEMANTIC_AMBIGUOUS" not in outcome.reason_codes
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED


def test_an_overridden_run_records_both_verdicts() -> None:
    """A reader of the outcome alone can tell the gate did not allow this.

    ``semantic_alignment`` is the verdict that was acted on; the override record
    is the gate's real one.  Collapsing them would make a measurement
    indistinguishable from a claim.
    """

    plan, query = _mismatching_plan_and_query()
    coordinator = _coordinator(
        plan,
        ScriptedRetrieval([[]]),
        alignment_override=_aligned_override(plan, query),
    )

    outcome = asyncio.run(coordinator.execute(_request(query)))

    record = outcome.runtime_metadata.get("semantic_alignment_override")
    assert isinstance(record, dict), outcome.runtime_metadata
    assert record["computed_status"] == "MISMATCH"
    assert record["computed_allowed"] is False
    assert "unrecognized_plan_metric:Deferred" in record["computed_mismatches"]

    effective = outcome.runtime_metadata.get("semantic_alignment")
    assert isinstance(effective, dict)
    assert effective["status"] == "ALIGNED"


def test_a_run_without_an_override_records_no_override() -> None:
    """The absence is the evidence that the other tests' record is real."""

    plan, query = _mismatching_plan_and_query()
    coordinator = _coordinator(plan, ScriptedRetrieval([[]]))

    outcome = asyncio.run(coordinator.execute(_request(query)))

    assert "semantic_alignment_override" not in outcome.runtime_metadata
