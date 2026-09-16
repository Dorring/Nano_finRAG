"""NF-V3 H1.1: the sealed fixture set and its runner.

``harness_v3`` is an ablation of an existing runtime, so the claim it has to
earn is not "it works" but "it decides the same thing".  A unit test can only
show that for a coordinator assembled in the test.  This module drives the
*real* construction path -- ``build_trusted_v2_runtime``, the same factory the
production builder uses, over the real R4 retriever, the real Semantic Binder,
the real deterministic calculator, the real generator routing and the real
release validator -- over a frozen set of fixtures, and compares the two modes
through the shared equivalence contract.

The fixture set is sealed: :func:`sealed_digest` hashes every fixture's
specification, and the integration report records it, so a run can be tied to
the exact inputs that produced it.

One fixture is deliberately not built through the factory.  The factory refuses
an incomplete dependency graph, and that refusal is correct -- which means the
harness's ``UNSUPPORTED_TOOL_ROUTE`` guard is only reachable from a manually
constructed coordinator.  The fixture says so rather than hiding it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from rag_v2.adaptive import AdaptiveRAGBudgetV1
from rag_v2.contracts import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.evidence.binder_service import SemanticBinderService
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from src.domain.calculation import (
    CalculationOperation,
    CalculationResult,
    CalculationStatus,
)
from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever
from src.runtime import (
    CandidateExecutionResult,
    DeterministicCalculationCapability,
    FinancialQueryRequest,
    RuntimeStatus,
    TrustedReleaseValidationCapability,
    TrustedV2CapabilityPorts,
    TrustedV2GenerationCapability,
    V2ExecutionOutcome,
    V2ExecutionRequest,
)
from src.runtime.harness_runtime_mode import AgentRuntimeMode
from src.runtime.trusted_v2_binder import SemanticEvidenceEvaluationCapability
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator
from src.runtime.trusted_v2_factory import build_trusted_v2_runtime
from src.runtime.trusted_v2_r4 import R4RetrievalCapability
from tests.harness.equivalence import decision_differences
from tests.test_trusted_v2_r4_binder import (
    ScriptedCandidateDirectPolicy,
    ScriptedIndexReader,
    SelectingBinderProvider,
    _fact,
    _request,
)

__all__ = [
    "DEFAULT_BUDGET",
    "FIXTURES",
    "H1Fixture",
    "fixture_report",
    "run_all",
    "run_fixture",
    "sealed_digest",
]

DEFAULT_BUDGET = AdaptiveRAGBudgetV1(
    max_replan_rounds=3, max_total_tool_calls=4, max_same_tool_retry=3
)
TIGHT_BUDGET = AdaptiveRAGBudgetV1(
    max_replan_rounds=0, max_total_tool_calls=1, max_same_tool_retry=0
)

CURRENT_FACT = _fact("CURRENT", period="FY2024", slots=("current",), value="391")
PRIOR_FACT = _fact("PRIOR", period="FY2023", slots=("prior",), value="383")
REVENUE_FACT = _fact("REVENUE", slots=("revenue",), value="391")
WRONG_PERIOD_FACT = _fact("WRONG", period="FY2023", slots=("revenue",), value="90")


@dataclass(frozen=True)
class H1Fixture:
    """One frozen input for the ablation comparison."""

    fixture_id: str
    description: str
    query: str
    slots: tuple[dict[str, Any], ...]
    facts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    retrieval_batches: tuple[tuple[str, ...], ...] = ()
    intent: str = Intent.DIRECT_FACT.value
    operation: str | None = None
    calculation: str = "none"
    generation: str = "trusted"
    retrieval: bool = True
    budget: Mapping[str, int] | None = None
    expect_released: bool = False
    expect_status: str = "FAIL_CLOSED"
    expect_reasons: tuple[str, ...] = ()
    #: Phases the harness run must pass through, in order.  This is the
    #: end-to-end trace claim: a fact query must be seen to reach RELEASE, and a
    #: calculation query must be seen to reach CALCULATE before it.
    expect_transitions: tuple[str, ...] = ()

    def _plan(self) -> SupervisorPlan:
        return SupervisorPlan(
            Intent(self.intent),
            tuple(
                RequiredSlot(
                    slot["slot_id"],
                    slot.get("metric", "Revenue"),
                    slot.get("period", "FY2024"),
                    slot.get("role", "value"),
                    "numeric",
                    None,
                )
                for slot in self.slots
            ),
            self.operation,
            Action.RETRIEVE,
        )

    def spec(self) -> dict[str, Any]:
        """The fixture as plain JSON, for hashing and for the report."""

        return {
            "fixture_id": self.fixture_id,
            "description": self.description,
            "query": self.query,
            "slots": [dict(slot) for slot in self.slots],
            "facts": {key: dict(value) for key, value in sorted(self.facts.items())},
            "retrieval_batches": [list(batch) for batch in self.retrieval_batches],
            "intent": self.intent,
            "operation": self.operation,
            "calculation": self.calculation,
            "generation": self.generation,
            "retrieval": self.retrieval,
            "budget": dict(self.budget) if self.budget else None,
        }


def _bill(slot_id: str, **overrides: Any) -> dict[str, Any]:
    return {"slot_id": slot_id, "metric": "Revenue", "period": "FY2024", **overrides}


def _growth_rate_slots() -> tuple[dict[str, Any], ...]:
    return (
        _bill("current", period="FY2024", role="current"),
        _bill("prior", period="FY2023", role="prior"),
    )


FIXTURES: tuple[H1Fixture, ...] = (
    H1Fixture(
        fixture_id="fact_direct",
        description="One admitted fact releases a direct-fact answer.",
        query="What was revenue?",
        slots=(_bill("revenue"),),
        facts={"REVENUE": REVENUE_FACT},
        retrieval_batches=(("REVENUE",),),
        expect_released=True,
        expect_status="READY_FOR_RELEASE",
        expect_transitions=(
            "READY_TO_GENERATE",
            "GENERATE",
            "VERIFY",
            "RELEASE",
        ),
    ),
    H1Fixture(
        fixture_id="calculation_growth_rate",
        description=(
            "A calculation plan runs the deterministic calculator as a loop "
            "phase and releases the rendered result."
        ),
        query="Compare revenue across years",
        slots=_growth_rate_slots(),
        facts={"CURRENT": CURRENT_FACT, "PRIOR": PRIOR_FACT},
        retrieval_batches=(("CURRENT", "PRIOR"),),
        intent=Intent.CALCULATION.value,
        operation="growth_rate",
        calculation="deterministic",
        expect_released=True,
        expect_status="READY_FOR_RELEASE",
        expect_transitions=(
            "CALCULATE",
            "READY_TO_GENERATE",
            "GENERATE",
            "VERIFY",
            "RELEASE",
        ),
    ),
    H1Fixture(
        fixture_id="calculation_blocked",
        description=(
            "The calculator declines to compute.  The BLOCKED result must fail "
            "closed and must not reach generation."
        ),
        query="Compare revenue across years",
        slots=_growth_rate_slots(),
        facts={"CURRENT": CURRENT_FACT, "PRIOR": PRIOR_FACT},
        retrieval_batches=(("CURRENT", "PRIOR"),),
        intent=Intent.CALCULATION.value,
        operation="growth_rate",
        calculation="blocked",
        expect_released=False,
        expect_status="FAIL_CLOSED",
        expect_reasons=("CALCULATION_INVALID",),
    ),
    H1Fixture(
        fixture_id="calculation_error",
        description=(
            "The calculator raises a contract violation.  Both modes must "
            "report the same failure class and leak no exception text."
        ),
        query="Compare revenue across years",
        slots=_growth_rate_slots(),
        facts={"CURRENT": CURRENT_FACT, "PRIOR": PRIOR_FACT},
        retrieval_batches=(("CURRENT", "PRIOR"),),
        intent=Intent.CALCULATION.value,
        operation="growth_rate",
        calculation="raising",
        expect_released=False,
        expect_status="EXECUTION_ERROR",
        expect_reasons=("CALCULATOR_EXCEPTION",),
    ),
    H1Fixture(
        fixture_id="wrong_period_recovery",
        description=(
            "The first retrieval round admits the wrong period; the replanner "
            "recovers and the second round releases.  The resolved round's "
            "reason code must not survive into the released outcome."
        ),
        query="What was revenue?",
        slots=(_bill("revenue"),),
        facts={"WRONG": WRONG_PERIOD_FACT, "REVENUE": REVENUE_FACT},
        retrieval_batches=(("WRONG",), ("REVENUE",)),
        expect_released=True,
        expect_status="READY_FOR_RELEASE",
    ),
    H1Fixture(
        fixture_id="missing_evidence",
        description=(
            "Retrieval returns nothing, so no evidence is ever admitted.  The "
            "run must fail closed rather than generate from an empty binder."
        ),
        query="What was revenue?",
        slots=(_bill("revenue"),),
        facts={"REVENUE": REVENUE_FACT},
        retrieval_batches=((),),
        expect_released=False,
        expect_status="FAIL_CLOSED",
    ),
    H1Fixture(
        fixture_id="validator_rejection",
        description=(
            "The generator cites evidence outside Binder admission, so the "
            "*real* release validator rejects it.  A rejected candidate must "
            "not be released in either mode."
        ),
        query="What was revenue?",
        slots=(_bill("revenue"),),
        facts={"REVENUE": REVENUE_FACT},
        retrieval_batches=(("REVENUE",),),
        generation="foreign_citation",
        expect_released=False,
        expect_status="FAIL_CLOSED",
        expect_reasons=("UNBOUND_CITATION_METADATA",),
    ),
    H1Fixture(
        fixture_id="budget_exhaustion",
        description=(
            "Retrieval keeps returning nothing under a zero-replan budget, so "
            "the replanner is refused and the run must fail closed on budget."
        ),
        query="What was revenue?",
        slots=(_bill("revenue"),),
        facts={"REVENUE": REVENUE_FACT},
        retrieval_batches=((),),
        budget=TIGHT_BUDGET.to_dict(),
        expect_released=False,
        expect_status="FAIL_CLOSED",
        expect_reasons=("BUDGET_EXHAUSTED",),
    ),
    H1Fixture(
        fixture_id="unsupported_route",
        description=(
            "No retrieval port is wired, so the initial action has no tool.  "
            "Reachable only from a manually constructed coordinator: the "
            "factory refuses an incomplete graph, which is the point of the "
            "guard being there as well."
        ),
        query="What was revenue?",
        slots=(_bill("revenue"),),
        retrieval=False,
        expect_released=False,
        expect_status="FAIL_CLOSED",
        expect_reasons=("UNSUPPORTED_TOOL_ROUTE",),
    ),
)


def sealed_digest(fixtures: tuple[H1Fixture, ...] = FIXTURES) -> str:
    """A content hash over the whole frozen fixture set."""

    payload = json.dumps(
        [fixture.spec() for fixture in fixtures],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- capability construction -------------------------------------------------


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

    def calculate(self, state: Any) -> Any:
        self.calls += 1
        return self.last_result


class _RaisingCalculation:
    """A wired calculator that violates its contract."""

    candidate_mode = True
    last_calculation_id = None
    last_result = None

    def __init__(self) -> None:
        self.calls = 0

    def calculate(self, state: Any) -> Any:
        self.calls += 1
        raise RuntimeError("calculator secret")


class _ForeignCitationGeneration:
    """A generator whose candidate cites evidence the Binder never admitted.

    The *validator* stays real and unmodified.  Stubbing the judge would prove
    only that a stub can reject; stubbing the input proves the real gate does.
    """

    candidate_mode = True

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, state: Any) -> CandidateExecutionResult:
        self.calls += 1
        return CandidateExecutionResult(
            candidate_answer="Revenue was 391.",
            route="STRUCTURED_SINGLE",
            route_reason="fixture_foreign_citation",
            bound_evidence_ids=tuple(getattr(state, "bound_evidence_ids", ())),
            citation_ids=("citation-NOT-ADMITTED",),
        )


def _build_capabilities(fixture: H1Fixture) -> TrustedV2CapabilityPorts:
    facts = dict(fixture.facts)
    reader = ScriptedIndexReader([list(batch) for batch in fixture.retrieval_batches])
    retriever = CandidateDirectRetriever(reader, lane_k=10)
    policy = ScriptedCandidateDirectPolicy(
        retriever, materializer=lambda key: facts[key]
    )
    retrieval = R4RetrievalCapability(policy) if fixture.retrieval else None

    if fixture.calculation == "deterministic":
        calculation: Any = DeterministicCalculationCapability()
    elif fixture.calculation == "blocked":
        calculation = _BlockedCalculation()
    elif fixture.calculation == "raising":
        calculation = _RaisingCalculation()
    else:
        calculation = None

    generation: Any = (
        _ForeignCitationGeneration()
        if fixture.generation == "foreign_citation"
        else TrustedV2GenerationCapability()
    )
    return TrustedV2CapabilityPorts(
        retrieval=retrieval,
        evidence_evaluator=SemanticEvidenceEvaluationCapability(
            SemanticBinderService(SelectingBinderProvider())
        ),
        calculation=calculation,
        generation=generation,
        release_validator=TrustedReleaseValidationCapability(),
    )


def _budget(fixture: H1Fixture) -> AdaptiveRAGBudgetV1:
    return (
        AdaptiveRAGBudgetV1(**dict(fixture.budget))
        if fixture.budget
        else DEFAULT_BUDGET
    )


def _ports_complete(capabilities: TrustedV2CapabilityPorts) -> bool:
    return all(
        getattr(capabilities, name) is not None
        for name in (
            "retrieval",
            "evidence_evaluator",
            "calculation",
            "generation",
            "release_validator",
        )
    )


def run_fixture(fixture: H1Fixture, mode: AgentRuntimeMode) -> V2ExecutionOutcome:
    """Execute one fixture in one mode, through the real construction path.

    ``build_trusted_v2_runtime`` is the seam the production builder calls, so
    the capability graph, the budget validation and the runtime-mode coercion
    are the production ones.  The call goes to the coordinator rather than to
    ``TrustedFinancialRuntimeV2.execute`` because that adapter maps the outcome
    down to ``FinancialQueryResult``, which carries no turn trace -- and the
    turn trace is the thing this report exists to show.  ``adapter_agrees``
    below closes that gap by checking the transport mapping separately.
    """

    plan = fixture._plan()
    capabilities = _build_capabilities(fixture)
    supervisor = SupervisorService(
        DeterministicFallbackProvider({fixture.query: plan})
    )
    budget = _budget(fixture)
    request = _request(fixture.query, f"h1-{fixture.fixture_id}")
    complete = _ports_complete(capabilities)

    if not complete:
        # The factory refuses an incomplete graph on purpose; see the
        # unsupported_route fixture.
        coordinator = BoundedTrustedV2Coordinator(
            supervisor,
            capabilities=capabilities,
            budget=budget,
            runtime_mode=mode,
        )
        return asyncio.run(coordinator.execute(request))

    runtime = build_trusted_v2_runtime(
        supervisor,
        capabilities=capabilities,
        budget=budget,
        runtime_mode=mode,
    )
    return asyncio.run(runtime.coordinator.execute(request))


def adapter_agrees(fixture: H1Fixture, mode: AgentRuntimeMode) -> list[str]:
    """Check the transport adapter maps the coordinator's verdict faithfully."""

    plan = fixture._plan()
    capabilities = _build_capabilities(fixture)
    if not _ports_complete(capabilities):
        return []
    supervisor = SupervisorService(
        DeterministicFallbackProvider({fixture.query: plan})
    )
    runtime = build_trusted_v2_runtime(
        supervisor,
        capabilities=capabilities,
        budget=_budget(fixture),
        runtime_mode=mode,
    )
    request = FinancialQueryRequest(
        request_id=f"h1-{fixture.fixture_id}",
        user_id="user-7",
        session_id="session-1",
        original_query=fixture.query,
    )
    result = asyncio.run(runtime.execute(request))
    coordinator_outcome = asyncio.run(
        runtime.coordinator.execute(V2ExecutionRequest.from_financial_request(request))
    )

    problems: list[str] = []
    released = coordinator_outcome.release_status.value == "RELEASED"
    expected_status = (
        RuntimeStatus.ANSWER if released else RuntimeStatus.FAIL_CLOSED
    )
    if coordinator_outcome.status.value == "EXECUTION_ERROR":
        expected_status = RuntimeStatus.ERROR
    if result.status is not expected_status:
        problems.append(
            f"adapter reported {result.status.value}, coordinator said "
            f"{coordinator_outcome.status.value}"
        )
    if (result.release_status.value == "RELEASED") != released:
        problems.append("adapter release_status disagrees with the coordinator")
    if result.answer != coordinator_outcome.answer:
        problems.append("adapter answer differs from the coordinator's")
    return problems


def _turns(outcome: V2ExecutionOutcome) -> list[str]:
    trace = outcome.debug_metadata.get("trace", {})
    return list(trace.get("action_trace") or [])


def _transitions(outcome: V2ExecutionOutcome) -> list[str]:
    trace = outcome.debug_metadata.get("trace", {})
    return [item["to"] for item in trace.get("transitions", ())]


def _is_subsequence(expected: tuple[str, ...], actual: list[str]) -> bool:
    remaining = list(actual)
    for name in expected:
        if name not in remaining:
            return False
        remaining = remaining[remaining.index(name) + 1 :]
    return True


def _failure_reasons(
    fixture: H1Fixture, outcome: V2ExecutionOutcome, mode: AgentRuntimeMode
) -> list[str]:
    problems: list[str] = []
    released = outcome.release_status.value == "RELEASED"
    if released is not fixture.expect_released:
        problems.append(f"expected released={fixture.expect_released}, got {released}")
    if outcome.status.value != fixture.expect_status:
        problems.append(
            f"expected status={fixture.expect_status}, got {outcome.status.value}"
        )
    for reason in fixture.expect_reasons:
        if reason not in outcome.reason_codes:
            problems.append(f"expected reason {reason} in {outcome.reason_codes}")
    if mode is AgentRuntimeMode.HARNESS_V3 and fixture.expect_transitions:
        actual = _transitions(outcome)
        if not _is_subsequence(fixture.expect_transitions, actual):
            problems.append(
                f"expected phase sequence {list(fixture.expect_transitions)} "
                f"in order within {actual}"
            )
    return problems


def fixture_report(fixture: H1Fixture) -> dict[str, Any]:
    """Run one fixture in both modes and describe the comparison."""

    legacy = run_fixture(fixture, AgentRuntimeMode.LEGACY)
    harness = run_fixture(fixture, AgentRuntimeMode.HARNESS_V3)

    differences = decision_differences(legacy, harness)
    failures = [
        f"legacy: {problem}"
        for problem in _failure_reasons(fixture, legacy, AgentRuntimeMode.LEGACY)
    ] + [
        f"harness_v3: {problem}"
        for problem in _failure_reasons(fixture, harness, AgentRuntimeMode.HARNESS_V3)
    ]
    failures += [
        f"transport[{mode.value}]: {problem}"
        for mode in (AgentRuntimeMode.LEGACY, AgentRuntimeMode.HARNESS_V3)
        for problem in adapter_agrees(fixture, mode)
    ]

    return {
        "fixture_id": fixture.fixture_id,
        "description": fixture.description,
        "sealed_spec": fixture.spec(),
        "legacy": {
            "status": legacy.status.value,
            "route": legacy.route,
            "released": legacy.release_status.value == "RELEASED",
            "answer": legacy.answer,
            "reason_codes": list(legacy.reason_codes),
            "turns": _turns(legacy),
        },
        "harness_v3": {
            "status": harness.status.value,
            "route": harness.route,
            "released": harness.release_status.value == "RELEASED",
            "answer": harness.answer,
            "reason_codes": list(harness.reason_codes),
            "turns": _turns(harness),
            "transitions": _transitions(harness),
            "turn_count": harness.debug_metadata.get("trace", {}).get("turn_count"),
        },
        "decision_equivalent": not differences,
        "differences": differences,
        "expectation_failures": failures,
    }


def run_all(fixtures: tuple[H1Fixture, ...] = FIXTURES) -> dict[str, Any]:
    """Run every fixture and summarise the ablation."""

    reports = [fixture_report(fixture) for fixture in fixtures]
    return {
        "harness": "nf-v3-h1-harness-core",
        "sealed_digest": sealed_digest(fixtures),
        "fixture_count": len(reports),
        "fixtures": reports,
        "summary": {
            "decision_equivalence": sum(
                1 for report in reports if report["decision_equivalent"]
            ),
            "expectation_failures": sum(
                len(report["expectation_failures"]) for report in reports
            ),
            "release_bypass": sum(
                1
                for report in reports
                if report["harness_v3"]["released"]
                and not report["legacy"]["released"]
            ),
            "false_calculation_release": sum(
                1
                for report in reports
                if report["fixture_id"] in {"calculation_blocked", "calculation_error"}
                and report["harness_v3"]["released"]
            ),
            "infinite_loop": 0,
        },
    }
