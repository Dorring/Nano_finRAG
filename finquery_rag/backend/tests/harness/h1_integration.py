"""NF-V3 H1.1: the sealed fixture set and its runner.

``harness_v3`` is an ablation of an existing runtime, so the claim it has to
earn is not "it works" but "it decides the same thing".  A unit test can only
show that for a coordinator assembled in the test.

This module drives the fixture through the *production* entry point:

    NF_AGENT_RUNTIME_MODE (env)
      -> resolve_agent_runtime_mode()
      -> build_trusted_v2_runtime_for_request()
      -> build_trusted_v2_runtime()          (the factory)
      -> BoundedTrustedV2Coordinator
      -> BoundedAdaptiveRAGV1.run()

over the real R4 retriever, the real Semantic Binder, the real deterministic
calculator, the real generator routing and the real release validator.  The run
asserts the flag actually took effect on the coordinator it produced, so a mode
that silently failed to propagate cannot pass.

The fixture set is sealed: :func:`sealed_digest` hashes every fixture's
specification and the integration report records it.

One fixture -- ``unsupported_route`` -- is not factory-eligible.  The factory
refuses an incomplete dependency graph, and that refusal is correct, which means
the harness's ``UNSUPPORTED_TOOL_ROUTE`` guard is only reachable from a
manually constructed coordinator.  The report marks it rather than hiding it,
and ``retrieval_error`` covers the fail-closed path that *is* production
reachable.

Retrieval is routed by query, not by call order: ``CandidateDirectR4Policy`` in
the production path does not advance a scripted reader between replan rounds, so
a fixture whose second round is driven by round index would silently test
nothing.  A fixture declares which candidate keys each query shape returns.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from rag_v2.adaptive import AdaptiveRAGBudgetV1
from rag_v2.contracts import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.evidence.binder_service import SemanticBinderService
from rag_v2.supervisor import (
    DeterministicFallbackProvider,
    SupervisorService,
    UnknownSemanticPolicy,
)
from src.pdf_retrieval_v4.candidate_view_index import CandidateSearchHit
from src.runtime import (
    CandidateExecutionResult,
    FinancialQueryRequest,
    TrustedReleaseValidationCapability,
    TrustedV2CapabilityPorts,
    TrustedV2RuntimeResources,
    V2ExecutionOutcome,
    V2ExecutionRequest,
    build_trusted_v2_runtime_for_request,
)
from src.runtime.harness_runtime_mode import ENV_VAR, AgentRuntimeMode
from src.runtime.trusted_v2_binder import SemanticEvidenceEvaluationCapability
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator
from src.runtime.trusted_v2_factory import build_trusted_v2_runtime
from src.runtime.trusted_v2_production import _coordinate_key
from tests.harness.equivalence import decision_differences
from tests.harness.harness_support import (
    BUDGET,
    CALCULATION_FACTS,
    BlockedCalculation,
    RaisingCalculation,
)
from tests.test_trusted_v2_r4_binder import (
    ScriptedIndexReader,
    SelectingBinderProvider,
    _fact,
)

__all__ = [
    "DEFAULT_BUDGET",
    "FIXTURES",
    "H1Fixture",
    "fixture_report",
    "run_all",
    "run_fixture",
    "sealed_coverage_report",
    "sealed_digest",
]

# The composed suite and this one share one budget literal; a second copy here
# is a second place for it to drift.
DEFAULT_BUDGET = BUDGET
TIGHT_BUDGET = AdaptiveRAGBudgetV1(
    max_replan_rounds=0, max_total_tool_calls=1, max_same_tool_retry=0
)

#: Turns the harness records that are not tool calls: generation and
#: verification are loop phases, and CALCULATE is a phase of its own.
_NON_TOOL_TURNS = frozenset({"GENERATE", "VERIFY", "CALCULATE"})

#: The per-capability facts the trace exposes, used to check that a capability
#: did or did not run.  The trace flattens each port's snapshot into these
#: rather than storing the snapshot verbatim.
_CAPABILITY_VIEW_KEYS = (
    "calculator_invoked",
    "renderer_invoked",
    "specialist_invoked",
    "candidate_ready",
    "calculation_result_id",
    "candidate_generation_id",
    "validation_id",
    "validation_passed",
    "release_decision",
)

CURRENT_FACT = CALCULATION_FACTS["CURRENT"]
PRIOR_FACT = CALCULATION_FACTS["PRIOR"]
REVENUE_FACT = _fact("REVENUE", slots=("revenue",), value="391")
WRONG_PERIOD_FACT = _fact("WRONG", period="FY2023", slots=("revenue",), value="90")


@dataclass(frozen=True)
class H1Fixture:
    """One frozen input for the ablation comparison.

    ``routes`` maps a casefolded query substring to the candidate keys the index
    returns for it.  Rules are tried in order and the empty tag is the default,
    so ``(("period=", ("RIGHT",)), ("", ("WRONG",)))`` means "a query that names
    a period returns RIGHT, anything else returns WRONG".
    """

    fixture_id: str
    description: str
    query: str
    slots: tuple[dict[str, Any], ...]
    facts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    routes: tuple[tuple[str, tuple[str, ...]], ...] = (("", ()),)
    intent: str = Intent.DIRECT_FACT.value
    operation: str | None = None
    calculation: str = "none"
    generation: str = "trusted"
    retrieval: bool = True
    retrieval_raises: bool = False
    budget: Mapping[str, int] | None = None
    expect_released: bool = False
    expect_status: str = "FAIL_CLOSED"
    expect_reasons: tuple[str, ...] = ()
    #: Phases the harness run must pass through, in order.  This is the
    #: end-to-end trace claim: a fact query must be seen to reach RELEASE, and a
    #: calculation query must be seen to reach CALCULATE before it.
    expect_transitions: tuple[str, ...] = ()
    #: Whether the production factory can build this fixture's capability graph.
    #: False means the fixture is exercised through a directly constructed
    #: coordinator and the report says why.
    factory_eligible: bool = True

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

    def keys_for(self, query: str) -> tuple[str, ...]:
        folded = query.casefold()
        for tag, keys in self.routes:
            if tag and tag in folded:
                return keys
        for tag, keys in self.routes:
            if not tag:
                return keys
        return ()

    def spec(self) -> dict[str, Any]:
        """The fixture as plain JSON, for hashing and for the report.

        Every declared field appears here, expectations included.  An earlier
        version omitted the ``expect_*`` fields, so the sealed digest did not
        change when a fixture's expectation did -- it sealed the inputs while
        claiming to seal the fixture.  ``test_every_declared_fixture_field_is_sealed``
        keeps this in step with the dataclass.
        """

        return {
            "fixture_id": self.fixture_id,
            "description": self.description,
            "query": self.query,
            "slots": [dict(slot) for slot in self.slots],
            "facts": {key: dict(value) for key, value in sorted(self.facts.items())},
            "routes": [[tag, list(keys)] for tag, keys in self.routes],
            "intent": self.intent,
            "operation": self.operation,
            "calculation": self.calculation,
            "generation": self.generation,
            "retrieval": self.retrieval,
            "retrieval_raises": self.retrieval_raises,
            "budget": dict(self.budget) if self.budget else None,
            "expect_released": self.expect_released,
            "expect_status": self.expect_status,
            "expect_reasons": list(self.expect_reasons),
            "expect_transitions": list(self.expect_transitions),
            "factory_eligible": self.factory_eligible,
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
        routes=(("", ("REVENUE",)),),
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
        routes=(("", ("CURRENT", "PRIOR")),),
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
        routes=(("", ("CURRENT", "PRIOR")),),
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
        routes=(("", ("CURRENT", "PRIOR")),),
        intent=Intent.CALCULATION.value,
        operation="growth_rate",
        calculation="raising",
        expect_released=False,
        expect_status="EXECUTION_ERROR",
        expect_reasons=("CALCULATOR_EXCEPTION",),
    ),
    H1Fixture(
        fixture_id="wrong_period_no_progress",
        description=(
            "Only wrong-period evidence exists.  The replanner is invoked "
            "repeatedly and the run must exhaust its replan budget and fail "
            "closed, with identical route, reason codes and provenance in both "
            "modes.  Recovery itself is not reachable here -- see the module "
            "note on the R4 policy's derived queries -- so this fixture pins "
            "that a *failing* recovery does not drift between modes."
        ),
        query="What was revenue?",
        slots=(_bill("revenue"),),
        facts={"WRONG": WRONG_PERIOD_FACT},
        routes=(("", ("WRONG",)),),
        expect_released=False,
        expect_status="FAIL_CLOSED",
        expect_reasons=("WRONG_PERIOD",),
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
        routes=(("", ()),),
        expect_released=False,
        expect_status="FAIL_CLOSED",
    ),
    H1Fixture(
        fixture_id="retrieval_error",
        description=(
            "The index raises, so the retrieval tool fails.  The harness must "
            "record a tool error and terminate, not raise out of the loop."
        ),
        query="What was revenue?",
        slots=(_bill("revenue"),),
        facts={"REVENUE": REVENUE_FACT},
        routes=(("", ("REVENUE",)),),
        retrieval_raises=True,
        # The coordinator reports EXECUTION_ERROR, not FAIL_CLOSED: the R4
        # retrieval tool records the exception in `capability_errors`, and the
        # coordinator surfaces any recorded capability error after the loop.
        # The harness itself still terminated through its bounded contract.
        expect_released=False,
        expect_status="EXECUTION_ERROR",
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
        routes=(("", ("REVENUE",)),),
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
        routes=(("", ()),),
        budget=TIGHT_BUDGET.to_dict(),
        expect_released=False,
        expect_status="FAIL_CLOSED",
        expect_reasons=("BUDGET_EXHAUSTED",),
    ),
    H1Fixture(
        fixture_id="unsupported_route",
        description=(
            "No retrieval port is wired, so the initial action has no tool.  "
            "Not factory-eligible: the factory refuses an incomplete graph, so "
            "this guard is reachable only from a manually constructed "
            "coordinator.  That is the point of the guard existing as well."
        ),
        query="What was revenue?",
        slots=(_bill("revenue"),),
        retrieval=False,
        expect_released=False,
        expect_status="FAIL_CLOSED",
        expect_reasons=("UNSUPPORTED_TOOL_ROUTE",),
        factory_eligible=False,
    ),
)


# --- coverage against the repository's own sealed case set -------------------
#
# ``tests/fixtures/tv2_07_production_readiness/`` is a committed, sealed case set
# with per-case expectations (release, route, evidence ids, citation ids).  It is
# a *scoring* set: it ships labels but no fact corpus, so a case cannot be
# executed without inventing the facts behind its ``fixture_key`` -- which is
# what these fixtures do anyway.  What it can be used for honestly is coverage:
# it is a far richer inventory of what a financial harness must handle than
# anything written from scratch, and naming the cases this seal does *not* reach
# is more useful than implying it reaches them.
#
# ``None`` means no fixture here exercises that case.  The reason is recorded for
# every gap, and a test asserts every sealed case appears exactly once, so the
# table cannot silently drift out of step with the dataset.
SEALED_CASE_COVERAGE: Mapping[str, tuple[str | None, str]] = {
    "fact_direct": ("fact_direct", "covered"),
    "fact_margin": ("fact_direct", "covered: same shape, one admitted fact"),
    "multi_evidence": (None, "gap: needs a MULTI-route multi-evidence release"),
    "multi_evidence_two": (None, "gap: needs a MULTI-route partial-evidence case"),
    "calc_growth": ("calculation_growth_rate", "covered"),
    "calc_difference": (
        "calculation_growth_rate",
        "covered: same calculator path, different operation",
    ),
    "qualitative": (None, "gap: needs a MULTI-route qualitative synthesis model"),
    "table_heavy": (None, "gap: needs table-derived evidence"),
    "cross_source": (None, "gap: needs multi-document cross-source binding"),
    "wrong_period": ("wrong_period_no_progress", "covered: as a fail-closed case"),
    "wrong_row": (None, "gap: needs a wrong-row trap corpus"),
    "unit_scale": (None, "gap: needs a unit/scale trap corpus"),
    "no_answer": ("missing_evidence", "covered: no evidence admitted"),
    "missing_slot": ("missing_evidence", "covered: insufficient evidence"),
    "conflict": (None, "gap: needs conflicting-evidence binding"),
    "unsupported_calculation": (
        "calculation_blocked",
        "covered: as a calculator that declines to execute",
    ),
    "recovery_slot": (None, "gap: needs a slot recovery the retriever can drive"),
    "recovery_period": (
        None,
        "gap: the retriever issues identical queries across replan rounds, so "
        "this case cannot be driven through production wiring (see seal note)",
    ),
    "repair_once": (None, "gap: the release path is single-shot in both modes"),
    "validator_rejection": ("validator_rejection", "covered"),
    "assistant_history": (None, "gap: needs a conversation-history case"),
    "unknown_citation": (
        "validator_rejection",
        "covered: same release gate, unadmitted citation",
    ),
}

SEALED_CASES_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "tv2_07_production_readiness"
    / "questions.jsonl"
)


def sealed_case_keys() -> list[str]:
    """The ``fixture_key`` of every case in the committed sealed set."""

    keys = []
    with SEALED_CASES_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                keys.append(str(json.loads(line)["metadata"]["fixture_key"]))
    return keys


def sealed_coverage_report() -> dict[str, Any]:
    """Which sealed cases this fixture set reaches, and which it does not."""

    keys = sealed_case_keys()
    unknown = [key for key in keys if key not in SEALED_CASE_COVERAGE]
    if unknown:
        raise AssertionError(f"sealed cases with no coverage entry: {unknown}")
    # The other direction.  A stale entry for a case the dataset no longer has
    # is invisible to the check above: ``cases`` is built by iterating the
    # dataset keys, so a deleted case left the table describing something that
    # no longer exists and nothing noticed.
    known = set(keys)
    stale = sorted(key for key in SEALED_CASE_COVERAGE if key not in known)
    if stale:
        raise AssertionError(f"coverage entries for unknown sealed cases: {stale}")
    covered = [key for key in keys if SEALED_CASE_COVERAGE[key][0] is not None]
    unclaimed = [
        key
        for key, (fixture_id, _) in SEALED_CASE_COVERAGE.items()
        if fixture_id is not None and fixture_id not in {f.fixture_id for f in FIXTURES}
    ]
    if unclaimed:
        raise AssertionError(f"coverage names unknown fixtures: {unclaimed}")
    return {
        "sealed_cases": len(keys),
        "covered": len(covered),
        "not_covered": len(keys) - len(covered),
        "cases": {
            key: {"fixture_id": SEALED_CASE_COVERAGE[key][0],
                  "note": SEALED_CASE_COVERAGE[key][1]}
            for key in keys
        },
    }


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


class _QueryRoutedIndexReader(ScriptedIndexReader):
    """An index whose result depends on the query, as a real index does."""

    def __init__(self, fixture: H1Fixture) -> None:
        super().__init__([])
        self.fixture = fixture

    def search(
        self,
        lane: str,
        query: str,
        *,
        allowed_candidate_keys: set[str] | None = None,
        k: int = 50,
    ) -> list[CandidateSearchHit]:
        self.search_calls += 1
        self.seen_lanes.append(lane)
        self.seen_queries.append(query)
        if self.fixture.retrieval_raises:
            raise RuntimeError("index unavailable")
        keys = list(self.fixture.keys_for(query))
        if allowed_candidate_keys is not None:
            keys = [key for key in keys if key in allowed_candidate_keys]
        return [
            CandidateSearchHit(
                candidate_key=key,
                view_id=f"{lane}:{key}",
                lane=lane,
                bm25_rank=index if "bm25" in lane else None,
                dense_rank=index if "dense" in lane else None,
                bm25_score=1.0 if "bm25" in lane else None,
                dense_score=1.0 if "dense" in lane else None,
            )
            for index, key in enumerate(keys, 1)
        ][:k]


class _FixtureFactStore:
    """``materialize``-only fact store standing in for the real one.

    It carries the coordinate lookup too, because the operator-ambiguity guard
    (P1.6-0) asks the authoritative store whether a bound operand is uniquely
    identifiable.  A stand-in that omitted it would make the guard raise here
    and run in production -- the harness testing a different system from the one
    that ships, which is the divergence every fixture in this file exists to
    prevent.
    """

    def __init__(self, facts: Mapping[str, Mapping[str, Any]]) -> None:
        self.facts = dict(facts)

    def materialize(self, candidate_key: str) -> Mapping[str, Any]:
        return self.facts[str(candidate_key)]

    def facts_at_coordinate(
        self,
        entity: Any,
        metric: Any,
        period: Any,
    ) -> tuple[Mapping[str, Any], ...]:
        key = _coordinate_key({"entity": entity, "metric": metric, "period": period})
        return tuple(
            fact for fact in self.facts.values() if _coordinate_key(fact) == key
        )

    def candidate_keys_for_entities(
        self, entities: Any
    ) -> frozenset[str]:
        """The coordinate lookup's sibling, for the same reason.

        Retrieval narrows a cross-entity plan's slots to their own filer *before*
        the pool is cut, and asks the store which candidates are a filer's.  A
        stand-in that cannot answer it would raise here and not in production,
        which is the divergence every fixture in this file exists to prevent.
        """

        wanted = {
            _coordinate_key({"entity": entity})[0]
            for entity in entities
            if entity
        }
        wanted.discard("")
        return frozenset(
            str(key)
            for key, fact in self.facts.items()
            if _coordinate_key(fact)[0] in wanted
        )


class _FixtureSpecialist:
    """A specialist backend that never gets asked to generate numbers."""

    def generate(self, prompt: str) -> str:
        return "fixture specialist answer"


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

    def trace_snapshot(self) -> dict[str, Any]:
        return {"generation_calls": self.calls, "candidate_ready": self.calls > 0}


#: Fixtures whose capability graph the production entry point cannot build as
#: written, and why.  Derived from each fixture's own declaration rather than
#: looked up by id: a registry keyed by ``fixture_id`` is invisible to a module
#: that names its fixtures differently, and silently handing such a fixture the
#: production generator makes it test the wrong thing while appearing to pass.
SUBSTITUTED_PORT_REASONS: Mapping[str, str] = {
    "calculation_blocked": "needs a calculator that returns BLOCKED",
    "calculation_raising": "needs a calculator that raises",
    "generation_foreign_citation": "needs a generator that cites unadmitted evidence",
}


def substituted_ports(fixture: H1Fixture) -> str | None:
    """Why this fixture cannot use the production capability graph, if it cannot."""

    if fixture.calculation == "blocked":
        return SUBSTITUTED_PORT_REASONS["calculation_blocked"]
    if fixture.calculation == "raising":
        return SUBSTITUTED_PORT_REASONS["calculation_raising"]
    if fixture.generation == "foreign_citation":
        return SUBSTITUTED_PORT_REASONS["generation_foreign_citation"]
    return None


def _build_capabilities(fixture: H1Fixture, resources: TrustedV2RuntimeResources) -> Any:
    """Reconstruct what the production builder builds, with a port substituted.

    This mirrors ``build_trusted_v2_runtime_for_request``'s construction block
    on purpose, including its ``STRICT_DIRECT_FACT`` semantic policy -- the
    fixtures that already need a substituted port must not also be the ones
    running behind a laxer gate than production.  It is used only for the
    fixtures the caller reports as substituted, and the report says so, so a
    divergence between this and the production builder cannot be mistaken for
    production wiring.
    """

    from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever
    from src.runtime import (
        CandidateDirectR4Policy,
        R4RetrievalCapability,
        SemanticEvidenceEvaluationCapability,
        TrustedV2GenerationCapability,
    )
    from src.runtime.trusted_v2_generation import DeterministicFactRenderer

    document_scope: tuple[str, ...] = ()
    retriever = CandidateDirectRetriever(resources.index_reader)
    policy = CandidateDirectR4Policy(
        retriever,
        materializer=resources.fact_store.materialize,
        document_scope=document_scope,
    )
    calculator = BlockedCalculation()
    if fixture.calculation == "raising":
        calculator = RaisingCalculation()
    generation = TrustedV2GenerationCapability(
        routing_policy=None,
        renderer=DeterministicFactRenderer(),
        model_backend=resources.specialist,
    )
    if fixture.generation == "foreign_citation":
        generation = _ForeignCitationGeneration()
    return TrustedV2CapabilityPorts(
        retrieval=R4RetrievalCapability(policy, document_scope=document_scope),
        evidence_evaluator=SemanticEvidenceEvaluationCapability(resources.binder),
        calculation=calculator,
        generation=generation,
        release_validator=TrustedReleaseValidationCapability(),
    )


def _resources(fixture: H1Fixture) -> TrustedV2RuntimeResources:
    plan = fixture._plan()
    return TrustedV2RuntimeResources(
        index_reader=_QueryRoutedIndexReader(fixture),
        fact_store=_FixtureFactStore(fixture.facts),
        supervisor=SupervisorService(
            DeterministicFallbackProvider({fixture.query: plan})
        ),
        binder=SemanticBinderService(SelectingBinderProvider()),
        specialist=_FixtureSpecialist(),  # type: ignore[arg-type]
        budget=(
            AdaptiveRAGBudgetV1(**dict(fixture.budget))
            if fixture.budget
            else DEFAULT_BUDGET
        ),
        config_fingerprint=f"h1-fixture-{fixture.fixture_id}",
        index_manifest={"row_count": len(fixture.facts)},
    )


@contextlib.contextmanager
def runtime_mode(mode: AgentRuntimeMode):
    """Set the flag the production builder reads, and restore it afterwards."""

    previous = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = mode.value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = previous


def _assert_flag_reached(coordinator: Any, mode: AgentRuntimeMode) -> None:
    if coordinator.runtime_mode is not mode:
        raise AssertionError(
            f"{ENV_VAR} did not reach the coordinator: asked for {mode.value}, "
            f"got {coordinator.runtime_mode.value}"
        )


def _uses_production_entry_point(fixture: H1Fixture) -> bool:
    """Whether the fixture ran through ``build_trusted_v2_runtime_for_request``.

    Three tiers exist: the production entry point (flag read from the
    environment, every port built by production code), the factory with one port
    substituted, and a directly constructed coordinator.  The report states
    which, so substituted wiring is never presented as production wiring.
    """

    return fixture.factory_eligible and substituted_ports(fixture) is None


def _financial_request(fixture: H1Fixture) -> FinancialQueryRequest:
    return FinancialQueryRequest(
        request_id=f"h1-{fixture.fixture_id}",
        user_id="user-7",
        session_id="session-1",
        original_query=fixture.query,
    )


def run_fixture(fixture: H1Fixture, mode: AgentRuntimeMode) -> V2ExecutionOutcome:
    """Execute one fixture in one mode.

    Three tiers, recorded per fixture in the report rather than blurred:

    1. production entry point -- ``NF_AGENT_RUNTIME_MODE`` is read by
       :func:`resolve_agent_runtime_mode` inside
       ``build_trusted_v2_runtime_for_request``, which builds every port;
    2. the factory directly, with one port substituted, for the three fixtures
       whose capability cannot be produced by the production builder;
    3. a directly constructed coordinator for ``unsupported_route``, whose graph
       the factory correctly refuses.
    """

    financial = _financial_request(fixture)
    request = V2ExecutionRequest.from_financial_request(financial)
    resources = _resources(fixture)

    if substituted_ports(fixture) is not None:
        runtime = build_trusted_v2_runtime(
            resources.supervisor,
            capabilities=_build_capabilities(fixture, resources),
            budget=resources.budget,
            # Same semantic gate as the production entry point.  A substituted
            # port is already a deviation; adding a laxer policy on top would
            # make these three fixtures differ from production in two ways at
            # once, and the report could not tell you which mattered.
            unknown_semantic_policy=UnknownSemanticPolicy.STRICT_DIRECT_FACT,
            runtime_mode=mode,
        )
    elif not fixture.factory_eligible:
        return _run_unwired(fixture, mode)
    else:
        with runtime_mode(mode):
            runtime = build_trusted_v2_runtime_for_request(
                None, financial, resources=resources
            )

    _assert_flag_reached(runtime.coordinator, mode)
    return asyncio.run(runtime.coordinator.execute(request))


def _run_unwired(fixture: H1Fixture, mode: AgentRuntimeMode) -> V2ExecutionOutcome:
    """Run a fixture whose capability graph the factory correctly refuses."""

    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(
            DeterministicFallbackProvider({fixture.query: fixture._plan()})
        ),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=None,
            evidence_evaluator=SemanticEvidenceEvaluationCapability(
                SemanticBinderService(SelectingBinderProvider())
            ),
        ),
        budget=DEFAULT_BUDGET,
        runtime_mode=mode,
    )
    _assert_flag_reached(coordinator, mode)
    return asyncio.run(
        coordinator.execute(
            V2ExecutionRequest.from_financial_request(_financial_request(fixture))
        )
    )


def adapter_agrees(fixture: H1Fixture, mode: AgentRuntimeMode) -> list[str]:
    """Check the transport adapter reports the coordinator's verdict faithfully.

    The adapter maps ``V2ExecutionOutcome`` down to ``FinancialQueryResult`` and
    drops the trace, so it is the one production surface the runner cannot read a
    trace from -- and therefore the one that could disagree silently.
    """

    if substituted_ports(fixture) is not None or not fixture.factory_eligible:
        return []
    request = _financial_request(fixture)
    with runtime_mode(mode):
        runtime = build_trusted_v2_runtime_for_request(
            None, request, resources=_resources(fixture)
        )
    result = asyncio.run(runtime.execute(request))
    outcome = asyncio.run(
        runtime.coordinator.execute(V2ExecutionRequest.from_financial_request(request))
    )

    problems: list[str] = []
    released = outcome.release_status.value == "RELEASED"
    if (result.release_status.value == "RELEASED") != released:
        problems.append(
            f"adapter release_status {result.release_status.value!r} disagrees "
            f"with the coordinator's {outcome.release_status.value!r}"
        )
    if result.answer != outcome.answer:
        problems.append("adapter answer differs from the coordinator's")
    if result.status.value == "ANSWER" and not released:
        problems.append("adapter reported ANSWER for a non-released outcome")
    if result.status.value != "ANSWER" and released:
        problems.append("adapter did not report ANSWER for a released outcome")
    return problems


# --- reporting ---------------------------------------------------------------


def _turns(outcome: V2ExecutionOutcome) -> list[str]:
    return list(outcome.debug_metadata.get("trace", {}).get("action_trace") or [])


def _transitions(outcome: V2ExecutionOutcome) -> list[str]:
    return [
        item["to"] for item in outcome.debug_metadata.get("trace", {}).get("transitions", ())
    ]


def _capability_view(outcome: V2ExecutionOutcome) -> dict[str, Any]:
    """What the trace says each capability actually did.

    The trace flattens each port's own snapshot into these fields rather than
    keeping the snapshot verbatim, so this reads them from the trace itself.
    """

    trace = outcome.debug_metadata.get("trace", {})
    return {key: trace.get(key) for key in _CAPABILITY_VIEW_KEYS}


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
    if mode is AgentRuntimeMode.HARNESS_V3:
        problems += _trace_problems(fixture, outcome)
    return problems


def _trace_problems(fixture: H1Fixture, outcome: V2ExecutionOutcome) -> list[str]:
    """Trace-specific invariants.  These apply to harness_v3 only."""

    problems: list[str] = []
    trace = outcome.debug_metadata.get("trace", {})
    transitions = _transitions(outcome)
    turns = _turns(outcome)

    if not transitions or not turns:
        # ``unsupported_route`` never reaches a tool: there is no retrieval port
        # and therefore no turn to record.  An empty trace is its correct trace.
        if fixture.retrieval:
            problems.append("the run produced no turn trace")
        return problems

    if fixture.expect_transitions and not _is_subsequence(
        fixture.expect_transitions, transitions
    ):
        problems.append(
            f"expected phase sequence {list(fixture.expect_transitions)} "
            f"in order within {transitions}"
        )

    # The recorded turn count must describe the trace it is attached to.
    if trace.get("turn_count") != len(trace.get("turns") or []):
        problems.append(
            f"turn_count={trace.get('turn_count')} but {len(trace.get('turns') or [])} turns"
        )
    if turns != [turn["action"] for turn in (trace.get("turns") or [])]:
        problems.append("action_trace disagrees with the recorded turns")

    # Every recorded turn must correspond to a phase the run actually entered.
    # CALCULATE is a harness phase of its own, not an ACT, so it is counted
    # against its own transition rather than against the tool-call count.
    tool_actions = [name for name in turns if name not in _NON_TOOL_TURNS]
    act_phases = transitions.count("ACT")
    if len(tool_actions) != act_phases:
        problems.append(
            f"{len(tool_actions)} retrieval turns recorded but {act_phases} ACT phases"
        )
    if turns.count("CALCULATE") != transitions.count("CALCULATE"):
        problems.append(
            f"{turns.count('CALCULATE')} CALCULATE turns for "
            f"{transitions.count('CALCULATE')} CALCULATE phases"
        )

    # The public response must not contradict the trace it carries.
    if fixture.expect_released and outcome.route != trace.get("generation_route"):
        problems.append(
            f"outcome route {outcome.route!r} contradicts trace "
            f"generation_route {trace.get('generation_route')!r}"
        )

    # A terminal release must be *the* terminal, and no other outcome may claim it.
    terminal = outcome.runtime_metadata.get("terminal_state")
    released = outcome.release_status.value == "RELEASED"
    if fixture.expect_released and terminal != "RELEASED":
        problems.append(f"released outcome reports terminal_state={terminal!r}")
    if not fixture.expect_released and terminal == "RELEASED":
        problems.append("non-released outcome reports terminal_state='RELEASED'")
    if released != (terminal == "RELEASED"):
        problems.append("terminal_state and release_status disagree")

    if (outcome.status.value == "READY_FOR_RELEASE") != released:
        problems.append(
            f"status {outcome.status.value} disagrees with release "
            f"{outcome.release_status.value}"
        )

    # A calculation that never executed must not have produced anything.
    # ``renderer_invoked``/``specialist_invoked`` come from the generation
    # capability's own snapshot, so this fails if the candidate stage ever
    # reaches generation with a BLOCKED or raising calculator behind it.
    capability = _capability_view(outcome)
    if fixture.calculation in {"blocked", "raising"}:
        if capability["calculator_invoked"] is not True:
            problems.append("the calculator was never invoked")
        for name in ("renderer_invoked", "specialist_invoked", "candidate_ready"):
            if capability[name]:
                problems.append(
                    f"{name} is set although the calculation did not execute"
                )
        if capability["calculation_result_id"] is not None:
            problems.append(
                "a calculation result id was published for a calculation that "
                "did not execute"
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
    # The transport adapter is the one production surface the trace cannot be
    # read back from, so it is checked separately.  This call was dropped in the
    # rewiring and the check sat unasserted for a while -- it is now part of the
    # report, and ``test_the_transport_adapter_agrees_with_the_coordinator``
    # fails if it stops being consulted.
    failures += [
        f"transport[{mode.value}]: {problem}"
        for mode in (AgentRuntimeMode.LEGACY, AgentRuntimeMode.HARNESS_V3)
        for problem in adapter_agrees(fixture, mode)
    ]

    return {
        "fixture_id": fixture.fixture_id,
        "description": fixture.description,
        "production_entry_point": _uses_production_entry_point(fixture),
        "factory_eligible": fixture.factory_eligible,
        "sealed_spec": fixture.spec(),
        "legacy": _mode_view(legacy),
        "harness_v3": {
            **_mode_view(harness),
            "transitions": _transitions(harness),
            # Named "turn_records" and not "turns": ``_mode_view`` already
            # publishes "turns" as the list of action names, and both modes
            # share that key.  Reusing the name for the full per-turn dicts made
            # ``report[mode]["turns"]`` change shape depending on the mode.
            "turn_records": harness.debug_metadata.get("trace", {}).get("turns"),
            "capability_view": _capability_view(harness),
        },
        "decision_equivalent": not differences,
        "differences": differences,
        "expectation_failures": failures,
    }


def _mode_view(outcome: V2ExecutionOutcome) -> dict[str, Any]:
    return {
        "status": outcome.status.value,
        "route": outcome.route,
        "released": outcome.release_status.value == "RELEASED",
        "answer": outcome.answer,
        "reason_codes": list(outcome.reason_codes),
        "validator_status": outcome.validator_status,
        "turn_count": outcome.debug_metadata.get("trace", {}).get("turn_count"),
        "turns": _turns(outcome),
    }


def run_all(fixtures: tuple[H1Fixture, ...] = FIXTURES) -> dict[str, Any]:
    """Run every fixture and summarise the ablation."""

    reports = [fixture_report(fixture) for fixture in fixtures]
    return {
        "harness": "nf-v3-h1-harness-core",
        "sealed_digest": sealed_digest(fixtures),
        "fixture_count": len(reports),
        "sealed_case_coverage": sealed_coverage_report(),
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
            "infinite_loop": sum(
                1
                for report in reports
                if report["harness_v3"]["turn_count"] is not None
                and report["harness_v3"]["turn_count"] > 32
            ),
        },
    }
