"""TV2-06 V1-primary/V2-shadow integration tests."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from rag_v2.contracts import Intent, SupervisorPlan
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from src.runtime import (
    DeterministicCalculationCapability,
    FinancialQueryRequest,
    FinancialRuntimeModeError,
    FinancialRuntimeRouter,
    FinancialQueryResult,
    InMemoryShadowObservationSink,
    ReleaseStatus,
    RuntimeStatus,
    RuntimeVersion,
    FinancialRuntimeMode,
    TrustedFinancialRuntimeV2,
    TrustedReleaseValidationCapability,
    TrustedV2CapabilityPorts,
    TrustedV2GenerationCapability,
    V2ExecutionOutcome,
    V2ExecutionStatus,
    build_trusted_v2_runtime,
    resolve_financial_runtime_mode,
)
from src.runtime.query_lifecycle import (
    QueryExecutionService,
    QueryLifecycleService,
    UserTurnExecutionRequest,
)
from src.runtime.trusted_v2_generation import CandidateExecutionResult
from src.runtime.trusted_v2_contracts import V2ExecutionRequest

from tests.test_trusted_v2_r4_binder import (
    SelectingBinderProvider,
    _fact,
    _plan,
    _real_capabilities,
    _slot,
)


def _request_contract(
    query: str = "What was revenue?",
    *,
    request_id: str = "tv2-06",
    resolved_query: str | None = None,
) -> FinancialQueryRequest:
    standalone = resolved_query or query
    return FinancialQueryRequest(
        request_id=request_id,
        user_id="42",
        session_id="session-tv2-06",
        original_query=query,
        standalone_query=standalone,
        query_as_resolved=resolved_query is not None,
        request_metadata={
            "conversation_history": [{"role": "assistant", "content": "old answer"}],
            "memory_profile": {"preferred_currency": "USD"},
        },
    )


def _result(
    answer: str = "Revenue: 100 USD million",
    *,
    version: RuntimeVersion = RuntimeVersion.V1,
    status: RuntimeStatus = RuntimeStatus.ANSWER,
    release: ReleaseStatus = ReleaseStatus.RELEASED,
    evidence_ids: list[str] | None = None,
    citation_ids: list[str] | None = None,
    calculation_ids: list[str] | None = None,
    reason_codes: list[str] | None = None,
    debug_metadata: dict[str, Any] | None = None,
) -> FinancialQueryResult:
    return FinancialQueryResult(
        status=status,
        answer=answer if status is not RuntimeStatus.CLARIFICATION_REQUIRED else None,
        citations=[],
        evidence_ids=evidence_ids or [],
        citation_ids=citation_ids or [],
        calculation_ids=calculation_ids or [],
        reason_codes=reason_codes or [],
        runtime_version=version,
        release_status=release,
        debug_metadata=debug_metadata or {},
    )


class _StubRuntime:
    def __init__(
        self,
        result: FinancialQueryResult | None = None,
        *,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.delay = delay
        self.error = error
        self.calls = 0
        self.requests: list[FinancialQueryRequest] = []

    async def execute(self, request: FinancialQueryRequest) -> FinancialQueryResult:
        self.calls += 1
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise AssertionError("stub result was not configured")
        return self.result


class _CapturingCoordinator:
    def __init__(self, outcome: V2ExecutionOutcome) -> None:
        self.outcome = outcome
        self.request: V2ExecutionRequest | None = None
        self.calls = 0

    async def execute(self, request: V2ExecutionRequest) -> V2ExecutionOutcome:
        self.calls += 1
        self.request = request
        return self.outcome


def _v2_result(
    *,
    status: V2ExecutionStatus = V2ExecutionStatus.FAIL_CLOSED,
) -> V2ExecutionOutcome:
    return V2ExecutionOutcome(
        status=status,
        release_status=(
            ReleaseStatus.RELEASED
            if status is V2ExecutionStatus.READY_FOR_RELEASE
            else ReleaseStatus.NOT_RELEASED
        ),
        answer="V2 candidate" if status is V2ExecutionStatus.READY_FOR_RELEASE else None,
        reason_codes=[] if status is V2ExecutionStatus.READY_FOR_RELEASE else ["MISSING_SLOT"],
    )


def _real_v2_fact_runtime(
    *,
    batches: list[list[str]] | None = None,
    facts: dict[str, dict[str, Any]] | None = None,
    generation: Any | None = None,
    plan: SupervisorPlan | None = None,
    query: str = "What was revenue?",
    provider: SelectingBinderProvider | None = None,
    calculation: Any | None = None,
) -> tuple[TrustedFinancialRuntimeV2, Any, Any, Any]:
    facts = facts or {"E1": _fact("E1", value="100")}
    retrieval, binder, policy, reader, _ = _real_capabilities(
        batches or [["E1"]],
        facts,
        provider,
    )
    generation = generation or TrustedV2GenerationCapability()
    calculation = calculation or DeterministicCalculationCapability()
    validator = TrustedReleaseValidationCapability()
    capabilities = TrustedV2CapabilityPorts(
        retrieval=retrieval,
        evidence_evaluator=binder,
        calculation=calculation,
        generation=generation,
        release_validator=validator,
    )
    supervisor = SupervisorService(
        DeterministicFallbackProvider(
            {query: plan or _plan(_slot("revenue"))},
        ),
    )
    runtime = build_trusted_v2_runtime(supervisor, capabilities=capabilities)
    return runtime, policy, retrieval, validator


def test_mode_defaults_to_v1_and_rejects_active_v2() -> None:
    assert resolve_financial_runtime_mode() == "v1"
    assert resolve_financial_runtime_mode(FinancialRuntimeMode.V1) == "v1"
    assert resolve_financial_runtime_mode(environ={"FINANCIAL_RUNTIME_MODE": "shadow"}) == "shadow"
    with pytest.raises(FinancialRuntimeModeError):
        resolve_financial_runtime_mode("v2")


def test_v1_mode_executes_only_primary() -> None:
    primary = _StubRuntime(_result())
    shadow = _StubRuntime(_result("shadow"))
    router = FinancialRuntimeRouter(primary, shadow_runtime=shadow, mode="v1")
    result = asyncio.run(router.execute(_request_contract()))
    assert result is primary.result
    assert primary.calls == 1
    assert shadow.calls == 0
    assert router.v2_calls == 0
    assert router.last_observation is None


def test_shadow_primary_authority_and_same_request() -> None:
    primary = _StubRuntime(_result("V1 official"))
    shadow = _StubRuntime(
        _result(
            "V2 different",
            version=RuntimeVersion.V2,
            evidence_ids=["E2"],
            citation_ids=["C2"],
        ),
    )
    sink = InMemoryShadowObservationSink()
    router = FinancialRuntimeRouter(
        primary,
        shadow_runtime=shadow,
        mode="shadow",
        observation_sink=sink,
    )
    request = _request_contract(resolved_query="Apple FY2023 revenue")
    result = asyncio.run(router.execute(request))

    assert result is primary.result
    assert primary.calls == shadow.calls == 1
    assert primary.requests[0] is shadow.requests[0]
    assert primary.requests[0].standalone_query == "Apple FY2023 revenue"
    assert primary.requests[0].query_as_resolved is True
    observation = sink.observations[0]
    assert observation.v1_answer == "V1 official"
    assert observation.v2_answer == "V2 different"
    assert observation.comparison["category"] == "ANSWER_DISAGREEMENT"
    assert observation.comparison["needs_review"] is True


def test_v2_exception_does_not_change_primary_or_write_session() -> None:
    primary = _StubRuntime(_result("official"))
    shadow = _StubRuntime(error=RuntimeError("shadow boom"))
    sink = InMemoryShadowObservationSink()
    router = FinancialRuntimeRouter(
        primary,
        shadow_runtime=shadow,
        mode="shadow",
        observation_sink=sink,
    )
    result = asyncio.run(router.execute(_request_contract()))
    assert result.answer == "official"
    observation = sink.observations[0]
    assert observation.shadow_status == "ERROR"
    assert observation.shadow_error_stage == "V2_EXCEPTION"
    assert observation.comparison["category"] == "V2_ERROR"


def test_v2_timeout_does_not_change_primary() -> None:
    primary = _StubRuntime(_result("official"))
    shadow = _StubRuntime(_result("late"), delay=0.2)
    sink = InMemoryShadowObservationSink()
    router = FinancialRuntimeRouter(
        primary,
        shadow_runtime=shadow,
        mode="shadow",
        shadow_timeout_ms=10,
        observation_sink=sink,
    )
    started = time.perf_counter()
    result = asyncio.run(router.execute(_request_contract()))
    elapsed = time.perf_counter() - started
    assert result.answer == "official"
    assert elapsed < 0.15
    assert router.v2_timeout_count == 1
    assert sink.observations[0].shadow_status == "TIMEOUT"
    assert sink.observations[0].comparison["category"] == "V2_TIMEOUT"


def test_primary_error_is_not_replaced_by_v2_success() -> None:
    primary = _StubRuntime(error=RuntimeError("primary failure"))
    shadow = _StubRuntime(_result("shadow success", version=RuntimeVersion.V2))
    router = FinancialRuntimeRouter(
        primary,
        shadow_runtime=shadow,
        mode="shadow",
        shadow_timeout_ms=1000,
        observation_sink=InMemoryShadowObservationSink(),
    )
    with pytest.raises(RuntimeError, match="primary failure"):
        asyncio.run(router.execute(_request_contract()))
    assert router.last_observation is not None
    assert router.last_observation.v1_status == "ERROR"


def test_v2_boundary_receives_standalone_query_without_raw_context() -> None:
    coordinator = _CapturingCoordinator(_v2_result())
    shadow = TrustedFinancialRuntimeV2(coordinator)
    primary = _StubRuntime(_result("official"))
    router = FinancialRuntimeRouter(
        primary,
        shadow_runtime=shadow,
        mode="shadow",
        observation_sink=InMemoryShadowObservationSink(),
    )
    request = _request_contract(resolved_query="Apple FY2023 revenue")
    asyncio.run(router.execute(request))
    assert coordinator.request is not None
    assert coordinator.request.standalone_query == "Apple FY2023 revenue"
    assert "conversation_history" not in coordinator.request.request_metadata
    assert "memory_profile" not in coordinator.request.request_metadata


def test_real_factory_fact_shadow_reaches_release() -> None:
    runtime, policy, retrieval, validator = _real_v2_fact_runtime()
    primary = _StubRuntime(_result("V1 official"))
    sink = InMemoryShadowObservationSink()
    router = FinancialRuntimeRouter(
        primary,
        shadow_runtime=runtime,
        mode="shadow",
        observation_sink=sink,
    )
    result = asyncio.run(router.execute(_request_contract()))
    observation = sink.observations[0]
    assert result.answer == "V1 official"
    assert policy.calls == 1
    assert retrieval.calls == 1
    assert validator.validation_calls == 1
    assert observation.v2_status == "ANSWER"
    assert observation.v2_release_status == "RELEASED"
    assert observation.v2_evidence_ids == ("E1",)


def test_real_factory_calculation_shadow_reaches_release() -> None:
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
    runtime, _, retrieval, validator = _real_v2_fact_runtime(
        facts=facts,
        batches=[["CURRENT", "PRIOR"]],
        plan=plan,
        query="Compare years",
        provider=SelectingBinderProvider(),
    )
    primary = _StubRuntime(_result("V1 calculation"))
    sink = InMemoryShadowObservationSink()
    router = FinancialRuntimeRouter(
        primary,
        shadow_runtime=runtime,
        mode="shadow",
        observation_sink=sink,
    )
    asyncio.run(router.execute(_request_contract("Compare years", request_id="tv2-06-calc")))
    observation = sink.observations[0]
    assert observation.v2_status == "ANSWER"
    assert observation.v2_calculation_ids
    assert observation.v2_route == "CALCULATION_SIMPLE"
    assert validator.validation_calls == 1
    assert retrieval.calls == 1


def test_real_factory_fail_closed_shadow_is_observation_only() -> None:
    runtime, _, retrieval, validator = _real_v2_fact_runtime(
        batches=[[]],
        facts={"E1": _fact("E1", value="100")},
    )
    primary = _StubRuntime(_result("V1 still official"))
    sink = InMemoryShadowObservationSink()
    router = FinancialRuntimeRouter(
        primary,
        shadow_runtime=runtime,
        mode="shadow",
        observation_sink=sink,
    )
    result = asyncio.run(router.execute(_request_contract(request_id="tv2-06-fail")))
    observation = sink.observations[0]
    assert result.answer == "V1 still official"
    assert observation.v2_status == "FAIL_CLOSED"
    assert observation.v2_release_status == "NOT_RELEASED"
    assert retrieval.calls <= 4
    assert validator.validation_calls == 0


class _BadCandidateGeneration:
    candidate_mode = True

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, state: Any) -> CandidateExecutionResult:
        self.calls += 1
        return CandidateExecutionResult(
            candidate_answer="Revenue (FY2024): 999 USD million [citation-E1]",
            route="STRUCTURED_SINGLE",
            route_reason="shadow-repair-fixture",
            bound_evidence_ids=("E1",),
            citation_ids=("citation-E1",),
        )


def test_real_factory_repair_once_is_shadow_only() -> None:
    generation = _BadCandidateGeneration()
    runtime, _, _, validator = _real_v2_fact_runtime(generation=generation)
    primary = _StubRuntime(_result("V1 official"))
    sink = InMemoryShadowObservationSink()
    router = FinancialRuntimeRouter(
        primary,
        shadow_runtime=runtime,
        mode="shadow",
        observation_sink=sink,
    )
    asyncio.run(router.execute(_request_contract(request_id="tv2-06-repair")))
    observation = sink.observations[0]
    assert observation.v2_status == "ANSWER"
    assert observation.v2_repair_count == 1
    assert validator.repair_calls == 1
    assert observation.v1_answer == "V1 official"


def test_shared_query_lifecycle_uses_router_for_one_json_or_sse_business_result() -> None:
    primary = _StubRuntime(
        _result(
            "official",
            debug_metadata={"legacy_response": {"answer": "official", "sources": [], "searched_docs": []}},
        ),
    )
    shadow = _StubRuntime(_result("shadow", version=RuntimeVersion.V2))
    sink = InMemoryShadowObservationSink()
    router = FinancialRuntimeRouter(
        primary,
        shadow_runtime=shadow,
        mode="shadow",
        observation_sink=sink,
    )

    class _Session:
        def __init__(self) -> None:
            self.messages: list[tuple[Any, ...]] = []

        def get_recent_messages(self, session_id: str, user_id: int) -> list[dict[str, Any]]:
            return []

        def add_message(self, *args: Any, **kwargs: Any) -> None:
            self.messages.append((args, kwargs))

    class _Memory:
        def get_profile(self, user_id: int) -> dict[str, Any]:
            return {}

    service = QueryLifecycleService(
        session_manager=_Session(),
        memory_store=_Memory(),
        get_rag_engine=lambda: object(),
        get_conversation_service=lambda: None,
        financial_runtime_adapter_enabled=lambda: True,
        active_query_requires_context=lambda query: False,
        active_query_is_out_of_scope=lambda query: False,
        assistant_session_metadata=lambda **kwargs: {},
        execution_service_factory=QueryExecutionService,
        financial_runtime_factory=lambda engine, request: router,
    )
    result = asyncio.run(
        service.execute_user_turn(
            UserTurnExecutionRequest(
                request_id="tv2-06-lifecycle",
                user_id=42,
                original_query="What was revenue?",
                session_id="session-tv2-06",
                conversation_mode="off",
            ),
        ),
    )
    assert result.answer == "official"
    assert result.runtime_version == "V1"
    assert primary.calls == shadow.calls == 1
    assert len(sink.observations) == 1
    assert len(service.session_manager.messages) == 2
