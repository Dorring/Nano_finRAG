"""TV2-03 real R4 and Semantic Binder integration tests."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

import pytest

from rag_v2.adaptive import AdaptiveRAGBudgetV1
from rag_v2.contracts import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding
from rag_v2.evidence.binder_provider import (
    BinderCallMetadata,
    BinderProviderError,
    BinderProviderResult,
)
from rag_v2.evidence.binder_service import SemanticBinderService
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever
from src.pdf_retrieval_v4.candidate_view_index import CandidateSearchHit
from src.runtime import V2ExecutionRequest, V2ExecutionStatus
from src.runtime.trusted_v2_binder import SemanticEvidenceEvaluationCapability
from src.runtime.trusted_v2_capabilities import TrustedV2CapabilityPorts
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator
from src.runtime.trusted_v2_r4 import (
    CandidateDirectR4Policy,
    R4RetrievalCapability,
)


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


def _request(query: str, request_id: str = "tv2-03-request") -> V2ExecutionRequest:
    return V2ExecutionRequest(
        request_id=request_id,
        user_id="user-7",
        session_id="session-1",
        original_query=query,
        standalone_query=query,
    )


def _fact(
    fact_id: str,
    *,
    period: str = "FY2024",
    slots: tuple[str, ...] = ("revenue",),
    metric: str = "Revenue",
    value: str = "100",
) -> dict[str, Any]:
    return {
        "evidence_id": fact_id,
        "fact_id": fact_id,
        "candidate_id": fact_id,
        "metric": metric,
        "value": value,
        "period": period,
        "entity": "Apple",
        "scope": "consolidated",
        "unit": "USD",
        "currency": "USD",
        "scale": "million",
        "citation_id": f"citation-{fact_id}",
        "source": "fixture",
        "physical_source_id": f"source-{fact_id}",
        "document_id": f"doc-{fact_id}",
        "pdf_page": 1,
        "slots": list(slots),
        "provenance_complete": True,
    }


class ScriptedIndexReader:
    """Deterministic index reader used behind the real CandidateDirectRetriever."""

    def __init__(
        self,
        batches: list[list[str]],
        *,
        lane_batches: Mapping[str, list[str]] | None = None,
    ) -> None:
        self.batches = [list(batch) for batch in batches]
        self.lane_batches = {
            str(lane): list(values)
            for lane, values in (lane_batches or {}).items()
        }
        self.round = 0
        self.search_calls = 0
        self.seen_lanes: list[str] = []
        self.seen_queries: list[str] = []

    def _batch(self) -> list[str]:
        if not self.batches:
            return []
        return self.batches[min(self.round, len(self.batches) - 1)]

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
        keys = list(self.lane_batches.get(lane, self._batch()))
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

    def candidate_keys_for_documents(
        self,
        lane: str,
        document_ids: set[str],
    ) -> set[str]:
        return set(self._batch())

    def advance(self) -> None:
        self.round += 1


class ScriptedCandidateDirectPolicy(CandidateDirectR4Policy):
    def retrieve(self, request):
        result = super().retrieve(request)
        reader = self.retriever.reader
        if hasattr(reader, "advance"):
            reader.advance()
        return result


class SelectingBinderProvider:
    provider_name = "fixture"
    model_name = "deterministic-binder"
    last_call = None

    def __init__(self, *, preferred_ids: Mapping[str, str] | None = None) -> None:
        self.preferred_ids = dict(preferred_ids or {})
        self.calls = 0

    def bind(self, request: Mapping[str, Any]) -> BinderProviderResult:
        self.calls += 1
        facts = list(request["financial_facts"])
        bindings: dict[str, tuple[str, ...]] = {}
        missing: list[str] = []
        for slot in request["required_slots"]:
            slot_id = str(slot["slot_id"])
            matches = [
                fact
                for fact in facts
                if slot_id in tuple(str(item) for item in fact.get("slots", ()))
                and str(fact.get("metric", "")).casefold()
                == str(slot.get("metric", "")).casefold()
                and str(fact.get("period", "")).casefold()
                == str(slot.get("period", "")).casefold()
            ]
            preferred = self.preferred_ids.get(slot_id)
            if preferred:
                matches = [
                    fact for fact in matches if str(fact.get("fact_id")) == preferred
                ] or matches
            if matches:
                bindings[slot_id] = (str(matches[0]["fact_id"]),)
            else:
                missing.append(slot_id)
        status = (
            BindingStatus.BOUND.value
            if not missing
            else BindingStatus.MISSING.value
        )
        binding = EvidenceBinding(
            status=status,
            slot_bindings=bindings,
            missing_slots=tuple(missing),
        )
        metadata = BinderCallMetadata(
            provider=self.provider_name,
            model=self.model_name,
            provider_role="evidence_binder",
            model_role="deterministic_fixture",
            latency_ms=0.1,
            provider_response_success=True,
            structured_output_success=True,
        )
        self.last_call = metadata
        return BinderProviderResult(binding=binding, metadata=metadata)


class StructurallyValidWrongMetricProvider:
    """Deliberately binds a wrong-metric fact to exercise the semantic firewall."""

    provider_name = "fixture"
    model_name = "deterministic-wrong-metric"
    last_call = None

    def __init__(self) -> None:
        self.calls = 0

    def bind(self, request: Mapping[str, Any]) -> BinderProviderResult:
        self.calls += 1
        slot_id = str(request["required_slots"][0]["slot_id"])
        fact_id = str(request["financial_facts"][0]["fact_id"])
        binding = EvidenceBinding(
            status=BindingStatus.BOUND.value,
            slot_bindings={slot_id: (fact_id,)},
        )
        metadata = BinderCallMetadata(
            provider=self.provider_name,
            model=self.model_name,
            provider_role="evidence_binder",
            model_role="deterministic_fixture",
            latency_ms=0.1,
            provider_response_success=True,
            structured_output_success=True,
        )
        self.last_call = metadata
        return BinderProviderResult(binding=binding, metadata=metadata)


class AmbiguousBinderProvider(SelectingBinderProvider):
    def bind(self, request: Mapping[str, Any]) -> BinderProviderResult:
        self.calls += 1
        slot_id = str(request["required_slots"][0]["slot_id"])
        binding = EvidenceBinding(
            status=BindingStatus.AMBIGUOUS.value,
            slot_bindings={},
            ambiguous_slots=(slot_id,),
        )
        metadata = BinderCallMetadata(
            provider=self.provider_name,
            model=self.model_name,
            provider_role="evidence_binder",
            model_role="deterministic_fixture",
            latency_ms=0.1,
            provider_response_success=True,
            structured_output_success=True,
        )
        return BinderProviderResult(binding=binding, metadata=metadata)


class InvalidSchemaBinderProvider(SelectingBinderProvider):
    def bind(self, request: Mapping[str, Any]) -> BinderProviderResult:
        error = BinderProviderError("malformed binder payload")
        error.schema_valid = False
        raise error


def _real_capabilities(
    batches: list[list[str]],
    facts: Mapping[str, Mapping[str, Any]],
    provider: SelectingBinderProvider | None = None,
):
    reader = ScriptedIndexReader(batches)
    retriever = CandidateDirectRetriever(reader, lane_k=10)
    policy = ScriptedCandidateDirectPolicy(
        retriever,
        materializer=lambda key: facts[key],
    )
    retrieval = R4RetrievalCapability(policy)
    binder_provider = provider or SelectingBinderProvider()
    binder = SemanticEvidenceEvaluationCapability(
        SemanticBinderService(binder_provider),
    )
    return retrieval, binder, policy, reader, binder_provider


def _coordinator(
    query: str,
    plan: SupervisorPlan,
    retrieval: R4RetrievalCapability,
    binder: SemanticEvidenceEvaluationCapability,
    *,
    calculation: Any = None,
    budget: AdaptiveRAGBudgetV1 | None = None,
) -> BoundedTrustedV2Coordinator:
    provider = DeterministicFallbackProvider({query: plan})
    return BoundedTrustedV2Coordinator(
        SupervisorService(provider),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=binder,
            calculation=calculation,
        ),
        budget=budget
        or AdaptiveRAGBudgetV1(
            max_replan_rounds=3,
            max_total_tool_calls=4,
            max_same_tool_retry=3,
        ),
    )


def test_p0_environment_and_real_component_imports() -> None:
    from enum import StrEnum

    import numpy
    import openai
    import jose

    assert StrEnum is not None
    assert numpy.__version__
    assert openai.__version__
    assert jose is not None
    assert CandidateDirectRetriever is not None
    assert SemanticBinderService is not None


def test_real_r4_and_real_binder_one_shot_produce_bound_provenance() -> None:
    facts = {"E1": _fact("E1")}
    retrieval, binder, policy, reader, provider = _real_capabilities(
        [["E1"]],
        facts,
    )
    coordinator = _coordinator(
        "What was revenue?",
        _plan(_slot("revenue")),
        retrieval,
        binder,
    )

    outcome = asyncio.run(coordinator.execute(_request("What was revenue?")))

    assert policy.calls > 0
    assert retrieval.calls == 1
    assert provider.calls == 1
    assert binder.calls == 1
    assert reader.search_calls > 0
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert outcome.release_status.value == "NOT_RELEASED"
    assert outcome.evidence_ids == ["E1"]
    assert outcome.citation_ids == ["citation-E1"]
    assert "DOWNSTREAM_EXECUTION_NOT_WIRED" in outcome.reason_codes
    trace = outcome.debug_metadata["trace"]
    assert trace["binder_status_per_round"] == ["BOUND"]
    assert trace["bound_evidence_ids"] == ["E1"]
    assert trace["candidate_ids_per_round"] == [["E1"]]


def test_semantic_firewall_rejects_structurally_bound_wrong_metric() -> None:
    facts = {"E1": _fact("E1", metric="Net Income")}
    wrong_metric_provider = StructurallyValidWrongMetricProvider()
    retrieval, binder, policy, reader, _ = _real_capabilities(
        [["E1"]],
        facts,
        wrong_metric_provider,
    )
    query = "What was Apple FY2024 operating income?"
    coordinator = _coordinator(
        query,
        _plan(_slot("value", metric="Operating Income")),
        retrieval,
        binder,
    )

    outcome = asyncio.run(coordinator.execute(_request(query)))

    assert policy.calls == 1
    assert reader.search_calls > 0
    assert wrong_metric_provider.calls == 1
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "QUERY_EVIDENCE_SEMANTIC_MISMATCH" in outcome.reason_codes
    assert outcome.evidence_ids == []
    semantic_check = binder.trace_snapshot()["semantic_checks"][0]
    assert semantic_check["status"] == "MISMATCH"
    assert "fact_metric_not_matching_slot:E1:value" in semantic_check["mismatches"]


def test_wrong_period_drives_real_targeted_recovery() -> None:
    facts = {
        "WRONG": _fact("WRONG", period="FY2023"),
        "RIGHT": _fact("RIGHT", period="FY2024"),
    }
    retrieval, binder, policy, _, provider = _real_capabilities(
        [["WRONG"], ["RIGHT"]],
        facts,
    )
    coordinator = _coordinator(
        "What was revenue?",
        _plan(_slot("revenue")),
        retrieval,
        binder,
    )

    outcome = asyncio.run(coordinator.execute(_request("What was revenue?")))

    assert policy.calls == 2
    assert retrieval.calls == 2
    assert provider.calls == 2
    assert outcome.evidence_ids == ["RIGHT"]
    trace = outcome.debug_metadata["trace"]
    assert trace["binder_status_per_round"] == ["MISSING", "BOUND"]
    assert "WRONG_PERIOD" in trace["reason_codes"]
    assert trace["wrong_period_slots"] == ["revenue"]


def test_missing_operand_recovery_never_calls_calculator() -> None:
    class CountingCalculation:
        calls = 0

        def calculate(self, state):
            self.calls += 1
            return "must not run"

    facts = {
        "CURRENT": _fact("CURRENT", slots=("current",), period="FY2024"),
        "PRIOR": _fact("PRIOR", slots=("prior",), period="FY2023"),
    }
    retrieval, binder, _, _, provider = _real_capabilities(
        [["CURRENT"], ["PRIOR"]],
        facts,
    )
    calculation = CountingCalculation()
    coordinator = _coordinator(
        "Compare years",
        _plan(
            _slot("current", period="FY2024", role="current"),
            _slot("prior", period="FY2023", role="prior"),
            intent=Intent.CALCULATION,
            operation="growth_rate",
        ),
        retrieval,
        binder,
        calculation=calculation,
    )

    outcome = asyncio.run(coordinator.execute(_request("Compare years")))

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert provider.calls == 2
    assert outcome.evidence_ids == ["CURRENT", "PRIOR"]
    assert calculation.calls == 0
    assert "MISSING_OPERAND" in outcome.debug_metadata["trace"]["reason_codes"]
    assert outcome.debug_metadata["trace"]["missing_operand_slots"] == ["prior"]


def test_candidate_and_bound_provenance_are_separate() -> None:
    facts = {
        "E1": _fact("E1"),
        "E2": _fact("E2"),
        "E3": _fact("E3"),
    }
    retrieval, binder, _, _, _ = _real_capabilities(
        [["E1", "E2", "E3"]],
        facts,
        SelectingBinderProvider(preferred_ids={"revenue": "E2"}),
    )
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
        ).execute(_request("What was revenue?"))
    )

    assert outcome.evidence_ids == ["E2"]
    trace = outcome.debug_metadata["trace"]
    assert set(trace["candidate_ids_per_round"][0]) == {"E1", "E2", "E3"}
    assert trace["bound_evidence_ids"] == ["E2"]


def test_r4_candidate_schema_failure_is_execution_error() -> None:
    facts = {"E1": {"metric": "Revenue", "period": "FY2024"}}
    retrieval, binder, _, _, _ = _real_capabilities(
        [["E1"]],
        facts,
    )
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
        ).execute(_request("What was revenue?"))
    )
    assert outcome.status is V2ExecutionStatus.EXECUTION_ERROR
    assert "CAPABILITY_EXCEPTION" in outcome.reason_codes


def test_binder_schema_failure_is_execution_error() -> None:
    facts = {"E1": _fact("E1")}
    retrieval, binder, _, _, _ = _real_capabilities(
        [["E1"]],
        facts,
        InvalidSchemaBinderProvider(),
    )
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
        ).execute(_request("What was revenue?"))
    )
    assert outcome.status is V2ExecutionStatus.EXECUTION_ERROR
    assert "CAPABILITY_EXCEPTION" in outcome.reason_codes


def test_unresolved_binder_conflict_never_reaches_downstream() -> None:
    facts = {"E1": _fact("E1")}
    retrieval, binder, _, _, provider = _real_capabilities(
        [["E1"]],
        facts,
        AmbiguousBinderProvider(),
    )
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
        ).execute(_request("What was revenue?"))
    )
    assert provider.calls == 1
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "EVIDENCE_CONFLICT" in outcome.reason_codes
    assert outcome.evidence_ids == []
    assert outcome.release_status.value == "NOT_RELEASED"


def test_candidate_dedup_is_stable_and_bound_slots_remain_explicit() -> None:
    facts = {"E1": _fact("E1")}
    retrieval, binder, _, _, _ = _real_capabilities(
        [["E1", "E1", "E1"]],
        facts,
    )
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
        ).execute(_request("What was revenue?"))
    )
    trace = outcome.debug_metadata["trace"]
    assert trace["candidate_ids_per_round"][0] == ["E1"]
    assert trace["bound_slot_ids"] == ["revenue"]


@pytest.mark.parametrize(
    "query,slot",
    [
        ("What was revenue?", _slot("revenue")),
        (
            "Compare years",
            _slot("current", period="FY2024", role="current"),
        ),
    ],
)
def test_r4_policy_uses_standalone_query_and_returns_candidates(
    query: str,
    slot: RequiredSlot,
) -> None:
    facts = {"E1": _fact("E1", slots=(slot.slot_id,), period=slot.period)}
    retrieval, _, policy, _, _ = _real_capabilities([["E1"]], facts)
    coordinator = _coordinator(query, _plan(slot), retrieval, SemanticEvidenceEvaluationCapability(
        SemanticBinderService(SelectingBinderProvider())
    ))
    outcome = asyncio.run(coordinator.execute(_request(query)))
    assert policy.calls == 1
    assert outcome.evidence_ids == ["E1"]


def test_real_r4_structured_lane_recovers_secondary_slot_under_crowding() -> None:
    facts = {
        "PRIMARY": _fact("PRIMARY"),
        "SECONDARY": _fact("SECONDARY"),
    }
    reader = ScriptedIndexReader(
        [["PRIMARY"]],
        lane_batches={
            "candidate_raw_bm25": ["PRIMARY"],
            "candidate_raw_dense": ["PRIMARY"],
            "candidate_structured_bm25": ["SECONDARY"],
            "candidate_structured_dense": ["SECONDARY"],
        },
    )
    retriever = CandidateDirectRetriever(reader, lane_k=10)
    policy = ScriptedCandidateDirectPolicy(
        retriever,
        materializer=lambda key: facts[key],
    )
    retrieval = R4RetrievalCapability(policy)
    binder_provider = SelectingBinderProvider(
        preferred_ids={"revenue": "SECONDARY"},
    )
    binder = SemanticEvidenceEvaluationCapability(
        SemanticBinderService(binder_provider),
    )
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
        ).execute(_request("What was revenue?"))
    )

    assert "candidate_structured_bm25" in reader.seen_lanes
    assert "candidate_structured_dense" in reader.seen_lanes
    trace = outcome.debug_metadata["trace"]
    assert set(trace["candidate_ids_per_round"][0]) == {"PRIMARY", "SECONDARY"}
    assert outcome.evidence_ids == ["SECONDARY"]
