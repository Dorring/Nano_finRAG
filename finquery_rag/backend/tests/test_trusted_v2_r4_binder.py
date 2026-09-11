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
from src.pdf_retrieval_v4.candidate_query_builder import (
    append_missing_query_terms,
    build_slot_query,
    build_slot_query_variants,
    extract_scope_phrase,
)
from src.pdf_retrieval_v4.planner import build_query_plan
from src.runtime import V2ExecutionRequest, V2ExecutionStatus
from src.runtime.trusted_v2_binder import SemanticEvidenceEvaluationCapability
from src.runtime.trusted_v2_capabilities import TrustedV2CapabilityPorts
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator
from src.runtime.trusted_v2_r4 import (
    CandidateDirectR4Policy,
    R4RetrievalCapability,
    R4RetrievalRequest,
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
            str(lane): list(values) for lane, values in (lane_batches or {}).items()
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


class QuerySensitiveIndexReader(ScriptedIndexReader):
    """Small deterministic reader proving alias variants affect retrieval."""

    def search(
        self,
        lane: str,
        query: str,
        *,
        allowed_candidate_keys: set[str] | None = None,
        k: int = 50,
    ) -> list[CandidateSearchHit]:
        # A filing may call Revenue "Total net sales".  The broad raw query
        # intentionally returns a distractor; only the specific alias query
        # returns the canonical fact.
        key = "RIGHT" if "total net sales" in query.casefold() else "DISTRACTOR"
        if allowed_candidate_keys is not None and key not in allowed_candidate_keys:
            return []
        return [
            CandidateSearchHit(
                candidate_key=key,
                view_id=f"{lane}:{key}",
                lane=lane,
                bm25_rank=1 if "bm25" in lane else None,
                dense_rank=1 if "dense" in lane else None,
                bm25_score=1.0 if "bm25" in lane else None,
                dense_score=1.0 if "dense" in lane else None,
            )
        ][:k]


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
            BindingStatus.BOUND.value if not missing else BindingStatus.MISSING.value
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


class OverselectingBinderProvider(SelectingBinderProvider):
    """Return equivalent duplicate rows for each slot.

    Some OpenAI-compatible Binder responses select every duplicate table row
    instead of the one-fact-per-slot contract.  The capability adapter should
    repair this only when the candidate packet provides a strict consensus.
    """

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
            if matches:
                bindings[slot_id] = tuple(str(fact["fact_id"]) for fact in matches)
            else:
                missing.append(slot_id)
        status = (
            BindingStatus.BOUND.value if not missing else BindingStatus.MISSING.value
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


class WrongPeriodSelectionProvider:
    """Select a structurally valid but semantically wrong-period duplicate."""

    provider_name = "fixture"
    model_name = "deterministic-wrong-period"
    last_call = None

    def __init__(self, fact_id: str, *, current_id: str = "CURRENT-1") -> None:
        self.fact_id = fact_id
        self.current_id = current_id
        self.calls = 0

    def bind(self, request: Mapping[str, Any]) -> BinderProviderResult:
        self.calls += 1
        binding = EvidenceBinding(
            status=BindingStatus.BOUND.value,
            slot_bindings={
                "current": (self.current_id,),
                "previous": (self.fact_id,),
            },
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


def test_duplicate_bound_rows_are_repaired_from_packet_consensus() -> None:
    facts = {
        "E1": _fact("E1", value="391035"),
        "E2": _fact("E2", value="391035"),
    }
    provider = OverselectingBinderProvider()
    retrieval, binder, policy, reader, _ = _real_capabilities(
        [["E1", "E2"]],
        facts,
        provider,
    )
    query = "What was Apple FY2024 revenue?"
    outcome = asyncio.run(
        _coordinator(
            query,
            _plan(_slot("revenue")),
            retrieval,
            binder,
        ).execute(_request(query))
    )

    assert policy.calls == 1
    assert reader.search_calls > 0
    assert provider.calls == 1
    assert binder.last_semantic_repair is not None
    assert binder.last_semantic_repair["strategy"] == (
        "deterministic_candidate_consensus"
    )
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "DOWNSTREAM_EXECUTION_NOT_WIRED" in outcome.reason_codes
    assert outcome.evidence_ids == ["E1"]
    assert outcome.citation_ids == ["citation-E1"]


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


def test_semantic_firewall_repairs_wrong_period_selection_from_packet_consensus() -> (
    None
):
    facts = {
        "CURRENT-1": _fact("CURRENT-1", period="FY2025", value="416161"),
        "CURRENT-2": _fact("CURRENT-2", period="FY2025", value="416161"),
        "PREVIOUS-1": _fact("PREVIOUS-1", period="FY2024", value="391035"),
        "PREVIOUS-2": _fact("PREVIOUS-2", period="FY2024", value="391035"),
        # This candidate is structurally valid and has the current-period
        # label, but its extracted value belongs to the previous-period row.
        "WRONG-PERIOD": _fact("WRONG-PERIOD", period="FY2025", value="391035"),
    }
    provider = WrongPeriodSelectionProvider("WRONG-PERIOD")
    retrieval, binder, policy, reader, _ = _real_capabilities(
        [["CURRENT-1", "CURRENT-2", "PREVIOUS-1", "PREVIOUS-2", "WRONG-PERIOD"]],
        facts,
        provider,
    )
    query = "What was Apple FY2025 revenue compared with FY2024?"
    plan = _plan(
        _slot("current", period="FY2025", role="minuend"),
        _slot("previous", period="FY2024", role="subtrahend"),
        intent=Intent.CALCULATION,
        operation="difference",
    )
    outcome = asyncio.run(
        _coordinator(query, plan, retrieval, binder).execute(_request(query))
    )

    assert policy.calls == 1
    assert reader.search_calls > 0
    assert provider.calls == 1
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert outcome.evidence_ids == ["CURRENT-1", "PREVIOUS-1"]
    trace = binder.trace_snapshot()
    assert trace["bound_evidence_ids"] == ["CURRENT-1", "PREVIOUS-1"]
    repair = trace["binder_rounds"][0]["semantic_repair"]
    assert repair["strategy"] == "deterministic_candidate_consensus"
    assert repair["replaced_slot_bindings"]["previous"]["from"] == ["WRONG-PERIOD"]
    assert repair["replaced_slot_bindings"]["previous"]["to"] == ["PREVIOUS-1"]


def test_semantic_firewall_repairs_same_period_value_misalignment() -> None:
    facts = {
        "RIGHT-1": _fact("RIGHT-1", slots=("value",), value="100"),
        "RIGHT-2": _fact("RIGHT-2", slots=("value",), value="100"),
        # Metric and period are correct, but the value is from a neighboring
        # extracted column. A semantic period check alone cannot detect this.
        "WRONG-VALUE": _fact("WRONG-VALUE", slots=("value",), value="90"),
    }
    provider = SelectingBinderProvider(preferred_ids={"value": "WRONG-VALUE"})
    retrieval, binder, _, _, _ = _real_capabilities(
        [["WRONG-VALUE", "RIGHT-1", "RIGHT-2"]],
        facts,
        provider,
    )
    query = "What was Apple FY2024 revenue?"
    outcome = asyncio.run(
        _coordinator(query, _plan(_slot("value")), retrieval, binder).execute(
            _request(query)
        )
    )

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert outcome.evidence_ids == ["RIGHT-1"]
    repair = binder.trace_snapshot()["binder_rounds"][0]["semantic_repair"]
    assert repair["strategy"] == "deterministic_candidate_consensus"
    assert repair["replaced_slot_bindings"]["value"]["from"] == ["WRONG-VALUE"]
    assert repair["replaced_slot_bindings"]["value"]["to"] == ["RIGHT-1"]


def test_semantic_firewall_does_not_repair_conflicting_packet_values() -> None:
    facts = {
        "CURRENT": _fact("CURRENT", period="FY2025", value="100"),
        "PREVIOUS": _fact("PREVIOUS", period="FY2024", value="90"),
        "PREVIOUS-CONFLICT": _fact("PREVIOUS-CONFLICT", period="FY2024", value="80"),
        "WRONG": _fact("WRONG", period="FY2025", value="90"),
    }
    provider = WrongPeriodSelectionProvider("WRONG", current_id="CURRENT")
    retrieval, binder, _, _, _ = _real_capabilities(
        [["CURRENT", "PREVIOUS", "PREVIOUS-CONFLICT", "WRONG"]],
        facts,
        provider,
    )
    query = "What was Apple FY2025 revenue compared with FY2024?"
    plan = _plan(
        _slot("current", period="FY2025", role="minuend"),
        _slot("previous", period="FY2024", role="subtrahend"),
        intent=Intent.CALCULATION,
        operation="difference",
    )
    outcome = asyncio.run(
        _coordinator(query, plan, retrieval, binder).execute(_request(query))
    )

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "QUERY_EVIDENCE_SEMANTIC_MISMATCH" in outcome.reason_codes
    assert outcome.evidence_ids == []
    assert binder.trace_snapshot()["binder_rounds"][0]["semantic_repair"] is None


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


def test_ambiguous_binder_recovers_only_with_independent_exact_consensus() -> None:
    facts = {
        "E1": _fact("E1", value="391035"),
        "E2": _fact("E2", value="391035"),
    }
    retrieval, binder, _, _, provider = _real_capabilities(
        [["E1", "E2"]],
        facts,
        AmbiguousBinderProvider(),
    )
    query = "What was Apple FY2024 revenue?"

    outcome = asyncio.run(
        _coordinator(query, _plan(_slot("revenue")), retrieval, binder).execute(
            _request(query)
        )
    )

    assert provider.calls == 1
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert outcome.evidence_ids == ["E1"]
    assert "DOWNSTREAM_EXECUTION_NOT_WIRED" in outcome.reason_codes
    repair = binder.trace_snapshot()["binder_rounds"][0]["semantic_repair"]
    assert repair["strategy"] == "deterministic_ambiguous_packet_consensus"
    assert repair["consensus_size_by_slot"] == {"revenue": 2}


def test_ambiguous_binder_does_not_count_duplicate_physical_source_as_consensus() -> (
    None
):
    facts = {
        "E1": _fact("E1", value="391035"),
        "E2": _fact("E2", value="391035"),
    }
    facts["E2"]["physical_source_id"] = facts["E1"]["physical_source_id"]
    retrieval, binder, _, _, provider = _real_capabilities(
        [["E1", "E2"]],
        facts,
        AmbiguousBinderProvider(),
    )
    query = "What was Apple FY2024 revenue?"

    outcome = asyncio.run(
        _coordinator(query, _plan(_slot("revenue")), retrieval, binder).execute(
            _request(query)
        )
    )

    assert provider.calls == 1
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "EVIDENCE_CONFLICT" in outcome.reason_codes
    assert outcome.evidence_ids == []
    assert binder.trace_snapshot()["binder_rounds"][0]["semantic_repair"] is None


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


def test_r4_attaches_row_scope_and_promotes_total_revenue_for_unqualified_fact() -> (
    None
):
    class SourceAwareReader(ScriptedIndexReader):
        def __init__(self) -> None:
            super().__init__([["SEGMENT", "TOTAL"]])
            self.views = {}
            for lane in (
                "candidate_raw_bm25",
                "candidate_raw_dense",
                "candidate_structured_bm25",
                "candidate_structured_dense",
            ):
                self.views[(lane, "SEGMENT")] = {
                    "view_id": f"{lane}:SEGMENT",
                    "document_id": "msft_fy2025",
                    "retrieval_text": "| Productivity and Business Processes | $50,838 | Revenue | FY2024 |",
                    "metadata": {"metric_paths": ["Revenue"], "periods": ["FY2024"]},
                }
                self.views[(lane, "TOTAL")] = {
                    "view_id": f"{lane}:TOTAL",
                    "document_id": "msft_fy2025",
                    "retrieval_text": "| Total Revenue | $245,122 | FY2024 |",
                    "metadata": {"metric_paths": ["Total"], "periods": ["FY2024"]},
                }

        def view_for_candidate(
            self, lane: str, candidate_key: str
        ) -> Mapping[str, Any] | None:
            return self.views.get((lane, candidate_key))

    reader = SourceAwareReader()
    policy = CandidateDirectR4Policy(
        CandidateDirectRetriever(reader),
        materializer=lambda key: {
            "candidate_key": key,
            "candidate_id": key,
            "fact_id": key,
            "evidence_id": key,
            "metric": "Revenue" if key == "SEGMENT" else "Total",
            "period": "FY2024",
            "entity": "Microsoft",
            "value": "50,838" if key == "SEGMENT" else "245,122",
            "citation_id": f"citation-{key}",
            "provenance_complete": True,
            "physical_source_id": f"source-{key}",
        },
    )
    result = policy.retrieve(
        R4RetrievalRequest(
            request_id="scope-test",
            standalone_query="What was Microsoft FY2024 revenue?",
            plan=_plan(_slot("value")),
            reason_code="MISSING_SLOT",
        )
    )
    assert [item["candidate_key"] for item in result.candidate_evidence] == [
        "TOTAL",
        "SEGMENT",
    ]
    total = result.candidate_evidence[0]
    segment = result.candidate_evidence[1]
    assert total["metric"] == "Revenue"
    assert total["metric_normalization"] == "structured_total_revenue"
    assert total["scope"] == "consolidated"
    assert segment["scope"] == "segment"
    assert segment["scope_label"] == "Productivity and Business Processes"


def test_targeted_slot_pool_preserves_source_view_metadata() -> None:
    """Targeted Slot retrieval must retain scope metadata for Binder safety."""

    class SlotSourceReader(ScriptedIndexReader):
        def __init__(self) -> None:
            super().__init__([["TOTAL"]])

        def view_for_candidate(
            self,
            lane: str,
            candidate_key: str,
        ) -> Mapping[str, Any] | None:
            if candidate_key != "TOTAL":
                return None
            return {
                "view_id": f"{lane}:TOTAL",
                "document_id": "msft_fy2025",
                "retrieval_text": "| Total Revenue | $245,122 | FY2024 |",
                "metadata": {
                    "metric_paths": ["Total"],
                    "periods": ["FY2024"],
                },
            }

    policy = CandidateDirectR4Policy(
        CandidateDirectRetriever(SlotSourceReader()),
        materializer=lambda key: {
            "candidate_key": key,
            "candidate_id": key,
            "fact_id": key,
            "evidence_id": key,
            "metric": "Total",
            "period": "FY2024",
            "entity": "Microsoft",
            "value": "245,122",
            "citation_id": f"citation-{key}",
            "provenance_complete": True,
            "physical_source_id": f"source-{key}",
        },
    )

    result = policy.retrieve(
        R4RetrievalRequest(
            request_id="scope-targeted-test",
            standalone_query="What was Microsoft FY2024 revenue?",
            plan=_plan(_slot("value")),
            reason_code="MISSING_SLOT",
            target_slots=("value",),
        )
    )

    candidate = result.candidate_evidence[0]
    assert candidate["scope"] == "consolidated"
    assert candidate["metric"] == "Revenue"
    assert candidate["metric_normalization"] == "structured_total_revenue"
    assert candidate["retrieval_context"]["source_view_ids"]


def test_scope_phrase_variant_preserves_explicit_segment_label() -> None:
    query_plan = build_query_plan(
        "What was Microsoft FY2024 Productivity and Business Processes revenue?",
        ("msft_fy2025",),
    )
    raw_phrase = query_plan.operand_slots[0].raw_metric_phrase

    assert extract_scope_phrase(raw_phrase) == "productivity and business processes"
    variants = build_slot_query_variants(
        query_plan,
        {"raw_metric_phrase": raw_phrase, "period": "FY2024"},
    )
    assert any(
        variant.casefold().startswith("productivity and business processes |")
        for variant in variants
    )


def test_r4_does_not_reorder_explicit_segment_name_query() -> None:
    class SourceAwareReader(ScriptedIndexReader):
        def __init__(self) -> None:
            super().__init__([["SEGMENT", "TOTAL"]])

        def view_for_candidate(
            self, lane: str, candidate_key: str
        ) -> Mapping[str, Any] | None:
            if candidate_key == "SEGMENT":
                text = "| Productivity and Business Processes | $50,838 | Revenue | FY2024 |"
                paths = ["Revenue"]
            else:
                text = "| Total Revenue | $245,122 | FY2024 |"
                paths = ["Total"]
            return {
                "view_id": f"{lane}:{candidate_key}",
                "document_id": "msft_fy2025",
                "retrieval_text": text,
                "metadata": {"metric_paths": paths, "periods": ["FY2024"]},
            }

    policy = CandidateDirectR4Policy(
        CandidateDirectRetriever(SourceAwareReader()),
        materializer=lambda key: {
            "candidate_key": key,
            "candidate_id": key,
            "fact_id": key,
            "evidence_id": key,
            "metric": "Revenue" if key == "SEGMENT" else "Total",
            "period": "FY2024",
            "entity": "Microsoft",
            "value": "50,838" if key == "SEGMENT" else "245,122",
            "citation_id": f"citation-{key}",
            "provenance_complete": True,
            "physical_source_id": f"source-{key}",
        },
    )
    result = policy.retrieve(
        R4RetrievalRequest(
            request_id="scope-test-segment",
            standalone_query=(
                "What was Microsoft FY2024 Productivity and Business Processes revenue?"
            ),
            plan=_plan(_slot("value")),
            reason_code="MISSING_SLOT",
        )
    )
    assert [item["candidate_key"] for item in result.candidate_evidence] == [
        "SEGMENT",
        "TOTAL",
    ]


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
    coordinator = _coordinator(
        query,
        _plan(slot),
        retrieval,
        SemanticEvidenceEvaluationCapability(
            SemanticBinderService(SelectingBinderProvider())
        ),
    )
    outcome = asyncio.run(coordinator.execute(_request(query)))
    assert policy.calls == 1
    assert outcome.evidence_ids == ["E1"]


def test_targeted_query_does_not_repeat_metric_or_period_terms() -> None:
    facts = {"E1": _fact("E1")}
    retrieval, _, _, reader, _ = _real_capabilities([["E1"]], facts)
    query = "What was Apple FY2024 revenue?"
    outcome = asyncio.run(
        _coordinator(
            query,
            _plan(_slot("revenue")),
            retrieval,
            SemanticEvidenceEvaluationCapability(
                SemanticBinderService(SelectingBinderProvider())
            ),
        ).execute(_request(query))
    )

    assert outcome.evidence_ids == ["E1"]
    assert append_missing_query_terms(query, ["Revenue", "FY2024"]) == query
    assert all("revenue Revenue" not in item.casefold() for item in reader.seen_queries)
    assert all("fy2024 fy2024" not in item.casefold() for item in reader.seen_queries)


def test_slot_query_expands_only_safe_revenue_filing_aliases() -> None:
    plan = build_query_plan("What was Apple FY2024 revenue?", ())
    query = build_slot_query(
        plan,
        {"raw_metric_phrase": "revenue", "period": "FY2024"},
    )

    assert query.startswith("revenue | net sales | total net sales | total revenue")
    assert "FY2024" in query
    # Similar-looking but distinct financial concepts must not be expanded.
    assert "operating income" not in query
    assert "net income" not in query


def test_slot_query_variants_recover_canonical_filing_label() -> None:
    plan = build_query_plan("What was Apple FY2024 revenue?", ())
    variants = build_slot_query_variants(
        plan,
        {"raw_metric_phrase": "revenue", "period": "FY2024"},
    )

    assert variants[0].startswith("revenue | FY2024")
    assert any("total net sales" in item.casefold() for item in variants)
    assert any("aapl" in item.casefold() for item in variants)
    assert all("operating income" not in item.casefold() for item in variants)
    assert all("net income" not in item.casefold() for item in variants)


def test_alias_variant_search_is_fused_into_slot_pool() -> None:
    plan = build_query_plan("What was Apple FY2024 revenue?", ())
    reader = QuerySensitiveIndexReader([[]])
    retriever = CandidateDirectRetriever(reader, lane_k=10)

    result = retriever.retrieve(plan, document_scope=set())

    assert result["slot_pools"]["fact"][0].candidate_key == "RIGHT"
    assert "DISTRACTOR" in {hit.candidate_key for hit in result["rrf_hits"]}
    assert any(
        "total net sales" in query.casefold()
        for query in result["slot_query_variants"]["fact"]
    )


def test_supervisor_slot_id_maps_to_query_plan_slot_pool() -> None:
    facts = {
        "RIGHT": _fact("RIGHT"),
        "DISTRACTOR": _fact("DISTRACTOR"),
    }
    reader = QuerySensitiveIndexReader([[]])
    retriever = CandidateDirectRetriever(reader, lane_k=10)
    policy = CandidateDirectR4Policy(
        retriever,
        materializer=lambda key: facts[key],
    )
    result = policy.retrieve(
        R4RetrievalRequest(
            request_id="slot-map",
            standalone_query="What was Apple FY2024 revenue?",
            plan=_plan(_slot("revenue")),
            reason_code="MISSING_SLOT",
            target_slots=("revenue",),
        )
    )

    assert result.candidate_ids[0] == "RIGHT"
    assert result.source_branch_metadata["planner_slot_ids"] == ["fact"]


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
