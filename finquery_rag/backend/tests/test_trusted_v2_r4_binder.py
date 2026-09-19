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
        selected_fact_ids: set[str] = set()
        for slot in request["required_slots"]:
            slot_id = str(slot["slot_id"])
            matches = [
                fact
                for fact in facts
                if str(fact.get("metric", "")).casefold()
                == str(slot.get("metric", "")).casefold()
                and str(fact.get("period", "")).casefold()
                == str(slot.get("period", "")).casefold()
            ]
            preferred = self.preferred_ids.get(slot_id)
            if preferred:
                matches = [
                    fact for fact in matches if str(fact.get("fact_id")) == preferred
                ] or matches
            matches = [
                fact
                for fact in matches
                if str(fact.get("fact_id")) not in selected_fact_ids
            ]
            if matches:
                fact_id = str(matches[0]["fact_id"])
                bindings[slot_id] = (fact_id,)
                selected_fact_ids.add(fact_id)
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
                if str(fact.get("metric", "")).casefold()
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


class PartiallyAmbiguousBinderProvider(SelectingBinderProvider):
    """Preserve one provider binding while marking another slot ambiguous."""

    def __init__(
        self,
        *,
        preserved_slot_id: str,
        preserved_fact_id: str,
        ambiguous_slot_id: str,
    ) -> None:
        super().__init__()
        self.preserved_slot_id = preserved_slot_id
        self.preserved_fact_id = preserved_fact_id
        self.ambiguous_slot_id = ambiguous_slot_id

    def bind(self, request: Mapping[str, Any]) -> BinderProviderResult:
        self.calls += 1
        binding = EvidenceBinding(
            status=BindingStatus.AMBIGUOUS.value,
            slot_bindings={self.preserved_slot_id: (self.preserved_fact_id,)},
            ambiguous_slots=(self.ambiguous_slot_id,),
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


class PartiallyAmbiguousDuplicateBinderProvider(PartiallyAmbiguousBinderProvider):
    """Return equal duplicate bindings for the otherwise non-ambiguous slot."""

    def __init__(
        self,
        *,
        preserved_slot_id: str,
        preserved_fact_ids: tuple[str, ...],
        ambiguous_slot_id: str,
    ) -> None:
        super().__init__(
            preserved_slot_id=preserved_slot_id,
            preserved_fact_id=preserved_fact_ids[0],
            ambiguous_slot_id=ambiguous_slot_id,
        )
        self.preserved_fact_ids = preserved_fact_ids

    def bind(self, request: Mapping[str, Any]) -> BinderProviderResult:
        result = super().bind(request)
        binding = EvidenceBinding(
            status=BindingStatus.AMBIGUOUS.value,
            slot_bindings={self.preserved_slot_id: self.preserved_fact_ids},
            ambiguous_slots=(self.ambiguous_slot_id,),
        )
        return BinderProviderResult(binding=binding, metadata=result.metadata)


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
    # H2A-2C-2: this used to assert a repair was *recorded*, because an
    # overselecting provider was by definition wrong -- BOUND permitted one fact
    # per slot, so two rows had to be collapsed into one.  It is no longer
    # wrong: E1 and E2 are two distinct physical sources stating 391035, so the
    # packet consensus for this slot *is* the pair, and reconciliation is
    # reached without changing anything.  `last_semantic_repair` staying None
    # means nothing needed correcting, not that nothing was checked.
    #
    # The safety intent is preserved and pinned below -- the bound set is the
    # consensus support set, not whatever the provider emitted.  The cases where
    # a row falls *outside* the consensus are the two
    # `test_semantic_firewall_repairs_*` tests beside this one.
    assert outcome.evidence_ids == ["E1", "E2"]
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "DOWNSTREAM_EXECUTION_NOT_WIRED" in outcome.reason_codes
    assert outcome.citation_ids == ["citation-E1", "citation-E2"]


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
    # H2A-2C-2: each slot keeps both of its independent supports.
    assert outcome.evidence_ids == [
        "CURRENT-1",
        "CURRENT-2",
        "PREVIOUS-1",
        "PREVIOUS-2",
    ]
    trace = binder.trace_snapshot()
    assert trace["bound_evidence_ids"] == [
        "CURRENT-1",
        "CURRENT-2",
        "PREVIOUS-1",
        "PREVIOUS-2",
    ]
    repair = trace["binder_rounds"][0]["semantic_repair"]
    assert repair["strategy"] == "deterministic_candidate_consensus"
    assert repair["replaced_slot_bindings"]["previous"]["from"] == ["WRONG-PERIOD"]
    assert repair["replaced_slot_bindings"]["previous"]["to"] == [
        "PREVIOUS-1",
        "PREVIOUS-2",
    ]


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
    # H2A-2C-2: both independent supports.
    assert outcome.evidence_ids == ["RIGHT-1", "RIGHT-2"]
    repair = binder.trace_snapshot()["binder_rounds"][0]["semantic_repair"]
    assert repair["strategy"] == "deterministic_candidate_consensus"
    assert repair["replaced_slot_bindings"]["value"]["from"] == ["WRONG-VALUE"]
    assert repair["replaced_slot_bindings"]["value"]["to"] == [
        "RIGHT-1",
        "RIGHT-2",
    ]


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
    # The reason code is now the precise one.  This fixture is two admissible
    # candidates for one slot that disagree -- PREVIOUS (FY2024, 90) and
    # PREVIOUS-CONFLICT (FY2024, 80) -- and H2A-1E added a same-slot conflict
    # gate, so it reports EVIDENCE_CONFLICT rather than the catch-all
    # QUERY_EVIDENCE_SEMANTIC_MISMATCH it used to share with every other
    # semantic failure.  Every other property of this test is unchanged and was
    # already true: it fails closed, binds nothing, and attempts no repair.
    # The three assertions below are the ones that carry the safety property.
    assert "EVIDENCE_CONFLICT" in outcome.reason_codes
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

    # H2A-2C-2: the independent consensus, not one representative of it.
    assert outcome.evidence_ids == ["E1", "E2", "E3"]
    trace = outcome.debug_metadata["trace"]
    assert set(trace["candidate_ids_per_round"][0]) == {"E1", "E2", "E3"}
    assert trace["bound_evidence_ids"] == ["E1", "E2", "E3"]


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
    # H2A-2C-2: both independent supports of the consensus.
    assert outcome.evidence_ids == ["E1", "E2"]
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


def test_partial_ambiguity_recovers_only_the_consensus_slot() -> None:
    facts = {
        "CURRENT-1": _fact("CURRENT-1", value="391035"),
        "CURRENT-2": _fact("CURRENT-2", value="391035"),
        "PRIOR-1": _fact("PRIOR-1", period="FY2023", value="383285"),
    }
    provider = PartiallyAmbiguousBinderProvider(
        preserved_slot_id="previous",
        preserved_fact_id="PRIOR-1",
        ambiguous_slot_id="current",
    )
    retrieval, binder, _, _, _ = _real_capabilities(
        [["CURRENT-1", "CURRENT-2", "PRIOR-1"]], facts, provider
    )
    query = "What was Apple's revenue growth from FY2023 to FY2024?"
    plan = _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("previous", period="FY2023", role="base"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )

    outcome = asyncio.run(
        _coordinator(query, plan, retrieval, binder).execute(_request(query))
    )

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    # H2A-2C-2: CURRENT keeps both independent supports; PRIOR is untouched.
    assert set(outcome.evidence_ids) == {"CURRENT-1", "CURRENT-2", "PRIOR-1"}
    assert "DOWNSTREAM_EXECUTION_NOT_WIRED" in outcome.reason_codes
    repair = binder.trace_snapshot()["binder_rounds"][0]["semantic_repair"]
    assert repair["strategy"] == "deterministic_partial_ambiguous_packet_consensus"
    assert repair["preserved_slot_bindings"] == {"previous": ["PRIOR-1"]}
    # The consensus slot is repaired to its whole winning group; the other
    # slot is left exactly as the provider bound it, which the
    # `preserved_slot_bindings` assertion above pins separately.
    assert repair["replaced_slot_bindings"] == {
        "current": {"to": ["CURRENT-1", "CURRENT-2"]}
    }
    assert repair["consensus_size_by_slot"] == {"current": 2}


def test_partial_ambiguity_normalizes_matching_duplicate_bound_slot() -> None:
    facts = {
        "CURRENT-1": _fact("CURRENT-1", value="391035"),
        "CURRENT-2": _fact("CURRENT-2", value="391035"),
        "PRIOR-1": _fact("PRIOR-1", period="FY2023", value="383285"),
        "PRIOR-2": _fact("PRIOR-2", period="FY2023", value="383285"),
    }
    provider = PartiallyAmbiguousDuplicateBinderProvider(
        preserved_slot_id="previous",
        preserved_fact_ids=("PRIOR-1", "PRIOR-2"),
        ambiguous_slot_id="current",
    )
    retrieval, binder, _, _, _ = _real_capabilities(
        [["CURRENT-1", "CURRENT-2", "PRIOR-1", "PRIOR-2"]], facts, provider
    )
    query = "What was Apple's revenue growth from FY2023 to FY2024?"
    plan = _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("previous", period="FY2023", role="base"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )

    outcome = asyncio.run(
        _coordinator(query, plan, retrieval, binder).execute(_request(query))
    )

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    # H2A-2C-2: PRIOR keeps both independent supports.
    assert set(outcome.evidence_ids) == {
        "CURRENT-1",
        "CURRENT-2",
        "PRIOR-1",
        "PRIOR-2",
    }
    repair = binder.trace_snapshot()["binder_rounds"][0]["semantic_repair"]
    assert repair["normalized_slot_bindings"] == {
        "previous": {"from": ["PRIOR-1", "PRIOR-2"], "to": ["PRIOR-1", "PRIOR-2"]}
    }
    assert repair["consensus_size_by_slot"] == {"current": 2, "previous": 2}


def test_partial_ambiguity_rejects_conflicting_duplicate_bound_slot() -> None:
    facts = {
        "CURRENT-1": _fact("CURRENT-1", value="391035"),
        "CURRENT-2": _fact("CURRENT-2", value="391035"),
        "PRIOR-1": _fact("PRIOR-1", period="FY2023", value="383285"),
        "PRIOR-2": _fact("PRIOR-2", period="FY2023", value="999999"),
    }
    provider = PartiallyAmbiguousDuplicateBinderProvider(
        preserved_slot_id="previous",
        preserved_fact_ids=("PRIOR-1", "PRIOR-2"),
        ambiguous_slot_id="current",
    )
    retrieval, binder, _, _, _ = _real_capabilities(
        [["CURRENT-1", "CURRENT-2", "PRIOR-1", "PRIOR-2"]], facts, provider
    )
    query = "What was Apple's revenue growth from FY2023 to FY2024?"
    plan = _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("previous", period="FY2023", role="base"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )

    outcome = asyncio.run(
        _coordinator(query, plan, retrieval, binder).execute(_request(query))
    )

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "EVIDENCE_CONFLICT" in outcome.reason_codes
    assert outcome.evidence_ids == []
    assert binder.trace_snapshot()["binder_rounds"][0]["semantic_repair"] is None


def test_partial_ambiguity_rejects_same_source_pseudo_consensus() -> None:
    facts = {
        "CURRENT-1": _fact("CURRENT-1", value="391035"),
        "CURRENT-2": _fact("CURRENT-2", value="391035"),
        "PRIOR-1": _fact("PRIOR-1", period="FY2023", value="383285"),
    }
    facts["CURRENT-2"]["physical_source_id"] = facts["CURRENT-1"][
        "physical_source_id"
    ]
    provider = PartiallyAmbiguousBinderProvider(
        preserved_slot_id="previous",
        preserved_fact_id="PRIOR-1",
        ambiguous_slot_id="current",
    )
    retrieval, binder, _, _, _ = _real_capabilities(
        [["CURRENT-1", "CURRENT-2", "PRIOR-1"]], facts, provider
    )
    query = "What was Apple's revenue growth from FY2023 to FY2024?"
    plan = _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("previous", period="FY2023", role="base"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )

    outcome = asyncio.run(
        _coordinator(query, plan, retrieval, binder).execute(_request(query))
    )

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


@pytest.mark.parametrize(
    ("alias_expansion", "expected"),
    [(False, "DISTRACTOR"), (True, "RIGHT")],
    ids=["arm-A-literal-query", "arm-B-slot-local-aliases"],
)
def test_a_targeted_slot_searches_for_its_own_requirement(
    alias_expansion: bool, expected: str
) -> None:
    """A targeted slot is queried from its own RequiredSlot, not the question.

    This replaces an assertion on ``source_branch_metadata["planner_slot_ids"]
    == ["fact"]`` -- the mapping from a Supervisor slot id onto the retrieval
    planner's own slot id.  That mapping existed only to undo the planner's
    slot-count guess and is gone, so what it was protecting is asserted
    directly: a targeted slot selects the candidate for *that* slot rather than
    drifting to the unscoped pool.

    ``QuerySensitiveIndexReader`` answers ``RIGHT`` only for a query containing
    "total net sales", which makes the two arms separate cleanly.  Arm A -- the
    literal ``Revenue FY2024`` -- never reaches it; Arm B does, by expanding the
    metric to its filing-label aliases while keeping the slot's own entity and
    period.  Both halves are pinned so that removing the expansion again cannot
    pass unnoticed, and so the alias capability is recorded as load-bearing
    rather than as legacy decoration.
    """

    facts = {
        "RIGHT": _fact("RIGHT"),
        "DISTRACTOR": _fact("DISTRACTOR"),
    }
    reader = QuerySensitiveIndexReader([[]])
    retriever = CandidateDirectRetriever(reader, lane_k=10)
    policy = CandidateDirectR4Policy(
        retriever,
        materializer=lambda key: facts[key],
        alias_expansion=alias_expansion,
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

    assert result.candidate_ids[0] == expected
    assert result.source_branch_metadata["slot_ids"] == ["revenue"]


def test_single_missing_slot_uses_canonical_slot_query_not_original_multi_query() -> None:
    """Recovery for net income must not repeatedly retrieve total net sales."""

    class MultiSlotReader(ScriptedIndexReader):
        def search(
            self,
            lane: str,
            query: str,
            *,
            allowed_candidate_keys: set[str] | None = None,
            k: int = 50,
        ) -> list[CandidateSearchHit]:
            key = "NET" if "net income" in query.casefold() and "total net sales" not in query.casefold() else "SALES"
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
            ]

    facts = {
        "SALES": _fact("SALES", metric="Total net sales"),
        "NET": _fact("NET", metric="Net income"),
    }
    policy = CandidateDirectR4Policy(
        CandidateDirectRetriever(MultiSlotReader([["SALES", "NET"]]), lane_k=10),
        materializer=lambda key: facts[key],
    )
    plan = _plan(
        _slot("sales", metric="Total net sales"),
        _slot("net_income", metric="Net income"),
        intent=Intent.MULTI_EVIDENCE,
    )

    result = policy.retrieve(
        R4RetrievalRequest(
            request_id="targeted-net-income",
            standalone_query="Compare Apple FY2024 total net sales and net income.",
            plan=plan,
            reason_code="MISSING_SLOT",
            target_slots=("net_income",),
        )
    )

    assert result.candidate_ids == ("NET",)
    assert result.source_branch_metadata["query"] == "Net income FY2024"

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
    # H2A-2C-2: the provider *prefers* SECONDARY and both lanes return it,
    # but PRIMARY is an equally independent source of the same fact, so the
    # bound set is the pair. The assertion above still pins what this test is
    # named for -- the candidate lanes are not the bound provenance.
    assert outcome.evidence_ids == ["PRIMARY", "SECONDARY"]


def test_facts_promotes_nested_metadata_qualifiers_without_overwriting_top_level() -> None:
    from rag_v2.adaptive import AdaptiveRAGStateV1

    plan = _plan(_slot("revenue"))
    state = AdaptiveRAGStateV1.new(
        request_id="req-metadata-promote",
        query="What was Apple revenue?",
        plan={"supervisor_plan": plan.to_dict()},
    )
    state.evidence_packets = [
        {
            "fact_id": "fact-1",
            "entity": "TopLevelEntity",
            "metadata": {
                "fact_id": "ignored-fact-id",
                "entity": "NestedEntity",
                "metric": "Net Sales",
                "normalized_metric": "net sales",
                "period": "FY2024",
                "currency": "USD",
                "scale": "million",
                "unit": "$",
                "scope": "consolidated",
                "statement_type": "income_statement",
            },
        }
    ]
    facts = SemanticEvidenceEvaluationCapability._facts(state)
    assert len(facts) == 1
    fact = facts[0]
    # Existing top-level keys must be preserved
    assert fact["entity"] == "TopLevelEntity"
    assert fact["fact_id"] == "fact-1"
    # Nested metadata keys must be promoted to top-level
    assert fact["metric"] == "Net Sales"
    assert fact["normalized_metric"] == "net sales"
    assert fact["period"] == "FY2024"
    assert fact["currency"] == "USD"
    assert fact["scale"] == "million"
    assert fact["unit"] == "$"
    assert fact["scope"] == "consolidated"
    assert fact["statement_type"] == "income_statement"


def test_repair_missing_binding_recovers_when_binder_returns_empty_tuple_slot_bindings() -> None:
    from rag_v2.adaptive import AdaptiveRAGStateV1

    class MissingEmptyTupleBinderProvider:
        provider_name = "fixture"
        model_name = "missing-empty-tuple"

        def bind(self, request: Mapping[str, Any]) -> BinderProviderResult:
            slots = [str(s["slot_id"]) for s in request["required_slots"]]
            return BinderProviderResult(
                binding=EvidenceBinding(
                    status=BindingStatus.MISSING.value,
                    slot_bindings={s: () for s in slots},
                    missing_slots=tuple(slots),
                ),
                metadata=BinderCallMetadata(
                    provider=self.provider_name,
                    model=self.model_name,
                    provider_role="evidence_binder",
                    model_role="deterministic_fixture",
                    latency_ms=0.1,
                    provider_response_success=True,
                    structured_output_success=True,
                ),
                raw_response={"status": "MISSING"},
            )

    fact_2023 = _fact(
        "REV_2023",
        period="FY2023",
        metric="Total net revenue",
        value="1000",
        slots=("revenue_2023",),
    )
    fact_2023["entity"] = "Apple"
    fact_2023["source"] = "10K-2023"
    fact_2023["physical_source_id"] = "10K-2023"

    fact_2024 = _fact(
        "REV_2024",
        period="FY2024",
        metric="Total net revenue",
        value="1200",
        slots=("revenue_2024",),
    )
    fact_2024["entity"] = "Apple"
    fact_2024["source"] = "10K-2024"
    fact_2024["physical_source_id"] = "10K-2024"

    plan = _plan(
        _slot("revenue_2023", metric="Total net revenue", period="FY2023"),
        _slot("revenue_2024", metric="Total net revenue", period="FY2024"),
        intent=Intent.CALCULATION,
        operation="difference",
    )

    binder = SemanticEvidenceEvaluationCapability(
        SemanticBinderService(MissingEmptyTupleBinderProvider()),
    )
    state = AdaptiveRAGStateV1.new(
        request_id="calc-recovery-empty-tuple",
        query="What was Apple net revenue change from FY2023 to FY2024?",
        intent="CALCULATION",
        plan={"supervisor_plan": plan.to_dict()},
    )
    state.evidence_packets = [fact_2023, fact_2024]

    evaluation = binder.evaluate(state)
    assert evaluation.decision.value == "SUFFICIENT"
    assert binder.last_run.binding.status == BindingStatus.BOUND.value
    assert binder.last_run.binding.slot_bindings == {
        "revenue_2023": ("REV_2023",),
        "revenue_2024": ("REV_2024",),
    }
    assert evaluation.supported_slots == ("revenue_2023", "revenue_2024")
    assert binder.last_bound_evidence_ids == ("REV_2023", "REV_2024")


def test_evaluate_does_not_mark_empty_tuple_slot_bindings_as_supported_slots() -> None:
    from rag_v2.adaptive import AdaptiveRAGStateV1

    class UnrecoverableMissingBinderProvider:
        provider_name = "fixture"
        model_name = "unrecoverable-missing"

        def bind(self, request: Mapping[str, Any]) -> BinderProviderResult:
            slots = [str(s["slot_id"]) for s in request["required_slots"]]
            return BinderProviderResult(
                binding=EvidenceBinding(
                    status=BindingStatus.MISSING.value,
                    slot_bindings={s: () for s in slots},
                    missing_slots=tuple(slots),
                ),
                metadata=BinderCallMetadata(
                    provider=self.provider_name,
                    model=self.model_name,
                    provider_role="evidence_binder",
                    model_role="deterministic_fixture",
                    latency_ms=0.1,
                    provider_response_success=True,
                    structured_output_success=True,
                ),
                raw_response={"status": "MISSING"},
            )

    fact_distractor = _fact(
        "DISTRACTOR",
        period="FY2020",
        metric="Other Expense",
        value="50",
    )
    plan = _plan(
        _slot("revenue_2099", metric="Total net revenue", period="FY2099"),
        intent=Intent.DIRECT_FACT,
    )
    binder = SemanticEvidenceEvaluationCapability(
        SemanticBinderService(UnrecoverableMissingBinderProvider()),
    )
    state = AdaptiveRAGStateV1.new(
        request_id="missing-with-distractor",
        query="What was Apple FY2099 revenue?",
        intent="DIRECT_FACT",
        plan={"supervisor_plan": plan.to_dict()},
    )
    state.evidence_packets = [fact_distractor]
    evaluation = binder.evaluate(state)
    assert evaluation.decision.value == "REPAIRABLE"
    assert evaluation.supported_slots == ()
    assert binder.last_bound_slot_bindings == {}
    trace = binder.trace_snapshot()
    assert trace["bound_slot_ids"] == []


def test_fact_value_key_normalizes_currency_footnotes_and_parenthesized_negatives() -> None:
    fact1 = {"value": "$ 177,556(g)", "unit": "USD", "currency": "USD"}
    fact2 = {"value": "177,556", "unit": "USD", "currency": "USD"}
    fact3 = {"value": "(3,037)", "unit": "USD", "currency": "USD"}
    fact4 = {"value": "-3037", "unit": "USD", "currency": "USD"}

    key1 = SemanticEvidenceEvaluationCapability._fact_value_key(fact1)
    key2 = SemanticEvidenceEvaluationCapability._fact_value_key(fact2)
    key3 = SemanticEvidenceEvaluationCapability._fact_value_key(fact3)
    key4 = SemanticEvidenceEvaluationCapability._fact_value_key(fact4)

    # The key is a private identity, not a published format, so what is pinned
    # here is what it must *say*: the two spellings are one quantity, and the
    # magnitude in the key is the one that was written -- 177,556 folded
    # exactly, and a parenthesised value folded to a negative.
    #
    # The separators and the trailing parts are not asserted: pinning them
    # pinned a layout, and the layout changed in H2A-2B when the key began
    # carrying the representation kind as well as the unit and currency.
    assert key1 == key2
    assert key3 == key4
    assert key1 is not None and key1.startswith("177556|")
    assert key3 is not None and key3.startswith("-3037|")
    assert key1 != key3


def test_slot_metric_matches_hardened_against_adversarial_subsets() -> None:
    from src.runtime.trusted_v2_r4 import _slot_metric_matches

    # 1. Valid compound plan phrase matching
    assert _slot_metric_matches(
        "total net revenue",
        "JPMorganChase total net revenue change from to",
    )
    assert _slot_metric_matches(
        "JPMorganChase total net revenue change from to",
        "total net revenue",
    )

    # 2. Adversarial metric subsets must NEVER match:
    # "income" must not match "net income"
    assert not _slot_metric_matches("income", "net income")
    assert not _slot_metric_matches("net income", "income")

    # "revenue" must not match "cost of revenue"
    assert not _slot_metric_matches("revenue", "cost of revenue")
    assert not _slot_metric_matches("cost of revenue", "revenue")

    # "operating income" must not match "net income"
    assert not _slot_metric_matches("operating income", "net income")
    assert not _slot_metric_matches("net income", "operating income")


def test_candidate_direct_r4_interleaves_multi_slot_pools_and_respects_entity_priority() -> None:
    from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever

    class MockSlotRetriever(CandidateDirectRetriever):
        """Two slot pools, merged the way production merges them.

        Overrides ``retrieve_for_requests``, which is now the seam the policy
        calls; it previously overrode ``retrieve(plan, ...)``, which the policy
        no longer calls at all.  The merge is the same ``build_slot_pool``
        production uses, so what this test still asserts about is the policy's
        entity-priority pass over a round-robin pool -- the property it was
        written for.

        Nothing here is weakened to make the test pass: the mocked pools, the
        expected counts and the entity-priority claim are unchanged.
        """

        def __init__(self) -> None:
            self.reader = None
            self.final_pool_k = 4

        def retrieve_for_requests(
            self,
            requests: Any,
            *,
            document_scope: Any = None,
            total_k: int | None = None,
            alias_expansion: bool = False,
        ) -> dict[str, Any]:
            from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit
            from src.pdf_retrieval_v4.candidate_slot_pool import build_slot_pool

            def hits(*keys: str) -> list[CandidateRRFHit]:
                return [
                    CandidateRRFHit(
                        candidate_key=key,
                        rrf_score=1.0 / (index + 1),
                        lane_ranks={"candidate_raw_bm25": index + 1},
                        supporting_view_ids={"candidate_raw_bm25": f"view:{key}"},
                    )
                    for index, key in enumerate(keys)
                ]

            pools = {
                "period_1": hits("PFIZER_2023_1", "JPM_2023_1", "JPM_2023_2"),
                "period_2": hits("PFIZER_2024_1", "JPM_2024_1"),
            }
            return {
                "candidate_direct_pool": build_slot_pool(
                    pools, total_k=total_k or max(80, self.final_pool_k * 2)
                ),
                "slot_pools": pools,
                "slot_query_variants": {},
                "lane_hits": {},
                "rrf_hits": [],
            }

    def mock_materializer(key: str) -> dict[str, Any]:
        entity = "JPMorgan" if "JPM" in key else "Pfizer"
        period = "FY2023" if "2023" in key else "FY2024"
        return {
            "candidate_key": key,
            "evidence_id": key,
            "fact_id": key,
            "metric": "Revenue",
            "period": period,
            "entity": entity,
            "scope": "consolidated",
            "value": "100",
            "unit": "USD",
            "currency": "USD",
            "provenance_complete": True,
        }

    plan = _plan(
        _slot("slot_1", metric="Revenue", period="FY2023"),
        _slot("slot_2", metric="Revenue", period="FY2024"),
        intent=Intent.CALCULATION,
        operation="difference",
    )

    policy = CandidateDirectR4Policy(
        MockSlotRetriever(),  # type: ignore[arg-type]
        materializer=mock_materializer,
    )
    result = policy.retrieve(
        R4RetrievalRequest(
            request_id="test-interleaving",
            standalone_query="How much did JPMorgan revenue change from FY2023 to FY2024?",
            plan=plan,
            reason_code="PLAN_REQUIRES_DETERMINISTIC_CALCULATION",
        )
    )

    cand_keys = [cand["candidate_key"] for cand in result.candidate_evidence]
    # JPMorgan candidates should take entity priority over Pfizer candidates,
    # and both slots should be present within the top 4
    assert len(cand_keys) == 4
    jpm_keys = [k for k in cand_keys if "JPM" in k]
    assert len(jpm_keys) == 3  # JPM_2023_1, JPM_2023_2, JPM_2024_1 must all be prioritized
    assert "JPM_2024_1" in cand_keys
    assert "JPM_2023_1" in cand_keys


def test_candidate_direct_r4_entity_priority_hardened_against_similar_entity_names() -> None:
    from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever

    class MockSingleSlotRetriever(CandidateDirectRetriever):
        """A single lane's candidates, in the shape the policy now consumes.

        Overrides ``retrieve_for_requests`` rather than ``retrieve(plan, ...)``,
        which the policy no longer calls.  The candidate keys and the assertion
        are unchanged -- only the seam moved.
        """

        def __init__(self) -> None:
            self.reader = None
            self.final_pool_k = 10

        def retrieve_for_requests(
            self,
            requests: Any,
            *,
            document_scope: Any = None,
            total_k: int | None = None,
            alias_expansion: bool = False,
        ) -> dict[str, Any]:
            return {
                "candidate_direct_pool": [
                    {"candidate_key": "HARTFORD_INSURANCE"},
                    {"candidate_key": "STANFORD_HEALTH"},
                    {"candidate_key": "FORD_MOTOR"},
                ],
                "slot_pools": {},
                "slot_query_variants": {},
                "lane_hits": {},
                "rrf_hits": [],
            }

    def mock_materializer(key: str) -> dict[str, Any]:
        entity_map = {
            "HARTFORD_INSURANCE": "Hartford Insurance",
            "STANFORD_HEALTH": "Stanford Healthcare",
            "FORD_MOTOR": "Ford Motor Co",
        }
        return {
            "candidate_key": key,
            "evidence_id": key,
            "fact_id": key,
            "metric": "Revenue",
            "period": "FY2024",
            "entity": entity_map[key],
            "scope": "consolidated",
            "value": "100",
            "unit": "USD",
            "currency": "USD",
            "provenance_complete": True,
        }

    plan = _plan(_slot("slot_1", metric="Revenue", period="FY2024"))
    policy = CandidateDirectR4Policy(
        MockSingleSlotRetriever(),  # type: ignore[arg-type]
        materializer=mock_materializer,
    )
    result = policy.retrieve(
        R4RetrievalRequest(
            request_id="test-entity-adversarial",
            standalone_query="What was Ford total revenue in FY2024?",
            plan=plan,
            reason_code="MISSING_SLOT",
        )
    )

    cand_keys = [cand["candidate_key"] for cand in result.candidate_evidence]
    # Ford Motor must be ranked first because Hartford and Stanford must NOT receive entity priority
    assert cand_keys[0] == "FORD_MOTOR"

    # Also test Apple vs Pineapple / Snapple
    class MockAppleRetriever(CandidateDirectRetriever):
        def __init__(self) -> None:
            self.reader = None
            self.final_pool_k = 10

        def retrieve_for_requests(
            self,
            requests: Any,
            *,
            document_scope: Any = None,
            total_k: int | None = None,
            alias_expansion: bool = False,
        ) -> dict[str, Any]:
            return {
                "candidate_direct_pool": [
                    {"candidate_key": "PINEAPPLE_INC"},
                    {"candidate_key": "SNAPPLE_BEVERAGE"},
                    {"candidate_key": "APPLE_INC"},
                ],
                "slot_pools": {},
                "slot_query_variants": {},
                "lane_hits": {},
                "rrf_hits": [],
            }

    def mock_apple_materializer(key: str) -> dict[str, Any]:
        entity_map = {
            "PINEAPPLE_INC": "Pineapple Inc",
            "SNAPPLE_BEVERAGE": "Snapple Beverage Corp",
            "APPLE_INC": "Apple Inc",
        }
        return {
            "candidate_key": key,
            "evidence_id": key,
            "fact_id": key,
            "metric": "Revenue",
            "period": "FY2024",
            "entity": entity_map[key],
            "scope": "consolidated",
            "value": "100",
            "unit": "USD",
            "currency": "USD",
            "provenance_complete": True,
        }

    policy_apple = CandidateDirectR4Policy(
        MockAppleRetriever(),  # type: ignore[arg-type]
        materializer=mock_apple_materializer,
    )
    result_apple = policy_apple.retrieve(
        R4RetrievalRequest(
            request_id="test-apple-adversarial",
            standalone_query="What was Apple revenue in FY2024?",
            plan=plan,
            reason_code="MISSING_SLOT",
        )
    )
    apple_cand_keys = [cand["candidate_key"] for cand in result_apple.candidate_evidence]
    # Apple Inc must be ranked first because Pineapple and Snapple must NOT receive entity priority
    assert apple_cand_keys[0] == "APPLE_INC"




