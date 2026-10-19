"""End-to-end integration tests for single-agent multi-skill workflow.

Validates the synergy between:
1. SourceRouter: dynamic intent & slot data source preference
2. DocumentHierarchyTree: 4-level 10-K hierarchy index and >80% candidate pruning
3. DiagnosticCritic: CRAG self-reflection and failure mode diagnosis
4. BoundedTrustedV2Coordinator: Single-agent fail-closed coordinator with Decimal validation
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any, Mapping

from rag_v2.adaptive import (
    AdaptiveRAGBudgetV1,
    AdaptiveRAGStateV1,
    DiagnosticCritic,
    CriticDiagnosisCategory,
    ReplanActionV1,
    ToolCapability,
)
from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.routing import DataSourceType, SourceRouter
from rag_v2.supervisor import (
    DeterministicFallbackProvider,
    PlanSemanticAlignment,
    SemanticAlignmentStatus,
    SupervisorService,
)
from src.domain.calculation import CalculationResult, CalculationStatus
from src.pdf_retrieval_v4.document_tree import DocumentHierarchyTree
from src.runtime.runtime_contract import ContextTrustLevel
from src.runtime.trusted_v2_capabilities import (
    CalculationCapability,
    EvidenceEvaluationCapability,
    GenerationCapability,
    ReleaseValidationCapability,
    RetrievalCapability,
    TrustedV2CapabilityPorts,
)
from src.runtime.trusted_v2_contracts import (
    V2ExecutionOutcome,
    V2ExecutionRequest,
    V2ExecutionStatus,
)
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator
from src.runtime.trusted_v2_generation import CandidateExecutionResult
from src.runtime.trusted_v2_validation import V2ValidationResult


class _MockRetrievalPort(RetrievalCapability):
    """Retrieval port powered by DocumentHierarchyTree with pruning telemetry."""

    def __init__(self, tree: DocumentHierarchyTree) -> None:
        self.tree = tree
        self.retrieval_calls: list[dict[str, Any]] = []

    def retrieve(self, action: ReplanActionV1, state: AdaptiveRAGStateV1) -> list[dict[str, Any]]:
        self.retrieval_calls.append({
            "query": action.query,
            "constraints": action.constraints,
            "round": state.replan_rounds,
        })
        section_hint = action.constraints.get("section_hint")
        pruned_chunks = self.tree.prune(section_hint=section_hint)

        # In round 0 (initial), if simulating a quarterly mismatch, return quarterly chunk
        if state.replan_rounds == 0 and "simulate_quarterly" in action.constraints:
            return [{
                "evidence_id": "chk_q3_ops",
                "metric": "Net Sales",
                "value": "85000",
                "period": "Three Months Ended Jun 2024",
                "source_text": "Three Months Ended June 29, 2024 Net sales $85,000",
                "title": "Consolidated Statements of Operations",
                "document_id": "AAPL-2024-10K",
            }]

        # Return relevant admitted chunk from pruned set
        if "chk_ops_annual" in pruned_chunks:
            return [{
                "evidence_id": "chk_ops_annual",
                "metric": "Net Sales",
                "value": "391035",
                "period": "FY2024",
                "source_text": "Years ended September 28, 2024 Total net sales $391,035",
                "title": "Consolidated Statements of Operations",
                "document_id": "AAPL-2024-10K",
            }]
        if "chk_note19_services" in pruned_chunks:
            return [{
                "evidence_id": "chk_note19_services",
                "metric": "Services Revenue",
                "value": "85200",
                "period": "FY2024",
                "source_text": "Note 19 Segment Information: Services net sales $85,200",
                "title": "Note 19 - Segment Information and Geographic Data",
                "document_id": "AAPL-2024-10K",
            }]

        return []


class _MockEvaluator(EvidenceEvaluationCapability):
    """Evaluator that admits only annual consolidated evidence."""

    def __init__(self) -> None:
        self.last_bound_evidence_ids: tuple[str, ...] = ()
        self.last_citation_ids: tuple[str, ...] = ()
        self.last_bound_slot_bindings: dict[str, tuple[str, ...]] = {}

    def evaluate(self, state: AdaptiveRAGStateV1) -> Any:
        packets = state.evidence_packets
        admitted = []
        for ev in packets:
            text = str(ev.get("source_text") or "")
            period = str(ev.get("period") or "")
            if "Three Months Ended" in text or "Three Months Ended" in period:
                # Reject quarterly evidence
                continue
            admitted.append(ev)

        missing = []
        for slot in state.required_slots:
            slot_id = slot["slot_id"]
            matched = any(
                slot_id in e.get("slots", ())
                or slot_id in e.get("slot_ids", ())
                or e.get("metric", "").lower() in slot.get("metric", "").lower()
                or slot.get("metric", "").lower() in e.get("metric", "").lower()
                for e in admitted
            )
            if not matched:
                missing.append(slot_id)


        from rag_v2.adaptive import EvidenceDecision, EvidenceEvaluationV1, ReasonCode
        if not missing and admitted:
            self.last_bound_evidence_ids = tuple(str(e.get("evidence_id")) for e in admitted)
            self.last_citation_ids = self.last_bound_evidence_ids
            self.last_bound_slot_bindings = {
                slot["slot_id"]: tuple(str(e.get("evidence_id")) for e in admitted)
                for slot in state.required_slots
            }
            return EvidenceEvaluationV1(
                decision=EvidenceDecision.SUFFICIENT,
                reason_codes=(),
                requested_slots=tuple(slot["slot_id"] for slot in state.required_slots),
                supported_slots=tuple(slot["slot_id"] for slot in state.required_slots),
                missing_slots=(),
                supporting_evidence_ids=self.last_bound_evidence_ids,
                temporal_status="CONSISTENT",
                calculation_ready=True,
            )
        else:
            self.last_bound_evidence_ids = ()
            self.last_citation_ids = ()
            self.last_bound_slot_bindings = {}
            return EvidenceEvaluationV1(
                decision=EvidenceDecision.REPAIRABLE,
                reason_codes=(ReasonCode.WRONG_PERIOD,),
                requested_slots=tuple(slot["slot_id"] for slot in state.required_slots),
                supported_slots=(),
                missing_slots=tuple(missing or [slot["slot_id"] for slot in state.required_slots]),
                supporting_evidence_ids=(),
                temporal_status="PERIOD_MISMATCH",
                calculation_ready=False,
            )


class _MockCalculator(CalculationCapability):
    def calculate(self, state: AdaptiveRAGStateV1) -> CalculationResult:
        return CalculationResult(
            calculation_id="calc_1",
            status=CalculationStatus.SUCCESS,
            value=Decimal("391035"),
            formula="slot_rev",
            operands={"slot_rev": Decimal("391035")},
        )


class _MockGenerator(GenerationCapability):
    def generate(self, state: AdaptiveRAGStateV1) -> str:
        return "Apple's total net sales in FY2024 was $391,035 million."


class _MockValidator(ReleaseValidationCapability):
    def validate(self, state: AdaptiveRAGStateV1, candidate: str) -> bool:
        return True


def _build_test_tree() -> DocumentHierarchyTree:
    tree = DocumentHierarchyTree(doc_id="AAPL-2024-10K")
    tree.index_chunk("chk_ops_annual", {
        "title": "Consolidated Statements of Operations",
        "statement_type": "income_statement",
        "text": "Years ended September 28, 2024 Total net sales $391,035",
    })
    tree.index_chunk("chk_bs", {
        "title": "Consolidated Balance Sheets",
        "statement_type": "balance_sheet",
        "text": "Total assets $364,980",
    })
    tree.index_chunk("chk_cf", {
        "title": "Consolidated Statements of Cash Flows",
        "statement_type": "cash_flow",
        "text": "Cash generated by operating activities",
    })
    tree.index_chunk("chk_note19_services", {
        "title": "Note 19 - Segment Information and Geographic Data",
        "text": "Services revenue $85,200",
    })
    tree.index_chunk("chk_note12_debt", {
        "title": "Note 12 - Debt",
        "text": "Term loan obligations $95,000",
    })
    tree.index_chunk("chk_mda_1", {
        "section": "Item 7",
        "title": "Management's Discussion and Analysis",
        "text": "Gross margin expanded due to product mix",
    })
    return tree


def _make_request(query: str, request_id: str = "req_test") -> V2ExecutionRequest:
    return V2ExecutionRequest(
        request_id=request_id,
        user_id="user_test",
        session_id="session_test",
        original_query=query,
        standalone_query=query,
    )


def test_single_agent_routing_and_tree_pruning_workflow() -> None:
    """Verify SourceRouter annotates plan, DocTree prunes 80% chunks, and calculation releases cleanly."""
    plan = SupervisorPlan(
        intent=Intent.DIRECT_FACT,
        required_slots=(
            RequiredSlot(
                slot_id="rev",
                metric="Total Net Sales",
                period="FY2024",
                role="value",
                value_type="numeric",
                unit=None,
                entity="Apple",
            ),
        ),
        operation=None,
        next_action=Action.RETRIEVE,
    )
    tree = _build_test_tree()
    retrieval = _MockRetrievalPort(tree)

    coordinator = BoundedTrustedV2Coordinator(
        supervisor=SupervisorService(
            DeterministicFallbackProvider({"What was Apple's total net sales in FY2024?": plan})
        ),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=_MockEvaluator(),
            calculation=_MockCalculator(),
            generation=_MockGenerator(),
            release_validator=_MockValidator(),
        ),
        source_router=SourceRouter,
        critic=DiagnosticCritic(),
        allow_test_release=True,
        alignment_override=PlanSemanticAlignment(
            status=SemanticAlignmentStatus.ALIGNED,
        ),
    )

    request = _make_request(
        query="What was Apple's total net sales in FY2024?",
        request_id="req_test_1",
    )

    outcome = asyncio.run(coordinator.execute(request))

    # 1. Verification of routing annotation
    assert len(outcome.evidence_ids) > 0
    assert "chk_ops_annual" in outcome.evidence_ids

    # 2. Verification of DocTree pruning in retrieval
    assert len(retrieval.retrieval_calls) >= 1
    first_call = retrieval.retrieval_calls[0]
    # Pruned down to Operations statements (1 chunk out of 6 chunks -> >80% pruning)
    assert first_call["constraints"].get("section_hint") == "Item 8. Consolidated Statements of Operations"
    pruned = tree.prune(first_call["constraints"].get("section_hint"))
    assert pruned == ["chk_ops_annual"]
    assert tree.pruning_ratio(pruned) >= 0.80

    # 3. Verification of final release
    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert outcome.answer == "Apple's total net sales in FY2024 was $391,035 million."


def test_single_agent_critic_reflection_and_self_correction() -> None:
    """Verify DiagnosticCritic diagnoses quarterly defect and triggers self-corrected round 2 retrieval."""
    plan = SupervisorPlan(
        intent=Intent.DIRECT_FACT,
        required_slots=(
            RequiredSlot(
                slot_id="rev",
                metric="Total Net Sales",
                period="FY2024",
                role="value",
                value_type="numeric",
                unit=None,
                entity="Apple",
            ),
        ),
        operation=None,
        next_action=Action.RETRIEVE,
    )
    tree = _build_test_tree()
    retrieval = _MockRetrievalPort(tree)

    # In round 0, instruct retrieval mock to simulate returning quarterly evidence
    def custom_retrieve(action: ReplanActionV1, state: AdaptiveRAGStateV1) -> list[dict[str, Any]]:
        retrieval.retrieval_calls.append({
            "query": action.query,
            "constraints": action.constraints,
            "round": state.replan_rounds,
        })
        if state.replan_rounds == 0:
            return [{
                "evidence_id": "chk_q3_ops",
                "metric": "Total Net Sales",
                "value": "85000",
                "period": "Three Months Ended Jun 2024",
                "source_text": "Three Months Ended June 29, 2024 Net sales $85,000",
                "title": "Consolidated Statements of Operations",
                "document_id": "AAPL-2024-10K",
            }]
        # In round 1, critic has guided retrieval to annual audited statement
        return [{
            "evidence_id": "chk_ops_annual",
            "metric": "Total Net Sales",
            "value": "391035",
            "period": "FY2024",
            "source_text": "Years ended September 28, 2024 Total net sales $391,035",
            "title": "Consolidated Statements of Operations",
            "document_id": "AAPL-2024-10K",
        }]

    retrieval.retrieve = custom_retrieve  # type: ignore

    coordinator = BoundedTrustedV2Coordinator(
        supervisor=SupervisorService(
            DeterministicFallbackProvider({"What was Apple's total net sales in FY2024?": plan})
        ),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=_MockEvaluator(),
            calculation=_MockCalculator(),
            generation=_MockGenerator(),
            release_validator=_MockValidator(),
        ),
        source_router=SourceRouter,
        critic=DiagnosticCritic(),
        allow_test_release=True,
        alignment_override=PlanSemanticAlignment(
            status=SemanticAlignmentStatus.ALIGNED,
        ),
    )

    request = _make_request(
        query="What was Apple's total net sales in FY2024?",
        request_id="req_test_critic",
    )

    outcome = asyncio.run(coordinator.execute(request))

    # Should succeed after self-correction
    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert "chk_ops_annual" in outcome.evidence_ids

    # Verify that Critic was invoked and diagnosed QUARTERLY_VS_ANNUAL
    assert len(retrieval.retrieval_calls) == 2
    round2_call = retrieval.retrieval_calls[1]
    diagnosis = round2_call["constraints"].get("critic_diagnosis")
    assert diagnosis is not None
    assert diagnosis["category"] == CriticDiagnosisCategory.QUARTERLY_VS_ANNUAL.value
    assert "annual consolidated" in round2_call["query"]


def test_single_agent_segment_footnote_routing() -> None:
    """Verify segment queries route to FOOTNOTE_DISCLOSURES and locate Note 19."""
    plan = SupervisorPlan(
        intent=Intent.DIRECT_FACT,
        required_slots=(
            RequiredSlot(
                slot_id="srv",
                metric="Services revenue",
                period="FY2024",
                role="value",
                value_type="numeric",
                unit=None,
                entity="Apple",
                scope="segment",
            ),
        ),
        operation=None,
        next_action=Action.RETRIEVE,
    )
    tree = _build_test_tree()
    retrieval = _MockRetrievalPort(tree)

    coordinator = BoundedTrustedV2Coordinator(
        supervisor=SupervisorService(
            DeterministicFallbackProvider({"What was Apple's Services segment revenue in FY2024?": plan})
        ),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=_MockEvaluator(),
            calculation=_MockCalculator(),
            generation=_MockGenerator(),
            release_validator=_MockValidator(),
        ),
        source_router=SourceRouter,
        allow_test_release=True,
        alignment_override=PlanSemanticAlignment(
            status=SemanticAlignmentStatus.ALIGNED,
        ),
    )

    request = _make_request(
        query="What was Apple's Services segment revenue in FY2024?",
        request_id="req_test_segment",
    )

    outcome = asyncio.run(coordinator.execute(request))
    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert "chk_note19_services" in outcome.evidence_ids

    first_call = retrieval.retrieval_calls[0]
    assert "Note" in first_call["constraints"].get("section_hint", "")
    assert "Segment" in first_call["constraints"].get("section_hint", "")
    pruned = tree.prune(first_call["constraints"].get("section_hint"))
    assert pruned == ["chk_note19_services"]


def test_single_agent_multi_source_sql_and_market_end_to_end() -> None:
    """Verify single agent Coordinator dispatches to SQL wide table and Market API simultaneously."""
    from rag_v2.sources import MultiSourceRetrievalAdapter, SandboxMarketProvider, SqlFactStore

    query = "What was Apple's FY2024 total revenue and what is Apple's current PE ratio?"
    plan = SupervisorPlan(
        intent=Intent.DIRECT_FACT,
        required_slots=(
            RequiredSlot(
                slot_id="s_rev",
                metric="Total Net Sales",
                period="FY2024",
                role="value",
                value_type="numeric",
                unit="USD",
                entity="Apple",
                scope="consolidated",
                source_preference=DataSourceType.STRUCTURED_STORE.value,
            ),
            RequiredSlot(
                slot_id="s_pe",
                metric="P/E",
                period="REALTIME",
                role="value",
                value_type="numeric",
                unit="ratio",
                entity="Apple",
                scope="consolidated",
                source_preference=DataSourceType.MARKET_API.value,
            ),
        ),
        operation=None,
        next_action=Action.RETRIEVE,
    )

    sql_store = SqlFactStore(":memory:")
    sql_store.seed_benchmark_data()
    market_provider = SandboxMarketProvider()
    tree = _build_test_tree()
    pdf_retriever = _MockRetrievalPort(tree)

    multi_adapter = MultiSourceRetrievalAdapter(
        sql_store=sql_store,
        market_provider=market_provider,
        pdf_retriever=pdf_retriever,
    )

    coordinator = BoundedTrustedV2Coordinator(
        supervisor=SupervisorService(
            DeterministicFallbackProvider({query: plan})
        ),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=multi_adapter,
            evidence_evaluator=_MockEvaluator(),
            calculation=_MockCalculator(),
            generation=_MockGenerator(),
            release_validator=_MockValidator(),
        ),
        source_router=SourceRouter,
        allow_test_release=True,
        alignment_override=PlanSemanticAlignment(
            status=SemanticAlignmentStatus.ALIGNED,
        ),
    )

    request = _make_request(query=query, request_id="req_test_multi_source")
    outcome = asyncio.run(coordinator.execute(request))

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    evidence_ids = outcome.evidence_ids
    assert any("sql_fact_" in eid for eid in evidence_ids)
    assert any("market_quote_" in eid for eid in evidence_ids)
    assert multi_adapter.last_dispatch_stats["sql_hits"] == 1
    assert multi_adapter.last_dispatch_stats["market_hits"] == 1

    sql_store.close()

