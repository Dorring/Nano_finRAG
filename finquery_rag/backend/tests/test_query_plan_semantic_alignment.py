"""Regression tests for the deterministic Query -> Plan semantic gate."""

from __future__ import annotations

import asyncio
from typing import Any

from rag_v2.contracts import Action, Intent, SupervisorPlan
from rag_v2.supervisor import (
    BoundEvidenceAlignmentStatus,
    DeterministicFallbackProvider,
    SemanticAlignmentStatus,
    SupervisorService,
    UnknownSemanticPolicy,
    align_bound_evidence_to_query,
    align_query_to_plan,
    canonical_entity_id,
    canonical_operation_id,
    canonical_metric_id,
    canonical_period_id,
    extract_query_semantic_frame,
    metric_alias_registry,
)
from src.runtime import V2ExecutionRequest, V2ExecutionStatus
from src.runtime.trusted_v2_capabilities import TrustedV2CapabilityPorts
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator


def _plan(metric: str, *, period: str = "FY2024") -> SupervisorPlan:
    return SupervisorPlan.from_dict(
        {
            "intent": Intent.DIRECT_FACT.value,
            "required_slots": [
                {
                    "slot_id": "value",
                    "metric": metric,
                    "period": period,
                    "role": "value",
                    "value_type": "numeric",
                    "unit": None,
                },
            ],
            "operation": None,
            "next_action": Action.RETRIEVE.value,
        },
    )


def _request(query: str) -> V2ExecutionRequest:
    return V2ExecutionRequest(
        request_id="semantic-alignment-test",
        user_id="user-1",
        session_id="session-1",
        original_query=query,
        standalone_query=query,
    )


class _CountingRetrieval:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, action: Any, state: Any) -> list[dict[str, Any]]:
        del action, state
        self.calls += 1
        return []


def test_canonical_metric_vocabulary_keeps_operating_and_net_income_distinct() -> None:
    assert canonical_metric_id("operating income") == "operating_income"
    assert canonical_metric_id("net income") == "net_income"
    assert canonical_metric_id("operating income") != canonical_metric_id("net income")
    frame = extract_query_semantic_frame(
        "What was Apple FY2024 operating income?",
    )
    assert frame.metric_ids == ("operating_income",)


def test_metric_ontology_is_shared_and_returns_a_defensive_copy() -> None:
    registry = metric_alias_registry()
    assert "operating_income" in registry
    assert "operating income" in registry["operating_income"]
    assert "net income" in registry["net_income"]
    assert "net income" not in registry["operating_income"]
    registry["operating_income"] = ("incorrect alias",)
    assert canonical_metric_id("operating income") == "operating_income"


def test_period_vocabulary_is_normalized_without_changing_plan_contract() -> None:
    assert canonical_period_id("FY2024") == "FY2024"
    assert canonical_period_id("fiscal year 2024") == "FY2024"
    assert canonical_period_id("Q1 2024") == "FY2024-Q1"
    assert canonical_period_id("2024 Q1") == "FY2024-Q1"

    frame = extract_query_semantic_frame(
        "What was Apple fiscal year 2024 operating income?",
    )
    assert frame.period_ids == ("FY2024",)
    assert frame.to_dict()["period_mentions"][0]["period_id"] == "FY2024"


def test_explicit_metric_alias_aligns_to_canonical_plan() -> None:
    result = align_query_to_plan("What was Apple FY2024 sales?", _plan("revenue"))
    assert result.status is SemanticAlignmentStatus.ALIGNED
    assert result.allowed
    assert result.query_metric_ids == ("revenue",)
    assert result.plan_metric_ids == ("revenue",)


def test_explicit_period_mismatch_is_rejected_before_retrieval() -> None:
    result = align_query_to_plan(
        "What was Apple FY2024 revenue?",
        _plan("revenue", period="FY2023"),
    )

    assert result.status is SemanticAlignmentStatus.MISMATCH
    assert not result.allowed
    assert result.query_period_ids == ("FY2024",)
    assert result.plan_period_ids == ("FY2023",)
    assert "query_period_not_planned:FY2024" in result.mismatches


def test_derived_comparison_periods_do_not_trigger_false_mismatch() -> None:
    plan = SupervisorPlan.from_dict(
        {
            "intent": Intent.CALCULATION.value,
            "required_slots": [
                {
                    "slot_id": "current",
                    "metric": "revenue",
                    "period": "FY2024",
                    "role": "current",
                    "value_type": "numeric",
                    "unit": None,
                },
                {
                    "slot_id": "prior",
                    "metric": "revenue",
                    "period": "FY2023",
                    "role": "prior",
                    "value_type": "numeric",
                    "unit": None,
                },
            ],
            "operation": "growth_rate",
            "next_action": Action.RETRIEVE.value,
        },
    )

    result = align_query_to_plan(
        "Compare Apple FY2024 revenue with the previous year.",
        plan,
    )

    assert result.status is SemanticAlignmentStatus.ALIGNED
    assert result.query_period_ids == ("FY2024",)
    assert result.plan_period_ids == ("FY2024", "FY2023")


def test_wrong_metric_plan_is_rejected_before_retrieval() -> None:
    query = "What was Apple FY2024 operating income?"
    retrieval = _CountingRetrieval()
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(
            DeterministicFallbackProvider({query: _plan("net income")}),
        ),
        capabilities=TrustedV2CapabilityPorts(retrieval=retrieval),
    )

    outcome = asyncio.run(coordinator.execute(_request(query)))

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "QUERY_PLAN_SEMANTIC_MISMATCH" in outcome.reason_codes
    assert retrieval.calls == 0
    alignment = outcome.runtime_metadata["semantic_alignment"]
    assert alignment["status"] == SemanticAlignmentStatus.MISMATCH.value
    assert "planned_metric_not_in_query:net_income" in alignment["mismatches"]


def test_period_mismatch_coordinator_stops_before_retrieval() -> None:
    query = "What was Apple FY2024 revenue?"
    retrieval = _CountingRetrieval()
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(
            DeterministicFallbackProvider({query: _plan("revenue", period="FY2023")}),
        ),
        capabilities=TrustedV2CapabilityPorts(retrieval=retrieval),
    )

    outcome = asyncio.run(coordinator.execute(_request(query)))

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "QUERY_PLAN_SEMANTIC_MISMATCH" in outcome.reason_codes
    assert retrieval.calls == 0
    trace_alignment = outcome.debug_metadata["trace"]["semantic_alignment"]
    assert trace_alignment["status"] == SemanticAlignmentStatus.MISMATCH.value
    assert "query_period_not_planned:FY2024" in trace_alignment["mismatches"]


def test_aligned_semantics_are_preserved_in_runtime_metadata_and_trace() -> None:
    query = "What was Apple FY2024 revenue?"
    retrieval = _CountingRetrieval()
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(
            DeterministicFallbackProvider({query: _plan("revenue")}),
        ),
        capabilities=TrustedV2CapabilityPorts(retrieval=retrieval),
    )

    outcome = asyncio.run(coordinator.execute(_request(query)))

    alignment = outcome.runtime_metadata["semantic_alignment"]
    assert alignment["status"] == SemanticAlignmentStatus.ALIGNED.value
    assert alignment["query_period_ids"] == ["FY2024"]
    assert outcome.debug_metadata["trace"]["semantic_alignment"] == alignment


def test_strict_unknown_coordinator_fails_closed_before_retrieval() -> None:
    query = "What was the value in FY2024?"
    retrieval = _CountingRetrieval()
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(
            DeterministicFallbackProvider({query: _plan("revenue")}),
        ),
        capabilities=TrustedV2CapabilityPorts(retrieval=retrieval),
        unknown_semantic_policy=UnknownSemanticPolicy.STRICT_DIRECT_FACT,
    )

    outcome = asyncio.run(coordinator.execute(_request(query)))

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "QUERY_PLAN_SEMANTIC_UNKNOWN" in outcome.reason_codes
    assert retrieval.calls == 0
    alignment = outcome.runtime_metadata["semantic_alignment"]
    assert alignment["unknown_blocked"] is True
    assert outcome.debug_metadata["trace"]["semantic_alignment"] == alignment


def test_queries_without_a_recognized_metric_remain_backward_compatible() -> None:
    result = align_query_to_plan("Compare the two years", _plan("revenue"))
    assert result.status is SemanticAlignmentStatus.UNKNOWN
    assert result.allowed


def test_explicit_period_is_checked_even_when_metric_is_unknown() -> None:
    result = align_query_to_plan(
        "Compare FY2024",
        _plan("revenue", period="FY2023"),
    )

    assert result.status is SemanticAlignmentStatus.MISMATCH
    assert not result.allowed
    assert result.query_metric_ids == ()
    assert result.query_period_ids == ("FY2024",)
    assert "query_period_not_planned:FY2024" in result.mismatches


def test_strict_unknown_policy_blocks_direct_fact_without_guessing_metric() -> None:
    result = align_query_to_plan(
        "What was the value in FY2024?",
        _plan("revenue"),
        unknown_policy=UnknownSemanticPolicy.STRICT_DIRECT_FACT,
    )

    assert result.status is SemanticAlignmentStatus.UNKNOWN
    assert not result.allowed
    assert result.unknown_blocked
    assert result.unknown_query_fields == ("metric",)
    payload = result.to_dict()
    assert payload["unknown_policy"] == "strict_direct_fact"
    assert payload["unknown_blocked"] is True


def test_strict_unknown_policy_preserves_operation_only_calculation_queries() -> None:
    plan = SupervisorPlan.from_dict(
        {
            "intent": Intent.CALCULATION.value,
            "required_slots": [
                {
                    "slot_id": "current",
                    "metric": "revenue",
                    "period": "FY2024",
                    "role": "current",
                    "value_type": "numeric",
                    "unit": None,
                },
                {
                    "slot_id": "prior",
                    "metric": "revenue",
                    "period": "FY2023",
                    "role": "prior",
                    "value_type": "numeric",
                    "unit": None,
                },
            ],
            "operation": "growth_rate",
            "next_action": Action.RETRIEVE.value,
        },
    )

    result = align_query_to_plan(
        "Compare the two years.",
        plan,
        unknown_policy=UnknownSemanticPolicy.STRICT_DIRECT_FACT,
    )

    assert result.status is SemanticAlignmentStatus.UNKNOWN
    assert result.allowed
    assert not result.unknown_blocked


def test_multiple_explicit_metrics_are_ambiguous_for_a_single_fact_plan() -> None:
    result = align_query_to_plan(
        "What were Apple FY2024 revenue and net income?",
        _plan("revenue"),
    )

    assert result.status is SemanticAlignmentStatus.AMBIGUOUS
    assert not result.allowed
    assert result.ambiguous_query_fields == ("metric",)
    assert result.to_dict()["ambiguity_blocked"] is True


def test_explicit_operation_mismatch_is_rejected_before_retrieval() -> None:
    plan = SupervisorPlan.from_dict(
        {
            "intent": Intent.CALCULATION.value,
            "required_slots": [
                {
                    "slot_id": "current",
                    "metric": "revenue",
                    "period": "FY2024",
                    "role": "current",
                    "value_type": "numeric",
                    "unit": None,
                },
                {
                    "slot_id": "prior",
                    "metric": "revenue",
                    "period": "FY2023",
                    "role": "prior",
                    "value_type": "numeric",
                    "unit": None,
                },
            ],
            "operation": "difference",
            "next_action": Action.RETRIEVE.value,
        },
    )

    result = align_query_to_plan(
        "What was Apple's year-over-year revenue growth in FY2024?",
        plan,
    )

    assert canonical_operation_id("year over year") == "growth_rate"
    assert result.status is SemanticAlignmentStatus.MISMATCH
    assert "query_operation_not_planned:growth_rate" in result.mismatches


def test_optional_entity_expectation_rejects_topic_mismatch() -> None:
    result = align_query_to_plan(
        "What was Microsoft FY2024 revenue?",
        _plan("revenue"),
        semantic_context={"entity": "Apple"},
    )

    assert canonical_entity_id("AAPL") == "aapl"
    assert result.status is SemanticAlignmentStatus.MISMATCH
    assert "query_entity_not_expected:msft" in result.mismatches


def test_authorized_metric_and_period_expectations_reject_plan_drift() -> None:
    result = align_query_to_plan(
        "What was Apple FY2024 revenue?",
        _plan("net income", period="FY2023"),
        semantic_context={
            "resolved_metric": "Revenue",
            "resolved_period": "FY2024",
        },
    )

    assert result.status is SemanticAlignmentStatus.MISMATCH
    assert result.expected_metric_ids == ("revenue",)
    assert result.expected_period_ids == ("FY2024",)
    assert "planned_metric_not_expected:net_income" in result.mismatches
    assert "planned_period_not_expected:FY2023" in result.mismatches
    payload = result.to_dict()
    assert payload["expected_metric_ids"] == ["revenue"]
    assert payload["expected_period_ids"] == ["FY2024"]


def test_bound_evidence_cross_check_rejects_wrong_metric_and_entity() -> None:
    facts = [
        {
            "fact_id": "F1",
            "metric": "Net Income",
            "period": "FY2024",
            "entity": "Microsoft",
        },
    ]
    result = align_bound_evidence_to_query(
        "What was Apple FY2024 operating income?",
        _plan("operating income"),
        facts,
        {"value": ("F1",)},
    )

    assert result.status is BoundEvidenceAlignmentStatus.MISMATCH
    assert not result.allowed
    assert "fact_metric_not_matching_slot:F1:value" in result.mismatches
    assert "fact_metric_not_in_query:F1" in result.mismatches
    assert "fact_entity_not_in_query:F1" in result.mismatches


def test_ambiguous_plan_is_fail_closed_without_retrieval() -> None:
    query = "What were Apple FY2024 revenue and net income?"
    retrieval = _CountingRetrieval()
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(
            DeterministicFallbackProvider({query: _plan("revenue")}),
        ),
        capabilities=TrustedV2CapabilityPorts(retrieval=retrieval),
    )

    outcome = asyncio.run(coordinator.execute(_request(query)))

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "QUERY_PLAN_SEMANTIC_AMBIGUOUS" in outcome.reason_codes
    assert retrieval.calls == 0
    assert outcome.runtime_metadata["semantic_alignment"]["status"] == "AMBIGUOUS"
