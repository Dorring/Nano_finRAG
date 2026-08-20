"""Observation-driven, bounded action selection."""
from __future__ import annotations

from typing import Any, Mapping

from .adaptive_budget import AdaptiveRAGBudgetV1
from .adaptive_contracts import (
    AdaptiveRAGStateV1,
    EvidenceDecision,
    EvidenceEvaluationV1,
    ReasonCode,
    ReplanActionV1,
    ToolCapability,
)


class BoundedReplannerV1:
    """Map a concrete evaluator reason to one permitted capability."""

    _CAPABILITY_BY_REASON = {
        ReasonCode.MISSING_SLOT: ToolCapability.SEMANTIC_RETRIEVAL,
        ReasonCode.LOW_EVIDENCE_COVERAGE: ToolCapability.SEMANTIC_RETRIEVAL,
        ReasonCode.WRONG_PERIOD: ToolCapability.STRUCTURED_FINANCIAL_LOOKUP,
        ReasonCode.AMBIGUOUS_PERIOD: ToolCapability.DOCUMENT_METADATA_LOOKUP,
        ReasonCode.MISSING_OPERAND: ToolCapability.STRUCTURED_FINANCIAL_LOOKUP,
        ReasonCode.WRONG_ENTITY_SCOPE: ToolCapability.STRUCTURED_FINANCIAL_LOOKUP,
        ReasonCode.EVIDENCE_CONFLICT: ToolCapability.DOCUMENT_METADATA_LOOKUP,
        ReasonCode.VERSION_CONFLICT: ToolCapability.DOCUMENT_METADATA_LOOKUP,
        ReasonCode.MISSING_AUTHORITATIVE_VERSION: ToolCapability.DOCUMENT_METADATA_LOOKUP,
        ReasonCode.TOOL_ERROR: ToolCapability.LEXICAL_RETRIEVAL,
    }

    def __init__(self, budget: AdaptiveRAGBudgetV1 | None = None) -> None:
        self.budget = budget or AdaptiveRAGBudgetV1()

    def replan(
        self,
        state: AdaptiveRAGStateV1,
        evaluation: EvidenceEvaluationV1,
        *,
        history: list[Mapping[str, Any]] | None = None,
    ) -> ReplanActionV1 | None:
        if evaluation.decision is not EvidenceDecision.REPAIRABLE:
            return None
        if state.replan_rounds >= self.budget.max_replan_rounds:
            return None
        if ReasonCode.NO_PROGRESS in evaluation.reason_codes:
            return None
        reason = next((item for item in evaluation.reason_codes if item in self._CAPABILITY_BY_REASON), None)
        if reason is None:
            return None
        capability = self._CAPABILITY_BY_REASON[reason]
        # Never create a new tool name from a prompt.  The capability-to-tool
        # registry is supplied by the caller and may reject unavailable tools.
        constraint: dict[str, Any] = {}
        if reason in {ReasonCode.WRONG_PERIOD, ReasonCode.AMBIGUOUS_PERIOD}:
            constraint["period_constraint"] = [
                slot.get("period") for slot in state.required_slots if slot.get("period")
            ]
        if reason is ReasonCode.MISSING_OPERAND:
            constraint["operand_slots"] = list(evaluation.missing_slots)
        if reason is ReasonCode.WRONG_ENTITY_SCOPE:
            constraint["scope_constraint"] = [
                slot.get("scope") or slot.get("entity") for slot in state.required_slots
            ]
        query = state.normalized_query
        if reason in {ReasonCode.WRONG_PERIOD, ReasonCode.AMBIGUOUS_PERIOD}:
            query = f"{query} [period={','.join(map(str, constraint.get('period_constraint', [])))}]"
        return ReplanActionV1(
            capability=capability,
            query=query,
            reason_code=reason,
            target_slots=tuple(evaluation.missing_slots),
            constraints=constraint,
        )
