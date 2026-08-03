"""Calculation pipeline: orchestrates the deterministic calculation flow.

The pipeline ties together the operation router, evidence extractor, plan
builder, executor, and renderer into a single ``try_calculate`` entry
point. It is called by the RAG orchestrator after retrieval and context
build, before the LLM/deterministic answer extractor.

Flow:
1. Route the question through ``route_calculation``.
   - NOT_APPLICABLE -> return ``NOT_APPLICABLE_RESULT`` (orchestrator
     continues the normal RAG flow).
2. Extract operands from retrieved evidence via ``extract_operands``.
3. Build a ``CalculationPlan`` from the routing decision and extraction
   result. If required operands are missing, the plan is BLOCKED.
4. Execute the plan via ``execute_plan``.
5. Return the ``CalculationResult``. The orchestrator inspects the
   status to decide whether to bypass the LLM.

The pipeline is purely deterministic: no LLM calls, no retrieval, no
side effects. It imports from ``src.domain`` and ``src.finance`` only.

Layer dependency: ``domain -> finance -> application -> services``.
"""

from __future__ import annotations

from typing import Any

from src.domain.calculation import (
    NOT_APPLICABLE_RESULT,
    CalculationOperation,
    CalculationPlan,
    CalculationResult,
    CalculationStatus,
)
from src.domain.evidence import EvidenceItem
from src.finance.calculation_executor import execute_plan
from src.finance.calculation_registry import get_operation_entry
from src.finance.evidence_extractor import extract_operands
from src.finance.calculation_intent import detect_calculation_intent
from src.finance.operation_router import RoutingDecision, route_calculation
from src.finance.structured_operand_binding import (
    adapt_bound_operands,
    bind_operands,
    build_operand_specs,
    extract_financial_facts,
)

ENABLE_STRUCTURED_OPERAND_BINDING = False


class CalculationPipeline:
    """Orchestrates the deterministic financial calculation pipeline.

    The pipeline is stateless and safe to call concurrently. The
    orchestrator constructs one instance and calls ``try_calculate``
    on every query that requires retrieval.
    """

    def __init__(
        self,
        *,
        allow_derived_document_qa: bool = False,
        enable_structured_operand_binding: bool = ENABLE_STRUCTURED_OPERAND_BINDING,
    ):
        self._allow_derived_document_qa = allow_derived_document_qa
        self._enable_structured_operand_binding = enable_structured_operand_binding

    def try_calculate(
        self,
        question: str,
        intent: dict[str, Any],
        evidence: tuple[EvidenceItem, ...],
    ) -> CalculationResult:
        """Attempt a deterministic calculation for the given question.

        Args:
            question: The user's (possibly rewritten) question text.
            intent: The intent dict returned by ``classify_query_intent``.
            evidence: The retrieved evidence items from the RAG pipeline.

        Returns:
            A ``CalculationResult``. The orchestrator inspects
            ``result.status``:
            - ``EXECUTED``  -> bypass LLM, render the result as the answer.
            - ``BLOCKED``   -> bypass LLM, render a deterministic refusal.
            - ``NOT_APPLICABLE`` -> continue the normal RAG flow.
            - ``FAILED``    -> bypass LLM, return safe failure message
              (never fall back to LLM for numeric hallucination safety).
        """
        # 1. Route the question through the conservative 3-gate router.
        routing = route_calculation(
            question,
            intent,
            allow_derived_document_qa=self._allow_derived_document_qa,
        )
        if routing.status is CalculationStatus.NOT_APPLICABLE:
            return NOT_APPLICABLE_RESULT

        # 2. Extract numeric operands from retrieved evidence.
        extraction = extract_operands(evidence, routing)

        # 3. Build a CalculationPlan from the routing decision and extraction.
        plan = self._build_plan(routing, extraction)

        # 4. Execute the plan deterministically.
        return execute_plan(plan)

    def try_structured_shadow(
        self,
        question: str,
        intent: dict[str, Any],
        evidence: tuple[EvidenceItem, ...],
    ) -> CalculationResult:
        """Evaluate structured binding without changing the production result."""
        if not self._enable_structured_operand_binding:
            return NOT_APPLICABLE_RESULT
        routing = route_calculation(
            question,
            intent,
            allow_derived_document_qa=self._allow_derived_document_qa,
        )
        if (
            routing.status is CalculationStatus.NOT_APPLICABLE
            or routing.operation is None
        ):
            return NOT_APPLICABLE_RESULT
        calculation_intent = detect_calculation_intent(question)
        specs = build_operand_specs(
            question=question,
            routing_decision=routing,
            calculation_intent=calculation_intent,
        )
        if not specs:
            return execute_plan(
                CalculationPlan(
                    operation=routing.operation,
                    operands=(),
                    formula_version=routing.formula_version or "unknown.v1",
                    target_metric=routing.metric or routing.operation.value,
                    status=CalculationStatus.BLOCKED,
                    block_reason="OPERAND_MISSING: no structured operand specification",
                )
            )
        binding = bind_operands(specs, extract_financial_facts(evidence))
        if not binding.success:
            return execute_plan(
                CalculationPlan(
                    operation=routing.operation,
                    operands=(),
                    formula_version=routing.formula_version or "unknown.v1",
                    target_metric=routing.metric or routing.operation.value,
                    status=CalculationStatus.BLOCKED,
                    block_reason=f"{binding.block_reason}: structured binding blocked",
                )
            )
        operands = adapt_bound_operands(binding.operands, routing.operation.value)
        return execute_plan(
            CalculationPlan(
                operation=routing.operation,
                operands=operands,
                formula_version=routing.formula_version or "unknown.v1",
                target_metric=routing.metric or routing.operation.value,
            )
        )

    @staticmethod
    def _build_plan(
        routing: RoutingDecision,
        extraction: Any,
    ) -> CalculationPlan:
        """Build a ``CalculationPlan`` from routing and extraction results.

        If required operand roles are missing, the plan is created with
        status ``BLOCKED`` so the executor returns a deterministic refusal
        without invoking any primitive.
        """
        operation = routing.operation or CalculationOperation.SUM
        formula_version = routing.formula_version or "unknown.v1"
        target_metric = routing.metric or operation.value

        # SCALE_CONVERSION needs special handling for source/target scale.
        if operation is CalculationOperation.SCALE_CONVERSION:
            return CalculationPipeline._build_scale_conversion_plan(
                routing, extraction, formula_version, target_metric
            )

        # If the routing expected specific roles but some are missing,
        # block the plan immediately.
        if extraction.expected_roles and extraction.missing_roles:
            missing = ", ".join(extraction.missing_roles)
            return CalculationPlan(
                operation=operation,
                operands=extraction.operands,
                formula_version=formula_version,
                target_metric=target_metric,
                status=CalculationStatus.BLOCKED,
                block_reason=f"missing operands: {missing}",
            )

        # For generic operations (no fixed roles), check that the extractor
        # found at least one operand. If not, block with a clear reason.
        if not extraction.expected_roles and not extraction.operands:
            entry = get_operation_entry(operation)
            min_needed = entry.min_operands if entry else 1
            return CalculationPlan(
                operation=operation,
                operands=(),
                formula_version=formula_version,
                target_metric=target_metric,
                status=CalculationStatus.BLOCKED,
                block_reason=(
                    f"no operands extracted for generic operation "
                    f"'{operation.value}' (requires at least {min_needed})"
                ),
            )
        return CalculationPlan(
            operation=operation,
            operands=extraction.operands,
            formula_version=formula_version,
            target_metric=target_metric,
        )

    @staticmethod
    def _build_scale_conversion_plan(
        routing: RoutingDecision,
        extraction: Any,
        formula_version: str,
        target_metric: str,
    ) -> CalculationPlan:
        """Build a plan for SCALE_CONVERSION with source/target scale."""
        from src.domain.calculation import CalculationOperation

        operation = CalculationOperation.SCALE_CONVERSION
        target_scale = routing.target_scale

        if target_scale == "__CURRENCY__":
            return CalculationPlan(
                operation=operation,
                operands=extraction.operands,
                formula_version=formula_version,
                target_metric=target_metric,
                target_scale=target_scale,
                status=CalculationStatus.BLOCKED,
                block_reason="CURRENCY_NOT_SUPPORTED: currency conversion is not supported",
            )

        if not target_scale:
            return CalculationPlan(
                operation=operation,
                operands=extraction.operands,
                formula_version=formula_version,
                target_metric=target_metric,
                status=CalculationStatus.BLOCKED,
                block_reason="UNIT_AMBIGUOUS: target scale not specified in question",
            )

        if not extraction.source_scale:
            return CalculationPlan(
                operation=operation,
                operands=extraction.operands,
                formula_version=formula_version,
                target_metric=target_metric,
                target_scale=target_scale,
                status=CalculationStatus.BLOCKED,
                block_reason="UNIT_AMBIGUOUS: source scale could not be inferred from evidence",
            )

        if not extraction.operands:
            return CalculationPlan(
                operation=operation,
                operands=(),
                formula_version=formula_version,
                target_metric=target_metric,
                target_scale=target_scale,
                source_scale=extraction.source_scale,
                status=CalculationStatus.BLOCKED,
                block_reason="INSUFFICIENT_OPERANDS: no numeric value found in evidence",
            )

        return CalculationPlan(
            operation=operation,
            operands=extraction.operands,
            formula_version=formula_version,
            target_metric=target_metric,
            source_scale=extraction.source_scale,
            target_scale=target_scale,
        )
