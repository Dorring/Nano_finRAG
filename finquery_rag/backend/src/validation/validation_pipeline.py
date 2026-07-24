"""Grounded validation pipeline facade (Phase 4 Commit 7).

The ``GroundedValidationPipeline`` is the top-level facade that combines
the pre-generation ``AnswerabilityEvaluator`` with the post-generation
``ResponseValidator``. The RAG orchestrator delegates to this pipeline
to decide:

1. *Before* generation — whether the LLM may be invoked at all.
2. *After* generation — whether the generated answer is safe to return.

The pipeline is deterministic and never calls the LLM. It does not
perform retrieval or modify state.

Layer dependency: ``domain <- validation <- application``. This module
imports from ``src.domain`` and from sibling ``src.validation`` modules
only. It must NOT import from ``src.application``, ``src.services``,
or ``src.api``.
"""
from __future__ import annotations

from typing import Any

from src.domain.calculation import CalculationResult
from src.domain.evidence import EvidenceItem
from src.domain.validation import (
    AnswerabilityResult,
    GroundedResponseResult,
    ValidationResult,
)
from src.retrieval.context_builder import SufficiencyResult
from src.validation.answerability import AnswerabilityEvaluator
from src.validation.response_validator import ResponseValidator


class GroundedValidationPipeline:
    """Top-level facade combining answerability + response validation.

    The pipeline exposes two methods that the orchestrator calls at
    different stages:

    - :meth:`evaluate_answerability` — call BEFORE generation. Returns an
      ``AnswerabilityResult`` that tells the orchestrator whether the
      LLM may be invoked.
    - :meth:`validate_response` — call AFTER generation. Returns a
      ``ValidationResult`` that tells the orchestrator whether the
      answer is safe to return, repairable, or must be blocked.

    A convenience :meth:`build_grounded_response` combines an
    answerability result, a validation result, and the (possibly
    repaired / fallback) answer into a single ``GroundedResponseResult``
    suitable for the orchestrator's return value.
    """

    def __init__(self) -> None:
        self._answerability = AnswerabilityEvaluator()
        self._response_validator = ResponseValidator()

    # -----------------------------------------------------------------
    # Pre-generation
    # -----------------------------------------------------------------

    def evaluate_answerability(
        self,
        *,
        question: str,
        intent: str,
        evidence: tuple[EvidenceItem, ...],
        sufficiency_result: SufficiencyResult,
        calculation_result: CalculationResult | None,
        requested_documents: tuple[str, ...],
        has_query_matched_evidence: bool = False,
    ) -> AnswerabilityResult:
        """Run the pre-generation answerability gate.

        Returns an ``AnswerabilityResult``. The orchestrator MUST inspect
        ``status`` and:
        - ``NOT_ANSWERABLE``      -> do NOT invoke the LLM; return the
          deterministic refusal built from ``missing_requirements``.
        - ``CALCULATION_BLOCKED`` -> do NOT invoke the LLM; return the
          Phase 3 calculation safe-response.
        - ``ANSWERABLE`` / ``PARTIALLY_ANSWERABLE`` -> proceed to LLM.
        """
        return self._answerability.evaluate(
            question=question,
            intent=intent,
            evidence=evidence,
            sufficiency_result=sufficiency_result,
            calculation_result=calculation_result,
            requested_documents=requested_documents,
            has_query_matched_evidence=has_query_matched_evidence,
        )

    # -----------------------------------------------------------------
    # Post-generation
    # -----------------------------------------------------------------

    def validate_response(
        self,
        *,
        answer: str,
        intent: str,
        evidence: tuple[EvidenceItem, ...],
        calculation_result: CalculationResult | None,
        sources: tuple[dict[str, Any], ...] = (),
    ) -> ValidationResult:
        """Run the post-generation response validation.

        Returns a ``ValidationResult``. Never raises — internal errors
        produce ``ValidationStatus.FAILED`` (fail-closed).

        ``sources`` is the tuple of source objects returned to the API
        consumer. The CitationValidator uses it to verify chunk_id,
        document_name, and page consistency.

        The orchestrator MUST inspect ``status`` and:
        - ``PASSED``         -> return the answer as-is.
        - ``REPAIRABLE``     -> apply the deterministic repair (Commit 8)
          and return the repaired answer.
        - ``BLOCKED``        -> do NOT return the answer; use the safe
          fallback.
        - ``FAILED``         -> do NOT return the answer; use the safe
          fallback (fail-closed).
        - ``NOT_APPLICABLE`` -> return the answer as-is (e.g. conversation).
        """
        return self._response_validator.validate(
            answer=answer,
            intent=intent,
            evidence=evidence,
            calculation_result=calculation_result,
            sources=sources,
        )

    # -----------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------

    def build_grounded_response(
        self,
        *,
        answer: str,
        sources: tuple[dict[str, Any], ...],
        answerability: AnswerabilityResult | None,
        validation: ValidationResult | None,
        warnings: tuple[str, ...] = (),
    ) -> GroundedResponseResult:
        """Combine the pipeline outputs into a single ``GroundedResponseResult``.

        This is the value the orchestrator returns to the HTTP / SSE
        layer. The ``answer`` field is the final text that is safe to
        show the user (possibly repaired or fallback). ``sources`` is the
        evidence source list. ``warnings`` are non-blocking, user-facing
        notices.
        """
        return GroundedResponseResult(
            answer=answer,
            sources=sources,
            answerability=answerability,
            validation=validation,
            warnings=warnings,
        )
