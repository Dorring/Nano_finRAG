"""Pre-generation answerability evaluation (Phase 4).

The ``AnswerabilityEvaluator`` decides whether the RAG orchestrator is
allowed to invoke the LLM at all. It runs *before* generation and produces
one of four verdicts:

- ``ANSWERABLE``           -> evidence is sufficient; generation allowed.
- ``PARTIALLY_ANSWERABLE`` -> some requested documents are missing but
  enough evidence exists for a limited answer.
- ``NOT_ANSWERABLE``       -> evidence is missing or irrelevant; the LLM
  must NOT be invoked and a deterministic refusal is returned.
- ``CALCULATION_BLOCKED``  -> the deterministic calculation pipeline
  returned BLOCKED / FAILED; the LLM must NOT be invoked and the Phase 3
  safe response is returned.

The evaluator is purely deterministic — no LLM, no retrieval, no side
effects. It inspects the sufficiency result, the calculation result, the
evidence set, and the requested documents.

Layer dependency: ``domain <- validation``. This module imports only from
``src.domain`` and stdlib.
"""
from __future__ import annotations

from src.domain.calculation import CalculationResult, CalculationStatus
from src.domain.evidence import EvidenceItem
from src.domain.validation import AnswerabilityResult, AnswerabilityStatus
from src.retrieval.context_builder import SufficiencyResult
from src.validation.validation_policy import get_policy_for_intent


# Reason codes (machine-readable, used in trace and for orchestrator logic).
REASON_CALCULATION_BLOCKED = "calculation_blocked"
REASON_CALCULATION_FAILED = "calculation_failed"
REASON_NO_EVIDENCE = "no_evidence"
REASON_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
REASON_MISSING_DOCUMENTS = "missing_documents"
REASON_NO_RETRIEVAL_REQUIRED = "no_retrieval_required"
REASON_QUERY_MATCHED_EVIDENCE = "query_matched_evidence"


class AnswerabilityEvaluator:
    """Deterministic pre-generation answerability gate.

    The evaluator never calls the LLM and never performs retrieval. It
    inspects the already-computed sufficiency result, calculation result,
    and evidence set to decide whether generation should proceed.
    """

    def evaluate(
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
        """Evaluate whether the question can be answered with the available evidence.

        Parameters
        ----------
        question:
            The user's question (unused for logic but available for
            future extension; currently not inspected to remain
            deterministic and avoid NLP heuristics).
        intent:
            The classified intent string (e.g. ``"financial_calculation"``,
            ``"conversation"``, ``"document_qa"``).
        evidence:
            The tuple of retrieved ``EvidenceItem`` objects.
        sufficiency_result:
            The sufficiency evaluation from the context builder.
        calculation_result:
            The calculation pipeline result, or ``None`` if not run.
        requested_documents:
            The document names the user asked about (may be empty).
        """
        policy = get_policy_for_intent(intent)

        # --- 1. Calculation blocked / failed -> CALCULATION_BLOCKED ---
        if calculation_result is not None:
            calc_status = calculation_result.status
            if calc_status is CalculationStatus.BLOCKED:
                return self._build_calculation_blocked(
                    evidence, sufficiency_result, calculation_result
                )
            if calc_status is CalculationStatus.FAILED:
                return self._build_calculation_failed(
                    evidence, sufficiency_result, calculation_result
                )

        # --- 2. Conversation / unsupported -> no evidence required ---
        if not policy.require_evidence:
            return AnswerabilityResult(
                status=AnswerabilityStatus.ANSWERABLE,
                reason_codes=(REASON_NO_RETRIEVAL_REQUIRED,),
                evidence_count=len(evidence),
                document_count=self._count_unique_documents(evidence),
                best_score=sufficiency_result.best_score,
                average_score=sufficiency_result.average_score,
            )

        # --- 3. No evidence at all -> NOT_ANSWERABLE ---
        if not evidence:
            missing = self._missing_requirements(
                requested_documents=requested_documents,
                evidence=evidence,
            )
            return AnswerabilityResult(
                status=AnswerabilityStatus.NOT_ANSWERABLE,
                reason_codes=(REASON_NO_EVIDENCE,),
                evidence_count=0,
                document_count=0,
                best_score=sufficiency_result.best_score,
                average_score=sufficiency_result.average_score,
                missing_requirements=missing,
            )

        # --- 4. Insufficient evidence -> NOT_ANSWERABLE ---
        # A deterministic extractor is only allowed to set this signal after
        # finding query-matched text in the actual model context.  It is a
        # generic evidence-strength signal, not a document/page/answer rule.
        # This prevents rank-scale calibration from rejecting a verifiable
        # answer merely because it came from a single-retriever RRF hit.
        if not sufficiency_result.is_sufficient:
            if has_query_matched_evidence:
                return AnswerabilityResult(
                    status=AnswerabilityStatus.ANSWERABLE,
                    reason_codes=(REASON_QUERY_MATCHED_EVIDENCE,),
                    evidence_count=len(evidence),
                    document_count=self._count_unique_documents(evidence),
                    best_score=sufficiency_result.best_score,
                    average_score=sufficiency_result.average_score,
                )
            missing = self._missing_requirements(
                requested_documents=requested_documents,
                evidence=evidence,
            )
            return AnswerabilityResult(
                status=AnswerabilityStatus.NOT_ANSWERABLE,
                reason_codes=(REASON_INSUFFICIENT_EVIDENCE,),
                evidence_count=len(evidence),
                document_count=self._count_unique_documents(evidence),
                best_score=sufficiency_result.best_score,
                average_score=sufficiency_result.average_score,
                missing_requirements=missing,
            )

        # --- 5. Missing requested documents -> PARTIALLY_ANSWERABLE ---
        missing_docs = self._missing_documents(requested_documents, evidence)
        if missing_docs:
            missing = tuple(f"document: {d}" for d in missing_docs)
            return AnswerabilityResult(
                status=AnswerabilityStatus.PARTIALLY_ANSWERABLE,
                reason_codes=(REASON_MISSING_DOCUMENTS,),
                evidence_count=len(evidence),
                document_count=self._count_unique_documents(evidence),
                best_score=sufficiency_result.best_score,
                average_score=sufficiency_result.average_score,
                missing_requirements=missing,
            )

        # --- 6. All checks passed -> ANSWERABLE ---
        return AnswerabilityResult(
            status=AnswerabilityStatus.ANSWERABLE,
            reason_codes=(),
            evidence_count=len(evidence),
            document_count=self._count_unique_documents(evidence),
            best_score=sufficiency_result.best_score,
            average_score=sufficiency_result.average_score,
        )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _count_unique_documents(evidence: tuple[EvidenceItem, ...]) -> int:
        """Count distinct document names in the evidence set."""
        names: set[str | None] = set()
        for item in evidence:
            names.add(item.document_name)
        # Exclude None from the count (evidence without a document name).
        names.discard(None)
        return len(names)

    @staticmethod
    def _missing_documents(
        requested: tuple[str, ...],
        evidence: tuple[EvidenceItem, ...],
    ) -> tuple[str, ...]:
        """Return requested document names that have no evidence chunks."""
        if not requested:
            return ()
        available: set[str | None] = {item.document_name for item in evidence}
        missing: list[str] = []
        for doc in requested:
            if doc not in available:
                missing.append(doc)
        return tuple(missing)

    @staticmethod
    def _missing_requirements(
        *,
        requested_documents: tuple[str, ...],
        evidence: tuple[EvidenceItem, ...],
    ) -> tuple[str, ...]:
        """Build a human-readable list of what is missing for the answer."""
        missing: list[str] = []
        missing_docs = AnswerabilityEvaluator._missing_documents(
            requested_documents, evidence
        )
        for doc in missing_docs:
            missing.append(f"document: {doc}")
        if not evidence:
            missing.append("evidence: no relevant passages retrieved")
        return tuple(missing)

    @staticmethod
    def _build_calculation_blocked(
        evidence: tuple[EvidenceItem, ...],
        sufficiency: SufficiencyResult,
        calc: CalculationResult,
    ) -> AnswerabilityResult:
        """Build a CALCULATION_BLOCKED result.

        The orchestrator must NOT invoke the LLM; the Phase 3 safe
        refusal is returned instead.
        """
        return AnswerabilityResult(
            status=AnswerabilityStatus.CALCULATION_BLOCKED,
            reason_codes=(REASON_CALCULATION_BLOCKED,),
            evidence_count=len(evidence),
            document_count=AnswerabilityEvaluator._count_unique_documents(evidence),
            best_score=sufficiency.best_score,
            average_score=sufficiency.average_score,
            missing_requirements=(
                f"calculation: {calc.error_code or 'blocked'}",
            ),
        )

    @staticmethod
    def _build_calculation_failed(
        evidence: tuple[EvidenceItem, ...],
        sufficiency: SufficiencyResult,
        calc: CalculationResult,
    ) -> AnswerabilityResult:
        """Build a CALCULATION_BLOCKED result for a FAILED calculation.

        FAILED calculations also bypass the LLM (Phase 3 invariant) to
        avoid reintroducing numeric hallucinations via LLM free-form text.
        """
        return AnswerabilityResult(
            status=AnswerabilityStatus.CALCULATION_BLOCKED,
            reason_codes=(REASON_CALCULATION_FAILED,),
            evidence_count=len(evidence),
            document_count=AnswerabilityEvaluator._count_unique_documents(evidence),
            best_score=sufficiency.best_score,
            average_score=sufficiency.average_score,
            missing_requirements=(
                f"calculation: {calc.error_code or 'failed'}",
            ),
        )
