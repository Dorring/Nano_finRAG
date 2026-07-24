"""RAG Orchestrator: coordinates retrieval, context building, and generation.

This is the top-level orchestration layer that ties together the retrieval
pipeline, context builder, sufficiency evaluator, deterministic answer
extractor, and LLM gateway into a single query flow.

The orchestrator exposes a typed boundary:
- Input:  ``QueryRequest`` (domain object)
- Output: ``AnswerResult`` (domain object)

The legacy ``query`` method is retained as a thin compatibility shim that
forwards to ``answer`` and unwraps the legacy dict. Production code
(``RAGEngine``) must use ``answer`` directly so the typed boundary is
exercised on every request.

Phase 3 Commit 8 adds an optional ``calculation_pipeline`` constructor
param. When set, the orchestrator invokes the deterministic calculation
pipeline after context build and before the LLM. If the pipeline returns
``EXECUTED`` or ``BLOCKED``, the LLM is bypassed and the rendered
calculation result is returned directly. When ``calculation_pipeline`` is
None (the default), the orchestrator behaves exactly as before.

Phase 4 Commit 9 adds an optional ``validation_pipeline`` constructor
param. When set, the orchestrator:
1. Runs pre-generation answerability evaluation (before LLM). If the
   verdict is ``NOT_ANSWERABLE``, the LLM is bypassed and a deterministic
   refusal is returned.
2. Runs post-generation response validation (after LLM). If the verdict
   is ``BLOCKED`` or ``FAILED``, the answer is replaced with a safe
   fallback. If ``REPAIRABLE``, a single deterministic repair is applied.
When ``validation_pipeline`` is None (the default), the orchestrator
behaves exactly as before.
"""

import re
import time
import uuid

from src.domain.answer import AnswerPath, AnswerResult
from src.domain.calculation import CalculationResult, CalculationStatus
from src.domain.evidence import EvidenceItem
from src.domain.query import QueryRequest
from src.domain.validation import (
    AnswerabilityResult,
    AnswerabilityStatus,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidationResult,
)
from src.finance.calculation_pipeline import CalculationPipeline
from src.finance.calculation_renderer import render_calculation_result
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.retrieval.context_builder import ContextBuilder, EvidenceSufficiencyEvaluator
from src.generation.llm_gateway import LLMGateway
from src.generation.deterministic_answers import DeterministicAnswerExtractor
from src.retrieval.candidate_fusion import summarize_retrieved_chunks
from src.validation.validation_pipeline import GroundedValidationPipeline
from src.validation.response_repair import RepairResult, ResponseRepair


class RAGOrchestrator:
    """Orchestrates the full RAG query pipeline.

    Coordinates:
    1. Query rewriting (follow-up resolution)
    2. Intent classification
    3. Conversational query handling
    4. Document retrieval (single/multi)
    5. Front-matter deterministic answering
    6. Context sufficiency evaluation
    7. Context building
    8. Deterministic calculation pipeline (Phase 3, when enabled)
    9. Deterministic context answering
    10. LLM generation (when needed)
    11. Trace logging
    """

    def __init__(
        self,
        *,
        query_processor: QueryProcessor,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder,
        sufficiency_evaluator: EvidenceSufficiencyEvaluator,
        llm_gateway: LLMGateway,
        deterministic_extractor: DeterministicAnswerExtractor,
        trace_logger,
        intent_classifier,
        list_all_documents_fn,
        get_front_matter_chunks_fn,
        get_page_chunks_fn=None,
        numeric_rrf_floor: float = 0.008,
        numeric_dense_floor: float = 0.08,
        model_name: str = "nanochat",
        calculation_pipeline: CalculationPipeline | None = None,
        validation_pipeline: GroundedValidationPipeline | None = None,
    ):
        self._query_processor = query_processor
        self._retrieval_pipeline = retrieval_pipeline
        self._context_builder = context_builder
        self._sufficiency_evaluator = sufficiency_evaluator
        self._llm_gateway = llm_gateway
        self._deterministic_extractor = deterministic_extractor
        self._trace_logger = trace_logger
        self._classify_intent = intent_classifier
        self._list_all_documents = list_all_documents_fn
        self._get_front_matter_chunks = get_front_matter_chunks_fn
        self._get_page_chunks = get_page_chunks_fn
        self._numeric_rrf_floor = numeric_rrf_floor
        self._numeric_dense_floor = numeric_dense_floor
        self._model_name = model_name
        self._calculation_pipeline = calculation_pipeline
        self._validation_pipeline = validation_pipeline
        self._response_repair = ResponseRepair() if validation_pipeline else None

    # ------------------------------------------------------------------
    # Typed boundary (production entry point)
    # ------------------------------------------------------------------
    async def answer(
        self,
        request: QueryRequest,
        *,
        n_results: int = 3,
    ) -> AnswerResult:
        """Execute the full RAG pipeline and return a typed ``AnswerResult``.

        ``RAGEngine.query`` must call this method, not the legacy ``query``
        shim, so that ``QueryRequest`` and ``AnswerResult`` are exercised on
        every production request.
        """
        t0 = time.time()
        trace_data: dict = {
            "tenant_id": request.user_id,
            "query_original": request.question,
        }

        question = request.question
        if request.conversation_history:
            question = await self._llm_gateway.rewrite_query(
                question,
                list(request.conversation_history),
                request.memory_profile,
            )
            trace_data["query_rewritten"] = question

        intent = self._classify_intent(question)
        trace_data["intent"] = intent["intent"]

        had_history = bool(request.conversation_history)
        conversational_response = self._handle_conversational_query(question)
        if conversational_response:
            return self._build_conversational_result(
                answer=conversational_response,
                intent=intent,
                rewritten_question=question if had_history else None,
                had_conversation_history=had_history,
            )

        if not intent["requires_retrieval"]:
            return self._build_no_retrieval_result(
                intent=intent,
                rewritten_question=question if had_history else None,
                had_conversation_history=had_history,
            )

        doc_names: list[str]
        if request.document_names:
            doc_names = list(request.document_names)
        else:
            all_docs = self._list_all_documents(request.user_id)
            doc_names = [doc["name"] for doc in all_docs]

        if not doc_names:
            return self._build_no_documents_result(
                rewritten_question=question if had_history else None,
                had_conversation_history=had_history,
            )

        # 1. Retrieve relevant chunks. Front-matter facts use direct metadata lookup first.
        chunks = self._retrieve_front_matter_chunks(
            doc_names, question, request.user_id
        )
        if not chunks:
            if len(doc_names) == 1:
                chunks = self._retrieval_pipeline.retrieve_single(
                    document_name=doc_names[0],
                    query=question,
                    user_id=request.user_id,
                    top_k=n_results,
                )
            else:
                chunks = await self._retrieval_pipeline.retrieve_multiple(
                    document_names=doc_names,
                    query=question,
                    user_id=request.user_id,
                    top_k=n_results,
                )

        front_matter_answer = self._deterministic_extractor.answer_front_matter_query(
            question, chunks
        )
        deterministic_answer = None
        low_confidence_numeric_override = False
        calculation_result = None
        calculation_answer = None
        answerability_result = None
        validation_result = None
        repair_result = None
        initial_validation_result = None
        evidence_for_validation: tuple[EvidenceItem, ...] = ()

        if front_matter_answer:
            chunks = front_matter_answer["chunks"]
            context, sources = self._context_builder.build(chunks)
            answer = front_matter_answer["answer"]
            is_sufficient = True
            confidence = 1.0
            deterministic_answer = front_matter_answer["diagnostic"]
            # Build evidence for front matter validation.
            evidence_for_validation = tuple(EvidenceItem.from_chunk(c) for c in chunks)
            # Run answerability + validation for front matter path.
            if self._validation_pipeline is not None:
                from src.retrieval.context_builder import SufficiencyResult

                fm_sufficiency = SufficiencyResult(
                    is_sufficient=True,
                    best_score=1.0,
                    average_score=1.0,
                )
                answerability_result = self._validation_pipeline.evaluate_answerability(
                    question=question,
                    intent="front_matter",
                    evidence=evidence_for_validation,
                    sufficiency_result=fm_sufficiency,
                    calculation_result=None,
                    requested_documents=tuple(doc_names),
                )
                if answerability_result.status is AnswerabilityStatus.NOT_ANSWERABLE:
                    answer = (
                        "I cannot answer this question based on the "
                        "available evidence. The retrieved documents "
                        "do not contain sufficient information to "
                        "provide a verified response."
                    )
                    is_sufficient = False
                    deterministic_answer = {
                        "path": "answerability_blocked",
                        "status": answerability_result.status.value,
                        "reason_codes": list(answerability_result.reason_codes),
                    }
                else:
                    (
                        answer,
                        validation_result,
                        repair_result,
                        initial_validation_result,
                    ) = self._validate_and_repair_once(
                        answer=answer,
                        intent="front_matter",
                        sources=tuple(sources),
                        evidence=evidence_for_validation,
                        calculation_result=None,
                        answerability=answerability_result,
                    )
        else:
            # Phase 3: Check context sufficiency
            sufficiency = self._sufficiency_evaluator.evaluate(chunks)
            is_sufficient = sufficiency.is_sufficient
            confidence = self._sufficiency_evaluator.confidence(chunks)

            # Numeric evidence is selected from the original child chunks
            # before ContextBuilder expands parents.  This prevents a nearby
            # table row in a parent section from displacing the retrieved
            # row that actually matches the requested metric.
            raw_numeric_answer = self._deterministic_extractor.answer_numeric_query_from_chunks(
                question, chunks
            )

            # 2. Build context (with dedup and score threshold)
            context, sources = self._context_builder.build(chunks)

            # 3. Deterministic calculation pipeline (Phase 3 Commit 8).
            if self._calculation_pipeline is not None:
                evidence = tuple(EvidenceItem.from_chunk(c) for c in chunks)
                calculation_result = self._calculation_pipeline.try_calculate(
                    question,
                    intent,
                    evidence,
                )
                if calculation_result.status is not CalculationStatus.NOT_APPLICABLE:
                    calculation_answer = render_calculation_result(calculation_result)

            # Build validation evidence from the exact context representation
            # (parent-expanded and token-truncated), not stale child chunks.
            context_evidence = getattr(
                self._context_builder, "last_context_evidence", None
            )
            if not isinstance(context_evidence, list) or (
                not context_evidence and chunks and context
            ):
                # Compatibility for lightweight test doubles and alternate
                # context builders that do not expose the new contract yet.
                # A non-empty context without recorded evidence is also an
                # incomplete implementation of that contract; fall back to
                # the retrieved chunks rather than incorrectly refusing an
                # otherwise answerable question as having no evidence.
                context_evidence = chunks
            evidence_for_validation = tuple(
                EvidenceItem.from_chunk(c) for c in context_evidence
            )

            # Extractors operate only on retrieved context and may provide a
            # generic query-matched-evidence signal to answerability.  This
            # does not bypass post-generation grounding validation.
            deterministic_context_answer = None
            if (
                calculation_result is None
                or calculation_result.status is CalculationStatus.NOT_APPLICABLE
            ):
                deterministic_context_answer = raw_numeric_answer or (
                    self._deterministic_extractor.answer_deterministic_query_from_context(
                        question,
                        context,
                        sources,
                    )
                )

            # Preserve the exact child evidence used by the numeric selector.
            # ContextBuilder may replace a child with an expanded parent; that
            # is desirable for LLM context, but validation must also see the
            # row/window that the deterministic answer was derived from.
            if raw_numeric_answer and raw_numeric_answer.get("chunks"):
                raw_evidence = tuple(
                    EvidenceItem.from_chunk(chunk)
                    for chunk in raw_numeric_answer["chunks"]
                )
                evidence_for_validation = raw_evidence + tuple(
                    item for item in evidence_for_validation
                    if item.chunk_id not in {raw.chunk_id for raw in raw_evidence}
                )

            # Phase 4 hotfix: Run answerability for ALL paths (including
            # calculation EXECUTED/BLOCKED/FAILED).
            answerability_blocked = False
            if self._validation_pipeline is not None:
                answerability_result = self._validation_pipeline.evaluate_answerability(
                    question=question,
                    intent=intent["intent"],
                    evidence=evidence_for_validation,
                    sufficiency_result=sufficiency,
                    calculation_result=calculation_result,
                    requested_documents=tuple(doc_names),
                    has_query_matched_evidence=bool(deterministic_context_answer),
                )
                if answerability_result.status is AnswerabilityStatus.NOT_ANSWERABLE:
                    answer = (
                        "I cannot answer this question based on the "
                        "available evidence. The retrieved documents "
                        "do not contain sufficient information to "
                        "provide a verified response."
                    )
                    is_sufficient = False
                    deterministic_answer = {
                        "path": "answerability_blocked",
                        "status": answerability_result.status.value,
                        "reason_codes": list(answerability_result.reason_codes),
                    }
                    answerability_blocked = True
                elif (
                    answerability_result.status
                    is AnswerabilityStatus.CALCULATION_BLOCKED
                ):
                    # Calculation BLOCKED/FAILED: use safe fallback.
                    # No LLM, no generation. Create explicit ValidationResult
                    # so the public API always includes validation status.
                    is_failed = (
                        calculation_result is not None
                        and calculation_result.status is CalculationStatus.FAILED
                    )
                    if is_failed:
                        validation_result = ValidationResult(
                            status=ValidationStatus.FAILED,
                            issues=(
                                ValidationIssue(
                                    code=(
                                        calculation_result.error_code
                                        or "CALCULATION_FAILED"
                                    ),
                                    severity=ValidationSeverity.CRITICAL,
                                    message="Calculation execution failed.",
                                    claim_text="",
                                    evidence_ids=(),
                                    public_message=(
                                        "计算执行失败，本次未返回未经验证的结果。"
                                    ),
                                ),
                            ),
                        )
                    else:
                        validation_result = ValidationResult(
                            status=ValidationStatus.BLOCKED,
                            issues=(
                                ValidationIssue(
                                    code=(
                                        calculation_result.error_code
                                        if calculation_result is not None
                                        else "CALCULATION_BLOCKED"
                                    ),
                                    severity=ValidationSeverity.ERROR,
                                    message="Calculation blocked: insufficient data.",
                                    claim_text="",
                                    evidence_ids=(),
                                    public_message=("计算所需的数据或条件不足。"),
                                ),
                            ),
                        )
                    if self._response_repair is not None:
                        repair_result = self._response_repair.safe_fallback(
                            validation=validation_result,
                            answerability=answerability_result,
                        )
                        answer = repair_result.answer
                    else:
                        answer = (
                            "The requested calculation could not be "
                            "completed with the available data."
                        )
                    is_sufficient = False
                    confidence = 0.0
                    diagnostic_path = (
                        "calculation_failed" if is_failed else "calculation_blocked"
                    )
                    deterministic_answer = {
                        "path": diagnostic_path,
                        "status": answerability_result.status.value,
                        "reason_codes": list(answerability_result.reason_codes),
                        "error_code": (
                            calculation_result.error_code
                            if calculation_result is not None
                            else None
                        ),
                    }
                    answerability_blocked = True

            if not answerability_blocked:
                # Determine which generation path to use.
                is_calculation_executed = (
                    calculation_result is not None
                    and calculation_result.status is CalculationStatus.EXECUTED
                )
                is_calculation_blocked_or_failed = (
                    calculation_result is not None
                    and calculation_result.status
                    in (CalculationStatus.BLOCKED, CalculationStatus.FAILED)
                )

                if is_calculation_executed:
                    # Calculation EXECUTED: use the rendered calculation answer.
                    answer = calculation_answer
                    is_sufficient = True
                    confidence = 1.0
                    deterministic_answer = {
                        "path": "calculation_pipeline",
                        "status": calculation_result.status.value,
                        "operation": (
                            calculation_result.operation.value
                            if calculation_result.operation
                            else None
                        ),
                        "formula_version": calculation_result.formula_version,
                        "error_code": calculation_result.error_code,
                    }
                elif is_calculation_blocked_or_failed:
                    # Phase 3 fallback: calculation BLOCKED/FAILED without
                    # validation pipeline — use rendered calculation answer
                    # (safe refusal) and skip LLM.
                    answer = calculation_answer
                    is_sufficient = False
                    confidence = 0.0
                    deterministic_answer = {
                        "path": (
                            "calculation_failed"
                            if calculation_result.status is CalculationStatus.FAILED
                            else "calculation_blocked"
                        ),
                        "status": calculation_result.status.value,
                        "error_code": calculation_result.error_code,
                    }
                else:
                    # Non-calculation: deterministic context or LLM.
                    if deterministic_context_answer:
                        answer = deterministic_context_answer["answer"]
                        is_sufficient = True
                        deterministic_answer = {
                            "path": deterministic_context_answer["diagnostic"],
                            "selection": deterministic_context_answer.get("selection", []),
                        }
                        low_confidence_numeric_override = False
                    else:
                        low_confidence_numeric_override = (
                            self._query_processor.should_generate_with_low_confidence(
                                question,
                                chunks,
                                numeric_rrf_floor=self._numeric_rrf_floor,
                                numeric_dense_floor=self._numeric_dense_floor,
                            )
                        )
                        if low_confidence_numeric_override:
                            is_sufficient = True

                        if not is_sufficient:
                            answer = "I couldn't find sufficiently relevant information in the documents to answer this question reliably."
                        else:
                            answer = await self._llm_gateway.generate(context, question)

                    # Apply PARTIALLY_ANSWERABLE restricted prefix.
                    if (
                        answerability_result is not None
                        and answerability_result.status
                        is AnswerabilityStatus.PARTIALLY_ANSWERABLE
                    ):
                        answer = self._apply_partial_prefix(
                            answer, answerability_result
                        )

            # Phase 4 hotfix: Post-generation validation for ALL paths
            # (calculation EXECUTED, deterministic, LLM) — but NOT for
            # answerability-blocked paths (which already have a safe fallback).
            if self._validation_pipeline is not None and not answerability_blocked:
                answer, validation_result, repair_result, initial_validation_result = (
                    self._validate_and_repair_once(
                        answer=answer,
                        intent=intent["intent"],
                        sources=tuple(sources),
                        evidence=evidence_for_validation,
                        calculation_result=calculation_result,
                        answerability=answerability_result,
                    )
                )

        # 5. Log trace (Phase 4 hotfix: redact full context and answer).
        elapsed_ms = (time.time() - t0) * 1000
        import hashlib as _hashlib

        context_str = context or ""
        answer_str = answer or ""
        context_hash = _hashlib.sha256(context_str.encode("utf-8")).hexdigest()[:16]
        answer_hash = _hashlib.sha256(answer_str.encode("utf-8")).hexdigest()[:16]

        trace_data.update(
            {
                "filter_conditions": {"doc_names": doc_names, "n_results": n_results},
                "candidates": [
                    {
                        "doc_id": c.get("doc_id", ""),
                        "score": c.get("score", 0),
                        "rerank_score": c.get("rerank_score"),
                        "reranker": c.get("reranker"),
                    }
                    for c in chunks
                ],
                # Phase 4 hotfix: do not store full context/answer in trace.
                "final_context": None,
                "answer": None,
                "sources": sources,
                "diagnostics": {
                    "confidence": confidence,
                    "context_sufficient": is_sufficient,
                    "intent_confidence": intent["confidence"],
                    "deterministic_answer": deterministic_answer,
                    "low_confidence_numeric_override": (
                        low_confidence_numeric_override
                        if not front_matter_answer
                        else False
                    ),
                    "calculation": (
                        calculation_result.to_trace_dict()
                        if calculation_result is not None
                        and calculation_result.status
                        is not CalculationStatus.NOT_APPLICABLE
                        else None
                    ),
                    # Phase 4 hotfix: content diagnostics (hashes + lengths).
                    "context_length": len(context_str),
                    "context_sha256": context_hash,
                    "answer_length": len(answer_str),
                    "answer_sha256": answer_hash,
                },
                "model_name": self._model_name,
                "latency_ms": elapsed_ms,
            }
        )
        # Phase 4: Add validation diagnostics to trace only when the
        # validation pipeline actually ran (non-None results).
        if self._validation_pipeline is not None:
            trace_data["diagnostics"]["answerability"] = (
                answerability_result.to_trace_dict()
                if answerability_result is not None
                else None
            )
            trace_data["diagnostics"]["initial_validation"] = (
                initial_validation_result.to_trace_dict()
                if initial_validation_result is not None
                else None
            )
            trace_data["diagnostics"]["validation"] = (
                validation_result.to_trace_dict()
                if validation_result is not None
                else None
            )
            trace_data["diagnostics"]["repair"] = (
                repair_result.to_trace_dict() if repair_result is not None else None
            )
        trace_id = None
        try:
            trace_id = self._trace_logger.log(**trace_data)
        except Exception:
            pass  # tracing must never break the query path
        if trace_id is None:
            # Fallback: ensure every response carries a trace_id even when
            # sampling skips or the logger fails. This ID is not stored in
            # the trace DB, but allows clients to correlate requests.
            trace_id = uuid.uuid4().hex

        # Build calculations tuple for AnswerResult (additive field).
        calculations: tuple[dict, ...] = ()
        if (
            calculation_result is not None
            and calculation_result.status is not CalculationStatus.NOT_APPLICABLE
        ):
            calculations = (calculation_result.to_public_dict(),)

        return AnswerResult(
            answer=answer,
            sources=tuple(sources),
            context=context,
            searched_docs=tuple(doc_names),
            confidence=confidence,
            context_sufficient=is_sufficient,
            intent=intent["intent"],
            intent_confidence=intent["confidence"],
            rewritten_question=question if had_history else None,
            retrieved_chunks=tuple(summarize_retrieved_chunks(chunks)),
            retrieval_debug=dict(self._retrieval_pipeline.last_retrieval_debug),
            trace_id=trace_id,
            path=AnswerPath.FULL,
            had_conversation_history=had_history,
            calculations=calculations,
            answerability=(
                answerability_result.to_public_dict()
                if answerability_result is not None
                else None
            ),
            validation=(
                validation_result.to_public_dict()
                if validation_result is not None
                else None
            ),
            repair=(
                repair_result.to_public_dict() if repair_result is not None else None
            ),
        )

    # ------------------------------------------------------------------
    # Legacy compatibility shim (not for production use)
    # ------------------------------------------------------------------
    async def query(
        self,
        question: str,
        doc_names: list[str] | None = None,
        user_id: int = None,
        n_results: int = 3,
        conversation_history: list = None,
        memory_profile: dict | None = None,
    ) -> dict:
        """Legacy entry point retained for tests that mock the orchestrator.

        Production code must call ``answer`` with a ``QueryRequest``.
        """
        request = QueryRequest(
            question=question,
            document_names=tuple(doc_names or ()),
            user_id=user_id,
            conversation_history=tuple(conversation_history or ()),
            memory_profile=memory_profile,
        )
        result = await self.answer(request, n_results=n_results)
        return result.to_legacy_dict()

    # ------------------------------------------------------------------
    # Branch constructors for early-return paths
    # ------------------------------------------------------------------
    @staticmethod
    def _build_conversational_result(
        *,
        answer: str,
        intent: dict,
        rewritten_question: str | None,
        had_conversation_history: bool = False,
    ) -> AnswerResult:
        return AnswerResult(
            answer=answer,
            sources=(),
            context=None,
            searched_docs=(),
            confidence=None,
            context_sufficient=True,
            intent="conversation",
            intent_confidence=intent["confidence"],
            rewritten_question=rewritten_question,
            retrieved_chunks=(),
            retrieval_debug={},
            trace_id=uuid.uuid4().hex,
            path=AnswerPath.CONVERSATIONAL,
            had_conversation_history=had_conversation_history,
        )

    @staticmethod
    def _build_no_retrieval_result(
        *,
        intent: dict,
        rewritten_question: str | None,
        had_conversation_history: bool = False,
    ) -> AnswerResult:
        return AnswerResult(
            answer=(
                "This question appears to be outside the uploaded financial documents. "
                "Please ask about your uploaded reports or financial data."
            ),
            sources=(),
            context=None,
            searched_docs=(),
            confidence=None,
            context_sufficient=True,
            intent=intent["intent"],
            intent_confidence=intent["confidence"],
            rewritten_question=rewritten_question,
            retrieved_chunks=(),
            retrieval_debug={},
            trace_id=uuid.uuid4().hex,
            path=AnswerPath.NO_RETRIEVAL,
            had_conversation_history=had_conversation_history,
        )

    @staticmethod
    def _build_no_documents_result(
        *,
        rewritten_question: str | None,
        had_conversation_history: bool = False,
    ) -> AnswerResult:
        return AnswerResult(
            answer="No documents found in database. Please upload documents first.",
            sources=(),
            context=None,
            searched_docs=(),
            confidence=None,
            context_sufficient=True,
            intent=None,
            intent_confidence=None,
            rewritten_question=rewritten_question,
            retrieved_chunks=(),
            retrieval_debug={},
            trace_id=uuid.uuid4().hex,
            path=AnswerPath.NO_DOCUMENTS,
            had_conversation_history=had_conversation_history,
        )

    # ------------------------------------------------------------------
    # Phase 4 hotfix: unified validation + repair orchestration
    # ------------------------------------------------------------------
    def _validate_and_repair_once(
        self,
        *,
        answer: str,
        intent: str,
        sources: tuple[dict, ...],
        evidence: tuple[EvidenceItem, ...],
        calculation_result: CalculationResult | None,
        answerability: AnswerabilityResult | None,
    ) -> tuple[str, ValidationResult, RepairResult | None, ValidationResult | None]:
        """Run validation, repair once if needed, then revalidate.

        Pattern (Phase 4 hotfix):
        1. Initial validation.
        2. If REPAIRABLE, execute ONE repair.
        3. Revalidate the repaired answer.
        4. PASSED -> return repaired answer.
        5. BLOCKED/FAILED/REPAIRABLE -> safe fallback (no second repair).

        Constraints:
        - At most ONE repair.
        - At most ONE revalidation.
        - No second repair allowed.
        - Never calls the LLM.
        - The final ``validation`` field in the API response is the
          revalidation result (if repair was attempted) or the initial
          validation result (if no repair was needed).
        - The initial validation result is stored in trace as
          ``initial_validation``.

        Returns ``(final_answer, final_validation, repair_result,
        initial_validation)``.
        """
        if self._validation_pipeline is None or self._response_repair is None:
            # Guard: should never happen because callers check first.
            not_applicable = ValidationResult(
                status=ValidationStatus.NOT_APPLICABLE,
            )
            return answer, not_applicable, None, None

        # 1. Initial validation (fail-closed on exception).
        try:
            initial_validation = self._validation_pipeline.validate_response(
                answer=answer,
                intent=intent,
                evidence=evidence,
                calculation_result=calculation_result,
                sources=sources,
            )
        except Exception:
            initial_validation = ValidationResult(
                status=ValidationStatus.FAILED,
                issues=(
                    ValidationIssue(
                        code="VALIDATOR_EXCEPTION",
                        severity=ValidationSeverity.CRITICAL,
                        message="Initial validation raised an exception.",
                    ),
                ),
            )

        # 2. PASSED / NOT_APPLICABLE -> no repair needed.
        if initial_validation.status in (
            ValidationStatus.PASSED,
            ValidationStatus.NOT_APPLICABLE,
        ):
            no_op = RepairResult(
                answer=answer,
                was_repaired=False,
                fallback_used=False,
            )
            return answer, initial_validation, no_op, initial_validation

        # 3. BLOCKED / FAILED -> safe fallback immediately (no repair).
        if initial_validation.status in (
            ValidationStatus.BLOCKED,
            ValidationStatus.FAILED,
        ):
            fallback = self._response_repair.safe_fallback(
                validation=initial_validation,
                answerability=answerability,
            )
            return fallback.answer, initial_validation, fallback, initial_validation

        # 4. REPAIRABLE -> attempt ONE repair, then revalidate.
        repair_result = self._response_repair.repair(
            answer=answer,
            validation=initial_validation,
            answerability=answerability,
        )

        # If the repair used a fallback (empty / unrepairable), return it.
        if repair_result.fallback_used:
            return (
                repair_result.answer,
                initial_validation,
                repair_result,
                initial_validation,
            )

        # Revalidate the repaired answer (at most once).
        try:
            final_validation = self._validation_pipeline.validate_response(
                answer=repair_result.answer,
                intent=intent,
                evidence=evidence,
                calculation_result=calculation_result,
                sources=sources,
            )
        except Exception:
            final_validation = ValidationResult(
                status=ValidationStatus.FAILED,
                issues=(
                    ValidationIssue(
                        code="VALIDATOR_EXCEPTION",
                        severity=ValidationSeverity.CRITICAL,
                        message="Revalidation raised an exception.",
                    ),
                ),
            )

        # PASSED -> return the repaired answer.
        if final_validation.status in (
            ValidationStatus.PASSED,
            ValidationStatus.NOT_APPLICABLE,
        ):
            return (
                repair_result.answer,
                final_validation,
                repair_result,
                initial_validation,
            )

        # Still not PASSED -> safe fallback (NO second repair).
        fallback = self._response_repair.safe_fallback(
            validation=final_validation,
            answerability=answerability,
        )
        return fallback.answer, final_validation, fallback, initial_validation

    @staticmethod
    def _apply_partial_prefix(
        answer: str,
        answerability: AnswerabilityResult,
    ) -> str:
        """Apply a restricted prefix/suffix for PARTIALLY_ANSWERABLE responses.

        The prefix makes clear that only a partial answer is provided,
        and the suffix lists what could not be verified. This prevents
        the user from mistaking a partial answer for a complete one.
        """
        if answerability.missing_requirements:
            missing_text = "; ".join(answerability.missing_requirements)
        else:
            missing_text = "部分请求的文档或数据"
        return (
            f"根据当前检索到的资料，只能确认以下部分：\n"
            f"{answer}\n"
            f"未找到或无法验证：{missing_text}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _retrieve_front_matter_chunks(
        self,
        doc_names: list[str],
        query: str,
        user_id: int | None = None,
    ) -> list:
        """Direct metadata lookup for deterministic front-matter questions."""
        if not self._query_processor.is_front_matter_query(query) or not doc_names or user_id is None:
            return []
        chunks = []
        for doc_name in doc_names:
            has_valid_title = False
            for chunk in self._get_front_matter_chunks(
                doc_name=doc_name, user_id=user_id, subtype="title"
            ):
                title = re.sub(
                    r"^title\s*:\s*", "", chunk.get("content", ""), flags=re.IGNORECASE
                ).strip()
                if DeterministicAnswerExtractor._is_valid_deterministic_title(
                    DeterministicAnswerExtractor._clean_deterministic_title(title)
                ):
                    chunks.append(chunk)
                    has_valid_title = True
            # The title metadata can be partial on marketing-style covers. For
            # title+period questions we also need the adjacent cover evidence.
            # Page-one lookup is deterministic, tenant scoped, and avoids
            # relying on an unrelated vector hit from later in the document.
            if self._get_page_chunks and (
                not has_valid_title or "reporting period" in (query or "").lower()
            ):
                chunks.extend(
                    self._get_page_chunks(
                        doc_name, user_id, [1], limit_per_page=4
                    )
                )
        return chunks

    @staticmethod
    def _handle_conversational_query(query: str) -> str | None:
        """Handle conversational/meta questions without RAG retrieval."""
        query_lower = query.lower().strip()

        # Financial keyword guard - never classify as chitchat
        financial_indicators = [
            "revenue",
            "expense",
            "profit",
            "loss",
            "income",
            "cash",
            "balance",
            "debt",
            "equity",
            "margin",
            "growth",
            "quarter",
            "fiscal",
            "earnings",
            "dividend",
            "asset",
            "liability",
            "$",
            "%",
            "million",
            "billion",
            "q1",
            "q2",
            "q3",
            "q4",
            "fy",
            "yoy",
            "table",
            "page",
            "report",
            "statement",
            "cost",
            "营收",
            "利润",
            "亏损",
            "收入",
            "现金",
            "负债",
            "资产",
            "权益",
            "增长",
            "季度",
            "财报",
            "股息",
            "报表",
            "成本",
            "费用",
            "净利",
        ]
        if any(ind in query_lower for ind in financial_indicators):
            return None

        # Greetings
        greetings = [
            "hi",
            "hello",
            "hi there",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]
        if (
            any(query_lower.startswith(g) for g in greetings)
            and len(query_lower.split()) <= 3
        ):
            return "Hello! I'm FinQuery, your financial document assistant. I can help you find information in your uploaded documents. What would you like to know?"

        # Identity questions
        identity_keywords = [
            "what are you",
            "who are you",
            "what is finquery",
            "tell me about yourself",
            "what do you do",
            "what can you do",
            "how do you work",
            "what's your purpose",
        ]
        if any(keyword in query_lower for keyword in identity_keywords):
            return "I'm FinQuery, an AI assistant that helps you analyze financial documents. Upload PDFs of reports, statements, or other financial documents, and I'll answer questions about them using the exact information from those documents."

        # Capability questions
        capability_keywords = [
            "how does this work",
            "how to use",
            "help me",
            "what can i ask",
            "how do i use this",
        ]
        if any(keyword in query_lower for keyword in capability_keywords):
            return "Here's how to use FinQuery:\n1. Upload financial documents (PDFs)\n2. Ask questions about the content\n3. I'll provide answers with page citations\n\nTry: 'What was the revenue in Q3?' or 'Summarize key financial metrics'"

        # Thanks/gratitude
        thanks_keywords = ["thank you", "thanks", "thx", "appreciate"]
        if (
            any(keyword in query_lower for keyword in thanks_keywords)
            and len(query_lower.split()) <= 5
        ):
            return "You're welcome! Let me know if you have any other questions about your documents."

        # Goodbyes
        goodbye_keywords = ["bye", "goodbye", "see you", "exit", "quit"]
        if (
            any(keyword in query_lower for keyword in goodbye_keywords)
            and len(query_lower.split()) <= 3
        ):
            return "Goodbye! Feel free to come back anytime you need to analyze financial documents."

        return None
