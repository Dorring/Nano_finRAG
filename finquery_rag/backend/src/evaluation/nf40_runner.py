"""Frozen-context, local-only execution adapter for NF40 attribution."""
from __future__ import annotations

import inspect
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from src.evaluation.evaluation import EvaluationCase, Prediction, score_prediction
from src.evaluation.nf40_attribution import (
    StageEvaluation,
    classify_context_coverage,
    compute_nf40_metrics,
    determine_primary_failure,
)
from src.evaluation.nf40_frozen_context import FrozenCaseContext, as_evaluation_context
from src.evaluation.nf40_pipeline_observer import (
    AnswerPipelineTrace,
    EvaluationExecutionContext,
)


class NF40ConfigurationError(ValueError):
    """Raised before model invocation when the frozen experiment is invalid."""


def validate_labeled_cases(cases: Iterable[EvaluationCase], *, expected_count: int = 27) -> list[EvaluationCase]:
    rows = list(cases)
    if len(rows) != expected_count:
        raise NF40ConfigurationError(f"Expected {expected_count} cases, got {len(rows)}")
    identifiers = [case.case_id for case in rows]
    if len(identifiers) != len(set(identifiers)):
        raise NF40ConfigurationError("Duplicate case IDs")
    missing = [case.case_id for case in rows if not case.expected_no_answer and not case.expected_sources]
    if missing:
        raise NF40ConfigurationError(f"Answerable cases missing expected sources: {missing}")
    return rows


def _matched_gold_count(case: EvaluationCase, frozen: FrozenCaseContext) -> int:
    candidates = [
        {"filename": item.document_id, "page": item.page, "chunk_id": item.source_id}
        for item in frozen.candidates
    ]
    return sum(any(source.matches(candidate) for candidate in candidates) for source in case.expected_sources)


@dataclass(frozen=True)
class NF40CaseRun:
    evaluation: StageEvaluation
    public_record: dict[str, Any]
    private_record: dict[str, Any]


class FrozenContextEvaluationRunner:
    """Executes production answer components while proving retrieval is bypassed."""

    def __init__(self, *, rag_engine: Any, execution_context: EvaluationExecutionContext | None = None):
        self._rag_engine = rag_engine
        self._execution_context = execution_context or EvaluationExecutionContext()
        self._execution_context.validate()

    async def run_case(self, *, case: EvaluationCase, frozen: FrozenCaseContext, tenant_id: int) -> NF40CaseRun:
        matched = _matched_gold_count(case, frozen)
        coverage = classify_context_coverage(
            expected_no_answer=case.expected_no_answer,
            expected_source_count=len(case.expected_sources),
            matched_gold_source_count=matched,
        )
        trace = AnswerPipelineTrace(
            case_id=case.case_id,
            trace_id=uuid.uuid4().hex,
            context_hash=frozen.final_context_hash,
            context_coverage=coverage.value,
        )
        frozen_input = as_evaluation_context(frozen)
        started = time.monotonic()
        result = self._rag_engine.answer_frozen_evaluation(
            question=case.question,
            user_id=tenant_id,
            frozen_context=frozen_input,
            evaluation_observer=trace,
        )
        if inspect.isawaitable(result):
            result = await result
        latency_ms = (time.monotonic() - started) * 1000
        if result.get("context") != frozen_input.context:
            raise NF40ConfigurationError(f"{case.case_id}: answer pipeline changed frozen context")
        raw_prediction = Prediction(
            case_id=case.case_id,
            answer=trace._raw_generation_text or "",
            sources=tuple(frozen_input.sources),
            retrieved_chunks=tuple(frozen_input.chunks),
            calculations=tuple(result.get("calculations") or ()),
            intent=result.get("intent"),
            latency_ms=latency_ms,
        )
        released_prediction = Prediction(
            case_id=case.case_id,
            answer=str(result.get("answer") or ""),
            sources=tuple(result.get("sources") or ()),
            retrieved_chunks=tuple(frozen_input.chunks),
            calculations=tuple(result.get("calculations") or ()),
            intent=result.get("intent"),
            latency_ms=latency_ms,
        )
        raw_score = score_prediction(case, raw_prediction)
        released_score = score_prediction(case, released_prediction)
        released = bool(trace.released_response_type == "answer")
        evaluation = StageEvaluation(
            case_id=case.case_id,
            context_coverage=coverage,
            raw_answer_present=bool(trace._raw_generation_text),
            raw_fact_correct=raw_score["answer_contains"] == 1.0,
            raw_numeric_correct=raw_score["number_accuracy"] == 1.0,
            raw_unit_correct=True,
            raw_period_correct=True,
            raw_citation_correct=raw_score["citation_recall"] == 1.0,
            raw_answer_correct=bool(raw_score["passed"]),
            released_answer_correct=bool(released_score["passed"]),
            released=released,
            calculation_attempted=trace.calculation_attempted,
            calculation_failed=trace.calculation_status == "failed",
            validation_failures=tuple(trace.validation_failures),
            repair_attempted=trace.repair_attempted,
            repair_succeeded=trace.repair_status == "repaired",
            no_answer_correct=(released_score["no_answer_accuracy"] == 1.0 if case.expected_no_answer else None),
            latency_ms=latency_ms,
        )
        primary = determine_primary_failure(evaluation)
        public = {
            "case_id": case.case_id,
            "context_coverage": coverage.value,
            "matched_gold_source_count": matched,
            "expected_gold_source_count": len(case.expected_sources),
            "context_hash": frozen.final_context_hash,
            "raw_generation_hash": trace.raw_generation_hash,
            "released_answer_hash": trace.released_answer_hash,
            "raw_answer_correct": evaluation.raw_answer_correct,
            "released_answer_correct": evaluation.released_answer_correct,
            "validation_status": trace.validation_status,
            "validation_failures": list(trace.validation_failures),
            "repair_attempted": trace.repair_attempted,
            "repair_status": trace.repair_status,
            "primary_failure": primary.value,
            "latency_ms": latency_ms,
        }
        private = {
            "case_id": case.case_id,
            "trace_id": trace.trace_id,
            "raw_generation": trace._raw_generation_text,
            "released_answer": trace._released_answer_text,
            "context_hash": frozen.final_context_hash,
        }
        return NF40CaseRun(evaluation=evaluation, public_record=public, private_record=private)

    async def run(self, *, cases: Iterable[EvaluationCase], contexts: dict[str, FrozenCaseContext], tenant_id: int) -> tuple[list[NF40CaseRun], dict]:
        rows = validate_labeled_cases(cases)
        if set(contexts) != {case.case_id for case in rows}:
            raise NF40ConfigurationError("Frozen contexts do not match evaluation cases")
        runs = [await self.run_case(case=case, frozen=contexts[case.case_id], tenant_id=tenant_id) for case in rows]
        return runs, compute_nf40_metrics(run.evaluation for run in runs)
