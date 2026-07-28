"""Offline evaluation and trace replay helpers for FinQuery RAG.

The module intentionally avoids network, model, and vector-store dependencies.
It scores saved predictions and can convert trace records into replay cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
import json
import math
import os
import re
import tempfile
from statistics import mean
from typing import Any, Iterable

from src.services.answer_validation import validate_answer_calculations
from src.services.financial_tools import (
    convert_scale,
    format_ratio_percent,
    growth_rate,
    percentage_share,
    sum_values,
    verify_sum,
)


_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")


@dataclass(frozen=True)
class ExpectedSource:
    """A citation or retrieval target expected for a case."""

    filename: str | None = None
    page: int | str | None = None
    chunk_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectedSource":
        _ensure_mapping(data, "expected source")
        return cls(
            filename=data.get("filename") or data.get("doc_name"),
            page=data.get("page"),
            chunk_id=data.get("chunk_id") or data.get("doc_id"),
        )

    def matches(self, candidate: dict[str, Any]) -> bool:
        """Return True when candidate satisfies all fields set on this source."""
        candidate_id = candidate.get("chunk_id") or candidate.get("doc_id")
        if self.chunk_id and candidate_id != self.chunk_id:
            return False

        if self.filename:
            cand_filename = candidate.get("filename") or candidate.get("doc_name")
            if not cand_filename and candidate_id:
                cand_filename = _filename_from_doc_id(str(candidate_id))
            if cand_filename != self.filename:
                return False

        if self.page is not None:
            cand_page = candidate.get("page")
            if str(cand_page) != str(self.page):
                return False

        return True



@dataclass(frozen=True)
class ExpectedCalculation:
    """Expected deterministic financial calculation for a case."""

    calc_id: str
    operation: str
    args: dict[str, Any]
    expected_value: str
    tolerance: str = "0"
    unit: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectedCalculation":
        _ensure_mapping(data, "expected calculation")
        calc_id = data.get("id") or data.get("calc_id") or data.get("operation")
        operation = data.get("operation")
        if not calc_id:
            raise ValueError("expected calculation missing id/calc_id")
        if not operation:
            raise ValueError(f"expected calculation {calc_id!r} missing operation")
        return cls(
            calc_id=str(calc_id),
            operation=str(operation),
            args=dict(data.get("args", {})),
            expected_value=str(data.get("expected_value")),
            tolerance=str(data.get("tolerance", "0")),
            unit=data.get("unit"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.calc_id,
            "operation": self.operation,
            "args": self.args,
            "expected_value": self.expected_value,
            "tolerance": self.tolerance,
            "unit": self.unit,
        }

@dataclass(frozen=True)
class EvaluationCase:
    """A golden/replay RAG evaluation case."""

    case_id: str
    question: str
    expected_sources: tuple[ExpectedSource, ...] = ()
    expected_answer_contains: tuple[str, ...] = ()
    expected_numbers: tuple[str, ...] = ()
    expected_no_answer: bool = False
    expected_calculations: tuple[ExpectedCalculation, ...] = ()
    expected_intent: str | None = None
    document_names: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationCase":
        _ensure_mapping(data, "evaluation case")
        case_id = data.get("id") or data.get("case_id") or data.get("trace_id")
        question = data.get("question")
        if not case_id:
            raise ValueError("evaluation case missing id/case_id")
        if not question:
            raise ValueError(f"evaluation case {case_id!r} missing question")
        _ensure_list_fields(
            data,
            ("expected_sources", "expected_answer_contains", "expected_numbers", "expected_calculations", "document_names", "tags"),
            f"evaluation case {case_id!r}",
        )
        if data.get("metadata") is not None and not isinstance(data.get("metadata"), dict):
            raise ValueError(f"evaluation case {case_id!r} field metadata must be an object")

        return cls(
            case_id=str(case_id),
            question=str(question),
            expected_sources=tuple(
                ExpectedSource.from_dict(item)
                for item in data.get("expected_sources", [])
            ),
            expected_answer_contains=tuple(
                str(item) for item in data.get("expected_answer_contains", [])
            ),
            expected_numbers=tuple(
                _normalize_number(str(item)) for item in data.get("expected_numbers", [])
            ),
            expected_no_answer=bool(data.get("expected_no_answer", False)),
            expected_calculations=tuple(
                ExpectedCalculation.from_dict(item)
                for item in data.get("expected_calculations", [])
            ),
            expected_intent=(
                str(data["expected_intent"]) if data.get("expected_intent") else None
            ),
            document_names=tuple(str(item) for item in data.get("document_names", [])),
            tags=tuple(str(item) for item in data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "question": self.question,
            "document_names": list(self.document_names),
            "expected_sources": [
                {
                    "filename": s.filename,
                    "page": s.page,
                    "chunk_id": s.chunk_id,
                }
                for s in self.expected_sources
            ],
            "expected_answer_contains": list(self.expected_answer_contains),
            "expected_numbers": list(self.expected_numbers),
            "expected_no_answer": self.expected_no_answer,
            "expected_calculations": [calc.to_dict() for calc in self.expected_calculations],
            "expected_intent": self.expected_intent,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Prediction:
    """A saved RAG output for an EvaluationCase."""

    case_id: str
    answer: str
    sources: tuple[dict[str, Any], ...] = ()
    retrieved_chunks: tuple[dict[str, Any], ...] = ()
    calculations: tuple[dict[str, Any], ...] = ()
    intent: str | None = None
    intent_confidence: float | None = None
    latency_ms: float | None = None
    error: str | None = None
    error_detail: str | None = None
    retrieval_debug: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prediction":
        _ensure_mapping(data, "prediction")
        case_id = data.get("id") or data.get("case_id")
        if not case_id:
            raise ValueError("prediction missing id/case_id")
        _ensure_list_fields(data, ("sources", "retrieved_chunks", "calculations"), f"prediction {case_id!r}")
        for field_name in ("sources", "retrieved_chunks", "calculations"):
            for idx, item in enumerate(data.get(field_name, []) or []):
                if not isinstance(item, dict):
                    raise ValueError(f"prediction {case_id!r} field {field_name}[{idx}] must be an object")
        return cls(
            case_id=str(case_id),
            answer=str(data.get("answer", "")),
            sources=tuple(dict(item) for item in data.get("sources", [])),
            retrieved_chunks=tuple(dict(item) for item in data.get("retrieved_chunks", [])),
            calculations=tuple(dict(item) for item in data.get("calculations", [])),
            intent=str(data["intent"]) if data.get("intent") else None,
            intent_confidence=_optional_float(data.get("intent_confidence")),
            latency_ms=_optional_float(data.get("latency_ms")),
            error=str(data["error"]) if data.get("error") else None,
            error_detail=str(data["error_detail"]) if data.get("error_detail") else None,
            retrieval_debug=dict(data.get("retrieval_debug") or {}),
        )


def load_jsonl_cases(path: str | Path) -> list[EvaluationCase]:
    cases = []
    seen_ids = set()
    for line_no, item in _read_jsonl_rows(path):
        try:
            case = EvaluationCase.from_dict(item)
        except ValueError as exc:
            raise ValueError(f"invalid evaluation case at {path}:{line_no}: {exc}") from exc
        if case.case_id in seen_ids:
            raise ValueError(f"duplicate evaluation case id {case.case_id!r} at {path}:{line_no}")
        seen_ids.add(case.case_id)
        cases.append(case)
    return cases


def load_jsonl_predictions(path: str | Path) -> dict[str, Prediction]:
    predictions = {}
    for line_no, item in _read_jsonl_rows(path):
        try:
            pred = Prediction.from_dict(item)
        except ValueError as exc:
            raise ValueError(f"invalid prediction at {path}:{line_no}: {exc}") from exc
        if pred.case_id in predictions:
            raise ValueError(f"duplicate prediction id {pred.case_id!r} at {path}:{line_no}")
        predictions[pred.case_id] = pred
    return predictions


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL rows atomically.

    The caller gets either the complete new file or the previous file remains
    untouched. This matters for replay/eval artifacts because partial files can
    silently poison later regression comparisons.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=out.parent,
            prefix=f".{out.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_name = fh.name
            for row_no, row in enumerate(rows, 1):
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row {row_no} must be an object")
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp_name, out)
    except Exception:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def write_json_file(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a formatted JSON object atomically."""
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def score_prediction(case: EvaluationCase, prediction: Prediction) -> dict[str, Any]:
    """Score one prediction against one case."""
    answer = prediction.answer or ""
    answer_lower = answer.lower()

    source_stats = _score_sources(case.expected_sources, prediction.sources)
    retrieval_stats = _score_sources(case.expected_sources, prediction.retrieved_chunks)
    contains_score = _contains_score(answer_lower, case.expected_answer_contains)
    number_score = _number_score(answer, case.expected_numbers)
    no_answer_score = _no_answer_score(answer_lower, case.expected_no_answer)
    calculation_score = _calculation_score(case.expected_calculations, prediction.calculations)
    answer_calculation_consistency = _answer_calculation_consistency(
        answer,
        prediction.calculations,
    )
    intent_accuracy = _intent_score(case.expected_intent, prediction.intent)

    required_scores = []
    if case.expected_sources:
        required_scores.append(source_stats["recall"])
    if case.expected_answer_contains:
        required_scores.append(contains_score)
    if case.expected_numbers:
        required_scores.append(number_score)
    if case.expected_no_answer:
        required_scores.append(no_answer_score)
    if case.expected_calculations:
        required_scores.append(calculation_score)
    if prediction.calculations:
        required_scores.append(answer_calculation_consistency)
    if case.expected_intent:
        required_scores.append(intent_accuracy)

    passed = bool(required_scores) and all(score >= 1.0 for score in required_scores)
    if not required_scores:
        passed = True

    failure_category = _classify_failure(
        case,
        prediction,
        passed=passed,
        source_recall=source_stats["recall"],
        retrieval_recall=retrieval_stats["recall"],
        contains_score=contains_score,
        number_score=number_score,
        no_answer_score=no_answer_score,
        intent_accuracy=intent_accuracy,
    )

    return {
        "id": case.case_id,
        "passed": passed,
        "failure_category": failure_category,
        "citation_precision": source_stats["precision"],
        "citation_recall": source_stats["recall"],
        "retrieval_precision": retrieval_stats["precision"],
        "retrieval_recall": retrieval_stats["recall"],
        "answer_contains": contains_score,
        "number_accuracy": number_score,
        "no_answer_accuracy": no_answer_score,
        "calculation_accuracy": calculation_score,
        "answer_calculation_consistency": answer_calculation_consistency,
        "intent_accuracy": intent_accuracy,
        "expected_answer_contains": list(case.expected_answer_contains),
        "expected_numbers": list(case.expected_numbers),
        "expected_sources": [_expected_source_to_dict(source) for source in case.expected_sources],
        "actual_answer_preview": _preview_text(prediction.answer),
        "actual_sources": [_compact_source_dict(source) for source in prediction.sources],
        "actual_retrieved_chunks": [_compact_source_dict(chunk) for chunk in prediction.retrieved_chunks],
        "prediction_error": prediction.error,
        "prediction_error_detail": _preview_text(prediction.error_detail or "", limit=300),
        "expected_intent": case.expected_intent,
        "predicted_intent": prediction.intent,
        "intent_confidence": prediction.intent_confidence,
        "latency_ms": prediction.latency_ms,
        "tags": list(case.tags),
    }


def evaluate_payload(cases_payload: list[dict[str, Any]], predictions_payload: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate in-memory JSON-compatible cases and predictions."""
    cases = [EvaluationCase.from_dict(item) for item in cases_payload]
    predictions: dict[str, Prediction] = {}
    for item in predictions_payload:
        prediction = Prediction.from_dict(item)
        if prediction.case_id in predictions:
            raise ValueError(f"duplicate prediction id {prediction.case_id!r}")
        predictions[prediction.case_id] = prediction
    return evaluate_predictions(cases, predictions)

def evaluate_predictions(
    cases: Iterable[EvaluationCase],
    predictions: dict[str, Prediction],
) -> dict[str, Any]:
    """Aggregate case-level scores into a stable report."""
    case_list = list(cases)
    case_scores: list[dict[str, Any]] = []
    missing: list[str] = []
    seen_case_ids: set[str] = set()

    for case in case_list:
        seen_case_ids.add(case.case_id)
        pred = predictions.get(case.case_id)
        if pred is None:
            missing.append(case.case_id)
            continue
        case_scores.append(score_prediction(case, pred))

    extra_prediction_ids = sorted(set(predictions) - seen_case_ids)
    warnings = []
    if not case_list:
        warnings.append("no evaluation cases loaded")
    if extra_prediction_ids:
        warnings.append(f"{len(extra_prediction_ids)} predictions did not match any evaluation case")

    latencies = [
        score["latency_ms"]
        for score in case_scores
        if isinstance(score.get("latency_ms"), (int, float))
    ]

    return {
        "summary": {
            "total_cases": len(case_list),
            "scored_cases": len(case_scores),
            "missing_predictions": len(missing),
            "extra_predictions": len(extra_prediction_ids),
            "total_predictions": len(predictions),
            "pass_rate": _avg(score["passed"] for score in case_scores),
            "citation_precision": _avg(score["citation_precision"] for score in case_scores),
            "citation_recall": _avg(score["citation_recall"] for score in case_scores),
            "retrieval_precision": _avg(score["retrieval_precision"] for score in case_scores),
            "retrieval_recall": _avg(score["retrieval_recall"] for score in case_scores),
            "answer_contains": _avg(score["answer_contains"] for score in case_scores),
            "number_accuracy": _avg(score["number_accuracy"] for score in case_scores),
            "no_answer_accuracy": _avg(score["no_answer_accuracy"] for score in case_scores),
            "calculation_accuracy": _avg(score["calculation_accuracy"] for score in case_scores),
            "answer_calculation_consistency": _avg(
                score["answer_calculation_consistency"] for score in case_scores
            ),
            "intent_accuracy": _avg(score["intent_accuracy"] for score in case_scores),
            "p95_latency_ms": _percentile(latencies, 95),
        },
        "missing_case_ids": missing,
        "extra_prediction_ids": extra_prediction_ids,
        "warnings": warnings,
        "cases": case_scores,
    }


def build_interview_report(
    cases: Iterable[EvaluationCase],
    predictions: dict[str, Prediction],
    *,
    ks: Iterable[int] = (1, 3, 5),
    candidate_field: str = "retrieved_chunks",
    worst_limit: int = 5,
) -> dict[str, Any]:
    """Build a compact, demo-friendly quality report for interview/readme use.

    The regular scorer is intentionally detailed and CI-oriented. This wrapper
    keeps the same underlying metrics but groups them into an easier narrative:
    answer correctness, citation grounding, retrieval recall, no-answer behavior,
    and latency. It stays offline and deterministic so it can be generated from
    saved JSONL predictions without a model service.
    """
    case_list = list(cases)
    score_report = evaluate_predictions(case_list, predictions)
    retrieval_report = diagnose_retrieval(
        case_list,
        predictions,
        ks=ks,
        candidate_field=candidate_field,
        worst_limit=worst_limit,
    )
    summary = score_report.get("summary", {})
    retrieval_summary = retrieval_report.get("summary", {})
    recall_at_k = retrieval_summary.get("recall_at_k") or {}
    recall_metric_k = _select_display_recall_k(recall_at_k)

    no_answer_cases = [
        item.case_id for item in case_list
        if item.expected_no_answer
    ]
    citation_cases = [
        item.case_id for item in case_list
        if item.expected_sources
    ]
    calculation_cases = [
        item.case_id for item in case_list
        if item.expected_calculations or item.expected_numbers
    ]

    score_by_id = {
        str(item.get("id")): item
        for item in score_report.get("cases", [])
        if item.get("id") is not None
    }

    return {
        "summary": {
            "total_cases": summary.get("total_cases", 0),
            "scored_cases": summary.get("scored_cases", 0),
            "missing_predictions": summary.get("missing_predictions", 0),
            "answer_pass_rate": summary.get("pass_rate", 1.0),
            "answer_contains": summary.get("answer_contains", 1.0),
            "number_accuracy": summary.get("number_accuracy", 1.0),
            "no_answer_accuracy": summary.get("no_answer_accuracy", 1.0),
            "citation_recall": summary.get("citation_recall", 1.0),
            "retrieval_recall": summary.get("retrieval_recall", 1.0),
            "retrieval_recall_at_k": retrieval_summary.get("recall_at_k", {}),
            "retrieval_mrr": retrieval_summary.get("mrr", 1.0),
            "retrieval_full_recall_rate": retrieval_summary.get("full_recall_rate", 1.0),
            "intent_accuracy": summary.get("intent_accuracy", 1.0),
            "p95_latency_ms": summary.get("p95_latency_ms"),
        },
        "case_groups": {
            "no_answer": no_answer_cases,
            "citation": citation_cases,
            "calculation_or_number": calculation_cases,
        },
        "resume_metrics": [
            {
                "name": "Golden answer pass rate",
                "value": _format_percent(summary.get("pass_rate", 1.0)),
                "source": "offline JSONL eval",
            },
            {
                "name": "Citation recall",
                "value": _format_percent(summary.get("citation_recall", 1.0)),
                "source": "expected source match",
            },
            {
                "name": f"Retrieval Recall@{recall_metric_k}",
                "value": _format_percent(recall_at_k.get(str(recall_metric_k), 1.0)),
                "source": f"{candidate_field} diagnostics",
            },
            {
                "name": "Retrieval MRR",
                "value": _format_decimal(retrieval_summary.get("mrr", 1.0)),
                "source": f"{candidate_field} diagnostics",
            },
            {
                "name": "No-answer accuracy",
                "value": _format_percent(summary.get("no_answer_accuracy", 1.0)),
                "source": "expected_no_answer cases",
            },
        ],
        "weak_cases": _select_interview_weak_cases(
            score_by_id,
            retrieval_report.get("worst_cases", []),
            limit=worst_limit,
        ),
        "score_report": score_report,
        "retrieval_report": retrieval_report,
    }


def build_failure_analysis_markdown(
    cases: Iterable[EvaluationCase],
    predictions: dict[str, Prediction],
    *,
    limit: int | None = None,
) -> str:
    """Build a human-readable failure analysis report for real-document evals."""
    case_list = list(cases)
    score_report = evaluate_predictions(case_list, predictions)
    prediction_by_id = predictions
    failed = [
        score for score in score_report.get("cases", [])
        if not score.get("passed")
    ]
    if limit is not None:
        failed = failed[:max(0, int(limit))]

    lines = [
        "# FinQuery eval failure analysis",
        "",
        "## Summary",
        "",
        f"- Total cases: {score_report.get('summary', {}).get('total_cases', 0)}",
        f"- Scored cases: {score_report.get('summary', {}).get('scored_cases', 0)}",
        f"- Pass rate: {_format_percent(score_report.get('summary', {}).get('pass_rate', 0.0))}",
        f"- Failed cases included: {len(failed)}",
        "",
        "## Failure categories",
        "",
    ]
    category_counts: dict[str, int] = {}
    for score in score_report.get("cases", []):
        if score.get("passed"):
            continue
        category = str(score.get("failure_category") or "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {category}: {count}")
    if not category_counts:
        lines.append("- none")

    lines.extend(["", "## Failed cases", ""])
    case_by_id = {case.case_id: case for case in case_list}
    for score in failed:
        case_id = str(score.get("id"))
        case = case_by_id.get(case_id)
        prediction = prediction_by_id.get(case_id)
        lines.extend([
            f"### {case_id}",
            "",
            f"- Failure category: `{score.get('failure_category')}`",
            f"- Tags: {', '.join(score.get('tags', [])) or '-'}",
            f"- Scores: answer={score.get('answer_contains')}, number={score.get('number_accuracy')}, citation={score.get('citation_recall')}, retrieval={score.get('retrieval_recall')}, intent={score.get('intent_accuracy')}",
            f"- Question: {case.question if case else ''}",
            f"- Expected answer contains: {', '.join(score.get('expected_answer_contains') or []) or '-'}",
            f"- Expected numbers: {', '.join(score.get('expected_numbers') or []) or '-'}",
            f"- Expected sources: {_format_source_list(score.get('expected_sources') or [])}",
            f"- Actual sources: {_format_source_list(score.get('actual_sources') or [])}",
            f"- Retrieved chunks: {_format_source_list(score.get('actual_retrieved_chunks') or [])}",
        ])
        if prediction and prediction.error:
            lines.append(f"- Prediction error: `{prediction.error}` {prediction.error_detail or ''}")
        lines.extend([
            "",
            "Actual answer:",
            "",
            "```text",
            _preview_text(prediction.answer if prediction else "", limit=1200),
            "```",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def audit_evaluation_fixtures(
    cases: Iterable[EvaluationCase],
    *,
    min_cases: int = 1,
    required_tags: Iterable[str] = (),
    require_expected_source: bool = False,
    require_expected_intent: bool = False,
) -> dict[str, Any]:
    """Return quality and coverage diagnostics for golden/replay fixtures."""
    if min_cases < 0:
        raise ValueError("min_cases must be >= 0")
    required_tag_set = {str(tag) for tag in required_tags if str(tag)}
    case_list = list(cases)
    tag_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    coverage = {
        "expected_sources": 0,
        "expected_answer_contains": 0,
        "expected_numbers": 0,
        "expected_no_answer": 0,
        "expected_calculations": 0,
        "expected_intent": 0,
        "document_names": 0,
    }
    issues: list[dict[str, Any]] = []

    for case in case_list:
        for tag in case.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if case.expected_intent:
            intent_counts[case.expected_intent] = intent_counts.get(case.expected_intent, 0) + 1
        if case.expected_sources:
            coverage["expected_sources"] += 1
        if case.expected_answer_contains:
            coverage["expected_answer_contains"] += 1
        if case.expected_numbers:
            coverage["expected_numbers"] += 1
        if case.expected_no_answer:
            coverage["expected_no_answer"] += 1
        if case.expected_calculations:
            coverage["expected_calculations"] += 1
        if case.expected_intent:
            coverage["expected_intent"] += 1
        if case.document_names:
            coverage["document_names"] += 1

        expected_signal_count = sum(
            bool(value)
            for value in (
                case.expected_sources,
                case.expected_answer_contains,
                case.expected_numbers,
                case.expected_no_answer,
                case.expected_calculations,
                case.expected_intent,
            )
        )
        if expected_signal_count == 0:
            issues.append({
                "severity": "warning",
                "case_id": case.case_id,
                "code": "missing_expected_signal",
                "message": "case has no expected sources, answer checks, numbers, no-answer flag, calculations, or intent",
            })
        if require_expected_source and not case.expected_sources:
            issues.append({
                "severity": "error",
                "case_id": case.case_id,
                "code": "missing_expected_source",
                "message": "case is missing expected_sources",
            })
        if require_expected_intent and not case.expected_intent:
            issues.append({
                "severity": "error",
                "case_id": case.case_id,
                "code": "missing_expected_intent",
                "message": "case is missing expected_intent",
            })

    if len(case_list) < min_cases:
        issues.append({
            "severity": "error",
            "case_id": None,
            "code": "min_cases_not_met",
            "message": f"fixture has {len(case_list)} cases; minimum is {min_cases}",
        })

    missing_required_tags = sorted(required_tag_set - set(tag_counts))
    for tag in missing_required_tags:
        issues.append({
            "severity": "error",
            "case_id": None,
            "code": "missing_required_tag",
            "tag": tag,
            "message": f"required tag {tag!r} is not represented",
        })

    errors = [item for item in issues if item["severity"] == "error"]
    warnings = [item for item in issues if item["severity"] == "warning"]
    total = len(case_list)
    coverage_rates = {
        key: (value / total if total else 0.0)
        for key, value in coverage.items()
    }
    return {
        "passed": not errors,
        "summary": {
            "total_cases": total,
            "min_cases": min_cases,
            "tag_counts": dict(sorted(tag_counts.items())),
            "intent_counts": dict(sorted(intent_counts.items())),
            "coverage_counts": coverage,
            "coverage_rates": coverage_rates,
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
        "required_tags": sorted(required_tag_set),
        "missing_required_tags": missing_required_tags,
        "issues": issues,
        "errors": errors,
        "warnings": warnings,
    }


def diagnose_retrieval(
    cases: Iterable[EvaluationCase],
    predictions: dict[str, Prediction],
    *,
    ks: Iterable[int] = (1, 3, 5),
    candidate_field: str = "retrieved_chunks",
    worst_limit: int = 10,
) -> dict[str, Any]:
    """Return source-level retrieval diagnostics for saved predictions.

    The report is intentionally answer-independent. It helps separate retrieval
    failures from generation/validation failures by showing where expected
    sources appeared in the retrieved candidate list.
    """
    normalized_ks = _normalize_retrieval_ks(ks)
    if candidate_field not in {"retrieved_chunks", "sources"}:
        raise ValueError("candidate_field must be 'retrieved_chunks' or 'sources'")
    if worst_limit < 0:
        raise ValueError("worst_limit must be >= 0")

    case_list = list(cases)
    diagnostics: list[dict[str, Any]] = []
    total_expected_sources = 0
    hits_at_k = {str(k): 0 for k in normalized_ks}
    reciprocal_ranks: list[float] = []
    full_recall_count = 0
    cases_with_expected_sources = 0
    missing_predictions: list[str] = []
    no_expected_source_cases: list[str] = []

    for case in case_list:
        expected_sources = case.expected_sources
        if not expected_sources:
            no_expected_source_cases.append(case.case_id)
            diagnostics.append({
                "id": case.case_id,
                "tags": list(case.tags),
                "expected_source_count": 0,
                "retrieved_count": 0,
                "matched_expected_count": 0,
                "full_recall": True,
                "best_rank": None,
                "reciprocal_rank": 0.0,
                "hit_ranks": [],
                "missed_expected_sources": [],
                "missing_prediction": case.case_id not in predictions,
            })
            if case.case_id not in predictions:
                missing_predictions.append(case.case_id)
            continue

        cases_with_expected_sources += 1
        total_expected_sources += len(expected_sources)
        prediction = predictions.get(case.case_id)
        if prediction is None:
            missing_predictions.append(case.case_id)
            candidates: tuple[dict[str, Any], ...] = ()
        else:
            candidates = getattr(prediction, candidate_field)

        expected_details = []
        hit_ranks = []
        for expected in expected_sources:
            rank = _first_matching_rank(expected, candidates)
            hit_ranks.append(rank)
            expected_details.append({
                "expected": _expected_source_to_dict(expected),
                "rank": rank,
                "matched": rank is not None,
            })
            if rank is not None:
                for k in normalized_ks:
                    if rank <= k:
                        hits_at_k[str(k)] += 1

        matched_count = sum(1 for rank in hit_ranks if rank is not None)
        best_rank = min((rank for rank in hit_ranks if rank is not None), default=None)
        reciprocal_rank = 1.0 / best_rank if best_rank else 0.0
        reciprocal_ranks.append(reciprocal_rank)
        full_recall = matched_count == len(expected_sources)
        if full_recall:
            full_recall_count += 1

        diagnostics.append({
            "id": case.case_id,
            "tags": list(case.tags),
            "expected_source_count": len(expected_sources),
            "retrieved_count": len(candidates),
            "matched_expected_count": matched_count,
            "full_recall": full_recall,
            "best_rank": best_rank,
            "reciprocal_rank": reciprocal_rank,
            "hit_ranks": hit_ranks,
            "expected_sources": expected_details,
            "missed_expected_sources": [
                item["expected"] for item in expected_details if not item["matched"]
            ],
            "missing_prediction": prediction is None,
        })

    worst_cases = sorted(
        (
            item for item in diagnostics
            if item["expected_source_count"] > 0
        ),
        key=lambda item: (
            item["full_recall"],
            item["best_rank"] is not None,
            item["best_rank"] or 10**9,
            -item["expected_source_count"],
            item["id"],
        ),
    )[:worst_limit]

    recall_at_k = {
        key: (hits / total_expected_sources if total_expected_sources else 1.0)
        for key, hits in hits_at_k.items()
    }
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 1.0
    full_recall_rate = (
        full_recall_count / cases_with_expected_sources
        if cases_with_expected_sources
        else 1.0
    )

    return {
        "summary": {
            "total_cases": len(case_list),
            "cases_with_expected_sources": cases_with_expected_sources,
            "cases_without_expected_sources": len(no_expected_source_cases),
            "cases_with_predictions": len(case_list) - len(missing_predictions),
            "missing_predictions": len(missing_predictions),
            "total_expected_sources": total_expected_sources,
            "candidate_field": candidate_field,
            "recall_at_k": recall_at_k,
            "mrr": mrr,
            "full_recall_rate": full_recall_rate,
        },
        "missing_case_ids": missing_predictions,
        "no_expected_source_case_ids": no_expected_source_cases,
        "cases": diagnostics,
        "worst_cases": worst_cases,
        "warnings": (
            ["no expected sources found in evaluation cases"]
            if total_expected_sources == 0
            else []
        ),
    }


def diagnose_candidate_stages(
    cases: Iterable[EvaluationCase],
    predictions: dict[str, Prediction],
) -> dict[str, Any]:
    """Classify where labeled evidence disappears in the retrieval pipeline."""
    counts = {
        "not_in_bm25_or_dense_40": 0,
        "lost_during_rrf": 0,
        "lost_during_rerank": 0,
        "lost_during_final_top_k": 0,
        "evidence_present_answer_failed": 0,
        "passed": 0,
        "untraced": 0,
    }
    score_by_id = {
        item["id"]: item
        for item in evaluate_predictions(cases, predictions).get("cases", [])
    }
    items = []
    for case in cases:
        prediction = predictions.get(case.case_id)
        stages = (prediction.retrieval_debug.get("candidate_stages", {}) if prediction else {})
        if not case.expected_sources:
            items.append({"id": case.case_id, "failure_stage": "passed", "reason": "no_expected_sources"})
            counts["passed"] += 1
            continue
        if not stages:
            items.append({"id": case.case_id, "failure_stage": "untraced"})
            counts["untraced"] += 1
            continue

        def hits(stage: str, limit: int) -> bool:
            rows = stages.get(stage, [])[:limit]
            normalized = [
                {
                    "doc_id": row.get("evidence_id"),
                    "filename": row.get("document_id"),
                    "page": row.get("page"),
                }
                for row in rows
            ]
            return any(expected.matches(row) for expected in case.expected_sources for row in normalized)

        bm25_hit = hits("bm25", 40)
        dense_hit = hits("dense", 40)
        if not bm25_hit and not dense_hit:
            failure_stage = "not_in_bm25_or_dense_40"
        elif not hits("rrf", 40):
            failure_stage = "lost_during_rrf"
        elif not hits("reranker", 20):
            failure_stage = "lost_during_rerank"
        elif not hits("final", 5):
            failure_stage = "lost_during_final_top_k"
        elif not score_by_id.get(case.case_id, {}).get("passed", False):
            failure_stage = "evidence_present_answer_failed"
        else:
            failure_stage = "passed"
        counts[failure_stage] += 1
        items.append({
            "id": case.case_id,
            "failure_stage": failure_stage,
            "bm25_hit_40": bm25_hit,
            "dense_hit_40": dense_hit,
            "rrf_hit_40": hits("rrf", 40),
            "reranker_hit_20": hits("reranker", 20),
            "final_hit_5": hits("final", 5),
        })
    stage_recall = {}
    total_expected = sum(len(case.expected_sources) for case in cases)
    stages_and_limits = (
        ("bm25", 8), ("bm25", 20), ("bm25", 40),
        ("dense", 8), ("dense", 20), ("dense", 40),
        ("rrf", 8), ("rrf", 20), ("rrf", 40),
        ("reranker", 5), ("reranker", 8), ("reranker", 20),
        ("final", 5),
    )
    for stage, limit in stages_and_limits:
        hits = 0
        for case in cases:
            prediction = predictions.get(case.case_id)
            debug = prediction.retrieval_debug if prediction else {}
            rows = debug.get("candidate_stages", {}).get(stage, [])[:limit]
            normalized = [
                {"doc_id": row.get("evidence_id"), "filename": row.get("document_id"), "page": row.get("page")}
                for row in rows
            ]
            hits += sum(any(expected.matches(row) for row in normalized) for expected in case.expected_sources)
        stage_recall[f"{stage}_recall_at_{limit}"] = hits / total_expected if total_expected else 1.0
    return {
        "summary": {**counts, "total_expected_sources": total_expected, **stage_recall},
        "cases": items,
    }


def trace_to_replay_case(trace: dict[str, Any]) -> EvaluationCase:
    """Convert one TraceLogger row dict into a replay case.

    The generated case intentionally stores references to sources and documents,
    but not the full retrieved context. This keeps replay fixtures small and
    avoids copying potentially sensitive document content into golden files.
    """
    trace_id = trace.get("trace_id")
    question = trace.get("query_original")
    if not trace_id or not question:
        raise ValueError("trace row missing trace_id or query_original")

    filters = _loads_json_field(trace.get("filter_conditions"), "filter_conditions", dict, {})
    sources = _loads_json_field(trace.get("sources_json"), "sources_json", list, [])
    diagnostics = _loads_json_field(trace.get("diagnostics_json"), "diagnostics_json", dict, {})

    return EvaluationCase(
        case_id=str(trace_id),
        question=str(question),
        document_names=tuple(str(item) for item in filters.get("doc_names", []) or []),
        expected_sources=tuple(ExpectedSource.from_dict(item) for item in sources),
        expected_answer_contains=(),
        expected_numbers=tuple(_extract_numbers(trace.get("answer") or "")),
        expected_no_answer=_looks_like_no_answer(trace.get("answer") or ""),
        expected_intent=str(trace.get("intent")) if trace.get("intent") else None,
        tags=("trace_replay",),
        metadata={
            "trace_id": trace_id,
            "tenant_id": trace.get("tenant_id"),
            "model_name": trace.get("model_name"),
            "created_at": trace.get("created_at"),
            "n_results": filters.get("n_results"),
            "diagnostics": diagnostics,
        },
    )


def export_replay_cases_from_traces(
    traces: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> list[EvaluationCase]:
    cases = [trace_to_replay_case(trace) for trace in traces]
    _ensure_unique_case_ids(cases, "trace replay export")
    write_jsonl(output_path, (case.to_dict() for case in cases))
    return cases


def feedback_to_replay_case(feedback: dict[str, Any], trace: dict[str, Any]) -> EvaluationCase:
    """Convert one feedback row and its trace into a replay case.

    Feedback metadata is kept outside expected fields so exported cases still
    load as regular EvaluationCase fixtures. Down-rated cases are tagged for
    triage and replay prioritization.
    """
    case = trace_to_replay_case(trace)
    rating = feedback.get("rating")
    tags = list(case.tags)
    if rating:
        tags.append(f"feedback_{rating}")
    tags.append("feedback_replay")

    metadata = dict(case.metadata)
    metadata.update({
        "feedback_id": feedback.get("feedback_id"),
        "feedback_rating": rating,
        "feedback_comment": feedback.get("comment"),
        "feedback_created_at": feedback.get("created_at"),
    })

    return replace(
        case,
        tags=tuple(dict.fromkeys(tags)),
        metadata=metadata,
    )


def export_replay_cases_from_feedback(
    feedback_rows: Iterable[dict[str, Any]],
    trace_lookup,
    output_path: str | Path,
) -> list[EvaluationCase]:
    """Export replay cases for feedback rows that still have matching traces."""
    cases = []
    for feedback in feedback_rows:
        trace = trace_lookup(feedback.get("trace_id"))
        if trace is None:
            continue
        cases.append(feedback_to_replay_case(feedback, trace))
    _ensure_unique_case_ids(cases, "feedback replay export")
    write_jsonl(output_path, (case.to_dict() for case in cases))
    return cases


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    regression_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare two evaluation reports and flag metric regressions.

    The comparison is intentionally deterministic and schema-stable so it can
    be used as a CI quality gate once project-specific thresholds are defined.
    """
    _validate_report_shape(baseline, "baseline")
    _validate_report_shape(candidate, "candidate")
    tolerance = float(regression_tolerance)
    if tolerance < 0:
        raise ValueError("regression_tolerance must be >= 0")

    baseline_summary = baseline.get("summary", {})
    candidate_summary = candidate.get("summary", {})
    metric_names = [
        "pass_rate",
        "citation_precision",
        "citation_recall",
        "retrieval_precision",
        "retrieval_recall",
        "answer_contains",
        "number_accuracy",
        "no_answer_accuracy",
        "calculation_accuracy",
        "answer_calculation_consistency",
        "intent_accuracy",
    ]

    metrics = {}
    regressions = []
    regression_details = []
    failure_reasons = []
    for name in metric_names:
        base_value = _optional_float(baseline_summary.get(name)) or 0.0
        cand_value = _optional_float(candidate_summary.get(name)) or 0.0
        delta = cand_value - base_value
        detail = {
            "metric": name,
            "baseline": base_value,
            "candidate": cand_value,
            "delta": delta,
            "allowed_drop": tolerance,
        }
        metrics[name] = {
            "baseline": base_value,
            "candidate": cand_value,
            "delta": delta,
        }
        if delta < -tolerance:
            regressions.append(name)
            regression_details.append(detail)
            failure_reasons.append(
                "metric %s regressed by %.6f (baseline %.6f -> candidate %.6f, tolerance %.6f)"
                % (name, delta, base_value, cand_value, regression_tolerance)
            )

    latency_delta = None
    base_latency = _optional_float(baseline_summary.get("p95_latency_ms"))
    cand_latency = _optional_float(candidate_summary.get("p95_latency_ms"))
    if base_latency is not None and cand_latency is not None:
        latency_delta = cand_latency - base_latency
    latency = {
        "baseline_p95_ms": base_latency,
        "candidate_p95_ms": cand_latency,
        "delta_ms": latency_delta,
    }

    baseline_cases = {
        item["id"]: item
        for item in baseline.get("cases", [])
        if "id" in item
    }
    candidate_cases = {
        item["id"]: item
        for item in candidate.get("cases", [])
        if "id" in item
    }
    newly_failed = sorted(
        case_id
        for case_id, base_case in baseline_cases.items()
        if base_case.get("passed") is True
        and candidate_cases.get(case_id, {}).get("passed") is False
    )
    newly_passed = sorted(
        case_id
        for case_id, cand_case in candidate_cases.items()
        if cand_case.get("passed") is True
        and baseline_cases.get(case_id, {}).get("passed") is False
    )
    case_failure_details = [
        {
            "id": case_id,
            "baseline_passed": baseline_cases.get(case_id, {}).get("passed"),
            "candidate_passed": candidate_cases.get(case_id, {}).get("passed"),
            "tags": candidate_cases.get(case_id, {}).get("tags", []),
        }
        for case_id in newly_failed
    ]
    for case_id in newly_failed:
        failure_reasons.append("case %s newly failed" % case_id)

    passed = not regressions and not newly_failed

    return {
        "passed": passed,
        "regression_tolerance": regression_tolerance,
        "metric_deltas": metrics,
        "regressions": regressions,
        "regression_details": regression_details,
        "newly_failed": newly_failed,
        "newly_passed": newly_passed,
        "case_failure_details": case_failure_details,
        "failure_reasons": failure_reasons,
        "latency": latency,
        "p95_latency_delta_ms": latency_delta,
        "baseline_missing_predictions": baseline_summary.get("missing_predictions", 0),
        "candidate_missing_predictions": candidate_summary.get("missing_predictions", 0),
    }

def _normalize_retrieval_ks(ks: Iterable[int]) -> tuple[int, ...]:
    normalized = sorted({int(k) for k in ks})
    if not normalized:
        raise ValueError("at least one k value is required")
    if any(k < 1 for k in normalized):
        raise ValueError("k values must be >= 1")
    return tuple(normalized)


def _first_matching_rank(
    expected: ExpectedSource,
    candidates: tuple[dict[str, Any], ...],
) -> int | None:
    for idx, candidate in enumerate(candidates, 1):
        if expected.matches(candidate):
            return idx
    return None


def _expected_source_to_dict(expected: ExpectedSource) -> dict[str, Any]:
    return {
        "filename": expected.filename,
        "page": expected.page,
        "chunk_id": expected.chunk_id,
    }


def _select_interview_weak_cases(
    score_by_id: dict[str, dict[str, Any]],
    worst_retrieval_cases: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Return a compact weak-case list for human review."""
    if limit <= 0:
        return []

    weak_by_id: dict[str, dict[str, Any]] = {}
    for case_id, score in score_by_id.items():
        if score.get("passed"):
            continue
        weak_by_id[case_id] = {
            "id": case_id,
            "passed": False,
            "failure_category": score.get("failure_category"),
            "citation_recall": score.get("citation_recall"),
            "retrieval_recall": score.get("retrieval_recall"),
            "answer_contains": score.get("answer_contains"),
            "number_accuracy": score.get("number_accuracy"),
            "no_answer_accuracy": score.get("no_answer_accuracy"),
            "intent_accuracy": score.get("intent_accuracy"),
            "expected_answer_contains": score.get("expected_answer_contains", []),
            "expected_numbers": score.get("expected_numbers", []),
            "expected_sources": score.get("expected_sources", []),
            "actual_answer_preview": score.get("actual_answer_preview", ""),
            "actual_sources": score.get("actual_sources", []),
            "actual_retrieved_chunks": score.get("actual_retrieved_chunks", []),
            "prediction_error": score.get("prediction_error"),
            "tags": score.get("tags", []),
            "reason": "score_failed",
        }

    for item in worst_retrieval_cases:
        case_id = str(item.get("id"))
        if not case_id or item.get("full_recall"):
            continue
        weak_by_id.setdefault(case_id, {
            "id": case_id,
            "passed": score_by_id.get(case_id, {}).get("passed"),
            "best_rank": item.get("best_rank"),
            "matched_expected_count": item.get("matched_expected_count"),
            "expected_source_count": item.get("expected_source_count"),
            "tags": item.get("tags", []),
            "reason": "retrieval_miss",
        })

    return sorted(
        weak_by_id.values(),
        key=lambda item: (
            item.get("passed") is True,
            item.get("best_rank") is not None,
            item.get("best_rank") or 10**9,
            str(item.get("id")),
        ),
    )[:limit]


def _format_percent(value: Any) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        numeric = 0.0
    return f"{numeric * 100:.1f}%"


def _format_decimal(value: Any) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        numeric = 0.0
    return f"{numeric:.3f}"


def _select_display_recall_k(recall_at_k: dict[str, Any]) -> int:
    if "5" in recall_at_k:
        return 5
    numeric_ks = []
    for key in recall_at_k:
        try:
            numeric_ks.append(int(key))
        except (TypeError, ValueError):
            continue
    return max(numeric_ks) if numeric_ks else 5


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [item for _, item in _read_jsonl_rows(path)]


def _atomic_write_text(path: str | Path, content: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=out.parent,
            prefix=f".{out.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_name = fh.name
            fh.write(content)
        os.replace(tmp_name, out)
    except Exception:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _read_jsonl_rows(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"invalid JSONL at {path}:{line_no}: row must be an object")
            rows.append((line_no, item))
    return rows


def _ensure_mapping(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")


def _ensure_list_fields(data: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    for field_name in fields:
        if data.get(field_name) is not None and not isinstance(data.get(field_name), list):
            raise ValueError(f"{label} field {field_name} must be a list")


def _intent_score(expected_intent: str | None, predicted_intent: str | None) -> float:
    if not expected_intent:
        return 1.0
    return 1.0 if predicted_intent == expected_intent else 0.0


def _answer_calculation_consistency(answer: str, calculations: tuple[dict[str, Any], ...]) -> float:
    if not calculations:
        return 1.0
    validation = validate_answer_calculations(answer, list(calculations))
    return 1.0 if validation.ok else 0.0

def _calculation_score(
    expected_calculations: tuple[ExpectedCalculation, ...],
    predicted_calculations: tuple[dict[str, Any], ...],
) -> float:
    if not expected_calculations:
        return 1.0
    if not predicted_calculations:
        return 0.0

    predicted_by_id = {
        str(item.get("id") or item.get("calc_id") or item.get("operation")): item
        for item in predicted_calculations
    }
    hits = 0
    for expected in expected_calculations:
        predicted = predicted_by_id.get(expected.calc_id)
        if not predicted:
            continue
        if _prediction_matches_expected_calculation(expected, predicted):
            hits += 1
    return hits / len(expected_calculations)


def _prediction_matches_expected_calculation(
    expected: ExpectedCalculation,
    predicted: dict[str, Any],
) -> bool:
    if predicted.get("operation") and predicted.get("operation") != expected.operation:
        return False
    if expected.unit and predicted.get("unit") and predicted.get("unit") != expected.unit:
        return False

    predicted_value = _decimal_or_none(predicted.get("value"))
    if predicted_value is None:
        return False

    expected_value = _decimal_or_none(expected.expected_value)
    tolerance = _decimal_or_none(expected.tolerance)
    if expected_value is None or tolerance is None:
        return False

    deterministic = _run_expected_calculation(expected)
    if deterministic is None:
        return False

    return (
        abs(predicted_value - expected_value) <= abs(tolerance)
        and abs(deterministic - expected_value) <= abs(tolerance)
    )


def _run_expected_calculation(expected: ExpectedCalculation) -> Decimal | None:
    args = expected.args
    operation = expected.operation
    if operation == "growth_rate":
        result = growth_rate(args.get("current"), args.get("previous"))
    elif operation == "percentage_share":
        result = percentage_share(args.get("part"), args.get("total"))
    elif operation == "sum_values":
        result = sum_values(list(args.get("values", [])))
    elif operation == "verify_sum":
        result = verify_sum(
            list(args.get("components", [])),
            args.get("reported_total"),
            tolerance=args.get("tolerance", "0.01"),
        )
    elif operation == "convert_scale":
        result = convert_scale(
            args.get("value"),
            args.get("from_scale", ""),
            args.get("to_scale", ""),
        )
    elif operation == "format_ratio_percent":
        result = format_ratio_percent(args.get("value"))
    else:
        return None
    return result.value if result.ok else None


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None

def _score_sources(
    expected_sources: tuple[ExpectedSource, ...],
    candidates: tuple[dict[str, Any], ...],
) -> dict[str, float]:
    if not expected_sources:
        return {"precision": 1.0, "recall": 1.0}
    if not candidates:
        return {"precision": 0.0, "recall": 0.0}

    matched_expected = {
        idx
        for idx, expected in enumerate(expected_sources)
        if any(expected.matches(candidate) for candidate in candidates)
    }
    matched_candidates = sum(
        1 for candidate in candidates
        if any(expected.matches(candidate) for expected in expected_sources)
    )
    return {
        "precision": matched_candidates / len(candidates),
        "recall": len(matched_expected) / len(expected_sources),
    }


def _contains_score(answer_lower: str, expected_phrases: tuple[str, ...]) -> float:
    if not expected_phrases:
        return 1.0
    normalized_answer = _normalize_text_for_match(answer_lower)
    hits = 0
    for phrase in expected_phrases:
        alternatives = _expected_alternatives(phrase)
        if any(_expected_phrase_matches(normalized_answer, item) for item in alternatives):
            hits += 1
    return hits / len(expected_phrases)


def _expected_phrase_matches(normalized_answer: str, expected: str) -> bool:
    normalized_expected = _normalize_text_for_match(expected)
    if normalized_expected in normalized_answer:
        return True
    percent_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)", normalized_expected)
    if percent_match:
        value = re.escape(percent_match.group(1))
        return bool(re.search(rf"\b{value}\s*(?:%|percent|per cent)\b", normalized_answer))
    return False


def _number_score(answer: str, expected_numbers: tuple[str, ...]) -> float:
    if not expected_numbers:
        return 1.0
    found = set(_extract_numbers(answer))
    hits = 0
    for number in expected_numbers:
        expected = _normalize_number(number)
        if expected in found or _number_text_match(answer, expected):
            hits += 1
    return hits / len(expected_numbers)


def _no_answer_score(answer_lower: str, expected_no_answer: bool) -> float:
    if not expected_no_answer:
        return 1.0
    return 1.0 if _looks_like_no_answer(answer_lower) else 0.0


def _looks_like_no_answer(answer: str) -> bool:
    text = answer.lower()
    markers = [
        "couldn't find",
        "could not find",
        "sufficiently relevant",
        "no relevant",
        "not found",
        "无法",
        "没有找到",
        "未找到",
        "不足以",
    ]
    return any(marker in text for marker in markers)


def _extract_numbers(text: str) -> list[str]:
    return [_normalize_number(match.group(0)) for match in _NUMBER_RE.finditer(text)]


def _normalize_number(value: str) -> str:
    return value.replace(",", "").strip().rstrip("%")


def _number_text_match(answer: str, expected: str) -> bool:
    """Fallback for simple scaled forms, e.g. 42.2 vs '$42.2 million'."""
    if not expected:
        return False
    compact_answer = answer.replace(",", "").lower()
    if re.search(rf"(?<!\d){re.escape(expected)}(?!\d)", compact_answer):
        return True
    return False


def _expected_alternatives(phrase: str) -> list[str]:
    text = str(phrase)
    # Allow explicit OR groups in future fixtures without changing the schema.
    return [part.strip() for part in text.split("||") if part.strip()] or [text]


def _normalize_text_for_match(text: str) -> str:
    lowered = str(text).lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _classify_failure(
    case: EvaluationCase,
    prediction: Prediction,
    *,
    passed: bool,
    source_recall: float,
    retrieval_recall: float,
    contains_score: float,
    number_score: float,
    no_answer_score: float,
    intent_accuracy: float,
) -> str | None:
    if passed:
        return None
    if prediction.error:
        return "runtime_error"
    if case.expected_sources and retrieval_recall <= 0:
        return "retrieval_miss"
    if case.expected_sources and source_recall <= 0:
        return "citation_miss"
    if case.expected_numbers and number_score < 1.0:
        return "number_extraction"
    if case.expected_answer_contains and contains_score < 1.0:
        return "answer_mismatch"
    if case.expected_no_answer and no_answer_score < 1.0:
        return "no_answer_failure"
    if case.expected_intent and intent_accuracy < 1.0:
        return "intent_mismatch"
    return "score_failed"


def _preview_text(text: str, *, limit: int = 500) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _compact_source_dict(source: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("filename", "page", "chunk_id", "doc_id", "parent_id", "section_path"):
        value = source.get(key)
        if value not in (None, "", []):
            compact[key] = value
    if "score" in source:
        compact["score"] = source.get("score")
    if "rerank_score" in source:
        compact["rerank_score"] = source.get("rerank_score")
    return compact


def _format_source_list(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "-"
    parts = []
    for item in sources[:8]:
        filename = item.get("filename") or _filename_from_doc_id(str(item.get("doc_id", ""))) or "?"
        page = item.get("page", "?")
        chunk = item.get("chunk_id") or item.get("doc_id") or ""
        suffix = f" ({chunk})" if chunk else ""
        parts.append(f"{filename}:p{page}{suffix}")
    if len(sources) > 8:
        parts.append(f"... +{len(sources) - 8} more")
    return "; ".join(parts)


def _filename_from_doc_id(doc_id: str) -> str | None:
    prefix = doc_id.split("::", 1)[0]
    if prefix.startswith("user_"):
        parts = prefix.split("_")
        if len(parts) > 2:
            return "_".join(parts[2:])
    return prefix or None


def _validate_report_shape(report: dict[str, Any], label: str) -> None:
    if not isinstance(report.get("summary", {}), dict):
        raise ValueError(f"{label} report summary must be an object")
    if not isinstance(report.get("cases", []), list):
        raise ValueError(f"{label} report cases must be a list")
    for idx, case in enumerate(report.get("cases", [])):
        if not isinstance(case, dict):
            raise ValueError(f"{label} report cases[{idx}] must be an object")
        if not case.get("id"):
            raise ValueError(f"{label} report cases[{idx}] missing id")


def _ensure_unique_case_ids(cases: Iterable[EvaluationCase], label: str) -> None:
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            raise ValueError(f"duplicate replay case id {case.case_id!r} in {label}")
        seen.add(case.case_id)


def _loads_json_field(value: Any, label: str, expected_type: type, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, expected_type):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(parsed, expected_type):
        type_name = "object" if expected_type is dict else "array"
        raise ValueError(f"{label} must be a JSON {type_name}")
    return parsed


def _loads_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: Iterable[Any]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return mean(items)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
