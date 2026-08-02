"""Shared, production-independent helpers for Financial RAG Benchmark v1."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

QUESTION_ANSWER_TYPES = {
    "text", "numeric", "currency", "percentage", "date",
    "comparison", "aggregation", "financial_volume", "no_answer",
}
DIFFICULTIES = {"easy", "medium", "hard"}
SOURCE_TYPES = {"text", "table", "table_row", "front_matter"}
CALCULATION_OPERATIONS = {"growth_rate", "difference", "sum", "ratio", "custom"}


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def document_identity_payload(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": document["document_id"],
        "filename": document["filename"],
        "file_sha256": document["file_sha256"],
        "page_count": int(document["page_count"]),
        "chunk_count": int(document["chunk_count"]),
    }


def document_identity_hash(document: dict[str, Any]) -> str:
    return stable_json_hash(document_identity_payload(document))


def corpus_hash(documents: Iterable[dict[str, Any]]) -> str:
    payload = [document_identity_payload(document) for document in sorted(documents, key=lambda item: item["document_id"])]
    return stable_json_hash(payload)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def parse_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None


def normalized_question(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold(), flags=re.UNICODE).strip()


def _source_errors(source: dict[str, Any], *, corpus_by_id: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for field in ("document_id", "filename", "page", "evidence_type", "source_verified"):
        if field not in source:
            errors.append(f"source missing {field}")
    if errors:
        return errors
    document = corpus_by_id.get(source["document_id"])
    if document is None:
        return [f"source document_id not in corpus: {source['document_id']}"]
    if source["filename"] != document["filename"]:
        errors.append("source filename does not match corpus document")
    page = source["page"]
    if not isinstance(page, int) or isinstance(page, bool):
        errors.append("source page must be an integer")
    elif not 1 <= page <= int(document["page_count"]):
        errors.append("source page is outside the document range")
    if source["evidence_type"] not in SOURCE_TYPES:
        errors.append(f"unsupported evidence_type: {source['evidence_type']}")
    if not isinstance(source["source_verified"], bool):
        errors.append("source_verified must be boolean")
    for optional in ("section", "table_title", "row_label", "column_header", "period", "unit", "scale", "display_scale"):
        if optional in source and source[optional] is not None and not isinstance(source[optional], str):
            errors.append(f"source {optional} must be a string or null")
    return errors


def validate_dataset(*, corpus: dict[str, Any], questions: list[dict[str, Any]], labels: list[dict[str, Any]], review_records: list[dict[str, Any]], draft: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    documents = corpus.get("documents")
    if not isinstance(documents, list) or not documents:
        errors.append("corpus.documents must be a non-empty array")
        documents = []
    corpus_by_id = {item.get("document_id"): item for item in documents if isinstance(item, dict) and item.get("document_id")}
    question_by_id: dict[str, dict[str, Any]] = {}
    label_by_id: dict[str, dict[str, Any]] = {}
    review_by_id: dict[str, dict[str, Any]] = {}
    duplicate_question_ids: list[str] = []
    duplicate_label_ids: list[str] = []
    duplicate_review_ids: list[str] = []
    normalized_questions: Counter[str] = Counter()
    for question in questions:
        case_id = question.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            errors.append("question missing case_id")
            continue
        if case_id in question_by_id:
            duplicate_question_ids.append(case_id)
        question_by_id[case_id] = question
        text = question.get("question")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{case_id}: question text is required")
        else:
            normalized_questions[normalized_question(text)] += 1
        if question.get("benchmark_id") != corpus.get("benchmark_id", "financial-rag-v1"):
            errors.append(f"{case_id}: benchmark_id mismatch")
        if question.get("answer_type") not in QUESTION_ANSWER_TYPES:
            errors.append(f"{case_id}: invalid answer_type")
        if question.get("difficulty") not in DIFFICULTIES:
            errors.append(f"{case_id}: invalid difficulty")
        scope = question.get("document_scope")
        if not isinstance(scope, list) or not scope or any(item not in corpus_by_id for item in scope):
            errors.append(f"{case_id}: document_scope must reference corpus documents")
        if not isinstance(question.get("category"), list) or not question["category"]:
            errors.append(f"{case_id}: category is required")
        for flag in ("answerable", "requires_calculation", "requires_multiple_sources"):
            if not isinstance(question.get(flag), bool):
                errors.append(f"{case_id}: {flag} must be boolean")
        if question.get("draft_status") not in {"generated", "edited", "rejected"}:
            errors.append(f"{case_id}: invalid draft_status")
        if question.get("authoring_method") not in {"human", "assisted", "human_or_assisted"}:
            errors.append(f"{case_id}: invalid authoring_method")
    duplicate_question_texts = [text for text, count in normalized_questions.items() if count > 1]
    warnings.extend(f"duplicate question text: {text}" for text in duplicate_question_texts)
    for label in labels:
        case_id = label.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            errors.append("label missing case_id")
            continue
        if case_id in label_by_id:
            duplicate_label_ids.append(case_id)
        label_by_id[case_id] = label
        question = question_by_id.get(case_id)
        if question is None:
            errors.append(f"label without question: {case_id}")
            continue
        if label.get("label_status") not in {"draft", "reviewed", "golden", "sealed"}:
            errors.append(f"{case_id}: invalid label_status")
        if label.get("review_status") not in {"unreviewed", "in_review", "reviewed"}:
            errors.append(f"{case_id}: invalid review_status")
        expected_no_answer = label.get("expected_no_answer")
        if not isinstance(expected_no_answer, bool):
            errors.append(f"{case_id}: expected_no_answer must be boolean")
            expected_no_answer = False
        sources = label.get("expected_sources")
        if not isinstance(sources, list):
            errors.append(f"{case_id}: expected_sources must be an array")
            sources = []
        if question.get("answerable") and not expected_no_answer and not sources:
            errors.append(f"{case_id}: answerable question requires expected_sources")
        if expected_no_answer and sources:
            errors.append(f"{case_id}: no-answer question must not have expected_sources")
        for source in sources:
            if not isinstance(source, dict):
                errors.append(f"{case_id}: source must be an object")
            else:
                errors.extend(f"{case_id}: {message}" for message in _source_errors(source, corpus_by_id=corpus_by_id))
        expected_answer = label.get("expected_answer")
        if not isinstance(expected_answer, dict):
            errors.append(f"{case_id}: expected_answer must be an object")
        elif not draft and question.get("answerable") and expected_answer.get("canonical_value") is None and not expected_answer.get("text"):
            errors.append(f"{case_id}: reviewed answerable label needs an answer value")
        calculation = label.get("calculation")
        if question.get("requires_calculation") and not isinstance(calculation, dict):
            errors.append(f"{case_id}: calculation is required")
        if isinstance(calculation, dict):
            if calculation.get("operation") not in CALCULATION_OPERATIONS:
                errors.append(f"{case_id}: invalid calculation operation")
            if not isinstance(calculation.get("formula"), str) or not calculation["formula"].strip():
                errors.append(f"{case_id}: calculation formula is required")
            if not draft and not isinstance(calculation.get("operands"), list):
                errors.append(f"{case_id}: calculation operands are required")
    for review in review_records:
        case_id = review.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            errors.append("review record missing case_id")
            continue
        if case_id in review_by_id:
            duplicate_review_ids.append(case_id)
        review_by_id[case_id] = review
        for field in ("question_reviewed", "answer_reviewed", "source_reviewed", "calculation_reviewed", "ready_for_golden"):
            if not isinstance(review.get(field), bool):
                errors.append(f"{case_id}: review field {field} must be boolean")
    question_ids = set(question_by_id)
    if question_ids != set(label_by_id):
        errors.append("questions and labels must have identical case IDs")
    if question_ids != set(review_by_id):
        errors.append("questions and review records must have identical case IDs")
    if duplicate_question_ids or duplicate_label_ids or duplicate_review_ids:
        errors.append("duplicate case IDs detected")
    return {
        "errors": errors,
        "warnings": warnings,
        "duplicate_question_ids": duplicate_question_ids,
        "duplicate_label_ids": duplicate_label_ids,
        "duplicate_review_ids": duplicate_review_ids,
        "duplicate_question_texts": duplicate_question_texts,
        "question_count": len(questions),
        "label_count": len(labels),
        "review_count": len(review_records),
        "schema_valid": not errors,
    }
