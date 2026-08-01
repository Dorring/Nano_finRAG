"""Validation helpers for the Financial RAG Benchmark v1 annotation set.

The benchmark intentionally keeps human-authored labels separate from the
blind question file used by the evaluation runner.  This module is evaluation
infrastructure only; it has no production RAG dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


CASE_TYPES = frozenset(
    {
        "fact",
        "table_fact",
        "calculation",
        "multi_source",
        "no_answer",
        "unit_period_trap",
    }
)
DIFFICULTIES = frozenset({"easy", "medium", "hard"})


@dataclass(frozen=True)
class BenchmarkValidationIssue:
    record_id: str
    message: str


def validate_document_catalog(documents: Iterable[dict[str, Any]]) -> list[BenchmarkValidationIssue]:
    issues: list[BenchmarkValidationIssue] = []
    seen: set[str] = set()
    for document in documents:
        document_id = str(document.get("document_id") or "")
        if not document_id:
            issues.append(BenchmarkValidationIssue("<unknown>", "missing document_id"))
            continue
        if document_id in seen:
            issues.append(BenchmarkValidationIssue(document_id, "duplicate document_id"))
        seen.add(document_id)
        for field in ("company", "fiscal_year", "official_landing_url", "local_filename"):
            if not document.get(field):
                issues.append(BenchmarkValidationIssue(document_id, f"missing {field}"))
        if document.get("source_kind") != "issuer_investor_relations":
            issues.append(BenchmarkValidationIssue(document_id, "source_kind must be issuer_investor_relations"))
    return issues


def validate_annotation_cases(
    cases: Iterable[dict[str, Any]],
    *,
    allowed_document_ids: set[str],
) -> list[BenchmarkValidationIssue]:
    issues: list[BenchmarkValidationIssue] = []
    seen: set[str] = set()
    for case in cases:
        case_id = str(case.get("id") or "")
        if not case_id:
            issues.append(BenchmarkValidationIssue("<unknown>", "missing id"))
            continue
        if case_id in seen:
            issues.append(BenchmarkValidationIssue(case_id, "duplicate case id"))
        seen.add(case_id)
        if case.get("type") not in CASE_TYPES:
            issues.append(BenchmarkValidationIssue(case_id, "unsupported type"))
        if case.get("difficulty") not in DIFFICULTIES:
            issues.append(BenchmarkValidationIssue(case_id, "unsupported difficulty"))
        if not str(case.get("question") or "").strip():
            issues.append(BenchmarkValidationIssue(case_id, "missing question"))
        answerable = case.get("answerable")
        if not isinstance(answerable, bool):
            issues.append(BenchmarkValidationIssue(case_id, "answerable must be boolean"))
            continue
        sources = case.get("expected_sources") or []
        if answerable and not sources:
            issues.append(BenchmarkValidationIssue(case_id, "answerable case requires expected_sources"))
        if not answerable and case.get("type") != "no_answer":
            issues.append(BenchmarkValidationIssue(case_id, "unanswerable case must use no_answer type"))
        if not answerable and sources:
            issues.append(BenchmarkValidationIssue(case_id, "no_answer case must not declare expected_sources"))
        for source in sources:
            document_id = source.get("document_id")
            if document_id not in allowed_document_ids:
                issues.append(BenchmarkValidationIssue(case_id, "source references unknown document"))
            page = source.get("page")
            if not isinstance(page, int) or page < 1:
                issues.append(BenchmarkValidationIssue(case_id, "source page must be a positive integer"))
            if not str(source.get("section") or "").strip():
                issues.append(BenchmarkValidationIssue(case_id, "source section is required"))
        review = case.get("review") or {}
        if review.get("status") not in {"draft", "reviewed", "accepted"}:
            issues.append(BenchmarkValidationIssue(case_id, "review status is required"))
        if review.get("status") == "accepted" and not review.get("second_reviewer"):
            issues.append(BenchmarkValidationIssue(case_id, "accepted case requires second_reviewer"))
    return issues


def taxonomy_counts(cases: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {item: 0 for item in sorted(CASE_TYPES)}
    for case in cases:
        case_type = case.get("type")
        if case_type in counts:
            counts[case_type] += 1
    return counts
