"""Document-whitelist contracts for the financial benchmark.

This module is deliberately independent from the production RAG pipeline.  It
provides one small, reusable contract that evaluation adapters can apply after
each retrieval/ranking boundary without changing the global user index.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class BenchmarkScopeError(ValueError):
    """Raised when a benchmark scope or stage cannot be validated."""


def benchmark_document_ids(corpus: Mapping[str, Any]) -> frozenset[str]:
    """Return the exact document whitelist declared by ``corpus``."""

    documents = corpus.get("documents")
    if not isinstance(documents, list) or not documents:
        raise BenchmarkScopeError("corpus.documents must be a non-empty list")
    ids = {
        str(document.get("document_id"))
        for document in documents
        if isinstance(document, Mapping) and document.get("document_id")
    }
    if len(ids) != len(documents):
        raise BenchmarkScopeError("corpus document_id values must be unique and non-empty")
    return frozenset(ids)


def _document_id(candidate: Any) -> str | None:
    if isinstance(candidate, Mapping):
        value = candidate.get("document_id")
        if value is None:
            identity = candidate.get("identity")
            if isinstance(identity, Mapping):
                value = identity.get("document_id")
    else:
        value = getattr(candidate, "document_id", None)
        if value is None:
            identity = getattr(candidate, "identity", None)
            value = getattr(identity, "document_id", None)
    return str(value) if value else None


def filter_candidates(
    candidates: Iterable[Any],
    allowed_document_ids: frozenset[str],
) -> tuple[list[Any], int]:
    """Keep only candidates belonging to the benchmark whitelist.

    The returned integer counts rejected candidates.  Missing identities are
    rejected as out of scope rather than silently passing through.
    """

    accepted: list[Any] = []
    rejected = 0
    for candidate in candidates:
        if _document_id(candidate) in allowed_document_ids:
            accepted.append(candidate)
        else:
            rejected += 1
    return accepted, rejected


def filter_stage_candidates(
    stages: Mapping[str, Iterable[Any]],
    allowed_document_ids: frozenset[str],
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    """Apply the same whitelist independently to every pipeline stage."""

    filtered: dict[str, list[Any]] = {}
    rejected: dict[str, int] = {}
    for stage, candidates in stages.items():
        filtered[stage], rejected[stage] = filter_candidates(
            candidates,
            allowed_document_ids,
        )
    return filtered, rejected


def validate_scope_pipeline(
    stages: Mapping[str, Iterable[Any]],
    allowed_document_ids: frozenset[str],
    *,
    citations: Iterable[Any] = (),
) -> dict[str, Any]:
    """Return a non-sensitive scope report for retrieval and answer stages."""

    stage_rejected: dict[str, int] = {}
    for stage, candidates in stages.items():
        _, stage_rejected[stage] = filter_candidates(
            candidates,
            allowed_document_ids,
        )
    _, citation_rejected = filter_candidates(citations, allowed_document_ids)
    report = {
        "allowed_document_count": len(allowed_document_ids),
        "retrieved_out_of_scope_candidates": sum(
            stage_rejected.get(stage, 0)
            for stage in ("dense", "chroma", "bm25", "retrieved")
        ),
        "reranked_out_of_scope_candidates": stage_rejected.get("reranker", 0),
        "final_context_out_of_scope_candidates": stage_rejected.get("final", 0),
        "citation_out_of_scope_count": citation_rejected,
        "stage_rejected_counts": stage_rejected,
    }
    report["scope_integrity_passed"] = not any(
        report[key]
        for key in (
            "retrieved_out_of_scope_candidates",
            "reranked_out_of_scope_candidates",
            "final_context_out_of_scope_candidates",
            "citation_out_of_scope_count",
        )
    )
    return report

