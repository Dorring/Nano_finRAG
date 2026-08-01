from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluation.benchmark_scope import (
    benchmark_document_ids,
    filter_candidates,
    filter_stage_candidates,
    validate_scope_pipeline,
)


def _corpus() -> dict:
    return {
        "benchmark_id": "financial-rag-v1",
        "documents": [
            {"document_id": f"doc-{index}"}
            for index in range(8)
        ],
    }


def _candidate(document_id: str, evidence_id: str = "e1") -> dict:
    return {"document_id": document_id, "evidence_id": evidence_id}


def test_global_index_may_contain_legacy_documents() -> None:
    allowed = benchmark_document_ids(_corpus())
    candidates, rejected = filter_candidates(
        [_candidate("doc-0"), _candidate("legacy-wipo")],
        allowed,
    )
    assert [item["document_id"] for item in candidates] == ["doc-0"]
    assert rejected == 1


def test_benchmark_scope_contains_exactly_eight_documents() -> None:
    assert len(benchmark_document_ids(_corpus())) == 8


def test_dense_results_are_filtered_by_document_whitelist() -> None:
    allowed = benchmark_document_ids(_corpus())
    filtered, rejected = filter_candidates(
        [_candidate("doc-1"), _candidate("legacy-final")],
        allowed,
    )
    assert len(filtered) == 1
    assert rejected == 1


def test_bm25_results_are_filtered_by_document_whitelist() -> None:
    allowed = benchmark_document_ids(_corpus())
    filtered, rejected = filter_candidates(
        [_candidate("legacy-leac"), _candidate("doc-2")],
        allowed,
    )
    assert [item["document_id"] for item in filtered] == ["doc-2"]
    assert rejected == 1


def test_rrf_cannot_reintroduce_out_of_scope_documents() -> None:
    allowed = benchmark_document_ids(_corpus())
    stages, rejected = filter_stage_candidates(
        {
            "dense": [_candidate("doc-0")],
            "bm25": [_candidate("doc-0"), _candidate("legacy-wipo")],
            "rrf": [_candidate("doc-0"), _candidate("legacy-wipo")],
        },
        allowed,
    )
    assert [item["document_id"] for item in stages["rrf"]] == ["doc-0"]
    assert rejected["rrf"] == 1


def test_final_context_contains_only_benchmark_documents() -> None:
    report = validate_scope_pipeline(
        {"final": [_candidate("doc-3"), _candidate("legacy-final")]},
        benchmark_document_ids(_corpus()),
    )
    assert report["final_context_out_of_scope_candidates"] == 1
    assert report["scope_integrity_passed"] is False


def test_citations_contain_only_benchmark_documents() -> None:
    report = validate_scope_pipeline(
        {},
        benchmark_document_ids(_corpus()),
        citations=[_candidate("doc-4"), _candidate("legacy-wipo")],
    )
    assert report["citation_out_of_scope_count"] == 1
    assert report["scope_integrity_passed"] is False


def test_legacy_documents_remain_unchanged() -> None:
    root = Path(__file__).parents[2] / "benchmarks" / "financial_rag_legacy_v0"
    catalog = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
    names = {item["filename"] for item in catalog["documents"]}
    assert names == {
        "FINAL Annual Report.pdf",
        "leac203.pdf",
        "wipo_pub_rn2021_18e.pdf",
    }
    assert benchmark_document_ids(_corpus()).isdisjoint(names)


def test_object_candidates_can_be_filtered() -> None:
    allowed = benchmark_document_ids(_corpus())
    filtered, rejected = filter_candidates(
        [SimpleNamespace(document_id="doc-5"), SimpleNamespace(document_id="legacy")],
        allowed,
    )
    assert len(filtered) == 1
    assert rejected == 1

