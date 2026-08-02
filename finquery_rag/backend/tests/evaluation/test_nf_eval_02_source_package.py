from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.evaluation.audit_nf_eval_02_source_files import (
    SourceAuditError,
    collect_verified_source_files,
)
from scripts.evaluation.prepare_nf_eval_02_review_package import _make_worklist


def _write_registry(path: Path, *, filename: str, digest: str, status: str = "ready") -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE document_registry (document_id TEXT, tenant_id INTEGER, "
        "filename TEXT, file_hash TEXT, content_hash TEXT, chunk_count INTEGER, "
        "page_count INTEGER, version INTEGER, status TEXT, updated_at REAL)"
    )
    connection.execute(
        "INSERT INTO document_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("registry-1", 1, filename, digest, None, 1, 1, 1, status, 1.0),
    )
    connection.commit()
    connection.close()


def _write_manifests(root: Path, digest: str) -> tuple[Path, Path, Path]:
    source = root / "source.pdf"
    source.write_bytes(b"pdf")
    corpus = root / "corpus.json"
    runtime = root / "runtime.json"
    payload = {
        "benchmark_id": "financial-rag-v1",
        "documents": [
            {
                "document_id": "doc-1",
                "filename": "source.pdf",
                "file_sha256": digest,
                "page_count": 1,
                "chunk_count": 1,
            }
        ],
    }
    corpus.write_text(json.dumps(payload), encoding="utf-8")
    runtime.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "doc-1",
                        "local_filename": "source.pdf",
                        "sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return corpus, runtime, source


def test_source_audit_matches_exact_file_and_registry(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"pdf").hexdigest()
    corpus, runtime, source = _write_manifests(tmp_path, digest)
    registry = tmp_path / "registry.db"
    _write_registry(registry, filename=source.name, digest=digest)

    report, paths = collect_verified_source_files(
        corpus_path=corpus,
        runtime_manifest_path=runtime,
        registry_path=registry,
        source_root=tmp_path,
        tenant_id=1,
    )

    assert report["acceptance"]["passed"] is True
    assert paths["doc-1"] == source
    assert report["documents"][0]["source_path_field_available"] is False


def test_source_audit_fails_on_hash_mismatch(tmp_path: Path) -> None:
    wrong_digest = hashlib.sha256(b"not-the-file").hexdigest()
    corpus, runtime, source = _write_manifests(tmp_path, wrong_digest)
    registry = tmp_path / "registry.db"
    _write_registry(registry, filename=source.name, digest=wrong_digest)

    report, paths = collect_verified_source_files(
        corpus_path=corpus,
        runtime_manifest_path=runtime,
        registry_path=registry,
        source_root=tmp_path,
        tenant_id=1,
    )

    assert report["acceptance"]["passed"] is False
    assert paths == {}
    assert report["documents"][0]["file_hash_matches_manifest"] is False


def test_source_audit_missing_registry_stops(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"pdf").hexdigest()
    corpus, runtime, _ = _write_manifests(tmp_path, digest)

    with pytest.raises(SourceAuditError, match="document registry is missing"):
        collect_verified_source_files(
            corpus_path=corpus,
            runtime_manifest_path=runtime,
            registry_path=tmp_path / "missing.db",
            source_root=tmp_path,
            tenant_id=1,
        )


def test_worklist_counts_expected_sources_not_cases(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    labels = tmp_path / "labels.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    questions.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "company": "Example",
                "question": "Compare two values",
                "requires_calculation": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "expected_no_answer": False,
                "expected_sources": [{}, {}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reviews.write_text(
        json.dumps({"case_id": "case-1", "review_status": "unreviewed"})
        + "\n",
        encoding="utf-8",
    )

    rows = _make_worklist(
        questions_path=questions,
        labels_path=labels,
        review_path=reviews,
    )
    assert rows[0]["expected_source_count"] == 2
    assert rows[0]["calculation_review_status"] == "pending"
