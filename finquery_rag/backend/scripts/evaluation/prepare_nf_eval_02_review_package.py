"""Prepare a read-only local review package from ingested benchmark files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.evaluation.audit_nf_eval_02_source_files import (
    SourceAuditError,
    collect_verified_source_files,
)
from scripts.evaluation.benchmark_foundation import load_json, load_jsonl
from scripts.evaluation.build_benchmark_source_review_candidates import build_candidates


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _make_worklist(
    *,
    questions_path: Path,
    labels_path: Path,
    review_path: Path,
) -> list[dict[str, Any]]:
    questions = {item["case_id"]: item for item in load_jsonl(questions_path)}
    labels = {item["case_id"]: item for item in load_jsonl(labels_path)}
    reviews = {item["case_id"]: item for item in load_jsonl(review_path)}
    rows: list[dict[str, Any]] = []
    for case_id in sorted(questions):
        question = questions[case_id]
        label = labels.get(case_id, {})
        review = reviews.get(case_id, {})
        answerable = not bool(label.get("expected_no_answer"))
        sources = label.get("expected_sources") or []
        rows.append(
            {
                "case_id": case_id,
                "company": question.get("company"),
                "question": question.get("question"),
                "answerable": answerable,
                "question_review_status": "pending",
                "answer_review_status": "pending" if answerable else "not_applicable",
                "source_review_status": "pending" if answerable else "not_applicable",
                "calculation_review_status": (
                    "pending"
                    if question.get("requires_calculation")
                    else "not_applicable"
                ),
                "negative_evidence_review_status": (
                    "not_applicable" if answerable else "pending"
                ),
                "expected_source_count": len(sources),
                "verified_source_count": sum(
                    int(bool(source.get("source_verified"))) for source in sources
                ),
                "legacy_review_status": review.get("review_status", "unreviewed"),
                "reviewer": None,
                "reviewed_at": None,
                "review_notes": None,
            }
        )
    return rows


def _set_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            os.chmod(path, 0o555)
        else:
            os.chmod(path, 0o444)
    os.chmod(root, 0o555)


def prepare_package(
    *,
    corpus_path: Path,
    runtime_manifest_path: Path,
    registry_path: Path,
    source_root: Path,
    questions_path: Path,
    labels_path: Path,
    review_path: Path,
    package_dir: Path,
    bm25_db_path: Path,
    chroma_path: Path,
    tenant_id: int,
    build_review_candidates: bool = True,
) -> dict[str, Any]:
    audit, verified_paths = collect_verified_source_files(
        corpus_path=corpus_path,
        runtime_manifest_path=runtime_manifest_path,
        registry_path=registry_path,
        source_root=source_root,
        tenant_id=tenant_id,
    )
    if not audit["acceptance"]["passed"]:
        raise SourceAuditError("source audit failed; review package was not created")
    package_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = package_dir / "pdfs"
    candidate_dir = package_dir / "source-review-candidates"
    pdf_dir.mkdir(exist_ok=True)
    candidate_dir.mkdir(exist_ok=True)
    corpus = load_json(corpus_path)
    for document in corpus["documents"]:
        source_path = verified_paths[str(document["document_id"])]
        destination = pdf_dir / str(document["filename"])
        if destination.exists():
            os.chmod(destination, 0o644)
        shutil.copy2(source_path, destination)
        os.chmod(destination, 0o444)

    candidate_report: dict[str, Any] = {
        "candidate_count": 0,
        "candidate_files": [],
        "skipped": True,
    }
    if build_review_candidates:
        candidate_report = build_candidates(
            corpus_path=corpus_path,
            questions_path=questions_path,
            labels_path=labels_path,
            out_dir=candidate_dir,
            bm25_db_path=bm25_db_path,
            chroma_path=chroma_path,
            tenant_id=tenant_id,
            top_k=20,
        )
    worklist = _make_worklist(
        questions_path=questions_path,
        labels_path=labels_path,
        review_path=review_path,
    )
    _write_jsonl(package_dir / "annotation-worklist.jsonl", worklist)
    instructions = f"""# Financial RAG v1 source review package

This package contains read-only copies of the exact eight PDF files used by
the production ingestion for tenant {tenant_id}. It is a review aid only. It
is not an ingestion payload and must not be uploaded again.

Use the PDF page number printed by the PDF viewer (1-based). For every
answerable case, verify the document, page, section, table title, row, column,
period, unit/scale, and the candidate identity shown in the candidate files.
For no-answer cases, perform a full-document search and record the searched
terms and sections in the annotation workflow.

The draft questions and labels remain unverified. Do not mark a source or
answer as verified solely because it appears in a retrieval candidate list.
Do not replace a missing file with another download. Report any hash or page
mismatch as an anomaly.

Candidate builder summary: {candidate_report.get("candidate_count", 0)} metadata-only
candidates across {candidate_report.get("question_count", 0)} questions.
"""
    (package_dir / "review-instructions.md").write_text(
        instructions,
        encoding="utf-8",
    )
    manifest = {
        "artifact_schema": "nf-eval-02/review-package/v1",
        "benchmark_id": corpus.get("benchmark_id", "financial-rag-v1"),
        "corpus_hash": corpus.get("corpus_hash"),
        "tenant_id": tenant_id,
        "document_count": len(corpus["documents"]),
        "question_count": len(worklist),
        "pdf_count": len(list(pdf_dir.glob("*.pdf"))),
        "documents": [
            {
                "company": document.get("company"),
                "document_id": document["document_id"],
                "filename": document["filename"],
                "relative_pdf": f"pdfs/{document['filename']}",
                "file_sha256": document["file_sha256"],
                "page_count": document["page_count"],
            }
            for document in corpus["documents"]
        ],
        "candidate_report": {
            key: value
            for key, value in candidate_report.items()
            if key != "no_candidate_cases"
        },
        "source_audit_passed": True,
        "source_files_are_exact_ingested_copies": True,
        "read_only": True,
        "not_for_ingestion": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(package_dir / "review-package-manifest.json", manifest)
    _set_read_only(package_dir)
    return {
        "package_dir": str(package_dir),
        "source_audit": audit["acceptance"],
        "candidate_report": candidate_report,
        "pdf_count": manifest["pdf_count"],
        "question_count": manifest["question_count"],
        "read_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/financial_rag_v1/corpus.json"),
    )
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        default=Path("runtime/benchmark/financial_rag_v1/corpus-manifest.json"),
    )
    parser.add_argument("--registry", type=Path, default=Path("document_registry.db"))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("runtime/benchmark/financial_rag_v1/pdfs"),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("benchmarks/financial_rag_v1/data/questions.draft.jsonl"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("benchmarks/financial_rag_v1/data/labels.draft.jsonl"),
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("benchmarks/financial_rag_v1/data/review-status.jsonl"),
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("runtime/benchmark/financial_rag_v1/review-package"),
    )
    parser.add_argument("--bm25-db", type=Path, default=Path("rag_bm25.db"))
    parser.add_argument("--chroma-path", type=Path, default=Path("chroma_db"))
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument("--skip-candidates", action="store_true")
    args = parser.parse_args()
    result = prepare_package(
        corpus_path=args.corpus,
        runtime_manifest_path=args.runtime_manifest,
        registry_path=args.registry,
        source_root=args.source_root,
        questions_path=args.questions,
        labels_path=args.labels,
        review_path=args.review,
        package_dir=args.package_dir,
        bm25_db_path=args.bm25_db,
        chroma_path=args.chroma_path,
        tenant_id=args.tenant_id,
        build_review_candidates=not args.skip_candidates,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
