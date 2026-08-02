"""Audit the exact benchmark PDF files used by the production ingestion.

This is an evaluation-only, read-only audit. The document registry currently
stores the filename and file hash but not a source path, so the runtime corpus
manifest is used only to resolve the already-ingested file. No replacement
download or ingestion fallback is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


class SourceAuditError(RuntimeError):
    """Raised when the benchmark source cannot be proven to be usable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SourceAuditError(f"{path} must contain a JSON object")
    return value


def _read_registry_rows(
    registry_path: Path,
    *,
    tenant_id: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    if not registry_path.is_file():
        raise SourceAuditError(f"document registry is missing: {registry_path.name}")
    resolved = registry_path.resolve()
    uri = f"file:{resolved.as_posix()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        try:
            columns = [
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(document_registry)"
                )
            ]
            if not columns:
                raise SourceAuditError("document_registry table is missing")
            wanted = {
                "document_id",
                "tenant_id",
                "filename",
                "file_hash",
                "content_hash",
                "chunk_count",
                "page_count",
                "status",
                "version",
                "created_at",
                "updated_at",
            }
            available = sorted(wanted.intersection(columns))
            rows = connection.execute(
                "SELECT "
                + ", ".join(available)
                + " FROM document_registry WHERE tenant_id = ? "
                "ORDER BY filename, version DESC, updated_at DESC",
                (tenant_id,),
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise SourceAuditError(
            f"cannot read document registry: {type(exc).__name__}"
        ) from exc
    return [dict(zip(available, row)) for row in rows], set(columns)


def collect_verified_source_files(
    *,
    corpus_path: Path,
    runtime_manifest_path: Path,
    registry_path: Path,
    source_root: Path,
    tenant_id: int,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Return a non-sensitive audit report and verified source paths.

    The returned path map is only for the local review-package builder and is
    never serialized into a committed artifact.
    """

    corpus = _load_json(corpus_path)
    runtime_manifest = _load_json(runtime_manifest_path)
    documents = corpus.get("documents")
    runtime_documents = runtime_manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise SourceAuditError("corpus.documents must be a non-empty list")
    if not isinstance(runtime_documents, list):
        raise SourceAuditError("runtime corpus manifest has no documents")
    runtime_by_id = {
        str(item.get("document_id")): item
        for item in runtime_documents
        if isinstance(item, dict) and item.get("document_id")
    }
    registry_rows, registry_columns = _read_registry_rows(
        registry_path,
        tenant_id=tenant_id,
    )
    by_filename: dict[str, list[dict[str, Any]]] = {}
    for row in registry_rows:
        by_filename.setdefault(str(row.get("filename") or ""), []).append(row)

    reports: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    for document in documents:
        if not isinstance(document, dict):
            raise SourceAuditError("corpus document entry is not an object")
        benchmark_id = str(document.get("document_id") or "")
        filename = str(document.get("filename") or "")
        expected_hash = str(document.get("file_sha256") or "")
        runtime = runtime_by_id.get(benchmark_id)
        runtime_matches = bool(
            runtime
            and runtime.get("local_filename") == filename
            and runtime.get("sha256") == expected_hash
        )
        source_path = source_root / filename
        present = source_path.is_file()
        actual_hash = sha256_file(source_path) if present else None
        rows = by_filename.get(filename, [])
        ready_rows = [row for row in rows if row.get("status") == "ready"]
        ready_registry_ids = sorted(
            str(row.get("document_id"))
            for row in ready_rows
            if row.get("document_id")
        )
        registry_hashes = sorted(
            {
                str(row.get("file_hash"))
                for row in ready_rows
                if row.get("file_hash")
            }
        )
        registry_hash_matches = bool(
            ready_rows
            and all(row.get("file_hash") == expected_hash for row in ready_rows)
        )
        duplicate_count = max(0, len(set(ready_registry_ids)) - 1)
        item = {
            "company": document.get("company"),
            "document_id": benchmark_id,
            "filename": filename,
            "tenant_id": tenant_id,
            "registry_document_ids": ready_registry_ids,
            "production_document_record": bool(ready_rows),
            "registry_ready_count": len(ready_rows),
            "registry_file_hashes": registry_hashes,
            "registry_file_hash_matches_manifest": registry_hash_matches,
            "source_path_field_available": bool(
                {"source_path", "file_path", "storage_path", "original_path"}
                & registry_columns
            ),
            "source_resolution": "runtime_manifest_pdf_directory",
            "runtime_manifest_matches_corpus": runtime_matches,
            "source_file_present": present,
            "file_sha256": actual_hash,
            "manifest_file_sha256": expected_hash,
            "file_hash_matches_manifest": bool(
                actual_hash and actual_hash == expected_hash
            ),
            "file_size_bytes": source_path.stat().st_size if present else None,
            "page_count": document.get("page_count"),
            "registry_page_count": (
                ready_rows[0].get("page_count") if ready_rows else None
            ),
            "chunk_count": document.get("chunk_count"),
            "registry_chunk_count": (
                ready_rows[0].get("chunk_count") if ready_rows else None
            ),
            "duplicate_document_count": duplicate_count,
        }
        item["passed"] = all(
            (
                item["production_document_record"],
                item["registry_ready_count"] == 1,
                item["tenant_id"] == tenant_id,
                item["runtime_manifest_matches_corpus"],
                item["source_file_present"],
                item["file_hash_matches_manifest"],
                item["registry_file_hash_matches_manifest"],
                item["duplicate_document_count"] == 0,
            )
        )
        reports.append(item)
        if item["passed"]:
            paths[benchmark_id] = source_path

    acceptance = {
        "document_count": len(documents),
        "production_document_record_count": sum(
            int(item["production_document_record"]) for item in reports
        ),
        "source_file_count": sum(int(item["source_file_present"]) for item in reports),
        "file_hash_match_count": sum(
            int(item["file_hash_matches_manifest"]) for item in reports
        ),
        "registry_hash_match_count": sum(
            int(item["registry_file_hash_matches_manifest"]) for item in reports
        ),
        "duplicate_document_count": sum(
            int(item["duplicate_document_count"]) for item in reports
        ),
        "passed": len(reports) == len(documents)
        and all(item["passed"] for item in reports),
    }
    report = {
        "artifact_schema": "nf-eval-02/source-file-audit/v1",
        "benchmark_id": corpus.get("benchmark_id", "financial-rag-v1"),
        "tenant_id": tenant_id,
        "corpus_hash": corpus.get("corpus_hash"),
        "runtime_manifest_path_type": "runtime_corpus_manifest",
        "registry_path_type": "local_document_registry",
        "registry_source_path_field_available": bool(
            {"source_path", "file_path", "storage_path", "original_path"}
            & registry_columns
        ),
        "documents": reports,
        "acceptance": acceptance,
    }
    return report, paths


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
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/evaluation/nf-eval-02/source-file-audit.json"),
    )
    args = parser.parse_args()
    try:
        report, _ = collect_verified_source_files(
            corpus_path=args.corpus,
            runtime_manifest_path=args.runtime_manifest,
            registry_path=args.registry,
            source_root=args.source_root,
            tenant_id=args.tenant_id,
        )
    except SourceAuditError as exc:
        report = {
            "artifact_schema": "nf-eval-02/source-file-audit/v1",
            "acceptance": {"passed": False, "error": str(exc)},
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("acceptance", {}).get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
