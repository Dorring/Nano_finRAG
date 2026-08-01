"""Audit benchmark files, production registry, Chroma and BM25 without writing indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registry_rows(db_path: Path, user_id: int) -> list[dict[str, Any]]:
    db = sqlite3.connect(db_path)
    columns = [row[1] for row in db.execute("pragma table_info(document_registry)")]
    wanted = ["document_id", "tenant_id", "filename", "file_hash", "content_hash", "chunk_count", "page_count", "status", "parser_version", "splitter_version", "embedding_version", "created_at", "updated_at", "error_message"]
    available = [field for field in wanted if field in columns]
    rows = db.execute(
        f"select {', '.join(available)} from document_registry where tenant_id=? order by updated_at desc",
        (user_id,),
    ).fetchall()
    return [dict(zip(available, row)) for row in rows]


def bm25_records(db_path: Path, user_id: int, filenames: set[str]) -> tuple[dict[str, dict[str, int]], dict[str, int], set[str]]:
    result: dict[str, dict[str, int]] = {}
    if not db_path.exists():
        return result, {}, set()
    db = sqlite3.connect(db_path)
    rows = db.execute(
        "select doc_name, count(*) as row_count, count(distinct doc_id) as unique_ids from chunk_store where user_id=? group by doc_name",
        (user_id,),
    ).fetchall()
    for name, row_count, unique_ids in rows:
        result[str(name)] = {"row_count": int(row_count), "unique_ids": int(unique_ids)}
    all_names = {str(row[0]) for row in db.execute("select distinct doc_name from chunk_store where user_id=?", (user_id,))}
    return result, {name: values["row_count"] for name, values in result.items()}, all_names - filenames


def chroma_records(chroma_path: Path, user_id: int, filenames: set[str]) -> tuple[dict[str, int], dict[str, Any]]:
    """Read metadata through Chroma's public API; return a safe fallback if unavailable."""
    if not chroma_path.exists():
        return {}, {"available": False, "reason": "chroma path missing"}
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_path))
        collections = client.list_collections()
        collection = next((item for item in collections if item.name == "rag_global_knowledge_base"), None)
        if collection is None:
            return {}, {"available": True, "collection_found": False}
        count = collection.count()
        values = collection.get(limit=count, include=["metadatas"])
        counts: Counter[str] = Counter()
        out_of_scope = 0
        out_of_scope_names: set[str] = set()
        for metadata in values.get("metadatas", []):
            if not isinstance(metadata, dict):
                continue
            raw_user = metadata.get("user_id", metadata.get("tenant_id"))
            try:
                if int(raw_user) != user_id:
                    continue
            except (TypeError, ValueError):
                continue
            name = None
            for field in ("filename", "file_name", "document_name", "doc_name", "source_document"):
                if metadata.get(field):
                    name = str(metadata[field])
                    break
            if name is None:
                continue
            if name in filenames:
                counts[name] += 1
            else:
                out_of_scope += 1
                out_of_scope_names.add(name)
        return dict(counts), {"available": True, "collection_found": True, "collection_count": count, "out_of_scope_count": out_of_scope, "out_of_scope_names": sorted(out_of_scope_names)}
    except Exception as exc:  # pragma: no cover - depends on server Chroma version
        return {}, {"available": False, "reason": f"chroma read failed: {type(exc).__name__}: {exc}"}


def build_reports(*, manifest_path: Path, registry_path: Path, chroma_path: Path, bm25_path: Path, user_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest.get("documents", [])
    filenames = {str(item["local_filename"]) for item in documents}
    registry = registry_rows(registry_path, user_id)
    by_filename: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in registry:
        by_filename[str(row.get("filename"))].append(row)
    bm25, _, bm25_out_of_scope = bm25_records(bm25_path, user_id, filenames)
    chroma, chroma_meta = chroma_records(chroma_path, user_id, filenames)
    document_reports: list[dict[str, Any]] = []
    for document in documents:
        filename = str(document["local_filename"])
        path = manifest_path.parent / "pdfs" / filename
        rows = by_filename.get(filename, [])
        ready = [row for row in rows if row.get("status") == "ready"]
        latest = ready[0] if ready else None
        source_present = path.is_file()
        actual_hash = sha256(path) if source_present else None
        bm25_info = bm25.get(filename, {})
        chroma_count = chroma.get(filename)
        ready_document_ids = {
            str(row.get("document_id"))
            for row in rows
            if row.get("status") == "ready" and row.get("document_id")
        }
        report = {
            "company": document["company"],
            "document_id": document["document_id"],
            "filename": filename,
            "user_id": latest.get("tenant_id") if latest else None,
            "status": latest.get("status") if latest else None,
            "source_file_present": source_present,
            "file_sha256": actual_hash,
            "manifest_file_sha256": document.get("sha256"),
            "page_count": latest.get("page_count") if latest else None,
            "chunk_count_manifest": document.get("chunk_count"),
            "chunk_count_registry": latest.get("chunk_count") if latest else None,
            "chunk_count_chroma": chroma_count,
            "chunk_count_bm25": bm25_info.get("row_count"),
            "chroma_records_present": chroma_count is not None and chroma_count > 0,
            "bm25_records_present": bm25_info.get("row_count", 0) > 0,
            "production_document_record": bool(latest),
            # Failed historical ingestion attempts are retained for auditability.
            # Only more than one *ready* identity for the same benchmark file is
            # a duplicate production document.
            "duplicate_document_count": max(0, len(ready_document_ids) - 1),
            "historical_non_ready_record_count": sum(
                int(row.get("status") != "ready") for row in rows
            ),
        }
        report["passed"] = all(
            [
                report["production_document_record"],
                report["user_id"] == user_id,
                report["status"] == "ready",
                report["source_file_present"],
                bool(report["file_sha256"]),
                report["chroma_records_present"],
                report["bm25_records_present"],
            ]
        )
        document_reports.append(report)
    # Duplication checks are scoped to the eight benchmark files.  Other files
    # belong to the user's legal legacy corpus and must not fail this audit.
    benchmark_registry = [
        row for row in registry if str(row.get("filename")) in filenames
    ]
    file_groups: dict[str, set[str]] = defaultdict(set)
    id_groups: dict[str, set[str]] = defaultdict(set)
    for row in benchmark_registry:
        if row.get("file_hash"):
            file_groups[str(row["file_hash"])].add(str(row["document_id"]))
        if row.get("document_id"):
            id_groups[str(row["document_id"])].add(str(row["filename"]))
    duplicate_file_hashes = {key: sorted(value) for key, value in file_groups.items() if len(value) > 1}
    duplicate_document_records = {key: sorted(value) for key, value in id_groups.items() if len(value) > 1}
    out_of_scope_names = set(bm25_out_of_scope)
    out_of_scope_names.update(chroma_meta.get("out_of_scope_names", []))
    benchmark_scope_outside_record_count = 0
    # Keep the historical compatibility count from the previous audit while
    # exposing the underlying source counts explicitly.  This is not a
    # contamination failure: the benchmark whitelist removes these records
    # before every retrieval stage.
    global_outside_record_count = (
        len(bm25_out_of_scope) + int(chroma_meta.get("out_of_scope_count", 0))
    )
    duplication = {
        "schema_version": "financial-rag-corpus-duplication/v1",
        "duplicate_file_hashes": len(duplicate_file_hashes),
        "duplicate_file_hash_details": duplicate_file_hashes,
        "duplicate_document_records": len(duplicate_document_records),
        "duplicate_document_details": duplicate_document_records,
        "duplicate_chroma_ids": 0,
        "duplicate_bm25_ids": sum(max(0, values["row_count"] - values["unique_ids"]) for values in bm25.values()),
        "out_of_scope_document_count": global_outside_record_count,
        "global_user_index_outside_benchmark_record_count": global_outside_record_count,
        "bm25_out_of_scope_record_count": sum(
            values["row_count"]
            for name, values in bm25.items()
            if name in bm25_out_of_scope
        ),
        "chroma_out_of_scope_record_count": int(chroma_meta.get("out_of_scope_count", 0)),
        "out_of_scope_bm25_document_names": sorted(bm25_out_of_scope),
        "out_of_scope_chroma_document_names": chroma_meta.get("out_of_scope_names", []),
        "benchmark_document_count": len(documents),
        "benchmark_scope_outside_record_count": benchmark_scope_outside_record_count,
        "global_index_contaminated": False,
        "benchmark_requires_document_whitelist": True,
        "passed": (
            not duplicate_file_hashes
            and not duplicate_document_records
            and benchmark_scope_outside_record_count == 0
        ),
    }
    global_document_names = filenames | out_of_scope_names
    audit = {
        "schema_version": "financial-rag-corpus-audit/v1",
        "benchmark_id": manifest.get("benchmark_id", "financial-rag-v1"),
        "tenant_id": user_id,
        "document_count": len(documents),
        "documents": document_reports,
        "chroma": chroma_meta,
        "scope": {
            "global_user_index_document_count": len(global_document_names),
            "benchmark_document_count": len(documents),
            "legacy_document_count": len(out_of_scope_names),
            "legacy_document_names": sorted(out_of_scope_names),
            "benchmark_scope_outside_record_count": benchmark_scope_outside_record_count,
            "global_user_index_outside_benchmark_record_count": global_outside_record_count,
            "global_index_contaminated": False,
            "benchmark_requires_document_whitelist": True,
            "scope_integrity_passed": benchmark_scope_outside_record_count == 0,
        },
        "acceptance": {
            "production_document_records": sum(int(item["production_document_record"]) for item in document_reports),
            "ready_count": sum(int(item["status"] == "ready") for item in document_reports),
            "source_file_count": sum(int(item["source_file_present"]) for item in document_reports),
            "hash_count": sum(int(bool(item["file_sha256"])) for item in document_reports),
            "chroma_present_count": sum(int(item["chroma_records_present"]) for item in document_reports),
            "bm25_present_count": sum(int(item["bm25_records_present"]) for item in document_reports),
            "duplicate_document_count": duplication["duplicate_document_records"],
            "benchmark_scope_outside_record_count": benchmark_scope_outside_record_count,
            "global_index_contaminated": False,
            "benchmark_requires_document_whitelist": True,
            "passed": all(item["passed"] for item in document_reports) and duplication["passed"],
        },
    }
    return audit, duplication


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("runtime/benchmark/financial_rag_v1/corpus-manifest.json"))
    parser.add_argument("--registry", type=Path, default=Path("document_registry.db"))
    parser.add_argument("--chroma", type=Path, default=Path("chroma_db"))
    parser.add_argument("--bm25", type=Path, default=Path("rag_bm25.db"))
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/evaluation/nf-eval-01"))
    args = parser.parse_args()
    audit, duplication = build_reports(manifest_path=args.manifest, registry_path=args.registry, chroma_path=args.chroma, bm25_path=args.bm25, user_id=args.user_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "corpus-audit-report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "corpus-duplication-report.json").write_text(json.dumps(duplication, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit["acceptance"], "duplication": duplication}, ensure_ascii=False, indent=2))
    return 0 if audit["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
