"""Build non-authoritative source candidates for human benchmark review.

The builder reads the existing BM25 and Chroma indexes. It only emits
metadata, ranks, and hashes; it never edits labels, indexes, or production
configuration. Candidate files are intended for the ignored runtime review
package, not for Golden promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from scripts.evaluation.benchmark_foundation import load_json, load_jsonl  # noqa: E402
from scripts.evaluation.benchmark_scope import benchmark_document_ids  # noqa: E402
from src.retrieval.candidate_fusion import rrf  # noqa: E402
from src.retrieval.candidate_identity import (  # noqa: E402
    candidate_key,
    identity_from_candidate,
)


def _hash_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "unknown"


def _metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    value = candidate.get("metadata")
    return value if isinstance(value, dict) else {}


def _normalize(
    candidate: dict[str, Any],
    *,
    case_id: str,
    tenant_id: int,
    document_id_by_filename: dict[str, str],
) -> dict[str, Any] | None:
    meta = _metadata(candidate)
    if meta.get("type") == "table_cell":
        return None
    raw_id = candidate.get("doc_id") or candidate.get("candidate_id")
    filename = meta.get("doc_name") or candidate.get("filename")
    document_id = document_id_by_filename.get(str(filename))
    if not isinstance(raw_id, str) or not raw_id.strip() or not document_id:
        return None
    item = dict(candidate)
    item["case_id"] = case_id
    item["tenant_id"] = tenant_id
    item["document_id"] = document_id
    item["evidence_id"] = raw_id
    item["candidate_id"] = raw_id
    item["block_type"] = meta.get("type", candidate.get("block_type", "text"))
    item["parent_id"] = meta.get("parent_id", candidate.get("parent_id"))
    item["parent_row_id"] = meta.get(
        "parent_row_id", candidate.get("parent_row_id")
    )
    item["metadata"] = meta
    try:
        identity = identity_from_candidate(item)
    except Exception:
        return None
    item["candidate_key"] = candidate_key(identity)
    item["benchmark_document_id"] = document_id
    item["filename"] = str(filename)
    item["page"] = meta.get("page", candidate.get("page"))
    item["content"] = (
        candidate.get("content") if isinstance(candidate.get("content"), str) else ""
    )
    return item


def _compact(item: dict[str, Any]) -> dict[str, Any]:
    meta = item["metadata"]
    section = meta.get("section_path") or meta.get("section_title")
    if isinstance(section, list):
        section = " > ".join(str(value) for value in section)
    content = item.get("content", "")
    return {
        "case_id": item["case_id"],
        "candidate_key": item["candidate_key"],
        "document_id": item["benchmark_document_id"],
        "filename": item["filename"],
        "pdf_page": item.get("page"),
        "printed_page": meta.get("printed_page"),
        "section": section,
        "table_title": meta.get("table_title"),
        "row_label": meta.get("row_label"),
        "column_header": meta.get("column_header"),
        "period": meta.get("period"),
        "unit": meta.get("unit"),
        "scale": meta.get("scale"),
        "block_type": item.get("block_type"),
        "parent_id": item.get("parent_id"),
        "content_hash": _hash_text(content) if content.strip() else None,
        "content_char_count": len(content),
        "retrieval_sources": item.get("retrieval_sources", []),
        "bm25_rank": item.get("bm25_rank"),
        "dense_rank": item.get("dense_rank"),
        "rrf_rank": item.get("rrf_rank"),
        "bm25_score": item.get("bm25_score"),
        "dense_score": item.get("dense_score"),
        "rrf_score": item.get("rrf_score"),
    }


def build_candidates(
    *,
    corpus_path: Path,
    questions_path: Path,
    labels_path: Path,
    out_dir: Path,
    bm25_db_path: Path,
    chroma_path: Path,
    tenant_id: int,
    top_k: int = 20,
) -> dict[str, Any]:
    corpus = load_json(corpus_path)
    questions = load_jsonl(questions_path)
    label_ids = {item["case_id"] for item in load_jsonl(labels_path)}
    allowed = benchmark_document_ids(corpus)
    documents = corpus["documents"]
    by_filename = {item["filename"]: item["document_id"] for item in documents}
    by_id = {item["document_id"]: item for item in documents}
    allowed_filenames = set(by_filename)
    if not allowed or any(item not in by_id for item in allowed):
        raise ValueError("benchmark corpus whitelist is invalid")
    if {item["case_id"] for item in questions} != label_ids:
        raise ValueError("questions and labels must have identical case IDs")

    os.environ["CHROMA_PATH"] = str(chroma_path)
    from src.services.retrieval import SqliteBM25Retriever
    from src.services import vector_store

    # Bypass the retriever constructor's schema migration hook. The review
    # builder must only read the already-built BM25 index.
    bm25 = object.__new__(SqliteBM25Retriever)
    bm25.db_path = str(bm25_db_path)
    chroma_client = vector_store.get_chroma_client()
    collection = chroma_client.get_collection(
        name=vector_store.GLOBAL_COLLECTION_NAME,
        embedding_function=vector_store.embed_fn,
    )

    def query_dense(query: str, filenames: list[str]) -> list[dict[str, Any]]:
        result = collection.query(
            query_texts=[query],
            n_results=max(top_k, top_k * 4),
            where={
                "$and": [
                    {"user_id": tenant_id},
                    {"doc_name": {"$in": filenames}},
                ]
            },
            include=["documents", "metadatas", "distances"],
        )
        rows: list[dict[str, Any]] = []
        for doc_id, content, metadata, distance in zip(
            result.get("ids", [[]])[0],
            result.get("documents", [[]])[0],
            result.get("metadatas", [[]])[0],
            result.get("distances", [[]])[0],
        ):
            metadata = metadata if isinstance(metadata, dict) else {}
            if metadata.get("type") == "table_cell":
                continue
            rows.append(
                {
                    "doc_id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "score": 1 - float(distance),
                }
            )
            if len(rows) >= top_k:
                break
        return rows
    output_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scope_rejected = 0
    candidate_count = 0
    no_candidate_cases: list[str] = []
    for question in questions:
        case_id = str(question["case_id"])
        scope_ids = [
            str(value)
            for value in question.get("document_scope", [])
            if str(value) in allowed
        ]
        scope_filenames = [by_id[document_id]["filename"] for document_id in scope_ids]
        if not scope_filenames:
            scope_filenames = [item["filename"] for item in documents]
        scope_filename_set = set(scope_filenames)
        raw_bm25 = bm25.search(
            str(question["question"]),
            k=min(100, max(top_k * 5, top_k)),
            user_id=tenant_id,
        )
        bm25_candidates: list[dict[str, Any]] = []
        for raw in raw_bm25:
            meta = _metadata(raw)
            filename = str(meta.get("doc_name") or "")
            if filename not in allowed_filenames:
                scope_rejected += 1
                continue
            if filename not in scope_filename_set:
                continue
            bm25_candidates.append(raw)
            if len(bm25_candidates) >= top_k:
                break
        raw_dense = query_dense(str(question["question"]), scope_filenames)
        dense_candidates = [
            item
            for item in raw_dense
            if str(_metadata(item).get("doc_name") or "") in allowed_filenames
        ][:top_k]
        for rank, item in enumerate(bm25_candidates, start=1):
            item["bm25_rank"] = rank
        for rank, item in enumerate(dense_candidates, start=1):
            item["dense_rank"] = rank

        fused = rrf(
            [
                [{"doc_id": item.get("doc_id"), **item} for item in bm25_candidates],
                [{"doc_id": item.get("doc_id"), **item} for item in dense_candidates],
            ]
        )[:top_k]
        merged: dict[str, dict[str, Any]] = {}
        for source_name, candidates in (
            ("bm25", bm25_candidates),
            ("dense", dense_candidates),
        ):
            for rank, item in enumerate(candidates, start=1):
                normalized = _normalize(
                    item,
                    case_id=case_id,
                    tenant_id=tenant_id,
                    document_id_by_filename=by_filename,
                )
                if normalized is None:
                    continue
                key = str(normalized["doc_id"])
                existing = merged.setdefault(key, normalized)
                existing.setdefault("retrieval_sources", [])
                if source_name not in existing["retrieval_sources"]:
                    existing["retrieval_sources"].append(source_name)
                existing[f"{source_name}_rank"] = rank
                existing[f"{source_name}_score"] = item.get("score")
        for rank, item in enumerate(fused, start=1):
            key = str(item.get("doc_id") or "")
            if key in merged:
                merged[key]["rrf_rank"] = rank
                merged[key]["rrf_score"] = item.get("fused_score")
                continue
            normalized = _normalize(
                item,
                case_id=case_id,
                tenant_id=tenant_id,
                document_id_by_filename=by_filename,
            )
            if normalized is not None:
                normalized["rrf_rank"] = rank
                normalized["rrf_score"] = item.get("fused_score")
                normalized["retrieval_sources"] = ["rrf"]
                merged[key] = normalized
        rows = [
            _compact(item)
            for item in sorted(
                merged.values(),
                key=lambda value: (
                    value.get("rrf_rank") is None,
                    value.get("rrf_rank") or 10**9,
                    value["candidate_key"],
                ),
            )
            if item.get("rrf_rank") is not None
        ]
        if not rows:
            no_candidate_cases.append(case_id)
        candidate_count += len(rows)
        company = str(question.get("company") or "unknown")
        output_records[_slug(company)].extend(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for company, rows in sorted(output_records.items()):
        destination = out_dir / f"{company}.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        files.append(destination.name)
    report = {
        "artifact_schema": "nf-eval-02/source-review-candidates/v1",
        "benchmark_id": corpus.get("benchmark_id", "financial-rag-v1"),
        "tenant_id": tenant_id,
        "allowed_document_count": len(allowed),
        "question_count": len(questions),
        "candidate_count": candidate_count,
        "candidate_top_k": top_k,
        "scope_rejected_count": scope_rejected,
        "no_candidate_case_count": len(no_candidate_cases),
        "no_candidate_cases": no_candidate_cases,
        "candidate_files": files,
        "labels_used_for_metadata_only": True,
        "labels_modified": False,
        "indexes_modified": False,
    }
    (out_dir / "candidate-builder-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/financial_rag_v1/corpus.json"),
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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bm25-db", type=Path, default=Path("rag_bm25.db"))
    parser.add_argument("--chroma-path", type=Path, default=Path("chroma_db"))
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    result = build_candidates(
        corpus_path=args.corpus,
        questions_path=args.questions,
        labels_path=args.labels,
        out_dir=args.out_dir,
        bm25_db_path=args.bm25_db,
        chroma_path=args.chroma_path,
        tenant_id=args.tenant_id,
        top_k=args.top_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
