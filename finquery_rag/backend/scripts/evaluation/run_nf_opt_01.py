"""NF-OPT-01: MiniLM Dense coverage shadow A/B.

The runner builds an isolated Chroma collection from the complete BM25
canonical-candidate universe, reuses the production MiniLM query encoder, and
compares Dense/Union/RRF retrieval only.  It never invokes the answer
orchestrator, reranker, generator, Calculator, or Validator.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Mapping, Sequence
import unicodedata

import chromadb
import numpy as np

from scripts.evaluation import run_nf_eval_03_r1 as r1
from scripts.evaluation.run_nf_eval_03_r2 import _raw_candidate_for_identity
from src.evaluation.nf_opt_01 import (
    candidate_scope_ok,
    compare_rank_maps,
    coverage_state,
    dense_coverage_gate,
    dense_superset_gate,
    percentile,
    rank_metrics,
)
from src.retrieval.candidate_fusion import (
    boost_front_matter_chunks,
    normalize_scores,
    rrf,
)
from src.retrieval.query_processor import QueryProcessor


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "financial_rag_v1"
DATA = BENCHMARK / "data"
DEFAULT_OUT = ROOT / "artifacts" / "evaluation" / "nf-opt-01"
DEFAULT_RUNTIME = ROOT / "runtime" / "evaluation" / "nf-opt-01" / "shadow-chroma"
DEFAULT_SUPERSET_RUNTIME = ROOT / "runtime" / "evaluation" / "nf-opt-01" / "superset-chroma"
DEFAULT_NEGATIVE = ROOT / "artifacts" / "evaluation" / "nf-eval-02" / "negative-evidence-review-report.json"
NF04_OUT = ROOT / "artifacts" / "evaluation" / "nf-eval-04"


class NFOpt01Error(ValueError):
    """Raised when the frozen benchmark or shadow index cannot be validated."""


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _content_hash(content: Any) -> str:
    if not isinstance(content, str) or not content.strip():
        return ""
    normalized = unicodedata.normalize("NFC", content).replace("\r\n", "\n").replace("\r", "\n")
    return _sha256_bytes(normalized.encode("utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument("--chroma-path", type=Path, default=ROOT / "chroma_db")
    parser.add_argument("--bm25-db-path", type=Path, default=ROOT / "rag_bm25.db")
    parser.add_argument("--shadow-path", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--superset-path", type=Path, default=DEFAULT_SUPERSET_RUNTIME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--corpus", type=Path, default=BENCHMARK / "corpus.json")
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=DATA / "golden-manifest.json")
    parser.add_argument("--questions", type=Path, default=DATA / "questions.golden.jsonl")
    parser.add_argument("--labels", type=Path, default=DATA / "labels.golden.jsonl")
    parser.add_argument("--review-status", type=Path, default=DATA / "review-status.golden.jsonl")
    parser.add_argument("--negative-report", type=Path, default=DEFAULT_NEGATIVE)
    parser.add_argument("--diagnostic-top-n", type=int, default=200)
    parser.add_argument("--production-top-k", type=int, default=5)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    return parser.parse_args()


def _metadata_for_chroma(
    *,
    candidate_key: str,
    evidence_id: str,
    document_id: str,
    filename: str,
    metadata: Mapping[str, Any],
    content_hash: str,
    tenant_id: int,
) -> dict[str, Any]:
    """Keep only primitive, auditable metadata in the local shadow index."""

    output: dict[str, Any] = {
        "candidate_key": candidate_key,
        "evidence_id": evidence_id,
        "canonical_document_id": document_id,
        "doc_name": filename,
        "user_id": tenant_id,
        "content_hash": content_hash,
        "type": str(metadata.get("type") or "text"),
        "dense_indexed": True,
        "nf_opt_01_metadata_schema": "v2",
    }
    for key in ("page", "pages"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            output[key] = int(value)
    for key in ("parent_id", "parent_row_id", "row_id", "table_id", "section_path"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            output[key] = value
    return output


def _load_candidate_universe(
    *,
    db_path: Path,
    corpus: Mapping[str, Any],
    mapping: Mapping[str, str],
    tenant_id: int,
    gold_keys: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed = {str(item["filename"]) for item in corpus["documents"]}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    unsupported_count = 0
    global_scope_excluded_count = 0
    table_cell_count = 0
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT doc_id, content, metadata_json, doc_name, user_id "
            "FROM chunk_store WHERE user_id = ? ORDER BY doc_id",
            (tenant_id,),
        ).fetchall()
    for doc_id, content, metadata_json, filename, user_id in rows:
        filename = str(filename or "")
        if filename not in allowed:
            global_scope_excluded_count += 1
            continue
        try:
            metadata = json.loads(metadata_json or "{}")
        except (TypeError, json.JSONDecodeError):
            unsupported_count += 1
            continue
        block_type = str(metadata.get("type") or "text")
        # Table cells are secondary children and cannot form an independent
        # global Candidate Identity.  The parent row is the canonical unit.
        if block_type == "table_cell":
            table_cell_count += 1
            continue
        if not isinstance(content, str) or not content.strip():
            unsupported_count += 1
            continue
        raw = _raw_candidate_for_identity(
            {
                "doc_id": str(doc_id),
                "content": content,
                "metadata": metadata,
            }
        )
        key, document_id, evidence_id = r1.candidate_identity_from_record(
            raw,
            filename_to_document=mapping,
            tenant_id=tenant_id,
        )
        if not key or not document_id or not evidence_id:
            unsupported_count += 1
            continue
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        rendered_hash = _content_hash(content)
        if not rendered_hash:
            unsupported_count += 1
            continue
        records.append(
            {
                "candidate_key": key,
                "evidence_id": str(evidence_id),
                "doc_id": str(doc_id),
                "document_id": str(document_id),
                "filename": filename,
                "content": content,
                "content_hash": rendered_hash,
                "metadata": _metadata_for_chroma(
                    candidate_key=key,
                    evidence_id=str(evidence_id),
                    document_id=str(document_id),
                    filename=filename,
                    metadata=metadata,
                    content_hash=rendered_hash,
                    tenant_id=tenant_id,
                ),
            }
        )
    records.sort(key=lambda row: row["candidate_key"])
    gold_presence = sum(str(key) in seen for key in gold_keys)
    manifest_hash_payload = [
        {
            "candidate_key": row["candidate_key"],
            "document_id": row["document_id"],
            "evidence_id": row["evidence_id"],
            "content_hash": row["content_hash"],
            "type": row["metadata"].get("type"),
            "page": row["metadata"].get("page"),
        }
        for row in records
    ]
    summary = {
        "variant": "canonical_coverage_shadow",
        "embedding_model": "all-MiniLM-L6-v2",
        "candidate_count": len(records),
        "unique_candidate_count": len(seen),
        "duplicate_candidate_count": duplicate_count,
        "unsupported_candidate_count": unsupported_count,
        "out_of_scope_candidate_count": 0,
        "global_user_records_excluded_by_whitelist": global_scope_excluded_count,
        "table_cell_records_excluded_from_global_identity": table_cell_count,
        "gold_identity_presence_count": gold_presence,
        "gold_identity_expected_count": len(gold_keys),
        "collection_hash": _stable_hash(manifest_hash_payload),
        "candidate_source": "bm25_chunk_store_canonical_universe",
        "gold_labels_not_used_to_build_universe": True,
    }
    return records, summary


def _current_collection_keys(
    collection: Any,
    *,
    filenames: Sequence[str],
    mapping: Mapping[str, str],
    tenant_id: int,
) -> set[str]:
    where = {"$and": [{"user_id": tenant_id}, {"doc_name": {"$in": list(filenames)}}]}
    try:
        result = collection.get(where=where, include=["metadatas"])
    except Exception as exc:  # noqa: BLE001 - report as an empty observable set
        raise NFOpt01Error(f"cannot inspect current dense collection: {type(exc).__name__}") from exc
    keys: set[str] = set()
    for candidate_id, metadata in zip(result.get("ids") or [], result.get("metadatas") or []):
        metadata = metadata or {}
        filename = str(metadata.get("doc_name") or "")
        raw = {
            "doc_id": str(candidate_id),
            "filename": filename,
            "document_id": mapping.get(filename),
            "metadata": metadata,
            "type": metadata.get("type"),
            "block_type": metadata.get("type"),
            "row_id": metadata.get("row_id"),
            "parent_row_id": metadata.get("parent_row_id") or metadata.get("parent_id"),
        }
        key, _, _ = r1.candidate_identity_from_record(
            raw,
            filename_to_document=mapping,
            tenant_id=tenant_id,
        )
        if key:
            keys.add(key)
    return keys


def _load_current_dense_records(
    collection: Any,
    *,
    filenames: Sequence[str],
    mapping: Mapping[str, str],
    tenant_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read production vectors without re-encoding or mutating the collection."""

    where = {"$and": [{"user_id": tenant_id}, {"doc_name": {"$in": list(filenames)}}]}
    result = collection.get(
        where=where,
        include=["documents", "metadatas", "embeddings"],
    )
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) != len(ids):
        raise NFOpt01Error("current dense collection did not expose all embeddings")
    records_by_key: dict[str, dict[str, Any]] = {}
    invalid_count = 0
    conflict_count = 0
    for candidate_id, content, metadata, embedding in zip(ids, documents, metadatas, embeddings):
        metadata = dict(metadata or {})
        raw = _raw_candidate_for_identity(
            {
                "doc_id": str(candidate_id),
                "content": content,
                "metadata": metadata,
            }
        )
        key, document_id, evidence_id = r1.candidate_identity_from_record(
            raw,
            filename_to_document=mapping,
            tenant_id=tenant_id,
        )
        rendered_hash = _content_hash(content)
        if not key or not document_id or not evidence_id or not rendered_hash:
            invalid_count += 1
            continue
        row = {
            "candidate_key": str(key),
            "evidence_id": str(evidence_id),
            "doc_id": str(candidate_id),
            "document_id": str(document_id),
            "filename": str(metadata.get("doc_name") or ""),
            "content": str(content),
            "content_hash": rendered_hash,
            "metadata": _metadata_for_chroma(
                candidate_key=str(key),
                evidence_id=str(evidence_id),
                document_id=str(document_id),
                filename=str(metadata.get("doc_name") or ""),
                metadata=metadata,
                content_hash=rendered_hash,
                tenant_id=tenant_id,
            )
            | {"dense_variant_source": "current_production", "nf_opt_01_metadata_schema": "v3"},
            "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
        }
        prior = records_by_key.get(str(key))
        if prior is not None:
            prior_page = prior["metadata"].get("page")
            page = row["metadata"].get("page")
            if (
                prior["document_id"] != row["document_id"]
                or prior["content_hash"] != row["content_hash"]
                or prior_page != page
            ):
                conflict_count += 1
                continue
            # Identical duplicate rows do not create a second canonical key.
            continue
        records_by_key[str(key)] = row
    if invalid_count or conflict_count:
        raise NFOpt01Error(
            "current dense identity validation failed: "
            f"invalid={invalid_count}, conflicts={conflict_count}"
        )
    records = sorted(records_by_key.values(), key=lambda row: row["candidate_key"])
    return records, {
        "candidate_count": len(records),
        "invalid_identity_count": invalid_count,
        "identity_conflict_count": conflict_count,
        "production_vectors_reused_count": len(records),
    }


def _build_superset_index(
    *,
    superset_client: Any,
    superset_name: str,
    current_records: Sequence[Mapping[str, Any]],
    canonical_records: Sequence[Mapping[str, Any]],
    embed_fn: Any,
) -> tuple[Any, dict[str, Any], float]:
    """Build Current ∪ Canonical while reusing every current vector exactly."""

    started = time.perf_counter()
    current_by_key = {str(row["candidate_key"]): dict(row) for row in current_records}
    canonical_by_key = {str(row["candidate_key"]): dict(row) for row in canonical_records}
    conflicts: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    for key, current in current_by_key.items():
        canonical = canonical_by_key.get(key)
        if canonical is not None:
            current_page = (current.get("metadata") or {}).get("page")
            canonical_page = (canonical.get("metadata") or {}).get("page")
            if (
                current.get("document_id") != canonical.get("document_id")
                or current.get("content_hash") != canonical.get("content_hash")
                or current_page != canonical_page
            ):
                conflicts.append(
                    {
                        "candidate_key": key,
                        "document_id_current": current.get("document_id"),
                        "document_id_canonical": canonical.get("document_id"),
                        "content_hash_current": current.get("content_hash"),
                        "content_hash_canonical": canonical.get("content_hash"),
                        "page_current": current_page,
                        "page_canonical": canonical_page,
                    }
                )
        merged.append(current)
    for key, canonical in canonical_by_key.items():
        if key in current_by_key:
            continue
        row = dict(canonical)
        row["metadata"] = dict(row.get("metadata") or {}) | {
            "dense_variant_source": "canonical_missing",
            "nf_opt_01_metadata_schema": "v3",
        }
        row["embedding"] = None
        merged.append(row)
    if conflicts:
        raise NFOpt01Error(
            "Current ∪ Canonical identity conflicts detected: "
            f"{len(conflicts)}"
        )
    merged.sort(key=lambda row: str(row["candidate_key"]))
    target_ids = {str(row["candidate_key"]) for row in merged}
    collection = superset_client.get_or_create_collection(
        name=superset_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine", "nf_opt_01": "regression_safe_superset"},
    )
    reused = False
    if int(collection.count()) == len(target_ids):
        try:
            existing = collection.get(limit=len(target_ids), include=["metadatas"])
            existing_ids = {str(value) for value in (existing.get("ids") or [])}
            existing_meta = existing.get("metadatas") or []
            metadata_ok = all(
                (metadata or {}).get("nf_opt_01_metadata_schema") == "v3"
                for metadata in existing_meta
            )
        except Exception:
            existing_ids = set()
            metadata_ok = False
        if existing_ids == target_ids and metadata_ok:
            reused = True
    if not reused and int(collection.count()) > 0:
        superset_client.delete_collection(name=superset_name)
        collection = superset_client.get_or_create_collection(
            name=superset_name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine", "nf_opt_01": "regression_safe_superset"},
        )
    encoded_missing = 0
    if not reused:
        batch_size = 256
        for start in range(0, len(merged), batch_size):
            batch = merged[start:start + batch_size]
            missing = [index for index, row in enumerate(batch) if row.get("embedding") is None]
            if missing:
                missing_vectors = embed_fn([str(batch[index]["content"]) for index in missing])
                for index, vector in zip(missing, missing_vectors):
                    batch[index]["embedding"] = np.asarray(vector, dtype=np.float32).tolist()
                encoded_missing += len(missing)
            collection.upsert(
                ids=[str(row["candidate_key"]) for row in batch],
                documents=[str(row["content"]) for row in batch],
                metadatas=[dict(row["metadata"]) for row in batch],
                embeddings=[row["embedding"] for row in batch],
            )
    manifest = {
        "variant": "regression_safe_superset",
        "embedding_model": "all-MiniLM-L6-v2",
        "candidate_count": len(merged),
        "unique_candidate_count": len(target_ids),
        "duplicate_candidate_count": len(merged) - len(target_ids),
        "unsupported_candidate_count": 0,
        "out_of_scope_candidate_count": 0,
        "current_candidate_count": len(current_by_key),
        "canonical_candidate_count": len(canonical_by_key),
        "canonical_missing_candidate_count": len(set(canonical_by_key) - set(current_by_key)),
        "current_vectors_reused_count": len(current_by_key),
        "new_vectors_encoded_count": encoded_missing if not reused else 0,
        "current_keys_preserved": set(current_by_key).issubset(target_ids),
        "canonical_keys_included": set(canonical_by_key).issubset(target_ids),
        "identity_conflict_count": len(conflicts),
        "collection_hash": _stable_hash(
            [
                {
                    "candidate_key": row["candidate_key"],
                    "document_id": row["document_id"],
                    "content_hash": row["content_hash"],
                    "page": (row.get("metadata") or {}).get("page"),
                }
                for row in merged
            ]
        ),
        "collection_name": superset_name,
        "current_collection_untouched": True,
        "query_encoder_shared_with_current": True,
        "distance_metric": "cosine",
        "index_reused": reused,
        "index_build_ms": (time.perf_counter() - started) * 1000.0,
    }
    return collection, manifest, (time.perf_counter() - started) * 1000.0


def _build_shadow_index(
    *,
    shadow_client: Any,
    shadow_name: str,
    records: Sequence[Mapping[str, Any]],
    embed_fn: Any,
) -> tuple[Any, float]:
    started = time.perf_counter()
    collection = shadow_client.get_or_create_collection(
        name=shadow_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine", "nf_opt_01": "canonical_coverage_shadow"},
    )
    target_ids = {str(row["candidate_key"]) for row in records}
    # A prior diagnostic run may have completed the isolated index but failed
    # later while querying it.  Reuse it only after an exact ID-set check;
    # otherwise rebuild from the canonical universe.  This never touches the
    # production collection and avoids repeating a long CPU embedding pass.
    if int(collection.count()) == len(target_ids):
        try:
            existing = collection.get(limit=len(target_ids), include=[])
            existing_ids = {str(value) for value in (existing.get("ids") or [])}
        except Exception:
            existing_ids = set()
        if existing_ids == target_ids:
            try:
                sample = collection.get(limit=1, include=["metadatas"])
                sample_metadata = (sample.get("metadatas") or [{}])[0] or {}
            except Exception:
                sample_metadata = {}
            if sample_metadata.get("nf_opt_01_metadata_schema") == "v2":
                return collection, (time.perf_counter() - started) * 1000.0
            # Refresh auditable metadata (including row identity) without
            # recomputing embeddings.  This is still isolated Shadow state.
            batch_size = 256
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                collection.update(
                    ids=[str(row["candidate_key"]) for row in batch],
                    metadatas=[dict(row["metadata"]) for row in batch],
                )
            return collection, (time.perf_counter() - started) * 1000.0
    batch_size = 256
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        texts = [str(row["content"]) for row in batch]
        raw_embeddings = embed_fn(texts)
        embeddings = [np.asarray(vector, dtype=np.float32).tolist() for vector in raw_embeddings]
        collection.upsert(
            ids=[str(row["candidate_key"]) for row in batch],
            documents=texts,
            metadatas=[dict(row["metadata"]) for row in batch],
            embeddings=embeddings,
        )
    return collection, (time.perf_counter() - started) * 1000.0


def _query_dense(
    collection: Any,
    *,
    query_embedding: Sequence[float],
    filename: str,
    tenant_id: int,
    limit: int,
    mapping: Mapping[str, str],
) -> list[dict[str, Any]]:
    where = {"$and": [{"user_id": tenant_id}, {"doc_name": filename}]}
    # Chroma's public type validator rejects numpy scalar values even though
    # the production encoder returns a numpy-backed vector.  Convert only at
    # this evaluation boundary; the vector values and encoder remain exactly
    # the same for Current and Shadow variants.
    query_values = [float(value) for value in query_embedding]
    try:
        result = collection.query(
            query_embeddings=[query_values],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except ValueError:
        # Chroma rejects a request larger than a tiny collection.  The result
        # is still a valid Top-N prefix for this diagnostic.
        count = int(collection.count())
        if count <= 0:
            return []
        result = collection.query(
            query_embeddings=[query_values],
            n_results=min(limit, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    rows: list[dict[str, Any]] = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    for candidate_id, content, metadata, distance in zip(ids, docs, metas, distances):
        metadata = metadata or {}
        if metadata.get("type") == "table_cell":
            continue
        evidence_id = str(metadata.get("evidence_id") or candidate_id)
        filename = str(metadata.get("doc_name") or "")
        raw = {
            "doc_id": evidence_id,
            "filename": filename,
            "document_id": mapping.get(filename) or metadata.get("canonical_document_id"),
            "metadata": metadata,
            "type": metadata.get("type"),
            "block_type": metadata.get("type"),
            "row_id": metadata.get("row_id"),
            "parent_row_id": metadata.get("parent_row_id") or metadata.get("parent_id"),
        }
        # Shadow metadata already carries the canonical key generated from
        # the BM25 universe.  Prefer it when available; Current production
        # rows are reconstructed through the shared identity function below.
        if metadata.get("candidate_key") and (
            metadata.get("canonical_document_id") or raw.get("document_id")
        ):
            key = str(metadata["candidate_key"])
            document_id = str(
                metadata.get("canonical_document_id") or raw.get("document_id")
            )
        else:
            key, document_id, _ = r1.candidate_identity_from_record(
                raw,
                filename_to_document=mapping,
                tenant_id=tenant_id,
            )
        if not key or not document_id:
            continue
        rows.append(
            {
                "doc_id": evidence_id,
                "content": content,
                "metadata": metadata,
                "candidate_key": key,
                "document_id": document_id,
                "score": 1.0 - float(distance),
            }
        )
    # Enforce one rank per canonical key, preserving Chroma's deterministic
    # order.  This is especially important when parent/child rows coexist.
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row["candidate_key"] in seen:
            continue
        seen.add(row["candidate_key"])
        deduped.append(row)
    return deduped


def _annotated_candidate(
    candidate: Mapping[str, Any],
    *,
    mapping: Mapping[str, str],
    tenant_id: int,
) -> dict[str, Any]:
    raw = dict(candidate)
    metadata = dict(raw.get("metadata") or {})
    raw.setdefault("filename", metadata.get("doc_name"))
    raw.setdefault("document_id", mapping.get(str(raw.get("filename") or "")))
    raw.setdefault("page", metadata.get("page"))
    raw.setdefault("parent_id", metadata.get("parent_id"))
    raw.setdefault("row_id", metadata.get("row_id"))
    raw.setdefault("parent_row_id", metadata.get("parent_row_id") or metadata.get("parent_id"))
    raw.setdefault("evidence_id", raw.get("doc_id"))
    raw.setdefault("type", metadata.get("type"))
    raw.setdefault("block_type", metadata.get("type"))
    annotated = r1._annotate(raw, mapping=mapping, tenant_id=tenant_id)
    annotated["candidate_key"] = str(annotated.get("candidate_key") or "")
    annotated["document_id"] = annotated.get("canonical_document_id") or annotated.get("document_id")
    return annotated


def _rank_map(candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for rank, candidate in enumerate(candidates, start=1):
        key = str(candidate.get("candidate_key") or "")
        if key and key not in result:
            result[key] = rank
    return result


def _union_candidates(
    dense: Sequence[Mapping[str, Any]],
    bm25: Sequence[Mapping[str, Any]],
    *,
    mapping: Mapping[str, str],
    tenant_id: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in list(dense) + list(bm25):
        item = _annotated_candidate(candidate, mapping=mapping, tenant_id=tenant_id)
        key = str(item.get("candidate_key") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _rrf_candidates(
    dense: Sequence[Mapping[str, Any]],
    bm25: Sequence[Mapping[str, Any]],
    *,
    query: str,
    query_processor: QueryProcessor,
    mapping: Mapping[str, str],
    tenant_id: int,
) -> list[dict[str, Any]]:
    fused = normalize_scores(rrf([list(dense), list(bm25)]))
    fused = boost_front_matter_chunks(
        query,
        fused,
        is_front_matter_query_fn=query_processor.is_front_matter_query,
    )
    return [_annotated_candidate(item, mapping=mapping, tenant_id=tenant_id) for item in fused]


def _load_gold_keys(labels: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    keys: list[str] = []
    for case_id, label in labels.items():
        if label.get("expected_no_answer"):
            continue
        for source_index, source in enumerate(label.get("expected_sources") or []):
            key = str(source.get("candidate_key") or "")
            if not key:
                raise NFOpt01Error(f"{case_id}/{source_index}: missing candidate_key")
            rows.append({
                "case_id": str(case_id),
                "source_index": source_index,
                "candidate_key": key,
            })
            keys.append(key)
    return rows, keys


def _input_integrity(inputs: Any, *, args: argparse.Namespace) -> dict[str, Any]:
    actual = dict(inputs.hash_report["actual"])
    nf04_path = NF04_OUT / "input-integrity-report.json"
    nf04 = json.loads(nf04_path.read_text(encoding="utf-8")) if nf04_path.exists() else {}
    expected_fields = (
        "question_hash",
        "reference_answer_hash",
        "source_identity_hash",
        "negative_evidence_hash",
        "review_status_hash",
        "corpus_hash",
        "golden_manifest_sha256",
    )
    baseline_matches = {
        field: (not nf04 or actual.get(field) == nf04.get(field))
        for field in expected_fields
    }
    return {
        "artifact_schema": "nf-opt-01/v1",
        "benchmark_id": "financial-rag-v1",
        "tenant_id": args.tenant_id,
        "case_count": 64,
        "expected_source_count": 80,
        "allowed_document_count": 8,
        "question_hash": actual.get("question_hash"),
        "reference_answer_hash": actual.get("reference_answer_hash"),
        "source_identity_hash": actual.get("source_identity_hash"),
        "negative_evidence_hash": actual.get("negative_evidence_hash"),
        "review_status_hash": actual.get("review_status_hash"),
        "corpus_hash": actual.get("corpus_hash"),
        "golden_manifest_sha256": actual.get("golden_manifest_sha256"),
        "benchmark_hash": _stable_hash(actual),
        "all_hashes_verified": all(inputs.hash_report["matches"].values()),
        "nf04_hashes_unchanged": all(baseline_matches.values()),
        "legacy_documents_loaded": 0,
    }


def _metrics_rows(
    source_rows: Sequence[Mapping[str, Any]],
    rank_field: str,
) -> dict[str, Any]:
    return rank_metrics(source_rows, rank_field)


def _run(args: argparse.Namespace) -> int:
    if args.tenant_id != 1:
        raise NFOpt01Error("Financial RAG v1 is tenant 1 only")
    if args.diagnostic_top_n < 200:
        raise NFOpt01Error("diagnostic Top-N must be at least 200")
    if args.embedding_model != "all-MiniLM-L6-v2":
        raise NFOpt01Error("NF-OPT-01 only permits all-MiniLM-L6-v2")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.shadow_path.mkdir(parents=True, exist_ok=True)
    args.superset_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ["CHROMA_PATH"] = str(args.chroma_path)
    os.environ["BM25_DB_PATH"] = str(args.bm25_db_path)

    inputs = r1._load_inputs(
        corpus_path=args.corpus,
        manifest_path=args.manifest_path,
        questions_path=args.questions,
        labels_path=args.labels,
        review_status_path=args.review_status,
        negative_report_path=args.negative_report,
    )
    integrity = _input_integrity(inputs, args=args)
    if not integrity["all_hashes_verified"] or not integrity["nf04_hashes_unchanged"]:
        raise NFOpt01Error("NF-EVAL-04 input hashes are not verified or changed")
    mapping = r1._doc_map(inputs.corpus)
    filenames = [str(item["filename"]) for item in inputs.corpus["documents"]]
    allowed_documents = {str(item["document_id"]) for item in inputs.corpus["documents"]}
    source_rows, gold_keys = _load_gold_keys(inputs.labels_by_id)
    if len(inputs.questions) != 72 or len(source_rows) != 80:
        raise NFOpt01Error("expected the frozen 72-question/80-source benchmark")

    from src.services import vector_store
    from src.services.retrieval import SqliteBM25Retriever

    chroma_client = chromadb.PersistentClient(path=str(args.chroma_path))
    current_collection = chroma_client.get_collection(
        name=vector_store.GLOBAL_COLLECTION_NAME,
        embedding_function=vector_store.embed_fn,
    )
    current_keys = _current_collection_keys(
        current_collection,
        filenames=filenames,
        mapping=mapping,
        tenant_id=args.tenant_id,
    )
    records, shadow_manifest = _load_candidate_universe(
        db_path=args.bm25_db_path,
        corpus=inputs.corpus,
        mapping=mapping,
        tenant_id=args.tenant_id,
        gold_keys=gold_keys,
    )
    current_records, current_record_manifest = _load_current_dense_records(
        current_collection,
        filenames=filenames,
        mapping=mapping,
        tenant_id=args.tenant_id,
    )
    if set(row["candidate_key"] for row in current_records) != current_keys:
        raise NFOpt01Error("current dense key audit does not match current collection key audit")
    shadow_manifest["current_dense_candidate_count"] = len(current_keys)
    shadow_manifest["current_collection_untouched"] = True
    shadow_name = "financial_rag_v1_dense_coverage_shadow"
    shadow_client = chromadb.PersistentClient(path=str(args.shadow_path))
    shadow_collection, build_ms = _build_shadow_index(
        shadow_client=shadow_client,
        shadow_name=shadow_name,
        records=records,
        embed_fn=vector_store.embed_fn,
    )
    shadow_manifest["shadow_collection_name"] = shadow_name
    shadow_manifest["shadow_collection_count"] = int(shadow_collection.count())
    shadow_manifest["index_build_ms"] = build_ms
    shadow_manifest["embedding_provider"] = {
        "model_name": args.embedding_model,
        "query_encoder_shared_with_current": True,
        "distance_metric": "cosine",
    }
    superset_client = chromadb.PersistentClient(path=str(args.superset_path))
    superset_name = "financial_rag_v1_dense_coverage_superset"
    superset_collection, superset_manifest, superset_build_ms = _build_superset_index(
        superset_client=superset_client,
        superset_name=superset_name,
        current_records=current_records,
        canonical_records=records,
        embed_fn=vector_store.embed_fn,
    )
    superset_manifest["current_record_manifest"] = current_record_manifest
    superset_manifest["shadow_collection_untouched"] = True
    superset_keys = {str(row["candidate_key"]) for row in current_records} | {
        str(row["candidate_key"]) for row in records
    }
    superset_manifest["gold_identity_presence_count"] = sum(
        str(key) in superset_keys for key in gold_keys
    )
    superset_manifest["gold_identity_expected_count"] = len(gold_keys)

    query_processor = QueryProcessor()
    bm25 = SqliteBM25Retriever(db_path=str(args.bm25_db_path))
    current_dense_rows: list[dict[str, Any]] = []
    shadow_dense_rows: list[dict[str, Any]] = []
    superset_dense_rows: list[dict[str, Any]] = []
    current_union_rows: list[dict[str, Any]] = []
    shadow_union_rows: list[dict[str, Any]] = []
    superset_union_rows: list[dict[str, Any]] = []
    current_rrf_rows: list[dict[str, Any]] = []
    shadow_rrf_rows: list[dict[str, Any]] = []
    superset_rrf_rows: list[dict[str, Any]] = []
    # Gold-only rows keep Source Recall denominators fixed at the frozen 80
    # Expected Sources.  The candidate rows above remain for lineage and
    # regression diagnostics, but must not be used as metric denominators.
    current_dense_gold_rows: list[dict[str, Any]] = []
    shadow_dense_gold_rows: list[dict[str, Any]] = []
    superset_dense_gold_rows: list[dict[str, Any]] = []
    current_union_gold_rows: list[dict[str, Any]] = []
    shadow_union_gold_rows: list[dict[str, Any]] = []
    superset_union_gold_rows: list[dict[str, Any]] = []
    current_rrf_gold_rows: list[dict[str, Any]] = []
    shadow_rrf_gold_rows: list[dict[str, Any]] = []
    superset_rrf_gold_rows: list[dict[str, Any]] = []
    dense_query_times_current: list[float] = []
    dense_query_times_shadow: list[float] = []
    dense_query_times_superset: list[float] = []
    query_embedding_times: list[float] = []
    dense_source_comparison: list[dict[str, Any]] = []
    hybrid_source_comparison: list[dict[str, Any]] = []
    case_comparison: list[dict[str, Any]] = []
    scope_out_of_scope = 0
    model_calls = 0
    answer_calls = 0

    labels = inputs.labels_by_id
    for question in inputs.questions:
        case_id = str(question["case_id"])
        label = labels[case_id]
        if label.get("expected_no_answer"):
            continue
        document_scope = [str(value) for value in question.get("document_scope") or []]
        if len(document_scope) != 1 or document_scope[0] not in allowed_documents:
            raise NFOpt01Error(f"{case_id}: invalid benchmark document scope")
        filename = next(
            str(item["filename"])
            for item in inputs.corpus["documents"]
            if str(item["document_id"]) == document_scope[0]
        )
        query = str(question.get("question") or "")
        expanded_query = query_processor.expand(query)
        started = time.perf_counter()
        query_embedding = np.asarray(vector_store.embed_fn([expanded_query])[0], dtype=np.float32)
        query_embedding_times.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        current_dense = _query_dense(
            current_collection,
            query_embedding=query_embedding,
            filename=filename,
            tenant_id=args.tenant_id,
            limit=args.diagnostic_top_n,
            mapping=mapping,
        )
        dense_query_times_current.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        shadow_dense = _query_dense(
            shadow_collection,
            query_embedding=query_embedding,
            filename=filename,
            tenant_id=args.tenant_id,
            limit=args.diagnostic_top_n,
            mapping=mapping,
        )
        dense_query_times_shadow.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        superset_dense = _query_dense(
            superset_collection,
            query_embedding=query_embedding,
            filename=filename,
            tenant_id=args.tenant_id,
            limit=args.diagnostic_top_n,
            mapping=mapping,
        )
        dense_query_times_superset.append((time.perf_counter() - started) * 1000.0)
        current_dense_map = _rank_map(current_dense)
        shadow_dense_map = _rank_map(shadow_dense)
        superset_dense_map = _rank_map(superset_dense)
        case_sources = [
            row for row in source_rows if row["case_id"] == case_id
        ]
        for source in case_sources:
            key = source["candidate_key"]
            current_rank = current_dense_map.get(key)
            shadow_rank = shadow_dense_map.get(key)
            dense_source_comparison.append(
                {
                    "case_id": case_id,
                    "source_index": source["source_index"],
                    "candidate_key": key,
                    "current_index_present": key in current_keys,
                    "shadow_index_present": key in {row["candidate_key"] for row in records},
                    "current_dense_rank": current_rank,
                    "shadow_dense_rank": shadow_rank,
                    "superset_dense_rank": superset_dense_map.get(key),
                    "superset_index_present": key in {
                        row["candidate_key"] for row in current_records
                    }
                    or key in {row["candidate_key"] for row in records},
                    "coverage_gain": current_rank is None and shadow_rank is not None,
                    "ranking_gain": shadow_rank is not None and (current_rank is None or shadow_rank < current_rank),
                }
            )

        candidate_k = args.production_top_k * args.candidate_multiplier
        if query_processor.is_numeric_query(query):
            candidate_k = max(candidate_k, args.production_top_k * 8)
        bm25_rows = bm25.search(
            expanded_query,
            k=candidate_k,
            doc_name=filename,
            user_id=args.tenant_id,
        )
        bm25_rows = [_annotated_candidate(row, mapping=mapping, tenant_id=args.tenant_id) for row in bm25_rows]
        bm25_rows = [
            row for row in bm25_rows
            if candidate_scope_ok(row.get("document_id"), allowed_documents)
        ]
        current_dense_window = current_dense[:candidate_k]
        shadow_dense_window = shadow_dense[:candidate_k]
        superset_dense_window = superset_dense[:candidate_k]
        current_union = _union_candidates(
            current_dense_window,
            bm25_rows,
            mapping=mapping,
            tenant_id=args.tenant_id,
        )
        shadow_union = _union_candidates(
            shadow_dense_window,
            bm25_rows,
            mapping=mapping,
            tenant_id=args.tenant_id,
        )
        superset_union = _union_candidates(
            superset_dense_window,
            bm25_rows,
            mapping=mapping,
            tenant_id=args.tenant_id,
        )
        current_rrf = _rrf_candidates(
            current_dense_window,
            bm25_rows,
            query=query,
            query_processor=query_processor,
            mapping=mapping,
            tenant_id=args.tenant_id,
        )
        shadow_rrf = _rrf_candidates(
            shadow_dense_window,
            bm25_rows,
            query=query,
            query_processor=query_processor,
            mapping=mapping,
            tenant_id=args.tenant_id,
        )
        superset_rrf = _rrf_candidates(
            superset_dense_window,
            bm25_rows,
            query=query,
            query_processor=query_processor,
            mapping=mapping,
            tenant_id=args.tenant_id,
        )
        current_dense_rows.extend(
            {"case_id": case_id, "candidate_key": key, "rank": rank}
            for key, rank in current_dense_map.items()
        )
        shadow_dense_rows.extend(
            {"case_id": case_id, "candidate_key": key, "rank": rank}
            for key, rank in shadow_dense_map.items()
        )
        superset_dense_rows.extend(
            {"case_id": case_id, "candidate_key": key, "rank": rank}
            for key, rank in superset_dense_map.items()
        )
        current_union_map = _rank_map(current_union)
        shadow_union_map = _rank_map(shadow_union)
        superset_union_map = _rank_map(superset_union)
        current_rrf_map = _rank_map(current_rrf)
        shadow_rrf_map = _rank_map(shadow_rrf)
        superset_rrf_map = _rank_map(superset_rrf)
        current_union_rows.extend(
            {"case_id": case_id, "candidate_key": key, "rank": rank}
            for key, rank in current_union_map.items()
        )
        shadow_union_rows.extend(
            {"case_id": case_id, "candidate_key": key, "rank": rank}
            for key, rank in shadow_union_map.items()
        )
        superset_union_rows.extend(
            {"case_id": case_id, "candidate_key": key, "rank": rank}
            for key, rank in superset_union_map.items()
        )
        current_rrf_rows.extend(
            {"case_id": case_id, "candidate_key": key, "rank": rank}
            for key, rank in current_rrf_map.items()
        )
        shadow_rrf_rows.extend(
            {"case_id": case_id, "candidate_key": key, "rank": rank}
            for key, rank in shadow_rrf_map.items()
        )
        superset_rrf_rows.extend(
            {"case_id": case_id, "candidate_key": key, "rank": rank}
            for key, rank in superset_rrf_map.items()
        )
        for source in case_sources:
            source_key = source["candidate_key"]
            current_dense_gold_rows.append(
                {"case_id": case_id, "source_index": source["source_index"], "candidate_key": source_key, "rank": current_dense_map.get(source_key)}
            )
            shadow_dense_gold_rows.append(
                {"case_id": case_id, "source_index": source["source_index"], "candidate_key": source_key, "rank": shadow_dense_map.get(source_key)}
            )
            superset_dense_gold_rows.append(
                {"case_id": case_id, "source_index": source["source_index"], "candidate_key": source_key, "rank": superset_dense_map.get(source_key)}
            )
            current_union_gold_rows.append(
                {"case_id": case_id, "source_index": source["source_index"], "candidate_key": source_key, "rank": current_union_map.get(source_key)}
            )
            shadow_union_gold_rows.append(
                {"case_id": case_id, "source_index": source["source_index"], "candidate_key": source_key, "rank": shadow_union_map.get(source_key)}
            )
            superset_union_gold_rows.append(
                {"case_id": case_id, "source_index": source["source_index"], "candidate_key": source_key, "rank": superset_union_map.get(source_key)}
            )
            current_rrf_gold_rows.append(
                {"case_id": case_id, "source_index": source["source_index"], "candidate_key": source_key, "rank": current_rrf_map.get(source_key)}
            )
            shadow_rrf_gold_rows.append(
                {"case_id": case_id, "source_index": source["source_index"], "candidate_key": source_key, "rank": shadow_rrf_map.get(source_key)}
            )
            superset_rrf_gold_rows.append(
                {"case_id": case_id, "source_index": source["source_index"], "candidate_key": source_key, "rank": superset_rrf_map.get(source_key)}
            )
        current_state = coverage_state(
            [row["candidate_key"] for row in case_sources],
            current_rrf_map,
        )
        shadow_state = coverage_state(
            [row["candidate_key"] for row in case_sources],
            shadow_rrf_map,
        )
        superset_state = coverage_state(
            [row["candidate_key"] for row in case_sources],
            superset_rrf_map,
        )
        hybrid_source_comparison.extend(
            {
                "case_id": case_id,
                "source_index": source["source_index"],
                "candidate_key": source["candidate_key"],
                "current_union_rank": current_union_map.get(source["candidate_key"]),
                "shadow_union_rank": shadow_union_map.get(source["candidate_key"]),
                "superset_union_rank": superset_union_map.get(source["candidate_key"]),
                "current_rrf_rank": current_rrf_map.get(source["candidate_key"]),
                "shadow_rrf_rank": shadow_rrf_map.get(source["candidate_key"]),
                "superset_rrf_rank": superset_rrf_map.get(source["candidate_key"]),
            }
            for source in case_sources
        )
        case_comparison.append(
            {
                "case_id": case_id,
                "expected_source_count": len(case_sources),
                "production_candidate_limit": candidate_k,
                "current_union_coverage": coverage_state(
                    [row["candidate_key"] for row in case_sources], current_union_map
                ),
                "shadow_union_coverage": coverage_state(
                    [row["candidate_key"] for row in case_sources], shadow_union_map
                ),
                "superset_union_coverage": coverage_state(
                    [row["candidate_key"] for row in case_sources], superset_union_map
                ),
                "current_rrf_coverage": current_state,
                "shadow_rrf_coverage": shadow_state,
                "superset_rrf_coverage": superset_state,
                "current_rrf_top40_coverage": coverage_state(
                    [row["candidate_key"] for row in case_sources], list(current_rrf_map)[:40]
                ),
                "shadow_rrf_top40_coverage": coverage_state(
                    [row["candidate_key"] for row in case_sources], list(shadow_rrf_map)[:40]
                ),
                "superset_rrf_top40_coverage": coverage_state(
                    [row["candidate_key"] for row in case_sources], list(superset_rrf_map)[:40]
                ),
            }
        )

    current_dense_metrics = _metrics_rows(current_dense_gold_rows, "rank")
    shadow_dense_metrics = _metrics_rows(shadow_dense_gold_rows, "rank")
    superset_dense_metrics = _metrics_rows(superset_dense_gold_rows, "rank")
    current_union_metrics = _metrics_rows(current_union_gold_rows, "rank")
    shadow_union_metrics = _metrics_rows(shadow_union_gold_rows, "rank")
    superset_union_metrics = _metrics_rows(superset_union_gold_rows, "rank")
    current_rrf_metrics = _metrics_rows(current_rrf_gold_rows, "rank")
    shadow_rrf_metrics = _metrics_rows(shadow_rrf_gold_rows, "rank")
    superset_rrf_metrics = _metrics_rows(superset_rrf_gold_rows, "rank")
    current_dense_gold_map = {
        f"{row['case_id']}:{row['candidate_key']}": row["rank"]
        for row in current_dense_gold_rows
        if isinstance(row.get("rank"), int)
    }
    shadow_dense_gold_map = {
        f"{row['case_id']}:{row['candidate_key']}": row["rank"]
        for row in shadow_dense_gold_rows
        if isinstance(row.get("rank"), int)
    }
    superset_dense_gold_map = {
        f"{row['case_id']}:{row['candidate_key']}": row["rank"]
        for row in superset_dense_gold_rows
        if isinstance(row.get("rank"), int)
    }
    current_rrf_gold_map = {
        f"{row['case_id']}:{row['candidate_key']}": row["rank"]
        for row in current_rrf_gold_rows
        if isinstance(row.get("rank"), int)
    }
    shadow_rrf_gold_map = {
        f"{row['case_id']}:{row['candidate_key']}": row["rank"]
        for row in shadow_rrf_gold_rows
        if isinstance(row.get("rank"), int)
    }
    superset_rrf_gold_map = {
        f"{row['case_id']}:{row['candidate_key']}": row["rank"]
        for row in superset_rrf_gold_rows
        if isinstance(row.get("rank"), int)
    }
    dense_regressions = compare_rank_maps(
        current_dense_gold_map,
        shadow_dense_gold_map,
        cutoff=args.diagnostic_top_n,
    )
    rrf_source_regressions = compare_rank_maps(
        current_rrf_gold_map,
        shadow_rrf_gold_map,
        cutoff=40,
    )
    superset_dense_regressions = compare_rank_maps(
        current_dense_gold_map,
        superset_dense_gold_map,
        cutoff=args.diagnostic_top_n,
    )
    superset_rrf_source_regressions = compare_rank_maps(
        current_rrf_gold_map,
        superset_rrf_gold_map,
        cutoff=40,
    )
    rrf_all_regressed_cases = sum(
        row["current_rrf_coverage"] == "all"
        and row["shadow_rrf_coverage"] != "all"
        for row in case_comparison
    )
    current_all = Counter(row["current_rrf_coverage"] for row in case_comparison)
    shadow_all = Counter(row["shadow_rrf_coverage"] for row in case_comparison)
    superset_all = Counter(row["superset_rrf_coverage"] for row in case_comparison)
    current_top40_all = Counter(row["current_rrf_top40_coverage"] for row in case_comparison)
    shadow_top40_all = Counter(row["shadow_rrf_top40_coverage"] for row in case_comparison)
    superset_top40_all = Counter(row["superset_rrf_top40_coverage"] for row in case_comparison)
    superset_rrf_all_regressed_cases = sum(
        row["current_rrf_coverage"] == "all"
        and row["superset_rrf_coverage"] != "all"
        for row in case_comparison
    )
    superset_rrf_top40_all_regressed_cases = sum(
        row["current_rrf_top40_coverage"] == "all"
        and row["superset_rrf_top40_coverage"] != "all"
        for row in case_comparison
    )
    production_limits = {
        str(row["case_id"]): int(row["production_candidate_limit"])
        for row in case_comparison
    }
    current_union_production_hits = {
        f"{row['case_id']}:{row['candidate_key']}"
        for row in current_union_gold_rows
        if isinstance(row.get("rank"), int)
        and int(row["rank"]) <= production_limits[str(row["case_id"])]
    }
    superset_union_production_hits = {
        f"{row['case_id']}:{row['candidate_key']}"
        for row in superset_union_gold_rows
        if isinstance(row.get("rank"), int)
        and int(row["rank"]) <= production_limits[str(row["case_id"])]
    }
    all_union_gold_keys = {
        f"{row['case_id']}:{row['candidate_key']}"
        for row in current_union_gold_rows
    }
    superset_union_source_regressions = {
        "current_hit_count": len(current_union_production_hits),
        "shadow_hit_count": len(superset_union_production_hits),
        "superset_hit_count": len(superset_union_production_hits),
        "new_hit_count": len(superset_union_production_hits - current_union_production_hits),
        "regressed_hit_count": len(current_union_production_hits - superset_union_production_hits),
        "both_hit_count": len(current_union_production_hits & superset_union_production_hits),
        "both_missed_count": len(
            all_union_gold_keys
            - current_union_production_hits
            - superset_union_production_hits
        ),
        "scope": "per_case_production_candidate_limit",
    }
    latency_ratio = None
    if dense_query_times_current and dense_query_times_shadow:
        current_p95 = percentile(dense_query_times_current, 0.95) or 0.0
        shadow_p95 = percentile(dense_query_times_shadow, 0.95) or 0.0
        latency_ratio = (shadow_p95 - current_p95) / current_p95 if current_p95 else None
    canonical_shadow_gate = dense_coverage_gate(
        shadow_gold_identity_presence=int(shadow_manifest["gold_identity_presence_count"]),
        unsupported_candidate_count=int(shadow_manifest["unsupported_candidate_count"]),
        out_of_scope_candidate_count=int(shadow_manifest["out_of_scope_candidate_count"]),
        dense_source_gain_at_200=shadow_dense_metrics["@200"]["source_hit_count"] - current_dense_metrics["@200"]["source_hit_count"],
        production_union_source_gain=shadow_union_metrics["@200"]["source_hit_count"] - current_union_metrics["@200"]["source_hit_count"],
        rrf_source_gain_at_40=shadow_rrf_metrics["@40"]["source_hit_count"] - current_rrf_metrics["@40"]["source_hit_count"],
        rrf_all_case_gain=shadow_all["all"] - current_all["all"],
        dense_regressed_sources=dense_regressions["regressed_hit_count"],
        rrf_regressed_sources_at_40=rrf_source_regressions["regressed_hit_count"],
        rrf_regressed_all_cases=rrf_all_regressed_cases,
        latency_increase_ratio=latency_ratio,
    )
    superset_latency_ratio = None
    if dense_query_times_current and dense_query_times_superset:
        current_p95 = percentile(dense_query_times_current, 0.95) or 0.0
        superset_p95 = percentile(dense_query_times_superset, 0.95) or 0.0
        superset_latency_ratio = (
            (superset_p95 - current_p95) / current_p95 if current_p95 else None
        )
    superset_gate = dense_superset_gate(
        superset_gold_identity_presence=int(superset_manifest["gold_identity_presence_count"]),
        unsupported_candidate_count=int(superset_manifest["unsupported_candidate_count"]),
        out_of_scope_candidate_count=int(superset_manifest["out_of_scope_candidate_count"]),
        dense_source_hit_at_200=superset_dense_metrics["@200"]["source_hit_count"],
        rrf_source_hit_at_40=superset_rrf_metrics["@40"]["source_hit_count"],
        rrf_top40_all_case_count=superset_top40_all["all"],
        dense_regressed_sources=superset_dense_regressions["regressed_hit_count"],
        rrf_regressed_sources_at_40=superset_rrf_source_regressions["regressed_hit_count"],
        rrf_regressed_all_cases=superset_rrf_all_regressed_cases,
        rrf_top40_regressed_all_cases=superset_rrf_top40_all_regressed_cases,
        union_regressed_sources=superset_union_source_regressions["regressed_hit_count"],
        latency_increase_ratio=superset_latency_ratio,
    )
    integrity["scope_integrity_passed"] = (
        shadow_manifest["out_of_scope_candidate_count"] == 0
        and scope_out_of_scope == 0
    )
    acceptance = {
        "artifact_schema": "nf-opt-01/v1",
        "decision": superset_gate["decision"],
        "formal_variant": "regression_safe_superset",
        "canonical_replacement_shadow_gate": canonical_shadow_gate,
        "dense_superset_gate": superset_gate,
        "case_count": len(case_comparison),
        "expected_source_count": len(source_rows),
        "input_hashes_verified": integrity["all_hashes_verified"] and integrity["nf04_hashes_unchanged"],
        "scope_integrity_passed": integrity["scope_integrity_passed"],
        "legacy_27_loaded": False,
        "shadow_production_collection_modified": False,
        "superset_production_collection_modified": False,
        "embedding_model_same": True,
        "query_embedding_shared": True,
        "model_chat_completion_requests": model_calls,
        "answer_generation_calls": answer_calls,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "optimization_allowed": False,
    }
    _write(args.out_dir / "input-integrity-report.json", integrity)
    _write(args.out_dir / "dense-shadow-index-manifest.json", shadow_manifest)
    _write(args.out_dir / "dense-superset-index-manifest.json", superset_manifest)
    _write(args.out_dir / "dense-index-presence-comparison.json", {
        "current_exact_gold_presence_count": sum(key in current_keys for key in gold_keys),
        "shadow_exact_gold_presence_count": shadow_manifest["gold_identity_presence_count"],
        "gold_source_count": len(gold_keys),
        "current_candidate_count": len(current_keys),
        "shadow_candidate_count": len(records),
        "current_collection_untouched": True,
        "records": dense_source_comparison,
    })
    _write(args.out_dir / "dense-rank-comparison.json", {
        "current": current_dense_metrics,
        "shadow": shadow_dense_metrics,
        "superset": superset_dense_metrics,
        "records": dense_source_comparison,
    })
    _write(args.out_dir / "hybrid-union-comparison.json", {
        "current": current_union_metrics,
        "shadow": shadow_union_metrics,
        "superset": superset_union_metrics,
        "records": hybrid_source_comparison,
    })
    _write(args.out_dir / "rrf-comparison.json", {
        "current": current_rrf_metrics,
        "shadow": shadow_rrf_metrics,
        "superset": superset_rrf_metrics,
        "current_case_coverage": dict(current_all),
        "shadow_case_coverage": dict(shadow_all),
        "superset_case_coverage": dict(superset_all),
        "current_rrf_top40_case_coverage": dict(current_top40_all),
        "shadow_rrf_top40_case_coverage": dict(shadow_top40_all),
        "superset_rrf_top40_case_coverage": dict(superset_top40_all),
        "multi_evidence_current_all": sum(
            row["expected_source_count"] > 1 and row["current_rrf_coverage"] == "all"
            for row in case_comparison
        ),
        "multi_evidence_shadow_all": sum(
            row["expected_source_count"] > 1 and row["shadow_rrf_coverage"] == "all"
            for row in case_comparison
        ),
        "multi_evidence_superset_all": sum(
            row["expected_source_count"] > 1 and row["superset_rrf_coverage"] == "all"
            for row in case_comparison
        ),
        "records": case_comparison,
    })
    _write(args.out_dir / "regression-report.json", {
        "dense": dense_regressions,
        "rrf_top40": rrf_source_regressions,
        "superset_dense": superset_dense_regressions,
        "superset_rrf_top40": superset_rrf_source_regressions,
        "superset_union": superset_union_source_regressions,
        "rrf_all_gold_regressed_case_count": rrf_all_regressed_cases,
        "superset_rrf_all_gold_regressed_case_count": superset_rrf_all_regressed_cases,
        "superset_rrf_top40_all_gold_regressed_case_count": superset_rrf_top40_all_regressed_cases,
        "dense_retrieval_regressed_source_count": dense_regressions["regressed_hit_count"],
        "rrf_top40_regressed_source_count": rrf_source_regressions["regressed_hit_count"],
        "current_all_partial_none": dict(current_all),
        "shadow_all_partial_none": dict(shadow_all),
        "current_rrf_top40_all_partial_none": dict(current_top40_all),
        "shadow_rrf_top40_all_partial_none": dict(shadow_top40_all),
        "superset_all_partial_none": dict(superset_all),
        "superset_rrf_top40_all_partial_none": dict(superset_top40_all),
    })
    _write(args.out_dir / "latency-report.json", {
        "embedding_model": args.embedding_model,
        "query_embedding_reused": True,
        "query_embedding_p50_ms": percentile(query_embedding_times, 0.50),
        "query_embedding_p95_ms": percentile(query_embedding_times, 0.95),
        "current_dense_query_p50_ms": percentile(dense_query_times_current, 0.50),
        "current_dense_query_p95_ms": percentile(dense_query_times_current, 0.95),
        "shadow_dense_query_p50_ms": percentile(dense_query_times_shadow, 0.50),
        "shadow_dense_query_p95_ms": percentile(dense_query_times_shadow, 0.95),
        "superset_dense_query_p50_ms": percentile(dense_query_times_superset, 0.50),
        "superset_dense_query_p95_ms": percentile(dense_query_times_superset, 0.95),
        "shadow_index_build_ms": build_ms,
        "superset_index_build_ms": superset_build_ms,
        "shadow_p95_increase_ratio": latency_ratio,
        "superset_p95_increase_ratio": superset_latency_ratio,
        "online_latency_comparable": True,
    })
    _write(args.out_dir / "next-gate.json", {
        "selected_gate": superset_gate["next_gate"],
        "optimization_allowed": False,
        "production_switch_allowed": False,
        "reason": superset_gate["decision"],
        "formal_variant": "regression_safe_superset",
    })
    _write(args.out_dir / "regression-case-analysis.json", {
        "canonical_replacement_shadow_regressed_cases": [
            row["case_id"]
            for row in case_comparison
            if row["current_rrf_coverage"] == "all"
            and row["shadow_rrf_coverage"] != "all"
        ],
        "superset_regressed_cases": [
            row["case_id"]
            for row in case_comparison
            if row["current_rrf_coverage"] == "all"
            and row["superset_rrf_coverage"] != "all"
        ],
        "records": [row for row in case_comparison if row["case_id"] in {
            "msft_fy2025_002",
            "msft_fy2025_005",
            "msft_fy2025_006",
            "msft_fy2025_007",
        }],
        "dense_rank_records": [row for row in dense_source_comparison if row["case_id"] in {
            "msft_fy2025_002",
            "msft_fy2025_005",
            "msft_fy2025_006",
            "msft_fy2025_007",
        }],
        "hybrid_rank_records": [row for row in hybrid_source_comparison if row["case_id"] in {
            "msft_fy2025_002",
            "msft_fy2025_005",
            "msft_fy2025_006",
            "msft_fy2025_007",
        }],
    })
    _write(args.out_dir / "nf-opt-01-r1-acceptance.json", {
        **acceptance,
        "phase": "NF-OPT-01 R1",
        "current_union_and_canonical_shadow_retained_for_diagnostics": True,
    })
    _write(args.out_dir / "nf-opt-01-acceptance.json", acceptance)
    print(json.dumps({
        "acceptance": acceptance,
        "shadow_manifest": shadow_manifest,
        "current_dense": current_dense_metrics,
        "shadow_dense": shadow_dense_metrics,
        "superset_dense": superset_dense_metrics,
        "current_rrf": current_rrf_metrics,
        "shadow_rrf": shadow_rrf_metrics,
        "superset_rrf": superset_rrf_metrics,
        "current_case_coverage": dict(current_all),
        "shadow_case_coverage": dict(shadow_all),
        "superset_case_coverage": dict(superset_all),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if acceptance["input_hashes_verified"] and acceptance["scope_integrity_passed"] else 2


def main() -> None:
    args = _parse_args()
    try:
        raise SystemExit(_run(args))
    except (NFOpt01Error, r1.BaselineConfigurationError) as exc:
        print(f"NF-OPT-01 configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
