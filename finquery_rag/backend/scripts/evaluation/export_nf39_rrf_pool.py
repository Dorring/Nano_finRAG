"""Export the frozen RRF Top-40 candidate pool for NF39.

This script freezes the RRF candidate pool once so that both the baseline
(Variant A: Current) and the experiment (Variant B: Rank Fusion) read from
the exact same candidates.  Neither variant re-runs Dense, BM25, or RRF.

Usage::

    python -m scripts.evaluation.export_nf39_rrf_pool \
        --corpus-dir artifacts/evaluation/nf38 \
        --cases eval_data/phase5/sealed/questions.jsonl \
        --out-dir artifacts/evaluation/nf39 \
        --bm25-db-path bm25.db \
        --bm25-user-id 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from src.evaluation.case_fingerprints import (  # noqa: E402
    label_fingerprint,
    question_fingerprint,
)
from src.evaluation.evaluation import load_jsonl_cases  # noqa: E402
from src.evaluation.nf38_corpus import hash_embedding_text  # noqa: E402
from src.evaluation.nf38_dense_index import build_dense_index  # noqa: E402
from src.evaluation.nf38_evaluator import (  # noqa: E402
    EvaluationScope,
    _stable_digest,
    freeze_bm25_pool,
    validate_labeled_cases,
    validate_scope_corpus,
)
from src.retrieval.candidate_fusion import rrf  # noqa: E402
from src.retrieval.embedding_provider import (  # noqa: E402
    ExistingMiniLMEmbeddingProvider,
)


def _load_canonical_records_with_texts(
    corpus_dir: Path, chroma_path: str | None = None
) -> tuple[list, dict]:
    from scripts.evaluation.build_nf38_embedding_indexes import (
        load_canonical_records,
        load_embedding_texts,
    )
    from src.evaluation.nf38_corpus import CanonicalEvidenceRecord

    records, manifest = load_canonical_records(corpus_dir)
    text_map = load_embedding_texts(
        chroma_path or str(BACKEND / "chroma_db"),
        "rag_global_knowledge_base",
    )
    return [
        CanonicalEvidenceRecord(
            evidence_id=record.evidence_id,
            document_id=record.document_id,
            page=record.page,
            block_type=record.block_type,
            parent_id=record.parent_id,
            table_id=record.table_id,
            section_path=record.section_path,
            embedding_text=text_map.get(record.evidence_id, ""),
            embedding_text_hash=hash_embedding_text(
                text_map.get(record.evidence_id, "")
            ),
        )
        for record in records
    ], manifest


def _to_rrf_format_full(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert candidates to RRF format with full identity metadata."""
    return [
        {
            "doc_id": c.get("candidate_id") or c.get("evidence_id") or "",
            "score": float(c.get("score", 0)),
            "metadata": {
                "doc_name": c.get("document_id", ""),
                "page": c.get("page"),
                "type": c.get("block_type", "text"),
                "parent_id": c.get("parent_id"),
                "table_id": c.get("table_id"),
            },
        }
        for c in candidates
    ]


def _from_rrf_format_full(fused: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert RRF output to candidate format with full identity metadata."""
    result = []
    for rank, item in enumerate(fused):
        meta = item.get("metadata") or {}
        result.append(
            {
                "candidate_id": item.get("doc_id", ""),
                "evidence_id": item.get("doc_id", ""),
                "document_id": meta.get("doc_name", ""),
                "page": meta.get("page"),
                "block_type": meta.get("type", "text"),
                "parent_id": meta.get("parent_id"),
                "table_id": meta.get("table_id"),
                "score": float(item.get("fused_score", item.get("score", 0))),
                "rrf_score": float(item.get("fused_score", item.get("score", 0))),
                "rank": rank,
            }
        )
    return result


def _enrich_dense_candidates(
    candidates: list[dict[str, Any]],
    record_lookup: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add parent_id and table_id to dense candidates from canonical records."""
    for c in candidates:
        eid = c.get("evidence_id", "")
        record = record_lookup.get(eid)
        if record:
            c["parent_id"] = record.parent_id
            c["table_id"] = record.table_id
        else:
            c.setdefault("parent_id", None)
            c.setdefault("table_id", None)
    return candidates


def _normalize_bm25_full(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize BM25 candidates with full identity metadata."""
    normalized = []
    for rank, c in enumerate(candidates):
        meta = c.get("metadata") or {}
        normalized.append(
            {
                "candidate_id": c.get("doc_id", ""),
                "evidence_id": c.get("doc_id", ""),
                "document_id": meta.get("doc_name", ""),
                "page": meta.get("page"),
                "block_type": meta.get("type", "text"),
                "parent_id": meta.get("parent_id"),
                "table_id": meta.get("table_id"),
                "score": float(c.get("score", 0)),
                "rank": rank,
            }
        )
    return normalized


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_bm25_search_fn(bm25_db_path: str):
    from src.services.retrieval import SqliteBM25Retriever

    retriever = SqliteBM25Retriever(db_path=bm25_db_path)

    def search(query: str, *, k: int, user_id: int) -> list[dict[str, Any]]:
        return retriever.search(query, k=k, user_id=user_id)

    return search


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export frozen RRF Top-40 candidate pool for NF39"
    )
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chroma-path", default=None)
    parser.add_argument("--bm25-db-path", required=True)
    parser.add_argument("--bm25-user-id", type=int, required=True)
    parser.add_argument("--expected-case-count", type=int, default=27)
    parser.add_argument("--rrf-top-n", type=int, default=40)
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation cases...")
    cases = load_jsonl_cases(args.cases)
    dataset_validation = validate_labeled_cases(
        cases, expected_count=args.expected_case_count
    )
    print(f"Loaded {dataset_validation['case_count']} cases")

    print("Loading canonical corpus...")
    records, corpus_manifest = _load_canonical_records_with_texts(
        corpus_dir, args.chroma_path
    )
    corpus_hash = corpus_manifest["corpus_hash"]
    evidence_ids_hash = corpus_manifest["evidence_ids_hash"]
    scope = EvaluationScope(
        tenant_id=args.bm25_user_id,
        allowed_document_ids=frozenset(
            str(r.document_id) for r in records
        ),
        expected_case_count=args.expected_case_count,
        expected_corpus_hash=corpus_hash,
        expected_evidence_ids_hash=evidence_ids_hash,
    )
    validate_scope_corpus(scope, records)
    print(
        f"Corpus: {len(records)} records / {len(scope.allowed_document_ids)} docs"
    )

    print("Building MiniLM index...")
    provider = ExistingMiniLMEmbeddingProvider()
    index = build_dense_index(records, provider, corpus_hash, evidence_ids_hash)
    index_fingerprint = index.manifest().index_fingerprint

    record_lookup = {r.evidence_id: r for r in records}

    print("Freezing BM25 pool...")
    frozen_bm25 = freeze_bm25_pool(
        cases,
        _make_bm25_search_fn(args.bm25_db_path),
        scope=scope,
        k=50,
        oversample_k=200,
    )

    print("Computing RRF Top-40 for each case...")
    rrf_pool: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        query_vector = provider.encode_queries([case.question])[0]
        dense_candidates = _enrich_dense_candidates(
            index.search(query_vector, k=max(args.rrf_top_n, 40)),
            record_lookup,
        )
        bm25_candidates = _normalize_bm25_full(
            frozen_bm25.candidates.get(case.case_id, [])
        )
        fused = rrf(
            [
                _to_rrf_format_full(dense_candidates),
                _to_rrf_format_full(bm25_candidates),
            ]
        )
        rrf_top = _from_rrf_format_full(fused)[: args.rrf_top_n]
        rrf_pool[case.case_id] = rrf_top

    print("Writing RRF candidate pool...")
    pool_data = {
        case.case_id: [
            {
                "candidate_id": c["candidate_id"],
                "document_id": c["document_id"],
                "page": c["page"],
                "block_type": c["block_type"],
                "parent_id": c.get("parent_id"),
                "table_id": c.get("table_id"),
                "rrf_rank": c["rank"] + 1,
                "rrf_score": c["rrf_score"],
            }
            for c in candidates
        ]
        for case, candidates in zip(cases, [rrf_pool[c.case_id] for c in cases])
    }
    _write_json(out_dir / "rrf-candidate-pool.json", pool_data)

    candidate_count = sum(len(v) for v in pool_data.values())
    candidate_pool_hash = _stable_digest(
        [
            {"case_id": cid, "candidates": pool_data[cid]}
            for cid in sorted(pool_data)
        ]
    )

    manifest = {
        "candidate_pool_hash": candidate_pool_hash,
        "case_count": len(cases),
        "candidate_count": candidate_count,
        "tenant_id": args.bm25_user_id,
        "allowed_document_ids_hash": frozen_bm25.scope_report()[
            "allowed_document_ids_hash"
        ],
        "rrf_top_n": args.rrf_top_n,
        "corpus_hash": corpus_hash,
        "evidence_ids_hash": evidence_ids_hash,
        "index_fingerprint": index_fingerprint,
        "embedding_provider": provider.name,
        "question_hash": question_fingerprint(cases),
        "label_hash": label_fingerprint(cases),
        "rrf_k": 60,
        "bm25_frozen": True,
    }
    _write_json(out_dir / "rrf-candidate-pool-manifest.json", manifest)

    baseline_manifest = {
        "case_count": len(cases),
        "question_hash": question_fingerprint(cases),
        "label_hash": label_fingerprint(cases),
        "tenant_id": args.bm25_user_id,
        "allowed_document_ids_hash": frozen_bm25.scope_report()[
            "allowed_document_ids_hash"
        ],
        "corpus_hash": corpus_hash,
        "index_fingerprint": index_fingerprint,
        "embedding_provider": provider.name,
        "reranker_provider": "heuristic",
        "rrf_top_n": args.rrf_top_n,
        "reranker_input_top_n": args.rrf_top_n,
        "reranker_output_top_n": 20,
        "final_top_k": 5,
        "dataset_validation": dataset_validation,
        "bm25_frozen": True,
        "rrf_weight": "unchanged",
        "query_expansion": "disabled",
        "bge_reranker": "disabled",
    }
    _write_json(out_dir / "baseline-manifest.json", baseline_manifest)

    print(f"Done. Pool: {candidate_count} candidates across {len(cases)} cases.")
    print(f"Pool hash: {candidate_pool_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
