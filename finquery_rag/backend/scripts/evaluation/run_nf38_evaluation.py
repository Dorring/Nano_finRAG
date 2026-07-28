"""Run the NF38 Embedding A/B evaluation.

This script orchestrates the full A/B comparison:
1. Load canonical corpus and evaluation cases
2. Build (or load) MiniLM and BGE-M3 dense indexes
3. Freeze BM25 candidate pool
4. Run Dense-only diagnostics for each variant
5. Run Hybrid Ranking Gate (Dense + BM25 via RRF) for each variant
6. Evaluate Ranking Gate decision
7. Write all artifacts

Usage:
    python -m scripts.evaluation.run_nf38_evaluation \
        --corpus-dir artifacts/evaluation/nf38 \
        --cases eval_data/phase5/sealed/questions.jsonl \
        --out-dir artifacts/evaluation/nf38 \
        --bge-device cuda:1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from src.evaluation.evaluation import load_jsonl_cases
from src.evaluation.nf38_corpus import hash_embedding_text
from src.evaluation.nf38_dense_index import (
    DenseIndex,
    assert_indexes_share_corpus,
    build_dense_index,
)
from src.evaluation.nf38_evaluator import (
    LatencyReport,
    compute_case_diff,
    compute_token_length_report,
    evaluate_ranking_gate,
    freeze_bm25_pool,
    get_gpu_memory_mb,
    get_process_rss_mb,
    measure_query_latencies,
    run_dense_diagnostic,
    run_hybrid_ranking,
)
from src.retrieval.embedding_provider import (
    BgeM3DenseEmbeddingProvider,
    EmbeddingProvider,
    ExistingMiniLMEmbeddingProvider,
)
from src.services.reranker import build_reranker


def _load_canonical_records_with_texts(
    corpus_dir: Path, chroma_path: str | None = None
) -> tuple[list, dict]:
    """Load canonical records with embedding texts filled in."""
    from scripts.evaluation.build_nf38_embedding_indexes import (
        load_canonical_records,
        load_embedding_texts,
    )

    records, manifest = load_canonical_records(corpus_dir)
    chroma_path = chroma_path or str(BACKEND / "chroma_db")
    text_map = load_embedding_texts(chroma_path, "rag_global_knowledge_base")

    from src.evaluation.nf38_corpus import CanonicalEvidenceRecord

    filled: list[CanonicalEvidenceRecord] = []
    for record in records:
        text = text_map.get(record.evidence_id, "")
        filled.append(
            CanonicalEvidenceRecord(
                evidence_id=record.evidence_id,
                document_id=record.document_id,
                page=record.page,
                block_type=record.block_type,
                parent_id=record.parent_id,
                table_id=record.table_id,
                section_path=record.section_path,
                embedding_text=text,
                embedding_text_hash=hash_embedding_text(text),
            )
        )
    return filled, manifest


def _make_bm25_search_fn(bm25_db_path: str, user_id: int = 9003):
    """Create a BM25 search function that closes over the DB path."""
    from src.services.retrieval import SqliteBM25Retriever

    retriever = SqliteBM25Retriever(db_path=bm25_db_path)

    def search(query: str, k: int = 50, user_id: int = user_id) -> list[dict]:
        return retriever.search(query, k=k, user_id=user_id)

    return search


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _measure_model_cold_start(provider: EmbeddingProvider) -> float:
    """Measure the time to load the model and encode a single text."""
    start = time.monotonic()
    provider.encode_documents(["warmup text"])
    return time.monotonic() - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NF38 Embedding A/B evaluation")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--cases", required=True, help="Path to questions.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chroma-path", default=None)
    parser.add_argument("--bm25-db-path", default=None)
    parser.add_argument("--bge-device", default="cuda:1")
    parser.add_argument("--bge-max-length", type=int, default=512)
    parser.add_argument("--skip-bge", action="store_true")
    parser.add_argument("--reranker", default="heuristic", help="Reranker name (default: heuristic)")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load cases
    print("Loading evaluation cases...")
    cases = load_jsonl_cases(args.cases)
    print(f"Loaded {len(cases)} cases")

    # 2. Load canonical corpus
    print("Loading canonical corpus...")
    records, corpus_manifest = _load_canonical_records_with_texts(corpus_dir, args.chroma_path)
    print(f"Loaded {len(records)} canonical records (corpus_hash={corpus_manifest['corpus_hash'][:12]}...)")

    corpus_hash = corpus_manifest["corpus_hash"]
    evidence_ids_hash = corpus_manifest["evidence_ids_hash"]

    # 3. Write baseline manifest
    baseline_manifest = {
        "corpus_hash": corpus_hash,
        "evidence_ids_hash": evidence_ids_hash,
        "record_count": len(records),
        "cases_count": len(cases),
        "bm25_frozen": True,
        "rrf_weight": "unchanged",
        "reranker": args.reranker,
        "query_expansion": "disabled",
        "bge_reranker": "disabled",
    }
    _write_json(out_dir / "baseline-manifest.json", baseline_manifest)

    # 4. Build MiniLM index + measure latency
    print("Building MiniLM index...")
    minilm_provider = ExistingMiniLMEmbeddingProvider()
    minilm_cold_start = _measure_model_cold_start(minilm_provider)
    minilm_index = build_dense_index(records, minilm_provider, corpus_hash, evidence_ids_hash)
    print(f"MiniLM index built: {len(minilm_index.records)} records, dim={minilm_index.dimension}")

    minilm_manifest = minilm_index.manifest().to_dict()
    _write_json(out_dir / "minilm-index-manifest.json", minilm_manifest)

    # 5. Build BGE-M3 index (or skip)
    bge_provider: BgeM3DenseEmbeddingProvider | None = None
    bge_index: DenseIndex | None = None
    bge_cold_start = 0.0

    if args.skip_bge:
        print("Skipping BGE-M3 index (--skip-bge)")
    else:
        print(f"Building BGE-M3 index (max_length={args.bge_max_length}, device={args.bge_device})...")
        bge_provider = BgeM3DenseEmbeddingProvider(
            device=args.bge_device,
            max_length=args.bge_max_length,
        )
        bge_cold_start = _measure_model_cold_start(bge_provider)
        bge_index = build_dense_index(records, bge_provider, corpus_hash, evidence_ids_hash)
        print(f"BGE-M3 index built: {len(bge_index.records)} records, dim={bge_index.dimension}")

        # Verify isolation
        assert_indexes_share_corpus(minilm_index, bge_index)
        print("Corpus isolation verified.")

        bge_manifest = bge_index.manifest().to_dict()
        _write_json(out_dir / "bge-m3-index-manifest.json", bge_manifest)

        # Token length report (Section 10)
        print("Computing token-length report...")
        token_report = compute_token_length_report(records, bge_provider)
        token_report["provider"] = bge_provider.name
        token_report["selected_max_length"] = args.bge_max_length
        _write_json(out_dir / "token-length-report.json", token_report)
        print(f"Token lengths: p50={token_report['p50']}, p95={token_report['p95']}, "
              f"truncated@512={token_report['truncated_count_at_512']}")

    # 6. Freeze BM25 pool
    print("Freezing BM25 candidate pool...")
    bm25_db_path = args.bm25_db_path or str(BACKEND / "indexes/phase5/sealed/rag_bm25.db")
    bm25_search = _make_bm25_search_fn(bm25_db_path)
    bm25_pool = freeze_bm25_pool(cases, bm25_search, k=50)
    print(f"BM25 pool frozen: {len(bm25_pool)} cases")

    # Save BM25 pool manifest (not the full pool — privacy)
    bm25_manifest = {
        "case_count": len(bm25_pool),
        "k": 50,
        "source": "SqliteBM25Retriever",
        "db_path_hash": hashlib.sha256(bm25_db_path.encode()).hexdigest()[:16],
    }
    _write_json(out_dir / "bm25-candidate-pool-manifest.json", bm25_manifest)

    # 7. Dense-only diagnostics
    print("\n=== Dense-only Diagnostics ===")
    minilm_dense = run_dense_diagnostic(cases, minilm_index, minilm_provider)
    _write_json(out_dir / "dense-comparison.json", {"minilm": minilm_dense})
    print(f"MiniLM Dense: CaseHit@5={minilm_dense['case_hit_rate_at_5']:.3f}, "
          f"Recall@40={minilm_dense['source_recall_at_40']:.3f}, MRR={minilm_dense['mrr']:.3f}")

    bge_dense: dict[str, Any] | None = None
    if bge_index is not None and bge_provider is not None:
        bge_dense = run_dense_diagnostic(cases, bge_index, bge_provider)
        dense_comparison = {"minilm": minilm_dense, "bge_m3": bge_dense}
        _write_json(out_dir / "dense-comparison.json", dense_comparison)
        print(f"BGE-M3 Dense:  CaseHit@5={bge_dense['case_hit_rate_at_5']:.3f}, "
              f"Recall@40={bge_dense['source_recall_at_40']:.3f}, MRR={bge_dense['mrr']:.3f}")

    # 8. Hybrid Ranking Gate
    print("\n=== Hybrid Ranking Gate ===")
    reranker = build_reranker(args.reranker)

    minilm_hybrid = run_hybrid_ranking(cases, minilm_index, minilm_provider, bm25_pool, reranker)
    print(f"MiniLM Hybrid: Final CaseHit@5={minilm_hybrid['final']['case_hit_rate_at_5']:.3f}, "
          f"MRR={minilm_hybrid['final']['mrr']:.3f}")

    bge_hybrid: dict[str, Any] | None = None
    if bge_index is not None and bge_provider is not None:
        bge_hybrid = run_hybrid_ranking(cases, bge_index, bge_provider, bm25_pool, reranker)
        hybrid_comparison = {"minilm": minilm_hybrid, "bge_m3": bge_hybrid}
        _write_json(out_dir / "hybrid-ranking-comparison.json", hybrid_comparison)
        print(f"BGE-M3 Hybrid:  Final CaseHit@5={bge_hybrid['final']['case_hit_rate_at_5']:.3f}, "
              f"MRR={bge_hybrid['final']['mrr']:.3f}")

        # 9. Ranking Gate decision
        print("\n=== Ranking Gate ===")
        gate_result = evaluate_ranking_gate(minilm_hybrid, bge_hybrid)
        gate_result["dense_minilm"] = minilm_dense
        gate_result["dense_bge_m3"] = bge_dense
        _write_json(out_dir / "nf38-acceptance.json", gate_result)
        print(f"Ranking Gate: {'PASSED' if gate_result['passed'] else 'NOT PASSED'}")
        print(f"  Candidate improved: {gate_result['candidate_improved']}")
        print(f"  No regression: {gate_result['no_regression']}")
        print(f"  Final improved: {gate_result['final_improved']}")

        # 10. Case diff report
        baseline_rankings = {
            case.case_id: minilm_index.search(minilm_provider.encode_queries([case.question])[0], k=50)
            for case in cases
        }
        variant_rankings = {
            case.case_id: bge_index.search(bge_provider.encode_queries([case.question])[0], k=50)
            for case in cases
        }
        case_diff = compute_case_diff(cases, baseline_rankings, variant_rankings)
        _write_json(out_dir / "case-diff-report.json", {"cases": case_diff})

        improved = sum(1 for d in case_diff if d["improved"])
        regressed = sum(1 for d in case_diff if d["regressed"])
        print(f"\nCase diff: {improved} improved, {regressed} regressed")

    # 11. Latency and resource report
    print("\n=== Latency & Resource Report ===")
    minilm_query_lat = measure_query_latencies(
        minilm_provider, [cases[0].question] if cases else []
    )

    minilm_latency = LatencyReport(
        model_cold_start_seconds=minilm_cold_start,
        index_build_seconds=minilm_index.build_time_seconds,
        chunks_encoded=len(records),
        chunks_per_second=len(records) / max(minilm_index.build_time_seconds, 0.001),
        query_embedding_p50_ms=minilm_query_lat["p50_ms"],
        query_embedding_p95_ms=minilm_query_lat["p95_ms"],
        dense_search_p50_ms=minilm_dense.get("query_latencies_p50", 0) * 1000,
        dense_search_p95_ms=minilm_dense.get("query_latencies_p95", 0) * 1000,
        gpu_memory_mb=get_gpu_memory_mb(),
        process_rss_mb=get_process_rss_mb(),
        index_disk_bytes=minilm_index.storage_bytes,
    )

    latency_report: dict[str, Any] = {"minilm": minilm_latency.to_dict()}

    if bge_provider is not None and bge_index is not None and bge_dense is not None:
        bge_query_lat = measure_query_latencies(
            bge_provider, [cases[0].question] if cases else []
        )
        bge_latency = LatencyReport(
            model_cold_start_seconds=bge_cold_start,
            index_build_seconds=bge_index.build_time_seconds,
            chunks_encoded=len(records),
            chunks_per_second=len(records) / max(bge_index.build_time_seconds, 0.001),
            query_embedding_p50_ms=bge_query_lat["p50_ms"],
            query_embedding_p95_ms=bge_query_lat["p95_ms"],
            dense_search_p50_ms=bge_dense.get("query_latencies_p50", 0) * 1000,
            dense_search_p95_ms=bge_dense.get("query_latencies_p95", 0) * 1000,
            gpu_memory_mb=get_gpu_memory_mb(),
            process_rss_mb=get_process_rss_mb(),
            index_disk_bytes=bge_index.storage_bytes,
        )
        latency_report["bge_m3"] = bge_latency.to_dict()

    _write_json(out_dir / "latency-resource-report.json", latency_report)
    print(f"MiniLM cold start: {minilm_cold_start:.2f}s, query P50: {minilm_query_lat['p50_ms']:.1f}ms")
    if bge_provider is not None:
        print(f"BGE-M3 cold start: {bge_cold_start:.2f}s, query P50: {bge_query_lat['p50_ms']:.1f}ms")

    print("\nDone. Artifacts written to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
