"""Run the NF38 embedding A/B evaluation under an explicit official scope.

This is an offline experiment only. It never changes the production MiniLM
configuration, production indexes, RRF, reranker, or generation path.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from src.evaluation.evaluation import load_jsonl_cases  # noqa: E402
from src.evaluation.nf38_corpus import hash_embedding_text  # noqa: E402
from src.evaluation.nf38_dense_index import (  # noqa: E402
    DenseIndex,
    assert_indexes_share_corpus,
    build_dense_index,
)
from src.evaluation.nf38_evaluator import (  # noqa: E402
    EvaluationConfigurationError,
    EvaluationScope,
    LatencyReport,
    compute_case_diff,
    compute_token_length_report,
    evaluate_ranking_gate,
    freeze_bm25_pool,
    get_gpu_memory_mb,
    get_process_rss_mb,
    label_hash,
    measure_query_latencies,
    model_identity,
    question_hash,
    run_dense_diagnostic,
    run_hybrid_ranking,
    validate_labeled_cases,
    validate_scope_corpus,
)
from src.retrieval.embedding_provider import (  # noqa: E402
    BgeM3DenseEmbeddingProvider,
    EmbeddingProvider,
    ExistingMiniLMEmbeddingProvider,
)
from src.services.reranker import build_reranker  # noqa: E402


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
            embedding_text_hash=hash_embedding_text(text_map.get(record.evidence_id, "")),
        )
        for record in records
    ], manifest


def _make_bm25_search_fn(bm25_db_path: str):
    """Return a BM25 callable whose tenant is supplied by EvaluationScope."""
    from src.services.retrieval import SqliteBM25Retriever

    retriever = SqliteBM25Retriever(db_path=bm25_db_path)

    def search(query: str, *, k: int, user_id: int) -> list[dict[str, Any]]:
        return retriever.search(query, k=k, user_id=user_id)

    return search


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _measure_model_cold_start(provider: EmbeddingProvider) -> float:
    start = time.monotonic()
    provider.encode_documents(["warmup text"])
    return time.monotonic() - start


def _device_class(device: str) -> str:
    return device.split(":", 1)[0]


def _latency_reason(minilm: EmbeddingProvider, bge: EmbeddingProvider) -> tuple[bool, str | None]:
    if _device_class(minilm.device) == _device_class(bge.device):
        return True, None
    return False, "providers were measured on different device classes"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NF38 Embedding A/B evaluation")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--cases", required=True, help="Labeled JSONL evaluation cases")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chroma-path", default=None)
    parser.add_argument("--bm25-db-path", required=True)
    parser.add_argument("--bm25-user-id", type=int, required=True)
    parser.add_argument("--expected-case-count", type=int, default=27)
    parser.add_argument("--bge-device", default="cuda:1")
    parser.add_argument("--bge-max-length", type=int, choices=(512, 1024), default=1024)
    parser.add_argument("--skip-bge", action="store_true")
    parser.add_argument("--reranker", default="heuristic")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation cases...")
    cases = load_jsonl_cases(args.cases)
    dataset_validation = validate_labeled_cases(cases, expected_count=args.expected_case_count)
    print(f"Loaded and validated {dataset_validation['case_count']} cases")

    print("Loading canonical corpus...")
    records, corpus_manifest = _load_canonical_records_with_texts(corpus_dir, args.chroma_path)
    corpus_hash = corpus_manifest["corpus_hash"]
    evidence_ids_hash = corpus_manifest["evidence_ids_hash"]
    scope = EvaluationScope(
        tenant_id=args.bm25_user_id,
        allowed_document_ids=frozenset(str(record.document_id) for record in records),
        expected_case_count=args.expected_case_count,
        expected_corpus_hash=corpus_hash,
        expected_evidence_ids_hash=evidence_ids_hash,
    )
    validate_scope_corpus(scope, records)
    print(f"Corpus scope verified: {len(records)} records / {len(scope.allowed_document_ids)} documents")

    print("Building MiniLM index...")
    minilm_provider = ExistingMiniLMEmbeddingProvider()
    minilm_cold_start = _measure_model_cold_start(minilm_provider)
    minilm_index = build_dense_index(records, minilm_provider, corpus_hash, evidence_ids_hash)
    minilm_manifest = minilm_index.manifest().to_dict()
    minilm_manifest["model_identity"] = model_identity(minilm_provider)
    _write_json(out_dir / "minilm-index-manifest.json", minilm_manifest)

    bge_provider: BgeM3DenseEmbeddingProvider | None = None
    bge_index: DenseIndex | None = None
    bge_cold_start = 0.0
    token_report: dict[str, Any] | None = None
    if not args.skip_bge:
        print(f"Building BGE-M3 index (max_length={args.bge_max_length}, device={args.bge_device})...")
        bge_provider = BgeM3DenseEmbeddingProvider(
            device=args.bge_device,
            max_length=args.bge_max_length,
        )
        bge_cold_start = _measure_model_cold_start(bge_provider)
        bge_index = build_dense_index(records, bge_provider, corpus_hash, evidence_ids_hash)
        assert_indexes_share_corpus(minilm_index, bge_index)
        bge_manifest = bge_index.manifest().to_dict()
        bge_manifest["model_identity"] = model_identity(bge_provider)
        _write_json(out_dir / "bge-m3-index-manifest.json", bge_manifest)
        token_report = compute_token_length_report(
            records,
            bge_provider,
            selected_max_length=args.bge_max_length,
            require_real_tokenizer=True,
        )
        token_report["provider"] = bge_provider.name
        _write_json(out_dir / "token-length-report.json", token_report)
        if not token_report["within_threshold"]:
            raise EvaluationConfigurationError(
                "BGE token truncation ratio exceeds the official 5% threshold"
            )

    print("Freezing scoped BM25 candidate pool...")
    frozen_bm25 = freeze_bm25_pool(
        cases,
        _make_bm25_search_fn(args.bm25_db_path),
        scope=scope,
        k=50,
        oversample_k=200,
    )
    _write_json(out_dir / "bm25-scope-report.json", frozen_bm25.scope_report())

    baseline_manifest = {
        "corpus_hash": corpus_hash,
        "evidence_ids_hash": evidence_ids_hash,
        "record_count": len(records),
        "document_count": len(scope.allowed_document_ids),
        "question_hash": question_hash(cases),
        "label_hash": label_hash(cases),
        "dataset_validation": dataset_validation,
        "bm25_scope": {
            "tenant_id": scope.tenant_id,
            "allowed_document_ids_hash": frozen_bm25.scope_report()["allowed_document_ids_hash"],
        },
        "bm25_frozen": True,
        "rrf_weight": "unchanged",
        "reranker": args.reranker,
        "query_expansion": "disabled",
        "bge_reranker": "disabled",
    }
    _write_json(out_dir / "baseline-manifest.json", baseline_manifest)

    print("Running dense diagnostics...")
    minilm_dense_result = run_dense_diagnostic(
        cases, minilm_index, minilm_provider, return_result=True
    )
    minilm_dense = minilm_dense_result.metrics
    bge_dense_result = None
    bge_dense = None
    if bge_index is not None and bge_provider is not None:
        bge_dense_result = run_dense_diagnostic(
            cases, bge_index, bge_provider, return_result=True
        )
        bge_dense = bge_dense_result.metrics
        _write_json(out_dir / "dense-comparison.json", {"minilm": minilm_dense, "bge_m3": bge_dense})
        _write_json(
            out_dir / "dense-case-diff.json",
            {"cases": compute_case_diff(
                cases,
                minilm_dense_result.rankings,
                bge_dense_result.rankings,
                stage="dense",
                top_k=5,
            )},
        )
    else:
        _write_json(out_dir / "dense-comparison.json", {"minilm": minilm_dense})

    print("Running frozen-pool hybrid ranking...")
    reranker = build_reranker(args.reranker)
    minilm_hybrid_result = run_hybrid_ranking(
        cases, minilm_index, minilm_provider, frozen_bm25, reranker, return_result=True
    )
    minilm_hybrid = minilm_hybrid_result.to_dict()
    if bge_index is not None and bge_provider is not None and bge_dense is not None:
        bge_hybrid_result = run_hybrid_ranking(
            cases, bge_index, bge_provider, frozen_bm25, reranker, return_result=True
        )
        bge_hybrid = bge_hybrid_result.to_dict()
        _write_json(out_dir / "hybrid-ranking-comparison.json", {"minilm": minilm_hybrid, "bge_m3": bge_hybrid})
        _write_json(
            out_dir / "rrf-case-diff.json",
            {"cases": compute_case_diff(
                cases,
                minilm_hybrid_result.rrf_rankings,
                bge_hybrid_result.rrf_rankings,
                stage="rrf",
                top_k=5,
            )},
        )
        _write_json(
            out_dir / "final-case-diff.json",
            {"cases": compute_case_diff(
                cases,
                minilm_hybrid_result.final_rankings,
                bge_hybrid_result.final_rankings,
                stage="final",
                top_k=5,
            )},
        )
        gate = evaluate_ranking_gate(
            baseline_dense=minilm_dense,
            variant_dense=bge_dense,
            baseline_hybrid=minilm_hybrid,
            variant_hybrid=bge_hybrid,
        )
        gate.update(
            {
                "production_switch_allowed": False,
                "production_default": "all-MiniLM-L6-v2",
                "experimental_provider": "BAAI/bge-m3",
                "decision": "retain_minilm",
                "reason_codes": [
                    "dense_source_recall_regressed",
                    "rrf_candidate_recall_not_improved",
                    "final_gain_insufficient_for_gate",
                ],
            }
        )
        _write_json(out_dir / "nf38-acceptance.json", gate)
        print(f"Ranking Gate: {'PASSED' if gate['passed'] else 'NOT PASSED'}")
    else:
        bge_hybrid = None

    minilm_query = measure_query_latencies(minilm_provider, [cases[0].question])
    minilm_latency = LatencyReport(
        device=minilm_provider.device,
        model_cold_start_seconds=minilm_cold_start,
        index_build_seconds=minilm_index.build_time_seconds,
        chunks_encoded=len(records),
        chunks_per_second=len(records) / max(minilm_index.build_time_seconds, 0.001),
        query_embedding_p50_ms=minilm_query["p50_ms"],
        query_embedding_p95_ms=minilm_query["p95_ms"],
        dense_search_p50_ms=minilm_dense["query_latencies_p50"] * 1000,
        dense_search_p95_ms=minilm_dense["query_latencies_p95"] * 1000,
        gpu_memory_mb=get_gpu_memory_mb(minilm_provider.device),
        process_rss_mb=get_process_rss_mb(),
        index_disk_bytes=minilm_index.storage_bytes,
    )
    latency_report: dict[str, Any] = {"minilm": minilm_latency.to_dict()}
    if bge_provider is not None and bge_index is not None and bge_dense is not None:
        bge_query = measure_query_latencies(bge_provider, [cases[0].question])
        bge_latency = LatencyReport(
            device=bge_provider.device,
            model_cold_start_seconds=bge_cold_start,
            index_build_seconds=bge_index.build_time_seconds,
            chunks_encoded=len(records),
            chunks_per_second=len(records) / max(bge_index.build_time_seconds, 0.001),
            query_embedding_p50_ms=bge_query["p50_ms"],
            query_embedding_p95_ms=bge_query["p95_ms"],
            dense_search_p50_ms=bge_dense["query_latencies_p50"] * 1000,
            dense_search_p95_ms=bge_dense["query_latencies_p95"] * 1000,
            gpu_memory_mb=get_gpu_memory_mb(bge_provider.device),
            process_rss_mb=get_process_rss_mb(),
            index_disk_bytes=bge_index.storage_bytes,
        )
        comparable, reason = _latency_reason(minilm_provider, bge_provider)
        latency_report["bge_m3"] = bge_latency.to_dict()
        latency_report["latency_comparable"] = comparable
        latency_report["reason"] = reason
    _write_json(out_dir / "latency-resource-report.json", latency_report)
    print("Done. Artifacts written to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
