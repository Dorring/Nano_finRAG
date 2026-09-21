"""NF-OPT-16 Gate B: fixed BGE-M3 neural sparse Shadow retrieval evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.nf_opt_16 import build_sparse_inverted_index, sparse_rank
from src.retrieval.candidate_fusion import boost_front_matter_chunks, normalize_scores, rrf
from src.retrieval.candidate_identity import candidate_key, identity_from_candidate
from src.retrieval.query_processor import QueryProcessor
from src.services.reranker import HeuristicReranker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "artifacts/evaluation/nf-eval-03-r2/case-results.json"
DEFAULT_CORPUS = ROOT / "benchmarks/financial_rag_v1/corpus.json"
DEFAULT_QUESTIONS = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"
DEFAULT_LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
DEFAULT_OUT = ROOT / "artifacts/evaluation/nf-opt-16-gate-b"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _candidate_from_row(
    *, doc_id: str, content: str, metadata: dict[str, Any], document: dict[str, Any], doc_name: str
) -> dict[str, Any]:
    evidence_type = str(metadata.get("type") or "text")
    raw = {
        "tenant_id": metadata.get("user_id"),
        "document_id": document["document_id"],
        "block_type": evidence_type,
        "evidence_id": doc_id,
        "doc_id": doc_id,
        "metadata": metadata,
    }
    return {
        "candidate_key": candidate_key(identity_from_candidate(raw)),
        "canonical_document_id": document["document_id"],
        "doc_id": doc_id,
        "evidence_id": doc_id,
        "content": content,
        "metadata": metadata,
        "filename": doc_name,
        "page": metadata.get("page"),
        "type": evidence_type,
        "block_type": evidence_type,
        "parent_id": metadata.get("parent_id"),
        "parent_candidate_key": metadata.get("parent_id"),
    }


def _load_candidates(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], int]:
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    documents = {str(item["filename"]): item for item in corpus["documents"]}
    connection = sqlite3.connect(f"file:{args.candidate_db}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT doc_id, content, metadata_json, doc_name FROM chunk_store").fetchall()
    finally:
        connection.close()
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc_id, content, metadata_json, doc_name in rows:
        document = documents.get(str(doc_name))
        if document is None or not str(content or "").strip():
            continue
        metadata = json.loads(metadata_json or "{}")
        by_document[str(document["document_id"])].append(
            _candidate_from_row(
                doc_id=str(doc_id), content=str(content), metadata=metadata, document=document, doc_name=str(doc_name)
            )
        )
    return by_document, len(rows)


def _source_indices(selected: list[dict[str, Any]], sources: list[dict[str, Any]]) -> set[int]:
    selected_keys = {str(item["candidate_key"]) for item in selected}
    return {index for index, source in enumerate(sources) if str(source["candidate_key"]) in selected_keys}


def _coverage(selected: list[dict[str, Any]], sources: list[dict[str, Any]]) -> str:
    matched = _source_indices(selected, sources)
    if len(matched) == len(sources):
        return "all"
    return "partial" if matched else "none"


def _encode_corpus(model: Any, by_document: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> tuple[dict[str, dict[str, list[tuple[int, float]]]], dict[str, int], float]:
    indexes: dict[str, dict[str, list[tuple[int, float]]]] = {}
    token_counts: dict[str, int] = {}
    started = time.perf_counter()
    for document_id, candidates in sorted(by_document.items()):
        output = model.encode(
            [str(item["content"]) for item in candidates],
            batch_size=args.batch_size,
            max_length=args.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        lexical_rows = output.get("lexical_weights")
        if not isinstance(lexical_rows, list) or len(lexical_rows) != len(candidates):
            raise ValueError(f"sparse output mismatch for {document_id}")
        indexes[document_id] = build_sparse_inverted_index(lexical_rows)
        token_counts[document_id] = sum(len(row) for row in lexical_rows)
    return indexes, token_counts, (time.perf_counter() - started) * 1000.0


def run(args: argparse.Namespace) -> int:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("NF-OPT-16 requires offline model environment")
    if not args.model_path.exists():
        raise FileNotFoundError(f"missing local BGE-M3 snapshot: {args.model_path}")
    trace = json.loads(args.cases.read_text(encoding="utf-8"))
    questions = {str(item["case_id"]): item for item in _jsonl(args.questions)}
    labels = {str(item["case_id"]): item for item in _jsonl(args.labels)}
    answerable = [item for item in trace["cases"] if not item.get("expected_no_answer")]
    no_answer = [item for item in trace["cases"] if item.get("expected_no_answer")]
    if len(answerable) != 64 or len(no_answer) != 8:
        raise ValueError("expected frozen 64 answerable and 8 no-answer cases")
    by_document, candidate_store_rows = _load_candidates(args)
    candidates_by_doc_id = {
        str(candidate["doc_id"]): candidate
        for candidates in by_document.values()
        for candidate in candidates
    }

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(str(args.model_path), use_fp16=True, devices=args.device, batch_size=args.batch_size)
    indexes, token_counts, index_build_ms = _encode_corpus(model, by_document, args)
    reranker = HeuristicReranker()
    query_processor = QueryProcessor()
    records: list[dict[str, Any]] = []
    sparse_latencies: list[float] = []
    baseline_hits: set[tuple[str, int]] = set()
    sparse20_hits: set[tuple[str, int]] = set()
    sparse40_hits: set[tuple[str, int]] = set()
    sparse200_hits: set[tuple[str, int]] = set()
    rrf_hits: set[tuple[str, int]] = set()
    final_hits: set[tuple[str, int]] = set()
    final_all: set[str] = set()
    baseline_final_hits: set[tuple[str, int]] = set()
    baseline_control_hits: set[tuple[str, int]] = set()

    for case in answerable:
        case_id = str(case["case_id"])
        question = questions[case_id]
        scope = [str(value) for value in question.get("document_scope") or ()]
        if len(scope) != 1 or scope[0] not in by_document:
            raise ValueError(f"unsupported or missing document scope for {case_id}: {scope}")
        document_id = scope[0]
        candidates = by_document[document_id]
        raw_query = str(question["question"])
        retrieval_query = query_processor.expand(raw_query)
        encoded_query = model.encode(
            [retrieval_query],
            batch_size=1,
            max_length=args.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        query_weights = encoded_query["lexical_weights"][0]
        started = time.perf_counter()
        ranked = sparse_rank(
            query_weights=query_weights,
            inverted_index=indexes[document_id],
            candidate_keys=[str(item["candidate_key"]) for item in candidates],
            limit=200,
        )
        sparse_latencies.append((time.perf_counter() - started) * 1000.0)
        sparse = []
        for rank, (index, score) in enumerate(ranked, start=1):
            item = dict(candidates[index])
            item["score"] = score
            item["stage_rank"] = rank
            item["score_kind"] = "bge_m3_sparse"
            sparse.append(item)
        dense = []
        for trace_candidate in case["retrieval_stages"]["dense"]:
            source_candidate = candidates_by_doc_id.get(str(trace_candidate["doc_id"]))
            if source_candidate is None:
                raise ValueError(f"missing frozen Dense candidate content: {trace_candidate['doc_id']}")
            if str(source_candidate["candidate_key"]) != str(trace_candidate["candidate_key"]):
                raise ValueError(f"Dense candidate identity drift: {trace_candidate['doc_id']}")
            dense_candidate = dict(source_candidate)
            dense_candidate.update(trace_candidate)
            dense.append(dense_candidate)
        fused = normalize_scores(rrf([dense, sparse[:20]], k=60))
        fused = boost_front_matter_chunks(
            raw_query,
            fused,
            is_front_matter_query_fn=query_processor.is_front_matter_query,
        )
        reranked = reranker.rerank(raw_query, fused, top_k=min(20, len(fused)))
        final = reranker.rerank(raw_query, fused, top_k=5)
        sources = list(labels[case_id].get("expected_sources") or [])
        baseline = list(case["retrieval_stages"]["final"])
        frozen_rrf = []
        for trace_candidate in case["retrieval_stages"]["rrf"]:
            source_candidate = candidates_by_doc_id.get(str(trace_candidate["doc_id"]))
            if source_candidate is None:
                raise ValueError(f"missing frozen RRF candidate content: {trace_candidate['doc_id']}")
            frozen_candidate = dict(source_candidate)
            frozen_candidate.update(trace_candidate)
            frozen_rrf.append(frozen_candidate)
        baseline_control = reranker.rerank(raw_query, frozen_rrf, top_k=5)
        baseline_case_hits = _source_indices(baseline, sources)
        sparse20_case_hits = _source_indices(sparse[:20], sources)
        sparse40_case_hits = _source_indices(sparse[:40], sources)
        sparse200_case_hits = _source_indices(sparse[:200], sources)
        rrf_case_hits = _source_indices(fused[:40], sources)
        final_case_hits = _source_indices(final, sources)
        baseline_hits.update((case_id, item) for item in baseline_case_hits)
        baseline_final_hits.update((case_id, item) for item in baseline_case_hits)
        baseline_control_hits.update((case_id, item) for item in _source_indices(baseline_control, sources))
        sparse20_hits.update((case_id, item) for item in sparse20_case_hits)
        sparse40_hits.update((case_id, item) for item in sparse40_case_hits)
        sparse200_hits.update((case_id, item) for item in sparse200_case_hits)
        rrf_hits.update((case_id, item) for item in rrf_case_hits)
        final_hits.update((case_id, item) for item in final_case_hits)
        if _coverage(final, sources) == "all":
            final_all.add(case_id)
        records.append(
            {
                "case_id": case_id,
                "document_scope": scope,
                "sparse_candidate_count": len(sparse),
                "sparse_top20_candidate_keys": [str(item["candidate_key"]) for item in sparse[:20]],
                "dense_candidate_keys": [str(item["candidate_key"]) for item in dense],
                "rrf_candidate_keys": [str(item["candidate_key"]) for item in fused],
                "reranker_top20_candidate_keys": [str(item["candidate_key"]) for item in reranked],
                "final_candidate_keys": [str(item["candidate_key"]) for item in final],
                "baseline_final_matched_source_count": len(baseline_case_hits),
                "baseline_control_matched_source_count": len(_source_indices(baseline_control, sources)),
                "sparse20_matched_source_count": len(sparse20_case_hits),
                "sparse40_matched_source_count": len(sparse40_case_hits),
                "sparse200_matched_source_count": len(sparse200_case_hits),
                "rrf40_matched_source_count": len(rrf_case_hits),
                "final_matched_source_count": len(final_case_hits),
                "final_gold_coverage": _coverage(final, sources),
            }
        )

    no_answer_records = [
        {"case_id": str(case["case_id"]), "final_behavior": "baseline_unchanged"}
        for case in no_answer
    ]
    source_count = sum(len(labels[str(case["case_id"])].get("expected_sources") or []) for case in answerable)
    if source_count != 80:
        raise ValueError(f"expected 80 source instances, got {source_count}")
    regressions = baseline_final_hits - final_hits
    new_hits = final_hits - baseline_final_hits
    baseline_all = {
        str(case["case_id"])
        for case in answerable
        if _coverage(list(case["retrieval_stages"]["final"]), list(labels[str(case["case_id"])].get("expected_sources") or [])) == "all"
    }
    all_regressions = baseline_all - final_all
    baseline_control_matches = baseline_control_hits == baseline_final_hits
    if not baseline_control_matches:
        raise ValueError("frozen RRF reranker control did not reproduce baseline strict hits")
    metrics = {
        "baseline_final_source_recall_at_5": {"matched_sources": len(baseline_hits), "source_count": source_count},
        "bge_m3_sparse_source_recall_at_20": {"matched_sources": len(sparse20_hits), "source_count": source_count},
        "bge_m3_sparse_source_recall_at_40": {"matched_sources": len(sparse40_hits), "source_count": source_count},
        "bge_m3_sparse_source_recall_at_200": {"matched_sources": len(sparse200_hits), "source_count": source_count},
        "hybrid_rrf_source_recall_at_40": {"matched_sources": len(rrf_hits), "source_count": source_count},
        "shadow_final_source_recall_at_5": {"matched_sources": len(final_hits), "source_count": source_count},
        "baseline_final_all_gold_case_count": len(baseline_all),
        "shadow_final_all_gold_case_count": len(final_all),
        "new_final_hits": len(new_hits),
        "regressed_final_hits": len(regressions),
        "regressed_final_all_gold_cases": len(all_regressions),
        "baseline_control_source_recall_at_5": {"matched_sources": len(baseline_control_hits), "source_count": source_count},
    }
    final_gain = len(final_hits) - len(baseline_hits)
    all_gain = len(final_all) - len(baseline_all)
    if final_gain >= 4 and all_gain >= 2 and not regressions and not all_regressions:
        decision = "bge_m3_sparse_shadow_validated"
        next_gate = "production_candidate_shadow_ab"
    elif len(sparse200_hits) > len(sparse20_hits) and final_gain < 4:
        decision = "bge_m3_sparse_transfer_failed_multivector_diagnostic_allowed"
        next_gate = "nf-opt-16-gate-c-frozen-multivector-shadow"
    else:
        decision = "bge_m3_sparse_shadow_gain_insufficient"
        next_gate = "stop_neural_sparse_before_multi_vector"
    configuration = {
        "retriever": "BAAI/bge-m3 lexical_weights only",
        "dense_vectors_used": False,
        "colbert_vectors_used": False,
        "scope": "frozen question document_scope",
        "candidate_corpus": "existing chunk_store read-only, eight frozen documents",
        "passage_max_length": args.max_length,
        "batch_size": args.batch_size,
        "sparse_score": "sum(query_lexical_weight * candidate_lexical_weight)",
        "sparse_top_k": 200,
        "hybrid": "current Dense Top-20 + BGE-M3 Sparse Top-20, equal RRF k=60",
        "reranker": "current heuristic reranker, RRF full ordered list",
        "final_top_k": 5,
        "configuration_frozen_before_benchmark_run": True,
    }
    acceptance = {
        "artifact_schema": "nf-opt-16/gate-b/acceptance/v1",
        "baseline_gate_a_commit": "d72794e",
        "question_count": 72,
        "answerable_case_count": len(answerable),
        "no_answer_case_count": len(no_answer),
        "gold_source_count": source_count,
        "input_hashes": {"cases_sha256": _sha(args.cases), "corpus_sha256": _sha(args.corpus), "questions_sha256": _sha(args.questions), "labels_sha256": _sha(args.labels)},
        "candidate_store_access_mode": "sqlite_read_only",
        "gold_used_only_for_posthoc_scoring": True,
        "dense_behavior_modified": False,
        "rrf_parameters_modified": False,
        "reranker_behavior_modified": False,
        "final_top_k_modified": False,
        "embedding_model_network_calls": 0,
        "embedding_model_calls": len(by_document) + len(answerable),
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "no_answer_final_behavior_unchanged": True,
        "frozen_rrf_reranker_control_matches_baseline": baseline_control_matches,
        "decision": decision,
        "next_gate": next_gate,
    }
    _write(args.out_dir / "neural-sparse-configuration.json", configuration)
    _write(args.out_dir / "sparse-index-manifest.json", {
        "candidate_store_rows_total": candidate_store_rows,
        "scope_candidate_count": sum(len(value) for value in by_document.values()),
        "document_count": len(by_document),
        "lexical_weight_count_by_document": token_counts,
        "index_build_ms": index_build_ms,
        "index_persisted": False,
        "production_index_written": False,
    })
    _write(args.out_dir / "bge-m3-sparse-stage-results.json", {"metrics": metrics, "cases": records})
    _write(args.out_dir / "hybrid-transfer-results.json", {"reranker_input_source": "rrf_all", "final_top_k": 5, "metrics": metrics, "no_answer_cases": no_answer_records})
    _write(args.out_dir / "strict-hit-regression-report.json", {"new_source_instances": sorted(new_hits), "regressed_source_instances": sorted(regressions), "regressed_all_gold_cases": sorted(all_regressions)})
    _write(args.out_dir / "latency-and-resource-report.json", {
        "sparse_query_ms": {"mean": sum(sparse_latencies) / len(sparse_latencies), "p50": _percentile(sparse_latencies, 0.5), "p95": _percentile(sparse_latencies, 0.95)},
        "index_build_ms": index_build_ms,
        "online_multivector_used": False,
        "online_pdf_parser_used": False,
    })
    _write(args.out_dir / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    _write(args.out_dir / "nf-opt-16-gate-b-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
