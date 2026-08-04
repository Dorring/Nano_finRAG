"""NF-OPT-16 Gate C: fixed BGE-M3 Sparse-to-late-interaction Shadow retrieval."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from src.evaluation.nf_opt_16 import rank_scores, sparse_rank
from src.retrieval.candidate_fusion import boost_front_matter_chunks, normalize_scores, rrf
from src.retrieval.query_processor import QueryProcessor
from src.services.reranker import HeuristicReranker

from scripts.evaluation.run_nf_opt_16_gate_b import (
    DEFAULT_CASES,
    DEFAULT_CORPUS,
    DEFAULT_LABELS,
    DEFAULT_QUESTIONS,
    ROOT,
    _coverage,
    _encode_corpus,
    _jsonl,
    _load_candidates,
    _percentile,
    _sha,
    _source_indices,
    _write,
)

DEFAULT_OUT = ROOT / "artifacts/evaluation/nf-opt-16-gate-c"


def _late_interaction_scores(query_vectors: Any, passage_vectors: list[Any], *, device: str, batch_size: int) -> list[float]:
    """Score only Sparse Top-200 candidates with GPU late interaction.

    Cell geometry, Gold values, and candidate labels are never inputs here.
    """
    import torch

    query = torch.as_tensor(query_vectors, device=device, dtype=torch.float32)
    values: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(passage_vectors), batch_size):
            batch = passage_vectors[start:start + batch_size]
            lengths = [int(value.shape[0]) for value in batch]
            if not all(length > 0 for length in lengths):
                raise ValueError("late-interaction passage contains zero token vectors")
            width = max(lengths)
            stacked = torch.zeros((len(batch), width, int(query.shape[1])), device=device, dtype=torch.float32)
            for index, value in enumerate(batch):
                stacked[index, : lengths[index]] = torch.as_tensor(value, device=device, dtype=torch.float32)
            token_scores = torch.einsum("qd,bpd->bqp", query, stacked)
            mask = torch.arange(width, device=device).unsqueeze(0) >= torch.tensor(lengths, device=device).unsqueeze(1)
            token_scores = token_scores.masked_fill(mask.unsqueeze(1), float("-inf"))
            values.extend(token_scores.max(dim=2).values.mean(dim=1).detach().cpu().tolist())
    return values


def run(args: argparse.Namespace) -> int:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("NF-OPT-16 requires offline model environment")
    trace = json.loads(args.cases.read_text(encoding="utf-8"))
    questions = {str(item["case_id"]): item for item in _jsonl(args.questions)}
    labels = {str(item["case_id"]): item for item in _jsonl(args.labels)}
    answerable = [item for item in trace["cases"] if not item.get("expected_no_answer")]
    no_answer = [item for item in trace["cases"] if item.get("expected_no_answer")]
    if len(answerable) != 64 or len(no_answer) != 8:
        raise ValueError("expected frozen 64 answerable and 8 no-answer cases")
    by_document, candidate_store_rows = _load_candidates(args)
    candidates_by_doc_id = {str(item["doc_id"]): item for values in by_document.values() for item in values}

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(str(args.model_path), use_fp16=True, devices=args.device, batch_size=args.batch_size)
    indexes, token_counts, index_build_ms = _encode_corpus(model, by_document, args)
    reranker = HeuristicReranker()
    query_processor = QueryProcessor()
    sparse_latencies: list[float] = []
    late_latencies: list[float] = []
    records: list[dict[str, Any]] = []
    baseline_hits: set[tuple[str, int]] = set()
    baseline_control_hits: set[tuple[str, int]] = set()
    sparse200_hits: set[tuple[str, int]] = set()
    late20_hits: set[tuple[str, int]] = set()
    rrf_hits: set[tuple[str, int]] = set()
    final_hits: set[tuple[str, int]] = set()
    final_all: set[str] = set()

    for case in answerable:
        case_id = str(case["case_id"])
        question = questions[case_id]
        raw_query = str(question["question"])
        scope = [str(value) for value in question.get("document_scope") or ()]
        if len(scope) != 1 or scope[0] not in by_document:
            raise ValueError(f"unsupported or missing document scope for {case_id}: {scope}")
        document_id = scope[0]
        candidates = by_document[document_id]
        encoded_query = model.encode(
            [query_processor.expand(raw_query)], batch_size=1, max_length=args.max_length,
            return_dense=False, return_sparse=True, return_colbert_vecs=True,
        )
        sparse_start = time.perf_counter()
        sparse_ranked = sparse_rank(
            query_weights=encoded_query["lexical_weights"][0],
            inverted_index=indexes[document_id],
            candidate_keys=[str(item["candidate_key"]) for item in candidates],
            limit=200,
        )
        sparse_latencies.append((time.perf_counter() - sparse_start) * 1000.0)
        sparse_candidates = [dict(candidates[index]) for index, _ in sparse_ranked]
        late_start = time.perf_counter()
        encoded_passages = model.encode(
            [str(item["content"]) for item in sparse_candidates], batch_size=args.batch_size, max_length=args.max_length,
            return_dense=False, return_sparse=False, return_colbert_vecs=True,
        )
        late_scores = _late_interaction_scores(
            encoded_query["colbert_vecs"][0], encoded_passages["colbert_vecs"], device=args.device, batch_size=args.late_batch_size,
        )
        late_latencies.append((time.perf_counter() - late_start) * 1000.0)
        late_ranked = rank_scores(late_scores, [str(item["candidate_key"]) for item in sparse_candidates], limit=20)
        late_candidates = []
        for rank, (index, score) in enumerate(late_ranked, start=1):
            item = dict(sparse_candidates[index])
            item.update({"score": score, "stage_rank": rank, "score_kind": "bge_m3_late_interaction"})
            late_candidates.append(item)
        dense = []
        for trace_candidate in case["retrieval_stages"]["dense"]:
            source_candidate = candidates_by_doc_id.get(str(trace_candidate["doc_id"]))
            if source_candidate is None or str(source_candidate["candidate_key"]) != str(trace_candidate["candidate_key"]):
                raise ValueError(f"Dense candidate identity drift: {trace_candidate['doc_id']}")
            item = dict(source_candidate)
            item.update(trace_candidate)
            dense.append(item)
        fused = normalize_scores(rrf([dense, late_candidates], k=60))
        fused = boost_front_matter_chunks(raw_query, fused, is_front_matter_query_fn=query_processor.is_front_matter_query)
        reranked = reranker.rerank(raw_query, fused, top_k=min(20, len(fused)))
        final = reranker.rerank(raw_query, fused, top_k=5)
        frozen_rrf = []
        for trace_candidate in case["retrieval_stages"]["rrf"]:
            source_candidate = candidates_by_doc_id.get(str(trace_candidate["doc_id"]))
            if source_candidate is None:
                raise ValueError(f"missing frozen RRF candidate content: {trace_candidate['doc_id']}")
            item = dict(source_candidate)
            item.update(trace_candidate)
            frozen_rrf.append(item)
        baseline_control = reranker.rerank(raw_query, frozen_rrf, top_k=5)
        sources = list(labels[case_id].get("expected_sources") or [])
        baseline = list(case["retrieval_stages"]["final"])
        baseline_case_hits = _source_indices(baseline, sources)
        sparse200_case_hits = _source_indices(sparse_candidates, sources)
        late20_case_hits = _source_indices(late_candidates, sources)
        rrf_case_hits = _source_indices(fused[:40], sources)
        final_case_hits = _source_indices(final, sources)
        baseline_hits.update((case_id, value) for value in baseline_case_hits)
        baseline_control_hits.update((case_id, value) for value in _source_indices(baseline_control, sources))
        sparse200_hits.update((case_id, value) for value in sparse200_case_hits)
        late20_hits.update((case_id, value) for value in late20_case_hits)
        rrf_hits.update((case_id, value) for value in rrf_case_hits)
        final_hits.update((case_id, value) for value in final_case_hits)
        if _coverage(final, sources) == "all":
            final_all.add(case_id)
        records.append({
            "case_id": case_id, "document_scope": scope,
            "sparse_top200_candidate_keys": [str(item["candidate_key"]) for item in sparse_candidates],
            "late_interaction_top20_candidate_keys": [str(item["candidate_key"]) for item in late_candidates],
            "rrf_candidate_keys": [str(item["candidate_key"]) for item in fused],
            "reranker_top20_candidate_keys": [str(item["candidate_key"]) for item in reranked],
            "final_candidate_keys": [str(item["candidate_key"]) for item in final],
            "sparse200_matched_source_count": len(sparse200_case_hits),
            "late20_matched_source_count": len(late20_case_hits),
            "rrf40_matched_source_count": len(rrf_case_hits),
            "final_matched_source_count": len(final_case_hits),
        })
    source_count = sum(len(labels[str(case["case_id"])].get("expected_sources") or []) for case in answerable)
    baseline_all = {str(case["case_id"]) for case in answerable if _coverage(list(case["retrieval_stages"]["final"]), list(labels[str(case["case_id"])].get("expected_sources") or [])) == "all"}
    regressions = baseline_hits - final_hits
    all_regressions = baseline_all - final_all
    control_matches = baseline_control_hits == baseline_hits
    if source_count != 80 or not control_matches:
        raise ValueError("baseline source count or frozen RRF control mismatch")
    final_gain = len(final_hits) - len(baseline_hits)
    all_gain = len(final_all) - len(baseline_all)
    decision = "bge_m3_multivector_shadow_validated" if final_gain >= 4 and all_gain >= 2 and not regressions and not all_regressions else "bge_m3_multivector_shadow_failed"
    next_gate = "production_candidate_shadow_ab" if decision.endswith("validated") else "stop_neural_retrieval_and_start_financial_hard_negative_reranker"
    metrics = {
        "baseline_final_source_recall_at_5": {"matched_sources": len(baseline_hits), "source_count": source_count},
        "baseline_control_source_recall_at_5": {"matched_sources": len(baseline_control_hits), "source_count": source_count},
        "sparse_stage_source_recall_at_200": {"matched_sources": len(sparse200_hits), "source_count": source_count},
        "late_interaction_source_recall_at_20": {"matched_sources": len(late20_hits), "source_count": source_count},
        "hybrid_rrf_source_recall_at_40": {"matched_sources": len(rrf_hits), "source_count": source_count},
        "shadow_final_source_recall_at_5": {"matched_sources": len(final_hits), "source_count": source_count},
        "baseline_final_all_gold_case_count": len(baseline_all), "shadow_final_all_gold_case_count": len(final_all),
        "new_final_hits": len(final_hits - baseline_hits), "regressed_final_hits": len(regressions), "regressed_final_all_gold_cases": len(all_regressions),
    }
    acceptance = {
        "artifact_schema": "nf-opt-16/gate-c/acceptance/v1", "case_count": 64, "no_answer_case_count": len(no_answer), "source_count": source_count,
        "input_hashes": {"cases_sha256": _sha(args.cases), "corpus_sha256": _sha(args.corpus), "questions_sha256": _sha(args.questions), "labels_sha256": _sha(args.labels)},
        "candidate_store_access_mode": "sqlite_read_only", "gold_used_only_for_posthoc_scoring": True,
        "sparse_first_stage_top_k": 200, "late_interaction_top_k": 20, "dense_behavior_modified": False, "rrf_parameters_modified": False, "reranker_behavior_modified": False, "final_top_k_modified": False,
        "embedding_model_network_calls": 0, "embedding_model_calls": len(by_document) + 2 * len(answerable), "answer_generation_calls": 0, "production_index_writes": 0, "production_behavior_changed": False, "production_switch_allowed": False, "no_answer_final_behavior_unchanged": True,
        "frozen_rrf_reranker_control_matches_baseline": control_matches, "decision": decision, "next_gate": next_gate,
    }
    _write(args.out_dir / "multivector-configuration.json", {"first_stage": "BGE-M3 sparse lexical top-200", "second_stage": "BGE-M3 ColBERT late interaction top-20", "dense_vectors_used": False, "gold_fields_read_for_selection": False, "configuration_frozen_before_benchmark_run": True})
    _write(args.out_dir / "sparse-index-manifest.json", {"candidate_store_rows_total": candidate_store_rows, "scope_candidate_count": sum(len(values) for values in by_document.values()), "document_count": len(by_document), "lexical_weight_count_by_document": token_counts, "index_build_ms": index_build_ms, "index_persisted": False})
    _write(args.out_dir / "late-interaction-stage-results.json", {"metrics": metrics, "cases": records})
    _write(args.out_dir / "hybrid-transfer-results.json", {"reranker_input_source": "rrf_all", "final_top_k": 5, "metrics": metrics})
    _write(args.out_dir / "strict-hit-regression-report.json", {"new_source_instances": sorted(final_hits - baseline_hits), "regressed_source_instances": sorted(regressions), "regressed_all_gold_cases": sorted(all_regressions)})
    _write(args.out_dir / "latency-and-resource-report.json", {"sparse_query_ms": {"mean": sum(sparse_latencies) / len(sparse_latencies), "p50": _percentile(sparse_latencies, .5), "p95": _percentile(sparse_latencies, .95)}, "late_interaction_ms": {"mean": sum(late_latencies) / len(late_latencies), "p50": _percentile(late_latencies, .5), "p95": _percentile(late_latencies, .95)}, "index_build_ms": index_build_ms, "full_corpus_colbert_index_persisted": False})
    _write(args.out_dir / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    _write(args.out_dir / "nf-opt-16-gate-c-acceptance.json", acceptance)
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
    parser.add_argument("--late-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
