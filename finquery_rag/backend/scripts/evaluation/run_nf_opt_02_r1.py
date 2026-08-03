"""NF-OPT-02 R1 transfer evaluation using the production reranker path."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import os
import time
from typing import Any

import chromadb
import numpy as np

from scripts.evaluation import run_nf_eval_03_r1 as eval_r1
from scripts.evaluation import run_nf_opt_01 as opt01
from src.evaluation.nf_opt_01 import candidate_scope_ok, percentile
from src.evaluation.nf_opt_02 import protected_dense_merge
from src.evaluation.nf_opt_02_r1 import (
    all_gold_cases,
    coverage_counts,
    full_coverage,
    hit_set,
    lineage_report,
    mrr,
    promotion_demotion,
    select_smallest_passing_variant,
    source_rows,
    transfer_gate,
)
from src.retrieval.query_processor import QueryProcessor
from src.services.reranker import build_reranker

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "financial_rag_v1"
DATA = BENCHMARK / "data"
OUT = ROOT / "artifacts" / "evaluation" / "nf-opt-02-r1"
RESIDUAL = ROOT / "runtime" / "evaluation" / "nf-opt-02" / "residual-chroma"
NEGATIVE = ROOT / "artifacts" / "evaluation" / "nf-eval-02" / "negative-evidence-review-report.json"
NF04 = ROOT / "artifacts" / "evaluation" / "nf-eval-04"
NF02 = ROOT / "artifacts" / "evaluation" / "nf-opt-02"


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument("--chroma-path", type=Path, default=ROOT / "chroma_db")
    parser.add_argument("--bm25-db-path", type=Path, default=ROOT / "rag_bm25.db")
    parser.add_argument("--residual-path", type=Path, default=RESIDUAL)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--corpus", type=Path, default=BENCHMARK / "corpus.json")
    parser.add_argument("--manifest", type=Path, default=DATA / "golden-manifest.json")
    parser.add_argument("--questions", type=Path, default=DATA / "questions.golden.jsonl")
    parser.add_argument("--labels", type=Path, default=DATA / "labels.golden.jsonl")
    parser.add_argument("--review-status", type=Path, default=DATA / "review-status.golden.jsonl")
    parser.add_argument("--negative-report", type=Path, default=NEGATIVE)
    return parser.parse_args()


def input_report(inputs: Any, args: argparse.Namespace) -> dict[str, Any]:
    actual = dict(inputs.hash_report["actual"])
    prior = json.loads((NF04 / "input-integrity-report.json").read_text(encoding="utf-8"))
    fields = ("question_hash", "reference_answer_hash", "source_identity_hash", "negative_evidence_hash", "review_status_hash", "corpus_hash", "golden_manifest_sha256")
    unchanged = {field: actual.get(field) == prior.get(field) for field in fields}
    return {
        "artifact_schema": "nf-opt-02-r1/v1",
        "tenant_id": args.tenant_id,
        "case_count": 64,
        "expected_source_count": 80,
        "allowed_document_count": 8,
        **{field: actual.get(field) for field in fields},
        "all_hashes_recomputed_and_verified": all(inputs.hash_report["matches"].values()),
        "nf_eval_04_hashes_unchanged": all(unchanged.values()),
        "legacy_documents_loaded": 0,
    }


def hit_diff(before: set[str], after: set[str]) -> dict[str, int]:
    return {
        "baseline_source_count": len(before),
        "variant_source_count": len(after),
        "new_source_count": len(after - before),
        "regressed_source_count": len(before - after),
        "both_source_count": len(before & after),
    }


def run(args: argparse.Namespace) -> int:
    if args.tenant_id != 1:
        raise ValueError("tenant 1 is required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    inputs = eval_r1._load_inputs(
        corpus_path=args.corpus,
        manifest_path=args.manifest,
        questions_path=args.questions,
        labels_path=args.labels,
        review_status_path=args.review_status,
        negative_report_path=args.negative_report,
    )
    integrity = input_report(inputs, args)
    if not integrity["all_hashes_recomputed_and_verified"] or not integrity["nf_eval_04_hashes_unchanged"]:
        raise ValueError("frozen input integrity failed")
    residual_manifest = json.loads((NF02 / "residual-index-manifest.json").read_text(encoding="utf-8"))
    if residual_manifest.get("gold_identity_presence_count") != 80:
        raise ValueError("residual index is not complete")
    mapping = eval_r1._doc_map(inputs.corpus)
    allowed = {str(item["document_id"]) for item in inputs.corpus["documents"]}
    sources, _ = opt01._load_gold_keys(inputs.labels_by_id)
    if len(inputs.questions) != 72 or len(sources) != 80:
        raise ValueError("expected 72 questions and 80 sources")

    from src.services import vector_store
    from src.services.retrieval import SqliteBM25Retriever

    current_client = chromadb.PersistentClient(path=str(args.chroma_path))
    current = current_client.get_collection(name=vector_store.GLOBAL_COLLECTION_NAME, embedding_function=vector_store.embed_fn)
    residual_client = chromadb.PersistentClient(path=str(args.residual_path))
    residual = residual_client.get_collection(name="financial_rag_v1_dense_residual", embedding_function=vector_store.embed_fn)
    reranker = build_reranker("heuristic")
    if reranker is None:
        raise ValueError("heuristic production reranker unavailable")
    processor = QueryProcessor()
    bm25 = SqliteBM25Retriever(db_path=str(args.bm25_db_path))
    variants = {"A": 0, "C10": 10, "C20": 20, "C40": 40}
    rows = {name: [] for name in variants}
    lineages = {name: [] for name in variants}
    latency = {name: defaultdict(list) for name in variants}

    for question in inputs.questions:
        case_id = str(question["case_id"])
        label = inputs.labels_by_id[case_id]
        if label.get("expected_no_answer"):
            continue
        scope = [str(value) for value in question.get("document_scope") or []]
        if len(scope) != 1 or scope[0] not in allowed:
            raise ValueError(f"{case_id}: out-of-scope question")
        filename = next(item["filename"] for item in inputs.corpus["documents"] if str(item["document_id"]) == scope[0])
        query = str(question["question"])
        started = time.perf_counter()
        embedding = np.asarray(vector_store.embed_fn([processor.expand(query)])[0], dtype=np.float32)
        embed_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        base = opt01._query_dense(current, query_embedding=embedding, filename=filename, tenant_id=args.tenant_id, limit=40, mapping=mapping)
        base_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        residual_rows = opt01._query_dense(residual, query_embedding=embedding, filename=filename, tenant_id=args.tenant_id, limit=40, mapping=mapping)
        residual_ms = (time.perf_counter() - started) * 1000.0
        candidate_k = 40 if processor.is_numeric_query(query) else 20
        started = time.perf_counter()
        sparse = [opt01._annotated_candidate(item, mapping=mapping, tenant_id=args.tenant_id) for item in bm25.search(processor.expand(query), k=candidate_k, doc_name=filename, user_id=args.tenant_id)]
        sparse = [item for item in sparse if candidate_scope_ok(item.get("document_id"), allowed)]
        bm25_ms = (time.perf_counter() - started) * 1000.0
        expected = [item for item in sources if item["case_id"] == case_id]
        for name, budget in variants.items():
            started = time.perf_counter()
            dense = protected_dense_merge(base_candidates=base, residual_candidates=residual_rows[:budget])
            merge_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            fused = opt01._rrf_candidates(dense, sparse, query=query, query_processor=processor, mapping=mapping, tenant_id=args.tenant_id)
            rrf_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            reranked = reranker.rerank(query, fused, top_k=min(20, len(fused)))
            reranker_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            final = reranker.rerank(query, fused, top_k=5)
            final_ms = (time.perf_counter() - started) * 1000.0
            rows[name].extend(source_rows(case_id=case_id, expected_sources=expected, stage_candidates={"rrf": fused, "reranker": reranked, "final": final}))
            lineages[name].append(lineage_report(rrf_input=fused, reranker_output=reranked, final_output=final, allowed_document_ids=allowed))
            measured = {"query_embedding_ms": embed_ms, "base_dense_query_ms": base_ms, "residual_dense_query_ms": residual_ms if budget else 0.0, "protected_merge_ms": merge_ms, "bm25_ms": bm25_ms, "rrf_ms": rrf_ms, "reranker_ms": reranker_ms, "final_selector_ms": final_ms}
            measured["total_retrieval_ms"] = sum(measured.values())
            for stage, value in measured.items():
                latency[name][stage].append(value)

    base_rows = rows["A"]
    base_r20, base_r10, base_f5 = hit_set(base_rows, "reranker", 20), hit_set(base_rows, "reranker", 10), hit_set(base_rows, "final", 5)
    base_r20_all, base_f5_all = all_gold_cases(base_rows, "reranker", 20), all_gold_cases(base_rows, "final", 5)
    comparisons = {}
    for name in variants:
        current_rows = rows[name]
        r20, r10, f5 = hit_set(current_rows, "reranker", 20), hit_set(current_rows, "reranker", 10), hit_set(current_rows, "final", 5)
        r20_all, f5_all = all_gold_cases(current_rows, "reranker", 20), all_gold_cases(current_rows, "final", 5)
        lineages_for_variant = lineages[name]
        lineage_ok = all(item["lineage_passed"] for item in lineages_for_variant)
        r20_diff, r10_diff, f5_diff = hit_diff(base_r20, r20), hit_diff(base_r10, r10), hit_diff(base_f5, f5)
        dense_values = [
            base_value + residual_value
            for base_value, residual_value in zip(
                latency[name]["base_dense_query_ms"],
                latency[name]["residual_dense_query_ms"],
            )
        ]
        base_dense_values = latency["A"]["base_dense_query_ms"]
        dense_p95 = percentile(dense_values, .95) or 0.0
        total_p95 = percentile(latency[name]["total_retrieval_ms"], .95) or 0.0
        base_dense_p95 = percentile(base_dense_values, .95) or 0.0
        base_total_p95 = percentile(latency["A"]["total_retrieval_ms"], .95) or 0.0
        latency_ok = (
            (dense_p95 - base_dense_p95) / base_dense_p95 <= .25
            and (total_p95 - base_total_p95) / base_total_p95 <= .25
        ) if base_dense_p95 and base_total_p95 else False
        gate = transfer_gate(
            completeness_passed=len(current_rows) == 80,
            lineage_passed=lineage_ok,
            model_calls=0,
            answer_generation_calls=0,
            reranker20_source_gain=r20_diff["new_source_count"],
            reranker10_source_gain=r10_diff["new_source_count"],
            final5_source_gain=f5_diff["new_source_count"],
            reranker20_all_gold_gain=len(r20_all - base_r20_all),
            final_all_gold_gain=len(f5_all - base_f5_all),
            reranker20_source_regression=r20_diff["regressed_source_count"],
            reranker20_all_gold_regression=len(base_r20_all - r20_all),
            final_source_regression=f5_diff["regressed_source_count"],
            final_all_gold_regression=len(base_f5_all - f5_all),
            latency_gate_passed=latency_ok,
        )
        comparisons[name] = {
            "rrf_full_pool": {"coverage": full_coverage(current_rows, "rrf"), "source_count": 80},
            "reranker": {"metrics": coverage_counts(current_rows, "reranker"), "mrr_at_20": mrr(current_rows, "reranker", cutoff=20), "promotion_demotion": promotion_demotion(current_rows, "reranker")},
            "final": {"metrics": coverage_counts(current_rows, "final", cutoffs=(5,)), "mrr_at_5": mrr(current_rows, "final", cutoff=5)},
            "regression": {"reranker20": r20_diff, "reranker10": r10_diff, "final5": f5_diff, "reranker20_all_gold_gain": len(r20_all - base_r20_all), "final_all_gold_gain": len(f5_all - base_f5_all)},
            "lineage": {"case_count": len(lineages_for_variant), "lineage_failure_count": sum(not item["lineage_passed"] for item in lineages_for_variant), "candidate_injection_count": sum(item["reranker_candidate_injection_count"] + item["final_candidate_injection_count"] for item in lineages_for_variant), "out_of_scope_count": sum(item["out_of_scope_candidate_count"] for item in lineages_for_variant), "reranker_input_source": "rrf_all"},
            "latency": {stage: {"p50_ms": percentile(values, .5), "p95_ms": percentile(values, .95), "mean_ms": sum(values) / len(values), "warmup_count": 0, "measured_run_count": len(values)} for stage, values in latency[name].items()},
            "gate": gate,
        }

    selected = select_smallest_passing_variant({name: comparisons[name]["gate"] for name in ("C10", "C20", "C40")})
    transfer_exists = any(comparisons[name]["gate"]["transfer_gain_passed"] and comparisons[name]["gate"]["regression_passed"] for name in ("C10", "C20", "C40"))
    latency_exists = any(comparisons[name]["gate"]["latency_gate_passed"] for name in ("C10", "C20", "C40"))
    if transfer_exists and not latency_exists:
        decision, next_gate = "protected_residual_transfer_validated_latency_blocked", "parallel_base_residual_query_ab"
    elif selected:
        decision, next_gate = "protected_residual_transfer_validated", "production_config_shadow_validation"
    else:
        decision, next_gate = "protected_residual_transfer_failed", "stop_residual_dense_and_pivot_bm25_window"
    acceptance = {"artifact_schema": "nf-opt-02-r1/v1", "decision": decision, "selected_variant": selected, "production_switch_allowed": False, "production_behavior_changed": False, "candidate_lineage_passed": all(comparisons[name]["lineage"]["lineage_failure_count"] == 0 for name in variants), "scope_integrity_passed": all(comparisons[name]["lineage"]["out_of_scope_count"] == 0 for name in variants), "model_chat_completion_requests": 0, "answer_generation_calls": 0, "reranker_calls": 64 * 4 * 2, "final_selector_calls": 64 * 4, "legacy_27_loaded": False}
    write(args.out_dir / "input-integrity-report.json", integrity)
    write(args.out_dir / "variant-manifest.json", {"variants": variants, "reranker_input_source": "rrf_all", "reranker": "heuristic", "final_selector": "current_production_top5", "candidate_multiplier": 4})
    write(args.out_dir / "candidate-lineage-report.json", {name: comparisons[name]["lineage"] for name in variants})
    write(args.out_dir / "rrf-full-pool-comparison.json", {name: comparisons[name]["rrf_full_pool"] for name in variants})
    write(args.out_dir / "reranker-transfer-comparison.json", {name: comparisons[name]["reranker"] for name in variants})
    write(args.out_dir / "final-transfer-comparison.json", {name: comparisons[name]["final"] for name in variants})
    write(args.out_dir / "regression-report.json", {name: comparisons[name]["regression"] for name in variants})
    write(args.out_dir / "latency-breakdown.json", {name: comparisons[name]["latency"] for name in variants})
    write(args.out_dir / "variant-selection.json", {"order": ["C10", "C20", "C40"], "selected_variant": selected, "comparisons": {name: comparisons[name]["gate"] for name in variants}})
    write(args.out_dir / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    write(args.out_dir / "nf-opt-02-r1-acceptance.json", acceptance)
    print(json.dumps({"acceptance": acceptance, "variants": {name: comparisons[name]["gate"] for name in variants}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (ValueError, KeyError, OSError) as exc:
        print(f"NF-OPT-02 R1 failed: {exc}")
        raise SystemExit(2)
