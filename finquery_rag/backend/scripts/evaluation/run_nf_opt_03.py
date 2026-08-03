"""NF-OPT-03 read-only BM25 candidate-window A/B."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import chromadb
import numpy as np

from scripts.evaluation import run_nf_eval_03_r1 as eval_r1
from scripts.evaluation import run_nf_eval_04 as eval_04
from scripts.evaluation import run_nf_opt_01 as opt01
from src.evaluation.nf_opt_01 import percentile
from src.evaluation.nf_opt_02_r1 import all_gold_cases, coverage_counts, full_coverage, mrr, source_rows
from src.evaluation.nf_opt_03 import (
    candidate_keys, compare_sources, dynamic_coverage,
    identity_integrity, latency_gate, multi_evidence_all_gold,
    prefix_integrity, select_smallest_passing_window, source_hit_set,
    window_gate,
)
from src.retrieval.query_processor import QueryProcessor
from src.services.reranker import build_reranker
from src.services.retrieval import SqliteBM25Retriever

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "financial_rag_v1"
DATA = BENCHMARK / "data"
OUT = ROOT / "artifacts" / "evaluation" / "nf-opt-03"
NF04 = ROOT / "artifacts" / "evaluation" / "nf-eval-04"
NF03 = ROOT / "artifacts" / "evaluation" / "nf-eval-03-r1"
NEGATIVE = ROOT / "artifacts" / "evaluation" / "nf-eval-02" / "negative-evidence-review-report.json"
LEGACY_FILES = {"FINAL Annual Report.pdf", "leac203.pdf", "wipo_pub_rn2021_18e.pdf"}


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument("--chroma-path", type=Path, default=ROOT / "chroma_db")
    parser.add_argument("--bm25-db-path", type=Path, default=ROOT / "rag_bm25.db")
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
    fields = ("question_hash", "reference_answer_hash", "source_identity_hash", "negative_evidence_hash", "review_status_hash", "corpus_hash", "benchmark_hash", "golden_manifest_sha256")
    actual["benchmark_hash"] = hashlib.sha256(json.dumps({
        key: actual.get(key) for key in (
            "question_hash", "reference_answer_hash", "source_identity_hash",
            "negative_evidence_hash", "review_status_hash", "corpus_hash",
            "golden_manifest_sha256",
        )
    }, sort_keys=True).encode("utf-8")).hexdigest()
    unchanged = {field: actual.get(field) == prior.get(field) for field in fields}
    return {
        "artifact_schema": "nf-opt-03/v1", "benchmark_id": "financial-rag-v1",
        "tenant_id": args.tenant_id, "case_count": 64, "expected_source_count": 80,
        "allowed_document_count": 8,
        **{field: actual.get(field) or prior.get(field) for field in fields},
        "all_hashes_recomputed_and_verified": all(inputs.hash_report["matches"].values()),
        "nf_eval_04_hashes_unchanged": all(unchanged.values()),
        "legacy_documents_loaded": 0,
    }


def production_config() -> dict[str, Any]:
    manifest = json.loads((NF03 / "baseline-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("embedding_model") != "all-MiniLM-L6-v2":
        raise ValueError("production embedding is not all-MiniLM-L6-v2")
    return {
        "embedding_model": manifest["embedding_model"],
        "n_results": int(manifest["n_results"]),
        "candidate_multiplier": int(manifest["retrieval_candidate_multiplier"]),
        "reranker": manifest["reranker"],
        "final_k": int(manifest["n_results"]),
    }


def current_limit(query: str, processor: QueryProcessor, config: dict[str, Any]) -> int:
    base = config["n_results"] * config["candidate_multiplier"]
    return max(base, config["n_results"] * 8) if processor.is_numeric_query(query) else base


def annotate(raw: list[dict[str, Any]], mapping: dict[str, str], tenant_id: int, allowed: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [opt01._annotated_candidate(item, mapping=mapping, tenant_id=tenant_id) for item in raw]
    for row in rows:
        if not row.get("content_hash"):
            row["content_hash"] = opt01._content_hash(row.get("content") or row.get("text"))
    return rows, identity_integrity(rows, allowed_document_ids=allowed)


def query_bm25(retriever: SqliteBM25Retriever, *, query: str, filename: str, limit: int, current: int, processor: QueryProcessor, mapping: dict[str, str], tenant_id: int, allowed: set[str]) -> tuple[list[dict[str, Any]], float, float, dict[str, Any]]:
    expanded = processor.expand(query)
    started = time.perf_counter()
    raw = retriever.search(expanded, k=limit, doc_name=filename, user_id=tenant_id) if limit == current else eval_04._direct_bm25_top_n(retriever, expanded, doc_name=filename, user_id=tenant_id, limit=limit)
    bm25_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    rows, audit = annotate(raw, mapping, tenant_id, allowed)
    return rows, bm25_ms, (time.perf_counter() - started) * 1000, audit


def lineage(stages: dict[str, list[dict[str, Any]]], allowed: set[str]) -> dict[str, Any]:
    """Audit stage identity and record explainable RRF deduplication.

    Production RRF keys by raw doc_id while the evaluation stages expose
    canonical candidate_key. Multiple canonical candidates can therefore
    collapse at the unchanged production RRF boundary. Such drops are
    reported explicitly; only injections, missing identities, scope escapes,
    or downstream subset violations fail lineage.
    """
    audits = {name: identity_integrity(items, allowed_document_ids=allowed) for name, items in stages.items()}
    union_keys = set(candidate_keys(stages["union"]))
    rrf_keys = set(candidate_keys(stages["rrf"]))
    rerank_keys = set(candidate_keys(stages["reranker"]))
    final_keys = set(candidate_keys(stages["final"]))
    rrf_dropped = union_keys - rrf_keys
    rrf_injected = rrf_keys - union_keys
    return {
        "stage_audits": audits,
        "reranker_input_source": "rrf_all",
        "reranker_input_count": len(stages["rrf"]),
        "reranker_output_count": len(stages["reranker"]),
        "final_output_count": len(stages["final"]),
        "rrf_missing_union_count": len(rrf_dropped),
        "rrf_dropped_candidate_count": len(rrf_dropped),
        "rrf_drop_reason": (
            "production_rrf_doc_id_deduplication"
            if rrf_dropped and not rrf_injected
            else None
        ),
        "rrf_candidate_injection_count": len(rrf_injected),
        "reranker_candidate_injection_count": len(rerank_keys - rrf_keys),
        "final_candidate_injection_count": len(final_keys - rerank_keys),
        "missing_identity_count": sum(int(a["missing_identity_count"]) for a in audits.values()),
        "out_of_scope_candidate_count": sum(int(a["out_of_scope_candidate_count"]) for a in audits.values()),
        "rrf_output_subset_of_union": rrf_keys <= union_keys,
        "reranker_output_subset_of_input": rerank_keys <= rrf_keys,
        "final_output_subset_of_reranker": final_keys <= rerank_keys,
        "lineage_passed": (
            all(a["passed"] for a in audits.values())
            and rrf_keys <= union_keys
            and rerank_keys <= rrf_keys
            and final_keys <= rerank_keys
        ),
    }

def dynamic_all(rows: list[dict[str, Any]], stage: str, limits: dict[str, int]) -> set[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    return {case for case, items in grouped.items() if all(isinstance(x.get(f"{stage}_rank"), int) and x[f"{stage}_rank"] <= limits[case] for x in items)}


def latency_summary(values: dict[str, list[float]], counts: list[int], warmups: int) -> dict[str, Any]:
    result = {stage: {"p50_ms": percentile(items, .5), "p95_ms": percentile(items, .95), "mean_ms": sum(items) / len(items), "warmup_count": warmups, "measured_run_count": len(items)} for stage, items in values.items()}
    result["candidate_count_p50"] = percentile([float(x) for x in counts], .5)
    result["candidate_count_p95"] = percentile([float(x) for x in counts], .95)
    return result


def run(args: argparse.Namespace) -> int:
    if args.tenant_id != 1:
        raise ValueError("tenant 1 is required")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inputs = eval_r1._load_inputs(corpus_path=args.corpus, manifest_path=args.manifest, questions_path=args.questions, labels_path=args.labels, review_status_path=args.review_status, negative_report_path=args.negative_report)
    integrity = input_report(inputs, args)
    if not integrity["all_hashes_recomputed_and_verified"] or not integrity["nf_eval_04_hashes_unchanged"]:
        raise ValueError("frozen benchmark input integrity failed")
    config = production_config()
    mapping = eval_r1._doc_map(inputs.corpus)
    allowed = {str(item["document_id"]) for item in inputs.corpus["documents"]}
    if LEGACY_FILES & {str(item.get("filename")) for item in inputs.corpus["documents"]}:
        raise ValueError("legacy document entered benchmark corpus")
    sources, _ = opt01._load_gold_keys(inputs.labels_by_id)
    answerable = [q for q in inputs.questions if not inputs.labels_by_id[str(q["case_id"])].get("expected_no_answer")]
    if len(answerable) != 64 or len(sources) != 80:
        raise ValueError("expected 64 answerable cases and 80 sources")
    processor = QueryProcessor()
    bm25 = SqliteBM25Retriever(db_path=str(args.bm25_db_path))
    client = chromadb.PersistentClient(path=str(args.chroma_path))
    from src.services import vector_store
    dense_collection = client.get_collection(name=vector_store.GLOBAL_COLLECTION_NAME, embedding_function=vector_store.embed_fn)
    reranker = build_reranker(config["reranker"])
    if reranker is None:
        raise ValueError("production reranker unavailable")
    variants: dict[str, int | None] = {"A": None, "B80": 80, "B120": 120, "B200": 200}
    rows = {name: [] for name in variants}
    lineages = {name: [] for name in variants}
    prefixes = {name: [] for name in variants}
    audits = {name: [] for name in variants}
    latencies = {name: defaultdict(list) for name in variants}
    counts = {name: [] for name in variants}
    current_limits: dict[str, int] = {}
    multi_cases = {str(row["case_id"]) for row in sources if sum(x["case_id"] == row["case_id"] for x in sources) > 1}

    def process(question: dict[str, Any], *, record: bool, index: int) -> None:
        case_id = str(question["case_id"])
        scope = [str(x) for x in question.get("document_scope") or []]
        if len(scope) != 1 or scope[0] not in allowed:
            raise ValueError(f"{case_id}: out-of-scope question")
        filename = next(str(x["filename"]) for x in inputs.corpus["documents"] if str(x["document_id"]) == scope[0])
        query = str(question["question"])
        limit = current_limit(query, processor, config)
        current_limits[case_id] = limit
        started = time.perf_counter()
        embedding = np.asarray(vector_store.embed_fn([processor.expand(query)])[0], dtype=np.float32)
        embed_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        dense = opt01._query_dense(dense_collection, query_embedding=embedding, filename=filename, tenant_id=args.tenant_id, limit=limit, mapping=mapping)
        dense, _dense_audit = annotate(dense, mapping, args.tenant_id, allowed)
        dense_ms = (time.perf_counter() - started) * 1000
        order = list(variants)
        shift = index % len(order)
        order = order[shift:] + order[:shift]
        sparse_by_name: dict[str, list[dict[str, Any]]] = {}
        times_by_name: dict[str, tuple[float, float]] = {}
        audits_by_name: dict[str, dict[str, Any]] = {}
        for name in order:
            target = limit if name == "A" else int(variants[name])
            sparse, bm25_ms, norm_ms, audit = query_bm25(bm25, query=query, filename=filename, limit=target, current=limit, processor=processor, mapping=mapping, tenant_id=args.tenant_id, allowed=allowed)
            sparse_by_name[name], times_by_name[name], audits_by_name[name] = sparse, (bm25_ms, norm_ms), audit
        expected = [x for x in sources if x["case_id"] == case_id]
        for name in order:
            sparse = sparse_by_name[name]
            started = time.perf_counter()
            union = opt01._union_candidates(dense, sparse, mapping=mapping, tenant_id=args.tenant_id)
            union_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            fused = opt01._rrf_candidates(dense, sparse, query=query, query_processor=processor, mapping=mapping, tenant_id=args.tenant_id)
            rrf_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            reranked = reranker.rerank(query, fused, top_k=min(20, len(fused)))
            reranker_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            final = reranker.rerank(query, fused, top_k=config["final_k"])
            final_ms = (time.perf_counter() - started) * 1000
            stages = {"dense": dense, "bm25": sparse, "union": union, "rrf": fused, "reranker": reranked, "final": final}
            lin = lineage(stages, allowed)
            pref = prefix_integrity(sparse_by_name["A"], sparse)
            if not record:
                continue
            rows[name].extend(source_rows(case_id=case_id, expected_sources=expected, stage_candidates=stages))
            lineages[name].append(lin)
            prefixes[name].append({"case_id": case_id, **pref})
            audits[name].append(audits_by_name[name])
            measured = {"query_embedding_ms": embed_ms, "dense_ms": dense_ms, "bm25_ms": times_by_name[name][0], "normalization_ms": times_by_name[name][1], "union_ms": union_ms, "rrf_ms": rrf_ms, "reranker_ms": reranker_ms, "final_selector_ms": final_ms}
            measured["total_retrieval_ms"] = sum(measured.values())
            for stage, value in measured.items():
                latencies[name][stage].append(value)
            counts[name].append(len(sparse))

    warmups = min(5, len(answerable))
    for index, question in enumerate(answerable[:warmups]):
        process(question, record=False, index=index)
    for index, question in enumerate(answerable):
        process(question, record=True, index=index)

    base = rows["A"]
    base_limits = {case: current_limits[case] for case in current_limits}
    base_bm25 = source_hit_set(base, "bm25")
    base_bm25_all = dynamic_all(base, "bm25", base_limits)
    base_rrf40, base_reranker20, base_final5 = source_hit_set(base, "rrf", 40), source_hit_set(base, "reranker", 20), source_hit_set(base, "final", 5)
    base_rrf40_all, base_reranker20_all, base_final5_all = all_gold_cases(base, "rrf", 40), all_gold_cases(base, "reranker", 20), all_gold_cases(base, "final", 5)
    comparisons: dict[str, Any] = {}
    for name, target in variants.items():
        current = rows[name]
        limits = base_limits if name == "A" else {case: int(target) for case in current_limits}
        bm25_window = dynamic_coverage(current, stage="bm25", limit_by_case=limits)
        bm25_all = dynamic_all(current, "bm25", limits)
        rrf40, reranker20, final5 = source_hit_set(current, "rrf", 40), source_hit_set(current, "reranker", 20), source_hit_set(current, "final", 5)
        rrf40_all, reranker20_all, final5_all = all_gold_cases(current, "rrf", 40), all_gold_cases(current, "reranker", 20), all_gold_cases(current, "final", 5)
        reg_bm25 = compare_sources(base, current, stage="bm25")
        reg_rrf = compare_sources(base, current, stage="rrf", cutoff=40)
        reg_reranker = compare_sources(base, current, stage="reranker", cutoff=20)
        reg_final = compare_sources(base, current, stage="final", cutoff=5)
        base_total, base_bm25_ms = percentile(latencies["A"]["total_retrieval_ms"], .95), percentile(latencies["A"]["bm25_ms"], .95)
        total_p95, bm25_p95 = percentile(latencies[name]["total_retrieval_ms"], .95), percentile(latencies[name]["bm25_ms"], .95)
        total_ratio = (total_p95 - base_total) / base_total if base_total else None
        bm25_ratio = (bm25_p95 - base_bm25_ms) / base_bm25_ms if base_bm25_ms else None
        lineage_ok = all(x["lineage_passed"] for x in lineages[name])
        prefix_ok = all(x["passed"] for x in prefixes[name])
        scope_ok = all(int(x["out_of_scope_candidate_count"]) == 0 for x in audits[name])
        gate = window_gate(
            complete=len(current) == 80, prefix_passed=prefix_ok, lineage_passed=lineage_ok, scope_passed=scope_ok, model_calls=0, answer_generation_calls=0,
            bm25_source_gain=bm25_window["source_hit_count"] - len(base_bm25), bm25_all_gold_gain=len(bm25_all - base_bm25_all),
            rrf40_source_gain=len(rrf40 - base_rrf40), reranker20_source_gain=len(reranker20 - base_reranker20), final5_source_gain=len(final5 - base_final5), final_all_gold_gain=len(final5_all - base_final5_all),
            bm25_source_regression=len(base_bm25 - source_hit_set(current, "bm25")), rrf40_source_regression=len(base_rrf40 - rrf40), reranker20_source_regression=len(base_reranker20 - reranker20), final5_source_regression=len(base_final5 - final5),
            rrf_all_gold_regression=len(base_rrf40_all - rrf40_all), reranker_all_gold_regression=len(base_reranker20_all - reranker20_all), final_all_gold_regression=len(base_final5_all - final5_all),
            latency_passed=latency_gate(total_ratio=total_ratio, bm25_ratio=bm25_ratio),
        )
        comparisons[name] = {
            "bm25_window_limit": "current_production" if name == "A" else target,
            "bm25_window_metrics": bm25_window,
            "bm25_rank_metrics": coverage_counts(current, "bm25", cutoffs=(20, 40, 80, 120, 200)),
            "production_union": {"source_hit_count": len(source_hit_set(current, "union")), "source_recall": len(source_hit_set(current, "union")) / 80, "coverage": full_coverage(current, "union")},
            "rrf": {"metrics": coverage_counts(current, "rrf", cutoffs=(20, 40, 100)), "full_pool_coverage": full_coverage(current, "rrf"), "top40_coverage": full_coverage([{**x, "rrf_rank": x["rrf_rank"] if isinstance(x.get("rrf_rank"), int) and x["rrf_rank"] <= 40 else None} for x in current], "rrf"), "multi_evidence_all_source_coverage": multi_evidence_all_gold(current, stage="rrf", multi_case_ids=multi_cases)},
            "reranker": {"metrics": coverage_counts(current, "reranker", cutoffs=(5, 10, 20)), "mrr_at_20": mrr(current, "reranker", cutoff=20)},
            "final": {"metrics": coverage_counts(current, "final", cutoffs=(5,)), "multi_evidence_all_source_coverage": multi_evidence_all_gold(current, stage="final", multi_case_ids=multi_cases, cutoff=5)},
            "regression": {"bm25": reg_bm25, "rrf40": reg_rrf, "reranker20": reg_reranker, "final5": reg_final, "rrf_all_gold_regressed_cases": len(base_rrf40_all - rrf40_all), "reranker_all_gold_regressed_cases": len(base_reranker20_all - reranker20_all), "final_all_gold_regressed_cases": len(base_final5_all - final5_all)},
            "lineage": {"case_count": len(lineages[name]), "lineage_failure_count": sum(not x["lineage_passed"] for x in lineages[name]), "missing_identity_count": sum(int(x["missing_identity_count"]) for x in lineages[name]), "candidate_injection_count": sum(int(x["rrf_candidate_injection_count"]) + int(x["reranker_candidate_injection_count"]) + int(x["final_candidate_injection_count"]) for x in lineages[name]), "out_of_scope_count": sum(int(x["out_of_scope_candidate_count"]) for x in lineages[name]), "reranker_input_source": "rrf_all"},
            "latency": latency_summary(latencies[name], counts[name], warmups),
            "gate": gate,
        }

    passing = {name: comparisons[name]["gate"] for name in ("B80", "B120", "B200")}
    selected = select_smallest_passing_window(passing)
    if selected:
        decision, next_gate = "bm25_window_validated", "production_config_shadow_validation"
    elif any(not comparisons[name]["gate"]["integrity_passed"] for name in variants):
        decision, next_gate = "bm25_window_integrity_failed", "stop_and_repair_evaluation"
    elif any(comparisons[name]["gate"]["bm25_gain_passed"] and len(source_hit_set(rows[name], "rrf", 40) - base_rrf40) >= 6 and len(source_hit_set(rows[name], "reranker", 20) - base_reranker20) >= 5 and not comparisons[name]["gate"]["transfer_gain_passed"] for name in ("B80", "B120", "B200")):
        decision, next_gate = "bm25_window_blocked_by_final_selector", "final_context_budget_ab"
    elif any(comparisons[name]["gate"]["bm25_gain_passed"] and not comparisons[name]["gate"]["transfer_gain_passed"] for name in ("B80", "B120", "B200")):
        decision, next_gate = "bm25_window_gain_not_safely_transferred", "final_evidence_budget_ab"
    else:
        decision, next_gate = "bm25_window_gain_insufficient", "retrieval_representation_or_query_analysis"

    acceptance = {
        "artifact_schema": "nf-opt-03/v1", "decision": decision, "selected_bm25_variant": selected,
        "selected_bm25_limit": None if selected is None else comparisons[selected]["bm25_window_limit"],
        "production_switch_allowed": False, "production_behavior_changed": False,
        "candidate_lineage_passed": all(comparisons[name]["lineage"]["lineage_failure_count"] == 0 for name in variants),
        "scope_integrity_passed": all(comparisons[name]["lineage"]["out_of_scope_count"] == 0 for name in variants),
        "model_chat_completion_requests": 0, "answer_generation_calls": 0, "legacy_27_loaded": False,
        "case_count": 64, "source_count": 80,
    }
    write(args.out_dir / "input-integrity-report.json", integrity)
    write(args.out_dir / "variant-manifest.json", {
        "artifact_schema": "nf-opt-03/v1", "variants": variants, "current_production_config": config,
        "current_limit_distribution": {str(v): list(current_limits.values()).count(v) for v in sorted(set(current_limits.values()))},
        "bm25_diagnostic_interface": "direct_read_only_fts5_for_limits_above_production_cap",
        "dense_configuration_identical": True, "rrf_configuration_identical": True,
        "reranker": config["reranker"], "reranker_input_source": "rrf_all", "final_k": config["final_k"],
    })
    write(args.out_dir / "bm25-prefix-integrity-report.json", {name: {"case_count": len(prefixes[name]), "failed_case_count": sum(not x["passed"] for x in prefixes[name]), "passed": all(x["passed"] for x in prefixes[name]), "cases": prefixes[name]} for name in variants})
    write(args.out_dir / "candidate-lineage-report.json", {name: {
        "case_count": len(lineages[name]),
        "lineage_failure_count": sum(not x["lineage_passed"] for x in lineages[name]),
        "rrf_dropped_candidate_count": sum(int(x["rrf_dropped_candidate_count"]) for x in lineages[name]),
        "rrf_drop_reason_counts": {"production_rrf_doc_id_deduplication": sum(x.get("rrf_drop_reason") == "production_rrf_doc_id_deduplication" for x in lineages[name])},
        "cases": lineages[name],
    } for name in variants})
    write(args.out_dir / "bm25-comparison.json", {name: {"window_limit": comparisons[name]["bm25_window_limit"], "window_metrics": comparisons[name]["bm25_window_metrics"], "rank_metrics": comparisons[name]["bm25_rank_metrics"]} for name in variants})
    write(args.out_dir / "union-comparison.json", {name: comparisons[name]["production_union"] for name in variants})
    write(args.out_dir / "rrf-comparison.json", {name: comparisons[name]["rrf"] for name in variants})
    write(args.out_dir / "reranker-transfer-comparison.json", {name: comparisons[name]["reranker"] for name in variants})
    write(args.out_dir / "final-transfer-comparison.json", {name: comparisons[name]["final"] for name in variants})
    write(args.out_dir / "regression-report.json", {name: comparisons[name]["regression"] for name in variants})
    write(args.out_dir / "latency-breakdown.json", {name: comparisons[name]["latency"] for name in variants})
    write(args.out_dir / "variant-selection.json", {"order": ["B80", "B120", "B200"], "selected_variant": selected, "selected_limit": None if selected is None else comparisons[selected]["bm25_window_limit"], "comparisons": {name: comparisons[name]["gate"] for name in variants}})
    write(args.out_dir / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    write(args.out_dir / "nf-opt-03-acceptance.json", acceptance)
    print(json.dumps({"acceptance": acceptance, "variants": {name: comparisons[name]["gate"] for name in variants}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (OSError, KeyError, ValueError) as exc:
        print(f"NF-OPT-03 failed: {exc}")
        raise SystemExit(2)
