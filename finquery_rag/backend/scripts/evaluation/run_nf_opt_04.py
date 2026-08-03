"""NF-OPT-04 read-only Final Evidence Budget A/B."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import chromadb
import numpy as np
import tiktoken

from scripts.evaluation import run_nf_eval_03_r1 as eval_r1
from scripts.evaluation import run_nf_opt_01 as opt01
from scripts.evaluation import run_nf_opt_03 as opt03
from src.evaluation.nf_opt_01 import percentile
from src.evaluation.nf_opt_02_r1 import source_rows
from src.evaluation.nf_opt_04 import (
    all_gold_cases, context_quality, coverage, final_budget_decision,
    final_budget_gate, hit_set, prefix_report, rank_bucket, rank_distribution, select_prefix,
    select_smallest_passing_variant, select_token_budget,
)
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.query_processor import QueryProcessor
from src.services import vector_store
from src.services.reranker import build_reranker
from src.services.retrieval import SqliteBM25Retriever

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "financial_rag_v1"
DATA = BENCHMARK / "data"
OUT = ROOT / "artifacts" / "evaluation" / "nf-opt-04"
NF04 = ROOT / "artifacts" / "evaluation" / "nf-eval-04"
NEGATIVE = ROOT / "artifacts" / "evaluation" / "nf-eval-02" / "negative-evidence-review-report.json"
LEGACY_FILES = {"FINAL Annual Report.pdf", "leac203.pdf", "wipo_pub_rn2021_18e.pdf"}
TOKENIZER_SCHEMA = "tiktoken/cl100k_base"


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


def input_integrity(inputs: Any, args: argparse.Namespace) -> dict[str, Any]:
    actual = dict(inputs.hash_report["actual"])
    expected = json.loads((NF04 / "input-integrity-report.json").read_text(encoding="utf-8"))
    keys = ("question_hash", "reference_answer_hash", "source_identity_hash", "negative_evidence_hash", "review_status_hash", "corpus_hash", "golden_manifest_sha256")
    actual["benchmark_hash"] = hashlib.sha256(json.dumps({key: actual.get(key) for key in keys}, sort_keys=True).encode("utf-8")).hexdigest()
    fields = (*keys, "benchmark_hash")
    return {
        "artifact_schema": "nf-opt-04/v1",
        "benchmark_id": "financial-rag-v1",
        "tenant_id": args.tenant_id,
        "case_count": 64,
        "expected_source_count": 80,
        "allowed_document_count": 8,
        **{field: actual.get(field) for field in fields},
        "all_hashes_recomputed_and_verified": all(inputs.hash_report["matches"].values()),
        "nf_eval_04_hashes_unchanged": all(actual.get(field) == expected.get(field) for field in fields),
        "legacy_documents_loaded": 0,
    }


def context_payload(candidates: list[dict[str, Any]], tokenizer: Any) -> tuple[int, int]:
    builder = ContextBuilder(max_context_tokens=1_000_000, tokenizer=tokenizer)
    context, _ = builder.build([dict(candidate) for candidate in candidates])
    return len(tokenizer.encode(context)), len(builder.last_context_evidence)


def context_record(case_id: str, candidates: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    tokens, rendered = context_payload(candidates, tokenizer)
    return {
        "case_id": case_id,
        "selected_candidate_count": len(candidates),
        "rendered_evidence_count": rendered,
        "context_token_count": tokens,
        **context_quality(candidates),
    }


def latency_report(values: dict[str, list[float]]) -> dict[str, Any]:
    return {
        name: {
            "mean_ms": sum(items) / len(items) if items else 0.0,
            "p50_ms": percentile(items, 0.5),
            "p95_ms": percentile(items, 0.95),
            "measured_run_count": len(items),
        }
        for name, items in values.items()
    }


def quality_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    def rate(field: str) -> float:
        return sum(int(int(record[field]) > 0) for record in records) / total if total else 0.0
    def pct(field: str, value: float) -> float | None:
        return percentile([float(record[field]) for record in records], value)
    return {
        "case_count": total,
        "evidence_count_mean": sum(record["selected_candidate_count"] for record in records) / total if total else 0.0,
        "evidence_count_p50": pct("selected_candidate_count", 0.5),
        "evidence_count_p95": pct("selected_candidate_count", 0.95),
        "context_tokens_mean": sum(record["context_token_count"] for record in records) / total if total else 0.0,
        "context_tokens_p50": pct("context_token_count", 0.5),
        "context_tokens_p95": pct("context_token_count", 0.95),
        "duplicate_evidence_case_rate": rate("duplicate_evidence_count"),
        "same_page_duplicate_case_rate": rate("same_page_duplicate_count"),
        "same_metric_duplicate_case_rate": rate("same_metric_duplicate_count"),
        "conflicting_period_case_count": sum(int(int(record["conflicting_period_count"]) > 0) for record in records),
        "conflicting_value_case_count": sum(int(int(record["conflicting_value_count"]) > 0) for record in records),
    }


def diff(before: set[str], after: set[str]) -> dict[str, int]:
    return {"baseline_count": len(before), "variant_count": len(after), "new_count": len(after - before), "regressed_count": len(before - after), "both_count": len(before & after)}


def run(args: argparse.Namespace) -> int:
    if args.tenant_id != 1:
        raise ValueError("tenant 1 is required")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    inputs = eval_r1._load_inputs(
        corpus_path=args.corpus,
        manifest_path=args.manifest,
        questions_path=args.questions,
        labels_path=args.labels,
        review_status_path=args.review_status,
        negative_report_path=args.negative_report,
    )
    integrity = input_integrity(inputs, args)
    if not integrity["all_hashes_recomputed_and_verified"] or not integrity["nf_eval_04_hashes_unchanged"]:
        raise ValueError("frozen benchmark input integrity failed")
    config = opt03.production_config()
    mapping = eval_r1._doc_map(inputs.corpus)
    allowed = {str(item["document_id"]) for item in inputs.corpus["documents"]}
    if LEGACY_FILES & {str(item.get("filename")) for item in inputs.corpus["documents"]}:
        raise ValueError("legacy document entered benchmark corpus")
    sources, _ = opt01._load_gold_keys(inputs.labels_by_id)
    answerable = [question for question in inputs.questions if not inputs.labels_by_id[str(question["case_id"])].get("expected_no_answer")]
    if len(answerable) != 64 or len(sources) != 80:
        raise ValueError("expected 64 answerable cases and 80 sources")
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        raise ValueError("current production tokenizer is unavailable") from exc
    processor = QueryProcessor()
    bm25 = SqliteBM25Retriever(db_path=str(args.bm25_db_path))
    client = chromadb.PersistentClient(path=str(args.chroma_path))
    dense_collection = client.get_collection(name=vector_store.GLOBAL_COLLECTION_NAME, embedding_function=vector_store.embed_fn)
    reranker = build_reranker(config["reranker"])
    if reranker is None:
        raise ValueError("production reranker unavailable")

    variants = {
        "F5": {"max_evidence": 5, "token_budget": None},
        "F8": {"max_evidence": 8, "token_budget": None},
        "F10": {"max_evidence": 10, "token_budget": None},
        "FT8": {"max_evidence": 8, "token_budget": 6000},
        "FT10": {"max_evidence": 10, "token_budget": 8000},
    }
    rows = {name: [] for name in variants}
    prefix_cases = {name: [] for name in variants if name != "F5"}
    lineages = {name: [] for name in variants}
    context_cases = {name: [] for name in variants}
    latency = {name: defaultdict(list) for name in variants}
    rank_rows: list[dict[str, Any]] = []

    for question in answerable:
        case_id = str(question["case_id"])
        scope = [str(value) for value in question.get("document_scope") or []]
        if len(scope) != 1 or scope[0] not in allowed:
            raise ValueError(f"{case_id}: out-of-scope question")
        filename = next(str(item["filename"]) for item in inputs.corpus["documents"] if str(item["document_id"]) == scope[0])
        query = str(question["question"])
        current_limit = opt03.current_limit(query, processor, config)
        started = time.perf_counter()
        embedding = np.asarray(vector_store.embed_fn([processor.expand(query)])[0], dtype=np.float32)
        embedding_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        dense_raw = opt01._query_dense(
            dense_collection,
            query_embedding=embedding,
            filename=filename,
            tenant_id=args.tenant_id,
            limit=current_limit,
            mapping=mapping,
        )
        dense, _ = opt03.annotate(dense_raw, mapping, args.tenant_id, allowed)
        dense_ms = (time.perf_counter() - started) * 1000.0
        sparse, bm25_ms, normalization_ms, _ = opt03.query_bm25(
            bm25,
            query=query,
            filename=filename,
            limit=current_limit,
            current=current_limit,
            processor=processor,
            mapping=mapping,
            tenant_id=args.tenant_id,
            allowed=allowed,
        )
        started = time.perf_counter()
        union = opt01._union_candidates(dense, sparse, mapping=mapping, tenant_id=args.tenant_id)
        union_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        fused = opt01._rrf_candidates(
            dense,
            sparse,
            query=query,
            query_processor=processor,
            mapping=mapping,
            tenant_id=args.tenant_id,
        )
        rrf_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        reranked = reranker.rerank(query, fused, top_k=min(20, len(fused)))
        reranker_ms = (time.perf_counter() - started) * 1000.0
        expected = [source for source in sources if source["case_id"] == case_id]
        rank_rows.extend(source_rows(case_id=case_id, expected_sources=expected, stage_candidates={"reranker": reranked}))
        selections: dict[str, list[dict[str, Any]]] = {}
        context_payload_cache: dict[tuple[str, ...], tuple[int, int]] = {}

        def cached_context_payload(values: list[dict[str, Any]]) -> tuple[int, int]:
            key = tuple(str(value["candidate_key"]) for value in values)
            if key not in context_payload_cache:
                context_payload_cache[key] = context_payload(values, tokenizer)
            return context_payload_cache[key]

        def cached_context_tokens(values: list[dict[str, Any]]) -> int:
            return cached_context_payload(values)[0]

        for name, settings in variants.items():
            select_started = time.perf_counter()
            if settings["token_budget"] is None:
                selected = select_prefix(reranked, max_evidence=int(settings["max_evidence"]))
                token_ms = 0.0
            else:
                token_started = time.perf_counter()
                selected = select_token_budget(
                    reranked,
                    max_evidence=int(settings["max_evidence"]),
                    token_budget=int(settings["token_budget"]),
                    count_context_tokens=cached_context_tokens,
                )
                token_ms = (time.perf_counter() - token_started) * 1000.0
            selection_ms = (time.perf_counter() - select_started) * 1000.0
            serialization_started = time.perf_counter()
            tokens, rendered = cached_context_payload(selected)
            record = {
                "case_id": case_id,
                "selected_candidate_count": len(selected),
                "rendered_evidence_count": rendered,
                "context_token_count": tokens,
                **context_quality(selected),
            }
            serialization_ms = (time.perf_counter() - serialization_started) * 1000.0
            selections[name] = selected
            context_cases[name].append(record)
            rows[name].extend(source_rows(case_id=case_id, expected_sources=expected, stage_candidates={"reranker": reranked, "final": selected}))
            lineages[name].append(opt03.lineage({"dense": dense, "bm25": sparse, "union": union, "rrf": fused, "reranker": reranked, "final": selected}, allowed))
            if name != "F5":
                prefix_cases[name].append({"case_id": case_id, **prefix_report(selections["F5"], selected)})
            measured = {
                "query_embedding_ms": embedding_ms,
                "dense_ms": dense_ms,
                "bm25_ms": bm25_ms,
                "normalization_ms": normalization_ms,
                "union_ms": union_ms,
                "rrf_ms": rrf_ms,
                "reranker_ms": reranker_ms,
                "final_selection_ms": selection_ms,
                "context_serialization_ms": serialization_ms,
                "token_counting_ms": token_ms,
            }
            measured["total_retrieval_ms"] = sum(measured.values())
            for field, value in measured.items():
                latency[name][field].append(value)


    base_rows = rows["F5"]
    base_sources = hit_set(base_rows, stage="final")
    base_all = all_gold_cases(base_rows, stage="final")
    multi_cases = {
        str(row["case_id"])
        for row in sources
        if sum(other["case_id"] == row["case_id"] for other in sources) > 1
    }
    base_multi = base_all & multi_cases
    base_quality = quality_report(context_cases["F5"])
    comparisons: dict[str, Any] = {}
    for name in variants:
        current_rows = rows[name]
        final_sources = hit_set(current_rows, stage="final")
        final_all = all_gold_cases(current_rows, stage="final")
        final_multi = final_all & multi_cases
        quality = quality_report(context_cases[name])
        source_diff = diff(base_sources, final_sources)
        all_diff = diff(base_all, final_all)
        multi_diff = diff(base_multi, final_multi)
        base_p95 = percentile(latency["F5"]["total_retrieval_ms"], 0.95) or 0.0
        current_p95 = percentile(latency[name]["total_retrieval_ms"], 0.95) or 0.0
        latency_ratio = (current_p95 - base_p95) / base_p95 if base_p95 else None
        lineage_ok = all(item["lineage_passed"] for item in lineages[name])
        prefix_ok = name == "F5" or all(item["passed"] for item in prefix_cases[name])
        gate = final_budget_gate(
            integrity_passed=len(current_rows) == 80 and lineage_ok and prefix_ok,
            source_hit_count=len(final_sources),
            all_gold_case_count=len(final_all),
            multi_evidence_all_gold_count=len(final_multi),
            new_source_count=source_diff["new_count"],
            new_all_gold_case_count=all_diff["new_count"],
            source_regression_count=source_diff["regressed_count"],
            all_gold_regression_count=all_diff["regressed_count"],
            multi_evidence_regression_count=multi_diff["regressed_count"],
            conflicting_period_case_increase=quality["conflicting_period_case_count"] - base_quality["conflicting_period_case_count"],
            conflicting_value_case_increase=quality["conflicting_value_case_count"] - base_quality["conflicting_value_case_count"],
            duplicate_case_rate=float(quality["duplicate_evidence_case_rate"]),
            context_token_p95=float(quality["context_tokens_p95"] or 0.0),
            total_latency_ratio=latency_ratio,
        )
        comparisons[name] = {
            "final": coverage(current_rows, stage="final"),
            "multi_evidence_all_source_coverage": len(final_multi),
            "regression": {"source": source_diff, "all_gold_case": all_diff, "multi_evidence_all_source": multi_diff},
            "context_quality": quality,
            "latency": latency_report(latency[name]),
            "lineage": {
                "case_count": len(lineages[name]),
                "lineage_failure_count": sum(not item["lineage_passed"] for item in lineages[name]),
                "missing_identity_count": sum(int(item["missing_identity_count"]) for item in lineages[name]),
                "out_of_scope_count": sum(int(item["out_of_scope_candidate_count"]) for item in lineages[name]),
                "candidate_injection_count": sum(int(item["reranker_candidate_injection_count"]) + int(item["final_candidate_injection_count"]) for item in lineages[name]),
            },
            "gate": gate,
        }

    selected = select_smallest_passing_variant({name: comparisons[name]["gate"] for name in variants})
    decision, next_gate = final_budget_decision(
        selected_variant=selected,
        context_quality_blocked=any(
            not comparisons[name]["gate"]["context_quality_passed"]
            for name in variants
            if name != "F5"
        ),
    )
    acceptance = {
        "artifact_schema": "nf-opt-04/v1",
        "decision": decision,
        "selected_variant": selected,
        "production_switch_allowed": False,
        "production_behavior_changed": False,
        "input_hashes_verified": integrity["all_hashes_recomputed_and_verified"] and integrity["nf_eval_04_hashes_unchanged"],
        "candidate_lineage_passed": all(comparisons[name]["lineage"]["lineage_failure_count"] == 0 for name in variants),
        "scope_integrity_passed": all(comparisons[name]["lineage"]["out_of_scope_count"] == 0 for name in variants),
        "model_chat_completion_requests": 0,
        "answer_generation_calls": 0,
        "legacy_27_loaded": False,
        "case_count": 64,
        "source_count": 80,
    }
    base_final_sources = hit_set(rows["F5"], stage="final")
    origin: dict[str, dict[str, int]] = {}
    for name in variants:
        new_ids = hit_set(rows[name], stage="final") - base_final_sources
        counts = Counter()
        for row in rows[name]:
            if f"{row['case_id']}:{row['source_index']}" in new_ids:
                counts[rank_bucket(row.get("reranker_rank"))] += 1
        origin[name] = {key: int(counts.get(key, 0)) for key in ("1_5", "6_8", "9_10", "11_20", "below_20")}


    write(args.out_dir / "input-integrity-report.json", integrity)
    write(args.out_dir / "variant-manifest.json", {
        "artifact_schema": "nf-opt-04/v1",
        "variants": variants,
        "token_counter": TOKENIZER_SCHEMA,
        "current_production_config": config,
        "reranker_input_source": "rrf_all",
        "gold_labels_not_used_for_selection": True,
        "answer_generation_called": False,
    })
    write(args.out_dir / "final-prefix-integrity-report.json", {
        name: {
            "case_count": len(prefix_cases[name]),
            "failed_case_count": sum(not record["passed"] for record in prefix_cases[name]),
            "passed": all(record["passed"] for record in prefix_cases[name]),
            "cases": prefix_cases[name],
        }
        for name in prefix_cases
    })
    write(args.out_dir / "reranker-rank-distribution.json", {
        "gold_source_rank_distribution": rank_distribution(rank_rows),
        "new_final_source_origin_by_variant": origin,
    })
    write(args.out_dir / "final-source-comparison.json", {name: comparisons[name]["final"] for name in variants})
    write(args.out_dir / "multi-evidence-comparison.json", {name: {"all_source_coverage": comparisons[name]["multi_evidence_all_source_coverage"]} for name in variants})
    write(args.out_dir / "context-quality-report.json", {name: comparisons[name]["context_quality"] for name in variants})
    write(args.out_dir / "regression-report.json", {name: comparisons[name]["regression"] for name in variants})
    write(args.out_dir / "latency-breakdown.json", {name: comparisons[name]["latency"] for name in variants})
    write(args.out_dir / "variant-selection.json", {
        "order": ["F8", "FT8", "F10", "FT10"],
        "selected_variant": selected,
        "comparisons": {name: comparisons[name]["gate"] for name in variants},
    })
    write(args.out_dir / "next-gate.json", {
        "decision": decision,
        "next_gate": next_gate,
        "production_switch_allowed": False,
    })
    write(args.out_dir / "nf-opt-04-acceptance.json", acceptance)
    print(json.dumps({"acceptance": acceptance, "variants": {name: comparisons[name]["gate"] for name in variants}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (KeyError, OSError, ValueError) as exc:
        print(f"NF-OPT-04 failed: {exc}")
        raise SystemExit(2)
