"""NF-OPT-02: protected residual Dense candidate-window A/B.

This evaluation-only runner preserves the current Dense Top-40 as an exact
prefix, appends Canonical-minus-Current residual candidates, and evaluates
only read-only BM25/Dense/RRF retrieval.  It never invokes reranking, final
selection, an answer pipeline, or a model service.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import chromadb
import numpy as np

from scripts.evaluation import run_nf_eval_03_r1 as r1
from scripts.evaluation import run_nf_opt_01 as opt01
from src.evaluation.nf_opt_01 import candidate_scope_ok, coverage_state, percentile, rank_metrics
from src.evaluation.nf_opt_02 import (
    NFOpt02Error,
    base_retention,
    compare_hit_sets,
    protected_dense_merge,
    protected_residual_gate,
    residual_candidate_keys,
    select_smallest_passing_variant,
)
from src.retrieval.query_processor import QueryProcessor


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "financial_rag_v1"
DATA = BENCHMARK / "data"
DEFAULT_OUT = ROOT / "artifacts" / "evaluation" / "nf-opt-02"
DEFAULT_RUNTIME = ROOT / "runtime" / "evaluation" / "nf-opt-02" / "residual-chroma"
DEFAULT_NEGATIVE = ROOT / "artifacts" / "evaluation" / "nf-eval-02" / "negative-evidence-review-report.json"
NF04_OUT = ROOT / "artifacts" / "evaluation" / "nf-eval-04"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument("--chroma-path", type=Path, default=ROOT / "chroma_db")
    parser.add_argument("--bm25-db-path", type=Path, default=ROOT / "rag_bm25.db")
    parser.add_argument("--residual-path", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--corpus", type=Path, default=BENCHMARK / "corpus.json")
    parser.add_argument("--manifest", type=Path, default=DATA / "golden-manifest.json")
    parser.add_argument("--questions", type=Path, default=DATA / "questions.golden.jsonl")
    parser.add_argument("--labels", type=Path, default=DATA / "labels.golden.jsonl")
    parser.add_argument("--review-status", type=Path, default=DATA / "review-status.golden.jsonl")
    parser.add_argument("--negative-report", type=Path, default=DEFAULT_NEGATIVE)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    return parser.parse_args()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _identity_conflicts(
    current: Mapping[str, Mapping[str, Any]],
    canonical: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    conflicts: list[str] = []
    for key in current.keys() & canonical.keys():
        left, right = current[key], canonical[key]
        if (
            left.get("document_id") != right.get("document_id")
            or left.get("content_hash") != right.get("content_hash")
            or (left.get("metadata") or {}).get("page") != (right.get("metadata") or {}).get("page")
        ):
            conflicts.append(key)
    return sorted(conflicts)


def _build_residual_index(
    *,
    client: Any,
    records: Sequence[Mapping[str, Any]],
    embed_fn: Any,
) -> tuple[Any, dict[str, Any]]:
    """Build or exactly reuse the isolated Canonical-only residual collection."""

    started = time.perf_counter()
    name = "financial_rag_v1_dense_residual"
    target_ids = {str(row["candidate_key"]) for row in records}
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine", "nf_opt_02": "protected_residual"},
    )
    reused = False
    if int(collection.count()) == len(target_ids):
        existing = collection.get(limit=len(target_ids), include=["metadatas"])
        metadata = existing.get("metadatas") or []
        if (
            {str(value) for value in existing.get("ids") or []} == target_ids
            and all((item or {}).get("nf_opt_02_metadata_schema") == "v1" for item in metadata)
        ):
            reused = True
    if not reused and int(collection.count()) > 0:
        client.delete_collection(name=name)
        collection = client.get_or_create_collection(
            name=name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine", "nf_opt_02": "protected_residual"},
        )
    encoded = 0
    if not reused:
        for start in range(0, len(records), 256):
            batch = records[start : start + 256]
            embeddings = embed_fn([str(row["content"]) for row in batch])
            collection.upsert(
                ids=[str(row["candidate_key"]) for row in batch],
                documents=[str(row["content"]) for row in batch],
                metadatas=[dict(row["metadata"]) | {"nf_opt_02_metadata_schema": "v1", "dense_variant_source": "canonical_residual"} for row in batch],
                embeddings=[np.asarray(vector, dtype=np.float32).tolist() for vector in embeddings],
            )
            encoded += len(batch)
    return collection, {
        "collection_name": name,
        "index_reused": reused,
        "new_vectors_encoded_count": encoded,
        "index_build_ms": (time.perf_counter() - started) * 1000.0,
    }


def _rank_map(candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return opt01._rank_map(candidates)


def _gold_hit_set(rows: Sequence[Mapping[str, Any]], *, cutoff: int | None = None) -> set[str]:
    return {
        f"{row['case_id']}:{row['candidate_key']}"
        for row in rows
        if isinstance(row.get("rank"), int) and (cutoff is None or int(row["rank"]) <= cutoff)
    }


def _coverage(rows: Sequence[Mapping[str, Any]], *, cutoff: int | None = None) -> Counter[str]:
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    return Counter(
        coverage_state(
            [str(item["candidate_key"]) for item in items],
            [str(item["candidate_key"]) for item in items if isinstance(item.get("rank"), int) and (cutoff is None or int(item["rank"]) <= cutoff)],
        )
        for items in by_case.values()
    )


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return rank_metrics(rows, "rank", cutoffs=(20, 40, 100))


def _input_integrity(inputs: Any, args: argparse.Namespace) -> dict[str, Any]:
    actual = dict(inputs.hash_report["actual"])
    prior = json.loads((NF04_OUT / "input-integrity-report.json").read_text(encoding="utf-8"))
    fields = ("question_hash", "reference_answer_hash", "source_identity_hash", "negative_evidence_hash", "review_status_hash", "corpus_hash", "golden_manifest_sha256")
    unchanged = {field: actual.get(field) == prior.get(field) for field in fields}
    return {
        "artifact_schema": "nf-opt-02/v1",
        "benchmark_id": "financial-rag-v1",
        "tenant_id": args.tenant_id,
        "case_count": 64,
        "expected_source_count": 80,
        "allowed_document_count": 8,
        **{field: actual.get(field) for field in fields},
        "all_hashes_verified": all(inputs.hash_report["matches"].values()),
        "nf_eval_04_hashes_unchanged": all(unchanged.values()),
        "legacy_documents_loaded": 0,
    }


def _variant_summary(
    *,
    name: str,
    budget: int,
    source_rows: Sequence[Mapping[str, Any]],
    dense_rows: Sequence[Mapping[str, Any]],
    union_rows: Sequence[Mapping[str, Any]],
    rrf_rows: Sequence[Mapping[str, Any]],
    base_audits: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    dense_ratio: float | None,
    retrieval_ratio: float | None,
) -> dict[str, Any]:
    full_cov, top40_cov = _coverage(rrf_rows), _coverage(rrf_rows, cutoff=40)
    union_hits = _gold_hit_set(union_rows)
    rrf_hits = _gold_hit_set(rrf_rows)
    rrf_top40_hits = _gold_hit_set(rrf_rows, cutoff=40)
    dense_hits = _gold_hit_set(dense_rows)
    base_missing = sum(int(row["base_candidate_missing_count"]) for row in base_audits)
    base_order = sum(int(row["base_candidate_order_changed_count"]) for row in base_audits)
    gate = protected_residual_gate(
        case_count=64,
        source_count=80,
        overlap_count=int(baseline["overlap_count"]),
        identity_conflict_count=int(baseline["identity_conflict_count"]),
        out_of_scope_count=int(baseline["out_of_scope_count"]),
        base_missing_count=base_missing,
        base_order_changed_count=base_order,
        dense_gold_regressions=compare_hit_sets(baseline_hits=set(baseline["dense_hits"]), variant_hits=dense_hits)["regressed_hit_count"],
        union_source_regressions=compare_hit_sets(baseline_hits=set(baseline["union_hits"]), variant_hits=union_hits)["regressed_hit_count"],
        rrf_full_source_regressions=compare_hit_sets(baseline_hits=set(baseline["rrf_hits"]), variant_hits=rrf_hits)["regressed_hit_count"],
        rrf_top40_source_regressions=compare_hit_sets(baseline_hits=set(baseline["rrf_top40_hits"]), variant_hits=rrf_top40_hits)["regressed_hit_count"],
        rrf_full_all_gold_regressions=sum(row["coverage"] == "all" and item != "all" for row, item in zip(baseline["full_coverage"], _case_coverage(rrf_rows))),
        rrf_top40_all_gold_regressions=sum(row["coverage"] == "all" and item != "all" for row, item in zip(baseline["top40_coverage"], _case_coverage(rrf_rows, cutoff=40))),
        union_source_hits=len(union_hits),
        rrf_full_source_hits=len(rrf_hits),
        rrf_top40_source_hits=len(rrf_top40_hits),
        rrf_top40_all_gold_cases=int(top40_cov["all"]),
        dense_latency_ratio=dense_ratio,
        retrieval_latency_ratio=retrieval_ratio,
    )
    return {
        "variant": name,
        "residual_budget": budget,
        "dense_metrics": _metrics(dense_rows),
        "production_union_source_recall": {"source_hit_count": len(union_hits), "source_recall": len(union_hits) / 80},
        "rrf_metrics": _metrics(rrf_rows),
        "rrf_full_pool_coverage": dict(full_cov),
        "rrf_top40_coverage": dict(top40_cov),
        "base_retention": {"base_candidate_retention_count": 64 * 40 - base_missing, "base_candidate_missing_count": base_missing, "base_candidate_order_changed_count": base_order},
        "regression": {
            "dense": compare_hit_sets(baseline_hits=set(baseline["dense_hits"]), variant_hits=dense_hits),
            "union": compare_hit_sets(baseline_hits=set(baseline["union_hits"]), variant_hits=union_hits),
            "rrf_full": compare_hit_sets(baseline_hits=set(baseline["rrf_hits"]), variant_hits=rrf_hits),
            "rrf_top40": compare_hit_sets(baseline_hits=set(baseline["rrf_top40_hits"]), variant_hits=rrf_top40_hits),
        },
        "gate": gate,
    }


def _case_coverage(rows: Sequence[Mapping[str, Any]], *, cutoff: int | None = None) -> list[dict[str, str]]:
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    return [
        {
            "case_id": case_id,
            "coverage": coverage_state(
                [str(item["candidate_key"]) for item in items],
                [str(item["candidate_key"]) for item in items if isinstance(item.get("rank"), int) and (cutoff is None or int(item["rank"]) <= cutoff)],
            ),
        }
        for case_id, items in sorted(by_case.items())
    ]


def _run(args: argparse.Namespace) -> int:
    if args.tenant_id != 1 or args.embedding_model != "all-MiniLM-L6-v2":
        raise NFOpt02Error("NF-OPT-02 is fixed to tenant 1 and all-MiniLM-L6-v2")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.residual_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    inputs = r1._load_inputs(corpus_path=args.corpus, manifest_path=args.manifest, questions_path=args.questions, labels_path=args.labels, review_status_path=args.review_status, negative_report_path=args.negative_report)
    integrity = _input_integrity(inputs, args)
    if not integrity["all_hashes_verified"] or not integrity["nf_eval_04_hashes_unchanged"]:
        raise NFOpt02Error("frozen benchmark input integrity failed")
    mapping, filenames = r1._doc_map(inputs.corpus), [str(item["filename"]) for item in inputs.corpus["documents"]]
    allowed = {str(item["document_id"]) for item in inputs.corpus["documents"]}
    source_rows, gold_keys = opt01._load_gold_keys(inputs.labels_by_id)
    if len(inputs.questions) != 72 or len(source_rows) != 80:
        raise NFOpt02Error("expected frozen 72-question / 80-source input")

    from src.services import vector_store
    from src.services.retrieval import SqliteBM25Retriever

    client = chromadb.PersistentClient(path=str(args.chroma_path))
    current_collection = client.get_collection(name=vector_store.GLOBAL_COLLECTION_NAME, embedding_function=vector_store.embed_fn)
    canonical_records, canonical_manifest = opt01._load_candidate_universe(db_path=args.bm25_db_path, corpus=inputs.corpus, mapping=mapping, tenant_id=args.tenant_id, gold_keys=gold_keys)
    current_records, current_manifest = opt01._load_current_dense_records(current_collection, filenames=filenames, mapping=mapping, tenant_id=args.tenant_id)
    canonical_by_key = {str(row["candidate_key"]): row for row in canonical_records}
    current_by_key = {str(row["candidate_key"]): row for row in current_records}
    residual_keys = residual_candidate_keys(canonical_keys=set(canonical_by_key), current_keys=set(current_by_key))
    conflicts = _identity_conflicts(current_by_key, canonical_by_key)
    if conflicts:
        raise NFOpt02Error(f"candidate identity conflicts: {len(conflicts)}")
    residual_records = [canonical_by_key[key] for key in sorted(residual_keys)]
    residual_client = chromadb.PersistentClient(path=str(args.residual_path))
    residual_collection, residual_build = _build_residual_index(client=residual_client, records=residual_records, embed_fn=vector_store.embed_fn)
    residual_manifest = {
        "artifact_schema": "nf-opt-02/v1", "variant": "canonical_only_residual", "embedding_model": args.embedding_model,
        "current_candidate_count": len(current_records), "canonical_candidate_count": len(canonical_records), "residual_candidate_count": len(residual_records),
        "current_residual_overlap_count": len(set(current_by_key) & residual_keys), "identity_conflict_count": len(conflicts), "duplicate_candidate_count": len(residual_records) - len(residual_keys),
        "out_of_scope_candidate_count": sum(not candidate_scope_ok(row["document_id"], allowed) for row in residual_records),
        "gold_identity_presence_count": sum(key in canonical_by_key for key in gold_keys), "gold_identity_expected_count": len(gold_keys),
        "gold_labels_not_used_to_build_universe": True, **residual_build,
    }
    if residual_manifest["out_of_scope_candidate_count"] or residual_manifest["current_residual_overlap_count"]:
        raise NFOpt02Error("residual universe violates scope or disjointness")

    processor, bm25 = QueryProcessor(), SqliteBM25Retriever(db_path=str(args.bm25_db_path))
    variants = {"A": 0, "C10": 10, "C20": 20, "C40": 40}
    rows = {name: {"dense": [], "union": [], "rrf": [], "audit": []} for name in variants}
    dense_times = {name: [] for name in variants}
    retrieval_times = {name: [] for name in variants}
    scope_out_of_scope = 0
    for question in inputs.questions:
        case_id = str(question["case_id"])
        label = inputs.labels_by_id[case_id]
        if label.get("expected_no_answer"):
            continue
        scope = [str(value) for value in question.get("document_scope") or []]
        if len(scope) != 1 or scope[0] not in allowed:
            raise NFOpt02Error(f"{case_id}: invalid document scope")
        filename = next(str(item["filename"]) for item in inputs.corpus["documents"] if str(item["document_id"]) == scope[0])
        query = str(question["question"])
        query_embedding = np.asarray(vector_store.embed_fn([processor.expand(query)])[0], dtype=np.float32)
        start = time.perf_counter()
        base = opt01._query_dense(
            current_collection,
            query_embedding=query_embedding,
            filename=filename,
            tenant_id=args.tenant_id,
            limit=40,
            mapping=mapping,
        )
        base_elapsed = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        residual = opt01._query_dense(
            residual_collection,
            query_embedding=query_embedding,
            filename=filename,
            tenant_id=args.tenant_id,
            limit=40,
            mapping=mapping,
        )
        residual_elapsed = (time.perf_counter() - start) * 1000.0
        candidate_k = 40 if processor.is_numeric_query(query) else 20
        bm25_rows = [opt01._annotated_candidate(row, mapping=mapping, tenant_id=args.tenant_id) for row in bm25.search(processor.expand(query), k=candidate_k, doc_name=filename, user_id=args.tenant_id)]
        bm25_rows = [row for row in bm25_rows if candidate_scope_ok(row.get("document_id"), allowed)]
        case_sources = [row for row in source_rows if row["case_id"] == case_id]
        for name, budget in variants.items():
            started = time.perf_counter()
            dense = protected_dense_merge(base_candidates=base, residual_candidates=residual[:budget])
            audit = base_retention(base_candidates=base, protected_candidates=dense) | {"case_id": case_id}
            union = opt01._union_candidates(dense, bm25_rows, mapping=mapping, tenant_id=args.tenant_id)
            fused = opt01._rrf_candidates(dense, bm25_rows, query=query, query_processor=processor, mapping=mapping, tenant_id=args.tenant_id)
            elapsed = (time.perf_counter() - started) * 1000.0
            dense_times[name].append(base_elapsed if budget == 0 else base_elapsed + residual_elapsed)
            retrieval_times[name].append(elapsed + base_elapsed + (0 if budget == 0 else residual_elapsed))
            for candidate in [*dense, *union, *fused]:
                if not candidate_scope_ok(candidate.get("document_id"), allowed):
                    scope_out_of_scope += 1
            dense_map, union_map, rrf_map = _rank_map(dense), _rank_map(union), _rank_map(fused)
            rows[name]["audit"].append(audit)
            for source in case_sources:
                common = {"case_id": case_id, "source_index": source["source_index"], "candidate_key": source["candidate_key"]}
                rows[name]["dense"].append(common | {"rank": dense_map.get(source["candidate_key"])})
                rows[name]["union"].append(common | {"rank": union_map.get(source["candidate_key"])})
                rows[name]["rrf"].append(common | {"rank": rrf_map.get(source["candidate_key"])})

    base = {
        "overlap_count": residual_manifest["current_residual_overlap_count"], "identity_conflict_count": len(conflicts), "out_of_scope_count": scope_out_of_scope,
        "dense_hits": _gold_hit_set(rows["A"]["dense"]), "union_hits": _gold_hit_set(rows["A"]["union"]), "rrf_hits": _gold_hit_set(rows["A"]["rrf"]), "rrf_top40_hits": _gold_hit_set(rows["A"]["rrf"], cutoff=40),
        "full_coverage": _case_coverage(rows["A"]["rrf"]), "top40_coverage": _case_coverage(rows["A"]["rrf"], cutoff=40),
    }
    current_dense_p95, current_retrieval_p95 = percentile(dense_times["A"], .95) or 0.0, percentile(retrieval_times["A"], .95) or 0.0
    summaries: dict[str, dict[str, Any]] = {}
    for name, budget in variants.items():
        dense_p95, retrieval_p95 = percentile(dense_times[name], .95) or 0.0, percentile(retrieval_times[name], .95) or 0.0
        summaries[name] = _variant_summary(name=name, budget=budget, source_rows=source_rows, dense_rows=rows[name]["dense"], union_rows=rows[name]["union"], rrf_rows=rows[name]["rrf"], base_audits=rows[name]["audit"], baseline=base, dense_ratio=((dense_p95-current_dense_p95)/current_dense_p95 if current_dense_p95 else None), retrieval_ratio=((retrieval_p95-current_retrieval_p95)/current_retrieval_p95 if current_retrieval_p95 else None))
    selected = select_smallest_passing_variant({name: summary["gate"] for name, summary in summaries.items() if name != "A"})
    acceptance = {
        "artifact_schema": "nf-opt-02/v1", "decision": ("protected_residual_dense_validated" if selected else "protected_residual_dense_not_validated"),
        "selected_variant": selected, "selected_residual_budget": variants[selected] if selected else None, "input_hashes_verified": True, "scope_integrity_passed": scope_out_of_scope == 0,
        "production_behavior_changed": False, "production_switch_allowed": False, "model_chat_completion_requests": 0, "answer_generation_calls": 0, "reranker_calls": 0, "legacy_27_loaded": False,
    }
    _write(args.out_dir / "input-integrity-report.json", integrity)
    _write(args.out_dir / "residual-index-manifest.json", residual_manifest)
    _write(args.out_dir / "base-retention-report.json", {name: summary["base_retention"] for name, summary in summaries.items()})
    _write(args.out_dir / "protected-dense-comparison.json", {name: summary["dense_metrics"] for name, summary in summaries.items()})
    _write(args.out_dir / "production-union-comparison.json", {name: summary["production_union_source_recall"] for name, summary in summaries.items()})
    _write(args.out_dir / "rrf-comparison.json", {name: {"metrics": summary["rrf_metrics"], "full_pool_coverage": summary["rrf_full_pool_coverage"], "top40_coverage": summary["rrf_top40_coverage"]} for name, summary in summaries.items()})
    _write(args.out_dir / "regression-report.json", {name: summary["regression"] for name, summary in summaries.items()})
    _write(args.out_dir / "latency-report.json", {name: {"dense_p50_ms": percentile(dense_times[name], .5), "dense_p95_ms": percentile(dense_times[name], .95), "retrieval_p50_ms": percentile(retrieval_times[name], .5), "retrieval_p95_ms": percentile(retrieval_times[name], .95)} for name in variants})
    _write(args.out_dir / "variant-selection.json", {"order": ["C10", "C20", "C40"], "selected_variant": selected, "variants": {name: summary["gate"] for name, summary in summaries.items() if name != "A"}})
    _write(args.out_dir / "next-gate.json", {"selected_gate": "production_config_shadow_validation" if selected else "protected_residual_candidate_ab", "production_switch_allowed": False, "optimization_allowed": bool(selected)})
    _write(args.out_dir / "nf-opt-02-acceptance.json", acceptance)
    print(json.dumps({"acceptance": acceptance, "variants": {name: summary["gate"] for name, summary in summaries.items()}}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    try:
        return _run(_parse_args())
    except (NFOpt02Error, ValueError, KeyError, OSError) as error:
        print(f"NF-OPT-02 failed: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
