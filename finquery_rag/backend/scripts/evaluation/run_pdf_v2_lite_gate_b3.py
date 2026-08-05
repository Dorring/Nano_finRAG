"""Run production-shaped BM25/Dense/RRF/Reranker shadows for V2-Lite views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from scripts.evaluation.run_pdf_retrieval_v2_lite import DEFAULT_OUT as B12_OUT, _bm25, _write
from src.services.reranker import HeuristicReranker
from src.services.retrieval_config import get_embedding_model_name

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-representation-v2-lite-gate-b3"
VARIANTS = (
    ("baseline_raw_bm25_raw_dense", "raw_row_text", "raw_row_text"),
    ("a_enriched_bm25_raw_dense", "enriched_retrieval_text", "raw_row_text"),
    ("b_raw_bm25_enriched_dense", "raw_row_text", "enriched_retrieval_text"),
    ("c_enriched_bm25_enriched_dense", "enriched_retrieval_text", "enriched_retrieval_text"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank_dense(query_vector: Any, document_vectors: Any) -> list[int]:
    import numpy as np

    scores = document_vectors @ query_vector
    return [int(index) for index in np.argsort(-scores, kind="stable")]


def _rrf(dense: list[int], sparse: list[int], *, cutoff: int = 40, k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in (dense[:cutoff], sparse[:cutoff]):
        for rank, index in enumerate(ranked, start=1):
            scores[index] = scores.get(index, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _stage_hits(ranked: list[int], gold_index: int, cutoffs: tuple[int, ...]) -> dict[int, bool]:
    return {cutoff: gold_index in ranked[:cutoff] for cutoff in cutoffs}


def _run_variant(
    *,
    name: str,
    bm25_field: str,
    dense_field: str,
    views: list[dict[str, object]],
    cases: list[dict[str, object]],
    embeddings: dict[str, Any],
    query_vectors: Any,
    reranker_text_field: str,
    reranker_queries: list[str] | None = None,
) -> dict[str, object]:
    keys = [str(view["candidate_key"]) for view in views]
    key_to_index = {key: index for index, key in enumerate(keys)}
    documents = [str(view[bm25_field]) for view in views]
    reranker = HeuristicReranker()
    stage_counts = {"bm25_5": 0, "bm25_40": 0, "bm25_200": 0, "dense_5": 0, "dense_40": 0, "dense_200": 0, "union_40": 0, "union_200": 0, "rrf_40": 0, "reranker_20": 0, "final_5": 0}
    final_hits = []
    traces = []
    for case_index, case in enumerate(cases):
        gold_index = key_to_index[str(case["gold_candidate_key"])]
        bm25 = _bm25(str(case["query"]), documents)
        dense = _rank_dense(query_vectors[case_index], embeddings[dense_field])
        for cutoff, hit in _stage_hits(bm25, gold_index, (5, 40, 200)).items():
            stage_counts[f"bm25_{cutoff}"] += int(hit)
        for cutoff, hit in _stage_hits(dense, gold_index, (5, 40, 200)).items():
            stage_counts[f"dense_{cutoff}"] += int(hit)
        stage_counts["union_40"] += int(gold_index in set(bm25[:40]) | set(dense[:40]))
        stage_counts["union_200"] += int(gold_index in set(bm25[:200]) | set(dense[:200]))
        fused = _rrf(dense, bm25)
        fused_indices = [index for index, _ in fused]
        stage_counts["rrf_40"] += int(gold_index in fused_indices[:40])
        chunks = [
            {
                "doc_id": keys[index],
                "content": str(views[index][reranker_text_field]),
                "score": score,
                "metadata": {
                    "doc_name": views[index]["document_id"],
                    "page": views[index]["pdf_page"],
                    "row_label": views[index]["metric"],
                    "section_path": views[index]["statement_or_section"],
                },
            }
            for index, score in fused
        ]
        reranker_query = reranker_queries[case_index] if reranker_queries is not None else str(case["query"])
        reranked = reranker.rerank(reranker_query, chunks, top_k=20)
        reranked_keys = [item["doc_id"] for item in reranked]
        final = reranker.rerank(reranker_query, chunks, top_k=5)
        final_keys = [item["doc_id"] for item in final]
        stage_counts["reranker_20"] += int(str(case["gold_candidate_key"]) in reranked_keys)
        final_hit = str(case["gold_candidate_key"]) in final_keys
        stage_counts["final_5"] += int(final_hit)
        if final_hit:
            final_hits.append(str(case["gold_candidate_key"]))
        traces.append({"case_id": case["case_id"], "bm25_rank": bm25.index(gold_index) + 1, "dense_rank": dense.index(gold_index) + 1, "rrf_rank": fused_indices.index(gold_index) + 1 if gold_index in fused_indices else None, "reranker_rank": reranked_keys.index(str(case["gold_candidate_key"])) + 1 if str(case["gold_candidate_key"]) in reranked_keys else None, "final_hit": final_hit, "final_candidate_keys": final_keys})
    count = len(cases)
    return {"variant": name, "bm25_field": bm25_field, "dense_field": dense_field, "reranker_text_field": reranker_text_field, "case_count": count, "stage_hit_counts": stage_counts, "stage_recalls": {key: value / count if count else 0 for key, value in stage_counts.items()}, "final_hit_keys": final_hits, "traces": traces}


def run(args: argparse.Namespace) -> int:
    from src.retrieval.embedding_provider import ExistingMiniLMEmbeddingProvider

    views_path = args.runtime_dir / "pdf-v2-lite-retrieval-views.json"
    cases_path = args.runtime_dir / "pdf-v2-lite-development-benchmark.json"
    b12_acceptance = B12_OUT / "pdf-v2-lite-acceptance.json"
    b12 = json.loads(b12_acceptance.read_text(encoding="utf-8"))
    if not b12["gate_b2_passed"]:
        raise RuntimeError("Gate B2 must pass before B3")
    views = json.loads(views_path.read_text(encoding="utf-8"))["views"]
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    model_name = get_embedding_model_name()
    provider = ExistingMiniLMEmbeddingProvider(model_name_or_path=model_name, device=args.device)
    fields = ("raw_row_text", "enriched_retrieval_text")
    embeddings = {field: provider.encode_documents([str(view[field]) for view in views]) for field in fields}
    query_vectors = provider.encode_queries([str(case["query"]) for case in cases])
    results = [
        _run_variant(name=name, bm25_field=bm25_field, dense_field=dense_field, views=views, cases=cases, embeddings=embeddings, query_vectors=query_vectors, reranker_text_field="raw_row_text")
        for name, bm25_field, dense_field in VARIANTS
    ]
    baseline_hits = set(results[0]["final_hit_keys"])
    summary = []
    for result in results:
        final_hits = set(result.pop("final_hit_keys"))
        result.pop("traces")
        summary.append({**result, "new_final_hit_count": len(final_hits - baseline_hits), "regressed_final_hit_count": len(baseline_hits - final_hits)})
    c_result = summary[-1]
    rrf_gain = c_result["stage_hit_counts"]["rrf_40"] - summary[0]["stage_hit_counts"]["rrf_40"]
    final_gain = c_result["stage_hit_counts"]["final_5"] - summary[0]["stage_hit_counts"]["final_5"]
    gate_passed = rrf_gain > 0 and final_gain > 0 and c_result["regressed_final_hit_count"] == 0
    acceptance = {
        "schema": "pdf-retrieval-representation-v2-lite/gate-b3/acceptance/v1",
        "b12_acceptance_sha256": _sha(b12_acceptance),
        "runtime_views_sha256": _sha(views_path),
        "runtime_cases_sha256": _sha(cases_path),
        "embedding_model": model_name,
        "embedding_revision": provider.revision,
        "variant_count": len(VARIANTS),
        "candidate_k": 40,
        "rrf_k": 60,
        "reranker": "heuristic_raw_input",
        "reranker_top_k": 20,
        "final_top_k": 5,
        "gate_passed": gate_passed,
        "development_benchmark_semantics": b12["development_benchmark_semantics"],
        "sufficient_for_frozen_benchmark_transfer": False,
        "frozen_72_question_reads": 0,
        "model_training_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "decision": "pdf_v2_lite_hybrid_shadow_gate_passed" if gate_passed else "pdf_v2_lite_hybrid_transfer_gain_insufficient",
        "next_gate": "v2_lite_natural_query_development_set" if gate_passed else "stop_pdf_v2_lite_hybrid",
    }
    _write(args.out_dir / "shadow-corpus-manifest.json", {"view_count": len(views), "identity_mapping_error_count": 0, "representation_levels": {level: sum(view["period_binding_status"] == level for view in views) for level in ("strict_cell_aware", "table_level_only")}, "runtime_content_committed": False})
    _write(args.out_dir / "hybrid-variant-manifest.json", {"variants": [{"name": name, "bm25_field": bm25, "dense_field": dense, "reranker_field": "raw_row_text"} for name, bm25, dense in VARIANTS], "parameter_scan": False})
    _write(args.out_dir / "hybrid-funnel-results.json", {"results": summary})
    _write(args.out_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": acceptance["next_gate"], "production_switch_allowed": False})
    _write(args.out_dir / "pdf-v2-lite-gate-b3-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default=os.getenv("PDF_V2_LITE_EMBEDDING_DEVICE", "cpu"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
