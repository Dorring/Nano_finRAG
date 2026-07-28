"""NF38 Embedding A/B evaluation logic.

This module provides the core evaluation functions for comparing MiniLM and
BGE-M3 dense embeddings while keeping all other variables fixed (BM25, RRF,
reranker, top-k). It is designed to be called from scripts and tested in
isolation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.evaluation import EvaluationCase
from src.evaluation.nf37_metrics import ranking_metrics
from src.retrieval.candidate_fusion import rrf
from src.retrieval.embedding_provider import EmbeddingProvider
from src.evaluation.nf38_dense_index import DenseIndex, build_dense_index


def freeze_bm25_pool(
    cases: list[EvaluationCase],
    bm25_search_fn,
    k: int = 50,
) -> dict[str, list[dict[str, Any]]]:
    """Freeze BM25 Top-k candidates for each case.

    The search_fn must accept (query, k, user_id) and return a list of
    chunk dicts with doc_id, content, metadata, score.
    """
    pool: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        candidates = bm25_search_fn(case.question, k=k, user_id=_user_id_for_case(case))
        pool[case.case_id] = _normalize_bm25_candidates(candidates)
    return pool


def _user_id_for_case(case: EvaluationCase) -> int:
    """Return the user_id for BM25 search. Uses sealed partition ID."""
    return 9003


def _normalize_bm25_candidates(candidates: list[dict]) -> list[dict[str, Any]]:
    """Convert BM25 results to the normalized candidate format."""
    normalized: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates):
        metadata = candidate.get("metadata") or {}
        normalized.append(
            {
                "candidate_id": candidate.get("doc_id", ""),
                "evidence_id": candidate.get("doc_id", ""),
                "document_id": metadata.get("doc_name", ""),
                "page": metadata.get("page"),
                "block_type": metadata.get("type", "text"),
                "score": float(candidate.get("score", 0)),
                "rank": rank,
            }
        )
    return normalized


def run_dense_diagnostic(
    cases: list[EvaluationCase],
    index: DenseIndex,
    provider: EmbeddingProvider,
    ks: tuple[int, ...] = (5, 8, 20, 40),
) -> dict[str, Any]:
    """Run Dense-only retrieval for each case and compute ranking metrics."""
    rankings: dict[str, list[dict[str, Any]]] = {}
    query_latencies: list[float] = []

    for case in cases:
        query_vector = provider.encode_queries([case.question])[0]
        start = time.monotonic()
        candidates = index.search(query_vector, k=max(ks))
        query_latencies.append(time.monotonic() - start)
        rankings[case.case_id] = candidates

    metrics = ranking_metrics(cases, rankings, ks)
    metrics["query_count"] = len(cases)
    metrics["query_latencies_p50"] = _percentile(query_latencies, 50)
    metrics["query_latencies_p95"] = _percentile(query_latencies, 95)
    metrics["embedding_model"] = provider.name
    metrics["embedding_dimension"] = provider.dimension
    return metrics


def run_hybrid_ranking(
    cases: list[EvaluationCase],
    index: DenseIndex,
    provider: EmbeddingProvider,
    bm25_pool: dict[str, list[dict[str, Any]]],
    reranker=None,
    top_k: int = 5,
    ks: tuple[int, ...] = (5, 8, 20, 40),
) -> dict[str, Any]:
    """Run Hybrid ranking (Dense + frozen BM25 via RRF) for each case."""
    rrf_rankings: dict[str, list[dict[str, Any]]] = {}
    final_rankings: dict[str, list[dict[str, Any]]] = {}

    for case in cases:
        query_vector = provider.encode_queries([case.question])[0]
        dense_candidates = index.search(query_vector, k=max(ks))
        bm25_candidates = bm25_pool.get(case.case_id, [])

        dense_list = _to_rrf_format(dense_candidates)
        bm25_list = _to_rrf_format(bm25_candidates)
        fused = rrf([dense_list, bm25_list])
        rrf_rankings[case.case_id] = _from_rrf_format(fused)

        if reranker is not None:
            reranked = reranker.rerank(case.question, fused, top_k=top_k)
            final_rankings[case.case_id] = _from_rrf_format(reranked)
        else:
            final_rankings[case.case_id] = _from_rrf_format(fused[:top_k])

    rrf_metrics = ranking_metrics(cases, rrf_rankings, ks)
    final_metrics = ranking_metrics(cases, final_rankings, (top_k,))

    return {
        "rrf": rrf_metrics,
        "final": {
            f"case_hit_rate_at_{top_k}": final_metrics[f"case_hit_rate_at_{top_k}"],
            f"source_recall_at_{top_k}": final_metrics[f"source_recall_at_{top_k}"],
            f"all_source_coverage_at_{top_k}": final_metrics[f"all_source_coverage_at_{top_k}"],
            "mrr": final_metrics["mrr"],
        },
        "reranker": reranker.name if reranker else "noop",
        "top_k": top_k,
    }


def _to_rrf_format(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert normalized candidates to the format expected by rrf()."""
    return [
        {
            "doc_id": candidate.get("candidate_id") or candidate.get("evidence_id") or "",
            "score": float(candidate.get("score", 0)),
            "metadata": {
                "doc_name": candidate.get("document_id", ""),
                "page": candidate.get("page"),
                "type": candidate.get("block_type", "text"),
            },
        }
        for candidate in candidates
    ]


def _from_rrf_format(fused: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert rrf() output back to the normalized candidate format."""
    result: list[dict[str, Any]] = []
    for rank, item in enumerate(fused):
        metadata = item.get("metadata") or {}
        result.append(
            {
                "candidate_id": item.get("doc_id", ""),
                "evidence_id": item.get("doc_id", ""),
                "document_id": metadata.get("doc_name", ""),
                "page": metadata.get("page"),
                "block_type": metadata.get("type", "text"),
                "score": float(item.get("fused_score", item.get("score", 0))),
                "rank": rank,
            }
        )
    return result


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a list."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * p / 100)
    index = min(index, len(sorted_values) - 1)
    return sorted_values[index]


def evaluate_ranking_gate(
    baseline_metrics: dict[str, Any],
    variant_metrics: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate whether the variant passes the Ranking Gate.

    Gate criteria (all must be met):
    1. Candidate pool improvement: at least one of Dense/RRF Case Hit@40 or
       Source Recall@40 increases.
    2. No significant regression: RRF Source Recall@40 and All-source
       Coverage@40 must not decrease.
    3. Final ranking improvement: Final Case Hit@5 increases by at least 1
       case, or Final MRR improves by at least 0.02.
    """
    baseline_rrf = baseline_metrics.get("rrf", {})
    variant_rrf = variant_metrics.get("rrf", {})
    baseline_final = baseline_metrics.get("final", {})
    variant_final = variant_metrics.get("final", {})

    # 1. Candidate pool improvement
    dense_hit_40_b = baseline_metrics.get("case_hit_rate_at_40", 0)
    dense_hit_40_v = variant_metrics.get("case_hit_rate_at_40", 0)
    rrf_hit_40_b = baseline_rrf.get("case_hit_rate_at_40", 0)
    rrf_hit_40_v = variant_rrf.get("case_hit_rate_at_40", 0)
    dense_recall_40_b = baseline_metrics.get("source_recall_at_40", 0)
    dense_recall_40_v = variant_metrics.get("source_recall_at_40", 0)
    rrf_recall_40_b = baseline_rrf.get("source_recall_at_40", 0)
    rrf_recall_40_v = variant_rrf.get("source_recall_at_40", 0)

    candidate_improved = (
        dense_hit_40_v > dense_hit_40_b
        or rrf_hit_40_v > rrf_hit_40_b
        or dense_recall_40_v > dense_recall_40_b
        or rrf_recall_40_v > rrf_recall_40_b
    )

    # 2. No regression
    no_regression = (
        rrf_recall_40_v >= rrf_recall_40_b
        and variant_rrf.get("all_source_coverage_at_40", 0)
        >= baseline_rrf.get("all_source_coverage_at_40", 0)
    )

    # 3. Final ranking improvement
    final_hit_b = baseline_final.get(f"case_hit_rate_at_{top_k}", 0)
    final_hit_v = variant_final.get(f"case_hit_rate_at_{top_k}", 0)
    final_mrr_b = baseline_final.get("mrr", 0)
    final_mrr_v = variant_final.get("mrr", 0)

    final_improved = final_hit_v > final_hit_b or (final_mrr_v - final_mrr_b) >= 0.02

    passed = candidate_improved and no_regression and final_improved

    return {
        "passed": passed,
        "candidate_improved": candidate_improved,
        "no_regression": no_regression,
        "final_improved": final_improved,
        "dense_case_hit_40": {"baseline": dense_hit_40_b, "variant": dense_hit_40_v},
        "rrf_case_hit_40": {"baseline": rrf_hit_40_b, "variant": rrf_hit_40_v},
        "rrf_source_recall_40": {"baseline": rrf_recall_40_b, "variant": rrf_recall_40_v},
        "final_case_hit_5": {"baseline": final_hit_b, "variant": final_hit_v},
        "final_mrr": {"baseline": final_mrr_b, "variant": final_mrr_v},
    }


def compute_case_diff(
    cases: list[EvaluationCase],
    baseline_rankings: dict[str, list[dict[str, Any]]],
    variant_rankings: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Compute per-case differences between baseline and variant."""
    from src.evaluation.nf37_metrics import candidate_to_source

    diffs: list[dict[str, Any]] = []
    for case in cases:
        if not case.expected_sources:
            continue

        baseline_candidates = baseline_rankings.get(case.case_id, [])
        variant_candidates = variant_rankings.get(case.case_id, [])

        baseline_hit = any(
            expected.matches(candidate_to_source(c))
            for expected in case.expected_sources
            for c in baseline_candidates[:5]
        )
        variant_hit = any(
            expected.matches(candidate_to_source(c))
            for expected in case.expected_sources
            for c in variant_candidates[:5]
        )

        diffs.append(
            {
                "case_id": case.case_id,
                "golden_pass_baseline": baseline_hit,
                "golden_pass_variant": variant_hit,
                "improved": variant_hit and not baseline_hit,
                "regressed": baseline_hit and not variant_hit,
            }
        )
    return diffs
