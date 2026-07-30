"""NF38 Embedding A/B evaluation logic with explicit reliability controls."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.evaluation import EvaluationCase
from src.evaluation.nf37_metrics import candidate_to_source, ranking_metrics
from src.evaluation.nf38_dense_index import DenseIndex
from src.retrieval.candidate_fusion import rrf
from src.retrieval.embedding_provider import EmbeddingProvider


class EvaluationConfigurationError(ValueError):
    """Raised when an official NF38 run is missing required configuration."""


class EvaluationDatasetError(ValueError):
    """Raised when the labeled evaluation dataset is invalid."""


class TokenizerUnavailableError(RuntimeError):
    """Raised when official token accounting cannot access the real tokenizer."""


@dataclass(frozen=True)
class EvaluationScope:
    """Immutable corpus and tenant boundary for an official NF38 run."""

    tenant_id: int
    allowed_document_ids: frozenset[str]
    expected_case_count: int
    expected_corpus_hash: str
    expected_evidence_ids_hash: str


@dataclass
class FrozenBm25Pool:
    """Scoped, privacy-preserving BM25 candidate pool."""

    candidates: dict[str, list[dict[str, Any]]]
    tenant_id: int
    requested_k: int
    oversample_k: int
    allowed_document_ids: frozenset[str]
    out_of_scope_candidate_count: int = 0
    cases_with_candidate_shortfall: list[str] = field(default_factory=list)
    candidate_document_ids: set[str] = field(default_factory=set)

    def scope_report(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "allowed_document_count": len(self.allowed_document_ids),
            "allowed_document_ids_hash": _stable_digest(sorted(self.allowed_document_ids)),
            "requested_k": self.requested_k,
            "oversample_k": self.oversample_k,
            "out_of_scope_candidate_count": self.out_of_scope_candidate_count,
            "cases_with_candidate_shortfall": len(self.cases_with_candidate_shortfall),
            "candidate_document_ids_hash": _stable_digest(sorted(self.candidate_document_ids)),
        }


@dataclass
class DenseRankingResult:
    metrics: dict[str, Any]
    rankings: dict[str, list[dict[str, Any]]]


@dataclass
class HybridRankingResult:
    rrf_metrics: dict[str, Any]
    final_metrics: dict[str, Any]
    rrf_rankings: dict[str, list[dict[str, Any]]]
    final_rankings: dict[str, list[dict[str, Any]]]
    reranker: str
    top_k: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rrf": self.rrf_metrics,
            "final": self.final_metrics,
            "reranker": self.reranker,
            "top_k": self.top_k,
        }


@dataclass
class LatencyReport:
    """Collect latency and resource metrics for an embedding variant."""

    device: str
    model_cold_start_seconds: float = 0.0
    index_build_seconds: float = 0.0
    chunks_encoded: int = 0
    chunks_per_second: float = 0.0
    query_embedding_p50_ms: float = 0.0
    query_embedding_p95_ms: float = 0.0
    dense_search_p50_ms: float = 0.0
    dense_search_p95_ms: float = 0.0
    full_retrieval_p50_ms: float = 0.0
    full_retrieval_p95_ms: float = 0.0
    end_to_end_p50_ms: float = 0.0
    end_to_end_p95_ms: float = 0.0
    gpu_memory_mb: int | None = None
    process_rss_mb: float = 0.0
    index_disk_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "model_cold_start_seconds": self.model_cold_start_seconds,
            "index_build_seconds": self.index_build_seconds,
            "chunks_encoded": self.chunks_encoded,
            "chunks_per_second": self.chunks_per_second,
            "query_embedding_p50_ms": self.query_embedding_p50_ms,
            "query_embedding_p95_ms": self.query_embedding_p95_ms,
            "dense_search_p50_ms": self.dense_search_p50_ms,
            "dense_search_p95_ms": self.dense_search_p95_ms,
            "full_retrieval_p50_ms": self.full_retrieval_p50_ms,
            "full_retrieval_p95_ms": self.full_retrieval_p95_ms,
            "end_to_end_p50_ms": self.end_to_end_p50_ms,
            "end_to_end_p95_ms": self.end_to_end_p95_ms,
            "gpu_memory_mb": self.gpu_memory_mb,
            "process_rss_mb": self.process_rss_mb,
            "index_disk_bytes": self.index_disk_bytes,
        }


def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_labeled_cases(
    cases: list[EvaluationCase],
    *,
    expected_count: int,
) -> dict[str, Any]:
    """Validate that an official run uses a complete, labeled, stable case set."""
    if len(cases) != expected_count:
        raise EvaluationDatasetError(f"Expected {expected_count} cases, got {len(cases)}")

    ids = [case.case_id for case in cases]
    duplicate_ids = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicate_ids:
        raise EvaluationDatasetError(f"Duplicate case IDs: {duplicate_ids}")

    missing_sources = [
        case.case_id
        for case in cases
        if not case.expected_no_answer and not case.expected_sources
    ]
    if missing_sources:
        raise EvaluationDatasetError(
            f"Answerable cases missing expected_sources: {missing_sources}"
        )

    answerable_count = sum(not case.expected_no_answer for case in cases)
    no_answer_count = len(cases) - answerable_count
    return {
        "case_count": len(cases),
        "answerable_case_count": answerable_count,
        "no_answer_case_count": no_answer_count,
        "cases_missing_expected_sources": len(missing_sources),
        "duplicate_case_ids": len(duplicate_ids),
        "status": "passed",
    }


def question_hash(cases: list[EvaluationCase]) -> str:
    """Hash questions and retrieval scope without incorporating labels."""
    payload = [
        {
            "case_id": case.case_id,
            "question": case.question,
            "document_scope": sorted(case.document_names),
        }
        for case in sorted(cases, key=lambda item: item.case_id)
    ]
    return _stable_digest(payload)


def label_hash(cases: list[EvaluationCase]) -> str:
    """Hash expected labels without incorporating question wording."""
    payload = [
        {
            "case_id": case.case_id,
            "expected_answer": list(case.expected_answer_contains),
            "expected_numbers": list(case.expected_numbers),
            "expected_sources": [
                {"filename": source.filename, "page": source.page, "chunk_id": source.chunk_id}
                for source in case.expected_sources
            ],
            "expected_no_answer": case.expected_no_answer,
            "expected_period": case.metadata.get("expected_period"),
            "expected_unit": case.metadata.get("expected_unit"),
        }
        for case in sorted(cases, key=lambda item: item.case_id)
    ]
    return _stable_digest(payload)


def validate_scope_corpus(scope: EvaluationScope, records: list[Any]) -> None:
    """Reject a corpus that differs from the frozen official scope."""
    from src.evaluation.nf38_corpus import build_corpus_manifest

    actual_documents = frozenset(str(record.document_id) for record in records)
    if actual_documents != scope.allowed_document_ids:
        raise EvaluationConfigurationError("Canonical corpus document scope does not match EvaluationScope")
    actual = build_corpus_manifest(records)
    if actual["corpus_hash"] != scope.expected_corpus_hash:
        raise EvaluationConfigurationError("Canonical corpus hash does not match EvaluationScope")
    if actual["evidence_ids_hash"] != scope.expected_evidence_ids_hash:
        raise EvaluationConfigurationError("Canonical evidence IDs hash does not match EvaluationScope")


def measure_query_latencies(
    provider: EmbeddingProvider,
    queries: list[str],
    warmup: int = 1,
    rounds: int = 10,
) -> dict[str, float]:
    if not queries:
        return {"p50_ms": 0.0, "p95_ms": 0.0}
    for _ in range(warmup):
        provider.encode_queries(queries[:1])
    latencies: list[float] = []
    for _ in range(rounds):
        start = time.monotonic()
        provider.encode_queries(queries[:1])
        latencies.append((time.monotonic() - start) * 1000)
    return {"p50_ms": _percentile(latencies, 50), "p95_ms": _percentile(latencies, 95)}


def get_gpu_memory_mb(device: str) -> int | None:
    """Return allocated memory for the provider's explicit CUDA device."""
    if not device.startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        suffix = device.partition(":")[2]
        index = int(suffix) if suffix else torch.cuda.current_device()
        return int(torch.cuda.memory_allocated(index) / (1024 * 1024))
    except (ImportError, ValueError, RuntimeError):
        return None


def get_process_rss_mb() -> float:
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except (ImportError, AttributeError):
        pass
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
    except (ImportError, OSError, RuntimeError):
        return 0.0


def model_identity(provider: EmbeddingProvider) -> dict[str, Any]:
    """Create a reproducible local identity without exposing a filesystem path."""
    paths = getattr(provider, "identity_files", lambda: {})()
    payloads = getattr(provider, "identity_payloads", lambda: {})()
    config_digest = _digest_first(paths.get("config", ()))
    tokenizer_digest = _digest_first(paths.get("tokenizer", ()))
    if config_digest is None and payloads.get("config") is not None:
        config_digest = _stable_digest(payloads["config"])
    if tokenizer_digest is None and payloads.get("tokenizer") is not None:
        tokenizer_digest = _stable_digest(payloads["tokenizer"])
    return {
        "model_name": provider.name,
        "declared_revision": provider.revision,
        "resolved_revision": getattr(provider, "resolved_revision", provider.revision),
        "config_identity_digest": config_digest,
        "tokenizer_identity_digest": tokenizer_digest,
        "dimension": provider.dimension,
        "max_length": provider.max_length,
        "device": provider.device,
        "normalization": "l2",
    }


def _digest_first(paths: tuple[str, ...] | list[str]) -> str | None:
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def compute_token_length_report(
    records: list[Any],
    provider: EmbeddingProvider,
    *,
    selected_max_length: int,
    threshold: float = 0.05,
    require_real_tokenizer: bool = True,
) -> dict[str, Any]:
    """Compute official BGE token distribution with the actual tokenizer only."""
    if selected_max_length not in {512, 1024}:
        raise EvaluationConfigurationError("NF38 supports only BGE max_length 512 or 1024")
    texts = [record.embedding_text for record in records]
    tokenizer = provider.get_tokenizer()
    if tokenizer is None and require_real_tokenizer:
        raise TokenizerUnavailableError("NF38 official evaluation requires the real BGE-M3 tokenizer")
    if tokenizer is None:
        raise TokenizerUnavailableError("No tokenizer is available for token reporting")
    lengths = [len(tokenizer.encode(text)) for text in texts]
    arr = np.array(lengths, dtype=np.float32) if lengths else np.array([0], dtype=np.float32)
    truncated_count = int(sum(length > selected_max_length for length in lengths))
    total = len(lengths)
    ratio = truncated_count / total if total else 0.0
    return {
        "token_length_method": "real_tokenizer",
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "truncated_count_at_512": int(sum(length > 512 for length in lengths)),
        "truncated_count_at_1024": int(sum(length > 1024 for length in lengths)),
        "selected_max_length": selected_max_length,
        "truncated_count": truncated_count,
        "total_records": total,
        "truncated_ratio": ratio,
        "threshold": threshold,
        "within_threshold": ratio <= threshold,
    }


def freeze_bm25_pool(
    cases: list[EvaluationCase],
    bm25_search_fn,
    *,
    scope: EvaluationScope,
    k: int = 50,
    oversample_k: int = 200,
) -> FrozenBm25Pool:
    """Freeze only candidates within the explicit tenant and canonical corpus."""
    pool: dict[str, list[dict[str, Any]]] = {}
    shortfalls: list[str] = []
    out_of_scope_count = 0
    candidate_documents: set[str] = set()

    for case in cases:
        raw = bm25_search_fn(case.question, k=oversample_k, user_id=scope.tenant_id)
        accepted: list[dict[str, Any]] = []
        for candidate in raw:
            document_id = _candidate_document_id(candidate)
            if document_id not in scope.allowed_document_ids:
                out_of_scope_count += 1
                continue
            accepted.append(candidate)
            candidate_documents.add(document_id)
            if len(accepted) >= k:
                break
        if not accepted:
            raise EvaluationConfigurationError(
                f"BM25 produced zero in-scope candidates for case {case.case_id!r}"
            )
        if len(accepted) < k:
            shortfalls.append(case.case_id)
        pool[case.case_id] = _normalize_bm25_candidates(
            accepted,
            tenant_id=scope.tenant_id,
        )

    return FrozenBm25Pool(
        candidates=pool,
        tenant_id=scope.tenant_id,
        requested_k=k,
        oversample_k=oversample_k,
        allowed_document_ids=scope.allowed_document_ids,
        out_of_scope_candidate_count=out_of_scope_count,
        cases_with_candidate_shortfall=shortfalls,
        candidate_document_ids=candidate_documents,
    )


def _candidate_document_id(candidate: dict[str, Any]) -> str:
    metadata = candidate.get("metadata") or {}
    return str(metadata.get("doc_name") or candidate.get("document_id") or "")


def _normalize_bm25_candidates(
    candidates: list[dict[str, Any]],
    *,
    tenant_id: int | None = None,
) -> list[dict[str, Any]]:
    """Normalize raw BM25 results without discarding their source identity."""
    normalized: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates):
        metadata = candidate.get("metadata") or {}
        evidence_id = candidate.get("doc_id")
        document_id = metadata.get("doc_name")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise EvaluationConfigurationError(
                f"BM25 candidate at rank {rank} is missing doc_id"
            )
        if not isinstance(document_id, str) or not document_id.strip():
            raise EvaluationConfigurationError(
                f"BM25 candidate at rank {rank} is missing document_id"
            )
        normalized.append(
            {
                "candidate_id": evidence_id,
                "evidence_id": evidence_id,
                "document_id": document_id,
                "tenant_id": tenant_id,
                "page": metadata.get("page"),
                "block_type": metadata.get("type", "text"),
                "parent_id": metadata.get("parent_id"),
                "table_id": metadata.get("table_id"),
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
    *,
    return_result: bool = False,
) -> dict[str, Any] | DenseRankingResult:
    """Run dense retrieval once and optionally retain those rankings for artifacts."""
    rankings: dict[str, list[dict[str, Any]]] = {}
    query_latencies: list[float] = []
    for case in cases:
        query_vector = provider.encode_queries([case.question])[0]
        start = time.monotonic()
        candidates = index.search(query_vector, k=max(ks))
        query_latencies.append(time.monotonic() - start)
        rankings[case.case_id] = candidates
    metrics = ranking_metrics(cases, rankings, ks)
    metrics.update(
        {
            "query_count": len(cases),
            "query_latencies_p50": _percentile(query_latencies, 50),
            "query_latencies_p95": _percentile(query_latencies, 95),
            "embedding_model": provider.name,
            "embedding_dimension": provider.dimension,
        }
    )
    result = DenseRankingResult(metrics=metrics, rankings=rankings)
    return result if return_result else result.metrics


def run_hybrid_ranking(
    cases: list[EvaluationCase],
    index: DenseIndex,
    provider: EmbeddingProvider,
    bm25_pool: FrozenBm25Pool | dict[str, list[dict[str, Any]]],
    reranker=None,
    top_k: int = 5,
    ks: tuple[int, ...] = (5, 8, 20, 40),
    *,
    return_result: bool = False,
) -> dict[str, Any] | HybridRankingResult:
    """Run Hybrid ranking once and retain RRF/final rankings when requested."""
    pool = bm25_pool.candidates if isinstance(bm25_pool, FrozenBm25Pool) else bm25_pool
    rrf_rankings: dict[str, list[dict[str, Any]]] = {}
    final_rankings: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        query_vector = provider.encode_queries([case.question])[0]
        dense_candidates = index.search(query_vector, k=max(ks))
        fused = rrf([_to_rrf_format(dense_candidates), _to_rrf_format(pool.get(case.case_id, []))])
        rrf_rankings[case.case_id] = _from_rrf_format(fused)
        final_rankings[case.case_id] = _from_rrf_format(
            reranker.rerank(case.question, fused, top_k=top_k) if reranker else fused[:top_k]
        )
    rrf_metrics = ranking_metrics(cases, rrf_rankings, ks)
    final_raw = ranking_metrics(cases, final_rankings, (top_k,))
    final_metrics = {
        f"case_hit_rate_at_{top_k}": final_raw[f"case_hit_rate_at_{top_k}"],
        f"source_recall_at_{top_k}": final_raw[f"source_recall_at_{top_k}"],
        f"all_source_coverage_at_{top_k}": final_raw[f"all_source_coverage_at_{top_k}"],
        "mrr": final_raw["mrr"],
    }
    result = HybridRankingResult(
        rrf_metrics=rrf_metrics,
        final_metrics=final_metrics,
        rrf_rankings=rrf_rankings,
        final_rankings=final_rankings,
        reranker=reranker.name if reranker else "noop",
        top_k=top_k,
    )
    return result if return_result else result.to_dict()


def _to_rrf_format(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    return [
        {
            "candidate_id": item.get("doc_id", ""),
            "evidence_id": item.get("doc_id", ""),
            "document_id": (item.get("metadata") or {}).get("doc_name", ""),
            "page": (item.get("metadata") or {}).get("page"),
            "block_type": (item.get("metadata") or {}).get("type", "text"),
            "score": float(item.get("fused_score", item.get("score", 0))),
            "rank": rank,
        }
        for rank, item in enumerate(fused)
    ]


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p / 100), len(ordered) - 1)]


def evaluate_ranking_gate(
    *,
    baseline_dense: dict[str, Any],
    variant_dense: dict[str, Any],
    baseline_hybrid: dict[str, Any],
    variant_hybrid: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate Dense, RRF, and final stages without mixing their metric inputs."""
    baseline_rrf = baseline_hybrid["rrf"]
    variant_rrf = variant_hybrid["rrf"]
    baseline_final = baseline_hybrid["final"]
    variant_final = variant_hybrid["final"]
    dense_candidate_improved = (
        variant_dense["case_hit_rate_at_40"] > baseline_dense["case_hit_rate_at_40"]
        or variant_dense["source_recall_at_40"] > baseline_dense["source_recall_at_40"]
    )
    rrf_candidate_improved = (
        variant_rrf["case_hit_rate_at_40"] > baseline_rrf["case_hit_rate_at_40"]
        or variant_rrf["source_recall_at_40"] > baseline_rrf["source_recall_at_40"]
    )
    no_regression = (
        variant_rrf["source_recall_at_40"] >= baseline_rrf["source_recall_at_40"]
        and variant_rrf["all_source_coverage_at_40"] >= baseline_rrf["all_source_coverage_at_40"]
    )
    final_improved = (
        variant_final[f"case_hit_rate_at_{top_k}"] > baseline_final[f"case_hit_rate_at_{top_k}"]
        or variant_final["mrr"] - baseline_final["mrr"] >= 0.02
    )
    candidate_improved = dense_candidate_improved or rrf_candidate_improved
    return {
        "passed": candidate_improved and no_regression and final_improved,
        "candidate_improved": candidate_improved,
        "dense_candidate_improved": dense_candidate_improved,
        "rrf_candidate_improved": rrf_candidate_improved,
        "no_regression": no_regression,
        "final_improved": final_improved,
        "dense_case_hit_40": {
            "baseline": baseline_dense["case_hit_rate_at_40"],
            "variant": variant_dense["case_hit_rate_at_40"],
        },
        "dense_source_recall_40": {
            "baseline": baseline_dense["source_recall_at_40"],
            "variant": variant_dense["source_recall_at_40"],
        },
        "rrf_case_hit_40": {
            "baseline": baseline_rrf["case_hit_rate_at_40"],
            "variant": variant_rrf["case_hit_rate_at_40"],
        },
        "rrf_source_recall_40": {
            "baseline": baseline_rrf["source_recall_at_40"],
            "variant": variant_rrf["source_recall_at_40"],
        },
        "final_case_hit_5": {
            "baseline": baseline_final[f"case_hit_rate_at_{top_k}"],
            "variant": variant_final[f"case_hit_rate_at_{top_k}"],
        },
        "final_source_recall_5": {
            "baseline": baseline_final[f"source_recall_at_{top_k}"],
            "variant": variant_final[f"source_recall_at_{top_k}"],
        },
        "final_mrr": {"baseline": baseline_final["mrr"], "variant": variant_final["mrr"]},
    }


def compute_case_diff(
    cases: list[EvaluationCase],
    baseline_rankings: dict[str, list[dict[str, Any]]],
    variant_rankings: dict[str, list[dict[str, Any]]],
    *,
    stage: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Compare supplied rankings only; this helper never invokes retrieval."""
    diffs: list[dict[str, Any]] = []
    for case in cases:
        if not case.expected_sources:
            continue
        baseline = baseline_rankings.get(case.case_id, [])
        variant = variant_rankings.get(case.case_id, [])
        baseline_ranks = _gold_ranks(case, baseline)
        variant_ranks = _gold_ranks(case, variant)
        baseline_hit = any(rank < top_k for rank in baseline_ranks)
        variant_hit = any(rank < top_k for rank in variant_ranks)
        diffs.append(
            {
                "case_id": case.case_id,
                "stage": stage,
                "minilm_gold_ranks": baseline_ranks,
                "bge_gold_ranks": variant_ranks,
                "retrieval_hit_baseline": baseline_hit,
                "retrieval_hit_variant": variant_hit,
                "improved": variant_hit and not baseline_hit,
                "regressed": baseline_hit and not variant_hit,
            }
        )
    return diffs


def _gold_ranks(case: EvaluationCase, candidates: list[dict[str, Any]]) -> list[int]:
    ranks: list[int] = []
    for expected in case.expected_sources:
        for rank, candidate in enumerate(candidates):
            if expected.matches(candidate_to_source(candidate)):
                ranks.append(rank + 1)
                break
    return ranks

