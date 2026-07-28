"""JSON-safe candidate-stage tracing for retrieval diagnostics."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class CandidateStageRank:
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_rank: int | None = None
    reranker_rank: int | None = None
    final_rank: int | None = None

@dataclass
class CandidateTrace:
    evidence_id: str
    document_id: str
    page: int | None
    block_type: str
    parent_id: str | None
    table_id: str | None
    query_variants: list[str] = field(default_factory=list)
    retrieval_sources: list[str] = field(default_factory=list)
    bm25_score: float | None = None
    dense_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None
    ranks: CandidateStageRank = field(default_factory=CandidateStageRank)

def summarize_candidates(candidates: list[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    """Return identifiers and scores only; never document content."""
    result = []
    for rank, candidate in enumerate(candidates[:limit] if limit else candidates, start=1):
        metadata = candidate.get("metadata") or {}
        result.append({
            "rank": rank,
            "evidence_id": candidate.get("doc_id"),
            "document_id": metadata.get("doc_name") or metadata.get("filename"),
            "page": metadata.get("page"),
            "block_type": metadata.get("type", "text"),
            "parent_id": metadata.get("parent_id"),
            "table_id": metadata.get("table_id"),
            "score": candidate.get("score"),
            "rrf_score": candidate.get("fused_score"),
            "reranker_score": candidate.get("rerank_score"),
            "query_variants": sorted({item.get("query_variant") for item in candidate.get("retrieval_provenance", []) if item.get("query_variant")}),
            "retrieval_sources": sorted({item.get("retriever") for item in candidate.get("retrieval_provenance", []) if item.get("retriever")}),
        })
    return result
