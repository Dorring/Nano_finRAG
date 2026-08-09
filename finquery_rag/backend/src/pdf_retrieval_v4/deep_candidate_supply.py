"""Deep Top200 candidate supply retrieval with frozen index semantics."""

from __future__ import annotations

from typing import Any

from src.pdf_retrieval_v4.candidate_field_index import CandidateFieldIndexReader
from src.pdf_retrieval_v4.candidate_rrf import fuse_candidate_hits
from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader
from src.pdf_retrieval_v4.field_family_normalizer import fuse_hierarchical_structured

SUPPLY_LANE_K = 200
RRF_K = 60
GENERAL_LANES = ("candidate_structured_bm25", "candidate_structured_dense")
RAW_LANES = ("candidate_raw_bm25", "candidate_raw_dense")
FIELD_NAMES = ("metric", "axis", "context", "evidence")


def _serialize_hits(items: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_key": item.candidate_key,
            "rank": item.bm25_rank or item.dense_rank,
            "score": item.bm25_score if item.bm25_rank else item.dense_score,
        }
        for item in items
    ]


def _serialize_rrf(items: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_key": item.candidate_key,
            "rank": rank,
            "rrf_score": item.rrf_score,
            "lane_ranks": item.lane_ranks,
        }
        for rank, item in enumerate(items, 1)
    ]


def retrieve_deep_supply(
    general_reader: CandidateViewIndexReader,
    field_reader: CandidateFieldIndexReader,
    *,
    general_query: str,
    field_queries: dict[str, str],
    document_scope: set[str],
    lane_k: int = SUPPLY_LANE_K,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if lane_k != SUPPLY_LANE_K:
        raise ValueError("supply_lane_k_must_equal_200")
    lane_hits: dict[str, list[Any]] = {}
    counts = {"bm25_searches": 0, "dense_searches": 0, "field_searches": 0}
    for lane in (*RAW_LANES, *GENERAL_LANES):
        allowed = general_reader.candidate_keys_for_documents(lane, document_scope) if document_scope else None
        lane_hits[lane] = general_reader.search(
            lane, general_query, allowed_candidate_keys=allowed, k=lane_k
        )
        counts["dense_searches" if lane.endswith("dense") else "bm25_searches"] += 1
    for field in FIELD_NAMES:
        query = field_queries.get(field, "")
        lane = f"structured_{field}_bm25"
        if not query:
            lane_hits[lane] = []
            continue
        allowed = field_reader.candidate_keys_for_documents(field, document_scope) if document_scope else None
        lane_hits[lane] = field_reader.search(
            field, query, allowed_candidate_keys=allowed, k=lane_k
        )
        counts["bm25_searches"] += 1
        counts["field_searches"] += 1
    raw_fused = fuse_candidate_hits(
        {lane: lane_hits[lane] for lane in RAW_LANES}, rrf_k=RRF_K
    )
    serialized = {lane: _serialize_hits(items) for lane, items in lane_hits.items()}
    _, structured_h1 = fuse_hierarchical_structured(serialized, rrf_k=RRF_K)
    return serialized, _serialize_rrf(raw_fused), structured_h1, counts
