"""Equal-RRF field-aware Structured retrieval without hard filters."""

from __future__ import annotations

from src.pdf_retrieval_v4.candidate_field_index import CandidateFieldIndexReader
from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit, fuse_candidate_hits
from src.pdf_retrieval_v4.candidate_view_index import CandidateSearchHit, CandidateViewIndexReader

GENERAL_LANES = ("candidate_structured_bm25", "candidate_structured_dense")
FIELDS = ("metric", "axis", "context", "evidence")


def retrieve_field_aware_structured(
    general_reader: CandidateViewIndexReader,
    field_reader: CandidateFieldIndexReader,
    *,
    general_query: str,
    field_queries: dict[str, str],
    document_scope: set[str],
    lane_k: int = 50,
    rrf_k: int = 60,
) -> tuple[dict[str, list[CandidateSearchHit]], dict[str, list[CandidateRRFHit]]]:
    if lane_k != 50 or rrf_k != 60:
        raise ValueError("frozen_retrieval_budget_mutation")
    hits: dict[str, list[CandidateSearchHit]] = {}
    for lane in GENERAL_LANES:
        allowed = general_reader.candidate_keys_for_documents(lane, document_scope) if document_scope else None
        hits[lane] = general_reader.search(lane, general_query, allowed_candidate_keys=allowed, k=lane_k)
    variants: dict[str, list[CandidateRRFHit]] = {"s0": fuse_candidate_hits(dict(hits), rrf_k=rrf_k)}
    cumulative = dict(hits)
    for number, field in enumerate(FIELDS, 1):
        query = field_queries.get(field, "")
        lane = f"structured_{field}_bm25"
        allowed = field_reader.candidate_keys_for_documents(field, document_scope) if document_scope else None
        field_hits = field_reader.search(field, query, allowed_candidate_keys=allowed, k=lane_k) if query else []
        hits[lane] = field_hits
        cumulative[lane] = field_hits
        variants[f"s{number}"] = fuse_candidate_hits(dict(cumulative), rrf_k=rrf_k)
    return hits, variants
