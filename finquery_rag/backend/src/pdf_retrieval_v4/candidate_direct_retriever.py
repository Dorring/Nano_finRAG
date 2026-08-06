"""Gate 08 R2 Candidate-aligned Direct Retrieval orchestrator.

Orchestrates 4-lane candidate-aligned retrieval + RRF fusion for a
single case.  Fixed parameters: lane_k=50, rrf_k=60, all weights=1.0,
final pool K=40.
"""

from __future__ import annotations

from typing import Any

from src.pdf_retrieval_v4.candidate_query_builder import build_all_queries
from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit, fuse_candidate_hits
from src.pdf_retrieval_v4.candidate_slot_pool import build_slot_pool
from src.pdf_retrieval_v4.candidate_view_index import (
    LANES,
    CandidateSearchHit,
    CandidateViewIndexReader,
)
from src.pdf_retrieval_v4.query_plan_models import QueryPlan


class CandidateDirectRetriever:
    """Orchestrate 4-lane retrieval + RRF fusion for a single case."""

    def __init__(
        self,
        reader: CandidateViewIndexReader,
        *,
        rrf_k: int = 60,
        lane_k: int = 50,
    ) -> None:
        self.reader = reader
        self.rrf_k = int(rrf_k)
        self.lane_k = int(lane_k)
        self.final_pool_k = 40

    def _allowed_keys_for_scope(
        self, document_scope: set[str]
    ) -> dict[str, set[str] | None]:
        """Compute allowed candidate_keys per lane for the document scope."""
        cache: dict[str, set[str] | None] = {}
        for lane in LANES:
            if document_scope:
                cache[lane] = self.reader.candidate_keys_for_documents(lane, document_scope)
            else:
                cache[lane] = None
        return cache

    def _search_lanes(
        self,
        query: str,
        allowed_keys: dict[str, set[str] | None],
    ) -> dict[str, list[CandidateSearchHit]]:
        """Run 4-lane search with a single query."""
        lane_hits: dict[str, list[CandidateSearchHit]] = {}
        for lane in LANES:
            hits = self.reader.search(
                lane,
                query,
                allowed_candidate_keys=allowed_keys[lane],
                k=self.lane_k,
            )
            lane_hits[lane] = hits
        return lane_hits

    def _pool_from_rrf(self, rrf_hits: list[CandidateRRFHit]) -> list[dict[str, Any]]:
        """Build a top-K pool from RRF-fused hits."""
        return [
            {
                "candidate_key": hit.candidate_key,
                "rrf_score": hit.rrf_score,
                "rank": rank,
                "lane_ranks": dict(hit.lane_ranks),
                "supporting_view_ids": dict(hit.supporting_view_ids),
            }
            for rank, hit in enumerate(rrf_hits[: self.final_pool_k], 1)
        ]

    def retrieve(
        self, plan: QueryPlan, *, document_scope: set[str]
    ) -> dict[str, Any]:
        """Run candidate-aligned direct retrieval for a single case.

        Returns a dict with::

            {
                "candidate_direct_pool": list[dict],
                "lane_hits": dict[str, list[CandidateSearchHit]],
                "rrf_hits": list[CandidateRRFHit],
                "slot_pools": dict[str, list[CandidateRRFHit]],
            }
        """
        allowed_keys = self._allowed_keys_for_scope(document_scope)

        # 1. Build raw_question query and 4-lane search.
        queries = build_all_queries(plan)
        raw_query = queries["raw_question"][0] if queries["raw_question"] else ""
        lane_hits = self._search_lanes(raw_query, allowed_keys)

        # 2. Fuse raw_question hits with candidate RRF.
        rrf_hits = fuse_candidate_hits(lane_hits, rrf_k=self.rrf_k)

        # 3. If plan has operand_slots, build slot queries and search.
        slot_queries = queries.get("slots", {})
        is_multi_slot = len(slot_queries) > 1

        slot_pools: dict[str, list[CandidateRRFHit]] = {}
        if slot_queries:
            for slot_id, query_list in slot_queries.items():
                if not query_list:
                    continue
                slot_query = query_list[0]
                slot_lane_hits = self._search_lanes(slot_query, allowed_keys)
                slot_rrf = fuse_candidate_hits(slot_lane_hits, rrf_k=self.rrf_k)
                slot_pools[slot_id] = slot_rrf

        # 4. Build final candidate_direct_pool.
        if is_multi_slot and slot_pools:
            candidate_direct_pool = build_slot_pool(slot_pools)
        else:
            candidate_direct_pool = self._pool_from_rrf(rrf_hits)

        return {
            "candidate_direct_pool": candidate_direct_pool,
            "lane_hits": lane_hits,
            "rrf_hits": rrf_hits,
            "slot_pools": slot_pools,
        }
