"""Gate 08 R2 Candidate-aligned Direct Retrieval orchestrator.

Orchestrates 4-lane candidate-aligned retrieval + RRF fusion for a
single case.  Fixed parameters: lane_k=50, rrf_k=60, all weights=1.0,
final pool K=40.
"""

from __future__ import annotations

from dataclasses import replace
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

    @staticmethod
    def _merge_variant_lane_hits(
        variant_hits: list[dict[str, list[CandidateSearchHit]]],
        *,
        variant_priorities: list[int] | None = None,
        rank_penalty: int = 1000,
    ) -> dict[str, list[CandidateSearchHit]]:
        """Merge bounded query-variant hits without double-counting a lane.

        Metric aliases are searched independently because the index reader
        tokenizes a single query as an OR expression.  A candidate appearing
        in multiple variants should still contribute at most one rank per
        lane to RRF; otherwise an alias could win merely by being repeated.
        Keep the best (lowest) lane rank, with score and identity as stable
        tie-breakers.
        """

        merged: dict[str, dict[str, CandidateSearchHit]] = {
            lane: {} for lane in LANES
        }

        def hit_key(lane: str, hit: CandidateSearchHit) -> tuple[int, float, str, str]:
            rank = hit.bm25_rank if "bm25" in lane else hit.dense_rank
            score = hit.bm25_score if "bm25" in lane else hit.dense_score
            return (
                int(rank or 10**9),
                -float(score or 0.0),
                str(hit.candidate_key),
                str(hit.view_id),
            )

        priorities = variant_priorities or [1] * len(variant_hits)
        max_priority = max(priorities, default=1)
        for variant_index, lane_hits in enumerate(variant_hits):
            priority = priorities[variant_index] if variant_index < len(priorities) else 1
            for lane, hits in lane_hits.items():
                lane_map = merged.setdefault(lane, {})
                for hit in hits:
                    key = str(hit.candidate_key)
                    if not key:
                        continue
                    # Specific filing labels (for example ``total net
                    # sales``) should outrank a broad alias (``revenue``)
                    # when both return a candidate at the same lane rank.
                    # Encode that bounded preference in the effective RRF
                    # rank while retaining the original score and identity.
                    raw_rank = hit.bm25_rank if "bm25" in lane else hit.dense_rank
                    effective_rank = int(raw_rank or 10**9) + (
                        max_priority - priority
                    ) * max(1, int(rank_penalty))
                    adjusted_hit = (
                        replace(hit, bm25_rank=effective_rank)
                        if "bm25" in lane
                        else replace(hit, dense_rank=effective_rank)
                    )
                    previous = lane_map.get(key)
                    if previous is None or hit_key(lane, adjusted_hit) < hit_key(lane, previous):
                        lane_map[key] = adjusted_hit

        return {
            lane: sorted(
                lane_map.values(),
                key=lambda hit: hit_key(lane, hit),
            )
            for lane, lane_map in merged.items()
        }

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

    @staticmethod
    def _variant_priority(query: str) -> int:
        """Return a bounded semantic priority for a metric alias variant.

        Query length is not a useful specificity signal here: the planner
        appends the same period and issuer terms to every variant, so the
        former implementation penalized ``total revenue`` merely because
        ``total net sales`` happened to contain one more token.  Keep the
        filing-label aliases at the same priority and let the broad
        ``revenue`` form act as the fallback.
        """

        phrase = " ".join(str(query).split("|", 1)[0].split()).casefold()
        if phrase in {"total net sales"}:
            return 4
        if phrase in {"total revenue", "total revenues", "total sales"}:
            return 3
        if phrase in {"net sales", "sales revenue", "revenues", "sales"}:
            return 2
        return 1

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
        slot_query_variants: dict[str, list[str]] = {}
        if slot_queries:
            for slot_id, query_list in slot_queries.items():
                if not query_list:
                    continue
                slot_query_variants[slot_id] = list(query_list)
                variant_lane_hits = [
                    self._search_lanes(slot_query, allowed_keys)
                    for slot_query in query_list
                ]
                variant_priorities = [
                    self._variant_priority(str(query))
                    for query in query_list
                ]
                slot_lane_hits = self._merge_variant_lane_hits(
                    variant_lane_hits,
                    variant_priorities=variant_priorities,
                    # Keep alias specificity meaningful without turning a
                    # valid result from one safe filing label into an
                    # effectively rank-56 hit merely because another alias
                    # has one extra word.  The lane itself is already
                    # bounded at ``lane_k``; a smaller deterministic penalty
                    # preserves recall while still preferring explicit
                    # labels over the broad ``revenue`` variant.
                    rank_penalty=max(5, self.lane_k // 5),
                )
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
            "slot_query_variants": slot_query_variants,
        }
