"""Gate 08 R2 Candidate-aligned Direct Retrieval orchestrator.

Orchestrates 4-lane candidate-aligned retrieval + RRF fusion for a
single case.  Fixed parameters: lane_k=50, rrf_k=60, all weights=1.0,
final pool K=40.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from src.pdf_retrieval_v4.candidate_query_builder import (
    _metric_aliases,
    build_all_queries,
)
from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit, fuse_candidate_hits
from src.pdf_retrieval_v4.candidate_slot_pool import build_slot_pool
from src.pdf_retrieval_v4.candidate_view_index import (
    LANES,
    CandidateSearchHit,
    CandidateViewIndexReader,
)
from src.pdf_retrieval_v4.query_plan_models import QueryPlan
from src.pdf_retrieval_v4.retrieval_demand import SlotRetrievalRequestV1


def _slot_query_variants(request: SlotRetrievalRequestV1) -> list[str]:
    """The literal slot query, then this slot's own alias expansions.

    Every variant inherits the slot's own ``entity`` and ``period``.  An alias is
    a different name for *the same requirement*; it is never licence to widen the
    search to another slot's terms.  That distinction is the whole defect this
    replaces -- the plan-wide `_entity_terms` put every company into every slot's
    query, so four lanes all searched the same crowded pool:

        "and JPMorganChase larger | FY2025 | tsla | jpmorganchase"

    Alias expansion is measurably load-bearing, not legacy decoration: four slots
    in the canonical fixture set have a question surface that differs from the
    filing label (``Revenues`` vs ``net sales``), one of them a case that
    currently releases.  It is therefore kept -- but kept slot-local, so it
    cannot re-introduce cross-slot contamination.

    The literal query is always first, so a caller that wants no expansion is the
    same code path with one variant rather than a second implementation.
    """

    variants = [request.query]

    def _render(term: str) -> str:
        parts = (request.entity, term, request.period)
        return " ".join(part.strip() for part in parts if part and part.strip())

    for alias in _metric_aliases(request.metric):
        candidate = _render(alias)
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


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

    def retrieve_for_requests(
        self,
        requests: Sequence[SlotRetrievalRequestV1],
        *,
        document_scope: set[str],
        total_k: int | None = None,
        alias_expansion: bool = False,
    ) -> dict[str, Any]:
        """One lane per retrieval demand, merged by rank into a bounded packet.

        The result's cardinality is ``len(requests)`` and nothing else.  Every
        other input to lane count has been removed, because the demand is the
        one thing already decided upstream and re-deciding it is what went
        wrong: `retrieve` above takes its count from `QueryPlan.operand_slots`,
        which is the retrieval planner's own guess at a number `SupervisorPlan`
        had already stated -- and for every cross-entity comparison and ranking
        that guess is 1.

        Each lane is searched with its own demand's deterministic query, so no
        lane carries another slot's entity.  Lanes are fused per slot and then
        merged by `build_slot_pool`, which interleaves them round-robin by rank
        with a per-slot minimum budget.  Rank-based rather than score-based is
        required, not stylistic: an RRF score is bounded by how many lanes a
        candidate appeared in, so two slots' scores are not on one scale and
        comparing them would compare lane counts.  Round-robin is what stops one
        company's evidence from occupying the whole packet -- the observed
        failure.

        A single demand keeps the plain top-``final_pool_k`` shape rather than
        going through the slot pool, whose per-slot truncation is sized for
        splitting a budget between several slots and would halve a lone slot's
        depth for no reason.

        Returns the same mapping shape as `retrieve`, so callers need no second
        code path.  ``lane_hits`` and ``rrf_hits`` are empty: they describe one
        globally fused search, and there is no longer one.
        """

        allowed_keys = self._allowed_keys_for_scope(document_scope)
        slot_pools: dict[str, list[CandidateRRFHit]] = {}
        slot_queries: dict[str, list[str]] = {}
        for request in requests:
            variants = (
                _slot_query_variants(request) if alias_expansion else [request.query]
            )
            slot_queries[request.slot_id] = list(variants)
            if len(variants) == 1:
                lane_hits = self._search_lanes(variants[0], allowed_keys)
            else:
                # The same bounded variant merge the legacy path used, so the
                # alias-priority penalty keeps its meaning.  What is new is that
                # every variant here belongs to *one* slot.
                lane_hits = self._merge_variant_lane_hits(
                    [self._search_lanes(variant, allowed_keys) for variant in variants],
                    variant_priorities=[
                        self._variant_priority(variant) for variant in variants
                    ],
                    rank_penalty=max(5, self.lane_k // 5),
                )
            slot_pools[request.slot_id] = fuse_candidate_hits(
                lane_hits, rrf_k=self.rrf_k
            )

        if not slot_pools:
            return {
                "candidate_direct_pool": [],
                "lane_hits": {},
                "rrf_hits": [],
                "slot_pools": {},
                "slot_query_variants": {},
            }

        if len(slot_pools) == 1:
            pool = self._pool_from_rrf(next(iter(slot_pools.values())))
        else:
            # Merged to a working depth, not to the final packet size.  The
            # caller still applies entity and scope ordering before its own cap
            # at `final_pool_k`, and truncating to the final size here would drop
            # the deeper candidates that ordering exists to rescue -- a slot's
            # second candidate is exactly the one an entity-priority pass lifts
            # over another slot's first.  The depth is the one the previous
            # interleave used for the same reason.
            pool = build_slot_pool(
                slot_pools,
                total_k=total_k or max(80, self.final_pool_k * 2),
            )

        return {
            "candidate_direct_pool": pool,
            "lane_hits": {},
            "rrf_hits": [],
            "slot_pools": slot_pools,
            "slot_query_variants": slot_queries,
        }
