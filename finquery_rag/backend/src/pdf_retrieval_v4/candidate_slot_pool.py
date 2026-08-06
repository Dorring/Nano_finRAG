"""Gate 08 R2 slot round-robin pool builder for multi-evidence cases.

When a case has multiple operand slots, each slot produces its own
RRF-fused candidate list.  This module builds a fair round-robin pool
that ensures every slot gets representation while capping the total
pool size.
"""

from __future__ import annotations

from typing import Any

from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit


def build_slot_pool(
    slot_hits: dict[str, list[CandidateRRFHit]],
    *,
    slot_top_k: int = 20,
    slot_min_budget: int = 10,
    total_k: int = 40,
) -> list[dict[str, Any]]:
    """Build a round-robin slot pool for multi-evidence cases.

    1. For each slot, take top-K (``slot_top_k``) candidates.
    2. Round-robin: slot_1 rank_1, slot_2 rank_1, slot_1 rank_2, ...
    3. Each slot gets at least ``slot_min_budget`` candidates.
    4. Total pool capped at ``total_k``.
    5. Same candidate can support multiple slots (dedup by candidate_key,
       keep first occurrence; ``supporting_slots`` lists all slots).
    """
    slot_order = list(slot_hits.keys())
    slot_top: dict[str, list[CandidateRRFHit]] = {
        slot_id: list(hits[:slot_top_k]) for slot_id, hits in slot_hits.items()
    }

    # Build candidate -> supporting_slots mapping (all slots where candidate
    # appears in top-K).
    candidate_support: dict[str, list[str]] = {}
    for slot_id, hits in slot_top.items():
        for hit in hits:
            candidate_support.setdefault(hit.candidate_key, []).append(slot_id)

    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    slot_counts: dict[str, int] = {slot_id: 0 for slot_id in slot_order}
    max_rank = max((len(hits) for hits in slot_top.values()), default=0)

    # Round-robin pass.
    for rank in range(1, max_rank + 1):
        for slot_id in slot_order:
            hits = slot_top[slot_id]
            if rank > len(hits):
                continue
            hit = hits[rank - 1]
            if hit.candidate_key in seen:
                continue
            if len(pool) >= total_k:
                break
            seen.add(hit.candidate_key)
            supporting = list(candidate_support.get(hit.candidate_key, []))
            pool.append(
                {
                    "candidate_key": hit.candidate_key,
                    "slot_id": slot_id,
                    "slot_rank": rank,
                    "supporting_slots": supporting,
                }
            )
            slot_counts[slot_id] += 1
        if len(pool) >= total_k:
            break

    # Ensure each slot has at least slot_min_budget candidates.
    for slot_id in slot_order:
        if slot_counts[slot_id] >= slot_min_budget:
            continue
        hits = slot_top[slot_id]
        for rank, hit in enumerate(hits, 1):
            if slot_counts[slot_id] >= slot_min_budget:
                break
            if len(pool) >= total_k:
                break
            if hit.candidate_key in seen:
                continue
            seen.add(hit.candidate_key)
            supporting = list(candidate_support.get(hit.candidate_key, []))
            pool.append(
                {
                    "candidate_key": hit.candidate_key,
                    "slot_id": slot_id,
                    "slot_rank": rank,
                    "supporting_slots": supporting,
                }
            )
            slot_counts[slot_id] += 1

    return pool[:total_k]
