"""Rank-preserving fusion of RRF and Reranker rankings (NF39).

This module implements equal-weight reciprocal rank fusion of two ranked
candidate lists (RRF and Reranker).  It uses **ranks only** — never raw
scores — so that candidates from different scoring spaces can be compared
fairly.

The fusion is **not** wired into the production retrieval pipeline by
default.  It is an offline experiment that may be enabled via
``FINAL_RANKING_MODE=rrf_reranker_fusion`` only after the NF39 Ranking Gate
passes.
"""
from __future__ import annotations

from typing import Any


def candidate_key(candidate: Any) -> str:
    """Return the canonical identity key for a candidate.

    Accepts both dict candidates (``summarize_candidates`` or production
    format) and objects with ``evidence_id``/``document_id``/``block_type``
    /``parent_id`` attributes.

    - ``table_cell`` → ``table_row:{document_id}:{parent_id}`` (cell maps
      to its parent row; raises if ``parent_id`` is missing).
    - ``table_row`` → ``table_row:{document_id}:{evidence_id}``.
    - everything else → ``block:{document_id}:{evidence_id}``.
    """
    if isinstance(candidate, dict):
        evidence_id = (
            candidate.get("evidence_id")
            or candidate.get("candidate_id")
            or candidate.get("doc_id")
            or ""
        )
        document_id = (
            candidate.get("document_id")
            or (candidate.get("metadata") or {}).get("doc_name")
            or ""
        )
        block_type = (
            candidate.get("block_type")
            or (candidate.get("metadata") or {}).get("type")
            or "text"
        )
        parent_id = (
            candidate.get("parent_id")
            or (candidate.get("metadata") or {}).get("parent_id")
        )
    else:
        evidence_id = getattr(candidate, "evidence_id", "") or getattr(
            candidate, "candidate_id", ""
        )
        document_id = getattr(candidate, "document_id", "")
        block_type = getattr(candidate, "block_type", "text")
        parent_id = getattr(candidate, "parent_id", None)

    if block_type == "table_cell":
        if not parent_id:
            raise ValueError("table_cell has no parent row")
        return f"table_row:{document_id}:{parent_id}"
    if block_type == "table_row":
        return f"table_row:{document_id}:{evidence_id}"
    return f"block:{document_id}:{evidence_id}"


def reciprocal_rank_score(
    rank: int | None,
    *,
    k: int = 60,
    missing_rank: int,
) -> float:
    """Return ``1 / (k + effective_rank)``.

    When ``rank`` is ``None`` the candidate was absent from that ranked
    list, so ``missing_rank`` (typically ``len(list) + 1``) is used to
    give it a small but non-zero score.
    """
    effective_rank = rank if rank is not None else missing_rank
    return 1.0 / (k + effective_rank)


def rank_preserving_fusion(
    *,
    rrf_candidates: list[Any],
    reranked_candidates: list[Any],
    fusion_k: int = 60,
) -> list[Any]:
    """Fuse RRF and Reranker rankings using equal reciprocal ranks.

    The fusion:

    1. Assigns 1-based ranks within each input list.
    2. For every candidate in the RRF list, computes
       ``rrf_score + reranker_score`` where each term is
       :func:`reciprocal_rank_score`.
    3. Candidates absent from the reranker output receive
       ``missing_rank = len(rrf_candidates) + 1`` for the reranker term.
    4. Sorts by fused score (desc), then RRF rank (asc), then reranker
       rank (asc), then canonical key (asc) for deterministic tie-breaking.

    Only candidates present in ``rrf_candidates`` participate — the
    reranker list is a subset (or re-ordering) of the same candidates.
    The candidate set is therefore preserved exactly.

    The two terms contribute equally (1:1).  No raw scores are used.
    """
    rrf_rank: dict[str, int] = {}
    for rank, candidate in enumerate(rrf_candidates, start=1):
        key = candidate_key(candidate)
        if key not in rrf_rank:
            rrf_rank[key] = rank

    reranker_rank: dict[str, int] = {}
    for rank, candidate in enumerate(reranked_candidates, start=1):
        key = candidate_key(candidate)
        if key not in reranker_rank:
            reranker_rank[key] = rank

    # Preserve the first occurrence of each candidate from the RRF list
    # so the returned objects are the original RRF candidates.
    candidate_map: dict[str, Any] = {}
    for candidate in rrf_candidates:
        key = candidate_key(candidate)
        if key not in candidate_map:
            candidate_map[key] = candidate

    missing_rank = len(rrf_candidates) + 1

    scored: list[tuple[Any, float, int, int, str]] = []
    for key, candidate in candidate_map.items():
        rr = rrf_rank.get(key, missing_rank)
        rer = reranker_rank.get(key, missing_rank)
        score = reciprocal_rank_score(
            rr, k=fusion_k, missing_rank=missing_rank
        ) + reciprocal_rank_score(
            rer, k=fusion_k, missing_rank=missing_rank
        )
        scored.append((candidate, score, rr, rer, key))

    scored.sort(
        key=lambda item: (
            -item[1],       # fused score descending
            item[2],        # rrf rank ascending
            item[3],        # reranker rank ascending
            item[4],        # canonical key ascending (deterministic)
        )
    )

    return [candidate for candidate, _, _, _, _ in scored]
