"""Pure identity and collision checks for NF-OPT-17 Gate D."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def shadow_candidate_key(
    *,
    document_id: str,
    table_index: int,
    row_index: int,
    content: str,
) -> str:
    """Create a query-independent table-row candidate identity."""
    identity = "|".join((document_id, str(table_index), str(row_index), content))
    return f"shadow:sec:{stable_hash(identity)}"


def metadata_hash(metadata: Mapping[str, Any]) -> str:
    return stable_hash(json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def classify_pair_distinguishability(
    *,
    positive: Mapping[str, Any],
    negative: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify whether a fact-level pair is learnable from candidate inputs."""
    key_distinct = positive["shadow_candidate_key"] != negative["shadow_candidate_key"]
    content_distinct = positive["content_sha256"] != negative["content_sha256"]
    metadata_distinct = positive["metadata_sha256"] != negative["metadata_sha256"]
    reranker_input_distinct = positive["reranker_input_sha256"] != negative["reranker_input_sha256"]
    cross_encoder_input_distinct = positive["cross_encoder_input_sha256"] != negative["cross_encoder_input_sha256"]
    if not key_distinct:
        status = "candidate_identity_collision"
    elif not reranker_input_distinct:
        status = "model_input_collision"
    elif not content_distinct and metadata_distinct:
        status = "metadata_only_distinguishable"
    else:
        status = "trainable_pair"
    return {
        "candidate_key_distinct": key_distinct,
        "candidate_content_distinct": content_distinct,
        "metadata_distinct": metadata_distinct,
        "reranker_input_distinct": reranker_input_distinct,
        "cross_encoder_input_distinct": cross_encoder_input_distinct,
        "status": status,
        "trainable_with_current_heuristic": status == "trainable_pair",
        "trainable_with_content_only_cross_encoder": status == "trainable_pair" and cross_encoder_input_distinct,
    }


def issuer_grouped_folds(
    *,
    issuer_to_document_ids: Mapping[str, Sequence[str]],
    issuer_to_candidate_keys: Mapping[str, Sequence[str]],
    issuer_to_query_ids: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Freeze leave-one-issuer-out folds and prove all identities are disjoint."""
    issuers = sorted(issuer_to_document_ids)
    folds: list[dict[str, Any]] = []
    for holdout in issuers:
        train = [issuer for issuer in issuers if issuer != holdout]
        holdout_documents = set(issuer_to_document_ids[holdout])
        holdout_candidates = set(issuer_to_candidate_keys[holdout])
        holdout_queries = set(issuer_to_query_ids[holdout])
        train_documents = set().union(*(set(issuer_to_document_ids[issuer]) for issuer in train))
        train_candidates = set().union(*(set(issuer_to_candidate_keys[issuer]) for issuer in train))
        train_queries = set().union(*(set(issuer_to_query_ids[issuer]) for issuer in train))
        if holdout_documents & train_documents:
            raise ValueError("issuer fold document overlap")
        if holdout_candidates & train_candidates:
            raise ValueError("issuer fold candidate overlap")
        if holdout_queries & train_queries:
            raise ValueError("issuer fold query overlap")
        folds.append(
            {
                "fold_id": f"leave-{holdout.casefold().replace(' ', '-')}-out",
                "holdout_issuer": holdout,
                "train_issuers": train,
                "train_document_count": len(train_documents),
                "holdout_document_count": len(holdout_documents),
                "train_candidate_count": len(train_candidates),
                "holdout_candidate_count": len(holdout_candidates),
                "train_query_count": len(train_queries),
                "holdout_query_count": len(holdout_queries),
                "document_overlap_count": 0,
                "candidate_overlap_count": 0,
                "query_overlap_count": 0,
            }
        )
    return folds


def render_structured_reranker_input(candidate: Mapping[str, Any]) -> str:
    """Render only source-side fields available to the shadow reranker."""
    required = (
        "issuer",
        "xbrl_concept",
        "period_end",
        "period_kind",
        "content",
    )
    missing = [field for field in required if not str(candidate.get(field) or "").strip()]
    if missing:
        raise ValueError(f"shadow reranker candidate is missing fields: {missing}")
    return "\n".join(
        (
            f"issuer: {candidate['issuer']}",
            f"metric: {candidate['xbrl_concept']}",
            f"period: {candidate['period_end']}",
            f"period kind: {candidate['period_kind']}",
            f"evidence: {candidate['content']}",
        )
    )


def rank_triplet(scores: Sequence[float], negative_types: Sequence[str]) -> dict[str, Any]:
    """Score one positive-first triplet without consulting benchmark labels."""
    if len(scores) != 3 or len(negative_types) != 2:
        raise ValueError("each development query requires one positive and two negatives")
    positive = float(scores[0])
    negative_scores = [float(value) for value in scores[1:]]
    order = sorted(range(3), key=lambda index: (-float(scores[index]), index))
    positive_rank = order.index(0) + 1
    return {
        "positive_rank": positive_rank,
        "positive_top1": positive_rank == 1,
        "reciprocal_rank": 1.0 / positive_rank,
        "pairwise": {
            str(negative_type): positive > negative_score
            for negative_type, negative_score in zip(negative_types, negative_scores, strict=True)
        },
    }
