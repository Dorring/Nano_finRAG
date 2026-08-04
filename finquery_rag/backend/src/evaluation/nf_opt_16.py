"""Offline BGE-M3 capability guards for the NF-OPT-16 Shadow experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


MODEL_INPUT_EXCLUDED_FIELDS = {
    "expected_answer",
    "expected_sources",
    "expected_value",
    "reference_answer",
    "gold_source",
    "source_index",
}


def validate_model_output(output: Mapping[str, Any], *, expected_rows: int) -> dict[str, Any]:
    """Validate BGE-M3 sparse and late-interaction output without scoring Gold.

    Dense vectors may be emitted by the installed FlagEmbedding release even
    when ``return_dense=False``.  The Shadow contract records that behavior
    but never reads the dense values.
    """
    sparse = output.get("lexical_weights")
    colbert = output.get("colbert_vecs")
    if not isinstance(sparse, Sequence) or len(sparse) != expected_rows:
        raise ValueError("BGE-M3 sparse output row count mismatch")
    if not isinstance(colbert, Sequence) or len(colbert) != expected_rows:
        raise ValueError("BGE-M3 ColBERT output row count mismatch")
    sparse_nonzero = sum(len(row) for row in sparse)
    colbert_token_count = sum(int(getattr(row, "shape", (0,))[0]) for row in colbert)
    if sparse_nonzero == 0:
        raise ValueError("BGE-M3 sparse output contains no lexical weights")
    if colbert_token_count == 0:
        raise ValueError("BGE-M3 ColBERT output contains no token vectors")
    return {
        "returned_keys": sorted(str(key) for key in output),
        "sparse_row_count": len(sparse),
        "sparse_nonzero_count": sparse_nonzero,
        "colbert_row_count": len(colbert),
        "colbert_token_count": colbert_token_count,
        "dense_output_ignored": "dense_vecs" in output,
    }


def assert_query_has_no_expected_fields(question: Mapping[str, Any]) -> None:
    """Fail closed if a query input accidentally carries benchmark answer data."""
    forbidden = MODEL_INPUT_EXCLUDED_FIELDS.intersection(question)
    if forbidden:
        raise ValueError(f"model query includes excluded expected fields: {sorted(forbidden)}")


def stable_smoke_sample(rows: Sequence[Mapping[str, Any]], *, limit: int = 8) -> list[Mapping[str, Any]]:
    """Pick a deterministic corpus-only sample, independent of labels or Gold."""
    usable = [row for row in rows if str(row.get("content") or "").strip()]
    ordered = sorted(usable, key=lambda item: (str(item.get("doc_name") or ""), str(item.get("doc_id") or "")))
    if len(ordered) < limit:
        raise ValueError(f"expected at least {limit} usable candidate rows")
    return ordered[:limit]


def build_sparse_inverted_index(
    lexical_rows: Sequence[Mapping[str, float]],
) -> dict[str, list[tuple[int, float]]]:
    """Build a deterministic in-memory inverted index from lexical weights."""
    inverted: dict[str, list[tuple[int, float]]] = {}
    for row_index, weights in enumerate(lexical_rows):
        for token_id, weight in weights.items():
            numeric_weight = float(weight)
            if numeric_weight <= 0.0:
                continue
            inverted.setdefault(str(token_id), []).append((row_index, numeric_weight))
    return inverted


def sparse_rank(
    *,
    query_weights: Mapping[str, float],
    inverted_index: Mapping[str, Sequence[tuple[int, float]]],
    candidate_keys: Sequence[str],
    limit: int,
) -> list[tuple[int, float]]:
    """Score a lexical query by sparse dot product with stable tie-breaking."""
    scores: dict[int, float] = {}
    for token_id, query_weight in query_weights.items():
        if float(query_weight) <= 0.0:
            continue
        for row_index, candidate_weight in inverted_index.get(str(token_id), ()):
            scores[row_index] = scores.get(row_index, 0.0) + float(query_weight) * candidate_weight
    ordered = sorted(scores.items(), key=lambda item: (-item[1], candidate_keys[item[0]]))
    return ordered[:limit]


def rank_scores(scores: Sequence[float], candidate_keys: Sequence[str], *, limit: int) -> list[tuple[int, float]]:
    """Rank an already-scored candidate set with the same stable identity tie-break."""
    if len(scores) != len(candidate_keys):
        raise ValueError("score and candidate-key counts must match")
    return sorted(enumerate(float(score) for score in scores), key=lambda item: (-item[1], candidate_keys[item[0]]))[:limit]
