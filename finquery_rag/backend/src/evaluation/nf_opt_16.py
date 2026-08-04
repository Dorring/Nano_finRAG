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
