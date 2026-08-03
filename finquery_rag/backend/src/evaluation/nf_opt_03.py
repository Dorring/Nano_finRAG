"""Pure helpers for NF-OPT-03 BM25 window evaluation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class NFOpt03Error(ValueError):
    """Raised when an NF-OPT-03 invariant fails."""


def candidate_keys(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = [str(item.get("candidate_key") or "") for item in candidates]
    if any(not key for key in keys):
        raise NFOpt03Error("candidate identity is missing")
    if len(keys) != len(set(keys)):
        raise NFOpt03Error("duplicate candidate identity")
    return keys


def identity_integrity(
    candidates: Sequence[Mapping[str, Any]],
    *,
    allowed_document_ids: set[str],
) -> dict[str, int | bool]:
    missing_identity = 0
    duplicate_keys = 0
    out_of_scope = 0
    seen: set[str] = set()
    for item in candidates:
        key = str(item.get("candidate_key") or "")
        document_id = str(item.get("document_id") or item.get("canonical_document_id") or "")
        content_hash = str(item.get("content_hash") or "")
        if not key or not content_hash:
            missing_identity += 1
        if key and key in seen:
            duplicate_keys += 1
        seen.add(key)
        if document_id not in allowed_document_ids:
            out_of_scope += 1
    return {
        "candidate_count": len(candidates),
        "missing_identity_count": missing_identity,
        "duplicate_candidate_count": duplicate_keys,
        "out_of_scope_candidate_count": out_of_scope,
        "passed": missing_identity == 0 and duplicate_keys == 0 and out_of_scope == 0,
    }


def prefix_integrity(
    current_candidates: Sequence[Mapping[str, Any]],
    expanded_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    current = candidate_keys(current_candidates)
    expanded = candidate_keys(expanded_candidates)
    prefix = expanded[: len(current)]
    return {
        "current_count": len(current),
        "expanded_count": len(expanded),
        "prefix_equal": prefix == current,
        "current_keys_missing_from_expanded": sorted(set(current) - set(expanded)),
        "order_changed": prefix != current,
        "passed": prefix == current,
    }


def source_hit_set(
    rows: Sequence[Mapping[str, Any]],
    stage: str,
    cutoff: int | None = None,
) -> set[str]:
    hits: set[str] = set()
    for row in rows:
        rank = row.get(f"{stage}_rank")
        if not isinstance(rank, int) or (cutoff is not None and rank > cutoff):
            continue
        source_id = row.get("source_index", row.get("candidate_key"))
        hits.add(f"{row['case_id']}:{source_id}")
    return hits


def all_gold_case_set(
    rows: Sequence[Mapping[str, Any]],
    stage: str,
    cutoff: int | None = None,
) -> set[str]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    return {
        case_id
        for case_id, items in grouped.items()
        if items
        and all(
            isinstance(item.get(f"{stage}_rank"), int)
            and (cutoff is None or item[f"{stage}_rank"] <= cutoff)
            for item in items
        )
    }


def compare_sources(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    cutoff: int | None = None,
) -> dict[str, int]:
    old = source_hit_set(before, stage, cutoff)
    new = source_hit_set(after, stage, cutoff)
    return {
        "baseline_source_count": len(old),
        "variant_source_count": len(new),
        "new_source_count": len(new - old),
        "regressed_source_count": len(old - new),
        "both_source_count": len(old & new),
    }


def dynamic_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    limit_by_case: Mapping[str, int],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    source_hits = [
        row
        for row in rows
        if isinstance(row.get(f"{stage}_rank"), int)
        and row[f"{stage}_rank"] <= int(limit_by_case[str(row["case_id"])])
    ]
    all_cases = 0
    partial_cases = 0
    none_cases = 0
    for case_id, items in grouped.items():
        matched = [
            item
            for item in items
            if isinstance(item.get(f"{stage}_rank"), int)
            and item[f"{stage}_rank"] <= int(limit_by_case[case_id])
        ]
        if len(matched) == len(items):
            all_cases += 1
        elif matched:
            partial_cases += 1
        else:
            none_cases += 1
    return {
        "source_hit_count": len(source_hits),
        "source_recall": len(source_hits) / len(rows) if rows else 0.0,
        "all_gold_case_count": all_cases,
        "partial_case_count": partial_cases,
        "none_case_count": none_cases,
    }


def multi_evidence_all_gold(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    multi_case_ids: set[str],
    cutoff: int | None = None,
) -> int:
    return len(all_gold_case_set(rows, stage, cutoff) & multi_case_ids)


def latency_gate(
    *,
    total_ratio: float | None,
    bm25_ratio: float | None,
) -> bool:
    return (
        total_ratio is not None
        and bm25_ratio is not None
        and total_ratio <= 0.20
        and bm25_ratio <= 0.50
    )


def window_gate(
    *,
    complete: bool,
    prefix_passed: bool,
    lineage_passed: bool,
    scope_passed: bool,
    model_calls: int,
    answer_generation_calls: int,
    bm25_source_gain: int,
    bm25_all_gold_gain: int,
    rrf40_source_gain: int,
    reranker20_source_gain: int,
    final5_source_gain: int,
    final_all_gold_gain: int,
    bm25_source_regression: int,
    rrf40_source_regression: int,
    reranker20_source_regression: int,
    final5_source_regression: int,
    rrf_all_gold_regression: int,
    reranker_all_gold_regression: int,
    final_all_gold_regression: int,
    latency_passed: bool,
) -> dict[str, Any]:
    integrity_passed = (
        complete
        and prefix_passed
        and lineage_passed
        and scope_passed
        and model_calls == 0
        and answer_generation_calls == 0
    )
    bm25_gain_passed = bm25_source_gain >= 8 and bm25_all_gold_gain >= 6
    transfer_gain_passed = (
        rrf40_source_gain >= 6
        and reranker20_source_gain >= 5
        and final5_source_gain >= 3
        and final_all_gold_gain >= 2
    )
    regression_passed = (
        bm25_source_regression == 0
        and rrf40_source_regression <= 1
        and reranker20_source_regression <= 1
        and final5_source_regression <= 1
        and rrf_all_gold_regression == 0
        and reranker_all_gold_regression == 0
        and final_all_gold_regression == 0
    )
    passed = integrity_passed and bm25_gain_passed and transfer_gain_passed and regression_passed and latency_passed
    return {
        "passed": passed,
        "integrity_passed": integrity_passed,
        "bm25_gain_passed": bm25_gain_passed,
        "transfer_gain_passed": transfer_gain_passed,
        "regression_passed": regression_passed,
        "latency_gate_passed": latency_passed,
    }


def select_smallest_passing_window(
    gates: Mapping[str, Mapping[str, Any]],
) -> str | None:
    for name in ("B80", "B120", "B200"):
        if gates.get(name, {}).get("passed"):
            return name
    return None


def lineage_subset_report(
    rrf_input: Sequence[Mapping[str, Any]],
    reranker_output: Sequence[Mapping[str, Any]],
    final_output: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rrf = set(candidate_keys(rrf_input))
    reranked = set(candidate_keys(reranker_output))
    final = set(candidate_keys(final_output))
    return {
        "reranker_input_count": len(rrf),
        "reranker_candidate_injection_count": len(reranked - rrf),
        "final_candidate_injection_count": len(final - reranked),
        "reranker_output_is_subset": reranked <= rrf,
        "final_output_is_subset": final <= reranked,
        "passed": reranked <= rrf and final <= reranked,
    }
