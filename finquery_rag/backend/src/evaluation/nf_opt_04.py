"""Pure helpers for NF-OPT-04 final evidence-budget evaluation."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import re
from typing import Any


class NFOpt04Error(ValueError):
    """Raised when a final-evidence budget invariant is violated."""


def ordered_keys(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = [str(candidate.get("candidate_key") or "") for candidate in candidates]
    if any(not key for key in keys):
        raise NFOpt04Error("candidate identity is missing")
    if len(keys) != len(set(keys)):
        raise NFOpt04Error("duplicate candidate identity")
    return keys


def select_prefix(candidates: Sequence[Mapping[str, Any]], *, max_evidence: int) -> list[dict[str, Any]]:
    if max_evidence < 1:
        raise NFOpt04Error("max_evidence must be positive")
    return [dict(candidate) for candidate in candidates[:max_evidence]]


def select_token_budget(
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_evidence: int,
    token_budget: int,
    count_context_tokens: Any,
) -> list[dict[str, Any]]:
    if max_evidence < 1 or token_budget < 1:
        raise NFOpt04Error("token budget and max evidence must be positive")
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if len(selected) >= max_evidence:
            break
        proposed = [*selected, dict(candidate)]
        if int(count_context_tokens(proposed)) > token_budget:
            continue
        selected.append(dict(candidate))
    return selected


def prefix_report(
    baseline: Sequence[Mapping[str, Any]],
    expanded: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base_keys = ordered_keys(baseline)
    expanded_keys = ordered_keys(expanded)
    return {
        "baseline_candidate_count": len(base_keys),
        "expanded_candidate_count": len(expanded_keys),
        "baseline_is_prefix": expanded_keys[: len(base_keys)] == base_keys,
        "baseline_missing_from_expanded_count": len(set(base_keys) - set(expanded_keys)),
        "passed": expanded_keys[: len(base_keys)] == base_keys,
    }


def hit_set(rows: Sequence[Mapping[str, Any]], *, stage: str) -> set[str]:
    return {
        f"{row['case_id']}:{row['source_index']}"
        for row in rows
        if isinstance(row.get(f"{stage}_rank"), int)
    }


def all_gold_cases(rows: Sequence[Mapping[str, Any]], *, stage: str) -> set[str]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    return {
        case_id
        for case_id, values in grouped.items()
        if values and all(isinstance(value.get(f"{stage}_rank"), int) for value in values)
    }


def coverage(rows: Sequence[Mapping[str, Any]], *, stage: str) -> dict[str, int | float]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    hits = hit_set(rows, stage=stage)
    all_cases = all_gold_cases(rows, stage=stage)
    partial = sum(
        any(isinstance(value.get(f"{stage}_rank"), int) for value in values)
        and case_id not in all_cases
        for case_id, values in grouped.items()
    )
    return {
        "source_hit_count": len(hits),
        "source_recall": len(hits) / len(rows) if rows else 0.0,
        "case_hit_count": sum(
            any(isinstance(value.get(f"{stage}_rank"), int) for value in values)
            for values in grouped.values()
        ),
        "all_gold_case_count": len(all_cases),
        "partial_case_count": partial,
        "none_case_count": len(grouped) - len(all_cases) - partial,
    }


def rank_bucket(rank: int | None) -> str:
    if not isinstance(rank, int):
        return "below_20"
    if rank <= 5:
        return "1_5"
    if rank <= 8:
        return "6_8"
    if rank <= 10:
        return "9_10"
    if rank <= 20:
        return "11_20"
    return "below_20"


def rank_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(rank_bucket(row.get("reranker_rank")) for row in rows)
    return {key: int(counts.get(key, 0)) for key in ("1_5", "6_8", "9_10", "11_20", "below_20")}


def _metric_key(candidate: Mapping[str, Any]) -> str | None:
    metadata = candidate.get("metadata") or {}
    for field in ("row_label", "metric", "table_title"):
        value = str(metadata.get(field) or "").strip().lower()
        if value:
            return value
    return None


def _periods(candidate: Mapping[str, Any]) -> set[str]:
    metadata = candidate.get("metadata") or {}
    value = str(metadata.get("period") or "")
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", value))
    if years:
        return years
    return set(re.findall(r"\b(?:19|20)\d{2}\b", str(candidate.get("content") or "")))


def _values(candidate: Mapping[str, Any]) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z0-9])\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?", str(candidate.get("content") or "")))


def context_quality(candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    hashes = [str(candidate.get("content_hash") or "") for candidate in candidates]
    duplicate_evidence = len(hashes) - len(set(hashes))
    pages = [
        (str(candidate.get("document_id") or ""), str(candidate.get("page") or (candidate.get("metadata") or {}).get("page") or ""))
        for candidate in candidates
    ]
    same_page = len(pages) - len(set(pages))
    metric_groups: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        key = _metric_key(candidate)
        if key:
            metric_groups.setdefault(key, []).append(candidate)
    same_metric = sum(len(items) - 1 for items in metric_groups.values() if len(items) > 1)
    conflicting_period = 0
    conflicting_value = 0
    for items in metric_groups.values():
        if len(items) < 2:
            continue
        periods = set().union(*(_periods(item) for item in items))
        values = set().union(*(_values(item) for item in items))
        conflicting_period += int(len(periods) > 1)
        conflicting_value += int(len(values) > 1)
    return {
        "duplicate_evidence_count": duplicate_evidence,
        "same_page_duplicate_count": same_page,
        "same_metric_duplicate_count": same_metric,
        "conflicting_period_count": conflicting_period,
        "conflicting_value_count": conflicting_value,
    }


def final_budget_gate(
    *,
    integrity_passed: bool,
    source_hit_count: int,
    all_gold_case_count: int,
    multi_evidence_all_gold_count: int,
    new_source_count: int,
    new_all_gold_case_count: int,
    source_regression_count: int,
    all_gold_regression_count: int,
    multi_evidence_regression_count: int,
    conflicting_period_case_increase: int,
    conflicting_value_case_increase: int,
    duplicate_case_rate: float,
    context_token_p95: float,
    total_latency_ratio: float | None,
) -> dict[str, bool]:
    gain = (
        source_hit_count >= 20
        and all_gold_case_count >= 15
        and multi_evidence_all_gold_count >= 4
        and new_source_count >= 7
        and new_all_gold_case_count >= 5
    )
    regression = (
        source_regression_count == 0
        and all_gold_regression_count == 0
        and multi_evidence_regression_count == 0
    )
    quality = (
        conflicting_period_case_increase <= 4
        and conflicting_value_case_increase <= 3
        and duplicate_case_rate <= 0.30
        and context_token_p95 <= 8000
    )
    latency = total_latency_ratio is not None and total_latency_ratio <= 0.15
    return {
        "passed": integrity_passed and gain and regression and quality and latency,
        "integrity_passed": integrity_passed,
        "gain_passed": gain,
        "regression_passed": regression,
        "context_quality_passed": quality,
        "latency_gate_passed": latency,
    }


def select_smallest_passing_variant(gates: Mapping[str, Mapping[str, Any]]) -> str | None:
    for name in ("F8", "FT8", "F10", "FT10"):
        if gates.get(name, {}).get("passed"):
            return name
    return None


def final_budget_decision(
    *,
    selected_variant: str | None,
    context_quality_blocked: bool,
) -> tuple[str, str]:
    """Return the next gate without conflating low gain with context conflict."""
    if selected_variant in {"F8", "FT8"}:
        return "final_evidence_budget_validated", "answer_chain_shadow_validation"
    if selected_variant:
        return (
            "final_evidence_budget_validated_with_large_context",
            "answer_chain_context_quality_validation",
        )
    if context_quality_blocked:
        return "final_budget_blocked_by_context_conflict", "evidence_deduplication_ab"
    return (
        "final_budget_gain_insufficient",
        "stop_context_expansion_and_start_calculation_route",
    )
