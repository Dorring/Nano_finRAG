"""Pure helpers for NF-OPT-02 R1 transfer evaluation."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any

class NFOpt02R1Error(ValueError):
    """Raised when a transfer-stage invariant is violated."""

def ordered_keys(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = [str(item.get("candidate_key") or "") for item in candidates]
    if any(not key for key in keys):
        raise NFOpt02R1Error("missing candidate identity")
    if len(keys) != len(set(keys)):
        raise NFOpt02R1Error("duplicate candidate identity")
    return keys

def lineage_report(*, rrf_input: Sequence[Mapping[str, Any]], reranker_output: Sequence[Mapping[str, Any]], final_output: Sequence[Mapping[str, Any]], allowed_document_ids: set[str]) -> dict[str, Any]:
    rrf_keys, reranker_keys, final_keys = ordered_keys(rrf_input), ordered_keys(reranker_output), ordered_keys(final_output)
    rrf_set, reranker_set, final_set = set(rrf_keys), set(reranker_keys), set(final_keys)
    out_of_scope = sum(str(item.get("document_id") or item.get("canonical_document_id") or "") not in allowed_document_ids for item in [*rrf_input, *reranker_output, *final_output])
    return {
        "reranker_input_source": "rrf_all",
        "reranker_input_count": len(rrf_keys),
        "reranker_output_count": len(reranker_keys),
        "final_output_count": len(final_keys),
        "reranker_input_missing_identity_count": 0,
        "reranker_candidate_injection_count": len(reranker_set - rrf_set),
        "final_candidate_injection_count": len(final_set - reranker_set),
        "out_of_scope_candidate_count": out_of_scope,
        "reranker_output_is_subset": reranker_set <= rrf_set,
        "final_output_is_subset": final_set <= reranker_set,
        "lineage_passed": reranker_set <= rrf_set and final_set <= reranker_set and out_of_scope == 0,
    }

def source_rows(*, case_id: str, expected_sources: Sequence[Mapping[str, Any]], stage_candidates: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    ranks = {stage: {str(item.get("candidate_key")): rank for rank, item in enumerate(items, start=1)} for stage, items in stage_candidates.items()}
    return [{"case_id": case_id, "source_index": index, "candidate_key": str(source.get("candidate_key") or ""), **{f"{stage}_rank": rank_map.get(str(source.get("candidate_key") or "")) for stage, rank_map in ranks.items()}} for index, source in enumerate(expected_sources)]

def coverage_counts(rows: Sequence[Mapping[str, Any]], stage: str, *, cutoffs: Sequence[int] = (5, 10, 20)) -> dict[str, Any]:
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    result: dict[str, Any] = {}
    for cutoff in cutoffs:
        hits = [row for row in rows if isinstance(row.get(f"{stage}_rank"), int) and row[f"{stage}_rank"] <= cutoff]
        result[f"@{cutoff}"] = {"source_hit_count": len(hits), "source_recall": len(hits) / len(rows) if rows else 0.0, "all_gold_case_count": sum(all(isinstance(item.get(f"{stage}_rank"), int) and item[f"{stage}_rank"] <= cutoff for item in items) for items in by_case.values()), "partial_case_count": sum(any(isinstance(item.get(f"{stage}_rank"), int) and item[f"{stage}_rank"] <= cutoff for item in items) and not all(isinstance(item.get(f"{stage}_rank"), int) and item[f"{stage}_rank"] <= cutoff for item in items) for items in by_case.values()), "none_case_count": sum(not any(isinstance(item.get(f"{stage}_rank"), int) and item[f"{stage}_rank"] <= cutoff for item in items) for items in by_case.values())}
    return result

def full_coverage(rows: Sequence[Mapping[str, Any]], stage: str) -> dict[str, int]:
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    return {"all": sum(all(isinstance(item.get(f"{stage}_rank"), int) for item in items) for items in by_case.values()), "partial": sum(any(isinstance(item.get(f"{stage}_rank"), int) for item in items) and not all(isinstance(item.get(f"{stage}_rank"), int) for item in items) for items in by_case.values()), "none": sum(not any(isinstance(item.get(f"{stage}_rank"), int) for item in items) for items in by_case.values())}

def mrr(rows: Sequence[Mapping[str, Any]], stage: str, *, cutoff: int | None = None) -> float:
    by_case: dict[str, list[int]] = {}
    for row in rows:
        rank = row.get(f"{stage}_rank")
        if isinstance(rank, int) and (cutoff is None or rank <= cutoff):
            by_case.setdefault(str(row["case_id"]), []).append(rank)
    return sum(1.0 / min(ranks) for ranks in by_case.values()) / len(by_case) if by_case else 0.0

def promotion_demotion(rows: Sequence[Mapping[str, Any]], stage: str) -> dict[str, int]:
    counts = {"gold_promotion_count": 0, "gold_demotion_count": 0, "gold_unchanged_count": 0}
    for row in rows:
        old, new = row.get("rrf_rank"), row.get(f"{stage}_rank")
        if not isinstance(old, int) or not isinstance(new, int):
            continue
        counts["gold_promotion_count" if new < old else "gold_demotion_count" if new > old else "gold_unchanged_count"] += 1
    return counts

def hit_set(rows: Sequence[Mapping[str, Any]], stage: str, cutoff: int) -> set[str]:
    return {f"{row['case_id']}:{row.get('source_index', row.get('candidate_key'))}" for row in rows if isinstance(row.get(f"{stage}_rank"), int) and row[f"{stage}_rank"] <= cutoff}

def all_gold_cases(rows: Sequence[Mapping[str, Any]], stage: str, cutoff: int) -> set[str]:
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    return {case_id for case_id, items in by_case.items() if all(isinstance(item.get(f"{stage}_rank"), int) and item[f"{stage}_rank"] <= cutoff for item in items)}

def transfer_gate(*, completeness_passed: bool, lineage_passed: bool, model_calls: int, answer_generation_calls: int, reranker20_source_gain: int, reranker10_source_gain: int, final5_source_gain: int, reranker20_all_gold_gain: int, final_all_gold_gain: int, reranker20_source_regression: int, reranker20_all_gold_regression: int, final_source_regression: int, final_all_gold_regression: int, latency_gate_passed: bool) -> dict[str, Any]:
    transfer_gain = reranker20_source_gain >= 8 and reranker10_source_gain >= 6 and final5_source_gain >= 4 and reranker20_all_gold_gain >= 5 and final_all_gold_gain >= 3
    regression_passed = reranker20_source_regression <= 1 and reranker20_all_gold_regression == 0 and final_source_regression <= 1 and final_all_gold_regression == 0
    complete = completeness_passed and lineage_passed and model_calls == 0 and answer_generation_calls == 0
    passed = complete and transfer_gain and regression_passed and latency_gate_passed
    if not complete or not transfer_gain or not regression_passed:
        decision, next_gate = "protected_residual_transfer_failed", "stop_residual_dense_and_pivot_bm25_window"
    elif not latency_gate_passed:
        decision, next_gate = "protected_residual_transfer_validated_latency_blocked", "parallel_base_residual_query_ab"
    else:
        decision, next_gate = "protected_residual_transfer_validated", "production_config_shadow_validation"
    return {"passed": passed, "transfer_gain_passed": transfer_gain, "regression_passed": regression_passed, "completeness_passed": complete, "latency_gate_passed": latency_gate_passed, "decision": decision, "next_gate": next_gate, "production_switch_allowed": False}

def select_smallest_passing_variant(variants: Mapping[str, Mapping[str, Any]]) -> str | None:
    for name in ("C10", "C20", "C40"):
        if variants.get(name, {}).get("passed"):
            return name
    return None
