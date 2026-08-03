"""Pure evaluation helpers for NF-OPT-01 Dense coverage shadow A/B."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class Opt01Gate(str, Enum):
    DENSE_COVERAGE_SHADOW = "dense_coverage_shadow"
    CANDIDATE_WINDOW = "candidate_window_expansion"
    DENSE_EMBEDDING = "dense_embedding_ab"
    RRF_ANALYSIS = "rrf_contribution_analysis"
    REGRESSION_STOP = "stop_and_analyze_regression"


def rank_metrics(
    rows: Sequence[Mapping[str, Any]],
    rank_field: str,
    *,
    cutoffs: Sequence[int] = (20, 40, 100, 200),
) -> dict[str, Any]:
    """Return integer-count and rate metrics for one ranked candidate stage."""

    source_count = len(rows)
    case_ids = {str(row.get("case_id")) for row in rows if row.get("case_id")}
    case_count = len(case_ids)
    output: dict[str, Any] = {
        "source_count": source_count,
        "case_count": case_count,
        "rank_field": rank_field,
    }
    for cutoff in cutoffs:
        hits = [
            row
            for row in rows
            if isinstance(row.get(rank_field), int)
            and int(row[rank_field]) <= cutoff
        ]
        source_hits = len(hits)
        case_hits = len({str(row["case_id"]) for row in hits})
        output[f"@{cutoff}"] = {
            "source_hit_count": source_hits,
            "source_recall": source_hits / source_count if source_count else 0.0,
            "case_hit_count": case_hits,
            "case_hit_rate": case_hits / case_count if case_count else 0.0,
        }
    return output


def coverage_state(expected_keys: Iterable[str], candidate_keys: Iterable[str]) -> str:
    expected = {str(key) for key in expected_keys if str(key).strip()}
    present = expected & {
        str(key) for key in candidate_keys if str(key).strip()
    }
    if not present:
        return "none"
    if present == expected:
        return "all"
    return "partial"


def compare_rank_maps(
    current: Mapping[str, int],
    shadow: Mapping[str, int],
    *,
    cutoff: int | None = None,
) -> dict[str, int]:
    """Compare two rank maps without hiding source-level regressions."""

    keys = set(current) | set(shadow)
    current_hits = {key for key in keys if key in current and (cutoff is None or current[key] <= cutoff)}
    shadow_hits = {key for key in keys if key in shadow and (cutoff is None or shadow[key] <= cutoff)}
    return {
        "current_hit_count": len(current_hits),
        "shadow_hit_count": len(shadow_hits),
        "new_hit_count": len(shadow_hits - current_hits),
        "regressed_hit_count": len(current_hits - shadow_hits),
        "both_hit_count": len(current_hits & shadow_hits),
        "both_missed_count": len(keys - current_hits - shadow_hits),
    }


def dense_coverage_gate(
    *,
    shadow_gold_identity_presence: int,
    unsupported_candidate_count: int,
    out_of_scope_candidate_count: int,
    dense_source_gain_at_200: int,
    production_union_source_gain: int,
    rrf_source_gain_at_40: int,
    rrf_all_case_gain: int,
    dense_regressed_sources: int,
    rrf_regressed_sources_at_40: int,
    rrf_regressed_all_cases: int,
    latency_increase_ratio: float | None,
) -> dict[str, Any]:
    """Apply the pre-registered NF-OPT-01 gate without changing production."""

    completeness_passed = (
        shadow_gold_identity_presence == 80
        and unsupported_candidate_count == 0
        and out_of_scope_candidate_count == 0
    )
    gain_passed = (
        dense_source_gain_at_200 >= 10
        and production_union_source_gain >= 8
        and rrf_source_gain_at_40 >= 5
        and rrf_all_case_gain >= 5
        and dense_regressed_sources <= 1
        and rrf_regressed_sources_at_40 <= 1
        and rrf_regressed_all_cases == 0
    )
    latency_passed = latency_increase_ratio is None or latency_increase_ratio <= 0.30
    passed = completeness_passed and gain_passed and latency_passed
    if passed:
        decision = "dense_coverage_shadow_passed"
        next_gate = Opt01Gate.CANDIDATE_WINDOW.value
    elif not latency_passed:
        decision = "dense_coverage_shadow_failed"
        next_gate = Opt01Gate.REGRESSION_STOP.value
    elif completeness_passed and (
        dense_regressed_sources > 1
        or rrf_regressed_sources_at_40 > 1
        or rrf_regressed_all_cases > 0
    ):
        decision = "dense_coverage_shadow_failed"
        next_gate = Opt01Gate.REGRESSION_STOP.value
    elif (
        completeness_passed
        and dense_source_gain_at_200 > 0
        and production_union_source_gain > 0
        and rrf_source_gain_at_40 >= 5
        and rrf_all_case_gain > 0
        and dense_regressed_sources <= 1
        and rrf_regressed_sources_at_40 <= 1
        and rrf_regressed_all_cases == 0
    ):
        # Coverage clearly helps the fused pool even if the stricter
        # production-union gate is not met; the next diagnostic variable is
        # the candidate window, not a production switch or a model change.
        decision = "dense_coverage_shadow_failed"
        next_gate = Opt01Gate.CANDIDATE_WINDOW.value
    elif completeness_passed and dense_source_gain_at_200 > 0 and rrf_source_gain_at_40 < 5:
        decision = "dense_coverage_shadow_failed"
        next_gate = Opt01Gate.RRF_ANALYSIS.value
    elif completeness_passed:
        decision = "dense_coverage_shadow_failed"
        next_gate = Opt01Gate.DENSE_EMBEDDING.value
    else:
        decision = "dense_coverage_shadow_failed"
        next_gate = Opt01Gate.DENSE_COVERAGE_SHADOW.value
    return {
        "decision": decision,
        "passed": passed,
        "production_switch_allowed": False,
        "completeness_passed": completeness_passed,
        "gain_passed": gain_passed,
        "latency_passed": latency_passed,
        "next_gate": next_gate,
        "optimization_allowed": False,
    }


def dense_superset_gate(
    *,
    superset_gold_identity_presence: int,
    unsupported_candidate_count: int,
    out_of_scope_candidate_count: int,
    dense_source_hit_at_200: int,
    rrf_source_hit_at_40: int,
    rrf_top40_all_case_count: int,
    dense_regressed_sources: int,
    rrf_regressed_sources_at_40: int,
    rrf_regressed_all_cases: int,
    rrf_top40_regressed_all_cases: int,
    union_regressed_sources: int,
    latency_increase_ratio: float | None,
) -> dict[str, Any]:
    """Apply the strict zero-regression gate to the Superset variant."""

    completeness_passed = (
        superset_gold_identity_presence == 80
        and unsupported_candidate_count == 0
        and out_of_scope_candidate_count == 0
    )
    gain_passed = (
        dense_source_hit_at_200 >= 50
        and rrf_source_hit_at_40 >= 33
        and rrf_top40_all_case_count >= 27
    )
    regression_passed = (
        dense_regressed_sources == 0
        and rrf_regressed_sources_at_40 == 0
        and rrf_regressed_all_cases == 0
        and rrf_top40_regressed_all_cases == 0
        and union_regressed_sources == 0
    )
    latency_passed = (
        latency_increase_ratio is not None
        and latency_increase_ratio <= 0.20
    )
    passed = completeness_passed and gain_passed and regression_passed and latency_passed
    if passed:
        decision = "dense_superset_passed"
        next_gate = Opt01Gate.CANDIDATE_WINDOW.value
    elif not regression_passed:
        decision = "dense_superset_failed_regression"
        next_gate = Opt01Gate.REGRESSION_STOP.value
    elif not latency_passed:
        decision = "dense_superset_failed_latency"
        next_gate = Opt01Gate.REGRESSION_STOP.value
    else:
        decision = "dense_superset_failed_target"
        next_gate = Opt01Gate.DENSE_COVERAGE_SHADOW.value
    return {
        "decision": decision,
        "passed": passed,
        "production_switch_allowed": False,
        "completeness_passed": completeness_passed,
        "gain_passed": gain_passed,
        "regression_passed": regression_passed,
        "latency_passed": latency_passed,
        "next_gate": next_gate,
        "optimization_allowed": False,
    }


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile suitable for small deterministic benchmark sets."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * fraction + 0.999999)) - 1))
    return ordered[index]


def candidate_scope_ok(document_id: Any, whitelist: set[str]) -> bool:
    return str(document_id or "") in whitelist
