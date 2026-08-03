"""Pure, production-independent helpers for NF-OPT-02.

The protected residual experiment keeps the production Dense Top-40 as an
ordered prefix and appends only candidates absent from that collection.  The
helpers deliberately operate on candidate identities rather than labels so
the evaluation runner cannot shape a candidate universe around Gold answers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class NFOpt02Error(ValueError):
    """Raised when protected-residual invariants are violated."""


def candidate_key(candidate: Mapping[str, Any]) -> str:
    key = str(candidate.get("candidate_key") or "").strip()
    if not key:
        raise NFOpt02Error("candidate_key is required")
    return key


def residual_candidate_keys(
    *,
    canonical_keys: set[str],
    current_keys: set[str],
) -> set[str]:
    """Return the label-independent Canonical-minus-Current universe."""

    if not current_keys.issubset(canonical_keys):
        raise NFOpt02Error("current candidate keys are not canonical candidates")
    return canonical_keys - current_keys


def protected_dense_merge(
    *,
    base_candidates: Sequence[Mapping[str, Any]],
    residual_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep Base ordering exactly and append unique Residual candidates."""

    base = [dict(candidate) for candidate in base_candidates]
    base_keys = [candidate_key(candidate) for candidate in base]
    if len(base_keys) != len(set(base_keys)):
        raise NFOpt02Error("base candidate keys must be unique")

    output = list(base)
    seen = set(base_keys)
    for candidate in residual_candidates:
        item = dict(candidate)
        key = candidate_key(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def base_retention(
    *,
    base_candidates: Sequence[Mapping[str, Any]],
    protected_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, int | bool]:
    """Audit exact Base prefix preservation before RRF receives candidates."""

    base_keys = [candidate_key(candidate) for candidate in base_candidates]
    protected_prefix = [
        candidate_key(candidate) for candidate in protected_candidates[: len(base_keys)]
    ]
    missing = len(set(base_keys) - {candidate_key(item) for item in protected_candidates})
    return {
        "base_candidate_count": len(base_keys),
        "base_candidate_retention_count": len(base_keys) - missing,
        "base_candidate_missing_count": missing,
        "base_candidate_order_changed_count": int(base_keys != protected_prefix),
        "base_prefix_preserved": base_keys == protected_prefix,
    }


def compare_hit_sets(
    *,
    baseline_hits: set[str],
    variant_hits: set[str],
) -> dict[str, int]:
    return {
        "baseline_hit_count": len(baseline_hits),
        "variant_hit_count": len(variant_hits),
        "new_hit_count": len(variant_hits - baseline_hits),
        "regressed_hit_count": len(baseline_hits - variant_hits),
        "both_hit_count": len(baseline_hits & variant_hits),
    }


def protected_residual_gate(
    *,
    case_count: int,
    source_count: int,
    overlap_count: int,
    identity_conflict_count: int,
    out_of_scope_count: int,
    base_missing_count: int,
    base_order_changed_count: int,
    dense_gold_regressions: int,
    union_source_regressions: int,
    rrf_full_source_regressions: int,
    rrf_top40_source_regressions: int,
    rrf_full_all_gold_regressions: int,
    rrf_top40_all_gold_regressions: int,
    union_source_hits: int,
    rrf_full_source_hits: int,
    rrf_top40_source_hits: int,
    rrf_top40_all_gold_cases: int,
    dense_latency_ratio: float | None,
    retrieval_latency_ratio: float | None,
) -> dict[str, Any]:
    """Apply NF-OPT-02's registered completeness, safety, gain and latency gates."""

    completeness_passed = (
        case_count == 64
        and source_count == 80
        and overlap_count == 0
        and identity_conflict_count == 0
        and out_of_scope_count == 0
    )
    regression_passed = (
        base_missing_count == 0
        and base_order_changed_count == 0
        and dense_gold_regressions == 0
        and union_source_regressions == 0
        and rrf_full_source_regressions == 0
        and rrf_top40_source_regressions == 0
        and rrf_full_all_gold_regressions == 0
        and rrf_top40_all_gold_regressions == 0
    )
    gain_passed = (
        union_source_hits >= 28
        and rrf_full_source_hits >= 36
        and rrf_top40_source_hits >= 30
        and rrf_top40_all_gold_cases >= 24
    )
    latency_passed = all(
        ratio is None or ratio <= 0.25
        for ratio in (dense_latency_ratio, retrieval_latency_ratio)
    )
    passed = completeness_passed and regression_passed and gain_passed and latency_passed
    return {
        "passed": passed,
        "decision": (
            "protected_residual_dense_validated"
            if passed
            else "protected_residual_dense_not_validated"
        ),
        "completeness_passed": completeness_passed,
        "regression_passed": regression_passed,
        "gain_passed": gain_passed,
        "latency_passed": latency_passed,
        "production_switch_allowed": False,
        "next_gate": (
            "production_config_shadow_validation"
            if passed
            else "protected_residual_candidate_ab"
        ),
    }


def select_smallest_passing_variant(
    variants: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Select the first passing budget in the pre-registered C10/C20/C40 order."""

    for name in ("C10", "C20", "C40"):
        if bool(variants.get(name, {}).get("passed")):
            return name
    return None
