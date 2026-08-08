"""Gate 03 R2 — Semantic Graph Validator.

Validates the semantic graph against gate thresholds:

  Resolved Metric Path Coverage       >= 95%
  Typed Evidence Admission            >= 97%
  Atomic Fact Admission               >= 85%
  Identity Conflict                   = 0
  Duplicate Semantic Fact             = 0
  Missing Source Traceback            = 0
  Equivalent-set Double Counting      = 0
  Metric Parent Cycle                 = 0
  Conflicting Parent Assignment       = 0
  False Scale Binding                 = 0
  Scale Conflict Auto-resolution      = 0
"""

from __future__ import annotations

from typing import Any

from src.pdf_retrieval_v4.semantic_graph_models import (
    AtomicFact,
    BucketFact,
    ComparisonFact,
    MetricPath,
    NarrativeEvidence,
    RowMatrix,
    ScaleResolution,
    SemanticAxisBinding,
    SemanticRow,
)
from src.pdf_retrieval_v4.semantic_scale_resolver import resolve_scale_keyword
from src.pdf_retrieval_v4.typed_evidence_emitters import (
    ADMITTED_OUTCOMES,
    ATOMIC_ELIGIBLE_KINDS,
    TYPED_ELIGIBLE_KINDS,
    compute_admission_outcomes,
)


def _safe_get(traceback: dict[str, Any], key: str) -> Any:
    return traceback.get(key)


def _has_conflicting_scale_candidates(sr: ScaleResolution) -> bool:
    """Check if raw_candidates contain keywords resolving to different units."""
    units: set[str] = set()
    for kw in sr.raw_candidates:
        resolved = resolve_scale_keyword(kw)
        if resolved:
            units.add(resolved[1])
    return len(units) > 1


def validate_semantic_graph(
    logical_tables: list[Any],
    semantic_rows: list[SemanticRow],
    metric_paths: list[MetricPath],
    axis_bindings: list[SemanticAxisBinding],
    scale_resolutions: list[ScaleResolution],
    atomic_facts: list[AtomicFact],
    comparison_facts: list[ComparisonFact],
    bucket_facts: list[BucketFact],
    row_matrices: list[RowMatrix],
    narrative_evidence: list[NarrativeEvidence],
    all_cells: list[dict[str, Any]],
    equivalent_double_counting: int = 0,
    parent_cycles: int = 0,
    conflicting_parents: int = 0,
) -> dict[str, Any]:
    """Run all validation checks and return a metrics + gates dict.

    Parameters
    ----------
    all_cells
        All raw cell dicts from adapter predictions (flattened across all
        tables).  Used to compute the pre-emission eligible numeric cell
        denominator for admission metrics.

    Returns a dict with:
    - ``metrics``: raw counts and ratios
    - ``gates``: boolean pass/fail per gate
    - ``all_passed``: True only if all gates pass
    """
    # --- Count eligible financial data rows ---
    eligible_rows = [sr for sr in semantic_rows if sr.is_financial_data_row]
    eligible_count = len(eligible_rows)

    # --- Metric Path Coverage (split: present / resolved / ambiguous / missing) ---
    present_paths = [
        mp for mp in metric_paths if mp.metric_status in ("resolved", "ambiguous")
    ]
    resolved_paths = [mp for mp in metric_paths if mp.metric_status == "resolved"]
    ambiguous_paths = [mp for mp in metric_paths if mp.metric_status == "ambiguous"]
    missing_paths = [mp for mp in metric_paths if mp.metric_status == "missing"]

    metric_path_present_coverage = (
        len(present_paths) / eligible_count if eligible_count > 0 else 0.0
    )
    metric_path_resolved_coverage = (
        len(resolved_paths) / eligible_count if eligible_count > 0 else 0.0
    )

    # --- Admission Outcomes (pre-emission denominator) ---
    admission_outcomes = compute_admission_outcomes(
        semantic_rows=semantic_rows,
        metric_paths=metric_paths,
        axis_bindings=axis_bindings,
        all_cells=all_cells,
        atomic_facts=atomic_facts,
        comparison_facts=comparison_facts,
        bucket_facts=bucket_facts,
        row_matrices=row_matrices,
    )

    eligible_numeric_cells = len(admission_outcomes)

    atomic_eligible_cells = [
        ao for ao in admission_outcomes if ao["temporal_kind"] in ATOMIC_ELIGIBLE_KINDS
    ]
    atomic_eligible_count = len(atomic_eligible_cells)

    typed_eligible_cells = [
        ao for ao in admission_outcomes if ao["temporal_kind"] in TYPED_ELIGIBLE_KINDS
    ]
    typed_eligible_count = len(typed_eligible_cells)

    atomic_admitted_count = sum(
        1 for ao in atomic_eligible_cells if "atomic" in ao["outcomes"]
    )

    typed_covered_count = sum(
        1 for ao in typed_eligible_cells if ao["outcomes"] & ADMITTED_OUTCOMES
    )

    atomic_admission = (
        atomic_admitted_count / atomic_eligible_count
        if atomic_eligible_count > 0
        else 0.0
    )
    typed_admission = (
        typed_covered_count / typed_eligible_count if typed_eligible_count > 0 else 0.0
    )

    # --- Identity Conflict (duplicate semantic_fact_id) ---
    all_fact_ids: list[str] = []
    all_fact_ids.extend(af.semantic_fact_id for af in atomic_facts)
    all_fact_ids.extend(cf.semantic_fact_id for cf in comparison_facts)
    all_fact_ids.extend(bf.semantic_fact_id for bf in bucket_facts)
    all_fact_ids.extend(rm.semantic_fact_id for rm in row_matrices)
    all_fact_ids.extend(ne.semantic_evidence_id for ne in narrative_evidence)

    id_counts: dict[str, int] = {}
    for fid in all_fact_ids:
        id_counts[fid] = id_counts.get(fid, 0) + 1
    identity_conflict = sum(1 for c in id_counts.values() if c > 1)

    # --- Duplicate Semantic Fact (duplicate semantic_fact_id within atomic) ---
    atomic_id_counts: dict[str, int] = {}
    for af in atomic_facts:
        atomic_id_counts[af.semantic_fact_id] = (
            atomic_id_counts.get(af.semantic_fact_id, 0) + 1
        )
    duplicate_semantic_facts = sum(1 for c in atomic_id_counts.values() if c > 1)

    # --- Missing Source Traceback ---
    all_facts_with_trace = (
        list(atomic_facts)
        + list(comparison_facts)
        + list(bucket_facts)
        + list(row_matrices)
        + list(narrative_evidence)
    )
    missing_traceback = 0
    for fact in all_facts_with_trace:
        tb = fact.source_traceback
        if (
            not tb
            or not _safe_get(tb, "document_id")
            or _safe_get(tb, "pdf_page") is None
        ):
            missing_traceback += 1

    # --- Scale metrics ---
    false_scale_binding = sum(
        1
        for sr in scale_resolutions
        if sr.scale_status == "resolved" and sr.scale_level in ("S5", "S6")
    )

    scale_conflict_auto_resolution = sum(
        1
        for sr in scale_resolutions
        if sr.scale_status == "resolved" and _has_conflicting_scale_candidates(sr)
    )

    scale_resolved = sum(1 for sr in scale_resolutions if sr.scale_status == "resolved")
    scale_candidate_only = sum(
        1 for sr in scale_resolutions if sr.scale_status == "candidate"
    )
    scale_conflict = sum(1 for sr in scale_resolutions if sr.scale_status == "conflict")
    scale_missing = sum(1 for sr in scale_resolutions if sr.scale_status == "missing")

    scale_conflict_detected = sum(
        1 for sr in scale_resolutions if _has_conflicting_scale_candidates(sr)
    )

    # --- Gates ---
    gates: dict[str, bool] = {
        "metric_path_resolved_coverage": metric_path_resolved_coverage >= 0.95,
        "typed_evidence_admission": typed_admission >= 0.97,
        "atomic_fact_admission": atomic_admission >= 0.85,
        "identity_conflict": identity_conflict == 0,
        "duplicate_semantic_fact": duplicate_semantic_facts == 0,
        "missing_source_traceback": missing_traceback == 0,
        "equivalent_set_double_counting": equivalent_double_counting == 0,
        "metric_parent_cycle": parent_cycles == 0,
        "conflicting_parent_assignment": conflicting_parents == 0,
        "false_scale_binding": false_scale_binding == 0,
        "scale_conflict_auto_resolution": scale_conflict_auto_resolution == 0,
    }

    # Admission outcome breakdown
    outcome_breakdown: dict[str, int] = {}
    for ao in admission_outcomes:
        for outcome in ao["outcomes"]:
            outcome_breakdown[outcome] = outcome_breakdown.get(outcome, 0) + 1

    metrics = {
        "logical_table_count": len(logical_tables),
        "semantic_row_count": len(semantic_rows),
        "eligible_financial_data_rows": eligible_count,
        "metric_path_count": len(metric_paths),
        "metric_path_present": len(present_paths),
        "metric_path_resolved": len(resolved_paths),
        "metric_path_ambiguous": len(ambiguous_paths),
        "metric_path_missing": len(missing_paths),
        "metric_path_present_coverage": round(metric_path_present_coverage, 4),
        "metric_path_resolved_coverage": round(metric_path_resolved_coverage, 4),
        "eligible_numeric_cells": eligible_numeric_cells,
        "atomic_eligible_cells": atomic_eligible_count,
        "typed_eligible_cells": typed_eligible_count,
        "atomic_admitted": atomic_admitted_count,
        "typed_covered": typed_covered_count,
        "atomic_fact_admission": round(atomic_admission, 4),
        "typed_evidence_admission": round(typed_admission, 4),
        "axis_binding_count": len(axis_bindings),
        "atomic_fact_count": len(atomic_facts),
        "comparison_fact_count": len(comparison_facts),
        "bucket_fact_count": len(bucket_facts),
        "row_matrix_count": len(row_matrices),
        "narrative_evidence_count": len(narrative_evidence),
        "identity_conflict": identity_conflict,
        "duplicate_semantic_fact": duplicate_semantic_facts,
        "missing_source_traceback": missing_traceback,
        "equivalent_set_double_counting": equivalent_double_counting,
        "metric_parent_cycle": parent_cycles,
        "conflicting_parent_assignment": conflicting_parents,
        "scale_resolved": scale_resolved,
        "scale_candidate_only": scale_candidate_only,
        "scale_conflict": scale_conflict,
        "scale_missing": scale_missing,
        "scale_conflict_detected": scale_conflict_detected,
        "false_scale_binding": false_scale_binding,
        "scale_conflict_auto_resolution": scale_conflict_auto_resolution,
        "admission_outcome_breakdown": outcome_breakdown,
    }

    all_passed = all(gates.values())

    return {
        "metrics": metrics,
        "gates": gates,
        "all_passed": all_passed,
    }
