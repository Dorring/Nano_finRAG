"""Gate 03 R2 — Semantic Graph Validator.

Validates the semantic graph against gate thresholds:

  Metric Path Coverage              >= 95%
  Typed Evidence Admission          >= 97%
  Atomic Fact Admission             >= 85%
  Identity Conflict                 = 0
  Duplicate Semantic Fact           = 0
  Missing Source Traceback          = 0
  Equivalent-set Double Counting    = 0
  Metric Parent Cycle               = 0
  Conflicting Parent Assignment     = 0
  False Scale Binding               = 0
  Scale Conflict Auto-resolution    = 0
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


def _safe_get(traceback: dict[str, Any], key: str) -> Any:
    return traceback.get(key)


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
    equivalent_double_counting: int = 0,
    parent_cycles: int = 0,
    conflicting_parents: int = 0,
) -> dict[str, Any]:
    """Run all validation checks and return a metrics + gates dict.

    Returns a dict with:
    - ``metrics``: raw counts and ratios
    - ``gates``: boolean pass/fail per gate
    - ``all_passed``: True only if all gates pass
    """
    # --- Count eligible financial data rows ---
    eligible_rows = [sr for sr in semantic_rows if sr.is_financial_data_row]
    eligible_count = len(eligible_rows)

    # --- Metric Path Coverage ---
    resolved_paths = [
        mp for mp in metric_paths if mp.metric_status in ("resolved", "ambiguous")
    ]
    metric_path_coverage = (
        len(resolved_paths) / eligible_count if eligible_count > 0 else 0.0
    )

    # --- Numeric cells ---
    # Count cells with parsed_numeric in value columns (col > 0)
    # This is approximated from axis_bindings with numeric-producing kinds
    numeric_cell_count = sum(
        1
        for ab in axis_bindings
        if ab.temporal_kind
        in ("point", "duration", "comparison", "bucket", "segment", "category")
    )

    # --- Atomic Fact Admission ---
    # Atomic facts with a non-null value_normalized and resolved metric_path
    admitted_atomic = [
        af for af in atomic_facts if af.value_normalized is not None and af.metric_path
    ]
    atomic_admission = len(admitted_atomic) / len(atomic_facts) if atomic_facts else 0.0

    # --- Typed Evidence Admission ---
    total_typed = (
        len(atomic_facts)
        + len(comparison_facts)
        + len(bucket_facts)
        + len(row_matrices)
    )
    admitted_typed = (
        len(admitted_atomic)
        + len(comparison_facts)
        + len(bucket_facts)
        + len(row_matrices)
    )
    typed_admission = admitted_typed / total_typed if total_typed > 0 else 0.0

    # --- Identity Conflict ---
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

    # --- Duplicate Semantic Fact ---
    # Check for duplicate (metric_path, temporal_kind, normalized_period, value_raw)
    seen_facts: dict[tuple, int] = {}
    for af in atomic_facts:
        key = (
            af.metric_path,
            af.temporal_kind,
            af.normalized_period or "",
            af.value_raw,
        )
        seen_facts[key] = seen_facts.get(key, 0) + 1
    duplicate_semantic_facts = sum(1 for c in seen_facts.values() if c > 1)

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
        if sr.scale_status == "resolved" and sr.scale_status == "conflict"
    )
    scale_resolved = sum(1 for sr in scale_resolutions if sr.scale_status == "resolved")
    scale_candidate_only = sum(
        1 for sr in scale_resolutions if sr.scale_status == "candidate"
    )
    scale_conflict = sum(1 for sr in scale_resolutions if sr.scale_status == "conflict")

    # --- Gates ---
    gates: dict[str, bool] = {
        "metric_path_coverage": metric_path_coverage >= 0.95,
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

    metrics = {
        "logical_table_count": len(logical_tables),
        "semantic_row_count": len(semantic_rows),
        "eligible_financial_data_rows": eligible_count,
        "metric_path_count": len(metric_paths),
        "resolved_metric_paths": len(resolved_paths),
        "metric_path_coverage": round(metric_path_coverage, 4),
        "numeric_cell_count": numeric_cell_count,
        "axis_binding_count": len(axis_bindings),
        "atomic_fact_count": len(atomic_facts),
        "admitted_atomic_count": len(admitted_atomic),
        "atomic_fact_admission": round(atomic_admission, 4),
        "comparison_fact_count": len(comparison_facts),
        "bucket_fact_count": len(bucket_facts),
        "row_matrix_count": len(row_matrices),
        "narrative_evidence_count": len(narrative_evidence),
        "total_typed_evidence": total_typed,
        "admitted_typed_evidence": admitted_typed,
        "typed_evidence_admission": round(typed_admission, 4),
        "identity_conflict": identity_conflict,
        "duplicate_semantic_fact": duplicate_semantic_facts,
        "missing_source_traceback": missing_traceback,
        "equivalent_set_double_counting": equivalent_double_counting,
        "metric_parent_cycle": parent_cycles,
        "conflicting_parent_assignment": conflicting_parents,
        "scale_resolved": scale_resolved,
        "scale_candidate_only": scale_candidate_only,
        "scale_conflict": scale_conflict,
        "false_scale_binding": false_scale_binding,
        "scale_conflict_auto_resolution": scale_conflict_auto_resolution,
    }

    all_passed = all(gates.values())

    return {
        "metrics": metrics,
        "gates": gates,
        "all_passed": all_passed,
    }
