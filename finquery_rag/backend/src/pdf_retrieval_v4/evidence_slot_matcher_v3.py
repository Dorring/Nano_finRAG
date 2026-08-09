from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize(value: object) -> str:
    return " ".join(
        re.findall(
            r"[a-z0-9]+", unicodedata.normalize("NFKC", str(value or "")).casefold()
        )
    )


def _metric(payload: dict[str, Any]) -> tuple[str, str, set[str]]:
    path = normalize(payload.get("metric_path"))
    leaf = normalize(payload.get("leaf_metric"))
    segments = {
        normalize(item)
        for item in str(payload.get("metric_path") or "").split("/")
        if normalize(item)
    }
    if leaf:
        segments.add(leaf)
    return path, leaf, segments


def metric_grade(slot: dict[str, Any], payload: dict[str, Any]) -> str | None:
    phrase = normalize(slot.get("raw_metric_phrase"))
    path, leaf, segments = _metric(payload)
    if not phrase:
        return "M3_concept"
    if phrase == leaf:
        return "M0_exact_leaf"
    if phrase == path:
        return "M1_exact_path"
    if phrase in segments:
        return "M2_path_segment"
    concepts = {
        normalize(item)
        for item in slot.get("concept_candidates") or []
        if normalize(item)
    }
    if leaf in concepts or path in concepts or concepts.intersection(segments):
        return "M3_concept"
    return None


def _statement_family(value: object) -> str | None:
    text = normalize(value)
    families = {
        "cash_flow": ("cash flow", "cash flows"),
        "balance_sheet": ("balance sheet", "balance sheets", "financial position"),
        "income_statement": (
            "income statement",
            "statements of operations",
            "statement of operations",
        ),
        "segment": ("segment", "segments"),
    }
    return next(
        (
            name
            for name, phrases in families.items()
            if any(phrase in text for phrase in phrases)
        ),
        None,
    )


def statement_compatibility(
    plan: dict[str, Any], evidence: dict[str, Any]
) -> tuple[str, int]:
    hint = plan.get("statement_hint")
    if not hint:
        return "neutral", 0
    context = evidence.get("context") or {}
    source = " ".join(
        str(context.get(key) or "")
        for key in ("statement_type", "table_title", "section_path")
    )
    hint_family, source_family = _statement_family(hint), _statement_family(source)
    if hint_family and source_family and hint_family != source_family:
        return "conflict", -1
    if hint_family and hint_family == source_family:
        return "compatible", 1
    if normalize(hint) and normalize(hint) in normalize(source):
        return "compatible", 1
    return "neutral", 0


def _dimension_for_slot(
    payload: dict[str, Any], slot: dict[str, Any]
) -> dict[str, Any] | None:
    period = normalize(slot.get("period"))
    segment = normalize(slot.get("segment_label"))
    bucket = normalize(slot.get("bucket_label"))
    for index, dimension in enumerate(payload.get("dimensions") or []):
        if period and normalize(dimension.get("normalized_period")) != period:
            continue
        if segment and normalize(dimension.get("segment_label")) != segment:
            continue
        if bucket and normalize(dimension.get("bucket_label")) != bucket:
            continue
        if period or segment or bucket:
            return {
                **dimension,
                "dimension_index": index,
                "dimension_identity": f"{payload.get('semantic_fact_id')}:{index}",
            }
    return None


def match_slot(
    plan: dict[str, Any], slot: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any] | None:
    required = str(slot.get("required_evidence_shape") or "")
    kind = str(evidence.get("evidence_type") or "")
    payload = evidence.get("semantic_payload") or {}
    grade = metric_grade(slot, payload) if kind != "raw_candidate" else "raw_fallback"
    dimension = _dimension_for_slot(payload, slot) if kind == "row_matrix" else None
    if required == "atomic_fact":
        compatible = kind == "atomic_fact" or (
            kind == "row_matrix" and dimension is not None
        )
    elif required == "comparison_fact":
        compatible = kind == "comparison_fact" or (
            kind == "row_matrix"
            and dimension is not None
            and len(payload.get("dimensions") or []) >= 2
        )
    elif required == "bucket_fact":
        compatible = kind == "bucket_fact" or (
            kind == "row_matrix"
            and dimension is not None
            and bool(normalize(slot.get("bucket_label")))
        )
    elif required == "row_matrix":
        compatible = kind == "row_matrix" and dimension is not None
    elif required in {"narrative_section", "narrative_evidence"}:
        compatible = kind == "narrative_evidence"
    elif required == "raw_fallback":
        compatible = kind == "raw_candidate"
    else:
        compatible = kind == required
    if not compatible or grade is None:
        return None
    period = normalize(slot.get("period"))
    if kind != "row_matrix" and period:
        periods = {
            normalize(payload.get(key))
            for key in ("normalized_period", "base_period", "compared_period")
            if normalize(payload.get(key))
        }
        if period not in periods:
            return None
    segment = normalize(slot.get("segment_label"))
    bucket = normalize(slot.get("bucket_label"))
    if (
        kind != "row_matrix"
        and segment
        and normalize(payload.get("segment_label")) != segment
    ):
        return None
    if (
        kind != "row_matrix"
        and bucket
        and normalize(payload.get("bucket_label")) != bucket
    ):
        return None
    context_status, context_score = statement_compatibility(plan, evidence)
    if context_status == "conflict":
        return None
    context = evidence.get("context") or {}
    return {
        "slot_id": slot["slot_id"],
        "evidence_id": evidence["evidence_id"],
        "candidate_key": evidence["candidate_key"],
        "supporting_candidate_keys": evidence.get("supporting_candidate_keys")
        or [evidence["candidate_key"]],
        "candidate_rank": evidence["candidate_rank"],
        "metric_grade": grade,
        "dimension_exact": bool(dimension)
        or not any(
            (slot.get("period"), slot.get("segment_label"), slot.get("bucket_label"))
        ),
        "matrix_dimension": dimension,
        "typed": kind != "raw_candidate",
        "evidence_type": kind,
        "statement_context": context_status,
        "statement_context_score": context_score,
        "row_id": context.get("row_id"),
        "table_fragment_id": context.get("table_fragment_id"),
        "equivalent_group_id": payload.get("equivalent_group_id"),
    }
