from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _metric_values(payload: dict[str, Any]) -> tuple[str, list[str]]:
    path = payload.get("metric_path") or []
    segments = (
        [normalize(item) for item in path if normalize(item)]
        if isinstance(path, list)
        else []
    )
    direct = normalize(payload.get("metric") or payload.get("normalized_metric"))
    if direct and direct not in segments:
        segments.append(direct)
    full = normalize(" ".join(map(str, path))) if path else direct
    return full, segments


def _periods(payload: dict[str, Any]) -> set[str]:
    values = [payload.get("period")]
    values.extend(payload.get("periods") or [])
    binding = payload.get("temporal_binding") or {}
    values.extend(
        [
            binding.get("period"),
            binding.get("base_period"),
            binding.get("current_period"),
        ]
    )
    return {normalize(value) for value in values if normalize(value)}


def _dimensions(payload: dict[str, Any], names: tuple[str, ...]) -> set[str]:
    values: list[object] = [payload.get(name) for name in names]
    dimensions = payload.get("dimensions") or payload.get("dimension_axes") or []
    for dimension in dimensions:
        if isinstance(dimension, dict):
            values.extend(dimension.get("labels") or [])
            values.extend([dimension.get("label"), dimension.get("value")])
    return {normalize(value) for value in values if normalize(value)}


def metric_grade(slot: dict[str, Any], payload: dict[str, Any]) -> str | None:
    phrase = normalize(slot.get("raw_metric_phrase"))
    full, segments = _metric_values(payload)
    if not phrase:
        return "M3_concept"
    if segments and phrase == segments[-1]:
        return "M0_exact_leaf"
    if phrase == full:
        return "M1_exact_path"
    if phrase in segments:
        return "M2_path_segment"
    concepts = {
        normalize(item)
        for item in slot.get("concept_candidates") or []
        if normalize(item)
    }
    if concepts.intersection(segments) or (full and full in concepts):
        return "M3_concept"
    return None


def matrix_covers_slot(payload: dict[str, Any], slot: dict[str, Any]) -> bool:
    if metric_grade(slot, payload) is None:
        return False
    period = normalize(slot.get("period"))
    segment = normalize(slot.get("segment_label"))
    bucket = normalize(slot.get("bucket_label"))
    if period and period not in _periods(payload):
        return False
    if segment and segment not in _dimensions(
        payload, ("segment", "segment_label", "segments")
    ):
        return False
    if bucket and bucket not in _dimensions(
        payload, ("bucket", "bucket_label", "buckets")
    ):
        return False
    return True


def match_slot(slot: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any] | None:
    required = str(slot.get("required_evidence_shape") or "")
    kind = str(evidence.get("evidence_type") or "")
    payload = evidence.get("payload") or {}
    if required == "atomic_fact":
        compatible = kind == "atomic_fact" or (
            kind == "row_matrix" and matrix_covers_slot(payload, slot)
        )
    elif required == "comparison_fact":
        compatible = kind == "comparison_fact" or (
            kind == "row_matrix"
            and matrix_covers_slot(payload, slot)
            and len(_periods(payload)) >= 2
        )
    elif required == "bucket_fact":
        compatible = kind == "bucket_fact" or (
            kind == "row_matrix"
            and matrix_covers_slot(payload, slot)
            and bool(normalize(slot.get("bucket_label")))
        )
    elif required == "row_matrix":
        compatible = kind == "row_matrix"
    elif required in {"narrative_section", "narrative_evidence"}:
        compatible = kind == "narrative_evidence"
    elif required == "raw_fallback":
        compatible = kind == "raw_candidate"
    else:
        compatible = kind == required
    if not compatible:
        return None
    grade = "raw_fallback" if kind == "raw_candidate" else metric_grade(slot, payload)
    if grade is None:
        return None
    period = normalize(slot.get("period"))
    periods = _periods(payload)
    if period and period not in periods:
        return None
    segment = normalize(slot.get("segment_label"))
    bucket = normalize(slot.get("bucket_label"))
    if segment and segment not in _dimensions(
        payload, ("segment", "segment_label", "segments")
    ):
        return None
    if bucket and bucket not in _dimensions(
        payload, ("bucket", "bucket_label", "buckets")
    ):
        return None
    trace = evidence.get("source_traceback") or []
    row_ids = sorted({str(item.get("row_id")) for item in trace if item.get("row_id")})
    table_ids = sorted(
        {
            str(item.get("table_fragment_id"))
            for item in trace
            if item.get("table_fragment_id")
        }
    )
    return {
        "slot_id": slot["slot_id"],
        "evidence_id": evidence["evidence_id"],
        "candidate_key": evidence["candidate_key"],
        "supporting_candidate_keys": evidence.get("supporting_candidate_keys")
        or [evidence["candidate_key"]],
        "candidate_rank": evidence["candidate_rank"],
        "metric_grade": grade,
        "period_match": not period or period in periods,
        "typed": kind != "raw_candidate",
        "evidence_type": kind,
        "row_ids": row_ids,
        "table_ids": table_ids,
    }
