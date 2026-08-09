from __future__ import annotations

import re
import unicodedata
from typing import Any

SHAPES = {
    "atomic_fact": {"atomic_fact", "row_matrix"},
    "comparison_fact": {"comparison_fact", "atomic_fact", "row_matrix"},
    "bucket_fact": {"bucket_fact", "row_matrix"},
    "row_matrix": {"row_matrix", "atomic_fact", "comparison_fact", "bucket_fact"},
    "narrative_section": {"narrative_evidence"},
    "narrative_evidence": {"narrative_evidence"},
    "raw_fallback": {"raw_candidate"},
}


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _metrics(payload: dict[str, Any]) -> list[str]:
    values = [payload.get("metric"), payload.get("normalized_metric")]
    path = payload.get("metric_path") or []
    if isinstance(path, list):
        values.extend(path)
        values.append(" / ".join(map(str, path)))
    return [normalize(value) for value in values if normalize(value)]


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


def match_slot(slot: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any] | None:
    required = str(slot.get("required_evidence_shape") or "")
    evidence_type = str(evidence.get("evidence_type") or "")
    if evidence_type not in SHAPES.get(required, {required}):
        return None
    payload = evidence.get("payload") or {}
    raw_metric = normalize(slot.get("raw_metric_phrase"))
    concepts = {
        normalize(item)
        for item in slot.get("concept_candidates") or []
        if normalize(item)
    }
    metrics = _metrics(payload)
    if evidence_type == "raw_candidate":
        grade = "raw_fallback"
    elif raw_metric and any(
        raw_metric == part or raw_metric in part or part in raw_metric
        for part in metrics
    ):
        grade = "A_exact"
    elif concepts and concepts.intersection(metrics):
        grade = "B_concept"
    elif evidence_type == "row_matrix" and metrics:
        grade = "C_matrix_cover"
    elif raw_metric:
        return None
    else:
        grade = "B_concept"
    expected_period = normalize(slot.get("period"))
    periods = _periods(payload)
    if expected_period and expected_period not in periods:
        return None
    expected_segment = normalize(slot.get("segment_label"))
    expected_bucket = normalize(slot.get("bucket_label"))
    text = normalize(payload)
    if expected_segment and expected_segment not in text:
        return None
    if expected_bucket and expected_bucket not in text:
        return None
    return {
        "slot_id": slot["slot_id"],
        "evidence_id": evidence["evidence_id"],
        "candidate_key": evidence["candidate_key"],
        "supporting_candidate_keys": evidence.get("supporting_candidate_keys")
        or [evidence["candidate_key"]],
        "candidate_rank": evidence["candidate_rank"],
        "metric_grade": grade,
        "period_match": not expected_period or expected_period in periods,
        "typed": evidence_type != "raw_candidate",
        "evidence_type": evidence_type,
    }
