"""Deterministic, evaluation-label-free reranker input views."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Iterable

RERANK_INSTRUCTION = (
    "Given a financial-report question and its structured query plan, judge whether the "
    "candidate evidence directly supports answering the question. Prioritize exact financial "
    "metric, reporting period, entity or segment, statement/table semantics, and factual numeric "
    "evidence. For multi-part questions, evidence supporting any required operand is relevant. "
    "Do not reward generic financial-topic similarity."
)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " / ".join(part for item in value if (part := _clean(item)))
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def _unique(values: Iterable[Any]) -> str:
    return " | ".join(dict.fromkeys(value for item in values if (value := _clean(item))))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_rerank_query_view(plan: dict[str, Any]) -> str:
    lines = ["[QUESTION]", _clean(plan["raw_question"]), "", "[QUERY PLAN]"]
    if value := _clean(plan.get("task_type")):
        lines.append(f"Task: {value}")
    if value := _clean(plan.get("operation")):
        lines.append(f"Operation: {value}")
    lines.append("Required evidence:")
    for index, slot in enumerate(plan.get("operand_slots", []), 1):
        fields = []
        for label, key in (
            ("metric", "raw_metric_phrase"), ("period", "period"),
            ("segment", "segment_label"), ("bucket", "bucket_label"),
            ("shape", "required_evidence_shape"),
        ):
            if value := _clean(slot.get(key)):
                fields.append(f"{label}={value}")
        lines.append(f"- Slot {index}: {', '.join(fields)}")
    return "\n".join(lines).strip()


def build_rerank_document_view(
    candidate: dict[str, Any], authoritative_evidence: list[dict[str, Any]]
) -> str:
    metadata = candidate.get("metadata", {})
    contexts = [item.get("context", {}) for item in authoritative_evidence]
    payloads = [item.get("semantic_payload", {}) for item in authoritative_evidence]
    lines = ["[DOCUMENT]"]
    if value := _clean(candidate.get("document_id") or metadata.get("document_id")):
        lines.append(f"Document: {value}")
    if value := _clean(metadata.get("issuer")):
        lines.append(f"Issuer: {value}")
    structure = [
        ("Statement", _unique(context.get("statement_type") for context in contexts)),
        ("Section", _unique(context.get("section_path") for context in contexts)),
        ("Table", _unique(context.get("table_title") for context in contexts)),
        ("Metric Path", _unique(
            [context.get("metric_path") for context in contexts]
            + [payload.get("metric_path") for payload in payloads]
            + list(metadata.get("metric_paths", []))
        )),
        ("Row", _unique(context.get("raw_row_label") for context in contexts)),
    ]
    if any(value for _, value in structure):
        lines.extend(["", "[STRUCTURE]"])
        lines.extend(f"{label}: {value}" for label, value in structure if value)
    evidence = [
        ("Type", _unique(item.get("evidence_type") for item in authoritative_evidence)),
        ("Metric", _unique(payload.get("leaf_metric") for payload in payloads)),
        ("Period", _unique(
            [payload.get("normalized_period") for payload in payloads]
            + [dimension.get("normalized_period") for payload in payloads for dimension in payload.get("dimensions", [])]
            + list(metadata.get("periods", []))
        )),
        ("Segment", _unique(
            [payload.get("segment_label") for payload in payloads]
            + [dimension.get("segment_label") for payload in payloads for dimension in payload.get("dimensions", [])]
        )),
        ("Bucket", _unique(
            [payload.get("bucket_label") for payload in payloads]
            + [dimension.get("bucket_label") for payload in payloads for dimension in payload.get("dimensions", [])]
        )),
        ("Value", _unique(
            [payload.get("value_normalized") for payload in payloads]
            + [dimension.get("value_normalized") for payload in payloads for dimension in payload.get("dimensions", [])]
        )),
        ("Scale", _unique(payload.get("scale_unit") or payload.get("scale") for payload in payloads)),
        ("Currency", _unique(payload.get("currency_code") for payload in payloads)),
    ]
    if any(value for _, value in evidence):
        lines.extend(["", "[EVIDENCE]"])
        lines.extend(f"{label}: {value}" for label, value in evidence if value)
    lines.extend(["", "[CONTENT]", str(candidate.get("raw_text", "")).strip()])
    return "\n".join(lines).strip()


def canonical_json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
