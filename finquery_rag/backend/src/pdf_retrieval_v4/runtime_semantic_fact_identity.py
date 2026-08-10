"""Runtime semantic fact identities for Gate 08 R8-SE1.

This module deliberately contains no I/O, retrieval, query, or benchmark logic.
It turns already-authoritative Gate03 evidence records into strict financial-fact
identities that exclude physical provenance such as rows, pages, and candidates.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

IDENTITY_SCHEMA = "pdf-retrieval-v4/runtime-semantic-fact/v1"
IDENTITY_FIELDS = (
    "document_id",
    "normalized_metric",
    "normalized_period",
    "normalized_segment",
    "normalized_bucket",
    "normalized_base_value",
    "normalized_scale",
    "normalized_currency",
)

_SCALE_ALIASES = {
    "base": "1",
    "one": "1",
    "ones": "1",
    "thousand": "1000",
    "thousands": "1000",
    "million": "1000000",
    "millions": "1000000",
    "billion": "1000000000",
    "billions": "1000000000",
}


def normalize_text(value: Any) -> str:
    """Normalize a semantic label without fuzzy matching or alias expansion."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def normalize_numeric(value: Any) -> str:
    """Return an exact canonical decimal, or an empty string when not numeric."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return ""
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("() ").replace(",", "").replace("%", "")
    text = re.sub(r"^[\$€£¥]", "", text).strip()
    try:
        number = Decimal(text)
    except InvalidOperation:
        return ""
    if negative:
        number = -number
    normalized = format(number.normalize(), "f")
    return "0" if Decimal(normalized) == 0 else normalized


def normalize_scale(scale: Any, scale_unit: Any = None) -> str:
    """Canonicalize only explicit, deterministic scale equivalences."""

    numeric = normalize_numeric(scale)
    if numeric:
        return numeric
    unit = normalize_text(scale_unit or scale)
    return _SCALE_ALIASES.get(unit, unit)


def normalize_currency(value: Any) -> str:
    return normalize_text(value).upper()


def build_semantic_fact(fields: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any] | None:
    """Build one strict fact identity, failing closed on incomplete core fields."""

    normalized = {
        "document_id": normalize_text(fields.get("document_id")),
        "normalized_metric": normalize_text(fields.get("normalized_metric")),
        "normalized_period": normalize_text(fields.get("normalized_period")),
        "normalized_segment": normalize_text(fields.get("normalized_segment")),
        "normalized_bucket": normalize_text(fields.get("normalized_bucket")),
        "normalized_base_value": normalize_numeric(fields.get("normalized_base_value")),
        "normalized_scale": normalize_scale(fields.get("normalized_scale"), fields.get("scale_unit")),
        "normalized_currency": normalize_currency(fields.get("normalized_currency")),
    }
    if not normalized["document_id"] or not normalized["normalized_metric"]:
        return None
    if not normalized["normalized_base_value"]:
        return None
    if not (
        normalized["normalized_period"]
        or normalized["normalized_segment"]
        or normalized["normalized_bucket"]
    ):
        return None
    identity_payload = "\x1f".join(normalized[field] for field in IDENTITY_FIELDS)
    semantic_fact_id = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
    return {
        "semantic_fact_id": semantic_fact_id,
        **normalized,
        "physical_provenance": provenance,
    }


def _provenance(record: dict[str, Any], evidence_id: str, dimension_index: int | None = None) -> dict[str, Any]:
    traceback = record.get("source_traceback") or {}
    result = {
        "authoritative_evidence_id": evidence_id,
        "authoritative_evidence_type": evidence_type(evidence_id),
        "document_id": record.get("document_id"),
        "pdf_page": traceback.get("pdf_page"),
        "table_fragment_id": record.get("table_fragment_id") or traceback.get("table_fragment_id"),
        "row_id": record.get("row_id") or traceback.get("row_id"),
        "cell_id": record.get("cell_id") or traceback.get("cell_id"),
    }
    if dimension_index is not None:
        result["matrix_dimension_index"] = dimension_index
    return result


def evidence_type(evidence_id: str) -> str:
    return str(evidence_id).partition(":")[0]


def expand_authoritative_evidence(evidence_id: str, record: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Expand one Gate03 record to zero or more cell-level semantic facts."""

    kind = evidence_type(evidence_id)
    metric = record.get("metric_path") or record.get("leaf_metric")
    common = {
        "document_id": record.get("document_id"),
        "normalized_metric": metric,
        "normalized_scale": record.get("scale"),
        "scale_unit": record.get("scale_unit"),
        "normalized_currency": record.get("currency_code"),
    }
    facts: list[dict[str, Any]] = []
    if kind == "atomic":
        fact = build_semantic_fact(
            {
                **common,
                "normalized_period": record.get("normalized_period"),
                "normalized_base_value": record.get("value_normalized"),
            },
            _provenance(record, evidence_id),
        )
        return ([fact] if fact else [], "expanded" if fact else "insufficient_for_expansion")
    if kind == "bucket":
        fact = build_semantic_fact(
            {
                **common,
                "normalized_period": record.get("normalized_period"),
                "normalized_segment": record.get("segment_label"),
                "normalized_bucket": record.get("bucket_label"),
                "normalized_base_value": record.get("value_normalized"),
            },
            _provenance(record, evidence_id),
        )
        return ([fact] if fact else [], "expanded" if fact else "insufficient_for_expansion")
    if kind == "matrix":
        for index, dimension in enumerate(record.get("dimensions") or []):
            fact = build_semantic_fact(
                {
                    **common,
                    "normalized_period": dimension.get("normalized_period"),
                    "normalized_segment": dimension.get("segment_label"),
                    "normalized_bucket": dimension.get("bucket_label"),
                    "normalized_base_value": dimension.get("value_normalized"),
                },
                {
                    **_provenance(record, evidence_id, index),
                    "cell_id": dimension.get("cell_id"),
                },
            )
            if fact:
                facts.append(fact)
        return (facts, "expanded" if facts else "insufficient_for_expansion")
    if kind == "comparison":
        for role in ("base", "compared"):
            fact = build_semantic_fact(
                {
                    **common,
                    "normalized_period": record.get(f"{role}_period"),
                    "normalized_base_value": record.get(f"{role}_value"),
                },
                {**_provenance(record, evidence_id), "comparison_operand_role": role},
            )
            if fact:
                facts.append(fact)
        status = "expanded" if len(facts) == 2 else "insufficient_for_expansion"
        return (facts if len(facts) == 2 else [], status)
    if kind == "narrative":
        return [], "semantic_expansion_not_supported"
    return [], "unsupported_evidence_type"


def deduplicate_facts(facts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse identical semantic identities while retaining all provenance."""

    by_id: dict[str, dict[str, Any]] = {}
    for fact in facts:
        fact_id = str(fact["semantic_fact_id"])
        if fact_id not in by_id:
            by_id[fact_id] = {**fact, "physical_provenance": [fact["physical_provenance"]]}
        else:
            provenance = fact["physical_provenance"]
            if provenance not in by_id[fact_id]["physical_provenance"]:
                by_id[fact_id]["physical_provenance"].append(provenance)
    for fact in by_id.values():
        fact["physical_provenance"] = sorted(
            fact["physical_provenance"], key=lambda item: json.dumps(item, sort_keys=True)
        )
    return [by_id[fact_id] for fact_id in sorted(by_id)]
