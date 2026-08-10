"""Strict metric and measurement-kind binding for Gate09 R5.1."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

MEASUREMENT_KINDS = {
    "monetary_amount",
    "percentage",
    "ratio",
    "per_share",
    "count",
    "dimensionless",
    "unknown",
}


def normalize(value: Any) -> str:
    text = " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())
    text = re.sub(r"\s*\([a-z0-9]{1,3}\)$", "", text)
    return re.sub(r"(?<=[a-z])\d{1,2}$", "", text).strip()


def infer_measurement_kind(metric: Any, explicit_unit: Any = None) -> str:
    explicit = normalize(explicit_unit)
    text = normalize(metric)
    if explicit in {"%", "percent", "percentage"} or any(
        phrase in text for phrase in ("percentage", "percent", "gross margin", "operating margin", "margin rate")
    ):
        return "percentage"
    if explicit in {"ratio", "multiple"} or " ratio" in f" {text}" or text.endswith(" ratio"):
        return "ratio"
    if explicit in {"per share", "share"} or any(phrase in text for phrase in ("per share", "earnings per share", " eps")):
        return "per_share"
    if any(phrase in text for phrase in ("number of", "headcount", "employee count", "transaction count")):
        return "count"
    monetary_terms = (
        "revenue", "sales", "income", "profit", "expense", "cost", "assets", "liabilities",
        "cash", "debt", "equity", "capital expenditure", "gross profit", "operating loss",
    )
    if any(term in text for term in monetary_terms):
        return "monetary_amount"
    return "unknown"


def bind_metric(slot: dict[str, Any], semantic_class: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic M0/M1 compatibility; concepts are metadata only."""

    phrase = normalize(slot.get("raw_metric_phrase"))
    metric = normalize(semantic_class.get("metric"))
    leaf = normalize(metric.rsplit("/", 1)[-1])
    concepts = {normalize(value) for value in slot.get("concept_candidates") or [] if normalize(value)}
    if phrase and phrase == metric:
        tier = "M0_exact_path"
    elif phrase and phrase == leaf:
        tier = "M1_exact_leaf"
    else:
        tier = None
    query_kind = infer_measurement_kind(phrase)
    evidence_kind = infer_measurement_kind(metric, semantic_class.get("unit_kind"))
    kind_conflict = query_kind != "unknown" and evidence_kind != "unknown" and query_kind != evidence_kind
    concept_hint = bool(concepts.intersection({metric, leaf}))
    return {
        "deterministic_compatible": bool(tier and not kind_conflict),
        "metric_tier": tier,
        "concept_hint_only": concept_hint,
        "query_measurement_kind": query_kind,
        "evidence_measurement_kind": evidence_kind,
        "measurement_kind_conflict": kind_conflict,
    }
