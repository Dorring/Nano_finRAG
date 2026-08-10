"""Resolve explicit unit context from frozen Gate03 structural artifacts."""

from __future__ import annotations

from typing import Any


def _unique(values: list[str]) -> tuple[str | None, str]:
    distinct = sorted({value for value in values if value})
    if len(distinct) == 1:
        return distinct[0], "resolved"
    if len(distinct) > 1:
        return None, "conflict"
    return None, "unresolved"


def resolve_unit_context(
    semantic_class: dict[str, Any],
    scale_by_table: dict[str, dict[str, Any]],
    currency_by_table: dict[str, dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    evidence_by_id = evidence_by_id or {}
    explicit_scale = str(semantic_class.get("scale") or "")
    explicit_currency = str(semantic_class.get("currency") or "")
    evidence_ids = sorted(
        {
            str(item.get("authoritative_evidence_id"))
            for item in semantic_class.get("physical_provenance") or []
            if item.get("authoritative_evidence_id")
        }
    )
    evidence_scales = [
        str(evidence_by_id[evidence_id].get("scale") or "")
        for evidence_id in evidence_ids
        if evidence_id in evidence_by_id and evidence_by_id[evidence_id].get("scale")
    ]
    evidence_currencies = [
        str(evidence_by_id[evidence_id].get("currency_code") or "")
        for evidence_id in evidence_ids
        if evidence_id in evidence_by_id and evidence_by_id[evidence_id].get("currency_code")
    ]
    table_ids = sorted(
        {
            str(item.get("table_fragment_id"))
            for item in semantic_class.get("physical_provenance") or []
            if item.get("table_fragment_id")
        }
    )
    table_scales = [
        str(scale_by_table[table_id].get("scale") or "")
        for table_id in table_ids
        if table_id in scale_by_table and scale_by_table[table_id].get("scale_status") == "resolved"
    ]
    table_currencies = [
        str(currency_by_table[table_id].get("currency_code") or "")
        for table_id in table_ids
        if table_id in currency_by_table and currency_by_table[table_id].get("currency_status") == "resolved"
    ]
    evidence_scale, evidence_scale_status = _unique(evidence_scales)
    evidence_currency, evidence_currency_status = _unique(evidence_currencies)
    if explicit_scale:
        scale, scale_status, scale_source = explicit_scale, "resolved", "semantic_evidence"
    elif evidence_scale_status != "unresolved":
        scale, scale_status = evidence_scale, evidence_scale_status
        scale_source = "authoritative_evidence" if scale_status == "resolved" else None
    else:
        scale, scale_status = _unique(table_scales)
        scale_source = "table_context" if scale_status == "resolved" else None
    if explicit_currency:
        currency, currency_status, currency_source = explicit_currency, "resolved", "semantic_evidence"
    elif evidence_currency_status != "unresolved":
        currency, currency_status = evidence_currency, evidence_currency_status
        currency_source = "authoritative_evidence" if currency_status == "resolved" else None
    else:
        currency, currency_status = _unique(table_currencies)
        currency_source = "table_context" if currency_status == "resolved" else None
    return {
        "scale": scale,
        "currency": currency,
        "scale_status": scale_status,
        "currency_status": currency_status,
        "scale_source": scale_source,
        "currency_source": currency_source,
        "table_fragment_ids": table_ids,
        "authoritative_evidence_ids": evidence_ids,
        "resolution_status": "conflict"
        if "conflict" in {scale_status, currency_status}
        else "resolved"
        if scale_status == currency_status == "resolved"
        else "partial"
        if "resolved" in {scale_status, currency_status}
        else "unresolved",
    }
