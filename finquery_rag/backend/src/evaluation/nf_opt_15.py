"""Fail-closed structured retrieval-view construction for NF-OPT-15 Gate A."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from src.retrieval.candidate_identity import candidate_key, identity_from_candidate


def normalize_metric(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"\(\d+\)", "", value)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
    return normalized or None


def periods_from_text(value: str) -> tuple[str, ...]:
    years = re.findall(r"\b(?:fy\s*|fiscal\s+)?(20\d{2})\b", value, flags=re.IGNORECASE)
    return tuple(dict.fromkeys(f"FY{year}" for year in years))


def scale_from_text(value: str) -> tuple[str | None, str | None]:
    lowered = value.lower()
    scale = next((item for item in ("billion", "million", "thousand") if item in lowered), None)
    currency = "USD" if "$" in value or "dollar" in lowered else None
    return currency, scale


def table_row_label(content: str, evidence_type: str) -> str | None:
    if evidence_type != "table_row" or "|" not in content:
        return None
    value = content.split("|", maxsplit=1)[0].strip()
    return value or None


def numeric_values(content: str, evidence_type: str) -> tuple[str, ...]:
    if evidence_type != "table_row":
        return ()
    return tuple(re.findall(r"(?<![A-Za-z])\(?\$?[\d][\d,]*(?:\.\d+)?%?\)?", content))


def build_retrieval_view(
    *,
    doc_id: str,
    content: str,
    metadata: dict[str, Any],
    document: dict[str, Any],
) -> dict[str, Any]:
    """Create a view using existing candidate fields only; never infer missing headers."""
    evidence_type = str(metadata.get("type") or "text")
    header_context = str(metadata.get("table_header_context") or "")
    row_label = table_row_label(content, evidence_type)
    metric = normalize_metric(row_label)
    periods = periods_from_text(header_context if evidence_type == "table_row" else content)
    currency, scale = scale_from_text(header_context + " " + content)
    section_path = str(metadata.get("section_path") or "") or None
    section_title = str(metadata.get("section_title") or "") or None
    candidate = {
        "tenant_id": metadata.get("user_id"),
        "document_id": document["document_id"],
        "block_type": evidence_type,
        "evidence_id": doc_id,
        "doc_id": doc_id,
        "metadata": metadata,
    }
    stable_key = candidate_key(identity_from_candidate(candidate))
    structured = {
        "view_schema": "nf-opt-15/retrieval-view/v1",
        "candidate_key": stable_key,
        "evidence_id": doc_id,
        "document_id": document["document_id"],
        "pdf_page": metadata.get("page"),
        "evidence_type": evidence_type,
        "document_field": {
            "company": document.get("company"),
            "report_type": document.get("source_type"),
            "fiscal_year": f"FY{document['fiscal_year']}",
        },
        "section_field": {
            "section_path": [section_path] if section_path else [],
            "statement_title": section_title,
            "table_title": None,
        },
        "metric_field": {
            "raw_metric": row_label,
            "normalized_metric": metric,
            "source": "table_row_label" if metric else None,
            "status": "present" if metric else "missing",
        },
        "period_field": {
            "periods": list(periods),
            "raw_headers": [header_context] if header_context else [],
            "source": "table_column_headers" if periods and evidence_type == "table_row" else ("raw_content" if periods else None),
            "status": "present" if periods else "missing",
        },
        "unit_field": {
            "currency": currency,
            "scale": scale,
            "source": "table_header_or_content" if currency or scale else None,
            "status": "present" if currency or scale else "missing",
        },
        "value_field": {
            "raw_values": list(numeric_values(content, evidence_type)),
            "normalized_values": [],
            "status": "present" if numeric_values(content, evidence_type) else "missing",
        },
        "field_lineage": {
            "metric": ["content.table_row_label"] if metric else [],
            "period": ["metadata.table_header_context"] if periods and evidence_type == "table_row" else (["content"] if periods else []),
            "scale": ["metadata.table_header_context", "content"] if currency or scale else [],
        },
    }
    identity_payload = json.dumps(structured, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    structured["retrieval_view_id"] = "view:v1:" + hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
    return structured
