"""Pure, production-isolated helpers for the NF-OPT-12 retrieval audit."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any


def _normal(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def family_components(item: dict[str, Any]) -> dict[str, str | int | None]:
    """Build family components using candidate metadata only, never gold data."""
    evidence_id = str(item.get("evidence_id") or item.get("doc_id") or "")
    document_id = str(item.get("canonical_document_id") or item.get("document_id") or "")
    page = item.get("page")
    table = re.search(r"::((?:layout_)?table_\d+)", evidence_id)
    row = re.search(r"::row_(\d+)", evidence_id)
    if table:
        table_identity = table.group(1)
        row_identity = row.group(1) if row else None
    else:
        table_identity = str(item.get("parent_id") or item.get("parent_candidate_key") or evidence_id)
        row_identity = None
    return {
        "document_id": document_id,
        "pdf_page": int(page) if isinstance(page, int) or str(page).isdigit() else None,
        "table_identity": _normal(table_identity),
        "row_identity": row_identity,
    }


def evidence_family_id(item: dict[str, Any]) -> str:
    parts = family_components(item)
    canonical = "|".join(
        str(parts[key] or "")
        for key in ("document_id", "pdf_page", "table_identity", "row_identity")
    )
    return "family:v1:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evidence_slot_id(item: dict[str, Any]) -> str:
    """Keep each concrete candidate slot distinct inside a logical family."""
    family = evidence_family_id(item)
    candidate = str(item.get("candidate_key") or item.get("evidence_id") or item.get("doc_id") or "")
    return "slot:v1:" + hashlib.sha256(f"{family}|{candidate}".encode("utf-8")).hexdigest()


def collapse_families(candidates: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Return the first-ranked member of each family without reordering it."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        family = evidence_family_id(candidate)
        if family in seen:
            continue
        seen.add(family)
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def parse_query_slots(question: dict[str, Any]) -> dict[str, Any]:
    """Derive a deterministic slot plan from question fields only."""
    text = str(question.get("question") or "")
    periods = re.findall(r"\b(?:fy\s*)?(20\d{2})\b", text, flags=re.IGNORECASE)
    normalized_periods = tuple(dict.fromkeys(f"FY{year}" for year in periods))
    calculation = bool(question.get("requires_calculation"))
    lowered = text.lower()
    operation = None
    if any(
        token in lowered
        for token in ("growth", "grow", "increase", "decrease", "decline", "percent change")
    ):
        operation = "growth_rate"
    elif any(token in lowered for token in ("difference", "how much higher", "how much lower", "by how much")):
        operation = "difference"
    elif any(token in lowered for token in ("what percentage of", "share of", "portion of")):
        operation = "percentage_share"
    elif "combined" in lowered or "sum of" in lowered:
        operation = "sum"
    elif "average" in lowered or "mean of" in lowered:
        operation = "average"
    if operation:
        calculation = True
    required = max(2 if calculation else 1, len(normalized_periods) if calculation else 1)
    slots = [
        {
            "slot_id": f"query-slot-{index + 1}",
            "period": period,
            "role": "operand" if calculation else "fact",
        }
        for index, period in enumerate(normalized_periods)
    ]
    while len(slots) < required:
        slots.append(
            {
                "slot_id": f"query-slot-{len(slots) + 1}",
                "period": None,
                "role": "operand" if calculation else "fact",
            }
        )
    return {
        "case_id": str(question["case_id"]),
        "document_scope": list(question.get("document_scope") or ()),
        "operation": operation,
        "requires_calculation": calculation,
        "required_evidence_count": required,
        "slots": slots,
        "input_fields": ["case_id", "question", "document_scope", "requires_calculation"],
        "expected_fields_read": False,
    }


def strict_hits(candidates: Iterable[dict[str, Any]], gold_keys: set[str]) -> set[str]:
    return {
        str(item.get("candidate_key"))
        for item in candidates
        if str(item.get("candidate_key")) in gold_keys
    }


def family_proxy_hits(candidates: Iterable[dict[str, Any]], gold_families: dict[str, str]) -> set[str]:
    retrieved = {evidence_family_id(item) for item in candidates}
    return {key for key, family in gold_families.items() if family in retrieved}
