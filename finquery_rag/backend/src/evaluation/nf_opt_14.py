"""Gold-independent query-slot contracts and compatibility helpers for NF-OPT-14."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%]+", " ", value.lower())).strip()


def _tokens(value: str | None) -> tuple[str, ...]:
    return tuple(token for token in _normalize(value or "").split() if len(token) > 2)


def _periods(text: str) -> tuple[str, ...]:
    values = re.findall(r"\b(?:fy\s*|fiscal\s+)?(20\d{2})\b", text, flags=re.IGNORECASE)
    return tuple(dict.fromkeys(f"FY{value}" for value in values))


def _metric_before(text: str, marker: str) -> str | None:
    match = re.search(marker, text, flags=re.IGNORECASE)
    return match.group("metric").strip(" ,?.") if match else None


def _slot(slot_id: str, role: str, metric: str | None, period: str | None) -> dict[str, Any]:
    return {
        "slot_id": slot_id,
        "role": role,
        "metric_phrase": metric,
        "normalized_metric_tokens": list(_tokens(metric)),
        "period": period,
    }


def parse_query_slot_contract(question: dict[str, Any]) -> dict[str, Any]:
    """Build a fixed query-only slot contract without reading expected fields."""
    text = str(question.get("question") or "")
    lowered = text.lower()
    periods = _periods(text)
    operation: str | None = None
    metric: str | None = None
    slots: list[dict[str, Any]] = []
    if re.search(r"\b(grow|growth|increase|decrease|decline|percent change)\b", lowered):
        operation = "growth_rate"
        metric = _metric_before(text, r"(?:how much did\s+)(?P<metric>.+?)(?:\s+(?:grow|increase|decrease|decline))")
        ordered = tuple(sorted(periods))
        if len(ordered) == 2 and metric:
            slots = [
                _slot("previous_operand", "previous", metric, ordered[0]),
                _slot("current_operand", "current", metric, ordered[1]),
            ]
    elif "percentage of" in lowered or "share of" in lowered or "portion of" in lowered:
        operation = "percentage_share"
        match = re.search(
            r"(?:percentage|share|portion) of\s+(?P<denominator>.+?)\s+(?:was|did)\s+(?P<numerator>.+?)(?:\s+(?:in|for)\s+|\?)",
            text,
            flags=re.IGNORECASE,
        )
        if match and len(periods) == 1:
            slots = [
                _slot("numerator", "numerator", match.group("numerator"), periods[0]),
                _slot("denominator", "denominator", match.group("denominator"), periods[0]),
            ]
    elif "difference between" in lowered:
        operation = "difference"
        match = re.search(r"difference between\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)(?:\s+(?:in|for)\s+|\?)", text, flags=re.IGNORECASE)
        if match:
            period = periods[0] if len(periods) == 1 else None
            slots = [
                _slot("minuend", "minuend", match.group("left"), period),
                _slot("subtrahend", "subtrahend", match.group("right"), period),
            ]
    elif "sum of" in lowered or "combined" in lowered or "average" in lowered or "mean of" in lowered:
        operation = "average" if "average" in lowered or "mean of" in lowered else "sum"
        match = re.search(r"(?:sum|average|mean) of\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)(?:\s+(?:in|for)\s+|\?)", text, flags=re.IGNORECASE)
        if match:
            period = periods[0] if len(periods) == 1 else None
            slots = [
                _slot("operand_1", "operand", match.group("left"), period),
                _slot("operand_2", "operand", match.group("right"), period),
            ]
    else:
        metric = _metric_before(text, r"(?:what was|what were)\s+(?P<metric>.+?)(?:\s+reported\b|\s+in\s+FY\d{4}|\?)")
        if metric:
            if len(periods) > 1:
                slots = [_slot(f"fact_period_{index + 1}", "fact", metric, period) for index, period in enumerate(periods)]
            else:
                slots = [_slot("fact", "fact", metric, periods[0] if periods else None)]
    complete = bool(slots) and all(slot["metric_phrase"] for slot in slots)
    return {
        "case_id": str(question["case_id"]),
        "document_scope": list(question.get("document_scope") or ()),
        "operation": operation,
        "slots": slots,
        "required_evidence_count": len(slots),
        "one_candidate_may_cover_multiple_slots": True,
        "contract_status": "complete" if complete else "incomplete",
        "query_fields_read": ["case_id", "question", "document_scope"],
        "expected_fields_read": False,
    }


def candidate_slot_compatibility(
    *, candidate: dict[str, Any], candidate_text: str, slot: dict[str, Any], document_scope: set[str]
) -> dict[str, Any]:
    """Score a candidate against a slot using only ordinary candidate/query data."""
    canonical_document = str(candidate.get("canonical_document_id") or "")
    document_match = canonical_document in document_scope
    text = _normalize(candidate_text + " " + jsonable_candidate_metadata(candidate))
    metric_tokens = tuple(slot.get("normalized_metric_tokens") or ())
    metric_match = bool(metric_tokens) and all(token in text for token in metric_tokens)
    period = slot.get("period")
    period_match = period is None or (period[2:] in text or period.lower() in text)
    compatibility = "strict" if document_match and metric_match and period_match else "partial" if document_match and (metric_match or period_match) else "none"
    return {
        "candidate_key": str(candidate.get("candidate_key")),
        "slot_id": str(slot["slot_id"]),
        "document_match": document_match,
        "metric_match": "strict" if metric_match else "none",
        "period_match": "strict" if period_match else "none",
        "role_match": "compatible" if metric_match else "none",
        "compatibility": compatibility,
        "signals_used": ["candidate_text", "candidate_metadata", "document_id", "period_token"],
    }


def jsonable_candidate_metadata(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(candidate.get(field) or "")
        for field in ("evidence_id", "doc_id", "type", "block_type", "page", "metadata_text")
    )


def deterministic_slot_selector(
    *, baseline_final: list[dict[str, Any]], reranked: list[dict[str, Any]], slots: list[dict[str, Any]], matrix: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fixed shadow policy: preserve baseline unless every requested slot is strict."""
    if len(slots) <= 1:
        return baseline_final
    compatible = {
        (str(item["candidate_key"]), str(item["slot_id"]))
        for item in matrix
        if item["compatibility"] == "strict"
    }
    selected: list[dict[str, Any]] = []
    for slot in slots:
        candidate = next(
            (item for item in reranked if (str(item.get("candidate_key")), str(slot["slot_id"])) in compatible),
            None,
        )
        if candidate is None:
            return baseline_final
        if str(candidate.get("candidate_key")) not in {str(value.get("candidate_key")) for value in selected}:
            selected.append(candidate)
    for candidate in reranked:
        if len(selected) == 5:
            break
        if str(candidate.get("candidate_key")) not in {str(value.get("candidate_key")) for value in selected}:
            selected.append(candidate)
    return selected[:5]
