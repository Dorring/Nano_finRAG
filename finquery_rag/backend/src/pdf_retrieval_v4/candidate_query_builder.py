"""Gate 08 R2 candidate-aligned query construction.

Builds retrieval queries from a Gate 07 QueryPlan for the 4-lane
candidate-aligned direct retrieval.  The Raw Metric Phrase is always
preserved as-is; concept features are appended, never replacing the
metric phrase.
"""

from __future__ import annotations

from typing import Any

from src.pdf_retrieval_v4.query_plan_models import OperandSlot, QueryPlan


def build_raw_question_query(plan: QueryPlan) -> str:
    """Build the raw-question query for 4-lane search.

    Combines: raw_question + issuer + metric_phrases + periods.
    """
    parts: list[str] = [plan.raw_question]
    if plan.issuer:
        parts.append(str(plan.issuer))
    parts.extend(str(p) for p in plan.metric_phrases if p)
    parts.extend(str(p) for p in plan.periods if p)
    return " | ".join(p for p in parts if p.strip())


def build_slot_query(plan: QueryPlan, slot: dict[str, Any]) -> str:
    """Build a per-slot query for 4-lane search.

    Combines: raw_metric_phrase + normalized_period + temporal_kind +
    top-3 concept features.

    The Raw Metric Phrase is preserved as-is.  Concept features are
    appended, not replacing the metric phrase.
    """
    parts: list[str] = []
    raw_phrase = slot.get("raw_metric_phrase")
    if raw_phrase:
        parts.append(str(raw_phrase))
    period = slot.get("period")
    if period:
        parts.append(str(period))
    temporal_kind = slot.get("temporal_kind")
    if temporal_kind:
        parts.append(str(temporal_kind))
    concepts = list(slot.get("concept_candidates") or ())
    parts.extend(str(c) for c in concepts[:3] if c)
    return " | ".join(p for p in parts if p.strip())


def _slot_to_dict(slot: OperandSlot) -> dict[str, Any]:
    return {
        "raw_metric_phrase": slot.raw_metric_phrase,
        "period": slot.period,
        "temporal_kind": slot.temporal_kind,
        "concept_candidates": list(slot.concept_candidates),
        "role": slot.role,
        "bucket_label": slot.bucket_label,
        "segment_label": slot.segment_label,
    }


def build_all_queries(plan: QueryPlan) -> dict[str, Any]:
    """Build all queries for a plan.

    Returns::

        {
            "raw_question": [query],
            "slots": {slot_id: [query]},
        }
    """
    raw_query = build_raw_question_query(plan)
    slot_queries: dict[str, list[str]] = {}
    for slot in plan.operand_slots:
        slot_dict = _slot_to_dict(slot)
        query = build_slot_query(plan, slot_dict)
        slot_queries[slot.slot_id] = [query] if query else []
    return {"raw_question": [raw_query], "slots": slot_queries}
