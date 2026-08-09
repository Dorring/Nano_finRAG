"""QueryPlan-only Gate 08 R5 field query projection."""

from __future__ import annotations

from typing import Any

from src.pdf_retrieval_v4.query_plan_models import OperandSlot, QueryPlan


def _join(values: list[object]) -> str:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split())
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return " | ".join(result)


def build_field_queries(plan: QueryPlan, slot: OperandSlot | None = None) -> dict[str, str]:
    metric_values: list[object] = list(plan.metric_phrases)
    axis_values: list[object] = []
    evidence_values: list[object] = list(plan.evidence_shapes)
    if slot is not None:
        metric_values.extend([slot.raw_metric_phrase, *slot.concept_candidates[:3]])
        axis_values.extend([slot.period, slot.temporal_kind, slot.segment_label, slot.bucket_label])
        evidence_values.append(slot.required_evidence_shape)
    else:
        axis_values.extend(plan.periods)
        for item in plan.operand_slots:
            metric_values.extend([item.raw_metric_phrase, *item.concept_candidates[:3]])
            axis_values.extend([item.period, item.temporal_kind, item.segment_label, item.bucket_label])
            evidence_values.append(item.required_evidence_shape)
    context_values: list[object] = [plan.statement_hint, plan.task_type, *plan.evidence_shapes]
    evidence_values.append(plan.operation)
    return {
        "metric": _join(metric_values),
        "axis": _join(axis_values),
        "context": _join(context_values),
        "evidence": _join(evidence_values),
    }


def field_queries_to_dict(plan: QueryPlan) -> dict[str, Any]:
    return {
        "main": build_field_queries(plan),
        "slots": {slot.slot_id: build_field_queries(plan, slot) for slot in plan.operand_slots},
    }
