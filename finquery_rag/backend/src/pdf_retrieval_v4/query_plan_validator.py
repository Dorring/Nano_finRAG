from __future__ import annotations

from .query_plan_models import QueryPlan


_TASK_TYPES = {
    "table_single_fact",
    "general_single_fact",
    "single_metric_multi_period",
    "multi_metric_comparison",
    "calculation_multi_operand",
    "narrative_or_note",
    "unsupported",
}


def validate_query_plan(plan: QueryPlan) -> tuple[str, ...]:
    errors: list[str] = []
    if not plan.raw_question.strip():
        errors.append("raw_question_empty")
    if not plan.document_scope:
        errors.append("document_scope_empty")
    if plan.task_type not in _TASK_TYPES:
        errors.append("invalid_task_type")
    if not plan.raw_protection_required:
        errors.append("raw_protection_disabled")
    if len({slot.slot_id for slot in plan.operand_slots}) != len(plan.operand_slots):
        errors.append("duplicate_slot_id")
    route_ids = {route.route_id for route in plan.retrieval_routes}
    if len(route_ids) != len(plan.retrieval_routes):
        errors.append("duplicate_route_id")
    slot_ids = {slot.slot_id for slot in plan.operand_slots}
    for route in plan.retrieval_routes:
        if not set(route.slot_ids).issubset(slot_ids):
            errors.append(f"route_slot_missing:{route.route_id}")
    if not any(route.index_type == "raw_production" and route.required for route in plan.retrieval_routes):
        errors.append("raw_route_missing")
    if plan.constraints.soft_continuation_expansion or plan.constraints.follow_soft_link or plan.constraints.merge_neighbor_table or plan.constraints.inherit_previous_header:
        errors.append("soft_continuation_expansion_enabled")
    if plan.requires_multiple_sources:
        if len(plan.operand_slots) < 2:
            errors.append("multi_source_slot_count")
        if "multi_operand_set" not in plan.evidence_shapes:
            errors.append("multi_operand_shape_missing")
    if plan.task_type == "calculation_multi_operand":
        if not plan.operation:
            errors.append("calculation_operation_missing")
        if len(plan.operand_slots) < 2 and plan.operation != "scale_conversion":
            errors.append("calculation_operands_missing")
        if not any(route.index_type == "atomic_fact" and route.required for route in plan.retrieval_routes):
            errors.append("calculation_atomic_route_missing")
    if plan.task_type == "narrative_or_note":
        if not any(route.index_type == "section" and route.required for route in plan.retrieval_routes):
            errors.append("narrative_section_route_missing")
        forbidden = {"atomic_fact", "comparison_fact", "bucket_fact", "cell"}
        if any(route.index_type in forbidden for route in plan.retrieval_routes):
            errors.append("narrative_structured_leakage")
    if plan.task_type == "unsupported":
        structured = [route for route in plan.retrieval_routes if route.index_type != "raw_production"]
        if structured:
            errors.append("unsupported_structured_route")
    if any(slot.bucket_label for slot in plan.operand_slots) and not any(route.index_type == "bucket_fact" and route.required for route in plan.retrieval_routes):
        errors.append("bucket_route_missing")
    if "comparison_fact" in plan.evidence_shapes and not any(route.index_type == "comparison_fact" for route in plan.retrieval_routes):
        errors.append("comparison_route_missing")
    return tuple(dict.fromkeys(errors))
