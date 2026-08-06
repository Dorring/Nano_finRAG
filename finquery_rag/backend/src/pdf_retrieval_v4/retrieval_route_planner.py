from __future__ import annotations

from .query_plan_models import RetrievalConstraints, RetrievalRoute


def _route(route_id: str, index_type: str, stage: str, *, slots=(), required=False, auxiliary=False, query_source="raw_question", filters=None) -> RetrievalRoute:
    return RetrievalRoute(route_id, index_type, stage, tuple(slots), query_source, required, auxiliary, dict(filters or {}))


def build_routes(task_type: str, operation: str | None, slots, *, bucket_label: str | None = None) -> tuple[RetrievalRoute, ...]:
    routes: list[RetrievalRoute] = []
    if task_type != "unsupported":
        routes.append(_route("raw_production", "raw_production", "raw", required=True))
    else:
        return (_route("raw_production", "raw_production", "raw", required=True),)
    if task_type == "narrative_or_note":
        routes.append(_route("section_context", "section", "context", required=True))
        return tuple(routes)
    if task_type == "general_single_fact":
        routes.append(_route("section_context", "section", "context", required=True))
        routes.append(_route("table_context", "table", "context", auxiliary=True))
        routes.append(_route("row_matrix", "row", "local", auxiliary=True))
        if slots and slots[0].period:
            routes.append(_route("atomic_fact", "atomic_fact", "fact", slots=(slots[0].slot_id,), auxiliary=True, query_source="raw_metric_phrase"))
        return tuple(routes)
    routes.extend([
        _route("table_context", "table", "context", required=True),
        _route("row_matrix", "row", "local", required=True),
    ])
    if bucket_label:
        routes.append(_route("bucket_fact", "bucket_fact", "fact", slots=tuple(s.slot_id for s in slots), required=True, query_source="raw_metric_phrase"))
        if slots:
            routes.append(_route("atomic_fact", "atomic_fact", "fact", slots=tuple(s.slot_id for s in slots), auxiliary=True, query_source="raw_metric_phrase"))
    elif slots:
        for slot in slots:
            routes.append(_route(f"atomic_{slot.slot_id}", "atomic_fact", "fact", slots=(slot.slot_id,), required=True, query_source="raw_metric_phrase", filters={"slot_id": slot.slot_id}))
        routes.append(_route("cell_context", "cell", "fact", slots=tuple(s.slot_id for s in slots), auxiliary=True, query_source="raw_metric_phrase"))
    if task_type in {"single_metric_multi_period", "multi_metric_comparison", "calculation_multi_operand"} and operation in {"difference", "growth_rate"}:
        routes.append(_route("comparison_fact", "comparison_fact", "fact", slots=tuple(s.slot_id for s in slots), auxiliary=True, query_source="raw_metric_phrase"))
    return tuple(routes)


def build_constraints(task_type: str, *, multi_source: bool) -> RetrievalConstraints:
    return RetrievalConstraints(
        same_document=True,
        prefer_same_logical_table=multi_source or task_type in {"table_single_fact", "general_single_fact"},
        prefer_same_row=task_type in {"table_single_fact", "single_metric_multi_period"},
        candidate_identity_unique=multi_source,
        soft_continuation_expansion=False,
        follow_soft_link=False,
        merge_neighbor_table=False,
        inherit_previous_header=False,
    )
