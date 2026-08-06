from __future__ import annotations

from .query_plan_models import OperandSlot, QueryPlan, RetrievalConstraints, RetrievalRoute


def query_plan_from_dict(value: dict[str, object]) -> QueryPlan:
    slots = tuple(OperandSlot(**slot) for slot in value.get("operand_slots", []))
    routes = tuple(RetrievalRoute(**route) for route in value.get("retrieval_routes", []))
    constraints = RetrievalConstraints(**value.get("constraints", {}))
    return QueryPlan(
        plan_id=str(value.get("plan_id", "")),
        plan_version=str(value.get("plan_version", "")),
        raw_question=str(value.get("raw_question", "")),
        document_scope=tuple(value.get("document_scope", [])),
        task_type=str(value.get("task_type", "")),
        operation=value.get("operation"),
        issuer=value.get("issuer"),
        metric_phrases=tuple(value.get("metric_phrases", [])),
        periods=tuple(value.get("periods", [])),
        evidence_shapes=tuple(value.get("evidence_shapes", [])),
        operand_slots=slots,
        retrieval_routes=routes,
        constraints=constraints,
        raw_protection_required=bool(value.get("raw_protection_required")),
        answerability_check_required=bool(value.get("answerability_check_required")),
        routing_reasons=tuple(value.get("routing_reasons", [])),
        unresolved_reasons=tuple(value.get("unresolved_reasons", [])),
        statement_hint=value.get("statement_hint"),
        requires_multiple_sources=bool(value.get("requires_multiple_sources")),
        plan_status=str(value.get("plan_status", "planned")),
        validation_errors=tuple(value.get("validation_errors", [])),
    )
