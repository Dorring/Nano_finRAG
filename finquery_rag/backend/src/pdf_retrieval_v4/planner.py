from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path

from src.retrieval_v3.query_router import route_question

from .evidence_shapes import classify_evidence_shapes, detect_bucket_label
from .operand_planner import build_operand_slots
from .query_plan_models import QueryPlan
from .query_plan_validator import validate_query_plan
from .retrieval_route_planner import build_constraints, build_routes


PLAN_VERSION = "pdf-v4-query-plan-v1"


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


class ConceptResolver:
    """Question-only, deterministic Top-3 label resolver."""

    def __init__(self, registry_path: Path | None = None):
        self.records: list[dict[str, object]] = []
        if registry_path and registry_path.exists():
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            self.records = list(payload.get("records", []))

    def resolve(self, phrase: str) -> tuple[str, ...]:
        query = _norm(phrase)
        if not query:
            return ()
        q_tokens = set(query.split())
        ranked: list[tuple[float, str, str]] = []
        for record in self.records:
            labels = [str(record.get("canonical_label") or "")]
            labels.extend(str(item) for item in record.get("labels", []) or [])
            labels.extend(str(item) for item in record.get("generic_aliases", []) or [])
            best = 0.0
            best_label = labels[0] if labels else ""
            for label in labels:
                normalized = _norm(label)
                if not normalized:
                    continue
                tokens = set(normalized.split())
                overlap = len(q_tokens & tokens) / max(1, len(q_tokens | tokens))
                char = SequenceMatcher(None, query, normalized).ratio()
                exact = 1.0 if query == normalized else 0.0
                score = exact * 2.0 + overlap + char * 0.25
                if score > best:
                    best, best_label = score, label
            if best >= 0.35:
                ranked.append((-best, _norm(best_label), best_label))
        ranked.sort()
        result: list[str] = []
        for _, _, label in ranked:
            if label not in result:
                result.append(label)
            if len(result) == 3:
                break
        return tuple(result)


def build_query_plan(question: str, document_scope: tuple[str, ...], resolver: ConceptResolver | None = None) -> QueryPlan:
    profile = route_question(question, document_scope=document_scope)
    resolver_fn = resolver.resolve if resolver is not None else None
    slots = build_operand_slots(profile, question, resolver_fn)
    bucket = detect_bucket_label(question)
    shapes = classify_evidence_shapes(
        profile.task_type,
        profile.operation,
        bucket_label=bucket,
        metric_count=len(profile.metric_phrases),
        period_count=len(profile.periods),
    )
    routes = build_routes(profile.task_type, profile.operation, slots, bucket_label=bucket)
    constraints = build_constraints(profile.task_type, multi_source=profile.requires_multiple_sources)
    normalized_scope = tuple(str(item) for item in document_scope if str(item))
    plan_key = json.dumps({"version": PLAN_VERSION, "question": question, "scope": normalized_scope}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    plan_id = hashlib.sha256(plan_key).hexdigest()
    plan = QueryPlan(
        plan_id=plan_id,
        plan_version=PLAN_VERSION,
        raw_question=question,
        document_scope=normalized_scope,
        task_type=profile.task_type,
        operation=profile.operation,
        issuer=profile.issuer,
        metric_phrases=tuple(item.raw_text for item in profile.metric_phrases),
        periods=tuple(item.normalized_period for item in profile.periods if item.normalized_period),
        evidence_shapes=shapes,
        operand_slots=slots,
        retrieval_routes=routes,
        constraints=constraints,
        raw_protection_required=True,
        answerability_check_required=profile.answerability_check_required,
        routing_reasons=profile.routing_reasons,
        unresolved_reasons=profile.unresolved_reasons,
        statement_hint=profile.statement_hint,
        requires_multiple_sources=profile.requires_multiple_sources,
    )
    errors = validate_query_plan(plan)
    return replace(plan, plan_status="blocked" if errors else "planned", validation_errors=errors)
