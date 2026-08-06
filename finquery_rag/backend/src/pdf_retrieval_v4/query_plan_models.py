from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


EvidenceShape = Literal[
    "narrative_section",
    "table_context",
    "row_matrix",
    "atomic_fact",
    "comparison_fact",
    "bucket_fact",
    "multi_operand_set",
    "raw_fallback",
]


@dataclass(frozen=True)
class OperandSlot:
    slot_id: str
    role: str
    raw_metric_phrase: str
    concept_candidates: tuple[str, ...] = ()
    period: str | None = None
    temporal_kind: str | None = None
    bucket_label: str | None = None
    segment_label: str | None = None
    required_evidence_shape: str = "atomic_fact"


@dataclass(frozen=True)
class RetrievalRoute:
    route_id: str
    index_type: str
    stage: str
    slot_ids: tuple[str, ...] = ()
    query_source: str = "raw_question"
    required: bool = False
    auxiliary: bool = False
    metadata_filters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalConstraints:
    same_document: bool = True
    prefer_same_logical_table: bool = False
    prefer_same_row: bool = False
    candidate_identity_unique: bool = True
    soft_continuation_expansion: bool = False
    follow_soft_link: bool = False
    merge_neighbor_table: bool = False
    inherit_previous_header: bool = False


@dataclass(frozen=True)
class QueryPlan:
    plan_id: str
    plan_version: str
    raw_question: str
    document_scope: tuple[str, ...]
    task_type: str
    operation: str | None
    issuer: str | None
    metric_phrases: tuple[str, ...]
    periods: tuple[str, ...]
    evidence_shapes: tuple[str, ...]
    operand_slots: tuple[OperandSlot, ...]
    retrieval_routes: tuple[RetrievalRoute, ...]
    constraints: RetrievalConstraints
    raw_protection_required: bool
    answerability_check_required: bool
    routing_reasons: tuple[str, ...] = ()
    unresolved_reasons: tuple[str, ...] = ()
    statement_hint: str | None = None
    requires_multiple_sources: bool = False
    plan_status: str = "planned"
    validation_errors: tuple[str, ...] = ()

