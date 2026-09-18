"""V2 supervisor contracts; model providers arrive in a later gate."""

from .api_provider import APIProvider
from .bailian_provider import BailianProvider
from .deterministic_fallback import DeterministicFallbackProvider
from .local_provider import LocalProvider
from .plan_validator import validate_plan, validate_plan_v2_01
from .plan_normalizer import (
    PlanNormalization,
    derive_slot_identities,
    normalize_supervisor_plan,
)
from .provider import SupervisorCallMetadata, SupervisorProvider, SupervisorProviderError
from .semantic_alignment import (
    BoundEvidenceAlignmentStatus,
    BoundEvidenceSemanticCheck,
    EvidenceScope,
    EvidenceScopeClassification,
    EntityMention,
    MetricDefinition,
    MetricMention,
    OperationMention,
    PlanSemanticAlignment,
    PeriodMention,
    QuerySemanticFrame,
    SemanticAlignmentStatus,
    UnknownSemanticPolicy,
    align_query_to_plan,
    align_bound_evidence_to_query,
    canonical_entity_id,
    canonical_metric_id,
    metric_alias_registry,
    canonical_operation_id,
    canonical_period_id,
    canonical_scope_id,
    classify_evidence_scope,
    coerce_unknown_semantic_policy,
    extract_query_semantic_frame,
    query_allows_evidence_scope,
)
from .service import SupervisorRun, SupervisorService
from .strong_general_provider import StrongGeneralAPIProvider

__all__ = [
    "APIProvider",
    "BailianProvider",
    "DeterministicFallbackProvider",
    "LocalProvider",
    "SupervisorCallMetadata",
    "SupervisorProvider",
    "SupervisorProviderError",
    "PlanNormalization",
    "normalize_supervisor_plan",
    "derive_slot_identities",
    "SupervisorRun",
    "SupervisorService",
    "StrongGeneralAPIProvider",
    "BoundEvidenceAlignmentStatus",
    "BoundEvidenceSemanticCheck",
    "EvidenceScope",
    "EvidenceScopeClassification",
    "EntityMention",
    "MetricDefinition",
    "MetricMention",
    "OperationMention",
    "PlanSemanticAlignment",
    "PeriodMention",
    "QuerySemanticFrame",
    "SemanticAlignmentStatus",
    "UnknownSemanticPolicy",
    "align_query_to_plan",
    "align_bound_evidence_to_query",
    "canonical_entity_id",
    "canonical_metric_id",
    "metric_alias_registry",
    "canonical_operation_id",
    "canonical_period_id",
    "canonical_scope_id",
    "classify_evidence_scope",
    "coerce_unknown_semantic_policy",
    "extract_query_semantic_frame",
    "query_allows_evidence_scope",
    "validate_plan",
    "validate_plan_v2_01",
]
