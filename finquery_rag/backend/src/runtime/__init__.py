"""Runtime contracts, ports, and compatibility adapters."""

from .runtime_adapters import (
    LegacyFinancialRuntimeAdapter,
    LegacyFinancialRuntimeAdapterError,
    UnsupportedResolvedQueryError,
)
from .query_execution_service import QueryExecutionService
from .query_lifecycle import (
    QueryLifecycleService,
    UserTurnExecutionRequest,
    UserTurnExecutionResult,
)
from .response_mapper import (
    LegacyResponseMappingError,
    to_legacy_query_dict,
)
from .runtime_contract import (
    ClarificationPayload,
    FinancialQARuntime,
    FinancialQueryRequest,
    FinancialQueryResult,
    ReleaseStatus,
    RouterMode,
    RuntimeMetadata,
    RuntimeRouterMode,
    RuntimeStatus,
    RuntimeVersion,
)
from .trusted_v2_adapter import TrustedFinancialRuntimeV2
from .trusted_v2_capabilities import (
    CalculationCapability,
    EvidenceEvaluationCapability,
    GenerationCapability,
    ReleaseValidationCapability,
    RetrievalCapability,
    TrustedV2CapabilityPorts,
)
from .trusted_v2_binder import (
    SemanticBinderCapabilityError,
    SemanticEvidenceEvaluationCapability,
)
from .trusted_v2_calculation import (
    DeterministicCalculationCapability,
    DeterministicCalculationCapabilityError,
    SUPPORTED_CALCULATION_OPERATIONS,
)
from .trusted_v2_generation import (
    CandidateExecutionResult,
    CandidateGenerationCapabilityError,
    DeterministicFactRenderer,
    LocalSpecialistGenerationAdapter,
    TrustedV2GenerationCapability,
)
from .trusted_v2_r4 import (
    CandidateDirectR4Policy,
    R4CandidateSchemaError,
    R4RetrievalCapability,
    R4RetrievalCapabilityError,
    R4RetrievalRequest,
    R4RetrievalResult,
)
from .trusted_v2_contracts import (
    TrustedV2ExecutionCoordinator,
    V2ExecutionOutcome,
    V2ExecutionRequest,
    V2ExecutionStatus,
)
from .trusted_v2_coordinator import BoundedTrustedV2Coordinator, V2ExecutionTrace

__all__ = [
    "ClarificationPayload",
    "FinancialQARuntime",
    "FinancialQueryRequest",
    "FinancialQueryResult",
    "LegacyFinancialRuntimeAdapter",
    "LegacyResponseMappingError",
    "QueryExecutionService",
    "QueryLifecycleService",
    "UserTurnExecutionRequest",
    "UserTurnExecutionResult",
    "LegacyFinancialRuntimeAdapterError",
    "ReleaseStatus",
    "RouterMode",
    "RuntimeMetadata",
    "RuntimeRouterMode",
    "RuntimeStatus",
    "RuntimeVersion",
    "UnsupportedResolvedQueryError",
    "TrustedFinancialRuntimeV2",
    "CalculationCapability",
    "EvidenceEvaluationCapability",
    "GenerationCapability",
    "ReleaseValidationCapability",
    "RetrievalCapability",
    "TrustedV2CapabilityPorts",
    "SemanticBinderCapabilityError",
    "SemanticEvidenceEvaluationCapability",
    "DeterministicCalculationCapability",
    "DeterministicCalculationCapabilityError",
    "SUPPORTED_CALCULATION_OPERATIONS",
    "CandidateExecutionResult",
    "CandidateGenerationCapabilityError",
    "DeterministicFactRenderer",
    "LocalSpecialistGenerationAdapter",
    "TrustedV2GenerationCapability",
    "CandidateDirectR4Policy",
    "R4CandidateSchemaError",
    "R4RetrievalCapability",
    "R4RetrievalCapabilityError",
    "R4RetrievalRequest",
    "R4RetrievalResult",
    "TrustedV2ExecutionCoordinator",
    "V2ExecutionOutcome",
    "V2ExecutionRequest",
    "V2ExecutionStatus",
    "BoundedTrustedV2Coordinator",
    "V2ExecutionTrace",
    "to_legacy_query_dict",
]
