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
from .runtime_router import FinancialRuntimeRouter
from .shadow_comparator import ShadowComparator, ShadowComparison
from .shadow_contracts import (
    FinancialRuntimeMode,
    FinancialRuntimeModeError,
    InMemoryShadowObservationSink,
    LoggingShadowObservationSink,
    ShadowObservationSink,
    V2ShadowObservation,
    resolve_financial_runtime_mode,
)
from .trusted_v2_adapter import TrustedFinancialRuntimeV2
from .trusted_v2_factory import TrustedV2FactoryError, build_trusted_v2_runtime
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
from .trusted_v2_validation import (
    CandidateRepairCapability,
    CandidateRepairError,
    CandidateRepairUnavailable,
    DeterministicCandidateRepair,
    TrustedReleaseValidationCapability,
    V2ValidationResult,
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
from .trusted_v2_production import (
    StructuredFactStore,
    TrustedV2ProductionConfigurationError,
    TrustedV2RuntimeResources,
    build_trusted_v2_runtime_for_request,
    clear_trusted_v2_production_cache,
    inspect_r4_index,
    validate_trusted_v2_production_configuration,
)

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
    "FinancialRuntimeMode",
    "FinancialRuntimeModeError",
    "FinancialRuntimeRouter",
    "InMemoryShadowObservationSink",
    "LoggingShadowObservationSink",
    "ShadowComparator",
    "ShadowComparison",
    "ShadowObservationSink",
    "V2ShadowObservation",
    "resolve_financial_runtime_mode",
    "UnsupportedResolvedQueryError",
    "TrustedFinancialRuntimeV2",
    "TrustedV2FactoryError",
    "build_trusted_v2_runtime",
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
    "CandidateRepairCapability",
    "CandidateRepairError",
    "CandidateRepairUnavailable",
    "DeterministicCandidateRepair",
    "TrustedReleaseValidationCapability",
    "V2ValidationResult",
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
    "StructuredFactStore",
    "TrustedV2ProductionConfigurationError",
    "TrustedV2RuntimeResources",
    "build_trusted_v2_runtime_for_request",
    "clear_trusted_v2_production_cache",
    "inspect_r4_index",
    "validate_trusted_v2_production_configuration",
    "to_legacy_query_dict",
]
