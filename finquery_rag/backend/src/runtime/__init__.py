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
from .trusted_v2_contracts import (
    TrustedV2ExecutionCoordinator,
    V2ExecutionOutcome,
    V2ExecutionRequest,
    V2ExecutionStatus,
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
    "UnsupportedResolvedQueryError",
    "TrustedFinancialRuntimeV2",
    "TrustedV2ExecutionCoordinator",
    "V2ExecutionOutcome",
    "V2ExecutionRequest",
    "V2ExecutionStatus",
    "to_legacy_query_dict",
]
