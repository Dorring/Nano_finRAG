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
    "to_legacy_query_dict",
]
