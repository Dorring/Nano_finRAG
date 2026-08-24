"""Narrow execution boundary used by the I3 /query migration."""

from __future__ import annotations

from .runtime_contract import (
    FinancialQARuntime,
    FinancialQueryRequest,
    FinancialQueryResult,
)


class QueryExecutionService:
    """Translate an endpoint-built request into one runtime execution.

    The service deliberately owns no authentication, session lifecycle,
    conversation state, response serialization, or V1/V2 routing. Those
    responsibilities remain at their current layers until later milestones.
    """

    def __init__(self, runtime: FinancialQARuntime) -> None:
        self._runtime = runtime

    async def execute(self, request: FinancialQueryRequest) -> FinancialQueryResult:
        """Execute exactly once through the injected runtime."""
        if not isinstance(request, FinancialQueryRequest):
            raise TypeError("request must be FinancialQueryRequest")
        return await self._runtime.execute(request)
