"""Thin FinancialQARuntime adapter for the future Trusted V2 coordinator.

TV2-01 is a contract shell, not a production V2 route.  The adapter accepts
an injected coordinator and deliberately does not import or call
TrustedRAGRuntimeV2.handle(), because that component starts after evidence
preparation.  A real coordinator will be assembled in TV2-02 through TV2-05.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from typing import Any

from .runtime_contract import (
    FinancialQARuntime,
    FinancialQueryRequest,
    FinancialQueryResult,
    ReleaseStatus,
    RuntimeMetadata,
    RuntimeStatus,
    RuntimeVersion,
)
from .trusted_v2_contracts import (
    TrustedV2ExecutionCoordinator,
    V2ExecutionOutcome,
    V2ExecutionRequest,
    V2ExecutionStatus,
)


class TrustedFinancialRuntimeV2(FinancialQARuntime):
    """Expose a future full V2 coordinator through the shared runtime port.

    The coordinator is injected so this phase cannot accidentally create a
    fake production execution root.  No instance of this adapter is registered
    by QueryLifecycleService in TV2-01.
    """

    def __init__(self, coordinator: TrustedV2ExecutionCoordinator) -> None:
        if not callable(getattr(coordinator, "execute", None)):
            raise TypeError("coordinator must expose an execute method")
        self._coordinator = coordinator

    @property
    def coordinator(self) -> TrustedV2ExecutionCoordinator:
        """Return the injected coordinator for explicit composition/testing."""

        return self._coordinator

    @staticmethod
    def _metadata(
        *,
        coordinator: TrustedV2ExecutionCoordinator,
        outcome_metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeMetadata:
        attributes = copy.deepcopy(dict(outcome_metadata or {}))
        attributes.setdefault("production_routing", False)
        attributes.setdefault("coordinator_type", type(coordinator).__name__)
        return RuntimeMetadata(
            implementation="trusted_v2_adapter",
            config_version="tv2-01",
            attributes=attributes,
        )

    def _error_result(
        self,
        *,
        reason_code: str,
        exception: Exception | None = None,
    ) -> FinancialQueryResult:
        debug_metadata: dict[str, Any] = {}
        if exception is not None:
            debug_metadata["exception_type"] = type(exception).__name__
        return FinancialQueryResult(
            status=RuntimeStatus.ERROR,
            answer=None,
            citations=[],
            evidence_ids=[],
            citation_ids=[],
            calculation_ids=[],
            reason_codes=[reason_code],
            runtime_version=RuntimeVersion.V2,
            release_status=ReleaseStatus.NOT_RELEASED,
            debug_metadata=debug_metadata,
            runtime_metadata=self._metadata(coordinator=self._coordinator),
        )

    async def execute(self, request: FinancialQueryRequest) -> FinancialQueryResult:
        """Execute one canonical standalone query through the injected V2 port."""

        if not isinstance(request, FinancialQueryRequest):
            raise TypeError("request must be a FinancialQueryRequest")
        v2_request = V2ExecutionRequest.from_financial_request(request)
        try:
            raw_outcome = self._coordinator.execute(v2_request)
            if inspect.isawaitable(raw_outcome):
                raw_outcome = await raw_outcome
        except Exception as exc:
            return self._error_result(
                reason_code="V2_COORDINATOR_EXCEPTION",
                exception=exc,
            )

        if not isinstance(raw_outcome, V2ExecutionOutcome):
            return self._error_result(
                reason_code="V2_OUTCOME_INVALID",
            )
        return self._map_outcome(raw_outcome)

    def _map_outcome(self, outcome: V2ExecutionOutcome) -> FinancialQueryResult:
        if outcome.status is V2ExecutionStatus.READY_FOR_RELEASE:
            status = RuntimeStatus.ANSWER
        elif outcome.status is V2ExecutionStatus.FAIL_CLOSED:
            status = RuntimeStatus.FAIL_CLOSED
        else:
            status = RuntimeStatus.ERROR

        runtime_attributes = copy.deepcopy(outcome.runtime_metadata)
        if outcome.route is not None:
            runtime_attributes.setdefault("route", outcome.route)
        if outcome.validator_status is not None:
            runtime_attributes.setdefault(
                "validator_status",
                outcome.validator_status,
            )
        for name in (
            "plan_id",
            "evidence_packet_id",
            "calculation_result_id",
        ):
            value = getattr(outcome, name)
            if value is not None:
                runtime_attributes.setdefault(name, value)
        debug_metadata = copy.deepcopy(outcome.debug_metadata)
        debug_metadata.setdefault("v2_execution_status", outcome.status.value)

        return FinancialQueryResult(
            status=status,
            answer=outcome.answer,
            citations=copy.deepcopy(outcome.citations),
            evidence_ids=list(outcome.evidence_ids),
            citation_ids=list(outcome.citation_ids),
            calculation_ids=list(outcome.calculation_ids),
            reason_codes=list(outcome.reason_codes),
            runtime_version=RuntimeVersion.V2,
            release_status=outcome.release_status,
            latency_metadata=copy.deepcopy(outcome.latency_metadata),
            debug_metadata=debug_metadata,
            runtime_metadata=self._metadata(
                coordinator=self._coordinator,
                outcome_metadata=runtime_attributes,
            ),
        )


__all__ = ["TrustedFinancialRuntimeV2"]
