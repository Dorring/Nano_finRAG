"""Explicit V1, V2-shadow, and V2-official runtime router.

The router is an implementation of the existing FinancialQARuntime port. In
v1 mode it invokes only the legacy runtime, in shadow mode it returns the V1
result while observing a bounded V2 branch, and in v2 mode it returns only the
Trusted V2 result. No mode silently falls back to another runtime.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from .runtime_contract import (
    FinancialQARuntime,
    FinancialQueryRequest,
    FinancialQueryResult,
)
from .shadow_comparator import ShadowComparator
from .shadow_contracts import (
    LoggingShadowObservationSink,
    ShadowObservationSink,
    V2ShadowObservation,
    resolve_financial_runtime_mode,
)

logger = logging.getLogger(__name__)


def _value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _tuple_ids(value: Any) -> tuple[str, ...]:
    if value is None or isinstance(value, (str, bytes)):
        return ()
    return tuple(dict.fromkeys(str(item) for item in value if str(item).strip()))


def _runtime_trace(result: FinancialQueryResult | None) -> Mapping[str, Any]:
    if result is None:
        return {}
    debug = result.debug_metadata
    if isinstance(debug, Mapping) and isinstance(debug.get("trace"), Mapping):
        return debug["trace"]
    return {}


def _error_stage(result: FinancialQueryResult | None) -> str | None:
    if result is None:
        return None
    codes = _tuple_ids(result.reason_codes)
    for stage in (
        "SUPERVISOR",
        "PLAN_VALIDATION",
        "RETRIEVAL",
        "BINDER",
        "CALCULATION",
        "GENERATION",
        "VALIDATION",
        "REPAIR",
    ):
        if any(stage in code for code in codes):
            return stage
    return "V2_RESULT" if _value(result.status) == "ERROR" else None


class FinancialRuntimeRouter(FinancialQARuntime):
    """Route explicit V1, V2-shadow, or V2-official execution modes."""

    def __init__(
        self,
        primary_runtime: FinancialQARuntime | None = None,
        *,
        shadow_runtime: FinancialQARuntime | None = None,
        v2_runtime: FinancialQARuntime | None = None,
        mode: str = "v1",
        shadow_timeout_ms: int = 5_000,
        observation_sink: ShadowObservationSink | Callable[[V2ShadowObservation], None] | None = None,
        comparator: ShadowComparator | None = None,
    ) -> None:
        normalized_mode = resolve_financial_runtime_mode(mode)
        if shadow_runtime is not None and v2_runtime is not None:
            raise ValueError("provide only one of shadow_runtime or v2_runtime")
        selected_v2 = v2_runtime if v2_runtime is not None else shadow_runtime
        if normalized_mode in {"v1", "shadow"} and not callable(
            getattr(primary_runtime, "execute", None),
        ):
            raise TypeError("primary_runtime must implement FinancialQARuntime")
        if selected_v2 is not None and not callable(getattr(selected_v2, "execute", None)):
            raise TypeError("v2 runtime must implement FinancialQARuntime")
        if normalized_mode in {"shadow", "v2"} and selected_v2 is None:
            raise ValueError(f"{normalized_mode} mode requires an explicit real V2 runtime")
        if isinstance(shadow_timeout_ms, bool) or int(shadow_timeout_ms) <= 0:
            raise ValueError("shadow_timeout_ms must be a positive integer")
        self.primary_runtime = primary_runtime
        self.shadow_runtime = selected_v2
        self.v2_runtime = selected_v2
        self.mode = normalized_mode
        self.shadow_timeout_ms = int(shadow_timeout_ms)
        self.comparator = comparator or ShadowComparator()
        self.observation_sink = observation_sink or LoggingShadowObservationSink(logger)
        self.last_observation: V2ShadowObservation | None = None
        self.v1_calls = 0
        self.v2_calls = 0
        self.v2_timeout_count = 0
        self.v2_error_count = 0

    async def _invoke(
        self,
        runtime: FinancialQARuntime,
        request: FinancialQueryRequest,
    ) -> FinancialQueryResult:
        result = runtime.execute(request)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, FinancialQueryResult):
            raise TypeError("runtime returned an invalid FinancialQueryResult")
        return result

    @staticmethod
    def _runtime_latency(result: FinancialQueryResult | None) -> float | None:
        if result is None:
            return None
        value = result.latency_metadata.get("latency_ms")
        try:
            return float(value) if isinstance(value, (int, float)) else None
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _observation_failure(
        request: FinancialQueryRequest,
        primary: FinancialQueryResult | None,
        shadow: FinancialQueryResult | None,
        error: Exception,
    ) -> V2ShadowObservation:
        return V2ShadowObservation(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            original_query=request.original_query,
            standalone_query=request.standalone_query,
            v1_status=_value(getattr(primary, "status", None)) or "ERROR",
            v1_release_status=_value(getattr(primary, "release_status", None)),
            v1_answer=getattr(primary, "answer", None),
            v2_status=_value(getattr(shadow, "status", None)) if shadow else None,
            v2_release_status=_value(getattr(shadow, "release_status", None)) if shadow else None,
            v2_answer=getattr(shadow, "answer", None) if shadow else None,
            shadow_status="ERROR",
            shadow_error_stage="OBSERVATION",
            shadow_error_code=type(error).__name__,
            comparison={
                "decision_parity": "UNAVAILABLE",
                "answer_semantic_parity": "UNAVAILABLE",
                "provenance_parity": "UNAVAILABLE",
                "calculation_parity": "UNAVAILABLE",
                "category": "V2_ERROR",
                "needs_review": True,
            },
        )

    def _observation(
        self,
        *,
        request: FinancialQueryRequest,
        primary: FinancialQueryResult | None,
        shadow: FinancialQueryResult | None,
        shadow_status: str,
        shadow_error_stage: str | None,
        shadow_error_code: str | None,
        v1_started: float,
        v2_started: float,
    ) -> V2ShadowObservation:
        trace = _runtime_trace(shadow)
        retrieval_rounds = trace.get("retrieval_rounds", ())
        if not isinstance(retrieval_rounds, (list, tuple)):
            retrieval_rounds = ()
        route = trace.get("generation_route")
        if route is None and shadow is not None and shadow.runtime_metadata is not None:
            attrs = shadow.runtime_metadata.attributes
            route = attrs.get("route") if isinstance(attrs, Mapping) else None
        comparison = self.comparator.compare(
            primary,
            shadow,
            shadow_status=shadow_status,
        )
        return V2ShadowObservation(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            original_query=request.original_query,
            standalone_query=request.standalone_query,
            primary_runtime_version=_value(primary.runtime_version) if primary else "V1",
            shadow_runtime_version=_value(shadow.runtime_version) if shadow else "V2",
            v1_status=_value(primary.status) if primary else "ERROR",
            v1_release_status=_value(primary.release_status) if primary else "NOT_RELEASED",
            v2_status=_value(shadow.status) if shadow else None,
            v2_release_status=_value(shadow.release_status) if shadow else None,
            v1_answer=primary.answer if primary else None,
            v2_answer=shadow.answer if shadow else None,
            v1_evidence_ids=_tuple_ids(primary.evidence_ids) if primary else (),
            v2_evidence_ids=_tuple_ids(shadow.evidence_ids) if shadow else (),
            v1_citation_ids=_tuple_ids(primary.citation_ids) if primary else (),
            v2_citation_ids=_tuple_ids(shadow.citation_ids) if shadow else (),
            v1_calculation_ids=_tuple_ids(primary.calculation_ids) if primary else (),
            v2_calculation_ids=_tuple_ids(shadow.calculation_ids) if shadow else (),
            v1_reason_codes=_tuple_ids(primary.reason_codes) if primary else (),
            v2_reason_codes=_tuple_ids(shadow.reason_codes) if shadow else (),
            v2_route=str(route) if route is not None else None,
            v1_latency_ms=self._runtime_latency(primary) or (time.perf_counter() - v1_started) * 1000,
            v2_latency_ms=self._runtime_latency(shadow) or (
                (time.perf_counter() - v2_started) * 1000 if shadow is not None else None
            ),
            v2_retrieval_rounds=len(retrieval_rounds),
            v2_repair_count=int(trace.get("repair_count", 0) or 0),
            shadow_status=shadow_status,
            shadow_error_stage=shadow_error_stage,
            shadow_error_code=shadow_error_code,
            comparison=comparison.to_dict(),
        )

    def _record(self, observation: V2ShadowObservation) -> None:
        self.last_observation = observation
        try:
            if callable(self.observation_sink):
                self.observation_sink(observation)
            else:
                self.observation_sink.record(observation)
        except Exception:
            # Observation failure is never allowed to change the official V1
            # result or HTTP semantics.
            logger.exception("v2 shadow observation sink failed")

    async def execute(self, request: FinancialQueryRequest) -> FinancialQueryResult:
        """Execute the selected runtime with no implicit fallback."""

        if not isinstance(request, FinancialQueryRequest):
            raise TypeError("request must be FinancialQueryRequest")

        if self.mode == "v2":
            v2_runtime = self.v2_runtime
            if v2_runtime is None:  # defensive: constructor already guards it
                raise RuntimeError("v2 runtime is not configured")
            self.v2_calls += 1
            return await self._invoke(v2_runtime, request)

        primary_runtime = self.primary_runtime
        if primary_runtime is None:  # defensive: constructor already guards it
            raise RuntimeError("primary runtime is not configured")
        self.v1_calls += 1
        v1_started = time.perf_counter()
        if self.mode == "v1":
            return await self._invoke(primary_runtime, request)

        shadow_runtime = self.shadow_runtime
        if shadow_runtime is None:  # defensive: constructor already guards it
            raise RuntimeError("shadow runtime is not configured")

        self.v2_calls += 1
        v2_started = time.perf_counter()
        primary_task = asyncio.create_task(self._invoke(primary_runtime, request))
        shadow_task = asyncio.create_task(self._invoke(shadow_runtime, request))
        primary: FinancialQueryResult | None = None
        primary_error: BaseException | None = None
        shadow: FinancialQueryResult | None = None
        shadow_status = "COMPLETED"
        shadow_error_stage: str | None = None
        shadow_error_code: str | None = None

        try:
            try:
                primary = await primary_task
            except BaseException as exc:
                primary_error = exc
            remaining_timeout = (
                self.shadow_timeout_ms / 1000
                - (time.perf_counter() - v2_started)
            )
            if remaining_timeout <= 0:
                self.v2_timeout_count += 1
                shadow_status = "TIMEOUT"
                shadow_error_stage = "TIMEOUT"
                shadow_error_code = "V2_SHADOW_TIMEOUT"
                shadow_task.cancel()
                await asyncio.gather(shadow_task, return_exceptions=True)
            else:
                shadow = await asyncio.wait_for(
                    shadow_task,
                    timeout=remaining_timeout,
                )
        except asyncio.TimeoutError:
            self.v2_timeout_count += 1
            shadow_status = "TIMEOUT"
            shadow_error_stage = "TIMEOUT"
            shadow_error_code = "V2_SHADOW_TIMEOUT"
        except asyncio.CancelledError:
            self.v2_timeout_count += 1
            shadow_status = "TIMEOUT"
            shadow_error_stage = "TIMEOUT"
            shadow_error_code = "V2_SHADOW_CANCELLED"
        except Exception as exc:
            self.v2_error_count += 1
            shadow_status = "ERROR"
            shadow_error_stage = "V2_EXCEPTION"
            shadow_error_code = type(exc).__name__
        finally:
            if not shadow_task.done():
                shadow_task.cancel()
                await asyncio.gather(shadow_task, return_exceptions=True)

        if shadow_status == "COMPLETED" and shadow is not None and _value(shadow.status) == "ERROR":
            shadow_status = "ERROR"
            shadow_error_stage = _error_stage(shadow)
            shadow_error_code = next(iter(_tuple_ids(shadow.reason_codes)), "V2_EXECUTION_ERROR")

        try:
            observation = self._observation(
                request=request,
                primary=primary,
                shadow=shadow,
                shadow_status=shadow_status,
                shadow_error_stage=shadow_error_stage,
                shadow_error_code=shadow_error_code,
                v1_started=v1_started,
                v2_started=v2_started,
            )
        except Exception as exc:
            self.v2_error_count += 1
            logger.exception("v2 shadow observation construction failed")
            observation = self._observation_failure(request, primary, shadow, exc)
        self._record(observation)

        if primary_error is not None:
            raise primary_error
        if primary is None:
            raise RuntimeError("primary runtime returned no result")
        return primary


__all__ = ["FinancialRuntimeRouter"]