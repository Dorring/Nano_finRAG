"""Contracts for V1-primary/V2-shadow execution.

TV2-06 keeps shadow output strictly outside the FinancialQueryResult returned
to callers. These types are observation-only and have no SessionManager or
ConversationStateStore dependency.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class FinancialRuntimeMode(str, Enum):
    """Deployment modes supported by TV2-06."""

    V1 = "v1"
    SHADOW = "shadow"


class FinancialRuntimeModeError(ValueError):
    """Raised for an unsupported or incomplete runtime mode."""


def resolve_financial_runtime_mode(
    mode: Any = None,
    *,
    environ: Mapping[str, Any] | None = None,
) -> str:
    """Validate FINANCIAL_RUNTIME_MODE without accepting active V2."""

    if mode is None and environ is not None:
        mode = environ.get("FINANCIAL_RUNTIME_MODE")
    raw_mode = getattr(mode, "value", mode)
    normalized = "v1" if raw_mode is None else str(raw_mode).strip().lower()
    if normalized in {
        FinancialRuntimeMode.V1.value,
        FinancialRuntimeMode.SHADOW.value,
    }:
        return normalized
    if normalized == "v2":
        raise FinancialRuntimeModeError(
            "FINANCIAL_RUNTIME_MODE=v2 is not available in TV2-06; "
            "use v1 or shadow",
        )
    raise FinancialRuntimeModeError(
        "FINANCIAL_RUNTIME_MODE must be one of: v1, shadow",
    )


class ShadowObservationSink(Protocol):
    """Observation-only sink; implementations must not mutate runtime state."""

    def record(self, observation: "V2ShadowObservation") -> None:
        ...


@dataclass(frozen=True)
class V2ShadowObservation:
    """Structured comparison of one primary V1 and shadow V2 attempt."""

    request_id: str
    user_id: str
    session_id: str
    original_query: str
    standalone_query: str
    primary_runtime_version: str = "V1"
    shadow_runtime_version: str = "V2"
    v1_status: str | None = None
    v1_release_status: str | None = None
    v2_status: str | None = None
    v2_release_status: str | None = None
    v1_answer: str | None = None
    v2_answer: str | None = None
    v1_evidence_ids: tuple[str, ...] = ()
    v2_evidence_ids: tuple[str, ...] = ()
    v1_citation_ids: tuple[str, ...] = ()
    v2_citation_ids: tuple[str, ...] = ()
    v1_calculation_ids: tuple[str, ...] = ()
    v2_calculation_ids: tuple[str, ...] = ()
    v1_reason_codes: tuple[str, ...] = ()
    v2_reason_codes: tuple[str, ...] = ()
    v2_route: str | None = None
    v1_latency_ms: float | None = None
    v2_latency_ms: float | None = None
    v2_retrieval_rounds: int = 0
    v2_repair_count: int = 0
    shadow_status: str = "COMPLETED"
    shadow_error_stage: str | None = None
    shadow_error_code: str | None = None
    comparison: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _tuple(value: Any) -> tuple[str, ...]:
        if value is None or isinstance(value, (str, bytes)):
            return ()
        return tuple(dict.fromkeys(str(item) for item in value if str(item).strip()))

    def __post_init__(self) -> None:
        for field_name in (
            "request_id",
            "user_id",
            "session_id",
            "original_query",
            "standalone_query",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in (
            "v1_evidence_ids",
            "v2_evidence_ids",
            "v1_citation_ids",
            "v2_citation_ids",
            "v1_calculation_ids",
            "v2_calculation_ids",
            "v1_reason_codes",
            "v2_reason_codes",
        ):
            object.__setattr__(self, field_name, self._tuple(getattr(self, field_name)))
        object.__setattr__(self, "comparison", copy.deepcopy(dict(self.comparison)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "original_query": self.original_query,
            "standalone_query": self.standalone_query,
            "primary_runtime_version": self.primary_runtime_version,
            "shadow_runtime_version": self.shadow_runtime_version,
            "v1_status": self.v1_status,
            "v1_release_status": self.v1_release_status,
            "v2_status": self.v2_status,
            "v2_release_status": self.v2_release_status,
            "v1_answer": self.v1_answer,
            "v2_answer": self.v2_answer,
            "v1_evidence_ids": list(self.v1_evidence_ids),
            "v2_evidence_ids": list(self.v2_evidence_ids),
            "v1_citation_ids": list(self.v1_citation_ids),
            "v2_citation_ids": list(self.v2_citation_ids),
            "v1_calculation_ids": list(self.v1_calculation_ids),
            "v2_calculation_ids": list(self.v2_calculation_ids),
            "v1_reason_codes": list(self.v1_reason_codes),
            "v2_reason_codes": list(self.v2_reason_codes),
            "v2_route": self.v2_route,
            "v1_latency_ms": self.v1_latency_ms,
            "v2_latency_ms": self.v2_latency_ms,
            "v2_retrieval_rounds": self.v2_retrieval_rounds,
            "v2_repair_count": self.v2_repair_count,
            "shadow_status": self.shadow_status,
            "shadow_error_stage": self.shadow_error_stage,
            "shadow_error_code": self.shadow_error_code,
            "comparison": copy.deepcopy(self.comparison),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


class InMemoryShadowObservationSink:
    """Bounded test/evaluation sink with no business side effects."""

    def __init__(self, *, max_records: int = 1000) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self.max_records = int(max_records)
        self._records: list[V2ShadowObservation] = []

    def record(self, observation: V2ShadowObservation) -> None:
        self._records.append(observation)
        if len(self._records) > self.max_records:
            del self._records[: len(self._records) - self.max_records]

    @property
    def observations(self) -> tuple[V2ShadowObservation, ...]:
        return tuple(self._records)


class LoggingShadowObservationSink:
    """Production-safe sink that logs only the structured observation."""

    def __init__(self, logger: Any) -> None:
        self.logger = logger

    def record(self, observation: V2ShadowObservation) -> None:
        self.logger.info("v2_shadow_observation %s", observation.to_json())


__all__ = [
    "FinancialRuntimeMode",
    "FinancialRuntimeModeError",
    "InMemoryShadowObservationSink",
    "LoggingShadowObservationSink",
    "ShadowObservationSink",
    "V2ShadowObservation",
    "resolve_financial_runtime_mode",
]
