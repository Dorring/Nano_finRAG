"""Contracts for the not-yet-wired Trusted Financial Runtime V2.

TV2-01 defines only the boundary between the shared runtime port and a
future V2 execution coordinator.  It deliberately does not construct or
invoke Supervisor, retrieval, Binder, Calculator, or TrustedRAGRuntimeV2.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .runtime_contract import (
    FinancialQueryRequest,
    ReleaseStatus,
)


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name)


def _copy_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return copy.deepcopy(dict(value))


def _normalize_ids(value: Iterable[Any] | None, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{field_name} must be an iterable of strings")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = _require_non_empty_string(item, field_name)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _normalize_citations(
    value: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError("citations must be an iterable of mappings")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("each citation must be a mapping")
        result.append(copy.deepcopy(dict(item)))
    return result


def _coerce_enum(value: Any, enum_type: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(str(member.value) for member in enum_type)
        raise ValueError(
            f"{field_name} must be one of: {allowed}",
        ) from exc


_RAW_CONTEXT_KEYS = frozenset(
    {
        "conversation_history",
        "raw_history",
        "raw_turns",
        "recent_turns",
        "messages",
        "memory_profile",
    },
)


def _without_raw_context(
    value: Mapping[str, Any] | None,
    field_name: str,
) -> dict[str, Any]:
    copied = _copy_mapping(value, field_name)
    return {
        key: item
        for key, item in copied.items()
        if str(key).strip().lower() not in _RAW_CONTEXT_KEYS
    }


class V2ExecutionStatus(str, Enum):
    """Stable terminal categories emitted by the future V2 coordinator."""

    READY_FOR_RELEASE = "READY_FOR_RELEASE"
    FAIL_CLOSED = "FAIL_CLOSED"
    EXECUTION_ERROR = "EXECUTION_ERROR"


@dataclass(frozen=True)
class V2ExecutionRequest:
    """Standalone-query input for a Trusted V2 execution coordinator.

    The V2 backend receives the already resolved financial question.  Raw
    conversation history is intentionally filtered from the metadata boundary;
    ConversationContextManager remains the owner of context resolution.
    """

    request_id: str
    user_id: str
    session_id: str
    original_query: str
    standalone_query: str
    conversation_resolved: bool = False
    request_metadata: dict[str, Any] = field(default_factory=dict)
    conversation_metadata: dict[str, Any] = field(default_factory=dict)
    runtime_budget: dict[str, Any] = field(default_factory=dict)
    trace_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _require_non_empty_string(self.request_id, "request_id"),
        )
        object.__setattr__(
            self,
            "user_id",
            _require_non_empty_string(self.user_id, "user_id"),
        )
        object.__setattr__(
            self,
            "session_id",
            _require_non_empty_string(self.session_id, "session_id"),
        )
        object.__setattr__(
            self,
            "original_query",
            _require_non_empty_string(self.original_query, "original_query"),
        )
        object.__setattr__(
            self,
            "standalone_query",
            _require_non_empty_string(self.standalone_query, "standalone_query"),
        )
        if not isinstance(self.conversation_resolved, bool):
            raise TypeError("conversation_resolved must be a bool")
        object.__setattr__(
            self,
            "request_metadata",
            _without_raw_context(self.request_metadata, "request_metadata"),
        )
        object.__setattr__(
            self,
            "conversation_metadata",
            _without_raw_context(
                self.conversation_metadata,
                "conversation_metadata",
            ),
        )
        object.__setattr__(
            self,
            "runtime_budget",
            _copy_mapping(self.runtime_budget, "runtime_budget"),
        )
        object.__setattr__(
            self,
            "trace_metadata",
            _copy_mapping(self.trace_metadata, "trace_metadata"),
        )

    @classmethod
    def from_financial_request(
        cls,
        request: FinancialQueryRequest,
    ) -> "V2ExecutionRequest":
        if not isinstance(request, FinancialQueryRequest):
            raise TypeError("request must be a FinancialQueryRequest")
        return cls(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            original_query=request.original_query,
            standalone_query=request.standalone_query,
            conversation_resolved=request.query_as_resolved,
            request_metadata=request.request_metadata,
            conversation_metadata=request.conversation_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "original_query": self.original_query,
            "standalone_query": self.standalone_query,
            "conversation_resolved": self.conversation_resolved,
            "request_metadata": copy.deepcopy(self.request_metadata),
            "conversation_metadata": copy.deepcopy(self.conversation_metadata),
            "runtime_budget": copy.deepcopy(self.runtime_budget),
            "trace_metadata": copy.deepcopy(self.trace_metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V2ExecutionRequest":
        if not isinstance(value, Mapping):
            raise TypeError("V2 execution request must be a mapping")
        return cls(
            request_id=value.get("request_id"),
            user_id=value.get("user_id"),
            session_id=value.get("session_id"),
            original_query=value.get("original_query"),
            standalone_query=value.get("standalone_query"),
            conversation_resolved=value.get("conversation_resolved", False),
            request_metadata=value.get("request_metadata"),
            conversation_metadata=value.get("conversation_metadata"),
            runtime_budget=value.get("runtime_budget"),
            trace_metadata=value.get("trace_metadata"),
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str) -> "V2ExecutionRequest":
        if not isinstance(value, str):
            raise TypeError("V2 execution request JSON must be a string")
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise TypeError("V2 execution request JSON must contain an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class V2ExecutionOutcome:
    """Terminal output of a future full V2 execution coordinator.

    A candidate answer may be present on a FAIL_CLOSED outcome.  Release is
    controlled exclusively by status and release_status, never by answer text.
    """

    status: V2ExecutionStatus
    answer: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)
    calculation_ids: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    release_status: ReleaseStatus = ReleaseStatus.NOT_RELEASED
    route: str | None = None
    validator_status: str | None = None
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    latency_metadata: dict[str, Any] = field(default_factory=dict)
    debug_metadata: dict[str, Any] = field(default_factory=dict)
    plan_id: str | None = None
    evidence_packet_id: str | None = None
    calculation_result_id: str | None = None

    def __post_init__(self) -> None:
        status = _coerce_enum(
            self.status,
            V2ExecutionStatus,
            "status",
        )
        release_status = _coerce_enum(
            self.release_status,
            ReleaseStatus,
            "release_status",
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "release_status", release_status)
        if self.answer is not None and not isinstance(self.answer, str):
            raise TypeError("answer must be a string or None")
        if status is V2ExecutionStatus.READY_FOR_RELEASE:
            if release_status is not ReleaseStatus.RELEASED:
                raise ValueError(
                    "READY_FOR_RELEASE requires release_status=RELEASED",
                )
            if self.answer is None or not self.answer.strip():
                raise ValueError(
                    "READY_FOR_RELEASE requires a non-empty answer",
                )
        elif release_status is not ReleaseStatus.NOT_RELEASED:
            raise ValueError(
                f"{status.value} requires release_status=NOT_RELEASED",
            )
        object.__setattr__(
            self,
            "citations",
            _normalize_citations(self.citations),
        )
        for field_name in (
            "evidence_ids",
            "citation_ids",
            "calculation_ids",
            "reason_codes",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_ids(getattr(self, field_name), field_name),
            )
        for field_name in (
            "runtime_metadata",
            "latency_metadata",
            "debug_metadata",
        ):
            object.__setattr__(
                self,
                field_name,
                _copy_mapping(getattr(self, field_name), field_name),
            )
        for field_name in (
            "route",
            "validator_status",
            "plan_id",
            "evidence_packet_id",
            "calculation_result_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_string(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "answer": self.answer,
            "citations": copy.deepcopy(self.citations),
            "evidence_ids": list(self.evidence_ids),
            "citation_ids": list(self.citation_ids),
            "calculation_ids": list(self.calculation_ids),
            "reason_codes": list(self.reason_codes),
            "release_status": self.release_status.value,
            "route": self.route,
            "validator_status": self.validator_status,
            "runtime_metadata": copy.deepcopy(self.runtime_metadata),
            "latency_metadata": copy.deepcopy(self.latency_metadata),
            "debug_metadata": copy.deepcopy(self.debug_metadata),
            "plan_id": self.plan_id,
            "evidence_packet_id": self.evidence_packet_id,
            "calculation_result_id": self.calculation_result_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V2ExecutionOutcome":
        if not isinstance(value, Mapping):
            raise TypeError("V2 execution outcome must be a mapping")
        return cls(
            status=value.get("status"),
            answer=value.get("answer"),
            citations=value.get("citations"),
            evidence_ids=value.get("evidence_ids"),
            citation_ids=value.get("citation_ids"),
            calculation_ids=value.get("calculation_ids"),
            reason_codes=value.get("reason_codes"),
            release_status=value.get(
                "release_status",
                ReleaseStatus.NOT_RELEASED,
            ),
            route=value.get("route"),
            validator_status=value.get("validator_status"),
            runtime_metadata=value.get("runtime_metadata"),
            latency_metadata=value.get("latency_metadata"),
            debug_metadata=value.get("debug_metadata"),
            plan_id=value.get("plan_id"),
            evidence_packet_id=value.get("evidence_packet_id"),
            calculation_result_id=value.get("calculation_result_id"),
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str) -> "V2ExecutionOutcome":
        if not isinstance(value, str):
            raise TypeError("V2 execution outcome JSON must be a string")
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise TypeError("V2 execution outcome JSON must contain an object")
        return cls.from_dict(decoded)


@runtime_checkable
class TrustedV2ExecutionCoordinator(Protocol):
    """Future full standalone-query-to-answer V2 execution boundary."""

    async def execute(
        self,
        request: V2ExecutionRequest,
    ) -> V2ExecutionOutcome:
        """Run bounded Supervisor/retrieval/binding/generation execution."""
        ...


__all__ = [
    "TrustedV2ExecutionCoordinator",
    "V2ExecutionOutcome",
    "V2ExecutionRequest",
    "V2ExecutionStatus",
]
