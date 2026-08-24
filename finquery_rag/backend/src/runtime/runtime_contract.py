"""Stable boundary types for financial QA runtimes.

I1 defines this port without wiring it into the production HTTP or RAG
paths. The contract deliberately uses the current production identity
boundary (user_id, session_id) and keeps conversation resolution separate
from financial runtime execution.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class RuntimeStatus(str, Enum):
    """Outcome category returned by a financial runtime."""

    ANSWER = "ANSWER"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    FAIL_CLOSED = "FAIL_CLOSED"
    ERROR = "ERROR"


class ReleaseStatus(str, Enum):
    """Whether a runtime result crossed the trusted release boundary."""

    RELEASED = "RELEASED"
    NOT_RELEASED = "NOT_RELEASED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RuntimeVersion(str, Enum):
    """Implementation version, independent from router deployment mode."""

    V1 = "V1"
    V2 = "V2"


class RuntimeRouterMode(str, Enum):
    """How a runtime implementation is being invoked by a future router."""

    ACTIVE = "ACTIVE"
    SHADOW = "SHADOW"
    CANARY = "CANARY"


# Short alias for callers that prefer the simpler name. The longer enum
# remains canonical so it cannot be confused with RuntimeVersion.
RouterMode = RuntimeRouterMode


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


def _normalize_string_list(
    value: Iterable[Any] | None,
    field_name: str,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{field_name} must be an iterable of strings")
    normalized: list[str] = []
    for item in value:
        normalized.append(_require_non_empty_string(item, field_name))
    return normalized


def _normalize_citations(
    value: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError("citations must be an iterable of mappings")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("each citation must be a mapping")
        normalized.append(copy.deepcopy(dict(item)))
    return normalized


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


@dataclass(frozen=True)
class ClarificationPayload:
    """Structured control response for an unresolved conversation request."""

    question: str
    reason_codes: list[str] = field(default_factory=list)
    options: list[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "question",
            _require_non_empty_string(self.question, "clarification.question"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _normalize_string_list(
                self.reason_codes,
                "clarification.reason_codes",
            ),
        )
        if self.options is None:
            normalized_options = None
        else:
            normalized_options = _normalize_string_list(
                self.options,
                "clarification.options",
            )
        object.__setattr__(self, "options", normalized_options)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "reason_codes": list(self.reason_codes),
            "options": None if self.options is None else list(self.options),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClarificationPayload":
        if not isinstance(value, Mapping):
            raise TypeError("clarification must be a mapping")
        return cls(
            question=value.get("question"),
            reason_codes=value.get("reason_codes", []),
            options=value.get("options"),
        )


@dataclass(frozen=True)
class RuntimeMetadata:
    """Non-semantic implementation metadata attached to a runtime result."""

    implementation: str | None = None
    config_version: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "implementation",
            _optional_string(
                self.implementation,
                "runtime_metadata.implementation",
            ),
        )
        object.__setattr__(
            self,
            "config_version",
            _optional_string(
                self.config_version,
                "runtime_metadata.config_version",
            ),
        )
        object.__setattr__(
            self,
            "attributes",
            _copy_mapping(self.attributes, "runtime_metadata.attributes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "config_version": self.config_version,
            "attributes": copy.deepcopy(self.attributes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeMetadata":
        if not isinstance(value, Mapping):
            raise TypeError("runtime_metadata must be a mapping")
        return cls(
            implementation=value.get("implementation"),
            config_version=value.get("config_version"),
            attributes=value.get("attributes"),
        )


@dataclass(frozen=True)
class FinancialQueryRequest:
    """Input port shared by future V1 and V2 financial runtimes.

    original_query is the immutable user wording. standalone_query is the
    query that a financial runtime should execute; it defaults to the
    original wording until a conversation resolver supplies a resolved query.
    query_as_resolved is an explicit future rewrite gate and has no effect on
    the current production path during I1.
    """

    request_id: str
    user_id: str
    session_id: str
    original_query: str
    standalone_query: str | None = None
    query_as_resolved: bool = False
    conversation_metadata: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)

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
        original_query = _require_non_empty_string(
            self.original_query,
            "original_query",
        )
        object.__setattr__(self, "original_query", original_query)
        standalone_query = (
            original_query
            if self.standalone_query is None
            else _require_non_empty_string(
                self.standalone_query,
                "standalone_query",
            )
        )
        object.__setattr__(self, "standalone_query", standalone_query)
        if not isinstance(self.query_as_resolved, bool):
            raise TypeError("query_as_resolved must be a bool")
        object.__setattr__(
            self,
            "conversation_metadata",
            _copy_mapping(self.conversation_metadata, "conversation_metadata"),
        )
        object.__setattr__(
            self,
            "request_metadata",
            _copy_mapping(self.request_metadata, "request_metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "original_query": self.original_query,
            "standalone_query": self.standalone_query,
            "query_as_resolved": self.query_as_resolved,
            "conversation_metadata": copy.deepcopy(self.conversation_metadata),
            "request_metadata": copy.deepcopy(self.request_metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FinancialQueryRequest":
        if not isinstance(value, Mapping):
            raise TypeError("request must be a mapping")
        return cls(
            request_id=value.get("request_id"),
            user_id=value.get("user_id"),
            session_id=value.get("session_id"),
            original_query=value.get("original_query"),
            standalone_query=value.get("standalone_query"),
            query_as_resolved=value.get("query_as_resolved", False),
            conversation_metadata=value.get("conversation_metadata"),
            request_metadata=value.get("request_metadata"),
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str) -> "FinancialQueryRequest":
        if not isinstance(value, str):
            raise TypeError("request JSON must be a string")
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise TypeError("request JSON must contain an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class FinancialQueryResult:
    """Output port for a financial runtime.

    Provenance fields are intentionally optional and default to empty lists.
    I1 does not derive them from answer text; a later runtime adapter must
    populate them only from structured internal provenance.
    """

    status: RuntimeStatus
    answer: str | None = None
    clarification: ClarificationPayload | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)
    calculation_ids: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    runtime_version: RuntimeVersion = RuntimeVersion.V1
    router_mode: RuntimeRouterMode = RuntimeRouterMode.ACTIVE
    release_status: ReleaseStatus = ReleaseStatus.NOT_APPLICABLE
    latency_metadata: dict[str, Any] = field(default_factory=dict)
    debug_metadata: dict[str, Any] = field(default_factory=dict)
    runtime_metadata: RuntimeMetadata | None = None

    def __post_init__(self) -> None:
        status = _coerce_enum(self.status, RuntimeStatus, "status")
        runtime_version = _coerce_enum(
            self.runtime_version,
            RuntimeVersion,
            "runtime_version",
        )
        router_mode = _coerce_enum(
            self.router_mode,
            RuntimeRouterMode,
            "router_mode",
        )
        release_status = _coerce_enum(
            self.release_status,
            ReleaseStatus,
            "release_status",
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "runtime_version", runtime_version)
        object.__setattr__(self, "router_mode", router_mode)
        object.__setattr__(self, "release_status", release_status)

        if self.answer is not None and not isinstance(self.answer, str):
            raise TypeError("answer must be a string or None")
        if status is RuntimeStatus.CLARIFICATION_REQUIRED:
            if self.clarification is None:
                raise ValueError(
                    "clarification is required when status is CLARIFICATION_REQUIRED",
                )
            if self.answer is not None:
                raise ValueError(
                    "answer must be None when status is CLARIFICATION_REQUIRED",
                )
        elif self.clarification is not None:
            raise ValueError(
                "clarification is only valid when status is CLARIFICATION_REQUIRED",
            )

        clarification = self.clarification
        if isinstance(clarification, Mapping):
            clarification = ClarificationPayload.from_dict(clarification)
        elif clarification is not None and not isinstance(
            clarification,
            ClarificationPayload,
        ):
            raise TypeError("clarification must be ClarificationPayload or None")
        object.__setattr__(self, "clarification", clarification)

        object.__setattr__(self, "citations", _normalize_citations(self.citations))
        object.__setattr__(
            self,
            "evidence_ids",
            _normalize_string_list(self.evidence_ids, "evidence_ids"),
        )
        object.__setattr__(
            self,
            "citation_ids",
            _normalize_string_list(self.citation_ids, "citation_ids"),
        )
        object.__setattr__(
            self,
            "calculation_ids",
            _normalize_string_list(self.calculation_ids, "calculation_ids"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _normalize_string_list(self.reason_codes, "reason_codes"),
        )
        object.__setattr__(
            self,
            "latency_metadata",
            _copy_mapping(self.latency_metadata, "latency_metadata"),
        )
        object.__setattr__(
            self,
            "debug_metadata",
            _copy_mapping(self.debug_metadata, "debug_metadata"),
        )
        if self.runtime_metadata is not None and isinstance(
            self.runtime_metadata,
            Mapping,
        ):
            object.__setattr__(
                self,
                "runtime_metadata",
                RuntimeMetadata.from_dict(self.runtime_metadata),
            )
        elif self.runtime_metadata is not None and not isinstance(
            self.runtime_metadata,
            RuntimeMetadata,
        ):
            raise TypeError("runtime_metadata must be RuntimeMetadata or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "answer": self.answer,
            "clarification": (
                None if self.clarification is None else self.clarification.to_dict()
            ),
            "citations": copy.deepcopy(self.citations),
            "evidence_ids": list(self.evidence_ids),
            "citation_ids": list(self.citation_ids),
            "calculation_ids": list(self.calculation_ids),
            "reason_codes": list(self.reason_codes),
            "runtime_version": self.runtime_version.value,
            "router_mode": self.router_mode.value,
            "release_status": self.release_status.value,
            "latency_metadata": copy.deepcopy(self.latency_metadata),
            "debug_metadata": copy.deepcopy(self.debug_metadata),
            "runtime_metadata": (
                None
                if self.runtime_metadata is None
                else self.runtime_metadata.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FinancialQueryResult":
        if not isinstance(value, Mapping):
            raise TypeError("result must be a mapping")
        return cls(
            status=value.get("status"),
            answer=value.get("answer"),
            clarification=value.get("clarification"),
            citations=value.get("citations"),
            evidence_ids=value.get("evidence_ids"),
            citation_ids=value.get("citation_ids"),
            calculation_ids=value.get("calculation_ids"),
            reason_codes=value.get("reason_codes"),
            runtime_version=value.get("runtime_version", RuntimeVersion.V1),
            router_mode=value.get("router_mode", RuntimeRouterMode.ACTIVE),
            release_status=value.get(
                "release_status",
                ReleaseStatus.NOT_APPLICABLE,
            ),
            latency_metadata=value.get("latency_metadata"),
            debug_metadata=value.get("debug_metadata"),
            runtime_metadata=value.get("runtime_metadata"),
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str) -> "FinancialQueryResult":
        if not isinstance(value, str):
            raise TypeError("result JSON must be a string")
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise TypeError("result JSON must contain an object")
        return cls.from_dict(decoded)


@runtime_checkable
class FinancialQARuntime(Protocol):
    """Port implemented by future V1/V2 financial runtimes."""

    async def execute(
        self,
        request: FinancialQueryRequest,
    ) -> FinancialQueryResult:
        """Execute one request under the runtime's frozen contract."""
        ...
