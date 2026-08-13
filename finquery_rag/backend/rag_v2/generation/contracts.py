"""Typed contracts for the trusted generation runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ValidationSeverity(str, Enum):
    PASS = "PASS"
    SOFT_FAIL = "SOFT_FAIL"
    HARD_FAIL = "HARD_FAIL"


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class AnswerEnvelopeV1:
    query_id: str
    route: str
    answer_text: str
    citation_ids: tuple[str, ...]
    generator_provider: str
    generator_model: str
    attempt_index: int = 0
    generation_status: str = "complete"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.query_id, "query_id")
        _required_text(self.route, "route")
        if not isinstance(self.answer_text, str):
            raise ValueError("answer_text must be a string")
        if not isinstance(self.citation_ids, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.citation_ids
        ):
            raise ValueError("citation_ids must be a tuple of non-empty strings")
        _required_text(self.generator_provider, "generator_provider")
        _required_text(self.generator_model, "generator_model")
        if self.attempt_index not in (0, 1):
            raise ValueError("attempt_index must be 0 or 1")
        _required_text(self.generation_status, "generation_status")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, attempt_index: int = 0,
                  provider_id: str | None = None) -> "AnswerEnvelopeV1":
        if not isinstance(value, Mapping):
            raise ValueError("answer envelope must be an object")
        raw_ids = value.get("citation_ids", ())
        if not isinstance(raw_ids, (list, tuple)):
            raise ValueError("citation_ids must be an array")
        return cls(query_id=str(value.get("query_id", "")), route=str(value.get("route", "")),
                   answer_text=str(value.get("answer_text", "")),
                   citation_ids=tuple(str(item) for item in raw_ids),
                   generator_provider=str(value.get("generator_provider", provider_id or "")),
                   generator_model=str(value.get("generator_model", "")),
                   attempt_index=int(value.get("attempt_index", attempt_index)),
                   generation_status=str(value.get("generation_status", "complete")),
                   metadata=dict(value.get("metadata", {})) if isinstance(value.get("metadata", {}), Mapping) else {})

    def to_dict(self) -> dict[str, Any]:
        return {"query_id": self.query_id, "route": self.route, "answer_text": self.answer_text,
                "citation_ids": list(self.citation_ids), "generator_provider": self.generator_provider,
                "generator_model": self.generator_model, "attempt_index": self.attempt_index,
                "generation_status": self.generation_status, "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class GenerationInputV1:
    """Renderer seam; a future FinancialGenerationViewV1 can produce this object."""
    query_id: str
    route: str
    question: str
    packet: Mapping[str, Any]
    renderer_id: str = "generic_packet_renderer_v1"
    rendered_text: str | None = None
    view_sha256: str | None = None
    trusted_packet: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _required_text(self.query_id, "query_id")
        _required_text(self.route, "route")
        if not isinstance(self.packet, Mapping):
            raise ValueError("packet must be an object")
        if self.packet.get("query_id") != self.query_id:
            raise ValueError("packet query_id does not match generation input")
        if self.packet.get("validation_status") != "VERIFIED":
            raise ValueError("generation input requires a VERIFIED packet")
        if self.rendered_text is not None and not isinstance(self.rendered_text, str):
            raise ValueError("rendered_text must be a string when present")
        if self.trusted_packet is not None and not isinstance(self.trusted_packet, Mapping):
            raise ValueError("trusted_packet must be an object when present")


@dataclass(frozen=True)
class GenerationValidationFindingV1:
    code: str
    severity: ValidationSeverity
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity.value, "message": self.message}


@dataclass(frozen=True)
class GenerationValidationReportV1:
    status: ValidationSeverity
    findings: tuple[GenerationValidationFindingV1, ...] = ()
    checked_dimensions: tuple[str, ...] = ()

    @property
    def hard_fail(self) -> bool:
        return self.status is ValidationSeverity.HARD_FAIL

    @property
    def soft_fail(self) -> bool:
        return self.status is ValidationSeverity.SOFT_FAIL

    @property
    def passed(self) -> bool:
        return self.status is ValidationSeverity.PASS

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(item.code for item in self.findings if item.severity is not ValidationSeverity.PASS)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "findings": [item.to_dict() for item in self.findings],
                "failure_codes": list(self.failure_codes), "checked_dimensions": list(self.checked_dimensions)}


@dataclass(frozen=True)
class GenerationAttemptRecordV1:
    query_id: str
    attempt_index: int
    provider_id: str
    model_id: str
    answer_envelope: AnswerEnvelopeV1 | None
    validation_report: GenerationValidationReportV1 | None
    recovery_reason: str | None
    terminal_state: str
    latency_ms: float | None = None
    input_token_count: int | None = None
    output_token_count: int | None = None

    def __post_init__(self) -> None:
        _required_text(self.query_id, "query_id")
        if self.attempt_index not in (0, 1):
            raise ValueError("attempt_index must be 0 or 1")

    def to_dict(self) -> dict[str, Any]:
        return {"query_id": self.query_id, "attempt_index": self.attempt_index,
                "provider_id": self.provider_id, "model_id": self.model_id,
                "answer_envelope": self.answer_envelope.to_dict() if self.answer_envelope else None,
                "validation_report": self.validation_report.to_dict() if self.validation_report else None,
                "recovery_reason": self.recovery_reason, "terminal_state": self.terminal_state,
                "latency_ms": self.latency_ms, "input_token_count": self.input_token_count,
                "output_token_count": self.output_token_count}
