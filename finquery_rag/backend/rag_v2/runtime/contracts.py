"""Typed top-level runtime request, response, trace, and terminal reasons."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4

from rag_v2.contracts.plan import SupervisorPlan


class TerminalReason(str, Enum):
    TR0_RELEASED_PRIMARY = "TR0_RELEASED_PRIMARY"
    TR1_RELEASED_FALLBACK = "TR1_RELEASED_FALLBACK"
    TR2_NO_TRUSTED_EVIDENCE = "TR2_NO_TRUSTED_EVIDENCE"
    TR3_PRIMARY_VALIDATION_FAIL_NO_FALLBACK = "TR3_PRIMARY_VALIDATION_FAIL_NO_FALLBACK"
    TR4_FALLBACK_VALIDATION_FAIL = "TR4_FALLBACK_VALIDATION_FAIL"
    TR5_PROVIDER_ERROR = "TR5_PROVIDER_ERROR"
    TR6_BUDGET_EXHAUSTED = "TR6_BUDGET_EXHAUSTED"
    TR7_NO_ANSWER = "TR7_NO_ANSWER"
    TR8_CALCULATION_NOT_READY = "TR8_CALCULATION_NOT_READY"
    TR9_MULTI_NOT_READY = "TR9_MULTI_NOT_READY"
    TR10_OTHER = "TR10_OTHER"


@dataclass(frozen=True)
class TrustedRAGQueryV2:
    query_id: str
    question: str
    supervisor_plan: SupervisorPlan | Mapping[str, Any] | None
    trusted_evidence_packet: Mapping[str, Any] | None = None
    no_answer: bool = False
    trace_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must be non-empty")
        if not isinstance(self.question, str):
            raise ValueError("question must be a string")
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise ValueError("trace_id must be non-empty")


@dataclass(frozen=True)
class RuntimeTraceV1:
    query_id: str
    route: str | None
    supervisor_plan_valid: bool
    trusted_evidence_available: bool
    evidence_source: str | None
    generation_attempts: tuple[Mapping[str, Any], ...]
    primary_provider: str | None
    fallback_provider: str | None
    validator_codes: tuple[tuple[str, ...], ...]
    fallback_triggered: bool
    released: bool
    terminal_reason: TerminalReason
    latencies_ms: tuple[float, ...] = ()
    token_counts: tuple[Mapping[str, int], ...] = ()
    trace_id: str = field(default_factory=lambda: uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id, "query_id": self.query_id, "route": self.route,
            "supervisor_plan_valid": self.supervisor_plan_valid,
            "trusted_evidence_available": self.trusted_evidence_available,
            "evidence_source": self.evidence_source,
            "generation_attempts": [dict(item) for item in self.generation_attempts],
            "primary_provider": self.primary_provider, "fallback_provider": self.fallback_provider,
            "validator_codes": [list(item) for item in self.validator_codes],
            "fallback_triggered": self.fallback_triggered, "released": self.released,
            "terminal_reason": self.terminal_reason.value, "latencies_ms": list(self.latencies_ms),
            "token_counts": [dict(item) for item in self.token_counts],
        }


@dataclass(frozen=True)
class TrustedRAGResponseV2:
    query_id: str
    route: str | None
    status: str
    answer_text: str | None
    citation_ids: tuple[str, ...]
    generation_provider: str | None
    generation_model: str | None
    used_fallback: bool
    attempt_count: int
    validation_status: str | None
    terminal_reason: TerminalReason
    trace_id: str
    trace: RuntimeTraceV1

    @property
    def released(self) -> bool:
        return self.status == "RELEASED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id, "route": self.route, "status": self.status,
            "answer_text": self.answer_text, "citation_ids": list(self.citation_ids),
            "generation_provider": self.generation_provider, "generation_model": self.generation_model,
            "used_fallback": self.used_fallback, "attempt_count": self.attempt_count,
            "validation_status": self.validation_status, "terminal_reason": self.terminal_reason.value,
            "trace_id": self.trace_id, "trace": self.trace.to_dict(),
        }
