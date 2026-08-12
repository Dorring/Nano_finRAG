from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from rag_v2.contracts.plan import SupervisorPlan


class SupervisorProviderError(RuntimeError):
    """Raised when a provider cannot return one strictly structured plan."""


@dataclass(frozen=True)
class SupervisorCallMetadata:
    provider: str
    model: str
    latency_ms: float
    raw_response: str | None
    provider_role: str = "supervisor"
    model_role: str = "strong_general_llm"
    provider_response_success: bool | None = None
    structured_output_success: bool | None = None
    reasoning_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    parse_failure: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "provider_role": self.provider_role,
            "model_role": self.model_role,
            "provider_response_success": self.provider_response_success,
            "structured_output_success": self.structured_output_success,
            "reasoning_tokens": self.reasoning_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "raw_response": self.raw_response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "parse_failure": self.parse_failure,
            "error": self.error,
        }


class SupervisorProvider(Protocol):
    """Provider boundary for exactly one question-to-plan proposal."""

    provider_name: str
    model_name: str
    last_call: SupervisorCallMetadata | None

    def plan(self, question: str) -> SupervisorPlan:
        """Return one structured proposal; never execute a downstream tool."""
