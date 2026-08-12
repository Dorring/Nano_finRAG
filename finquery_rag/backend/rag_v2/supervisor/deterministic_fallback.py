from __future__ import annotations

from collections.abc import Mapping

from rag_v2.contracts.errors import ContractError
from rag_v2.contracts.plan import SupervisorPlan

from .provider import SupervisorCallMetadata


class DeterministicFallbackProvider:
    """Explicit test-only provider; it never infers or calls a model."""

    provider_name = "deterministic_fallback"
    model_name = "deterministic-fallback"

    def __init__(self, plans: Mapping[str, SupervisorPlan]) -> None:
        self._plans = dict(plans)
        self.last_call: SupervisorCallMetadata | None = None

    def plan(self, question: str) -> SupervisorPlan:
        if question not in self._plans:
            raise ContractError("deterministic fallback has no plan for this question")
        self.last_call = SupervisorCallMetadata(
            provider=self.provider_name,
            model=self.model_name,
            latency_ms=0.0,
            raw_response=None,
        )
        return self._plans[question]
