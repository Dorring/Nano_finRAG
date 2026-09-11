from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_v2.contracts.plan import SupervisorPlan

from .plan_validator import validate_plan_v2_01
from .plan_normalizer import PlanNormalization, normalize_supervisor_plan
from .provider import SupervisorCallMetadata, SupervisorProvider, SupervisorProviderError


@dataclass(frozen=True)
class SupervisorRun:
    question: str
    plan: SupervisorPlan | None
    plan_valid: bool
    error: str | None
    metadata: SupervisorCallMetadata | None
    normalization: PlanNormalization | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "plan_valid": self.plan_valid,
            "error": self.error,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }
        if self.normalization is not None:
            payload["normalization"] = self.normalization.to_dict()
        return payload


class SupervisorService:
    """One-call Supervisor facade; it never retries and never executes tools."""

    def __init__(self, provider: SupervisorProvider, *, max_calls_per_question: int = 1) -> None:
        if max_calls_per_question != 1:
            raise ValueError("NF-V2-01 freezes max_supervisor_calls at 1")
        self.provider = provider
        self.max_calls_per_question = max_calls_per_question

    def plan(self, question: str) -> SupervisorRun:
        try:
            proposed = self.provider.plan(question)
        except SupervisorProviderError as exc:
            metadata = getattr(self.provider, "last_call", None)
            return SupervisorRun(question, None, False, str(exc), metadata)
        except Exception as exc:
            metadata = getattr(self.provider, "last_call", None)
            return SupervisorRun(question, None, False, f"{type(exc).__name__}: {exc}", metadata)
        normalized, normalization = normalize_supervisor_plan(question, proposed)
        try:
            validated = validate_plan_v2_01(normalized)
        except Exception as exc:
            metadata = getattr(self.provider, "last_call", None)
            return SupervisorRun(
                question,
                normalized,
                False,
                str(exc),
                metadata,
                normalization,
            )
        return SupervisorRun(
            question,
            validated,
            True,
            None,
            getattr(self.provider, "last_call", None),
            normalization,
        )
