"""Capability ports used by the TV2-02 bounded coordinator.

These protocols are intentionally small.  They let the bounded control loop
be tested with deterministic fakes while TV2-03 through TV2-05 attach the
real retrieval, binding, calculation, generation, and release components.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from rag_v2.adaptive import (
    AdaptiveRAGStateV1,
    EvidenceEvaluationV1,
    ReplanActionV1,
)


@runtime_checkable
class RetrievalCapability(Protocol):
    """Retrieve typed candidate packets for one structured recovery action."""

    def retrieve(
        self,
        action: ReplanActionV1,
        state: AdaptiveRAGStateV1,
    ) -> Iterable[Mapping[str, Any]]:
        """Return candidate packets; do not decide release."""
        ...


@runtime_checkable
class EvidenceEvaluationCapability(Protocol):
    """Evaluate candidate evidence against the current required slots."""

    def evaluate(self, state: AdaptiveRAGStateV1) -> EvidenceEvaluationV1:
        """Return a deterministic evidence decision and reason codes."""
        ...


@runtime_checkable
class CalculationCapability(Protocol):
    """Reserved TV2-04 port for deterministic calculation execution."""

    def calculate(self, state: AdaptiveRAGStateV1) -> Any:
        """Calculate only after the later binding gate admits operands."""
        ...


@runtime_checkable
class GenerationCapability(Protocol):
    """Reserved downstream generation port for explicit test wiring."""

    def generate(self, state: AdaptiveRAGStateV1) -> Any:
        """Return a candidate result; this port is not release authority."""
        ...


@runtime_checkable
class ReleaseValidationCapability(Protocol):
    """Reserved release validator port for explicit test wiring."""

    def validate(self, state: AdaptiveRAGStateV1, candidate: Any) -> Any:
        """Return a structured validation result, never a release decision."""
        ...


@dataclass(frozen=True)
class TrustedV2CapabilityPorts:
    """Injected capability set for one bounded coordinator instance."""

    retrieval: RetrievalCapability | None = None
    evidence_evaluator: EvidenceEvaluationCapability | None = None
    calculation: CalculationCapability | None = None
    generation: GenerationCapability | None = None
    release_validator: ReleaseValidationCapability | None = None


__all__ = [
    "CalculationCapability",
    "EvidenceEvaluationCapability",
    "GenerationCapability",
    "ReleaseValidationCapability",
    "RetrievalCapability",
    "TrustedV2CapabilityPorts",
]
