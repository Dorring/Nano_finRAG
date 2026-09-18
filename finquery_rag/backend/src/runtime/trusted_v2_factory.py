"""Explicit TV2-05 factory for a complete, non-fallback V2 component graph.

The factory is intentionally not imported by QueryLifecycleService.  It is a
construction seam for TV2-06 shadow and later release decisions.
"""
from __future__ import annotations

from typing import Any

from rag_v2.adaptive import AdaptiveRAGBudgetV1
from rag_v2.supervisor import SupervisorService, UnknownSemanticPolicy

from .harness_runtime_mode import AgentRuntimeMode, coerce_agent_runtime_mode
from .trusted_v2_adapter import TrustedFinancialRuntimeV2
from .trusted_v2_capabilities import TrustedV2CapabilityPorts
from .trusted_v2_coordinator import BoundedTrustedV2Coordinator


class TrustedV2FactoryError(RuntimeError):
    """Raised when a complete V2 dependency graph cannot be constructed."""


_REQUIRED_PORTS = (
    ("retrieval", "R4 retrieval"),
    ("evidence_evaluator", "Semantic Evidence Binder"),
    ("calculation", "deterministic Calculator"),
    ("generation", "Generator routing"),
    ("release_validator", "Validator release gate"),
)


def build_trusted_v2_runtime(
    supervisor: SupervisorService,
    *,
    capabilities: TrustedV2CapabilityPorts,
    budget: AdaptiveRAGBudgetV1 | None = None,
    unknown_semantic_policy: UnknownSemanticPolicy | str = (
        UnknownSemanticPolicy.COMPATIBILITY
    ),
    runtime_mode: AgentRuntimeMode | str | None = None,
    alignment_override: Any | None = None,
) -> TrustedFinancialRuntimeV2:
    """Build the complete V2 runtime with explicit dependencies.

    Missing ports fail fast.  There is deliberately no V1 fallback and no
    production registration in this function.

    ``alignment_override`` is P1.2's seam and is ``None`` everywhere in
    production -- no environment variable reaches it and this function's own
    default is the production value.  It is threaded through here rather than
    set on the coordinator afterwards because a coordinator that had been
    configured by mutation would be indistinguishable from one that had not.
    """

    if not isinstance(supervisor, SupervisorService):
        raise TypeError("supervisor must be SupervisorService")
    if not isinstance(capabilities, TrustedV2CapabilityPorts):
        raise TypeError("capabilities must be TrustedV2CapabilityPorts")

    missing = [
        label
        for name, label in _REQUIRED_PORTS
        if getattr(capabilities, name) is None
    ]
    if missing:
        raise TrustedV2FactoryError(
            "complete V2 runtime requires: " + ", ".join(missing)
        )
    for name, label in (
        ("calculation", "deterministic Calculator"),
        ("generation", "Generator routing"),
        ("release_validator", "Validator release gate"),
    ):
        port: Any = getattr(capabilities, name)
        if not bool(getattr(port, "candidate_mode", False)):
            raise TrustedV2FactoryError(
                f"{label} is not a TV2 candidate/release implementation"
            )

    coordinator = BoundedTrustedV2Coordinator(
        supervisor,
        capabilities=capabilities,
        budget=budget,
        allow_test_release=False,
        unknown_semantic_policy=unknown_semantic_policy,
        runtime_mode=coerce_agent_runtime_mode(runtime_mode),
        alignment_override=alignment_override,
    )
    return TrustedFinancialRuntimeV2(coordinator)


__all__ = ["TrustedV2FactoryError", "build_trusted_v2_runtime"]
