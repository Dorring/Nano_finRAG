"""Shared fixtures for the NF-V3 H1 harness tests.

Both ``test_calculation_phase.py`` and ``test_closed_loop.py`` drive the same
coordinator over the same synthetic fixtures, differing only in which phases they
assert on.  The construction helpers live here so the two suites cannot drift
apart -- a change to how a test coordinator is wired must move both, not one.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rag_v2.adaptive import (
    AdaptiveRAGBudgetV1,
    AdaptiveRAGStateV1,
    BoundedAdaptiveRAGV1,
    ReasonCode,
    ReplanActionV1,
    ToolCapability,
)
from rag_v2.contracts import Intent
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from src.runtime import (
    TrustedReleaseValidationCapability,
    TrustedV2CapabilityPorts,
    TrustedV2GenerationCapability,
)
from src.runtime.harness_runtime_mode import AgentRuntimeMode
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator
from tests.test_trusted_v2_r4_binder import (
    _fact,
    _plan,
    _real_capabilities,
    _request,
    _slot,
)

__all__ = [
    "BUDGET",
    "CALCULATION_FACTS",
    "REVENUE_FACTS",
    "calculation_loop_state",
    "calculation_plan",
    "execute",
    "loop_state",
    "packet",
    "run_loop",
    "transitions",
]

BUDGET = AdaptiveRAGBudgetV1(
    max_replan_rounds=3, max_total_tool_calls=4, max_same_tool_retry=3
)

CALCULATION_FACTS = {
    "CURRENT": _fact("CURRENT", period="FY2024", slots=("current",), value="391"),
    "PRIOR": _fact("PRIOR", period="FY2023", slots=("prior",), value="383"),
}

REVENUE_FACTS = {"E1": _fact("E1", value="100")}


def calculation_plan() -> Any:
    return _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("prior", period="FY2023", role="prior"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )


def execute(
    mode: AgentRuntimeMode,
    query: str,
    plan: Any,
    facts: dict[str, Any],
    keys: list[list[str]],
    *,
    binder_provider: Any = None,
    calculation: Any = None,
    request_id: str = "h1-harness",
) -> Any:
    """Run one coordinator request in the given runtime mode."""

    retrieval, binder, _, _, _ = _real_capabilities(keys, facts, binder_provider)
    coordinator = BoundedTrustedV2Coordinator(
        SupervisorService(DeterministicFallbackProvider({query: plan})),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=binder,
            calculation=calculation,
            generation=TrustedV2GenerationCapability(),
            release_validator=TrustedReleaseValidationCapability(),
        ),
        budget=BUDGET,
        runtime_mode=mode,
    )
    return asyncio.run(coordinator.execute(_request(query, request_id)))


def transitions(outcome: Any) -> list[str]:
    return [item["to"] for item in outcome.debug_metadata["trace"]["transitions"]]


# --- harness-level fixtures (no coordinator) --------------------------------


def loop_state() -> AdaptiveRAGStateV1:
    return AdaptiveRAGStateV1.new(
        "q1",
        "What was revenue?",
        required_slots=[{"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"}],
    )


def calculation_loop_state() -> AdaptiveRAGStateV1:
    state = AdaptiveRAGStateV1.new(
        "q1",
        "Compare revenue across years",
        required_slots=[{"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"}],
        calculation_requirements={"operation": "growth_rate", "operand_slots": ["revenue"]},
    )
    # The evidence evaluator admits evidence before the loop may calculate; a
    # bare state has no admitted evidence, so the CALCULATE phase is not entered.
    state.bound_evidence_ids = ["e1"]
    return state


def packet() -> dict[str, Any]:
    return {
        "evidence_id": "e1",
        "metric": "Revenue",
        "value": "120",
        "period": "FY2024",
        "entity": "Acme",
        "scope": "consolidated",
        "source": "10-K",
        "document_id": "e1",
    }


def run_loop(state: AdaptiveRAGStateV1 | None = None, **kwargs: Any) -> Any:
    """Run the harness directly, with one tool that always returns ``packet()``."""

    state = state if state is not None else loop_state()
    return BoundedAdaptiveRAGV1().run(
        state,
        {ToolCapability.SEMANTIC_RETRIEVAL: lambda query, current: [packet()]},
        initial_action=ReplanActionV1(
            ToolCapability.SEMANTIC_RETRIEVAL,
            state.normalized_query,
            ReasonCode.MISSING_SLOT,
        ),
        **kwargs,
    )
