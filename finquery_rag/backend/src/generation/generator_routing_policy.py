"""Generator Routing Policy (NF-V2-21).

Defines routing decisions among deterministic renderers, deterministic calculators,
and the Local Financial Specialist Generator based on query and evidence properties.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class GeneratorTarget(str, Enum):
    DETERMINISTIC_RENDERER = "DETERMINISTIC_RENDERER"
    DETERMINISTIC_CALCULATOR = "DETERMINISTIC_CALCULATOR"
    LOCAL_SPECIALIST = "LOCAL_SPECIALIST"
    FAIL_CLOSED_PRE_GEN = "FAIL_CLOSED_PRE_GEN"


class RouteName(str, Enum):
    STRUCTURED_SINGLE = "STRUCTURED_SINGLE"
    CALCULATION_SIMPLE = "CALCULATION_SIMPLE"
    QUALITATIVE = "QUALITATIVE"
    MULTI = "MULTI"
    TEMPORAL_SYNTHESIS = "TEMPORAL_SYNTHESIS"
    CALCULATION_WITH_EXPLANATION = "CALCULATION_WITH_EXPLANATION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class GeneratorRouteDecision:
    route_name: RouteName
    target: GeneratorTarget
    reason: str
    requires_c1: bool = False
    fail_closed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_name": self.route_name.value,
            "target": self.target.value,
            "reason": self.reason,
            "requires_c1": self.requires_c1,
            "fail_closed": self.fail_closed,
        }


class GeneratorRoutingPolicy:
    """Evaluates query and evidence conditions to select generator target."""

    @staticmethod
    def route(
        query: str,
        evidence_items: list[dict[str, Any]],
        calculation_result: dict[str, Any] | None = None,
        route_hint: str | None = None,
    ) -> GeneratorRouteDecision:
        """Route request according to runtime safety and capability policy."""
        # 1. Check if evidence is insufficient / missing
        if not evidence_items and not calculation_result:
            return GeneratorRouteDecision(
                route_name=RouteName.INSUFFICIENT_EVIDENCE,
                target=GeneratorTarget.FAIL_CLOSED_PRE_GEN,
                reason="Pre-generation fail-closed: No verified evidence provided to Binder.",
                fail_closed=True,
            )

        # 2. Check route hint or infer from evidence structure
        norm_hint = (route_hint or "").upper()

        if "CALCULATION" in norm_hint or calculation_result is not None:
            # Check if query requests qualitative explanation or comparison along with calculation
            has_explanation_terms = any(
                term in query.lower()
                for term in [
                    "why",
                    "explain",
                    "describe",
                    "compare",
                    "difference",
                    "growth",
                    "context",
                    "detail",
                    "trend",
                ]
            )
            if has_explanation_terms or len(evidence_items) > 1:
                return GeneratorRouteDecision(
                    route_name=RouteName.CALCULATION_WITH_EXPLANATION,
                    target=GeneratorTarget.LOCAL_SPECIALIST,
                    reason="Calculation with synthesis: Local Specialist consumes pre-computed C1 and cites evidence.",
                    requires_c1=True,
                )
            return GeneratorRouteDecision(
                route_name=RouteName.CALCULATION_SIMPLE,
                target=GeneratorTarget.DETERMINISTIC_CALCULATOR,
                reason="Simple atomic calculation: Deterministic calculator authority.",
                requires_c1=True,
            )

        if "TEMPORAL" in norm_hint or any(
            t in query.lower() for t in ["year-over-year", "yoy", "consecutive", "historical", "prior year", "trend"]
        ):
            return GeneratorRouteDecision(
                route_name=RouteName.TEMPORAL_SYNTHESIS,
                target=GeneratorTarget.LOCAL_SPECIALIST,
                reason="Temporal/multi-period synthesis: Local Specialist synthesizes timeline.",
            )

        if "MULTI" in norm_hint or len(evidence_items) > 1:
            return GeneratorRouteDecision(
                route_name=RouteName.MULTI,
                target=GeneratorTarget.LOCAL_SPECIALIST,
                reason="Multi-evidence synthesis: Local Specialist combines multiple verified facts.",
            )

        if "QUALITATIVE" in norm_hint or any(
            q in query.lower() for q in ["risk", "strategy", "policy", "note", "accounting", "discuss", "outlook"]
        ):
            return GeneratorRouteDecision(
                route_name=RouteName.QUALITATIVE,
                target=GeneratorTarget.LOCAL_SPECIALIST,
                reason="Qualitative grounded QA: Local Specialist generates verified text answer.",
            )

        # Single evidence structured lookup
        if len(evidence_items) == 1:
            return GeneratorRouteDecision(
                route_name=RouteName.STRUCTURED_SINGLE,
                target=GeneratorTarget.DETERMINISTIC_RENDERER,
                reason="Single factual table/document lookup: Deterministic structured renderer.",
            )

        # Default fallback to Local Specialist
        return GeneratorRouteDecision(
            route_name=RouteName.QUALITATIVE,
            target=GeneratorTarget.LOCAL_SPECIALIST,
            reason="Grounded context generation: Local Specialist.",
        )
