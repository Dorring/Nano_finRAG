"""Generator Routing Policy (NF-V2-21).

Defines routing decisions among deterministic renderers, deterministic calculators,
and the Local Financial Specialist Generator based on query and evidence properties.

Route and target answer different questions, and the two are deliberately
separate fields:

    RouteName        the *shape* of the request -- how much evidence there is,
                     and what kind of thing is being asked.
    GeneratorTarget  which capability can actually produce the answer.

H2A-2C-1 separated them in the selection rule as well as in the type.  The rule
used to read ``len(evidence_items) > 1`` as MULTI and MULTI as "the Local
Specialist writes it", which is three questions collapsed into one number:
evidence cardinality, semantic arity, and execution capability.  Two independent
sources stating one figure is cardinality > 1 with arity 1, and the deterministic
renderer states it correctly -- asking a free-form generator to do so was asking
a capability to solve a problem it was never needed for.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from rag_v2.contracts.financial_semantics import quantity_identity


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
    def states_one_canonical_fact(evidence_items: list[dict[str, Any]]) -> bool:
        """Whether the admitted state reduces to one fact the renderer can state.

        This is the capability question, asked of the *state* rather than of a
        count: the deterministic structured renderer emits one metric, one
        period and one value, so a state it can express honestly is one where
        every admitted item names the same metric, the same period, and -- once
        the H2A-2B shared semantics have canonicalised it -- the same quantity.

        Metric and period and quantity are one question here rather than three
        because for a *renderer* they are one: any disagreement among them means
        a single rendering would have to drop or choose, and both are dishonest.
        ``1 million`` and ``1000 thousand`` are therefore the same fact; ``1
        million`` and ``1200 thousand`` are not, however alike they look.

        The canonicalisation is the shared one, not a second comparison: a
        routing rule that decided two sources disagreed because they spelled
        their scale differently would manufacture the conflict the binding layer
        exists to resolve.
        """

        if not evidence_items:
            return False
        keys = {
            (
                str(item.get("metric") or item.get("normalized_metric") or "")
                .strip()
                .casefold(),
                str(item.get("period") or item.get("normalized_period") or "")
                .strip()
                .casefold(),
                quantity_identity(
                    item.get("value")
                    if item.get("value") is not None
                    else item.get("parsed_numeric_value"),
                    scale=item.get("scale"),
                    unit=item.get("unit"),
                    currency=item.get("currency"),
                ),
            )
            for item in evidence_items
        }
        return len(keys) == 1

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
            # Route and target part company here.  More than one admitted item
            # is what MULTI names, and that is evidence cardinality -- the
            # sealed readiness gold labels it and it is not ours to reshape.
            # Whether a *generator* is needed is a different question, and the
            # answer is no when the items are independent sources of one
            # canonical fact: the structured renderer states that exactly, and
            # a free-form generator adds only the opportunity to paraphrase a
            # number.
            if GeneratorRoutingPolicy.states_one_canonical_fact(evidence_items):
                return GeneratorRouteDecision(
                    route_name=RouteName.MULTI,
                    target=GeneratorTarget.DETERMINISTIC_RENDERER,
                    reason="Multi-source single fact: the structured renderer states one canonical value with all supporting citations.",
                )
            return GeneratorRouteDecision(
                route_name=RouteName.MULTI,
                target=GeneratorTarget.LOCAL_SPECIALIST,
                reason="Multi-fact synthesis: Local Specialist combines distinct verified facts.",
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
