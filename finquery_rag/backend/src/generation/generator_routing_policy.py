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
from typing import Any, Mapping

from rag_v2.contracts.financial_semantics import quantity_identity, text_identity
from rag_v2.supervisor.semantic_alignment import (
    canonical_entity_id,
    canonical_metric_id,
    canonical_period_id,
    canonical_scope_id,
)
from src.domain.calculation import RELATIONAL_OPERATIONS


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
    def _semantic_slot_identity(item: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
        """The semantic slot an admitted item belongs to, and its quantity.

        Composed from the repository's existing canonical identities --
        `canonical_metric_id`, `canonical_period_id`, `canonical_entity_id`,
        `canonical_scope_id` -- rather than from a field tuple maintained here.
        Those are the same identities `_fact_matches_slot` uses to decide which
        slot a fact belongs to, so "one canonical fact" means the same thing in
        both places by construction instead of by agreement.

        Each falls back to normalised text when the vocabulary does not know the
        value, and the fallback direction is the conservative one: two spellings
        of one entity that the vocabulary cannot resolve read as different
        slots, which costs a deterministic answer and cannot produce a wrong
        one. Merging on a coincidence of text is the failure this prevents --
        `Company A Revenue FY2024 = 100` and `Company B Revenue FY2024 = 100`
        must not be one fact.
        """

        def _first(*keys: str) -> Any:
            for key in keys:
                value = item.get(key)
                if value is not None and str(value).strip():
                    return value
            return None

        metric = _first("metric", "normalized_metric", "raw_metric")
        period = _first("period", "normalized_period", "raw_period")
        entity = _first("entity", "company", "ticker")
        scope = _first("scope")

        return (
            canonical_metric_id(metric) or text_identity(metric),
            canonical_period_id(period) or text_identity(period),
            canonical_entity_id(entity) or text_identity(entity),
            canonical_scope_id(scope) or text_identity(scope),
            quantity_identity(
                item.get("value")
                if item.get("value") is not None
                else item.get("parsed_numeric_value"),
                scale=item.get("scale"),
                unit=item.get("unit"),
                currency=item.get("currency"),
            ),
        )

    @staticmethod
    def states_one_canonical_fact(evidence_items: list[dict[str, Any]]) -> bool:
        """Whether the admitted state reduces to one fact the renderer can state.

        This is the capability question, asked of the *state* rather than of a
        count: the deterministic structured renderer emits one metric, one
        period, one value and one citation set, so a state it can express
        honestly is one where every admitted item belongs to the same semantic
        slot and states the same quantity once H2A-2B's shared semantics have
        canonicalised it.

        Metric, period, entity, scope and quantity are one question here rather
        than five because for a *renderer* they are one: any disagreement among
        them means a single rendering would have to drop or choose, and both are
        dishonest. ``1 million`` and ``1000 thousand`` are therefore the same
        fact; ``1 million`` and ``1200 thousand`` are not, however alike they
        look. Entity and scope joined the identity in H2A-2C-2, because a
        quantity that matches across two issuers is two facts, not one.
        """

        if not evidence_items:
            return False
        keys = {
            GeneratorRoutingPolicy._semantic_slot_identity(item)
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
            # Check if the query requests qualitative explanation or comparison
            # along with the calculation.
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
            # H2A-2C-2: cardinality is not a reason to hand away numeric
            # authority.  This read `has_explanation_terms or
            # len(evidence_items) > 1`, which sent a calculation to a free-form
            # generator whenever it had more than one evidence item -- the same
            # defect family as MULTI -> Specialist, in a second branch.  Two
            # corroborating supports for *one* operand, or two operands, are
            # exactly what the deterministic calculator is for.
            #
            # The runtime already refuses this at a higher layer
            # (`trusted_v2_generation.py:_route` forces the deterministic
            # calculator for a CALCULATION plan), so the coupling was inert
            # there -- the correction belongs here regardless, because the
            # override is a second rule compensating for this one, and a policy
            # that is only correct when something above it disagrees is not
            # correct.
            # A relational result is its own answer, and must not be handed to a
            # free-form generator.
            #
            # "Visa > Apple" is not an approximation of something a model should
            # restate; it *is* the answer.  `compare-002` computed it correctly
            # -- operands Apple `-8077` and Visa `1926`, ordering `Visa > Apple`
            # -- and released "Apple's General and administrative was larger",
            # because the explanation branch below sent the calculation to the
            # Local Specialist.  The validator had nothing to check the prose
            # against, so it passed.
            #
            # An explanation may be added *around* a relation.  It may never be
            # stated in place of one.
            if str((calculation_result or {}).get("operation") or "") in {
                item.value for item in RELATIONAL_OPERATIONS
            }:
                return GeneratorRouteDecision(
                    route_name=RouteName.CALCULATION_SIMPLE,
                    target=GeneratorTarget.DETERMINISTIC_CALCULATOR,
                    reason=(
                        "Relational result: the ordering is the answer, and is "
                        "stated deterministically rather than restated."
                    ),
                    requires_c1=True,
                )
            if has_explanation_terms:
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
