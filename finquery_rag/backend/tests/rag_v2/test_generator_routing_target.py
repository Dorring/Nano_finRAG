"""H2A-2C-1: route classification and execution target are separate decisions.

The routing policy already carries two fields -- `RouteName` and
`GeneratorTarget` -- and `decision.target` is consumed at exactly one place
(`src/runtime/trusted_v2_generation.py:371-387`).  So the types were never the
problem.  The *selection rule* was: `len(evidence_items) > 1` was read as
"MULTI", and MULTI was read as "the Local Specialist must write the answer".

Those are three different questions and the rule answered all three with one
number:

    evidence/support cardinality   how many items are admitted
    semantic arity                 how many distinct facts those items state
    execution capability           who can actually produce the answer

`cross-source-001` is one canonical Revenue value with two independent sources.
There is nothing to synthesise: the deterministic structured renderer already
expresses exactly that, and it expresses it correctly.  Sending it to a
free-form generator asked a capability to solve a problem it was never needed
for -- and, in this repository's harness, that capability is a recording stub,
which is how the defect became visible as an unanswerable readiness case rather
than as a subtle quality problem.

Expected values below are literals authored here.  None is computed by calling
the selector being tested.
"""

from __future__ import annotations

from typing import Any

from src.generation.generator_routing_policy import (
    GeneratorRoutingPolicy,
    GeneratorTarget,
    RouteName,
)


def _fact(
    metric: str = "Revenue",
    period: str = "FY2024",
    value: str = "100",
    *,
    unit: str | None = "USD",
    currency: str | None = "USD",
    scale: str | None = None,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "period": period,
        "value": value,
        "unit": unit,
        "currency": currency,
        "scale": scale,
    }


def _route(query: str, items: list[dict[str, Any]], **kw: Any):
    return GeneratorRoutingPolicy.route(query, items, **kw)


# --- the case the phase exists for -------------------------------------------


def test_one_canonical_fact_with_two_sources_routes_multi_but_renders() -> None:
    """Two independent sources, one quantity, no synthesis required.

    1 million and 1000 thousand are the same figure under the H2A-2B shared
    semantics, so these are corroborating supports of one fact rather than two
    facts.  The route stays MULTI -- that is evidence cardinality, and the
    sealed readiness gold names it -- while the target is the deterministic
    renderer, because one canonical fact is precisely what it renders.
    """

    decision = _route(
        "Compare the annual report and filing revenue figures.",
        [_fact(value="1", scale="million"), _fact(value="1000", scale="thousand")],
    )

    assert decision.route_name is RouteName.MULTI
    assert decision.target is GeneratorTarget.DETERMINISTIC_RENDERER


def test_the_route_name_does_not_change_with_the_target() -> None:
    """The gold labels the route, so its name is not ours to reshape.

    If the route name were renamed to match the target this phase would be
    editing the instrument to agree with itself.
    """

    items = [_fact(value="1", scale="million"), _fact(value="1000", scale="thousand")]
    decision = _route("What was Apple's FY2024 revenue?", items)

    assert decision.route_name.value == "MULTI"
    assert decision.target is not GeneratorTarget.LOCAL_SPECIALIST


# --- cardinality alone must not decide ---------------------------------------


def test_two_distinct_facts_still_require_synthesis() -> None:
    """The negative that keeps the rule from being "many items -> renderer".

    Two different metrics cannot be expressed by a renderer that emits one
    metric and one value; collapsing them would answer half the question and
    look complete.  This is the shape the Specialist target exists for.
    """

    decision = _route(
        "Compare revenue and operating income.",
        [_fact(metric="Revenue", value="100"), _fact(metric="operating_income", value="50")],
    )

    assert decision.route_name is RouteName.MULTI
    assert decision.target is GeneratorTarget.LOCAL_SPECIALIST


def test_two_different_values_for_one_metric_still_require_synthesis() -> None:
    """Same metric and period, different quantities: not one canonical fact.

    The binding layer refuses this as a conflict, so it should never reach
    generation.  If it does, the renderer must not be told the state is
    renderable -- picking one of two disagreeing values and rendering it
    confidently is the failure mode this phase exists to prevent.
    """

    decision = _route(
        "What was Apple's FY2024 revenue?",
        [_fact(value="1", scale="million"), _fact(value="1200", scale="thousand")],
    )

    assert decision.target is not GeneratorTarget.DETERMINISTIC_RENDERER


def test_a_different_period_routes_away_from_the_single_fact_renderer() -> None:
    """Two periods are two facts, whatever the metric.

    The renderer emits one period and one value; a two-period state rendered by
    it would silently drop half the comparison.
    """

    decision = _route(
        "Compare FY2023 and FY2024 revenue.",
        [_fact(period="FY2024", value="100"), _fact(period="FY2023", value="90")],
    )

    assert decision.target is not GeneratorTarget.DETERMINISTIC_RENDERER


# --- what the specialist is genuinely for ------------------------------------


def test_a_qualitative_question_keeps_a_free_form_target() -> None:
    """The one capability the deterministic path does not have: prose.

    The renderer emits `metric (period): value unit [citation]`.  A question
    asking for risk, strategy or outlook cannot be answered in that shape, and
    no amount of structured state changes it -- so this is a genuine capability
    boundary rather than a cardinality rule, and it is the case that justifies
    keeping the Local Specialist wired at all.
    """

    decision = _route(
        "What does management say about the outlook for margins?",
        [_fact(value="100")],
    )

    assert decision.route_name is RouteName.QUALITATIVE
    assert decision.target is GeneratorTarget.LOCAL_SPECIALIST


# --- no regression on the shapes that were already right ---------------------


def test_a_single_support_still_routes_structured_single() -> None:
    decision = _route("What was Apple's FY2024 revenue?", [_fact(value="100")])

    assert decision.route_name is RouteName.STRUCTURED_SINGLE
    assert decision.target is GeneratorTarget.DETERMINISTIC_RENDERER


def test_no_evidence_still_fails_closed() -> None:
    decision = _route("What was Apple's FY2024 revenue?", [])

    assert decision.route_name is RouteName.INSUFFICIENT_EVIDENCE
    assert decision.target is GeneratorTarget.FAIL_CLOSED_PRE_GEN
    assert decision.fail_closed is True
