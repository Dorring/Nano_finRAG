"""P1.5-S2E: a relation is not something a model should restate.

`compare-002` computed the right answer and released the wrong one.  The
operands were Apple `-8077` and Visa `1926`, the ordering was `Visa > Apple`,
and the released answer was:

    "The verified calculation indicates that Apple's General and
     administrative was larger in FY2025 compared to V_fy2025."

The calculation was correct.  The *route* was `CALCULATION_WITH_EXPLANATION`,
which hands a pre-computed result to the Local Specialist to write prose around,
and the prose stated the opposite.  The validator had no structured claim to
check it against, so it passed.

`ranking` never showed the defect, because its questions carry no explanation
terms and it took the deterministic route -- `rank-002` and `rank-003` released
`Microsoft > Tesla > Apple` and `Microsoft > JPMorganChase > Visa > Tesla`,
both exactly right.

The rule is that an explanation may be added *around* a relation and never in
place of one.  These tests pin it at the policy, which is where the choice is
made.
"""

from __future__ import annotations

import pytest

from src.generation.generator_routing_policy import (
    GeneratorRoutingPolicy,
    GeneratorTarget,
    RouteName,
)


def _route(query: str, operation: str, *, extra_evidence: int = 0):
    policy = GeneratorRoutingPolicy()
    items = [
        {"candidate_key": f"C{index}", "value": "1", "metric": "Revenue"}
        for index in range(2 + extra_evidence)
    ]
    return policy.route(
        query,
        items,
        calculation_result={"operation": operation, "value": None, "status": "executed"},
        route_hint="CALCULATION",
    )


@pytest.mark.parametrize(
    "query",
    [
        "Compare Apple and Visa: which reported a larger General and administrative in FY2025?",
        "Rank the following companies by Research and development in FY2025 from highest to lowest: Apple, Microsoft, Tesla.",
    ],
    ids=["comparison-with-explain-terms", "ranking"],
)
def test_a_comparison_with_explanation_terms_still_goes_to_the_calculator(
    query: str,
) -> None:
    """The `compare` in the question is exactly what used to send it to the model."""

    decision = _route(query, "comparison")

    assert decision.target is GeneratorTarget.DETERMINISTIC_CALCULATOR
    assert decision.route_name is RouteName.CALCULATION_SIMPLE


def test_a_ranking_goes_to_the_calculator() -> None:
    decision = _route("Rank these companies by revenue", "ranking")

    assert decision.target is GeneratorTarget.DETERMINISTIC_CALCULATOR


def test_an_arithmetic_calculation_can_still_be_explained() -> None:
    """The rule is about relations, not about explanations.

    A quantity has an explanation worth writing around it, and the model
    restating "6" as "revenue rose by 6" cannot contradict the arithmetic the
    way a restated ordering can -- the quantity is checkable in the prose and
    the ordering was not.
    """

    decision = _route("Explain why revenue grew: what was the difference?", "difference")

    assert decision.target is GeneratorTarget.LOCAL_SPECIALIST
    assert decision.route_name is RouteName.CALCULATION_WITH_EXPLANATION


def test_a_relational_result_is_never_sent_to_the_specialist() -> None:
    """Even with every explanation term present, and plenty of evidence."""

    decision = _route(
        "Compare and explain the detail and trend of the difference: which is larger?",
        "comparison",
        extra_evidence=5,
    )

    assert decision.target is GeneratorTarget.DETERMINISTIC_CALCULATOR


def test_a_plain_calculation_is_unaffected() -> None:
    decision = _route("What was the difference?", "difference")

    assert decision.target is GeneratorTarget.DETERMINISTIC_CALCULATOR
    assert decision.route_name is RouteName.CALCULATION_SIMPLE
