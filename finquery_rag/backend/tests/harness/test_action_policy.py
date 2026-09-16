"""NF-V3 H1-D2: the action policy decides permission, not action.

``BoundedReplannerV1`` proposes the next capability from the evaluator's reason
codes.  ``AdaptiveActionPolicyV1`` decides whether executing it is still
allowed.  The two were previously entangled inside ``BoundedAdaptiveRAGV1.run``;
these tests pin the extracted policy on its own.
"""

from __future__ import annotations

import pytest

from rag_v2.adaptive import (
    AdaptiveActionPolicyV1,
    AdaptiveRAGBudgetV1,
    AdaptiveRAGStateV1,
    BoundedAdaptiveRAGV1,
    ReasonCode,
    ToolCapability,
)


def _state(*, tool_calls: int = 0, replan_rounds: int = 0, retries: dict | None = None) -> AdaptiveRAGStateV1:
    state = AdaptiveRAGStateV1.new("q1", "What was revenue?")
    state.tool_calls = tool_calls
    state.replan_rounds = replan_rounds
    state.same_tool_retries = dict(retries or {})
    return state


def test_tool_call_is_permitted_while_budget_remains() -> None:
    policy = AdaptiveActionPolicyV1(AdaptiveRAGBudgetV1(max_total_tool_calls=3))
    assert policy.check_tool_call(_state(tool_calls=0)) is None
    assert policy.check_tool_call(_state(tool_calls=2)) is None


def test_tool_call_is_denied_when_budget_is_spent() -> None:
    policy = AdaptiveActionPolicyV1(AdaptiveRAGBudgetV1(max_total_tool_calls=3))
    assert policy.check_tool_call(_state(tool_calls=3)) is ReasonCode.BUDGET_EXHAUSTED


def test_tool_retry_allows_exactly_the_configured_retries() -> None:
    policy = AdaptiveActionPolicyV1(AdaptiveRAGBudgetV1(max_same_tool_retry=1))
    key = ToolCapability.SEMANTIC_RETRIEVAL

    assert policy.check_tool_retry(_state(retries={key.value: 0}), key) is None
    assert policy.check_tool_retry(_state(retries={key.value: 1}), key) is None
    assert policy.check_tool_retry(_state(retries={key.value: 2}), key) is ReasonCode.BUDGET_EXHAUSTED


def test_tool_retry_accepts_enum_and_string_keys() -> None:
    policy = AdaptiveActionPolicyV1(AdaptiveRAGBudgetV1(max_same_tool_retry=0))
    key = ToolCapability.LEXICAL_RETRIEVAL
    state = _state(retries={key.value: 1})

    assert policy.check_tool_retry(state, key) is ReasonCode.BUDGET_EXHAUSTED
    assert policy.check_tool_retry(state, key.value) is ReasonCode.BUDGET_EXHAUSTED


def test_replan_is_denied_once_replan_rounds_are_spent() -> None:
    policy = AdaptiveActionPolicyV1(AdaptiveRAGBudgetV1(max_replan_rounds=2))
    assert policy.check_replan(_state(replan_rounds=1)) is None
    assert policy.check_replan(_state(replan_rounds=2)) is ReasonCode.BUDGET_EXHAUSTED


def test_replan_is_denied_when_tool_budget_is_spent() -> None:
    policy = AdaptiveActionPolicyV1(
        AdaptiveRAGBudgetV1(max_replan_rounds=5, max_total_tool_calls=2)
    )
    assert policy.check_replan(_state(replan_rounds=0, tool_calls=1)) is None
    assert policy.check_replan(_state(replan_rounds=0, tool_calls=2)) is ReasonCode.BUDGET_EXHAUSTED


@pytest.mark.parametrize("default_budget", [None, AdaptiveRAGBudgetV1()])
def test_policy_defaults_to_the_shared_budget(default_budget: AdaptiveRAGBudgetV1 | None) -> None:
    policy = AdaptiveActionPolicyV1(default_budget)
    assert policy.check_tool_call(_state(tool_calls=4)) is None
    assert policy.check_tool_call(_state(tool_calls=5)) is ReasonCode.BUDGET_EXHAUSTED


def test_loop_rejects_a_policy_with_a_different_budget() -> None:
    """Two sources of truth for the same limits make a run unpredictable.

    The loop guard reads one budget and the permission checks another, so the
    run could fail closed while reporting budget it never used.
    """

    loop_budget = AdaptiveRAGBudgetV1(max_total_tool_calls=5)
    other_budget = AdaptiveRAGBudgetV1(max_total_tool_calls=1)

    with pytest.raises(ValueError, match="share the loop budget"):
        BoundedAdaptiveRAGV1(
            budget=loop_budget,
            policy=AdaptiveActionPolicyV1(other_budget),
        )


def test_loop_accepts_a_policy_sharing_its_budget() -> None:
    budget = AdaptiveRAGBudgetV1(max_total_tool_calls=3)
    loop = BoundedAdaptiveRAGV1(budget=budget, policy=AdaptiveActionPolicyV1(budget))
    assert loop.policy.budget is budget
