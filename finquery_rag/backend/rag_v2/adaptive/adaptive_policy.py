"""Action permission for the bounded adaptive control plane.

``BoundedReplannerV1`` *proposes* the next capability.  This module decides
whether executing it is *allowed*.  Keeping the two apart is deliberate: a
proposal is a hypothesis about what would help, a permission is a statement
about what the run is still entitled to spend.

The checks here were previously inline in ``BoundedAdaptiveRAGV1.run``.  They
are extracted verbatim so that the loop's behaviour is unchanged; the extraction
exists to make the seam explicit and testable on its own.
"""
from __future__ import annotations

from .adaptive_budget import AdaptiveRAGBudgetV1
from .adaptive_contracts import AdaptiveRAGStateV1, ReasonCode, ToolCapability


class AdaptiveActionPolicyV1:
    """Decide whether a proposed action may execute.

    Every method returns ``None`` when the action is permitted, or the
    ``ReasonCode`` that denies it.  Denials are terminal in the caller: the run
    is expected to fail closed rather than pick a different action.
    """

    def __init__(self, budget: AdaptiveRAGBudgetV1 | None = None) -> None:
        self.budget = budget or AdaptiveRAGBudgetV1()

    def check_tool_call(self, state: AdaptiveRAGStateV1) -> ReasonCode | None:
        """Permit a tool call only while the total tool budget has room."""

        if state.tool_calls >= self.budget.max_total_tool_calls:
            return ReasonCode.BUDGET_EXHAUSTED
        return None

    def check_tool_retry(
        self,
        state: AdaptiveRAGStateV1,
        capability: ToolCapability | str,
    ) -> ReasonCode | None:
        """Permit a repeat of one capability only up to the retry allowance."""

        key = self._key(capability)
        if state.same_tool_retries.get(key, 0) > self.budget.max_same_tool_retry:
            return ReasonCode.BUDGET_EXHAUSTED
        return None

    def check_replan(self, state: AdaptiveRAGStateV1) -> ReasonCode | None:
        """Permit one more replan only while both replan and tool budget remain."""

        if state.replan_rounds >= self.budget.max_replan_rounds:
            return ReasonCode.BUDGET_EXHAUSTED
        if state.tool_calls >= self.budget.max_total_tool_calls:
            return ReasonCode.BUDGET_EXHAUSTED
        return None

    @staticmethod
    def _key(capability: ToolCapability | str) -> str:
        return capability.value if isinstance(capability, ToolCapability) else str(capability)
