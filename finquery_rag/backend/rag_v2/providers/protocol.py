from __future__ import annotations

from typing import Protocol

from rag_v2.contracts.plan import SupervisorPlan


class SupervisorProvider(Protocol):
    """Provider abstraction for a single V2-01 question-to-plan call."""

    def plan(self, question: str) -> SupervisorPlan:
        """Return one supervisor proposal; downstream tools are not executed."""
