from __future__ import annotations

from typing import Protocol

from rag_v2.contracts.plan import SupervisorPlan


class SupervisorProvider(Protocol):
    """Provider abstraction reserved for V2-01.

    Implementations may propose a plan, but the deterministic plan validator
    and state machine remain authoritative.
    """

    def propose_plan(self, question: str) -> SupervisorPlan:
        """Return a supervisor proposal for a question."""
