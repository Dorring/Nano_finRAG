from __future__ import annotations

from dataclasses import dataclass

from rag_v2.contracts.errors import ContractError


@dataclass(frozen=True)
class RepairBudget:
    """Frozen V2 repair/tool-step limits."""

    retrieval_repair_max: int = 1
    generation_repair_max: int = 1
    total_tool_steps_max: int = 8

    def __post_init__(self) -> None:
        if self.retrieval_repair_max < 0 or self.generation_repair_max < 0:
            raise ContractError("repair budgets must be non-negative")
        if self.total_tool_steps_max < 1:
            raise ContractError("total_tool_steps_max must be positive")
