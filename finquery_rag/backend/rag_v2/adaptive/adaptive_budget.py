"""Explicit, configurable bounds for adaptive retrieval."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AdaptiveRAGBudgetV1:
    max_replan_rounds: int = 2
    max_total_tool_calls: int = 5
    max_same_tool_retry: int = 1
    max_identical_query_retry: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("adaptive budgets must be non-negative")
        if self.max_total_tool_calls < 1:
            raise ValueError("max_total_tool_calls must be positive")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
