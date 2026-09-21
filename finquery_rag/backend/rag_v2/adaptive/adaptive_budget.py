"""Explicit, configurable bounds for adaptive retrieval."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import ClassVar, Mapping


@dataclass(frozen=True)
class AdaptiveRAGBudgetV1:
    max_replan_rounds: int = 2
    max_total_tool_calls: int = 5
    max_same_tool_retry: int = 1
    #: Reserved.  Carried so the configuration surface is stable, but no
    #: control-flow decision reads it -- see ``RESERVED_FIELDS``.
    max_identical_query_retry: int = 0

    #: Bounds that are declared and configurable but that nothing enforces.
    #:
    #: ``max_identical_query_retry`` counts retries of a byte-identical query.
    #: ``BoundedReplannerV1`` reuses the same query text for ``MISSING_SLOT``,
    #: so enforcing this field at its default of 0 would forbid a retry the
    #: legacy runtime has always performed -- a retrieval behaviour change
    #: smuggled in through a budget.  H1 is an ablation whose only claim is that
    #: ``harness_v3`` matches ``legacy``, so it must not start enforcing it.
    #: Enforcing it is a recovery-policy decision, deferred rather than dropped.
    RESERVED_FIELDS: ClassVar[Mapping[str, int]] = MappingProxyType(
        {"max_identical_query_retry": 0}
    )

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("adaptive budgets must be non-negative")
        if self.max_total_tool_calls < 1:
            raise ValueError("max_total_tool_calls must be positive")

    def unenforced_settings(self) -> dict[str, int]:
        """Reserved fields configured away from their default.

        A non-empty result means an operator has asked for a bound that will
        not take effect.  Callers that read configuration from the environment
        should say so out loud rather than accept the value silently.
        """

        return {
            name: getattr(self, name)
            for name, default in self.RESERVED_FIELDS.items()
            if getattr(self, name) != default
        }

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
