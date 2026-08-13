"""Deterministic one-recovery policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RecoveryAction(str, Enum):
    NO_RECOVERY = "NO_RECOVERY"
    FALLBACK_PROVIDER = "FALLBACK_PROVIDER"
    SAME_PROVIDER_REPAIR = "SAME_PROVIDER_REPAIR"


@dataclass(frozen=True)
class GenerationRecoveryPolicyV1:
    primary_provider: str
    fallback_provider: str | None = None
    action: RecoveryAction = RecoveryAction.NO_RECOVERY
    same_provider_repair_enabled: bool = False
    fallback_budget: int = 1

    def __post_init__(self) -> None:
        if self.action is RecoveryAction.FALLBACK_PROVIDER and not self.fallback_provider:
            raise ValueError("fallback provider is required for FALLBACK_PROVIDER")
        if self.action is RecoveryAction.SAME_PROVIDER_REPAIR and not self.same_provider_repair_enabled:
            raise ValueError("same-provider repair is disabled by default")
        if self.fallback_budget not in (0, 1):
            raise ValueError("fallback_budget must be 0 or 1")

    def choose(self, failure_codes: tuple[str, ...]) -> RecoveryAction:
        del failure_codes
        if self.action is RecoveryAction.FALLBACK_PROVIDER and self.fallback_budget:
            return RecoveryAction.FALLBACK_PROVIDER
        if self.action is RecoveryAction.SAME_PROVIDER_REPAIR and self.same_provider_repair_enabled:
            return RecoveryAction.SAME_PROVIDER_REPAIR
        return RecoveryAction.NO_RECOVERY

    def to_dict(self) -> dict[str, Any]:
        return {"primary_provider": self.primary_provider, "fallback_provider": self.fallback_provider,
                "action": self.action.value, "same_provider_repair_enabled": self.same_provider_repair_enabled,
                "fallback_budget": self.fallback_budget, "generation_attempt_budget": 2,
                "recovery_attempt_budget": 1}
