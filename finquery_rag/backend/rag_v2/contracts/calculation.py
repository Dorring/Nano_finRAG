from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .errors import ContractError


class CalculationStatus(str, Enum):
    """Status of the frozen deterministic calculator handoff."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    READY = "READY"
    EXECUTED = "EXECUTED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class CalculationResultPacket:
    """Typed result handoff; arithmetic remains owned by the V1 calculator."""

    status: CalculationStatus
    operation: str | None
    value: str | None
    period: str | None
    unit: str | None
    scale: str | None
    currency: str | None
    supporting_evidence_ids: tuple[str, ...]
    formula_version: str | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, CalculationStatus):
            raise ContractError("status must be a CalculationStatus enum")
        if self.status == CalculationStatus.EXECUTED:
            if not self.operation or not self.value:
                raise ContractError("EXECUTED result requires operation and value")
            if not self.supporting_evidence_ids:
                raise ContractError("EXECUTED result requires supporting evidence")
        if self.status in {CalculationStatus.BLOCKED, CalculationStatus.FAILED} and not self.failure_reason:
            raise ContractError(f"{self.status.value} result requires failure_reason")
        if len(self.supporting_evidence_ids) != len(set(self.supporting_evidence_ids)):
            raise ContractError("supporting_evidence_ids must be unique")
        if any(not isinstance(item, str) or not item.strip() for item in self.supporting_evidence_ids):
            raise ContractError("supporting_evidence_ids must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "operation": self.operation,
            "value": self.value,
            "period": self.period,
            "unit": self.unit,
            "scale": self.scale,
            "currency": self.currency,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "formula_version": self.formula_version,
            "failure_reason": self.failure_reason,
        }
