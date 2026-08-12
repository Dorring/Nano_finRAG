from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .errors import ContractError


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    ERROR = "ERROR"


class ValidationDecision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REPAIR = "REPAIR"


@dataclass(frozen=True)
class ValidationResult:
    """Deterministic final-answer validation contract."""

    answerability: CheckStatus
    claim: CheckStatus
    citation: CheckStatus
    numeric: CheckStatus
    period: CheckStatus
    unit: CheckStatus
    calculation: CheckStatus
    decision: ValidationDecision
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        checks = (
            self.answerability,
            self.claim,
            self.citation,
            self.numeric,
            self.period,
            self.unit,
            self.calculation,
        )
        if any(not isinstance(item, CheckStatus) for item in checks):
            raise ContractError("all validator components must be CheckStatus enums")
        if not isinstance(self.decision, ValidationDecision):
            raise ContractError("decision must be a ValidationDecision enum")
        if any(not isinstance(issue, str) or not issue.strip() for issue in self.issues):
            raise ContractError("validation issues must be non-empty strings")
        if self.decision == ValidationDecision.PASS and any(item in {CheckStatus.FAIL, CheckStatus.ERROR} for item in checks):
            raise ContractError("PASS decision cannot contain failed/error checks")

    def to_dict(self) -> dict[str, Any]:
        return {
            "answerability": self.answerability.value,
            "claim": self.claim.value,
            "citation": self.citation.value,
            "numeric": self.numeric.value,
            "period": self.period.value,
            "unit": self.unit.value,
            "calculation": self.calculation.value,
            "decision": self.decision.value,
            "issues": list(self.issues),
        }
