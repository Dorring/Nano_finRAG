"""Dependency-light typed contracts for the V2 shadow architecture."""

from .answer import AnswerEnvelope, CanonicalAnswer, CanonicalSource
from .calculation import CalculationResultPacket, CalculationStatus
from .evidence import BindingStatus, EvidenceBinding
from .errors import ContractError, PlanValidationError, StateTransitionError
from .plan import Action, Intent, RequiredSlot, SupervisorPlan
from .query import QuestionEnvelope
from .validation import CheckStatus, ValidationDecision, ValidationResult

__all__ = [
    "Action",
    "AnswerEnvelope",
    "BindingStatus",
    "CanonicalAnswer",
    "CanonicalSource",
    "CheckStatus",
    "CalculationResultPacket",
    "CalculationStatus",
    "ContractError",
    "EvidenceBinding",
    "Intent",
    "PlanValidationError",
    "QuestionEnvelope",
    "RequiredSlot",
    "StateTransitionError",
    "SupervisorPlan",
    "ValidationDecision",
    "ValidationResult",
]
