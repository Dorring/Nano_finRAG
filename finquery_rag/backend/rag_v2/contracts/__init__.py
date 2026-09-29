"""Dependency-light typed contracts for the V2 shadow architecture."""

from .answer import AnswerEnvelope, CanonicalAnswer, CanonicalSource
from .calculation import CalculationResultPacket, CalculationStatus
from .derived_metric import (
    DerivedMetricDefinition,
    get_derived_metric_definition,
    is_supported_derived_metric,
)
from .evidence import BindingStatus, EvidenceBinding
from .errors import ContractError, PlanValidationError, StateTransitionError
from .plan import (
    Action,
    ExecutionDAG,
    Intent,
    OperandRef,
    OperationStep,
    RequiredSlot,
    SupervisorPlan,
    slot_key_error,
)
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
    "DerivedMetricDefinition",
    "EvidenceBinding",
    "ExecutionDAG",
    "Intent",
    "OperandRef",
    "OperationStep",
    "PlanValidationError",
    "QuestionEnvelope",
    "RequiredSlot",
    "StateTransitionError",
    "SupervisorPlan",
    "ValidationDecision",
    "ValidationResult",
    "get_derived_metric_definition",
    "is_supported_derived_metric",
    "slot_key_error",
]
