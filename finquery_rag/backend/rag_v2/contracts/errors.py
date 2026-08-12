from __future__ import annotations


class ContractError(ValueError):
    """Raised when a V2 typed contract is malformed."""


class PlanValidationError(ContractError):
    """Raised when a supervisor plan violates the deterministic validator."""


class StateTransitionError(ContractError):
    """Raised when the V2 state machine receives an illegal transition."""
