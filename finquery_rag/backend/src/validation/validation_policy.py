"""Intent-aware validation policies (Phase 4).

A ``ValidationPolicy`` describes how strictly a given intent should be
validated. Not all intents use the same rules: a financial calculation
requires strict numeric grounding and citation for every operand, while a
plain conversation must not be rejected for lacking document evidence.

The ``get_policy_for_intent`` factory returns a frozen policy per intent
string. Intent strings are produced by ``src.services.intent.classify_query_intent``
(``financial_calculation``, ``document_qa``, ``document_summary``,
``multi_document_comparison`` (detected via multi-doc scope), ``conversation``,
``unsupported``, ``front_matter``).

Layer dependency: ``domain <- validation``. This module imports only from
``src.domain`` and stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.domain.validation import ValidationStatus


# Action strings used by validators to decide what to do when a claim lacks
# support or a citation is missing. ``block`` means the issue is CRITICAL and
# the answer must not be returned; ``warn`` means a WARNING is recorded but
# the answer may still pass (for lenient intents).
ACTION_BLOCK = "block"
ACTION_WARN = "warn"
_VALID_ACTIONS = {ACTION_BLOCK, ACTION_WARN}


@dataclass(frozen=True)
class ValidationPolicy:
    """How strictly to validate answers for a given intent.

    Attributes:
        require_evidence: if True, the absence of any retrieved evidence
            forces NOT_ANSWERABLE (no LLM).
        require_citations: if True, numeric/calculation claims must cite
            evidence that actually contains the value.
        validate_numeric_claims: if True, run NumericClaimValidator.
        validate_units: if True, run UnitPeriodValidator.
        validate_periods: if True, period/year mismatches are blocking.
        strict_numeric_grounding: if True, ANY unsupported numeric claim
            is CRITICAL (BLOCK); if False, unsupported numerics are ERROR
            and may be repairable.
        unsupported_numeric_action: ``block`` or ``warn``.
        missing_citation_action: ``block`` or ``warn``.
    """

    require_evidence: bool
    require_citations: bool
    validate_numeric_claims: bool
    validate_units: bool
    validate_periods: bool
    strict_numeric_grounding: bool
    unsupported_numeric_action: str
    missing_citation_action: str

    def __post_init__(self) -> None:
        if self.unsupported_numeric_action not in _VALID_ACTIONS:
            raise ValueError(
                "unsupported_numeric_action must be 'block' or 'warn', got %r"
                % (self.unsupported_numeric_action,)
            )
        if self.missing_citation_action not in _VALID_ACTIONS:
            raise ValueError(
                "missing_citation_action must be 'block' or 'warn', got %r"
                % (self.missing_citation_action,)
            )

    @property
    def applies_any_validation(self) -> bool:
        """True if at least one validator should run for this intent."""
        return (
            self.validate_numeric_claims
            or self.validate_units
            or self.validate_periods
            or self.require_citations
        )


# ---------------------------------------------------------------------------
# Per-intent policies
# ---------------------------------------------------------------------------

_FINANCIAL_CALCULATION = ValidationPolicy(
    require_evidence=True,
    require_citations=True,
    validate_numeric_claims=True,
    validate_units=True,
    validate_periods=True,
    strict_numeric_grounding=True,
    unsupported_numeric_action=ACTION_BLOCK,
    missing_citation_action=ACTION_BLOCK,
)

_DOCUMENT_QA = ValidationPolicy(
    require_evidence=True,
    require_citations=True,
    validate_numeric_claims=True,
    validate_units=True,
    validate_periods=True,
    # A retrieved-document answer can contain several independent claims.
    # Treat an ungrounded numeric claim as repairable so the response layer
    # can remove only that claim and keep the evidence-backed remainder.
    # Calculation and comparison intents remain fail-closed above.
    strict_numeric_grounding=False,
    unsupported_numeric_action=ACTION_BLOCK,
    missing_citation_action=ACTION_WARN,
)

_DOCUMENT_SUMMARY = ValidationPolicy(
    require_evidence=True,
    require_citations=False,
    validate_numeric_claims=True,
    validate_units=True,
    validate_periods=True,
    strict_numeric_grounding=True,
    unsupported_numeric_action=ACTION_BLOCK,
    missing_citation_action=ACTION_WARN,
)

_MULTI_DOCUMENT_COMPARISON = ValidationPolicy(
    require_evidence=True,
    require_citations=True,
    validate_numeric_claims=True,
    validate_units=True,
    validate_periods=True,
    strict_numeric_grounding=True,
    unsupported_numeric_action=ACTION_BLOCK,
    missing_citation_action=ACTION_BLOCK,
)

_FRONT_MATTER = ValidationPolicy(
    require_evidence=True,
    require_citations=True,
    validate_numeric_claims=False,
    validate_units=False,
    validate_periods=False,
    strict_numeric_grounding=False,
    unsupported_numeric_action=ACTION_WARN,
    missing_citation_action=ACTION_WARN,
)

_CONVERSATION = ValidationPolicy(
    require_evidence=False,
    require_citations=False,
    validate_numeric_claims=False,
    validate_units=False,
    validate_periods=False,
    strict_numeric_grounding=False,
    unsupported_numeric_action=ACTION_WARN,
    missing_citation_action=ACTION_WARN,
)

_UNSUPPORTED = ValidationPolicy(
    require_evidence=False,
    require_citations=False,
    validate_numeric_claims=False,
    validate_units=False,
    validate_periods=False,
    strict_numeric_grounding=False,
    unsupported_numeric_action=ACTION_WARN,
    missing_citation_action=ACTION_WARN,
)


_POLICY_BY_INTENT: dict[str, ValidationPolicy] = {
    "financial_calculation": _FINANCIAL_CALCULATION,
    "document_qa": _DOCUMENT_QA,
    "document_summary": _DOCUMENT_SUMMARY,
    "multi_document_comparison": _MULTI_DOCUMENT_COMPARISON,
    "front_matter": _FRONT_MATTER,
    "conversation": _CONVERSATION,
    "unsupported": _UNSUPPORTED,
}

# Default fallback for unknown intents: treat as document QA (conservative).
_DEFAULT_POLICY = _DOCUMENT_QA


def get_policy_for_intent(intent: str | None) -> ValidationPolicy:
    """Return the frozen ValidationPolicy for the given intent string.

    Unknown / None intents fall back to the document_qa policy so that
    numeric grounding is still enforced conservatively.
    """
    if not intent:
        return _DEFAULT_POLICY
    return _POLICY_BY_INTENT.get(intent, _DEFAULT_POLICY)


def validation_status_for_conversation() -> ValidationStatus:
    """Conversation intents are NOT_APPLICABLE for validation."""
    return ValidationStatus.NOT_APPLICABLE
