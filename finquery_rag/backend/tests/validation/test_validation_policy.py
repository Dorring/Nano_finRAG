"""Tests for intent-aware validation policies (src/validation/validation_policy.py)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.validation import ValidationStatus  # noqa: E402
from src.validation.validation_policy import (  # noqa: E402
    ACTION_BLOCK,
    ACTION_WARN,
    ValidationPolicy,
    get_policy_for_intent,
    validation_status_for_conversation,
)


class TestFinancialCalculationPolicy:
    def test_strict_grounding_and_block_actions(self):
        p = get_policy_for_intent("financial_calculation")
        assert p.require_evidence is True
        assert p.require_citations is True
        assert p.validate_numeric_claims is True
        assert p.validate_units is True
        assert p.validate_periods is True
        assert p.strict_numeric_grounding is True
        assert p.unsupported_numeric_action == ACTION_BLOCK
        assert p.missing_citation_action == ACTION_BLOCK
        assert p.applies_any_validation is True


class TestDocumentQAPolicy:
    def test_numeric_claims_are_repairable_but_citations_warn(self):
        p = get_policy_for_intent("document_qa")
        assert p.require_evidence is True
        assert p.validate_numeric_claims is True
        assert p.strict_numeric_grounding is False
        # Missing citations warn rather than block for document QA.
        assert p.missing_citation_action == ACTION_WARN
        assert p.unsupported_numeric_action == ACTION_BLOCK


class TestDocumentSummaryPolicy:
    def test_no_per_sentence_citation_required(self):
        p = get_policy_for_intent("document_summary")
        assert p.require_evidence is True
        assert p.require_citations is False
        assert p.validate_numeric_claims is True
        assert p.missing_citation_action == ACTION_WARN


class TestMultiDocumentComparisonPolicy:
    def test_block_on_missing_citation(self):
        p = get_policy_for_intent("multi_document_comparison")
        assert p.require_citations is True
        assert p.missing_citation_action == ACTION_BLOCK
        assert p.unsupported_numeric_action == ACTION_BLOCK


class TestConversationPolicy:
    def test_no_evidence_no_validation(self):
        p = get_policy_for_intent("conversation")
        assert p.require_evidence is False
        assert p.validate_numeric_claims is False
        assert p.applies_any_validation is False

    def test_conversation_status_is_not_applicable(self):
        assert validation_status_for_conversation() is ValidationStatus.NOT_APPLICABLE


class TestFrontMatterPolicy:
    def test_must_have_evidence_but_lenient_numeric(self):
        p = get_policy_for_intent("front_matter")
        assert p.require_evidence is True
        assert p.validate_numeric_claims is False
        assert p.strict_numeric_grounding is False


class TestFallback:
    def test_unknown_intent_falls_back_to_document_qa(self):
        p = get_policy_for_intent("unknown_intent_xyz")
        assert p.validate_numeric_claims is True
        assert p.require_evidence is True

    def test_none_intent_falls_back(self):
        p = get_policy_for_intent(None)
        assert p.validate_numeric_claims is True

    def test_empty_string_falls_back(self):
        p = get_policy_for_intent("")
        assert p.validate_numeric_claims is True


class TestPolicyValidation:
    def test_invalid_action_raises(self):
        with pytest.raises(ValueError):
            ValidationPolicy(
                require_evidence=True, require_citations=True,
                validate_numeric_claims=True, validate_units=True,
                validate_periods=True, strict_numeric_grounding=True,
                unsupported_numeric_action="invalid", missing_citation_action="block",
            )

    def test_policies_are_frozen(self):
        p = get_policy_for_intent("document_qa")
        with pytest.raises(Exception):
            p.require_evidence = False  # type: ignore[misc]
