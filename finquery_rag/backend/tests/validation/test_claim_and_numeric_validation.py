"""Tests for ClaimExtractor, NumericClaimValidator, and UnitPeriodValidator.

Phase 4 Commit 5 — covers claim extraction from answer text, numeric
grounding validation, and unit/period/currency consistency checks.
"""
from __future__ import annotations

import sys
import os
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.evidence import EvidenceItem
from src.domain.validation import ValidationSeverity
from src.validation.claim_extractor import ClaimExtractor
from src.validation.numeric_claim_validator import (
    CODE_NUMERIC_UNGROUND,
    NumericClaimValidator,
)
from src.validation.unit_period_validator import (
    CODE_CURRENCY_MISMATCH,
    CODE_PERIOD_MISMATCH,
    UnitPeriodValidator,
)
from src.validation.validation_policy import (
    ACTION_BLOCK,
    ACTION_WARN,
    ValidationPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evidence(
    content: str,
    *,
    chunk_id: str = "c1",
    document_name: str = "annual_report.pdf",
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        content=content,
        document_name=document_name,
        page=12,
        content_type="text",
        score=0.9,
        rerank_score=None,
        metadata={},
    )


def _strict_policy() -> ValidationPolicy:
    return ValidationPolicy(
        require_evidence=True,
        require_citations=True,
        validate_numeric_claims=True,
        validate_units=True,
        validate_periods=True,
        strict_numeric_grounding=True,
        unsupported_numeric_action=ACTION_BLOCK,
        missing_citation_action=ACTION_BLOCK,
    )


def _warn_policy() -> ValidationPolicy:
    return ValidationPolicy(
        require_evidence=True,
        require_citations=False,
        validate_numeric_claims=True,
        validate_units=True,
        validate_periods=True,
        strict_numeric_grounding=False,
        unsupported_numeric_action=ACTION_WARN,
        missing_citation_action=ACTION_WARN,
    )


def _disabled_policy() -> ValidationPolicy:
    return ValidationPolicy(
        require_evidence=False,
        require_citations=False,
        validate_numeric_claims=False,
        validate_units=False,
        validate_periods=False,
        strict_numeric_grounding=False,
        unsupported_numeric_action=ACTION_WARN,
        missing_citation_action=ACTION_WARN,
    )


# ---------------------------------------------------------------------------
# ClaimExtractor tests
# ---------------------------------------------------------------------------

class TestClaimExtractor:
    def test_empty_answer(self):
        assert ClaimExtractor().extract("") == ()
        assert ClaimExtractor().extract("   ") == ()

    def test_currency_amount(self):
        claims = ClaimExtractor().extract("Revenue was $1,000,000 for FY2025.")
        amount_claims = [c for c in claims if c.claim_type == "amount"]
        assert len(amount_claims) == 1
        assert amount_claims[0].value == Decimal("1000000")
        assert amount_claims[0].currency == "USD"

    def test_currency_with_scale(self):
        claims = ClaimExtractor().extract("Revenue was $1.5M for the year.")
        amount_claims = [c for c in claims if c.claim_type == "amount"]
        assert len(amount_claims) == 1
        assert amount_claims[0].value == Decimal("1500000")
        assert amount_claims[0].scale == "m"

    def test_plain_scaled_amount(self):
        claims = ClaimExtractor().extract("The company earned 2.5 million in revenue.")
        amount_claims = [c for c in claims if c.claim_type == "amount"]
        assert len(amount_claims) == 1
        assert amount_claims[0].value == Decimal("2500000")

    def test_percentage(self):
        claims = ClaimExtractor().extract("The gross margin was 40%.")
        pct_claims = [c for c in claims if c.claim_type == "percent"]
        assert len(pct_claims) == 1
        assert pct_claims[0].value == Decimal("40")
        assert pct_claims[0].unit == "percent"

    def test_colon_ratio(self):
        claims = ClaimExtractor().extract("The debt-to-equity ratio is 3:2.")
        ratio_claims = [c for c in claims if c.claim_type == "ratio"]
        assert len(ratio_claims) == 1
        assert ratio_claims[0].value == Decimal("1.5")

    def test_fiscal_year(self):
        claims = ClaimExtractor().extract("Revenue grew in FY2025.")
        period_claims = [c for c in claims if c.claim_type == "period"]
        assert len(period_claims) == 1
        assert period_claims[0].period == "FY2025"

    def test_quarter(self):
        claims = ClaimExtractor().extract("Q3 2024 revenue increased.")
        period_claims = [c for c in claims if c.claim_type == "period"]
        assert len(period_claims) == 1
        assert period_claims[0].period == "Q3 2024"

    def test_bare_year(self):
        claims = ClaimExtractor().extract("In 2024 the company grew.")
        period_claims = [c for c in claims if c.claim_type == "period"]
        assert len(period_claims) == 1
        assert period_claims[0].period == "2024"

    def test_citation_ref(self):
        claims = ClaimExtractor().extract("Revenue was $1M [1].")
        citation_claims = [c for c in claims if c.claim_type == "citation_ref"]
        assert len(citation_claims) == 1
        assert "1" in citation_claims[0].citation_refs

    def test_metric_proximity(self):
        claims = ClaimExtractor().extract("Total revenue was $1,000,000.")
        amount_claims = [c for c in claims if c.claim_type == "amount"]
        assert len(amount_claims) == 1
        assert amount_claims[0].metric == "revenue"

    def test_multiple_claims(self):
        claims = ClaimExtractor().extract(
            "Revenue was $1,000,000 and gross margin was 40% in FY2025."
        )
        assert len(claims) >= 3  # amount, percent, period

    def test_claims_sorted_by_position(self):
        claims = ClaimExtractor().extract(
            "FY2025 revenue was $1,000,000 with 40% margin."
        )
        # First claim should be the fiscal year (earliest in text).
        assert claims[0].claim_type == "period"

    def test_chinese_currency(self):
        claims = ClaimExtractor().extract("营收为¥100万。")
        amount_claims = [c for c in claims if c.claim_type == "amount"]
        assert len(amount_claims) == 1
        assert amount_claims[0].currency == "CNY"
        assert amount_claims[0].value == Decimal("1000000")


# ---------------------------------------------------------------------------
# NumericClaimValidator tests
# ---------------------------------------------------------------------------

class TestNumericClaimValidator:
    def test_supported_numeric_claim(self):
        """A value that appears in evidence should not produce an issue."""
        extractor = ClaimExtractor()
        validator = NumericClaimValidator()
        answer = "Revenue was $1,000,000."
        evidence = (_evidence("Total revenue was $1,000,000 for FY2025."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        # No issues because the value is grounded.
        assert len(issues) == 0

    def test_unsupported_numeric_claim(self):
        """A value not in evidence should produce an issue."""
        extractor = ClaimExtractor()
        validator = NumericClaimValidator()
        answer = "Revenue was $2,000,000."
        evidence = (_evidence("Total revenue was $1,000,000 for FY2025."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        assert len(issues) == 1
        assert issues[0].code == CODE_NUMERIC_UNGROUND
        assert issues[0].severity == ValidationSeverity.CRITICAL

    def test_policy_disabled(self):
        """When validate_numeric_claims is False, no issues."""
        extractor = ClaimExtractor()
        validator = NumericClaimValidator()
        answer = "Revenue was $2,000,000."
        evidence = (_evidence("Total revenue was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _disabled_policy())
        assert len(issues) == 0

    def test_no_evidence_all_unsupported(self):
        extractor = ClaimExtractor()
        validator = NumericClaimValidator()
        answer = "Revenue was $1,000,000."
        claims = extractor.extract(answer)
        issues = validator.validate(claims, (), _strict_policy())
        assert len(issues) >= 1
        assert all(i.code == CODE_NUMERIC_UNGROUND for i in issues)

    def test_warn_action_produces_warning(self):
        """When unsupported_numeric_action is 'warn', severity is WARNING."""
        extractor = ClaimExtractor()
        validator = NumericClaimValidator()
        answer = "Revenue was $2,000,000."
        evidence = (_evidence("Total revenue was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _warn_policy())
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.WARNING

    def test_percentage_grounded(self):
        """A percentage value found in evidence is grounded."""
        extractor = ClaimExtractor()
        validator = NumericClaimValidator()
        answer = "Gross margin was 40%."
        evidence = (_evidence("The gross margin was 40% in FY2025."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        assert len(issues) == 0

    def test_scale_conversion_grounded(self):
        """$1M in answer should match $1,000,000 in evidence."""
        extractor = ClaimExtractor()
        validator = NumericClaimValidator()
        answer = "Revenue was $1M."
        evidence = (_evidence("Total revenue was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        assert len(issues) == 0

    def test_decimal_scale_conversion_grounded(self):
        """A fractional million amount remains grounded after base-unit conversion."""
        extractor = ClaimExtractor()
        validator = NumericClaimValidator()
        answer = "Cash was $42.2 million."
        evidence = (_evidence("Cash and cash equivalents were $42.2 million."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        assert issues == ()

    def test_public_message_does_not_leak_internals(self):
        """The public_message should not contain internal values or paths."""
        extractor = ClaimExtractor()
        validator = NumericClaimValidator()
        answer = "Revenue was $2,000,000."
        evidence = (_evidence("Total revenue was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        assert len(issues) == 1
        public = issues[0].to_public_dict()
        assert "message" not in public
        assert "evidence_ids" not in public
        assert public["public_message"] is not None


# ---------------------------------------------------------------------------
# UnitPeriodValidator tests
# ---------------------------------------------------------------------------

class TestUnitPeriodValidator:
    def test_period_match_no_issue(self):
        """When the answer's period matches evidence, no issue."""
        extractor = ClaimExtractor()
        validator = UnitPeriodValidator()
        answer = "Revenue in FY2025 was $1,000,000."
        evidence = (_evidence("Total revenue for FY2025 was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        period_issues = [i for i in issues if i.code == CODE_PERIOD_MISMATCH]
        assert len(period_issues) == 0

    def test_period_mismatch(self):
        """When the answer mentions a different year, flag it."""
        extractor = ClaimExtractor()
        validator = UnitPeriodValidator()
        answer = "Revenue in FY2025 was $1,000,000."
        evidence = (_evidence("Total revenue for FY2024 was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        period_issues = [i for i in issues if i.code == CODE_PERIOD_MISMATCH]
        assert len(period_issues) == 1

    def test_currency_mismatch(self):
        """When the answer uses $ but evidence uses ¥, flag it."""
        extractor = ClaimExtractor()
        validator = UnitPeriodValidator()
        answer = "Revenue was $1,000,000."
        evidence = (_evidence("营收为¥1,000,000。"),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        currency_issues = [i for i in issues if i.code == CODE_CURRENCY_MISMATCH]
        assert len(currency_issues) == 1

    def test_currency_match_no_issue(self):
        """When both answer and evidence use the same currency, no issue."""
        extractor = ClaimExtractor()
        validator = UnitPeriodValidator()
        answer = "Revenue was $1,000,000."
        evidence = (_evidence("Total revenue was $1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        currency_issues = [i for i in issues if i.code == CODE_CURRENCY_MISMATCH]
        assert len(currency_issues) == 0

    def test_policy_disabled(self):
        """When validate_units and validate_periods are False, no issues."""
        extractor = ClaimExtractor()
        validator = UnitPeriodValidator()
        answer = "Revenue in FY2025 was $2,000,000."
        evidence = (_evidence("Total revenue for FY2024 was ¥1,000,000."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _disabled_policy())
        assert len(issues) == 0

    def test_no_evidence_no_issues(self):
        extractor = ClaimExtractor()
        validator = UnitPeriodValidator()
        answer = "Revenue in FY2025 was $1,000,000."
        claims = extractor.extract(answer)
        issues = validator.validate(claims, (), _strict_policy())
        assert len(issues) == 0

    def test_multiple_currencies_in_evidence_no_issue(self):
        """Mixed currencies in evidence should not trigger a mismatch."""
        extractor = ClaimExtractor()
        validator = UnitPeriodValidator()
        answer = "Revenue was $1,000,000."
        evidence = (
            _evidence("Revenue was $1,000,000."),
            _evidence("Cost was ¥500,000."),
        )
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        currency_issues = [i for i in issues if i.code == CODE_CURRENCY_MISMATCH]
        assert len(currency_issues) == 0

    def test_no_period_in_evidence_no_issue(self):
        """If evidence has no years, don't flag period mismatch."""
        extractor = ClaimExtractor()
        validator = UnitPeriodValidator()
        answer = "Revenue in FY2025 was $1,000,000."
        evidence = (_evidence("Revenue was one million dollars."),)
        claims = extractor.extract(answer)
        issues = validator.validate(claims, evidence, _strict_policy())
        period_issues = [i for i in issues if i.code == CODE_PERIOD_MISMATCH]
        assert len(period_issues) == 0
