"""Numeric claim grounding validator (Phase 4 Commit 5 + Phase 4 hotfix).

The ``NumericClaimValidator`` checks whether each numeric claim extracted
from the generated answer is grounded in the retrieved evidence. A numeric
claim is considered *grounded* if:

1. Its **value** (in some textual form) appears in at least one evidence
   chunk.
2. If the claim has a **metric**, the same evidence chunk must also
   contain that metric keyword (or a known alias).
3. If the claim has a **period**, the same evidence chunk must also
   contain that period (year).

All three checks must pass on the **same** evidence item. A value that
appears in one chunk but with a different metric or period is NOT
considered grounded — this prevents cross-contamination where a correct
number for the wrong year/metric passes validation.

This validator does NOT use an LLM. It performs deterministic text matching
with multiple value representations (plain, comma-formatted, scale-suffixed)
to handle format variations between the LLM output and the source text.

Layer dependency: ``domain <- validation``. Imports only from ``src.domain``
and stdlib.
"""
from __future__ import annotations

import re
from decimal import Decimal

from src.domain.evidence import EvidenceItem
from src.domain.validation import ExtractedClaim, ValidationIssue, ValidationSeverity
from src.validation.validation_policy import (
    ACTION_WARN,
    ValidationPolicy,
)
from src.validation.claim_extractor import _METRIC_CANONICAL


# Issue code for unsupported numeric claims.
CODE_NUMERIC_UNGROUND = "NUMERIC_UNGROUND"
CODE_NUMERIC_VALUE_MISMATCH = "NUMERIC_VALUE_MISMATCH"
CODE_METRIC_VALUE_MISMATCH = "METRIC_VALUE_MISMATCH"
CODE_PERIOD_VALUE_MISMATCH = "PERIOD_VALUE_MISMATCH"
CODE_PERIOD_AMBIGUOUS = "PERIOD_AMBIGUOUS"


class NumericClaimValidator:
    """Validates that numeric claims are grounded in evidence.

    The validator checks each ``amount``, ``percent``, and ``ratio`` claim
    to see if its value, metric, and period all appear in the same evidence
    chunk. Claims whose values cannot be found in any evidence chunk (or
    whose metric/period don't match on the same chunk) are flagged as
    unsupported.
    """

    def validate(
        self,
        claims: tuple[ExtractedClaim, ...],
        evidence: tuple[EvidenceItem, ...],
        policy: ValidationPolicy,
    ) -> tuple[ValidationIssue, ...]:
        """Validate numeric claims against the evidence set.

        Returns a tuple of ``ValidationIssue`` objects. If the policy
        disables numeric claim validation, returns an empty tuple.
        """
        if not policy.validate_numeric_claims:
            return ()

        numeric_claims = tuple(
            c for c in claims if c.claim_type in ("amount", "percent", "ratio")
        )
        if not numeric_claims:
            return ()

        if not evidence:
            # No evidence means all numeric claims are unsupported.
            return tuple(
                self._build_unsupported_issue(c, policy, evidence_ids=())
                for c in numeric_claims
            )

        issues: list[ValidationIssue] = []
        for claim in numeric_claims:
            supporting_ids = self._find_supporting_evidence(claim, evidence)
            if not supporting_ids:
                issues.append(
                    self._build_unsupported_issue(
                        claim, policy, evidence_ids=supporting_ids
                    )
                )

        return tuple(issues)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _find_supporting_evidence(
        claim: ExtractedClaim,
        evidence: tuple[EvidenceItem, ...],
    ) -> tuple[str, ...]:
        """Find evidence chunk IDs that support the claim.

        A chunk supports a claim if ALL of the following are true:
        1. The claim's value appears in the chunk text.
        2. If the claim has a metric, the chunk also contains that metric
           keyword (or an alias).
        3. If the claim has a period, the chunk also contains that period
           (year).

        Returns a tuple of chunk IDs. An empty tuple means no supporting
        evidence was found.
        """
        if claim.value is None:
            return ()

        representations = NumericClaimValidator._value_representations(claim)
        claim_metric_aliases = NumericClaimValidator._metric_aliases(claim.metric)
        claim_years = NumericClaimValidator._extract_years(claim.period)

        supporting: list[str] = []
        for item in evidence:
            text = item.content or ""

            # 1. Value must appear in the chunk.
            value_found = False
            for rep in representations:
                if rep and rep in text:
                    value_found = True
                    break
            if not value_found:
                continue

            # 2. If claim has a metric, the chunk must contain it.
            if claim_metric_aliases:
                metric_found = False
                text_lower = text.lower()
                for alias in claim_metric_aliases:
                    if alias in text_lower:
                        metric_found = True
                        break
                if not metric_found:
                    continue

            # 3. If claim has a period, the chunk must contain that year.
            if claim_years:
                evidence_years = set(
                    re.findall(r"(?<!\d)(20[0-2]\d|19[89]\d)(?!\d)", text)
                )
                if not claim_years.intersection(evidence_years):
                    continue

            supporting.append(item.chunk_id)
        return tuple(supporting)

    @staticmethod
    def _metric_aliases(metric: str | None) -> tuple[str, ...]:
        """Return all keyword aliases for a canonical metric.

        If the metric is None, returns an empty tuple (no metric constraint).
        """
        if metric is None:
            return ()
        aliases: list[str] = []
        for keyword, canonical in _METRIC_CANONICAL.items():
            if canonical == metric:
                aliases.append(keyword)
        # Also include the metric itself if it's not in the canonical map.
        if metric not in _METRIC_CANONICAL.values():
            aliases.append(metric)
        return tuple(aliases)

    @staticmethod
    def _extract_years(period: str | None) -> set[str]:
        """Extract 4-digit years from a period string."""
        if not period:
            return set()
        return set(re.findall(r"(?<!\d)(20[0-2]\d|19[89]\d)(?!\d)", period))

    @staticmethod
    def _value_representations(claim: ExtractedClaim) -> tuple[str, ...]:
        """Generate multiple string representations of the claim value.

        The LLM may format numbers differently from the source text:
        - ``1000000`` vs ``1,000,000`` vs ``1 million``
        - ``40%`` vs ``40.00%`` vs ``0.40`` (for percentages)
        """
        if claim.value is None:
            return ()

        value = claim.value
        reps: list[str] = []

        if claim.claim_type == "percent":
            # Percentage: 40% could appear as "40%", "40.00%", "40 percent"
            reps.append(f"{value}%")
            reps.append(f"{value:.2f}%")
            # Also check if it appears as a decimal ratio (0.40)
            ratio = value / Decimal("100")
            reps.append(str(ratio))
            reps.append(f"{ratio:.2f}")
            reps.append(f"{ratio:.4f}")

        elif claim.claim_type == "ratio":
            # Ratio: 0.4 could appear as "0.4", "0.40", "40%"
            reps.append(str(value))
            reps.append(f"{value:.2f}")
            reps.append(f"{value:.4f}")
            # Also as percentage
            pct = value * Decimal("100")
            reps.append(f"{pct}%")
            reps.append(f"{pct:.2f}%")

        elif claim.claim_type == "amount":
            # Amount: try plain, comma-formatted, and scale-suffixed
            reps.append(str(value))
            reps.append(f"{value:,}")
            # Try without trailing zeros
            reps.append(str(value.normalize()))
            # Try with 2 decimal places
            reps.append(f"{value:.2f}")
            # Try scale-suffixed representations
            reps.extend(NumericClaimValidator._scale_representations(value))

        # Always include the integer representation if the value is whole.
        if value == value.to_integral_value():
            int_val = int(value)
            reps.append(str(int_val))
            reps.append(f"{int_val:,}")

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for r in reps:
            if r and r not in seen:
                seen.add(r)
                unique.append(r)
        return tuple(unique)

    @staticmethod
    def _scale_representations(value: Decimal) -> tuple[str, ...]:
        """Generate normalized textual representations at common financial scales.

        Decimal amounts are stored in base units, so "$42.2 million" becomes
        "42,200,000" during claim extraction. Preserve fractional scale
        representations as well as whole-unit abbreviations before matching
        against source evidence.
        """
        reps: list[str] = []
        absolute = abs(value)
        for divisor, label, suffix in (
            (Decimal("1000000000"), "billion", "B"),
            (Decimal("1000000"), "million", "M"),
            (Decimal("1000"), "thousand", "K"),
        ):
            if absolute < divisor:
                continue
            scaled = value / divisor
            compact = format(scaled.normalize(), "f")
            reps.extend((
                f"{compact} {label}",
                f"{scaled:.2f} {label}",
                f"{compact}{suffix}",
            ))
        return tuple(reps)

    @staticmethod
    def _build_unsupported_issue(
        claim: ExtractedClaim,
        policy: ValidationPolicy,
        evidence_ids: tuple[str, ...],
    ) -> ValidationIssue:
        """Build a ValidationIssue for an unsupported numeric claim.

        Severity depends on the policy:
        - ``strict_numeric_grounding=True`` -> CRITICAL (always blocks).
        - ``strict_numeric_grounding=False`` -> ERROR (may be repairable).
        - ``unsupported_numeric_action="warn"`` -> WARNING (non-blocking).
        """
        if policy.unsupported_numeric_action == ACTION_WARN:
            severity = ValidationSeverity.WARNING
        elif policy.strict_numeric_grounding:
            severity = ValidationSeverity.CRITICAL
        else:
            severity = ValidationSeverity.ERROR

        metric_label = claim.metric or claim.claim_type
        period_label = claim.period or "unknown"
        return ValidationIssue(
            code=CODE_NUMERIC_UNGROUND,
            severity=severity,
            message=(
                f"Numeric claim '{claim.raw_text}' (metric: {metric_label}, "
                f"value: {claim.value}, period: {period_label}) not found "
                f"in any evidence chunk with matching metric and period."
            ),
            claim_text=claim.raw_text,
            evidence_ids=evidence_ids,
            public_message=(
                "A numeric value in the answer could not be verified "
                "against the source documents."
            ),
        )
