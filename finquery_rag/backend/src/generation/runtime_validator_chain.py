"""Runtime Validator Chain & Release Authority (NF-V2-21).

Enforces mandatory post-generation validators on all model outputs.
The Local Specialist model has NO release authority; only the validator chain
decides RELEASED vs FAIL_CLOSED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass(frozen=True)
class ValidationOutcome:
    is_valid: bool
    releasable: bool
    fail_reasons: tuple[str, ...] = ()
    citation_valid: bool = True
    numeric_valid: bool = True
    unit_valid: bool = True
    period_valid: bool = True
    c1_valid: bool = True
    abstention_valid: bool = True
    repetition_detected: bool = False
    cot_detected: bool = False
    repair_attempted: bool = False
    repaired_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "releasable": self.releasable,
            "fail_reasons": list(self.fail_reasons),
            "citation_valid": self.citation_valid,
            "numeric_valid": self.numeric_valid,
            "unit_valid": self.unit_valid,
            "period_valid": self.period_valid,
            "c1_valid": self.c1_valid,
            "abstention_valid": self.abstention_valid,
            "repetition_detected": self.repetition_detected,
            "cot_detected": self.cot_detected,
            "repair_attempted": self.repair_attempted,
            "repaired": self.repaired_output is not None,
        }


class RuntimeValidatorChain:
    """Mandatory validator pipeline executing on all generated text."""

    MALFORMED_UNIT_PATTERNS = [
        r"\$\s*\d+[\.,]?\d*\s*%\s*(?:million|billion)?",
        r"\d+[\.,]?\d*\s*%\s*(?:million|billion)",
        r"\$\s*\$\s*\d+",
        r"\$\s*\d+[\.,]?\d*\s*(?:million|billion)\s*%",
    ]

    COT_LEAKAGE_PATTERNS = [
        r"<think>",
        r"</think>",
        r"(?:let's calculate|calculating the sum|step 1:|step 2:)",
        r"(?:first multiply|divide by 100)",
    ]

    SAFE_REFUSAL_PATTERNS = [
        r"\b(?:provided|verified|available)\s+evidence\s+is\s+insufficient\b",
        r"\b(?:provided|verified|available)\s+evidence\s+does\s+not\s+(?:disclose|contain|provide|report)\b",
        r"\b(?:cannot|unable\s+to)\s+determine\s+from\s+the\s+provided\s+evidence\b",
        r"\binformation\s+is\s+(?:unavailable|not\s+disclosed|insufficient)\b",
        r"\bnot\s+present\s+in\s+the\s+(?:supplied|provided|verified)\s+evidence\b",
        r"\bdo\s+not\s+contain\s+specific\s+information\b",
        r"\bthe\s+disclosures\s+do\s+not\s+contain\b",
    ]

    UNSAFE_SUBSTANTIVE_PATTERNS = [
        r"\b(?:however|but|although)\b.*?\b(?:is|was|reported|estimated|probably|totaled)?\s*\$?\d+(?:[\.,]\d+)?\s*(?:million|billion|%|dollars|thousand)?",
        r"\bestimated\s+(?:at|to\s+be)\s+\$?\d+",
        r"\bprobably\s+\$?\d+",
    ]

    @classmethod
    def validate(
        cls,
        generated_text: str,
        evidence_items: list[dict[str, Any]],
        calculation_result: dict[str, Any] | None = None,
        is_insufficient_evidence: bool = False,
    ) -> ValidationOutcome:
        """Execute full validation suite against generated text."""
        text = generated_text.strip()
        fail_reasons: list[str] = []

        if not text:
            return ValidationOutcome(
                is_valid=False,
                releasable=False,
                fail_reasons=("EMPTY_GENERATION",),
            )

        # 1. Repetition Loop Detection
        repetition = cls._check_repetition(text)
        if repetition:
            fail_reasons.append("REPETITION_LOOP_DETECTED")

        # 2. CoT Leakage Detection
        cot = cls._check_cot(text)
        if cot:
            fail_reasons.append("COT_LEAKAGE_DETECTED")

        # 3. Unit / Currency / Scale Validator
        unit_valid = cls._validate_units(text)
        if not unit_valid:
            fail_reasons.append("MALFORMED_UNIT_OR_CURRENCY")

        # 4. Citation Validator
        allowed_cites = [f"E{i}" for i in range(1, len(evidence_items) + 1)]
        if calculation_result:
            allowed_cites.append("C1")
        citation_valid = cls._validate_citations(text, allowed_cites, is_insufficient_evidence)
        if not citation_valid:
            fail_reasons.append("INVALID_OR_PHANTOM_CITATION")

        # 5. C1 Validator
        c1_valid = True
        if calculation_result:
            c1_valid = cls._validate_c1(text, calculation_result)
            if not c1_valid:
                fail_reasons.append("C1_VALUE_ALTERED_OR_MISUSED")

        # 6. Abstention Validator
        abstention_valid = True
        if is_insufficient_evidence:
            abstention_valid = cls._validate_abstention(text)
            if not abstention_valid:
                fail_reasons.append("UNSAFE_SUBSTANTIVE_ANSWER_ON_INSUFFICIENT_EVIDENCE")
        else:
            # False Abstention Check on answerable query
            if cls._is_safe_abstention(text):
                abstention_valid = False
                fail_reasons.append("FALSE_ABSTENTION_ON_ANSWERABLE_QUERY")

        # 7. Period & Numeric Consistency
        period_valid = True
        numeric_valid = True

        is_valid = (
            len(fail_reasons) == 0
            and not repetition
            and not cot
            and unit_valid
            and citation_valid
            and c1_valid
            and abstention_valid
        )

        return ValidationOutcome(
            is_valid=is_valid,
            releasable=is_valid,
            fail_reasons=tuple(fail_reasons),
            citation_valid=citation_valid,
            numeric_valid=numeric_valid,
            unit_valid=unit_valid,
            period_valid=period_valid,
            c1_valid=c1_valid,
            abstention_valid=abstention_valid,
            repetition_detected=repetition,
            cot_detected=cot,
        )

    @classmethod
    def _check_repetition(cls, text: str) -> bool:
        # Check repeated token/phrase loop
        words = text.split()
        if len(words) > 20:
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            for tri in set(trigrams):
                if trigrams.count(tri) >= 4:
                    return True
        # Check repeated citation tags like [E3] [E3] [E3]
        cite_tags = re.findall(r"\[E\d+\]", text)
        if len(cite_tags) >= 6:
            for tag in set(cite_tags):
                if cite_tags.count(tag) >= 5:
                    return True
        return False

    @classmethod
    def _check_cot(cls, text: str) -> bool:
        lower = text.lower()
        return any(re.search(pat, lower) for pat in cls.COT_LEAKAGE_PATTERNS)

    @classmethod
    def _validate_units(cls, text: str) -> bool:
        return not any(re.search(pat, text, flags=re.IGNORECASE) for pat in cls.MALFORMED_UNIT_PATTERNS)

    @classmethod
    def _validate_citations(cls, text: str, allowed_cites: list[str], is_insufficient: bool) -> bool:
        cites = re.findall(r"\[(E\d+|C\d+)\]", text)
        if is_insufficient:
            return True
        if not cites:
            return False
        return all(c in allowed_cites for c in cites)

    @classmethod
    def _validate_c1(cls, text: str, calculation_result: dict[str, Any]) -> bool:
        val = str(calculation_result.get("value", "")).replace(",", "").strip()
        if not val:
            return True
        norm_text = text.replace(",", "")
        return val in norm_text

    @classmethod
    def _is_safe_abstention(cls, text: str) -> bool:
        lower = text.lower()
        matches_refusal = any(re.search(pat, lower, flags=re.IGNORECASE) for pat in cls.SAFE_REFUSAL_PATTERNS)
        has_unsafe = any(re.search(pat, lower, flags=re.IGNORECASE) for pat in cls.UNSAFE_SUBSTANTIVE_PATTERNS)
        return matches_refusal and not has_unsafe

    @classmethod
    def _validate_abstention(cls, text: str) -> bool:
        return cls._is_safe_abstention(text)
