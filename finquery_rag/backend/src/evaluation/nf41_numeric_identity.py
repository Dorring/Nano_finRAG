"""Evaluation-only numeric normalization for NF41 production traces."""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True)
class EvaluationNumericIdentity:
    canonical_value: Decimal
    value_type: str
    currency: str | None
    period: str | None = None


_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_numeric_identity(value: str | None, *, period: str | None = None) -> EvaluationNumericIdentity | None:
    """Normalize a reported value without changing production validation."""
    raw = (value or "").strip()
    if not raw:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    match = _NUMBER.search(raw)
    if match is None:
        return None
    try:
        amount = Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None
    if negative:
        amount = -abs(amount)
    lowered = raw.lower()
    value_type = "percentage" if "%" in raw or "per cent" in lowered else "amount"
    if value_type == "percentage":
        amount /= Decimal("100")
    else:
        multiplier = Decimal("1")
        if "billion" in lowered or re.search(r"\b\d(?:\.\d+)?\s*b\b", lowered):
            multiplier = Decimal("1000000000")
        elif "million" in lowered or re.search(r"\b\d(?:\.\d+)?\s*m\b", lowered):
            multiplier = Decimal("1000000")
        elif "thousand" in lowered or re.search(r"\b\d(?:\.\d+)?\s*k\b", lowered):
            multiplier = Decimal("1000")
        amount *= multiplier
    currency = "USD" if "$" in raw or "usd" in lowered else None
    if "swiss franc" in lowered or "chf" in lowered:
        currency = "CHF"
    return EvaluationNumericIdentity(amount, value_type, currency, period)


def value_matches_expected(raw_value: str | None, expected_numbers: tuple[str, ...] | list[str]) -> bool:
    """Compare labels that may omit the scale expressed in source evidence.

    The first branch is strict canonical comparison.  The second is a
    compatibility branch for existing labels such as ``42.2`` whose source
    spelling is ``$42.2 million``; it compares the unscaled mantissa only and
    never equates a percentage with a plain amount.
    """
    actual = normalize_numeric_identity(raw_value)
    if actual is None:
        return False
    raw_match = _NUMBER.search(raw_value or "")
    mantissa = Decimal(raw_match.group(0).replace(",", "")) if raw_match else None
    for expected in expected_numbers:
        target = normalize_numeric_identity(expected)
        if target is None:
            continue
        if actual.value_type == target.value_type and actual.canonical_value == target.canonical_value:
            return True
        if actual.value_type == "amount" and target.value_type == "amount" and mantissa == target.canonical_value:
            return True
    return False
