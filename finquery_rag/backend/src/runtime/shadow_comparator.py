"""Pure comparison logic for V1-primary/V2-shadow observations.

The comparator is diagnostic only. It never changes either result, triggers
repair, or decides which runtime is correct.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .runtime_contract import FinancialQueryResult


def _value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _released(result: FinancialQueryResult | None) -> bool:
    return bool(
        result is not None
        and _value(result.status) == "ANSWER"
        and _value(result.release_status) == "RELEASED"
    )


def _normal_answer(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", " ", value.strip().casefold()) or None


def _ids(values: Iterable[Any] | None) -> tuple[str, ...]:
    if values is None or isinstance(values, (str, bytes)):
        return ()
    return tuple(dict.fromkeys(str(item) for item in values if str(item).strip()))


@dataclass(frozen=True)
class ShadowComparison:
    """Structured diagnostic comparison with no release authority."""

    decision_parity: str
    answer_semantic_parity: str
    provenance_parity: str
    calculation_parity: str
    category: str
    needs_review: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_parity": self.decision_parity,
            "answer_semantic_parity": self.answer_semantic_parity,
            "provenance_parity": self.provenance_parity,
            "calculation_parity": self.calculation_parity,
            "category": self.category,
            "needs_review": self.needs_review,
        }


class ShadowComparator:
    """Compare structured outcomes without parsing answer text as facts."""

    def compare(
        self,
        primary: FinancialQueryResult | None,
        shadow: FinancialQueryResult | None,
        *,
        shadow_status: str = "COMPLETED",
    ) -> ShadowComparison:
        if shadow_status == "TIMEOUT":
            return ShadowComparison(
                decision_parity="UNAVAILABLE",
                answer_semantic_parity="UNAVAILABLE",
                provenance_parity="UNAVAILABLE",
                calculation_parity="UNAVAILABLE",
                category="V2_TIMEOUT",
                needs_review=True,
            )
        if shadow_status == "ERROR" or shadow is None:
            return ShadowComparison(
                decision_parity="UNAVAILABLE",
                answer_semantic_parity="UNAVAILABLE",
                provenance_parity="UNAVAILABLE",
                calculation_parity="UNAVAILABLE",
                category="V2_ERROR",
                needs_review=True,
            )
        if primary is None:
            return ShadowComparison(
                decision_parity="UNAVAILABLE",
                answer_semantic_parity="UNAVAILABLE",
                provenance_parity="UNAVAILABLE",
                calculation_parity="UNAVAILABLE",
                category="V1_ERROR",
                needs_review=True,
            )

        primary_status = _value(primary.status)
        shadow_status_value = _value(shadow.status)
        primary_release = _released(primary)
        shadow_release = _released(shadow)
        decision = (
            "MATCH"
            if (
                primary_status == shadow_status_value
                and _value(primary.release_status) == _value(shadow.release_status)
            )
            else "DIFFERENT"
        )

        if primary_release and shadow_release:
            answer_parity = (
                "MATCH"
                if _normal_answer(primary.answer) == _normal_answer(shadow.answer)
                else "DIFFERENT"
            )
        else:
            answer_parity = "NOT_APPLICABLE"

        primary_ids = (
            _ids(primary.evidence_ids),
            _ids(primary.citation_ids),
            _ids(primary.calculation_ids),
        )
        shadow_ids = (
            _ids(shadow.evidence_ids),
            _ids(shadow.citation_ids),
            _ids(shadow.calculation_ids),
        )
        if not any(primary_ids):
            provenance = "V1_DATA_UNAVAILABLE"
        else:
            provenance = "MATCH" if primary_ids == shadow_ids else "DIFFERENT"

        if primary.calculation_ids:
            calculation = (
                "MATCH"
                if _ids(primary.calculation_ids) == _ids(shadow.calculation_ids)
                else "DIFFERENT"
            )
        else:
            calculation = "V1_DATA_UNAVAILABLE"

        if primary_release and shadow_release:
            category = "AGREE_RELEASE" if answer_parity == "MATCH" else "ANSWER_DISAGREEMENT"
        elif primary_release and not shadow_release:
            category = "V1_ONLY_RELEASE"
        elif shadow_release and not primary_release:
            category = "V2_ONLY_RELEASE"
        elif primary_status == "FAIL_CLOSED" and shadow_status_value == "FAIL_CLOSED":
            category = "AGREE_FAIL_CLOSED"
        elif shadow_status_value == "ERROR":
            category = "V2_ERROR"
        else:
            category = "OUTCOME_DISAGREEMENT"

        needs_review = category not in {"AGREE_RELEASE", "AGREE_FAIL_CLOSED"}
        if provenance == "DIFFERENT" or calculation == "DIFFERENT":
            needs_review = True
            if category == "AGREE_RELEASE":
                category = (
                    "CALCULATION_DISAGREEMENT"
                    if calculation == "DIFFERENT"
                    else "PROVENANCE_DISAGREEMENT"
                )
        return ShadowComparison(
            decision_parity=decision,
            answer_semantic_parity=answer_parity,
            provenance_parity=provenance,
            calculation_parity=calculation,
            category=category,
            needs_review=needs_review,
        )


__all__ = ["ShadowComparator", "ShadowComparison"]