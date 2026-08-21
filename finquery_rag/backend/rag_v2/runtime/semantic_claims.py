"""Experimental deterministic post-generation semantic claim verification.

This module is intentionally separate from semantic_sufficiency. The
pre-generation gate checks whether trusted evidence can answer a request;
this verifier checks claims actually emitted by the generator. It never
consults Gold/reference data and never generates an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
import re
from typing import Any, Mapping

from rag_v2.generation.contracts import AnswerEnvelopeV1


class SemanticClaimDecision(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class SemanticClaimV1:
    claim_id: str
    claim_type: str
    text: str
    decision: SemanticClaimDecision
    evidence_ids: tuple[str, ...] = ()
    value: str | None = None
    period: str | None = None
    unit: str | None = None
    currency: str | None = None
    scale: str | None = None
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "text": self.text,
            "decision": self.decision.value,
            "evidence_ids": list(self.evidence_ids),
            "value": self.value,
            "period": self.period,
            "unit": self.unit,
            "currency": self.currency,
            "scale": self.scale,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class SemanticClaimVerificationResultV1:
    decision: SemanticClaimDecision
    claims: tuple[SemanticClaimV1, ...] = ()
    unsupported_claims: tuple[SemanticClaimV1, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "claims": [item.to_dict() for item in self.claims],
            "unsupported_claims": [item.to_dict() for item in self.unsupported_claims],
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "reason_codes": list(self.reason_codes),
        }


class SemanticClaimVerifierV1:
    """Deterministic, fail-closed checks for generated claims."""

    version = "SemanticClaimVerifierV1"
    # Accept the canonical E#/C# IDs and the existing runtime EV-/CV-style
    # evidence IDs.  The packet allow-list remains authoritative.
    _CITATION = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*)\]")
    _NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?")
    _PERIOD = re.compile(r"\b(?:FY\s*\d{4}|Q[1-4]\s*FY?\s*\d{4}|\d{4}\s*Q[1-4]|20\d{2})\b", re.I)
    _CURRENCY = re.compile(r"(?:\$|€|£|¥|\b(?:USD|EUR|GBP|JPY|CNY)\b)", re.I)
    _UNIT = re.compile(
        r"\b(?:cubic\s+feet?|square\s+feet?|feet?|meters?|metres?|kg|kilograms?|"
        r"millions?|billions?|thousands?|percent(?:age)?|ratio|shares?|dollars?)\b|%",
        re.I,
    )
    _RELATIONAL = re.compile(
        r"\b(?:increase(?:d)?|decrease(?:d)?|growth|grew|decline(?:d)?|higher|lower|"
        r"more|less|compared|versus|vs\.?|margin|rate)\b",
        re.I,
    )
    _STOPWORDS = {
        "a", "an", "and", "by", "for", "from", "how", "in", "is", "of", "on",
        "reported", "the", "this", "to", "was", "what", "were", "with", "would",
    }
    _NON_CLAIM_WORDS = {
        "available", "evidence", "information", "provided", "sufficient",
        "supplied", "supported", "verified",
    }

    @staticmethod
    def _text(value: Any) -> str:
        return "" if value is None else str(value)

    @classmethod
    def _norm(cls, value: Any) -> str:
        return " ".join(cls._text(value).lower().replace("’", "'").split())

    @classmethod
    def _tokens(cls, value: Any) -> set[str]:
        tokens = set(re.findall(r"[a-z][a-z0-9']*", cls._norm(value)))
        return {token.rstrip("0123456789") for token in tokens if token not in cls._STOPWORDS}

    @classmethod
    def _numbers(cls, value: Any) -> list[Decimal]:
        result: list[Decimal] = []
        for token in cls._NUMBER.findall(cls._text(value).replace("−", "-")):
            try:
                result.append(Decimal(token.replace(",", "").rstrip("%")))
            except InvalidOperation:
                continue
        return result

    @classmethod
    def _periods(cls, value: Any) -> set[str]:
        return {re.sub(r"\s+", "", item).upper() for item in cls._PERIOD.findall(cls._text(value))}

    @classmethod
    def _items(cls, packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return [item for item in packet.get("evidence_items", ()) if isinstance(item, Mapping)]

    @classmethod
    def _visible_text(cls, item: Mapping[str, Any]) -> str:
        fields = (
            "metric", "normalized_metric", "period", "scope", "entity", "unit",
            "currency", "scale", "value", "source_text", "text", "evidence_text",
            "content", "row_label", "column_header_path",
        )
        return " ".join(cls._text(item.get(key)) for key in fields)

    @classmethod
    def _evidence_values(cls, packet: Mapping[str, Any]) -> list[Decimal]:
        values: list[Decimal] = []
        for item in cls._items(packet):
            values.extend(cls._numbers(item.get("value")))
            values.extend(cls._numbers(item.get("source_text")))
        calculation = packet.get("calculation_result")
        if isinstance(calculation, Mapping):
            values.extend(cls._numbers(calculation.get("value")))
        return values

    @classmethod
    def _evidence_periods(cls, packet: Mapping[str, Any]) -> set[str]:
        periods: set[str] = set()
        for item in cls._items(packet):
            for key in ("period", "source_text", "evidence_text", "column_header_path"):
                periods |= cls._periods(item.get(key))
        calculation = packet.get("calculation_result")
        if isinstance(calculation, Mapping):
            periods |= cls._periods(calculation.get("period"))
        return periods

    @classmethod
    def _evidence_ids(cls, packet: Mapping[str, Any]) -> list[str]:
        return [
            cls._text(item.get("citation_id") or item.get("evidence_id") or item.get("fact_id"))
            for item in cls._items(packet)
            if item.get("citation_id") or item.get("evidence_id") or item.get("fact_id")
        ]

    @classmethod
    def _citation_map(cls, packet: Mapping[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for index, evidence_id in enumerate(cls._evidence_ids(packet), 1):
            result[f"E{index}"] = evidence_id
            result[evidence_id] = evidence_id
            result[f"C{index}"] = evidence_id
        return result

    @staticmethod
    def _close(left: Decimal, right: Decimal) -> bool:
        return abs(left - right) <= max(Decimal("0.0002"), abs(right) * Decimal("0.0005"))

    @classmethod
    def _units(cls, text: Any) -> set[str]:
        return {match.group(0).lower() for match in cls._UNIT.finditer(cls._text(text))}

    @classmethod
    def _packet_units(cls, packet: Mapping[str, Any]) -> set[str]:
        units: set[str] = set()
        for item in cls._items(packet):
            for key in ("unit", "currency", "scale"):
                if item.get(key) is not None:
                    units.add(cls._norm(item.get(key)))
            units |= cls._units(cls._visible_text(item))
            units |= {match.lower() for match in cls._CURRENCY.findall(cls._visible_text(item))}
        calculation = packet.get("calculation_result")
        if isinstance(calculation, Mapping):
            for key in ("unit", "currency", "scale"):
                if calculation.get(key) is not None:
                    units.add(cls._norm(calculation.get(key)))
        return {unit for unit in units if unit}

    @classmethod
    def _unit_supported(cls, answer_unit: str, packet: Mapping[str, Any]) -> bool:
        known = cls._packet_units(packet)
        if answer_unit in known:
            return True
        aliases = {
            "million": "1000000", "millions": "1000000",
            "billion": "1000000000", "billions": "1000000000",
            "thousand": "1000", "thousands": "1000",
        }
        return answer_unit in aliases and aliases[answer_unit] in known

    @classmethod
    def _metric_supported(cls, answer: str, packet: Mapping[str, Any]) -> bool | None:
        metrics = [
            cls._tokens(item.get("metric") or item.get("normalized_metric"))
            for item in cls._items(packet)
        ]
        metrics = [metric for metric in metrics if metric]
        if not metrics:
            return None
        # A cited answer may state only the supplied value and period.  In that
        # case it does not introduce a new metric claim that can be rejected.
        # Remove syntax/measurement tokens before checking lexical metric
        # support; substantive metric words are still checked strictly.
        metric_text = cls._CITATION.sub(" ", answer)
        metric_text = cls._NUMBER.sub(" ", metric_text)
        metric_text = cls._PERIOD.sub(" ", metric_text)
        metric_text = cls._CURRENCY.sub(" ", metric_text)
        metric_text = cls._UNIT.sub(" ", metric_text)
        answer_tokens = cls._tokens(metric_text)
        if not answer_tokens or answer_tokens <= cls._NON_CLAIM_WORDS:
            return True
        for metric in metrics:
            overlap = metric & answer_tokens
            if overlap and (overlap == metric or len(overlap) >= max(1, (len(metric) + 1) // 2)):
                return True
        return False

    def verify(
        self,
        packet: Mapping[str, Any],
        envelope: AnswerEnvelopeV1,
    ) -> SemanticClaimVerificationResultV1:
        answer = envelope.answer_text.strip()
        if not answer:
            claim = SemanticClaimV1(
                "claim-1", "answer", answer, SemanticClaimDecision.UNSUPPORTED,
                reason_codes=("SCV_EMPTY_ANSWER",),
            )
            return SemanticClaimVerificationResultV1(
                SemanticClaimDecision.UNSUPPORTED, (claim,), (claim,), (), ("SCV_EMPTY_ANSWER",)
            )

        citation_map = self._citation_map(packet)
        citations = [match.group(1).upper() for match in self._CITATION.finditer(answer)]
        envelope_ids = [self._text(value) for value in envelope.citation_ids]
        referenced_ids: list[str] = []
        claims: list[SemanticClaimV1] = []
        unsupported: list[SemanticClaimV1] = []
        ambiguous: list[SemanticClaimV1] = []
        reasons: list[str] = []

        unknown = [citation for citation in citations if citation not in citation_map]
        if unknown:
            claim = SemanticClaimV1(
                "claim-citation", "citation", ", ".join(unknown),
                SemanticClaimDecision.UNSUPPORTED, reason_codes=("SCV_UNKNOWN_CITATION",),
            )
            claims.append(claim)
            unsupported.append(claim)
            reasons.append("SCV_UNKNOWN_CITATION")
        else:
            referenced_ids.extend(citation_map[citation] for citation in citations)

        if not citations and not envelope_ids:
            claim = SemanticClaimV1(
                "claim-citation", "citation", "", SemanticClaimDecision.UNSUPPORTED,
                reason_codes=("SCV_CITATION_MISSING",),
            )
            claims.append(claim)
            unsupported.append(claim)
            reasons.append("SCV_CITATION_MISSING")
        elif envelope_ids:
            for value in envelope_ids:
                if value in citation_map:
                    referenced_ids.append(citation_map[value])

        answer_without_period = self._CITATION.sub(" ", answer)
        answer_without_period = self._PERIOD.sub(" ", answer_without_period)
        answer_numbers = self._numbers(answer_without_period)
        supported_numbers = self._evidence_values(packet)
        if answer_numbers and not all(
            any(self._close(number, supported) for supported in supported_numbers)
            for number in answer_numbers
        ):
            claim = SemanticClaimV1(
                "claim-value", "value", answer, SemanticClaimDecision.UNSUPPORTED,
                value=", ".join(str(number) for number in answer_numbers),
                evidence_ids=tuple(sorted(set(referenced_ids))),
                reason_codes=("SCV_VALUE_UNSUPPORTED",),
            )
            claims.append(claim)
            unsupported.append(claim)
            reasons.append("SCV_VALUE_UNSUPPORTED")
        elif answer_numbers:
            claims.append(SemanticClaimV1(
                "claim-value", "value", answer, SemanticClaimDecision.SUPPORTED,
                value=", ".join(str(number) for number in answer_numbers),
                evidence_ids=tuple(sorted(set(referenced_ids))),
            ))

        answer_periods = self._periods(answer)
        supported_periods = self._evidence_periods(packet)
        if answer_periods and not answer_periods <= supported_periods:
            claim = SemanticClaimV1(
                "claim-period", "period", answer, SemanticClaimDecision.UNSUPPORTED,
                period=", ".join(sorted(answer_periods)),
                evidence_ids=tuple(sorted(set(referenced_ids))),
                reason_codes=("SCV_PERIOD_UNSUPPORTED",),
            )
            claims.append(claim)
            unsupported.append(claim)
            reasons.append("SCV_PERIOD_UNSUPPORTED")
        elif answer_periods:
            claims.append(SemanticClaimV1(
                "claim-period", "period", answer, SemanticClaimDecision.SUPPORTED,
                period=", ".join(sorted(answer_periods)),
                evidence_ids=tuple(sorted(set(referenced_ids))),
            ))

        metric_status = self._metric_supported(answer, packet)
        if metric_status is False:
            claim = SemanticClaimV1(
                "claim-metric", "metric", answer, SemanticClaimDecision.UNSUPPORTED,
                evidence_ids=tuple(sorted(set(referenced_ids))),
                reason_codes=("SCV_METRIC_UNSUPPORTED",),
            )
            claims.append(claim)
            unsupported.append(claim)
            reasons.append("SCV_METRIC_UNSUPPORTED")
        elif metric_status is None:
            claim = SemanticClaimV1(
                "claim-metric", "metric", answer, SemanticClaimDecision.AMBIGUOUS,
                evidence_ids=tuple(sorted(set(referenced_ids))),
                reason_codes=("SCV_METRIC_AMBIGUOUS",),
            )
            claims.append(claim)
            ambiguous.append(claim)
            reasons.append("SCV_METRIC_AMBIGUOUS")
        else:
            claims.append(SemanticClaimV1(
                "claim-metric", "metric", answer, SemanticClaimDecision.SUPPORTED,
                evidence_ids=tuple(sorted(set(referenced_ids))),
            ))

        for index, unit in enumerate(sorted(self._units(answer)), 1):
            if self._unit_supported(unit, packet):
                claims.append(SemanticClaimV1(
                    f"claim-unit-{index}", "unit", unit, SemanticClaimDecision.SUPPORTED,
                    unit=unit, evidence_ids=tuple(sorted(set(referenced_ids))),
                ))
            else:
                claim = SemanticClaimV1(
                    f"claim-unit-{index}", "unit", unit, SemanticClaimDecision.UNSUPPORTED,
                    unit=unit, evidence_ids=tuple(sorted(set(referenced_ids))),
                    reason_codes=("SCV_UNIT_UNSUPPORTED",),
                )
                claims.append(claim)
                unsupported.append(claim)
                reasons.append("SCV_UNIT_UNSUPPORTED")

        calculation = packet.get("calculation_result")
        if self._RELATIONAL.search(answer) and not isinstance(calculation, Mapping):
            claim = SemanticClaimV1(
                "claim-relational", "relational", answer, SemanticClaimDecision.AMBIGUOUS,
                evidence_ids=tuple(sorted(set(referenced_ids))),
                reason_codes=("SCV_RELATION_AMBIGUOUS",),
            )
            claims.append(claim)
            ambiguous.append(claim)
            reasons.append("SCV_RELATION_AMBIGUOUS")

        unique_ids = tuple(sorted(set(referenced_ids)))
        decision = (
            SemanticClaimDecision.UNSUPPORTED
            if unsupported
            else SemanticClaimDecision.AMBIGUOUS if ambiguous else SemanticClaimDecision.SUPPORTED
        )
        return SemanticClaimVerificationResultV1(
            decision, tuple(claims), tuple(unsupported), unique_ids, tuple(dict.fromkeys(reasons))
        )


__all__ = [
    "SemanticClaimDecision",
    "SemanticClaimV1",
    "SemanticClaimVerificationResultV1",
    "SemanticClaimVerifierV1",
]
