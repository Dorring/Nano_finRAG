from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .calculation import CalculationResultPacket
from .errors import ContractError
from .plan import Intent


class BindingStatus(str, Enum):
    """Status values returned by the semantic binder."""

    BOUND = "BOUND"
    MISSING = "MISSING"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class BoundFact:
    """A FinancialFactV1 projection admitted into the V2 evidence packet.

    This is a reference/projection, not a replacement for FinancialFactV1.
    The value remains a string so the packet cannot silently change the
    authoritative numeric representation or scale.
    """

    fact_id: str
    candidate_id: str
    physical_source_id: str
    document_id: str
    pdf_page: int
    metric: str
    period: str
    value: str
    currency: str | None
    scale: str | None
    unit: str | None
    citation_id: str
    slot_id: str

    def __post_init__(self) -> None:
        for name in (
            "fact_id",
            "candidate_id",
            "physical_source_id",
            "document_id",
            "metric",
            "period",
            "value",
            "citation_id",
            "slot_id",
        ):
            _text(getattr(self, name), name)
        if not isinstance(self.pdf_page, int) or self.pdf_page < 0:
            raise ContractError("pdf_page must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "candidate_id": self.candidate_id,
            "physical_source_id": self.physical_source_id,
            "document_id": self.document_id,
            "pdf_page": self.pdf_page,
            "metric": self.metric,
            "period": self.period,
            "value": self.value,
            "currency": self.currency,
            "scale": self.scale,
            "unit": self.unit,
            "citation_id": self.citation_id,
            "slot_id": self.slot_id,
        }


@dataclass(frozen=True)
class EvidenceBinding:
    """Deterministic envelope around binder-selected fact IDs."""

    status: str
    slot_bindings: Mapping[str, tuple[str, ...]]
    missing_slots: tuple[str, ...] = ()
    ambiguous_slots: tuple[str, ...] = ()
    invalid_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {
            BindingStatus.BOUND,
            BindingStatus.MISSING,
            BindingStatus.AMBIGUOUS,
            BindingStatus.INVALID,
        }:
            raise ContractError(f"invalid binding status: {self.status}")
        normalized: dict[str, tuple[str, ...]] = {}
        for slot_id, fact_ids in self.slot_bindings.items():
            _text(slot_id, "slot_id")
            values = tuple(_text(fact_id, "fact_id") for fact_id in fact_ids)
            if len(values) != len(set(values)):
                raise ContractError(f"duplicate fact IDs in slot binding: {slot_id}")
            normalized[slot_id] = values
        object.__setattr__(self, "slot_bindings", MappingProxyType(normalized))
        for name, values in (
            ("missing_slots", self.missing_slots),
            ("ambiguous_slots", self.ambiguous_slots),
            ("invalid_reasons", self.invalid_reasons),
        ):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ContractError(f"{name} contains an empty value")
        if self.status == BindingStatus.BOUND:
            if not normalized or self.missing_slots or self.ambiguous_slots or self.invalid_reasons:
                raise ContractError("BOUND binding must be complete and error-free")
        elif self.status == BindingStatus.MISSING and not self.missing_slots:
            raise ContractError("MISSING binding must identify missing slots")
        elif self.status == BindingStatus.AMBIGUOUS and not self.ambiguous_slots:
            raise ContractError("AMBIGUOUS binding must identify ambiguous slots")
        elif self.status == BindingStatus.INVALID and not self.invalid_reasons:
            raise ContractError("INVALID binding must identify invalid reasons")

    @property
    def is_bound(self) -> bool:
        return self.status == BindingStatus.BOUND

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "slot_bindings": {key: list(value) for key, value in self.slot_bindings.items()},
            "missing_slots": list(self.missing_slots),
            "ambiguous_slots": list(self.ambiguous_slots),
            "invalid_reasons": list(self.invalid_reasons),
        }


@dataclass(frozen=True)
class VerifiedEvidencePacket:
    """The only evidence object permitted to enter a generator."""

    question: str
    intent: Intent
    bound_facts: tuple[BoundFact, ...]
    calculation_result: CalculationResultPacket | None
    allowed_citation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.question, "question")
        if not isinstance(self.intent, Intent):
            raise ContractError("intent must be an Intent enum")
        if not self.bound_facts:
            raise ContractError("VerifiedEvidencePacket requires bound_facts")
        fact_ids = [fact.fact_id for fact in self.bound_facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ContractError("bound_facts must not contain duplicate fact IDs")
        citation_ids = tuple(_text(item, "citation_id") for item in self.allowed_citation_ids)
        if len(citation_ids) != len(set(citation_ids)):
            raise ContractError("allowed_citation_ids must be unique")
        if not citation_ids:
            raise ContractError("VerifiedEvidencePacket requires allowed citations")
        missing = [fact.citation_id for fact in self.bound_facts if fact.citation_id not in citation_ids]
        if missing:
            raise ContractError(f"bound fact citations are not allowed: {missing}")
        if self.calculation_result is not None:
            unsupported = [
                source_id
                for source_id in self.calculation_result.supporting_evidence_ids
                if source_id not in citation_ids
            ]
            if unsupported:
                raise ContractError(f"calculation citations are not allowed: {unsupported}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent.value,
            "bound_facts": [fact.to_dict() for fact in self.bound_facts],
            "calculation_result": self.calculation_result.to_dict() if self.calculation_result else None,
            "allowed_citations": list(self.allowed_citation_ids),
        }
