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
