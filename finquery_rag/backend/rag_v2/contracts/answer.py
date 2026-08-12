from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .errors import ContractError


class CanonicalSource(str, Enum):
    """Authoritative origin of canonical answer fields."""

    FINANCIAL_FACT = "FINANCIAL_FACT"
    CALCULATOR = "CALCULATOR"


@dataclass(frozen=True)
class CanonicalAnswer:
    """Structured answer fields that a generator cannot alter."""

    value: str
    period: str
    currency: str | None
    scale: str | None
    unit: str | None
    source: CanonicalSource
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("value", "period"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ContractError(f"{name} must be a non-empty string")
        if not isinstance(self.source, CanonicalSource):
            raise ContractError("source must be a CanonicalSource enum")
        if not self.source_ids or any(not isinstance(item, str) or not item.strip() for item in self.source_ids):
            raise ContractError("canonical answer requires source_ids")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ContractError("canonical answer source_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "period": self.period,
            "currency": self.currency,
            "scale": self.scale,
            "unit": self.unit,
            "source": self.source.value,
            "source_ids": list(self.source_ids),
        }


@dataclass(frozen=True)
class AnswerEnvelope:
    """Generator output constrained by a canonical answer and citations."""

    canonical_answer: CanonicalAnswer
    answer_text: str
    citations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_answer, CanonicalAnswer):
            raise ContractError("canonical_answer must be a CanonicalAnswer")
        if not isinstance(self.answer_text, str) or not self.answer_text.strip():
            raise ContractError("answer_text must be non-empty")
        if not self.citations or any(not isinstance(item, str) or not item.strip() for item in self.citations):
            raise ContractError("AnswerEnvelope requires citations")
        if len(self.citations) != len(set(self.citations)):
            raise ContractError("citations must be unique")
        unsupported = set(self.citations) - set(self.canonical_answer.source_ids)
        if unsupported:
            raise ContractError(f"citations not present in canonical source IDs: {sorted(unsupported)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_answer": self.canonical_answer.to_dict(),
            "answer_text": self.answer_text,
            "citations": list(self.citations),
        }
