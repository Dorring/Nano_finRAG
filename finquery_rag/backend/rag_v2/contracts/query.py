from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import ContractError


@dataclass(frozen=True)
class QuestionEnvelope:
    """Minimal V2 input envelope; it contains no answer/Gold fields."""

    question_id: str
    question: str
    document_scope: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("question_id", "question"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{name} must be non-empty")
        if any(not isinstance(item, str) or not item.strip() for item in self.document_scope):
            raise ContractError("document_scope values must be non-empty strings")

    @classmethod
    def from_benchmark_record(cls, record: Mapping[str, Any]) -> "QuestionEnvelope":
        if not isinstance(record, Mapping):
            raise ContractError("benchmark record must be an object")
        question_id = record.get("case_id") or record.get("question_id")
        question = record.get("question") or record.get("query")
        scope = record.get("document_scope") or ()
        if isinstance(scope, str):
            scope = (scope,)
        if not isinstance(scope, (list, tuple)):
            raise ContractError("document_scope must be an array or string")
        return cls(str(question_id), str(question), tuple(str(item) for item in scope))

    def to_dict(self) -> dict[str, Any]:
        return {"question_id": self.question_id, "question": self.question, "document_scope": list(self.document_scope)}
