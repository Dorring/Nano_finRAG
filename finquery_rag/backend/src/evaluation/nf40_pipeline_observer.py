"""Evaluation-only, redacted observation types for the NF40 answer pipeline."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvaluationExecutionContext:
    """Explicitly prevents an NF40 invocation from mutating product state."""

    evaluation_mode: bool = True
    retrieval_enabled: bool = False
    trace_persistence_enabled: bool = False
    conversation_memory_enabled: bool = False
    feedback_write_enabled: bool = False
    document_write_enabled: bool = False
    public_http_enabled: bool = False

    def validate(self) -> None:
        if not self.evaluation_mode:
            raise ValueError("NF40 requires evaluation_mode")
        if any((self.retrieval_enabled, self.trace_persistence_enabled, self.conversation_memory_enabled, self.feedback_write_enabled, self.document_write_enabled, self.public_http_enabled)):
            raise ValueError("NF40 execution context permits a prohibited side effect")


@dataclass
class AnswerPipelineTrace:
    """Public-safe stage trace: content and answers are represented only by hashes."""

    case_id: str
    trace_id: str
    context_hash: str
    context_coverage: str
    raw_generation_hash: str | None = None
    parsed_answer: dict[str, Any] | None = None
    extracted_claims: list[dict[str, Any]] = field(default_factory=list)
    calculation_attempted: bool = False
    calculation_operation: str | None = None
    calculation_status: str | None = None
    validation_status: str | None = None
    validation_failures: list[str] = field(default_factory=list)
    repair_attempted: bool = False
    repair_status: str | None = None
    released_response_type: str | None = None
    released_answer_hash: str | None = None
    # Local-only values are intentionally excluded from public artifact
    # serialization. The NF40 runner writes them only to an ignored 0600
    # diagnostic snapshot when a real run is explicitly requested.
    _raw_generation_text: str | None = field(default=None, repr=False)
    _released_answer_text: str | None = field(default=None, repr=False)
    # Optional local-only deterministic trace used by NF41 R1. It remains
    # absent in normal production requests and is never public-serialized.
    deterministic_observer: Any | None = field(default=None, repr=False)

    def record_context(self, *, context_hash: str) -> None:
        self.context_hash = context_hash

    def record_calculation(self, *, status: str | None, operation: str | None) -> None:
        self.calculation_attempted = True
        self.calculation_status = status
        self.calculation_operation = operation

    def record_raw_generation(self, answer: str | None) -> None:
        self._raw_generation_text = answer
        self.raw_generation_hash = sha256_text(answer)

    def record_release(self, *, answer: str | None, response_type: str) -> None:
        self._released_answer_text = answer
        self.released_answer_hash = sha256_text(answer)
        self.released_response_type = response_type

    def record_validation(self, *, status: str | None, failures: list[str], repair_attempted: bool, repair_status: str | None) -> None:
        self.validation_status = status
        self.validation_failures = list(failures)
        self.repair_attempted = repair_attempted
        self.repair_status = repair_status
