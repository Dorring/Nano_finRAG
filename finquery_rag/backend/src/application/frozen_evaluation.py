"""Private application input used only by local frozen-context evaluations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrozenEvaluationContext:
    """Exact evidence selected by a prior, verified ranking experiment.

    This is intentionally an application-level value object, not an HTTP
    payload. Supplying it disables retrieval while retaining the normal
    production answer, calculation, validation, and repair pipeline.
    """

    context: str
    chunks: tuple[dict[str, Any], ...]
    sources: tuple[dict[str, Any], ...]
    document_names: tuple[str, ...]
    final_context_hash: str

    def validate(self) -> None:
        if not self.context.strip():
            raise ValueError("Frozen evaluation context must not be empty")
        if not self.chunks or not self.sources:
            raise ValueError("Frozen evaluation context requires evidence")
        if not self.final_context_hash:
            raise ValueError("Frozen evaluation context requires a context hash")
