from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    evidence_type: str
    candidate_key: str
    candidate_rank: int
    document_id: str
    payload: dict[str, Any]
    source_traceback: tuple[dict[str, Any], ...] = ()
