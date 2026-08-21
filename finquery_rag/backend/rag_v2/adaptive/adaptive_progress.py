"""Progress signatures used to stop identical adaptive retries."""
from __future__ import annotations

from typing import Any, Iterable

from .adaptive_contracts import AdaptiveRAGStateV1, EvidencePacketV1, stable_hash


class ProgressDetectorV1:
    """Detect no information gain without embeddings or model judgments."""

    @staticmethod
    def signature(
        *,
        query: str,
        capability: str,
        packets: Iterable[EvidencePacketV1],
        filled_slots: Iterable[str] = (),
        missing_slots: Iterable[str] = (),
        conflicts: Iterable[Any] = (),
        calculation_ready: bool = False,
    ) -> str:
        payload = {
            "query": " ".join(str(query).split()).casefold(),
            "capability": str(capability),
            "evidence": sorted(packet.content_hash for packet in packets),
            "filled": sorted(str(item) for item in filled_slots),
            "missing": sorted(str(item) for item in missing_slots),
            "conflicts": sorted(map(str, conflicts)),
            "calculation_ready": bool(calculation_ready),
        }
        return stable_hash(payload)

    def observe(self, state: AdaptiveRAGStateV1, signature: str) -> bool:
        """Return True when this observation is new and records it."""
        if signature in state.progress_signatures:
            return False
        state.progress_signatures.append(signature)
        return True
