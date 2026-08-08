"""Gate 05 R5 — Bridge Equivalence handler.

Preserves semantic equivalent-set grouping from Gate 03 R2 during bridge.

Rules:
  - If a candidate maps to ANY physical row in an equivalent set,
    bridge_status = A_equivalent
  - The canonical semantic fact ID represents the entire set
  - No double-counting: one candidate → one structured view even if
    multiple equivalent physical rows are matched
  - Disambiguation must NOT use row_index (pick first, etc.)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.pdf_retrieval_v4.candidate_bridge_models import (
    BridgeGrade,
    BridgeResult,
    SemanticEvidenceSignature,
)
from src.pdf_retrieval_v4.semantic_evidence_catalog import SemanticEvidenceCatalog


class BridgeEquivalence:
    """Handles equivalent-set logic for bridge results."""

    def __init__(self, catalog: SemanticEvidenceCatalog) -> None:
        self._catalog = catalog
        # Build: equivalent_group_id → list of evidence signatures
        self._groups: dict[str, list[SemanticEvidenceSignature]] = defaultdict(list)
        # Build: evidence_id → equivalent_group_id
        self._evidence_to_group: dict[str, str] = {}
        self._build_groups()

    def _build_groups(self) -> None:
        """Build equivalent set lookups from catalog."""
        for ev in self._catalog.get_all():
            if ev.equivalent_group_id:
                self._groups[ev.equivalent_group_id].append(ev)
                self._evidence_to_group[ev.evidence_id] = ev.equivalent_group_id

    @property
    def group_count(self) -> int:
        return len(self._groups)

    def get_group_for_evidence(self, evidence_id: str) -> str | None:
        """Get the equivalent group ID for an evidence ID."""
        return self._evidence_to_group.get(evidence_id)

    def get_group_members(self, group_id: str) -> list[SemanticEvidenceSignature]:
        """Get all evidence in an equivalent group."""
        return self._groups.get(group_id, [])

    def check_equivalent_bridge(
        self,
        result: BridgeResult,
    ) -> BridgeResult:
        """Check if a bridge result should be upgraded to A_equivalent.

        If the matched evidence belongs to an equivalent set, upgrade
        the grade to A_equivalent and record the canonical group.
        """
        if not BridgeGrade.is_grade_a(result.grade):
            return result

        # Check if any matched evidence is in an equivalent set
        for match in result.matches:
            group_id = self._evidence_to_group.get(match.evidence_id)
            if group_id:
                # Upgrade to A_equivalent
                return BridgeResult(
                    candidate_key=result.candidate_key,
                    grade=BridgeGrade.A_EQUIVALENT.value,
                    matches=result.matches,
                    failure_stage=None,
                    bridge_reasons=result.bridge_reasons
                    + (f"equivalent_group={group_id[:40]}...",),
                )

        return result

    def detect_double_count(
        self,
        results: list[BridgeResult],
    ) -> list[dict[str, Any]]:
        """Detect if any evidence in an equivalent set is matched to
        multiple candidates (double-counting).

        Returns a list of double-count violations.
        """
        # Map: evidence_id → list of candidate_keys that matched it
        evidence_to_candidates: dict[str, list[str]] = defaultdict(list)
        for result in results:
            if not BridgeGrade.is_grade_a(result.grade):
                continue
            for match in result.matches:
                evidence_to_candidates[match.evidence_id].append(result.candidate_key)

        violations: list[dict[str, Any]] = []
        for group_id, members in self._groups.items():
            # Check if multiple candidates map to different members of the same group
            group_candidates: dict[str, set[str]] = defaultdict(set)
            for member in members:
                candidates = evidence_to_candidates.get(member.evidence_id, [])
                for ck in candidates:
                    group_candidates[ck].add(member.evidence_id)

            # If multiple distinct candidates map to this group, it's a potential double-count
            if len(group_candidates) > 1:
                violations.append(
                    {
                        "group_id": group_id,
                        "candidate_keys": list(group_candidates.keys()),
                        "evidence_per_candidate": {
                            ck: list(eids) for ck, eids in group_candidates.items()
                        },
                    }
                )

        return violations

    def stats(self) -> dict[str, Any]:
        """Return equivalence statistics."""
        return {
            "equivalent_group_count": len(self._groups),
            "total_equivalent_evidence": sum(
                len(members) for members in self._groups.values()
            ),
        }
