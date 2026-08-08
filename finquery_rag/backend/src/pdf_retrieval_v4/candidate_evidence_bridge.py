"""Gate 05 R5 — Candidate Evidence Bridge orchestrator.

Dispatches each Production Candidate to the appropriate Bridge Mapper
based on ``block_type``:

  table_row  → RowBridge (A1/A2/A3) → fallback MultiRowBridge (A4)
  table      → TableBridge
  text       → NarrativeBridge (A5)
  front_matter → unmapped (not structured-eligible)

Reads NO Question / Gold / Governance data.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.pdf_retrieval_v4.candidate_bridge_models import (
    BridgeGrade,
    BridgeResult,
    CandidateSignature,
    is_structured_eligible,
)
from src.pdf_retrieval_v4.candidate_multirow_bridge import MultiRowBridge
from src.pdf_retrieval_v4.candidate_narrative_bridge import NarrativeBridge
from src.pdf_retrieval_v4.candidate_row_bridge import RowBridge
from src.pdf_retrieval_v4.candidate_table_bridge import TableBridge
from src.pdf_retrieval_v4.semantic_evidence_catalog import SemanticEvidenceCatalog


class CandidateEvidenceBridge:
    """Main orchestrator for bridging Production Candidates to Semantic Evidence."""

    def __init__(self, catalog: SemanticEvidenceCatalog) -> None:
        self._catalog = catalog
        self._row_bridge = RowBridge(catalog)
        self._multirow_bridge = MultiRowBridge(catalog)
        self._table_bridge = TableBridge(catalog)
        self._narrative_bridge = NarrativeBridge(catalog)

    def bridge_all(self, signatures: list[CandidateSignature]) -> list[BridgeResult]:
        """Bridge all candidate signatures to evidence."""
        results: list[BridgeResult] = []
        for sig in signatures:
            result = self.bridge_one(sig)
            results.append(result)
        return results

    def bridge_one(self, sig: CandidateSignature) -> BridgeResult:
        """Bridge a single candidate to evidence.

        Dispatches based on block_type:
          table_row → RowBridge, fallback to MultiRowBridge
          table     → TableBridge
          text      → NarrativeBridge
          other     → unmapped (not structured-eligible)
        """
        # Check structured eligibility
        if not is_structured_eligible(sig.block_type):
            return BridgeResult(
                candidate_key=sig.candidate_key,
                grade=BridgeGrade.UNMAPPED.value,
                matches=(),
                failure_stage="candidate_type_unsupported",
                bridge_reasons=(f"block_type={sig.block_type}_not_eligible",),
            )

        # Dispatch by block type
        if sig.block_type == "table_row":
            return self._bridge_table_row(sig)
        elif sig.block_type == "table":
            return self._table_bridge.bridge(sig)
        elif sig.block_type == "text":
            result = self._narrative_bridge.bridge(sig)
            # Fallback: if narrative bridge fails and candidate has numeric content,
            # try row/multi-row bridge (some table blocks are misclassified as text)
            if result.grade == BridgeGrade.UNMAPPED.value and sig.numeric_multiset:
                row_result = self._bridge_table_row(sig)
                if row_result.grade != BridgeGrade.UNMAPPED.value:
                    return row_result
            return result
        else:
            return BridgeResult(
                candidate_key=sig.candidate_key,
                grade=BridgeGrade.UNMAPPED.value,
                matches=(),
                failure_stage="candidate_type_unsupported",
                bridge_reasons=(f"block_type={sig.block_type}_no_mapper",),
            )

    def _bridge_table_row(self, sig: CandidateSignature) -> BridgeResult:
        """Bridge a table_row candidate.

        Try RowBridge first (A1/A2/A3). If unmapped with multirow_required
        indication, try MultiRowBridge (A4).
        """
        result = self._row_bridge.bridge(sig)

        # If unmapped, try multi-row bridge as fallback
        if result.grade == BridgeGrade.UNMAPPED.value:
            # Check if the candidate might need multi-row
            if sig.numeric_multiset and len(sig.text_tokens) > 5:
                multirow_result = self._multirow_bridge.bridge(sig)
                if multirow_result.grade == BridgeGrade.A4_MULTIROW.value:
                    return multirow_result

        return result

    def build_summary(self, results: list[BridgeResult]) -> dict[str, Any]:
        """Build summary statistics from bridge results."""
        grade_counts: dict[str, int] = defaultdict(int)
        failure_counts: dict[str, int] = defaultdict(int)
        for r in results:
            grade_counts[r.grade] += 1
            if r.failure_stage:
                failure_counts[r.failure_stage] += 1

        grade_a_count = sum(
            v for k, v in grade_counts.items() if BridgeGrade.is_grade_a(k)
        )

        return {
            "total_candidates": len(results),
            "grade_counts": dict(grade_counts),
            "grade_a_count": grade_a_count,
            "grade_b_count": grade_counts.get(BridgeGrade.B_AMBIGUOUS.value, 0),
            "unmapped_count": grade_counts.get(BridgeGrade.UNMAPPED.value, 0),
            "failure_stage_counts": dict(failure_counts),
        }
