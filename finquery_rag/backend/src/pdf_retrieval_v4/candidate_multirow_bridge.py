"""Gate 05 R5 — Multi-row Candidate Bridge.

Handles Production Candidates whose content spans multiple contiguous
semantic rows within the same logical table (e.g., a candidate that
includes a header + multiple data rows).

Grade: A4_multirow

Requirements:
  - Same logical table
  - Same page
  - Rows contiguous (by row_index)
  - Combined numeric recall = 1.0
  - Combined text coverage >= threshold
"""

from __future__ import annotations


from src.pdf_retrieval_v4.candidate_bridge_models import (
    BRIDGE_ELIGIBLE_ROW_TYPES,
    BridgeGrade,
    BridgeMatch,
    BridgeResult,
    CandidateSignature,
    SemanticEvidenceSignature,
)
from src.pdf_retrieval_v4.candidate_row_bridge import (
    compute_numeric_recall,
    compute_text_coverage,
    metric_compatible,
    period_compatible,
)
from src.pdf_retrieval_v4.semantic_evidence_catalog import SemanticEvidenceCatalog

# Multi-row: require at least 2 rows
MIN_MULTIROW_COUNT = 2

# Combined text coverage threshold for multi-row
MULTIROW_TEXT_COVERAGE_THRESHOLD = 0.35


class MultiRowBridge:
    """Bridge mapper for candidates spanning multiple contiguous rows."""

    def __init__(self, catalog: SemanticEvidenceCatalog) -> None:
        self._catalog = catalog

    def bridge(self, sig: CandidateSignature) -> BridgeResult:
        """Bridge a candidate to a contiguous set of semantic rows."""
        # Get all rows on the same page
        page_rows = self._catalog.get_rows_by_page(sig.document_id, sig.pdf_page)

        # Filter to eligible row types
        eligible = [
            r
            for r in page_rows
            if r.row_type in BRIDGE_ELIGIBLE_ROW_TYPES
            and r.table_id is not None
            and r.row_index is not None
        ]

        if len(eligible) < MIN_MULTIROW_COUNT:
            return BridgeResult(
                candidate_key=sig.candidate_key,
                grade=BridgeGrade.UNMAPPED.value,
                matches=(),
                failure_stage="multirow_required",
                bridge_reasons=("insufficient_eligible_rows",),
            )

        # Group by table_id
        by_table: dict[str, list[SemanticEvidenceSignature]] = {}
        for r in eligible:
            assert r.table_id is not None
            by_table.setdefault(r.table_id, []).append(r)

        best_result: BridgeResult | None = None
        best_score: float = 0.0

        for table_id, rows in by_table.items():
            # Sort by row_index
            rows.sort(key=lambda r: r.row_index or 0)

            # Try all contiguous subsequences of length >= 2
            for start in range(len(rows)):
                for end in range(start + MIN_MULTIROW_COUNT, len(rows) + 1):
                    subset = rows[start:end]

                    # Check contiguity
                    indices = [r.row_index for r in subset if r.row_index is not None]
                    if not indices:
                        continue
                    if indices != list(range(indices[0], indices[0] + len(indices))):
                        continue

                    result = self._score_subset(sig, subset)
                    if (
                        result is not None
                        and result.grade == BridgeGrade.A4_MULTIROW.value
                    ):
                        # Extract score from the first match
                        if result.matches:
                            score = result.matches[0].score
                            if score > best_score:
                                best_score = score
                                best_result = result

        if best_result is not None:
            return best_result

        # Check if multi-row was needed but failed
        return BridgeResult(
            candidate_key=sig.candidate_key,
            grade=BridgeGrade.UNMAPPED.value,
            matches=(),
            failure_stage="multirow_required",
            bridge_reasons=("no_contiguous_multirow_match",),
        )

    def _score_subset(
        self,
        sig: CandidateSignature,
        rows: list[SemanticEvidenceSignature],
    ) -> BridgeResult | None:
        """Score a contiguous subset of rows against the candidate."""
        # Combine all row numbers and text
        combined_nums: set[str] = set()
        combined_text_parts: list[str] = []
        combined_periods: set[str] = set()
        for r in rows:
            combined_nums.update(r.numeric_multiset)
            combined_text_parts.append(r.normalized_text)
            combined_periods.update(r.periods)

        combined_text = " ".join(combined_text_parts)

        # Check numeric recall
        nr = compute_numeric_recall(sig.numeric_multiset, tuple(sorted(combined_nums)))
        if nr < 1.0:
            return None

        # Check text coverage
        tc = compute_text_coverage(sig.text_tokens, combined_text)
        if tc < MULTIROW_TEXT_COVERAGE_THRESHOLD:
            return None

        # Check metric compatibility
        metric_ok = True
        for r in rows:
            mp = self._catalog.get_metric_path_for_row(r.row_id or "")
            if sig.existing_metric_paths:
                mc = metric_compatible(sig.existing_metric_paths[0], mp)
                if not mc:
                    metric_ok = False
                    break

        # Build match
        evidence_ids = tuple(r.evidence_id for r in rows)
        score = (
            nr * 0.4 + tc * 0.3 + (0.2 if metric_ok else 0.0) + 0.1 * (len(rows) / 10.0)
        )

        match = BridgeMatch(
            evidence_id=evidence_ids[0],  # Primary evidence
            evidence_type="multirow",
            grade=BridgeGrade.A4_MULTIROW.value,
            score=score,
            reasons=(
                f"row_count={len(rows)}",
                f"numeric_recall={nr:.3f}",
                f"text_coverage={tc:.3f}",
                f"metric_compatible={metric_ok}",
                f"evidence_ids={','.join(evidence_ids[:3])}{'...' if len(evidence_ids) > 3 else ''}",
            ),
            numeric_recall=nr,
            text_coverage=tc,
            bbox_overlap=0.0,
            metric_compatible=metric_ok,
            period_compatible=period_compatible(
                sig.period_tokens, tuple(sorted(combined_periods))
            ),
        )

        return BridgeResult(
            candidate_key=sig.candidate_key,
            grade=BridgeGrade.A4_MULTIROW.value,
            matches=(match,),
            failure_stage=None,
            bridge_reasons=match.reasons,
        )
