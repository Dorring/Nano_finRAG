"""Gate 05 R5 — Table Block Candidate Bridge.

Handles ``table`` type Production Candidates (whole table blocks).

Matches a table candidate to a LogicalTable, then extracts relevant
Row/Fact set. The Structured View binds to ``candidate_key`` — it does
NOT expand to all candidates within the table.

Grade: A2_bbox_signature (if bbox match) or A3_row_signature (signature only)

Key constraint: one table match does NOT mark all same-page candidates as Grade-A.
"""

from __future__ import annotations


from src.pdf_retrieval_v4.candidate_bridge_models import (
    BRIDGE_ELIGIBLE_ROW_TYPES,
    BridgeGrade,
    BridgeMatch,
    BridgeResult,
    CandidateSignature,
)
from src.pdf_retrieval_v4.candidate_row_bridge import (
    compute_bbox_coverage,
    compute_numeric_recall,
    compute_text_coverage,
    metric_compatible,
    BBOX_COVERAGE_THRESHOLD,
    SCORE_GAP_THRESHOLD,
    TEXT_COVERAGE_MODERATE,
)
from src.pdf_retrieval_v4.semantic_evidence_catalog import SemanticEvidenceCatalog


class TableBridge:
    """Bridge mapper for table block type Production Candidates."""

    def __init__(self, catalog: SemanticEvidenceCatalog) -> None:
        self._catalog = catalog

    def bridge(self, sig: CandidateSignature) -> BridgeResult:
        """Bridge a table block candidate to a logical table + its rows."""
        # Get logical tables on the same page
        page_evidence = self._catalog.get_by_page(sig.document_id, sig.pdf_page)
        logical_tables = [
            e for e in page_evidence if e.evidence_type == "logical_table"
        ]

        if not logical_tables:
            return BridgeResult(
                candidate_key=sig.candidate_key,
                grade=BridgeGrade.UNMAPPED.value,
                matches=(),
                failure_stage="candidate_text_signature_mismatch",
                bridge_reasons=("no_logical_tables_on_page",),
            )

        matches: list[tuple[BridgeMatch, str]] = []  # (match, table_id)

        for lt in logical_tables:
            # Get all rows for this table
            table_rows = self._catalog.get_rows_by_page(sig.document_id, sig.pdf_page)
            table_rows = [
                r
                for r in table_rows
                if r.table_id == lt.table_id and r.row_type in BRIDGE_ELIGIBLE_ROW_TYPES
            ]

            if not table_rows:
                continue

            # Combine all row numbers for the table
            combined_nums: set[str] = set()
            combined_text_parts: list[str] = []
            for r in table_rows:
                combined_nums.update(r.numeric_multiset)
                combined_text_parts.append(r.normalized_text)

            combined_text = " ".join(combined_text_parts)

            # Score the match
            nr = compute_numeric_recall(
                sig.numeric_multiset, tuple(sorted(combined_nums))
            )
            tc = compute_text_coverage(sig.text_tokens, combined_text)
            bbox_cov = compute_bbox_coverage((), lt.bbox) if lt.bbox else 0.0

            # For table blocks: require at least moderate text coverage
            if tc < TEXT_COVERAGE_MODERATE and nr < 0.5:
                continue

            # Check metric compatibility with any row's metric
            metric_ok = True
            for r in table_rows[:5]:  # Check first 5 rows
                mp = self._catalog.get_metric_path_for_row(r.row_id or "")
                if sig.existing_metric_paths:
                    mc = metric_compatible(sig.existing_metric_paths[0], mp)
                    if not mc:
                        metric_ok = False
                        break

            # Determine grade
            if bbox_cov >= BBOX_COVERAGE_THRESHOLD and nr >= 0.8:
                grade = BridgeGrade.A2_BBOX_SIGNATURE.value
            else:
                grade = BridgeGrade.A3_ROW_SIGNATURE.value

            score = (
                nr * 0.35
                + tc * 0.3
                + bbox_cov * 0.15
                + (0.15 if metric_ok else 0.0)
                + 0.05
            )

            match = BridgeMatch(
                evidence_id=lt.evidence_id,
                evidence_type=lt.evidence_type,
                grade=grade,
                score=score,
                reasons=(
                    f"table_id={lt.table_id[:40]}...",
                    f"row_count={len(table_rows)}",
                    f"numeric_recall={nr:.3f}",
                    f"text_coverage={tc:.3f}",
                    f"bbox_coverage={bbox_cov:.3f}",
                    f"metric_compatible={metric_ok}",
                ),
                numeric_recall=nr,
                text_coverage=tc,
                bbox_overlap=bbox_cov,
                metric_compatible=metric_ok,
                period_compatible=True,
            )
            matches.append((match, lt.table_id or ""))

        if not matches:
            return BridgeResult(
                candidate_key=sig.candidate_key,
                grade=BridgeGrade.UNMAPPED.value,
                matches=(),
                failure_stage="candidate_text_signature_mismatch",
                bridge_reasons=("no_table_match_above_threshold",),
            )

        # Sort by score
        matches.sort(key=lambda x: x[0].score, reverse=True)

        # Check uniqueness
        if len(matches) >= 2:
            gap = matches[0][0].score - matches[1][0].score
            if gap < SCORE_GAP_THRESHOLD:
                # Ambiguous
                return BridgeResult(
                    candidate_key=sig.candidate_key,
                    grade=BridgeGrade.B_AMBIGUOUS.value,
                    matches=tuple(m for m, _ in matches[:5]),
                    failure_stage="multiple_equal_matches",
                    bridge_reasons=(
                        f"top_score={matches[0][0].score:.3f}",
                        f"second_score={matches[1][0].score:.3f}",
                        "score_gap_below_threshold",
                    ),
                )

        best_match, best_table_id = matches[0]
        return BridgeResult(
            candidate_key=sig.candidate_key,
            grade=best_match.grade,
            matches=(best_match,),
            failure_stage=None,
            bridge_reasons=best_match.reasons,
        )
