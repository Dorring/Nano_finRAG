"""Gate 05 R5 — Candidate Structured View builder.

Builds a single ``CandidateStructuredView`` for each Grade-A candidate,
aggregating multiple facts (Atomic, Comparison, Bucket, RowMatrix) into
one view per candidate.

Key constraint: one candidate → one structured view.
Fields are stored separately (not pre-joined into a single string).
"""

from __future__ import annotations

from typing import Any

from src.pdf_retrieval_v4.candidate_bridge_models import (
    BridgeGrade,
    BridgeResult,
    CandidateSignature,
    CandidateStructuredView,
    SemanticEvidenceSignature,
)
from src.pdf_retrieval_v4.semantic_evidence_catalog import SemanticEvidenceCatalog


class StructuredViewBuilder:
    """Builds CandidateStructuredView from bridge results."""

    def __init__(self, catalog: SemanticEvidenceCatalog) -> None:
        self._catalog = catalog

    def build_view(
        self,
        sig: CandidateSignature,
        result: BridgeResult,
    ) -> CandidateStructuredView | None:
        """Build a structured view for a Grade-A bridge result.

        Returns None if the bridge grade is not A*.
        """
        if not BridgeGrade.is_grade_a(result.grade):
            return None

        # Collect all evidence IDs from matches
        evidence_ids: list[str] = []
        row_ids: set[str] = set()
        table_ids: set[str] = set()
        metric_paths: set[str] = set()
        periods: set[str] = set()
        segments: set[str] = set()
        buckets: set[str] = set()
        facts: list[dict[str, Any]] = []
        source_tracebacks: list[dict[str, Any]] = []
        table_title: str | None = None
        section_paths: set[str] = set()

        for match in result.matches:
            evidence_ids.append(match.evidence_id)

            # Get the evidence from catalog
            ev = self._catalog.get_by_evidence_id(match.evidence_id)
            if ev is None:
                # For multirow, evidence_id is the first row; try others
                continue

            # Collect structural IDs
            if ev.row_id:
                row_ids.add(ev.row_id)
            if ev.table_id:
                table_ids.add(ev.table_id)
            metric_paths.update(ev.metric_paths)
            periods.update(ev.periods)
            segments.update(ev.segments)
            buckets.update(ev.buckets)
            source_tracebacks.append(ev.source_traceback)

            # Build fact entry based on evidence type
            fact = self._build_fact(sig, ev)
            if fact:
                facts.append(fact)

            # Extract table title from logical table
            if ev.evidence_type == "logical_table" and ev.raw_text:
                table_title = ev.raw_text

            # Extract section path from narrative
            if ev.evidence_type == "narrative_evidence":
                sp = ev.source_traceback.get("section_path") or ""
                if sp:
                    section_paths.add(sp)

        # For multirow: also collect evidence from all rows in the match
        if result.grade == BridgeGrade.A4_MULTIROW.value and result.matches:
            # The match reasons contain evidence_ids
            for reason in result.matches[0].reasons:
                if reason.startswith("evidence_ids="):
                    ids_str = reason.split("=", 1)[1].rstrip("...")
                    for eid in ids_str.split(","):
                        eid = eid.strip()
                        if eid and eid not in evidence_ids:
                            evidence_ids.append(eid)
                            ev = self._catalog.get_by_evidence_id(eid)
                            if ev:
                                if ev.row_id:
                                    row_ids.add(ev.row_id)
                                if ev.table_id:
                                    table_ids.add(ev.table_id)
                                metric_paths.update(ev.metric_paths)
                                periods.update(ev.periods)
                                source_tracebacks.append(ev.source_traceback)
                                fact = self._build_fact(sig, ev)
                                if fact:
                                    facts.append(fact)

        # Also gather facts from all evidence on the same rows
        for row_id in row_ids:
            row_evidence = self._catalog.get_by_row(row_id)
            for ev in row_evidence:
                if ev.evidence_type in (
                    "atomic_fact",
                    "comparison_fact",
                    "bucket_fact",
                    "row_matrix",
                ):
                    metric_paths.update(ev.metric_paths)
                    periods.update(ev.periods)
                    segments.update(ev.segments)
                    buckets.update(ev.buckets)
                    fact = self._build_fact(sig, ev)
                    if fact and fact not in facts:
                        facts.append(fact)

        # Build row matrix if multiple periods
        row_matrix: dict[str, Any] | None = None
        if len(periods) >= 2 and row_ids:
            row_matrix = self._build_row_matrix(row_ids, periods)

        # Determine candidate type
        candidate_type = sig.block_type

        return CandidateStructuredView(
            candidate_key=sig.candidate_key,
            document_id=sig.document_id,
            pdf_page=sig.pdf_page,
            candidate_type=candidate_type,
            raw_content=sig.raw_content,
            section_path=tuple(sorted(section_paths)),
            table_title=table_title,
            metric_paths=tuple(sorted(metric_paths)),
            periods=tuple(sorted(periods)),
            facts=tuple(facts),
            segments=tuple(sorted(segments)),
            buckets=tuple(sorted(buckets)),
            row_matrix=row_matrix,
            semantic_evidence_ids=tuple(evidence_ids),
            row_ids=tuple(sorted(row_ids)),
            bridge_grade=result.grade,
            bridge_reasons=result.bridge_reasons,
            source_traceback=tuple(source_tracebacks),
        )

    def _build_fact(
        self,
        sig: CandidateSignature,
        ev: SemanticEvidenceSignature,
    ) -> dict[str, Any] | None:
        """Build a fact dict from an evidence signature."""
        if ev.evidence_type == "atomic_fact":
            return {
                "type": "atomic",
                "metric": ev.metric_paths[0] if ev.metric_paths else "",
                "period": ev.periods[0] if ev.periods else "",
                "value": ev.raw_values[0] if ev.raw_values else "",
                "scale": ev.source_traceback.get("scale"),
                "evidence_id": ev.evidence_id,
            }
        elif ev.evidence_type == "comparison_fact":
            return {
                "type": "comparison",
                "metric": ev.metric_paths[0] if ev.metric_paths else "",
                "periods": list(ev.periods),
                "evidence_id": ev.evidence_id,
            }
        elif ev.evidence_type == "bucket_fact":
            return {
                "type": "bucket",
                "metric": ev.metric_paths[0] if ev.metric_paths else "",
                "bucket": ev.buckets[0] if ev.buckets else "",
                "value": ev.raw_values[0] if ev.raw_values else "",
                "evidence_id": ev.evidence_id,
            }
        elif ev.evidence_type == "row_matrix":
            return {
                "type": "row_matrix",
                "metric": ev.metric_paths[0] if ev.metric_paths else "",
                "periods": list(ev.periods),
                "values": list(ev.raw_values),
                "evidence_id": ev.evidence_id,
            }
        elif ev.evidence_type == "narrative_evidence":
            return {
                "type": "narrative",
                "text": ev.raw_text[:500],
                "evidence_id": ev.evidence_id,
            }
        return None

    def _build_row_matrix(
        self,
        row_ids: set[str],
        periods: set[str],
    ) -> dict[str, Any]:
        """Build a row matrix summary from row IDs and periods."""
        return {
            "row_count": len(row_ids),
            "period_count": len(periods),
            "periods": sorted(periods),
            "row_ids": sorted(row_ids),
        }

    def build_all_views(
        self,
        signatures: list[CandidateSignature],
        results: list[BridgeResult],
    ) -> list[CandidateStructuredView]:
        """Build structured views for all Grade-A candidates."""
        views: list[CandidateStructuredView] = []
        sig_by_key = {s.candidate_key: s for s in signatures}
        for result in results:
            sig = sig_by_key.get(result.candidate_key)
            if sig is None:
                continue
            view = self.build_view(sig, result)
            if view is not None:
                views.append(view)
        return views
