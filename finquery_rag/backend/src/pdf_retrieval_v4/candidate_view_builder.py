"""Build candidate-aligned dual views from Production Candidate store + V4 structural universe.

For each Production Candidate:
  1. Generate Raw View from candidate content (issuer, document, page, raw text)
  2. If Grade-A structural mapping exists, generate Structured View aggregating
     metric_paths, periods, facts, temporal types from mapped V4 views

This runs over the FULL Production Candidate Universe — it does not know which
candidates are Gold-related.  No Gold/Question/Expected Value is read.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.pdf_retrieval_v4.candidate_aligned_view import (
    CandidateAlignedView,
    CandidateViewPair,
    make_raw_view_id,
    make_structured_view_id,
)
from src.pdf_retrieval_v4.structural_gold_mapper import StructuralGoldMapper


def _format_raw_view_text(candidate: dict[str, Any]) -> str:
    """Build retrieval text for Raw View."""
    metadata = candidate.get("metadata") or {}
    issuer = str(metadata.get("issuer") or "")
    document_id = str(candidate.get("document_id") or "")
    page = candidate.get("page")
    block_type = str(candidate.get("block_type") or "text")
    content = str(candidate.get("content") or "")

    parts = [
        f"Issuer: {issuer}" if issuer else "",
        f"Document: {document_id}" if document_id else "",
        f"Page: {page}" if page is not None else "",
        f"Block Type: {block_type}" if block_type else "",
        "",
        "Source:",
        content,
    ]
    return "\n".join(p for p in parts if p is not None)


def _format_structured_view_text(
    candidate: dict[str, Any],
    mapped_views: list[dict[str, Any]],
) -> str:
    """Build retrieval text for Structured View from V4 structural metadata."""
    metadata = candidate.get("metadata") or {}
    issuer = str(metadata.get("issuer") or "")
    document_id = str(candidate.get("document_id") or "")
    page = candidate.get("page")

    # Aggregate structural fields from mapped V4 views
    metric_paths: set[str] = set()
    periods: set[str] = set()
    temporal_types: set[str] = set()
    table_titles: set[str] = set()
    section_paths: set[str] = set()
    statements: set[str] = set()
    facts: list[str] = []

    for view in mapped_views:
        vm = view.get("metadata") or {}
        mp = vm.get("metric_path")
        if mp:
            metric_paths.add(str(mp))
        for p in vm.get("periods") or []:
            periods.add(str(p))
        tb = vm.get("temporal_binding") or {}
        if isinstance(tb, dict):
            kind = tb.get("kind")
            if kind:
                temporal_types.add(str(kind))
            for p_field in ("period", "base_period", "current_period", "reporting_period"):
                pv = tb.get(p_field)
                if pv:
                    periods.add(str(pv))
        tt = vm.get("table_title")
        if tt:
            table_titles.add(str(tt))
        sp = vm.get("section_path")
        if sp:
            section_paths.add(str(sp))
        st = vm.get("statement")
        if st:
            statements.add(str(st))
        # Build fact lines
        metric = str(mp or vm.get("metric") or "")
        period = ""
        if isinstance(tb, dict):
            period = str(tb.get("period") or tb.get("reporting_period") or "")
        raw_value = str(vm.get("raw_value") or vm.get("value") or "")
        scale = str(vm.get("scale") or vm.get("currency") or "")
        if metric or raw_value:
            facts.append(f"{metric} | {period} | {raw_value} | {scale}")

    parts: list[str] = []
    if issuer:
        parts.append(f"Issuer: {issuer}")
    if statements:
        parts.append(f"Statement: {', '.join(sorted(statements))}")
    if section_paths:
        parts.append(f"Section: {', '.join(sorted(section_paths))}")
    if table_titles:
        parts.append(f"Table: {', '.join(sorted(table_titles))}")
    if document_id:
        parts.append(f"Document: {document_id}")
    if page is not None:
        parts.append(f"Page: {page}")
    if metric_paths:
        parts.append("")
        parts.append("Metric Paths:")
        parts.append("\n".join(f"  {m}" for m in sorted(metric_paths)))
    if periods:
        parts.append("")
        parts.append("Periods:")
        parts.append("\n".join(f"  {p}" for p in sorted(periods)))
    if facts:
        parts.append("")
        parts.append("Facts:")
        parts.append("\n".join(f"  {f}" for f in facts))
    if temporal_types:
        parts.append("")
        parts.append("Temporal Types:")
        parts.append(", ".join(sorted(temporal_types)))

    # Always include raw content as fallback for lexical matching
    parts.append("")
    parts.append("Source:")
    parts.append(str(candidate.get("content") or ""))

    return "\n".join(parts)


def _determine_bridge_grade(
    mapped_views: list[dict[str, Any]],
) -> str:
    """Determine bridge grade from mapped V4 views.

    A1 = direct identity (candidate_key stored in view metadata)
    A2 = stable parent row mapping
    A3 = structural signature match
    raw_only = no structural mapping
    """
    if not mapped_views:
        return "raw_only"
    # Check if any view has direct candidate identity
    for view in mapped_views:
        vm = view.get("metadata") or {}
        if vm.get("original_candidate_identity") or vm.get("candidate_key"):
            return "A1"
    # Check for row-level mapping
    for view in mapped_views:
        if str(view.get("unit_type") or "") == "row":
            return "A2"
    # Default to A3 for fact/cell level
    return "A3"


def _aggregate_structural_ids(
    mapped_views: list[dict[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Aggregate logical_table_ids, row_ids, fact_ids from mapped views."""
    table_ids: set[str] = set()
    row_ids: set[str] = set()
    fact_ids: set[str] = set()
    metric_paths: set[str] = set()
    periods: set[str] = set()
    temporal_types: set[str] = set()

    for view in mapped_views:
        vm = view.get("metadata") or {}
        if vm.get("logical_table_id"):
            table_ids.add(str(vm["logical_table_id"]))
        if vm.get("row_id"):
            row_ids.add(str(vm["row_id"]))
        if vm.get("fact_id"):
            fact_ids.add(str(vm["fact_id"]))
        if vm.get("metric_path"):
            metric_paths.add(str(vm["metric_path"]))
        for p in vm.get("periods") or []:
            periods.add(str(p))
        tb = vm.get("temporal_binding") or {}
        if isinstance(tb, dict) and tb.get("kind"):
            temporal_types.add(str(tb["kind"]))

    return {
        "logical_table_ids": tuple(sorted(table_ids)),
        "row_ids": tuple(sorted(row_ids)),
        "fact_ids": tuple(sorted(fact_ids)),
        "metric_paths": tuple(sorted(metric_paths)),
        "periods": tuple(sorted(periods)),
        "temporal_types": tuple(sorted(temporal_types)),
    }


class CandidateViewBuilder:
    """Build candidate-aligned dual views from production candidates + V4 structural universe."""

    def __init__(
        self,
        mapper: Any,  # ProductionCandidateMapper
        gold_mapper: StructuralGoldMapper,
    ) -> None:
        self._mapper = mapper
        self._gold_mapper = gold_mapper

    def build_all(self) -> list[CandidateViewPair]:
        """Build view pairs for ALL production candidates."""
        pairs: list[CandidateViewPair] = []
        for candidate_key, candidate in self._mapper.by_key.items():
            pair = self._build_one(candidate_key, candidate)
            pairs.append(pair)
        return pairs

    def _build_one(
        self, candidate_key: str, candidate: dict[str, Any]
    ) -> CandidateViewPair:
        # Get V4 structural views mapped to this candidate
        mapped_views = self._gold_mapper.views_for_candidate(candidate_key)

        # Determine bridge grade
        bridge_grade = _determine_bridge_grade(mapped_views)

        # Aggregate structural IDs
        structural_ids = _aggregate_structural_ids(mapped_views)

        # Build Raw View
        raw_view = CandidateAlignedView(
            candidate_key=candidate_key,
            view_type="raw",
            view_id=make_raw_view_id(candidate_key),
            retrieval_text=_format_raw_view_text(candidate),
            document_id=str(candidate.get("document_id") or ""),
            pdf_page=candidate.get("page"),
            logical_table_ids=structural_ids["logical_table_ids"],
            row_ids=structural_ids["row_ids"],
            fact_ids=structural_ids["fact_ids"],
            metric_paths=structural_ids["metric_paths"],
            periods=structural_ids["periods"],
            temporal_types=structural_ids["temporal_types"],
            bridge_grade=bridge_grade,
        )

        # Build Structured View (only if structural mapping exists)
        structured_view: CandidateAlignedView | None = None
        if mapped_views:
            structured_view = CandidateAlignedView(
                candidate_key=candidate_key,
                view_type="structured",
                view_id=make_structured_view_id(candidate_key),
                retrieval_text=_format_structured_view_text(candidate, mapped_views),
                document_id=str(candidate.get("document_id") or ""),
                pdf_page=candidate.get("page"),
                logical_table_ids=structural_ids["logical_table_ids"],
                row_ids=structural_ids["row_ids"],
                fact_ids=structural_ids["fact_ids"],
                metric_paths=structural_ids["metric_paths"],
                periods=structural_ids["periods"],
                temporal_types=structural_ids["temporal_types"],
                bridge_grade=bridge_grade,
            )

        return CandidateViewPair(
            candidate_key=candidate_key,
            raw_view=raw_view,
            structured_view=structured_view,
        )

    def build_stats(self, pairs: list[CandidateViewPair]) -> dict[str, Any]:
        """Compute summary statistics for built view pairs."""
        total = len(pairs)
        with_structured = sum(1 for p in pairs if p.structured_view is not None)
        raw_only = total - with_structured
        grade_counts: dict[str, int] = defaultdict(int)
        for p in pairs:
            grade_counts[p.bridge_grade] += 1
        return {
            "total_candidates": total,
            "with_structured_view": with_structured,
            "raw_only": raw_only,
            "bridge_grade_counts": dict(grade_counts),
            "view_schema_version": "candidate-aligned-v1",
        }
