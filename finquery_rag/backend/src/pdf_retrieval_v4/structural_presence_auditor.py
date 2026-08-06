"""Structural presence auditor for Gate 08 R1.2.

Performs five-layer structural existence audit for Gold sources that
are either strict-mapped-not-retrieved (B-class) or structurally-absent
(D-class).  Does NOT modify structures or retrieval — audit only.

Five layers checked per Gold source:
  Layer 1: Target PDF page exists in V4 views
  Layer 2: Target Table or Narrative Block exists
  Layer 3: Target Row exists
  Layer 4: Target Cell/Fact exists
  Layer 5: Can map to Production Candidate

Six failure classes (mutually exclusive):
  S1: Structure exists, Candidate Bridge missing
  S2: Candidate granularity mismatch
  S3: Structure exists, Evidence Unit not emitted
  S4: Native PDF has content, MinerU structure missing
  S5: Narrative or non-table evidence
  S6: Production Candidate granularity or mapping error

Reads ONLY sealed Gate 06 R2 metadata store and production candidate
mapper.  Gold labels are used solely for offline audit.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


@dataclass(frozen=True)
class LayerCheckResult:
    """Result of a single layer check."""

    layer: int
    layer_name: str
    present: bool
    evidence: str = ""


@dataclass(frozen=True)
class StructuralPresenceAudit:
    """Complete five-layer audit result for one Gold source."""

    gold_source_identity: str
    case_id: str
    gold_candidate_key: str
    document_id: str
    pdf_page: int | None

    # Layer results
    layer1_page_present: bool
    layer2_table_present: bool
    layer3_row_present: bool
    layer4_cell_fact_present: bool
    layer5_candidate_bridge: bool

    # Supporting evidence
    page_view_ids: tuple[str, ...]
    matched_table_ids: tuple[str, ...]
    matched_row_ids: tuple[str, ...]
    matched_fact_ids: tuple[str, ...]

    # Classification
    failure_class: str  # S1-S6
    recommended_action: str
    pdf_reprocessing_required: bool
    audit_notes: str

    # Source metadata for debugging
    gold_metric: str | None
    gold_period: str | None
    gold_row_label: str | None
    gold_table_title: str | None
    gold_section: str | None
    gold_evidence_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gold_source_identity": self.gold_source_identity,
            "case_id": self.case_id,
            "gold_candidate_key": self.gold_candidate_key,
            "document_id": self.document_id,
            "pdf_page": self.pdf_page,
            "layer1_page_present": self.layer1_page_present,
            "layer2_table_present": self.layer2_table_present,
            "layer3_row_present": self.layer3_row_present,
            "layer4_cell_fact_present": self.layer4_cell_fact_present,
            "layer5_candidate_bridge": self.layer5_candidate_bridge,
            "mineru_table_present": self.layer2_table_present,
            "v4_row_present": self.layer3_row_present,
            "v4_fact_present": self.layer4_cell_fact_present,
            "candidate_bridge_present": self.layer5_candidate_bridge,
            "page_view_ids": list(self.page_view_ids),
            "matched_table_ids": list(self.matched_table_ids),
            "matched_row_ids": list(self.matched_row_ids),
            "matched_fact_ids": list(self.matched_fact_ids),
            "failure_class": self.failure_class,
            "recommended_action": self.recommended_action,
            "pdf_reprocessing_required": self.pdf_reprocessing_required,
            "audit_notes": self.audit_notes,
            "gold_metric": self.gold_metric,
            "gold_period": self.gold_period,
            "gold_row_label": self.gold_row_label,
            "gold_table_title": self.gold_table_title,
            "gold_section": self.gold_section,
            "gold_evidence_type": self.gold_evidence_type,
        }


class StructuralPresenceAuditor:
    """Audit structural existence of Gold sources in V4 metadata store.

    Reads ONLY the sealed Gate 06 R2 metadata store (SQLite).  No PDF
    reprocessing, no MinerU reruns.
    """

    def __init__(self, metadata_db_path: Path) -> None:
        uri = f"file:{metadata_db_path.absolute().as_posix()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True)
        self._views_by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._tables_by_doc: dict[str, list[dict[str, Any]]] = {}
        self._rows_by_table: dict[str, list[dict[str, Any]]] = {}
        self._facts_by_row: dict[str, list[dict[str, Any]]] = {}
        self._all_views: list[dict[str, Any]] = []
        self._load_metadata()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "StructuralPresenceAuditor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _load_metadata(self) -> None:
        # Load all retrieval views
        rows = self._conn.execute(
            "SELECT retrieval_view_id, evidence_unit_id, unit_type, "
            "retrieval_text, metadata_json FROM retrieval_views"
        ).fetchall()
        for view_id, evidence_id, unit_type, text, metadata_json in rows:
            metadata = json.loads(metadata_json or "{}")
            view = {
                "retrieval_view_id": str(view_id),
                "evidence_unit_id": str(evidence_id),
                "unit_type": str(unit_type),
                "retrieval_text": str(text or ""),
                "metadata": metadata,
            }
            self._all_views.append(view)

            doc_id = str(metadata.get("document_id") or "")
            pages = metadata.get("pdf_pages") or []
            if not isinstance(pages, list):
                pages = [pages]
            for page in pages:
                try:
                    page_int = int(page)
                except (TypeError, ValueError):
                    continue
                if page_int > 0 and doc_id:
                    self._views_by_page.setdefault(
                        (doc_id, page_int), []
                    ).append(view)

        # Load table_rows
        for row_id, logical_table_id, member_json in self._conn.execute(
            "SELECT row_id, logical_table_id, member_view_ids_json "
            "FROM table_rows"
        ).fetchall():
            row = {
                "row_id": str(row_id),
                "logical_table_id": str(logical_table_id),
                "member_view_ids": json.loads(member_json or "[]"),
            }
            self._rows_by_table.setdefault(
                str(logical_table_id), []
            ).append(row)

        # Load row_cells
        for cell_id, row_id, member_json in self._conn.execute(
            "SELECT cell_id, row_id, member_view_ids_json FROM row_cells"
        ).fetchall():
            cell = {
                "cell_id": str(cell_id),
                "row_id": str(row_id),
                "member_view_ids": json.loads(member_json or "[]"),
            }
            self._facts_by_row.setdefault(str(row_id), []).append(cell)

        # Load facts
        for fact_id, cell_id, member_json in self._conn.execute(
            "SELECT fact_id, cell_id, member_view_ids_json FROM facts"
        ).fetchall():
            fact = {
                "fact_id": str(fact_id),
                "cell_id": str(cell_id),
                "member_view_ids": json.loads(member_json or "[]"),
            }
            # Attach to cell's row
            # Find row for this cell
            for row_id, cells in self._facts_by_row.items():
                for existing_cell in cells:
                    if existing_cell["cell_id"] == str(cell_id):
                        if "facts" not in existing_cell:
                            existing_cell["facts"] = []
                        existing_cell["facts"].append(fact)
                        break

        # Build tables index from row logical_table_ids
        table_ids_seen: set[str] = set()
        for row_id, row in self._conn.execute(
            "SELECT row_id, logical_table_id FROM table_rows"
        ).fetchall():
            table_id = str(row)
            if table_id not in table_ids_seen:
                table_ids_seen.add(table_id)
        # Index views by table
        for view in self._all_views:
            metadata = view.get("metadata") or {}
            table_id = str(metadata.get("logical_table_id") or "")
            doc_id = str(metadata.get("document_id") or "")
            if table_id and doc_id:
                self._tables_by_doc.setdefault(doc_id, []).append(view)

    @property
    def total_views(self) -> int:
        return len(self._all_views)

    @property
    def total_tables(self) -> int:
        return sum(len(views) for views in self._tables_by_doc.values())

    def audit_gold_source(
        self,
        *,
        case_id: str,
        source_index: int,
        gold_candidate_key: str,
        gold_document_id: str | None = None,
        gold_page: int | None = None,
        gold_metric: str | None = None,
        gold_period: str | None = None,
        gold_row_label: str | None = None,
        gold_table_title: str | None = None,
        gold_section: str | None = None,
        gold_evidence_type: str | None = None,
        r1_strict_mapped: bool = False,
        r1_matched_view_id: str | None = None,
        r1_matched_unit_type: str | None = None,
    ) -> StructuralPresenceAudit:
        """Audit a single Gold source through five layers."""
        gold_source_identity = f"{case_id}#{source_index}"
        doc_id = str(gold_document_id or "")
        page = gold_page

        # Layer 1: Page present
        page_views: list[dict[str, Any]] = []
        if doc_id and page is not None and page > 0:
            page_views = list(
                self._views_by_page.get((doc_id, page), [])
            )
        layer1 = len(page_views) > 0
        page_view_ids = tuple(
            v["retrieval_view_id"] for v in page_views
        )

        # Layer 2: Table or Narrative Block present
        # Check if any view on this page has table/row/cell/fact type
        table_views = [
            v for v in page_views
            if v["unit_type"] in (
                "table", "row", "cell",
                "atomic_fact", "comparison_fact", "bucket_fact",
            )
        ]
        narrative_views = [
            v for v in page_views
            if v["unit_type"] in ("section", "narrative", "paragraph")
        ]
        layer2 = len(table_views) > 0 or len(narrative_views) > 0

        # Layer 3: Row present (match by row_label/metric)
        matched_row_ids: list[str] = []
        gold_row_norm = _norm_text(gold_row_label or gold_metric or "")
        if layer2 and gold_row_norm:
            for view in table_views:
                if view["unit_type"] != "row":
                    continue
                metadata = view.get("metadata") or {}
                view_metric = str(metadata.get("metric_path") or "")
                if _norm_text(view_metric) == gold_row_norm:
                    row_id = str(metadata.get("row_id") or "")
                    if row_id:
                        matched_row_ids.append(row_id)
        layer3 = len(matched_row_ids) > 0

        # Layer 4: Cell/Fact present
        matched_fact_ids: list[str] = []
        if layer3 and gold_period:
            gold_period_norm = _norm_text(gold_period)
            for row_id in matched_row_ids:
                cells = self._facts_by_row.get(row_id, [])
                for cell in cells:
                    for fact in cell.get("facts", []):
                        # Check fact metadata for period match
                        fact_member_ids = fact.get("member_view_ids", [])
                        for fid in fact_member_ids:
                            # Look up fact view
                            for view in self._all_views:
                                if view["retrieval_view_id"] == fid:
                                    metadata = view.get("metadata") or {}
                                    view_periods = metadata.get("periods") or []
                                    view_period_str = " ".join(
                                        str(p) for p in view_periods
                                    )
                                    if (
                                        gold_period_norm
                                        and _norm_text(view_period_str)
                                        == gold_period_norm
                                    ):
                                        matched_fact_ids.append(
                                            fact["fact_id"]
                                        )
                                    break
        # Also check R1 matched view for fact presence
        if not matched_fact_ids and r1_matched_view_id:
            matched_fact_ids.append(r1_matched_view_id)
        layer4 = len(matched_fact_ids) > 0

        # Layer 5: Candidate bridge (R1 strict mapping)
        layer5 = r1_strict_mapped

        # Classify failure (S1-S6)
        failure_class, action, reprocess, notes = self._classify(
            layer1=layer1,
            layer2=layer2,
            layer3=layer3,
            layer4=layer4,
            layer5=layer5,
            gold_evidence_type=gold_evidence_type,
            has_table_views=len(table_views) > 0,
            has_narrative_views=len(narrative_views) > 0,
            r1_strict_mapped=r1_strict_mapped,
            gold_row_label=gold_row_label,
        )

        # Collect matched table IDs
        matched_table_ids = tuple(
            str(v.get("metadata", {}).get("logical_table_id") or "")
            for v in table_views
            if v.get("metadata", {}).get("logical_table_id")
        )

        return StructuralPresenceAudit(
            gold_source_identity=gold_source_identity,
            case_id=case_id,
            gold_candidate_key=gold_candidate_key,
            document_id=doc_id,
            pdf_page=page,
            layer1_page_present=layer1,
            layer2_table_present=layer2,
            layer3_row_present=layer3,
            layer4_cell_fact_present=layer4,
            layer5_candidate_bridge=layer5,
            page_view_ids=page_view_ids,
            matched_table_ids=matched_table_ids,
            matched_row_ids=tuple(matched_row_ids),
            matched_fact_ids=tuple(matched_fact_ids),
            failure_class=failure_class,
            recommended_action=action,
            pdf_reprocessing_required=reprocess,
            audit_notes=notes,
            gold_metric=gold_metric,
            gold_period=gold_period,
            gold_row_label=gold_row_label,
            gold_table_title=gold_table_title,
            gold_section=gold_section,
            gold_evidence_type=gold_evidence_type,
        )

    def _classify(
        self,
        *,
        layer1: bool,
        layer2: bool,
        layer3: bool,
        layer4: bool,
        layer5: bool,
        gold_evidence_type: str | None,
        has_table_views: bool,
        has_narrative_views: bool,
        r1_strict_mapped: bool,
        gold_row_label: str | None,
    ) -> tuple[str, str, bool, str]:
        """Classify into S1-S6.

        Returns (failure_class, recommended_action,
                 pdf_reprocessing_required, audit_notes).
        """
        # S4: No views on page at all → MinerU structure missing
        if not layer1:
            return (
                "S4_mineru_structure_missing",
                "targeted_pdf_reprocessing",
                True,
                "No V4 views found on target PDF page; MinerU may have "
                "missed the page or table.",
            )

        # S5: Narrative evidence (non-table)
        ev_type = str(gold_evidence_type or "").lower()
        is_narrative = (
            ev_type in ("narrative", "text", "paragraph", "md")
            or (has_narrative_views and not has_table_views)
        )
        if is_narrative:
            return (
                "S5_narrative_evidence",
                "candidate_aligned_narrative_view",
                False,
                "Gold evidence type is narrative/non-table; current "
                "Table/Fact pipeline does not apply.",
            )

        # S1: Structure exists (table/row/fact) but no candidate bridge
        if layer2 and not layer5:
            if layer3 and layer4:
                return (
                    "S1_bridge_missing",
                    "candidate_bridge_expansion",
                    False,
                    "Table, row, and fact structures exist but no "
                    "Production Candidate bridge mapping.",
                )
            if layer3 and not layer4:
                return (
                    "S3_evidence_unit_not_emitted",
                    "evidence_unit_expansion",
                    False,
                    "Row exists but no fact/cell Evidence Unit was "
                    "emitted for this period/metric.",
                )
            # Table exists but row not found
            return (
                "S2_candidate_granularity_mismatch",
                "candidate_block_matrix_view",
                False,
                "Table exists but target row not found; possible "
                "granularity mismatch (multi-row candidate or "
                "header/row merge).",
            )

        # S6: Candidate bridge exists (r1_strict_mapped) but still
        # not retrieved — candidate granularity or mapping error
        if layer5 and r1_strict_mapped:
            return (
                "S6_candidate_mapping_error",
                "candidate_granularity_or_gold_identity_audit",
                False,
                "Strict candidate bridge exists but candidate was not "
                "retrieved; possible candidate granularity or Gold "
                "identity contract issue.",
            )

        # S3: Table exists but no row or fact matched
        if layer2 and not layer3:
            return (
                "S3_evidence_unit_not_emitted",
                "evidence_unit_expansion",
                False,
                "Table/narrative block exists but no matching row "
                "Evidence Unit was emitted.",
            )

        # Fallback: should not reach here
        return (
            "S4_mineru_structure_missing",
            "targeted_pdf_reprocessing",
            True,
            "Unclassified structural absence; defaulting to S4 for "
            "investigation.",
        )
