"""Offline Gold-to-structure mapping for Gate 08 R1 evaluation contract repair.

Maps Gold Source identities to V4 structural universe elements
(logical_table_id, row_id, cell_id, fact_id) using the sealed Gate 06 R2
metadata store and the production candidate mapper.

Gold is used ONLY for offline scoring — it never enters runtime code.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v4_gate08_pool import ProductionCandidateMapper


def _norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


@dataclass(frozen=True)
class GoldStructuralMatch:
    """Result of mapping a Gold source to the V4 structural universe."""

    gold_candidate_key: str
    case_id: str
    source_index: int

    logical_table_id: str | None = None
    row_id: str | None = None
    cell_id: str | None = None
    fact_id: str | None = None

    matched_retrieval_view_id: str | None = None
    matched_unit_type: str | None = None

    mapping_method: str = "unresolved"
    mapping_confidence: float = 0.0

    gold_document_id: str | None = None
    gold_page: int | None = None
    gold_metric: str | None = None
    gold_period: str | None = None
    gold_row_label: str | None = None

    @property
    def in_structured_universe(self) -> bool:
        return self.matched_retrieval_view_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gold_candidate_key": self.gold_candidate_key,
            "case_id": self.case_id,
            "source_index": self.source_index,
            "logical_table_id": self.logical_table_id,
            "row_id": self.row_id,
            "cell_id": self.cell_id,
            "fact_id": self.fact_id,
            "matched_retrieval_view_id": self.matched_retrieval_view_id,
            "matched_unit_type": self.matched_unit_type,
            "mapping_method": self.mapping_method,
            "mapping_confidence": self.mapping_confidence,
            "gold_document_id": self.gold_document_id,
            "gold_page": self.gold_page,
            "gold_metric": self.gold_metric,
            "gold_period": self.gold_period,
            "gold_row_label": self.gold_row_label,
            "in_structured_universe": self.in_structured_universe,
        }


class StructuralGoldMapper:
    """Map Gold sources to V4 structural universe elements.

    Reads ONLY the sealed Gate 06 R2 metadata store (SQLite) and the
    production candidate mapper.  Gold labels are used solely for offline
    scoring — they never enter runtime retrieval code.
    """

    def __init__(
        self,
        metadata_db_path: Path,
        mapper: ProductionCandidateMapper,
    ) -> None:
        self._conn = sqlite3.connect(
            f"file:{Path(metadata_db_path).absolute().as_posix()}?mode=ro",
            uri=True,
        )
        self._mapper = mapper
        self._views_by_type: dict[str, list[dict[str, Any]]] = {}
        self._views_by_candidate_key: dict[str, list[dict[str, Any]]] = {}
        self._load_views()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "StructuralGoldMapper":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _load_views(self) -> None:
        rows = self._conn.execute(
            "SELECT retrieval_view_id, evidence_unit_id, unit_type, "
            "retrieval_text, metadata_json FROM retrieval_views"
        )
        for view_id, evidence_id, unit_type, text, metadata_json in rows:
            typ = str(unit_type)
            metadata = json.loads(metadata_json or "{}")
            view = {
                "retrieval_view_id": str(view_id),
                "evidence_unit_id": str(evidence_id),
                "unit_type": typ,
                "retrieval_text": str(text or ""),
                "metadata": metadata,
            }
            self._views_by_type.setdefault(typ, []).append(view)

            mapping = self._mapper.map_view(view)
            status = str(mapping.get("strict_candidate_status") or "unmapped")
            candidate_key = mapping.get("candidate_key")
            if status == "unique" and candidate_key:
                self._views_by_candidate_key.setdefault(
                    str(candidate_key), []
                ).append(view)

    @property
    def view_counts(self) -> dict[str, int]:
        return {typ: len(views) for typ, views in self._views_by_type.items()}

    @property
    def total_view_count(self) -> int:
        return sum(len(views) for views in self._views_by_type.values())

    @property
    def mapped_candidate_count(self) -> int:
        return len(self._views_by_candidate_key)

    def views_for_candidate(self, candidate_key: str) -> list[dict[str, Any]]:
        """Return all V4 structural views strictly mapped to a candidate key."""
        return list(self._views_by_candidate_key.get(str(candidate_key), []))

    def universe_candidate_map_records(self) -> list[dict[str, Any]]:
        """Generate universe-candidate-map records for all views."""
        records: list[dict[str, Any]] = []
        for typ in (
            "section",
            "table",
            "row",
            "cell",
            "atomic_fact",
            "comparison_fact",
            "bucket_fact",
        ):
            for view in self._views_by_type.get(typ, []):
                metadata = view.get("metadata") or {}
                mapping = self._mapper.map_view(view)
                status = str(
                    mapping.get("strict_candidate_status") or "unmapped"
                )
                candidate_key = mapping.get("candidate_key")
                direct_identities = (
                    [str(candidate_key)]
                    if status == "unique" and candidate_key
                    else []
                )
                records.append(
                    {
                        "retrieval_view_id": view["retrieval_view_id"],
                        "unit_type": typ,
                        "document_id": str(
                            metadata.get("document_id") or ""
                        ),
                        "logical_table_id": str(
                            metadata.get("logical_table_id") or ""
                        ),
                        "row_id": str(metadata.get("row_id") or ""),
                        "cell_id": str(metadata.get("cell_id") or ""),
                        "fact_id": str(metadata.get("fact_id") or ""),
                        "direct_original_candidate_identities": direct_identities,
                        "bridge_candidate_identities": [],
                        "bridge_status": status,
                    }
                )
        return records

    def map_gold_source(
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
        gold_evidence_id: str | None = None,
    ) -> GoldStructuralMatch:
        """Map a single Gold source to the V4 structural universe.

        Priority:
            1. Direct candidate_key match in universe-candidate-map
            2. Document + Page + Row text signature match (row views)
            3. Document + Page + Metric + Period match (fact views)
            4. Document + Page only (weaker match)
            5. unresolved
        """
        base = GoldStructuralMatch(
            gold_candidate_key=gold_candidate_key,
            case_id=case_id,
            source_index=source_index,
            gold_document_id=gold_document_id,
            gold_page=gold_page,
            gold_metric=gold_metric,
            gold_period=gold_period,
            gold_row_label=gold_row_label,
        )

        # Priority 1: Direct candidate_key match
        views = self._views_by_candidate_key.get(gold_candidate_key, [])
        if views:
            view = views[0]
            metadata = view.get("metadata") or {}
            return GoldStructuralMatch(
                gold_candidate_key=gold_candidate_key,
                case_id=case_id,
                source_index=source_index,
                logical_table_id=(
                    str(metadata.get("logical_table_id") or "") or None
                ),
                row_id=str(metadata.get("row_id") or "") or None,
                cell_id=str(metadata.get("cell_id") or "") or None,
                fact_id=str(metadata.get("fact_id") or "") or None,
                matched_retrieval_view_id=view["retrieval_view_id"],
                matched_unit_type=view["unit_type"],
                mapping_method="direct_candidate_key",
                mapping_confidence=1.0,
                gold_document_id=gold_document_id,
                gold_page=gold_page,
                gold_metric=gold_metric,
                gold_period=gold_period,
                gold_row_label=gold_row_label,
            )

        # Priority 2: Document + Page + Row text signature
        if gold_document_id and gold_page is not None and gold_row_label:
            gold_row_norm = _norm_text(gold_row_label)
            if gold_row_norm:
                for view in self._views_by_type.get("row", []):
                    metadata = view.get("metadata") or {}
                    if (
                        str(metadata.get("document_id") or "")
                        != gold_document_id
                    ):
                        continue
                    pages = metadata.get("pdf_pages") or []
                    if gold_page not in pages:
                        continue
                    view_metric = str(metadata.get("metric_path") or "")
                    if _norm_text(view_metric) == gold_row_norm:
                        return GoldStructuralMatch(
                            gold_candidate_key=gold_candidate_key,
                            case_id=case_id,
                            source_index=source_index,
                            logical_table_id=(
                                str(metadata.get("logical_table_id") or "")
                                or None
                            ),
                            row_id=(
                                str(metadata.get("row_id") or "") or None
                            ),
                            cell_id=None,
                            fact_id=None,
                            matched_retrieval_view_id=view[
                                "retrieval_view_id"
                            ],
                            matched_unit_type="row",
                            mapping_method="document_page_row_text",
                            mapping_confidence=0.85,
                            gold_document_id=gold_document_id,
                            gold_page=gold_page,
                            gold_metric=gold_metric,
                            gold_period=gold_period,
                            gold_row_label=gold_row_label,
                        )

        # Priority 3: Document + Page + Metric + Period
        if gold_document_id and gold_page is not None:
            for fact_type in (
                "atomic_fact",
                "comparison_fact",
                "bucket_fact",
            ):
                for view in self._views_by_type.get(fact_type, []):
                    metadata = view.get("metadata") or {}
                    if (
                        str(metadata.get("document_id") or "")
                        != gold_document_id
                    ):
                        continue
                    pages = metadata.get("pdf_pages") or []
                    if gold_page not in pages:
                        continue
                    if gold_period:
                        view_periods = metadata.get("periods") or []
                        if gold_period not in view_periods:
                            continue
                    if gold_metric:
                        view_metric = str(
                            metadata.get("metric_path") or ""
                        )
                        if (
                            _norm_text(gold_metric)
                            not in _norm_text(view_metric)
                        ):
                            continue
                    return GoldStructuralMatch(
                        gold_candidate_key=gold_candidate_key,
                        case_id=case_id,
                        source_index=source_index,
                        logical_table_id=(
                            str(metadata.get("logical_table_id") or "")
                            or None
                        ),
                        row_id=(
                            str(metadata.get("row_id") or "") or None
                        ),
                        cell_id=(
                            str(metadata.get("cell_id") or "") or None
                        ),
                        fact_id=(
                            str(metadata.get("fact_id") or "") or None
                        ),
                        matched_retrieval_view_id=view["retrieval_view_id"],
                        matched_unit_type=fact_type,
                        mapping_method="document_page_metric_period",
                        mapping_confidence=0.75,
                        gold_document_id=gold_document_id,
                        gold_page=gold_page,
                        gold_metric=gold_metric,
                        gold_period=gold_period,
                        gold_row_label=gold_row_label,
                    )

        # Priority 4: Document + Page only (weaker match, table-level)
        if gold_document_id and gold_page is not None:
            for typ in ("row", "atomic_fact", "comparison_fact", "bucket_fact"):
                for view in self._views_by_type.get(typ, []):
                    metadata = view.get("metadata") or {}
                    if (
                        str(metadata.get("document_id") or "")
                        != gold_document_id
                    ):
                        continue
                    pages = metadata.get("pdf_pages") or []
                    if gold_page in pages:
                        return GoldStructuralMatch(
                            gold_candidate_key=gold_candidate_key,
                            case_id=case_id,
                            source_index=source_index,
                            logical_table_id=(
                                str(metadata.get("logical_table_id") or "")
                                or None
                            ),
                            row_id=(
                                str(metadata.get("row_id") or "") or None
                            ),
                            cell_id=(
                                str(metadata.get("cell_id") or "") or None
                            ),
                            fact_id=(
                                str(metadata.get("fact_id") or "") or None
                            ),
                            matched_retrieval_view_id=view[
                                "retrieval_view_id"
                            ],
                            matched_unit_type=typ,
                            mapping_method="document_page_only",
                            mapping_confidence=0.50,
                            gold_document_id=gold_document_id,
                            gold_page=gold_page,
                            gold_metric=gold_metric,
                            gold_period=gold_period,
                            gold_row_label=gold_row_label,
                        )

        return base
