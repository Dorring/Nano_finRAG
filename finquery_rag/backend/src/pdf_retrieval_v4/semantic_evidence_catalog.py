"""Gate 05 R5 — Semantic Evidence Catalog builder.

Loads all Gate 03 R2 Semantic Graph artifacts and converts them into
``SemanticEvidenceSignature`` objects for bridge matching.

Evidence types loaded:
  - semantic-rows.jsonl       → semantic_row evidence
  - atomic-facts.jsonl        → atomic_fact evidence
  - comparison-facts.jsonl    → comparison_fact evidence
  - bucket-facts.jsonl        → bucket_fact evidence
  - row-matrices.jsonl        → row_matrix evidence
  - narrative-evidence.jsonl  → narrative_evidence evidence
  - logical-tables.jsonl     → logical_table evidence

No Question / Gold / Expected-Value data is read.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from src.pdf_retrieval_v4.candidate_bridge_models import SemanticEvidenceSignature
from src.pdf_retrieval_v4.candidate_signature import (
    _normalize_number,
    extract_numeric_multiset,
    extract_period_tokens,
    normalize_text,
)


# ---------------------------------------------------------------------------
# BBox extraction
# ---------------------------------------------------------------------------


def _extract_bbox(source_traceback: dict[str, Any] | None) -> tuple[float, ...]:
    """Extract bbox from source_traceback."""
    if not source_traceback:
        return ()
    bbox = (
        source_traceback.get("bbox")
        or source_traceback.get("row_bbox")
        or source_traceback.get("table_bbox")
    )
    if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return tuple(float(x) for x in bbox)
    return ()


def _extract_raw_text(
    source_traceback: dict[str, Any] | None, fallback: str = ""
) -> str:
    """Extract raw_text from source_traceback."""
    if source_traceback:
        rt = source_traceback.get("raw_text")
        if rt:
            return str(rt)
    return fallback


# ---------------------------------------------------------------------------
# Semantic Row → Signature
# ---------------------------------------------------------------------------


def _row_to_signature(
    record: dict[str, Any],
    enrichment: dict[str, Any] | None = None,
) -> SemanticEvidenceSignature:
    """Convert a semantic-rows.jsonl record to SemanticEvidenceSignature.

    If enrichment is provided, numeric values, metric paths, and periods
    from associated atomic facts and row matrices are used to enrich
    the signature (the raw_label alone rarely contains numbers).
    """
    source_tb = record.get("source_traceback") or {}
    row_id = str(record.get("row_id") or "")
    document_id = str(record.get("document_id") or "")
    pdf_page = int(record.get("pdf_page") or 0)
    raw_label = str(record.get("raw_label") or "")
    table_fragment_id = str(record.get("table_fragment_id") or "")
    row_type = str(record.get("row_type") or "unknown")
    row_index = record.get("row_index")

    bbox = _extract_bbox(source_tb)
    raw_text = raw_label or _extract_raw_text(source_tb)

    if enrichment:
        numeric = tuple(sorted(enrichment.get("numeric_values", set())))
        metric_paths = tuple(sorted(enrichment.get("metric_paths", set())))
        periods = tuple(sorted(enrichment.get("periods", set())))
        raw_values = tuple(enrichment.get("raw_values", []))
        # Enrich raw_text with values for better text matching
        if raw_values:
            raw_text = raw_text + " | " + " | ".join(raw_values)
    else:
        numeric = extract_numeric_multiset(raw_text)
        metric_paths = ()
        periods = extract_period_tokens(raw_text)
        raw_values = ()

    norm = normalize_text(raw_text)

    return SemanticEvidenceSignature(
        evidence_id=row_id,
        evidence_type="semantic_row",
        document_id=document_id,
        pdf_page=pdf_page,
        table_id=table_fragment_id or None,
        row_id=row_id or None,
        cell_ids=(),
        bbox=bbox,
        metric_paths=metric_paths,
        periods=periods,
        segments=(),
        buckets=(),
        raw_values=raw_values,
        numeric_multiset=numeric,
        raw_text=raw_text,
        normalized_text=norm,
        source_traceback=source_tb,
        equivalent_group_id=record.get("equivalent_group_id"),
        row_type=row_type,
        row_index=row_index if isinstance(row_index, int) else None,
    )


# ---------------------------------------------------------------------------
# Atomic Fact → Signature
# ---------------------------------------------------------------------------


def _atomic_to_signature(record: dict[str, Any]) -> SemanticEvidenceSignature:
    """Convert an atomic-facts.jsonl record to SemanticEvidenceSignature."""
    source_tb = record.get("source_traceback") or {}
    fact_id = str(record.get("semantic_fact_id") or "")
    document_id = str(record.get("document_id") or "")
    pdf_page = int(source_tb.get("pdf_page") or 0)
    row_id = str(record.get("row_id") or "")
    cell_id = str(record.get("cell_id") or "")
    table_fragment_id = str(record.get("table_fragment_id") or "")
    metric_path = str(record.get("metric_path") or "")
    value_raw = str(record.get("value_raw") or "")
    value_norm = str(record.get("value_normalized") or "")
    period = str(record.get("normalized_period") or "")

    bbox = _extract_bbox(source_tb)
    raw_text = _extract_raw_text(source_tb, value_raw)

    metric_paths = (metric_path,) if metric_path else ()
    periods = (period,) if period else ()
    raw_values = (value_raw,) if value_raw else ()
    numeric = (_normalize_number(value_norm),) if value_norm else ()

    return SemanticEvidenceSignature(
        evidence_id=fact_id,
        evidence_type="atomic_fact",
        document_id=document_id,
        pdf_page=pdf_page,
        table_id=table_fragment_id or None,
        row_id=row_id or None,
        cell_ids=(cell_id,) if cell_id else (),
        bbox=bbox,
        metric_paths=metric_paths,
        periods=periods,
        segments=(),
        buckets=(),
        raw_values=raw_values,
        numeric_multiset=numeric,
        raw_text=raw_text,
        normalized_text=normalize_text(raw_text),
        source_traceback=source_tb,
        equivalent_group_id=record.get("equivalent_group_id"),
    )


# ---------------------------------------------------------------------------
# Comparison Fact → Signature
# ---------------------------------------------------------------------------


def _comparison_to_signature(record: dict[str, Any]) -> SemanticEvidenceSignature:
    """Convert a comparison-facts.jsonl record to SemanticEvidenceSignature."""
    source_tb = record.get("source_traceback") or {}
    fact_id = str(record.get("semantic_fact_id") or "")
    document_id = str(record.get("document_id") or "")
    pdf_page = int(source_tb.get("pdf_page") or 0)
    row_id = str(record.get("row_id") or "")
    table_fragment_id = str(record.get("table_fragment_id") or "")
    metric_path = str(record.get("metric_path") or "")
    base_period = str(record.get("base_period") or "")
    compared_period = str(record.get("compared_period") or "")

    bbox = _extract_bbox(source_tb)
    raw_text = _extract_raw_text(source_tb)

    periods = tuple(p for p in (base_period, compared_period) if p)
    metric_paths = (metric_path,) if metric_path else ()

    return SemanticEvidenceSignature(
        evidence_id=fact_id,
        evidence_type="comparison_fact",
        document_id=document_id,
        pdf_page=pdf_page,
        table_id=table_fragment_id or None,
        row_id=row_id or None,
        cell_ids=(),
        bbox=bbox,
        metric_paths=metric_paths,
        periods=periods,
        segments=(),
        buckets=(),
        raw_values=(),
        numeric_multiset=(),
        raw_text=raw_text,
        normalized_text=normalize_text(raw_text),
        source_traceback=source_tb,
        equivalent_group_id=record.get("equivalent_group_id"),
    )


# ---------------------------------------------------------------------------
# Bucket Fact → Signature
# ---------------------------------------------------------------------------


def _bucket_to_signature(record: dict[str, Any]) -> SemanticEvidenceSignature:
    """Convert a bucket-facts.jsonl record to SemanticEvidenceSignature."""
    source_tb = record.get("source_traceback") or {}
    fact_id = str(record.get("semantic_fact_id") or "")
    document_id = str(record.get("document_id") or "")
    pdf_page = int(source_tb.get("pdf_page") or 0)
    row_id = str(record.get("row_id") or "")
    cell_id = str(record.get("cell_id") or "")
    table_fragment_id = str(record.get("table_fragment_id") or "")
    metric_path = str(record.get("metric_path") or "")
    bucket_label = str(record.get("bucket_label") or "")
    value_raw = str(record.get("value_raw") or "")
    value_norm = str(record.get("value_normalized") or "")

    bbox = _extract_bbox(source_tb)
    raw_text = _extract_raw_text(source_tb, value_raw)

    metric_paths = (metric_path,) if metric_path else ()
    buckets = (bucket_label,) if bucket_label else ()
    raw_values = (value_raw,) if value_raw else ()
    numeric = (_normalize_number(value_norm),) if value_norm else ()

    return SemanticEvidenceSignature(
        evidence_id=fact_id,
        evidence_type="bucket_fact",
        document_id=document_id,
        pdf_page=pdf_page,
        table_id=table_fragment_id or None,
        row_id=row_id or None,
        cell_ids=(cell_id,) if cell_id else (),
        bbox=bbox,
        metric_paths=metric_paths,
        periods=(),
        segments=(),
        buckets=buckets,
        raw_values=raw_values,
        numeric_multiset=numeric,
        raw_text=raw_text,
        normalized_text=normalize_text(raw_text),
        source_traceback=source_tb,
        equivalent_group_id=record.get("equivalent_group_id"),
    )


# ---------------------------------------------------------------------------
# Row Matrix → Signature
# ---------------------------------------------------------------------------


def _matrix_to_signature(record: dict[str, Any]) -> SemanticEvidenceSignature:
    """Convert a row-matrices.jsonl record to SemanticEvidenceSignature."""
    source_tb = record.get("source_traceback") or {}
    fact_id = str(record.get("semantic_fact_id") or "")
    document_id = str(record.get("document_id") or "")
    pdf_page = int(source_tb.get("pdf_page") or 0)
    row_id = str(record.get("row_id") or "")
    table_fragment_id = str(record.get("table_fragment_id") or "")
    metric_path = str(record.get("metric_path") or "")
    dimensions = record.get("dimensions") or []

    bbox = _extract_bbox(source_tb)
    raw_text = _extract_raw_text(source_tb)

    metric_paths = (metric_path,) if metric_path else ()

    # Extract periods, segments, and values from dimensions
    periods: list[str] = []
    segments: list[str] = []
    raw_values: list[str] = []
    numeric: list[str] = []
    for dim in dimensions:
        if not isinstance(dim, dict):
            continue
        np = dim.get("normalized_period")
        if np:
            periods.append(str(np))
        sl = dim.get("segment_label")
        if sl:
            segments.append(str(sl))
        vr = dim.get("value_raw")
        if vr:
            raw_values.append(str(vr))
        vn = dim.get("value_normalized")
        if vn:
            numeric.append(_normalize_number(str(vn)))

    return SemanticEvidenceSignature(
        evidence_id=fact_id,
        evidence_type="row_matrix",
        document_id=document_id,
        pdf_page=pdf_page,
        table_id=table_fragment_id or None,
        row_id=row_id or None,
        cell_ids=(),
        bbox=bbox,
        metric_paths=metric_paths,
        periods=tuple(sorted(set(periods))),
        segments=tuple(sorted(set(segments))),
        buckets=(),
        raw_values=tuple(raw_values),
        numeric_multiset=tuple(sorted(set(numeric))),
        raw_text=raw_text,
        normalized_text=normalize_text(raw_text),
        source_traceback=source_tb,
        equivalent_group_id=record.get("equivalent_group_id"),
    )


# ---------------------------------------------------------------------------
# Narrative Evidence → Signature
# ---------------------------------------------------------------------------


def _narrative_to_signature(record: dict[str, Any]) -> SemanticEvidenceSignature:
    """Convert a narrative-evidence.jsonl record to SemanticEvidenceSignature."""
    source_tb = record.get("source_traceback") or {}
    evidence_id = str(record.get("semantic_evidence_id") or "")
    document_id = str(record.get("document_id") or "")
    pdf_page = int(record.get("pdf_page") or 0)
    section_path = str(record.get("section_path") or "")
    heading = str(record.get("heading") or "")
    raw_text = str(record.get("raw_text") or "")

    bbox = _extract_bbox(source_tb) or _extract_bbox(record)
    if not bbox:
        bbox_raw = record.get("bbox")
        if bbox_raw and isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            bbox = tuple(float(x) for x in bbox_raw)

    # Combine section_path + heading + raw_text for token extraction
    f"{section_path} {heading} {raw_text}".strip()
    numeric = extract_numeric_multiset(raw_text)
    periods = extract_period_tokens(raw_text)
    norm = normalize_text(raw_text)

    return SemanticEvidenceSignature(
        evidence_id=evidence_id,
        evidence_type="narrative_evidence",
        document_id=document_id,
        pdf_page=pdf_page,
        table_id=None,
        row_id=None,
        cell_ids=(),
        bbox=bbox,
        metric_paths=(),
        periods=periods,
        segments=(),
        buckets=(),
        raw_values=(),
        numeric_multiset=numeric,
        raw_text=raw_text,
        normalized_text=norm,
        source_traceback=source_tb,
        equivalent_group_id=None,
    )


# ---------------------------------------------------------------------------
# Logical Table → Signature
# ---------------------------------------------------------------------------


def _logical_table_to_signature(record: dict[str, Any]) -> SemanticEvidenceSignature:
    """Convert a logical-tables.jsonl record to SemanticEvidenceSignature."""
    source_tb = record.get("source_traceback") or {}
    table_fragment_id = str(record.get("table_fragment_id") or "")
    document_id = str(record.get("document_id") or "")
    pdf_page = int(record.get("pdf_page") or 0)
    table_title = str(record.get("table_title") or "")

    bbox = _extract_bbox(source_tb)

    norm = normalize_text(table_title)

    return SemanticEvidenceSignature(
        evidence_id=table_fragment_id,
        evidence_type="logical_table",
        document_id=document_id,
        pdf_page=pdf_page,
        table_id=table_fragment_id or None,
        row_id=None,
        cell_ids=(),
        bbox=bbox,
        metric_paths=(),
        periods=(),
        segments=(),
        buckets=(),
        raw_values=(),
        numeric_multiset=(),
        raw_text=table_title,
        normalized_text=norm,
        source_traceback=source_tb,
        equivalent_group_id=None,
    )


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Catalog Builder
# ---------------------------------------------------------------------------


class SemanticEvidenceCatalog:
    """In-memory catalog of all Semantic Evidence from Gate 03 R2.

    Indexed by (document_id, pdf_page) for fast candidate lookup.
    """

    def __init__(self) -> None:
        # All evidence signatures
        self._all: list[SemanticEvidenceSignature] = []
        # Index: (document_id, pdf_page) → list of evidence indices
        self._by_page: dict[tuple[str, int], list[int]] = defaultdict(list)
        # Index: row_id → list of evidence indices
        self._by_row: dict[str, list[int]] = defaultdict(list)
        # Index: table_id → list of evidence indices
        self._by_table: dict[str, list[int]] = defaultdict(list)
        # Index: evidence_id → index
        self._by_evidence_id: dict[str, int] = {}
        # Metric paths: row_id → metric_path
        self._row_metric_paths: dict[str, str] = {}

    @property
    def total_count(self) -> int:
        return len(self._all)

    def add(self, sig: SemanticEvidenceSignature) -> None:
        """Add an evidence signature to the catalog."""
        idx = len(self._all)
        self._all.append(sig)
        self._by_page[(sig.document_id, sig.pdf_page)].append(idx)
        if sig.row_id:
            self._by_row[sig.row_id].append(idx)
        if sig.table_id:
            self._by_table[sig.table_id].append(idx)
        self._by_evidence_id[sig.evidence_id] = idx

    def get_by_evidence_id(self, evidence_id: str) -> SemanticEvidenceSignature | None:
        idx = self._by_evidence_id.get(evidence_id)
        if idx is not None:
            return self._all[idx]
        return None

    def get_by_page(
        self, document_id: str, pdf_page: int
    ) -> list[SemanticEvidenceSignature]:
        """Get all evidence on a given page."""
        return [self._all[i] for i in self._by_page.get((document_id, pdf_page), [])]

    def get_by_row(self, row_id: str) -> list[SemanticEvidenceSignature]:
        """Get all evidence for a given row."""
        return [self._all[i] for i in self._by_row.get(row_id, [])]

    def get_by_table(self, table_id: str) -> list[SemanticEvidenceSignature]:
        """Get all evidence for a given table."""
        return [self._all[i] for i in self._by_table.get(table_id, [])]

    def get_rows_by_page(
        self, document_id: str, pdf_page: int
    ) -> list[SemanticEvidenceSignature]:
        """Get only semantic_row evidence on a given page."""
        return [
            self._all[i]
            for i in self._by_page.get((document_id, pdf_page), [])
            if self._all[i].evidence_type == "semantic_row"
        ]

    def get_narrative_by_page(
        self, document_id: str, pdf_page: int
    ) -> list[SemanticEvidenceSignature]:
        """Get only narrative_evidence on a given page."""
        return [
            self._all[i]
            for i in self._by_page.get((document_id, pdf_page), [])
            if self._all[i].evidence_type == "narrative_evidence"
        ]

    def get_facts_by_page(
        self, document_id: str, pdf_page: int
    ) -> list[SemanticEvidenceSignature]:
        """Get all fact-type evidence (atomic/comparison/bucket/matrix) on a page."""
        fact_types = {"atomic_fact", "comparison_fact", "bucket_fact", "row_matrix"}
        return [
            self._all[i]
            for i in self._by_page.get((document_id, pdf_page), [])
            if self._all[i].evidence_type in fact_types
        ]

    def get_metric_path_for_row(self, row_id: str) -> str | None:
        """Get the metric_path for a row from the metric-paths cross-reference."""
        return self._row_metric_paths.get(row_id)

    def set_row_metric_paths(self, row_id: str, metric_path: str) -> None:
        """Set the metric_path for a row (from metric-paths.jsonl)."""
        self._row_metric_paths[row_id] = metric_path

    def get_all(self) -> list[SemanticEvidenceSignature]:
        """Get all evidence signatures."""
        return list(self._all)

    def stats(self) -> dict[str, Any]:
        """Return catalog statistics."""
        type_counts: dict[str, int] = defaultdict(int)
        for sig in self._all:
            type_counts[sig.evidence_type] += 1
        return {
            "total_evidence": len(self._all),
            "by_type": dict(type_counts),
            "unique_pages": len(self._by_page),
            "unique_rows": len(self._by_row),
            "unique_tables": len(self._by_table),
        }


# ---------------------------------------------------------------------------
# Catalog Loader
# ---------------------------------------------------------------------------


def _build_row_enrichment(base: Path) -> dict[str, dict[str, Any]]:
    """Build enrichment lookup from atomic-facts and row-matrices.

    Returns: row_id → {numeric_values, metric_paths, periods, raw_values}
    """
    from collections import defaultdict

    enrichment: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "numeric_values": set(),
            "metric_paths": set(),
            "periods": set(),
            "raw_values": [],
        }
    )

    # Atomic facts
    atomic_path = base / "atomic-facts.jsonl"
    if atomic_path.exists():
        for record in _read_jsonl(atomic_path):
            row_id = str(record.get("row_id") or "")
            if not row_id:
                continue
            vn = str(record.get("value_normalized") or "")
            if vn:
                enrichment[row_id]["numeric_values"].add(_normalize_number(vn))
            mp = str(record.get("metric_path") or "")
            if mp:
                enrichment[row_id]["metric_paths"].add(mp)
            np = str(record.get("normalized_period") or "")
            if np:
                enrichment[row_id]["periods"].add(np)
            vr = str(record.get("value_raw") or "")
            if vr:
                enrichment[row_id]["raw_values"].append(vr)

    # Row matrices
    matrix_path = base / "row-matrices.jsonl"
    if matrix_path.exists():
        for record in _read_jsonl(matrix_path):
            row_id = str(record.get("row_id") or "")
            if not row_id:
                continue
            for dim in record.get("dimensions") or []:
                if not isinstance(dim, dict):
                    continue
                vn = dim.get("value_normalized")
                if vn:
                    enrichment[row_id]["numeric_values"].add(_normalize_number(str(vn)))
                np = dim.get("normalized_period")
                if np:
                    enrichment[row_id]["periods"].add(str(np))
                vr = dim.get("value_raw")
                if vr:
                    enrichment[row_id]["raw_values"].append(str(vr))
            mp = str(record.get("metric_path") or "")
            if mp:
                enrichment[row_id]["metric_paths"].add(mp)

    return dict(enrichment)


def load_catalog(gate_03_r2_dir: str | Path) -> SemanticEvidenceCatalog:
    """Load all Gate 03 R2 artifacts into a SemanticEvidenceCatalog.

    Args:
        gate_03_r2_dir: Path to artifacts/evaluation/pdf-retrieval-v4-gate-03-r2/

    Returns:
        Populated SemanticEvidenceCatalog
    """
    base = Path(gate_03_r2_dir)
    catalog = SemanticEvidenceCatalog()

    # 0. Build row enrichment from atomic-facts and row-matrices
    row_enrichment = _build_row_enrichment(base)
    enriched_count = sum(1 for v in row_enrichment.values() if v["numeric_values"])
    print(f"  Row enrichment: {len(row_enrichment)} rows, {enriched_count} with values")

    # 1. Load semantic rows (enriched with fact/matrix values)
    rows_path = base / "semantic-rows.jsonl"
    if rows_path.exists():
        for record in _read_jsonl(rows_path):
            row_id = str(record.get("row_id") or "")
            enrichment = row_enrichment.get(row_id)
            sig = _row_to_signature(record, enrichment)
            catalog.add(sig)

    # 2. Load metric paths (cross-reference for row metric_path)
    metric_paths_path = base / "metric-paths.jsonl"
    if metric_paths_path.exists():
        for record in _read_jsonl(metric_paths_path):
            row_id = str(record.get("row_id") or "")
            metric_path = str(record.get("metric_path") or "")
            if row_id and metric_path:
                catalog.set_row_metric_paths(row_id, metric_path)

    # 3. Load atomic facts
    atomic_path = base / "atomic-facts.jsonl"
    if atomic_path.exists():
        for record in _read_jsonl(atomic_path):
            sig = _atomic_to_signature(record)
            catalog.add(sig)

    # 4. Load comparison facts
    comparison_path = base / "comparison-facts.jsonl"
    if comparison_path.exists():
        for record in _read_jsonl(comparison_path):
            sig = _comparison_to_signature(record)
            catalog.add(sig)

    # 5. Load bucket facts
    bucket_path = base / "bucket-facts.jsonl"
    if bucket_path.exists():
        for record in _read_jsonl(bucket_path):
            sig = _bucket_to_signature(record)
            catalog.add(sig)

    # 6. Load row matrices
    matrix_path = base / "row-matrices.jsonl"
    if matrix_path.exists():
        for record in _read_jsonl(matrix_path):
            sig = _matrix_to_signature(record)
            catalog.add(sig)

    # 7. Load narrative evidence
    narrative_path = base / "narrative-evidence.jsonl"
    if narrative_path.exists():
        for record in _read_jsonl(narrative_path):
            sig = _narrative_to_signature(record)
            catalog.add(sig)

    # 8. Load logical tables
    logical_path = base / "logical-tables.jsonl"
    if logical_path.exists():
        for record in _read_jsonl(logical_path):
            sig = _logical_table_to_signature(record)
            catalog.add(sig)

    return catalog
