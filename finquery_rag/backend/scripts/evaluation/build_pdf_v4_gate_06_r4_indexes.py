#!/usr/bin/env python3
"""Gate 06 R4: Build Expanded Candidate Shadow Indexes.

Constructs 4 isolated shadow lanes from Production Candidate views:

  Raw Lane:        38,319 raw views (unchanged from Gate 08 R2)
  Structured Lane: 19,500 Grade-A structured views (from Gate 05 R5)

The Structured Lane expands from the previous 628 views to 19,500,
isolating "Coverage Expansion" as the sole variable for the upcoming
Coverage-only Retrieval Replay (Gate 08 R3).

No retrieval is performed. No Question/Gold/Governance data is read.

Usage:
    python3 scripts/evaluation/build_pdf_v4_gate_06_r4_indexes.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.candidate_aligned_view import (  # noqa: E402
    CandidateAlignedView,
    CandidateViewPair,
    make_structured_view_id,
)
from src.pdf_retrieval_v4.candidate_view_index import (  # noqa: E402
    CandidateViewIndexBuilder,
)

GATE08_R2_VIEWS = (
    ROOT
    / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2/candidate-views/view-pairs.jsonl"
)
R5_STRUCTURED_VIEWS = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5/structured-views.jsonl"
)
DEFAULT_OUT = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-06-r4/candidate-indexes"
)
STRUCTURED_TEXT_VERSION = "gate06-r4-v1"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_r5_structured_text(view: dict[str, Any]) -> str:
    """Serialize an R5 structured view into deterministic retrieval text.

    Field order is fixed. List items are sorted. Empty fields are omitted.
    """
    parts: list[str] = []
    document_id = str(view.get("document_id") or "")
    pdf_page = view.get("pdf_page")
    if document_id:
        parts.append(f"Document: {document_id}")
    if pdf_page is not None:
        parts.append(f"Page: {pdf_page}")
    section_path = sorted(s for s in (view.get("section_path") or []) if s)
    if section_path:
        parts.append("")
        parts.append("Section:")
        parts.append("\n".join(f"  {s}" for s in section_path))
    table_title = view.get("table_title")
    if table_title:
        parts.append(f"Table: {table_title}")
    metric_paths = sorted(m for m in (view.get("metric_paths") or []) if m)
    if metric_paths:
        parts.append("")
        parts.append("Metric Paths:")
        parts.append("\n".join(f"  {m}" for m in metric_paths))
    periods = sorted(p for p in (view.get("periods") or []) if p)
    if periods:
        parts.append("")
        parts.append("Periods:")
        parts.append("\n".join(f"  {p}" for p in periods))
    facts = view.get("facts") or []
    fact_lines: list[str] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        ftype = str(fact.get("type") or "")
        metric = str(fact.get("metric") or "")
        if ftype == "narrative":
            text_val = str(fact.get("text") or "")
            if text_val:
                fact_lines.append(f"narrative | {text_val}")
        elif ftype == "row_matrix":
            rm_periods = fact.get("periods") or []
            rm_values = fact.get("values") or []
            for p, v in zip(rm_periods, rm_values):
                fact_lines.append(f"{metric} | {p} | {v}")
        else:
            period = str(fact.get("period") or "")
            value = str(fact.get("value") or "")
            scale = str(fact.get("scale") or "")
            fact_lines.append(f"{metric} | {period} | {value} | {scale}")
    if fact_lines:
        parts.append("")
        parts.append("Facts:")
        parts.append("\n".join(f"  {f}" for f in sorted(fact_lines)))
    segments = sorted(s for s in (view.get("segments") or []) if s)
    if segments:
        parts.append("")
        parts.append("Segments:")
        parts.append(", ".join(segments))
    buckets = sorted(b for b in (view.get("buckets") or []) if b)
    if buckets:
        parts.append("")
        parts.append("Buckets:")
        parts.append(", ".join(buckets))
    raw_content = str(view.get("raw_content") or "")
    parts.append("")
    parts.append("Source:")
    parts.append(raw_content)
    return "\n".join(parts)


def load_raw_views(view_pairs_path: Path) -> dict[str, CandidateAlignedView]:
    """Load raw views from Gate 08 R2 view-pairs.jsonl."""
    raw_views: dict[str, CandidateAlignedView] = {}
    for line in view_pairs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        ck = record["candidate_key"]
        rv = record["raw_view"]
        raw_views[ck] = CandidateAlignedView(
            candidate_key=ck,
            view_type="raw",
            view_id=rv["view_id"],
            retrieval_text=rv["retrieval_text"],
            document_id=rv.get("document_id", ""),
            pdf_page=rv.get("pdf_page"),
            logical_table_ids=tuple(rv.get("logical_table_ids") or []),
            row_ids=tuple(rv.get("row_ids") or []),
            fact_ids=tuple(rv.get("fact_ids") or []),
            metric_paths=tuple(rv.get("metric_paths") or []),
            periods=tuple(rv.get("periods") or []),
            temporal_types=tuple(rv.get("temporal_types") or []),
            bridge_grade=rv.get("bridge_grade", "raw_only"),
        )
    return raw_views


def load_r5_structured_views(path: Path) -> dict[str, dict[str, Any]]:
    """Load R5 structured views keyed by candidate_key."""
    views: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        views[record["candidate_key"]] = record
    return views


def build_view_pairs(
    raw_views: dict[str, CandidateAlignedView],
    r5_structured: dict[str, dict[str, Any]],
) -> list[CandidateViewPair]:
    """Build CandidateViewPair list combining raw views with R5 structured views."""
    pairs: list[CandidateViewPair] = []
    for ck, raw_view in raw_views.items():
        structured_view: CandidateAlignedView | None = None
        r5_data = r5_structured.get(ck)
        if r5_data is not None:
            retrieval_text = format_r5_structured_text(r5_data)
            structured_view = CandidateAlignedView(
                candidate_key=ck,
                view_type="structured",
                view_id=make_structured_view_id(ck),
                retrieval_text=retrieval_text,
                document_id=str(r5_data.get("document_id") or raw_view.document_id),
                pdf_page=r5_data.get("pdf_page", raw_view.pdf_page),
                logical_table_ids=(),
                row_ids=tuple(r5_data.get("row_ids") or []),
                fact_ids=tuple(r5_data.get("semantic_evidence_ids") or []),
                metric_paths=tuple(sorted(r5_data.get("metric_paths") or [])),
                periods=tuple(sorted(r5_data.get("periods") or [])),
                temporal_types=(),
                bridge_grade=str(r5_data.get("bridge_grade") or "raw_only"),
            )
        pairs.append(
            CandidateViewPair(
                candidate_key=ck,
                raw_view=raw_view,
                structured_view=structured_view,
            )
        )
    return pairs


def compute_raw_parity_hash(pairs: list[CandidateViewPair]) -> dict[str, Any]:
    """Compute parity hash for raw views to verify unchanged from Gate 08 R2."""
    raw_texts: list[str] = []
    raw_view_ids: list[str] = []
    for pair in sorted(pairs, key=lambda p: p.candidate_key):
        raw_texts.append(pair.raw_view.retrieval_text)
        raw_view_ids.append(pair.raw_view.view_id)
    return {
        "raw_candidate_count": len(pairs),
        "raw_view_id_hash": _sha256_text("\n".join(raw_view_ids)),
        "raw_text_hash": _sha256_text("\n".join(raw_texts)),
    }


def main() -> int:
    print("=" * 70)
    print("Gate 06 R4 - Build Expanded Candidate Shadow Indexes")
    print("=" * 70)
    print("\nLoading raw views from Gate 08 R2...")
    raw_views = load_raw_views(GATE08_R2_VIEWS)
    print(f"  Raw views: {len(raw_views)}")
    print("Loading R5 structured views...")
    r5_structured = load_r5_structured_views(R5_STRUCTURED_VIEWS)
    print(f"  R5 structured views: {len(r5_structured)}")
    overlap = len(set(raw_views.keys()) & set(r5_structured.keys()))
    print(f"  Overlap: {overlap}")
    print("\nBuilding view pairs...")
    pairs = build_view_pairs(raw_views, r5_structured)
    structured_count = sum(1 for p in pairs if p.structured_view is not None)
    raw_only_count = sum(1 for p in pairs if p.structured_view is None)
    print(f"  Total pairs: {len(pairs)}")
    print(f"  With structured view: {structured_count}")
    print(f"  Raw-only: {raw_only_count}")
    parity = compute_raw_parity_hash(pairs)
    print("\nRaw parity:")
    print(f"  Candidate count: {parity['raw_candidate_count']}")
    print(f"  View ID hash: {parity['raw_view_id_hash'][:16]}...")
    print(f"  Text hash: {parity['raw_text_hash'][:16]}...")
    print("\nBuilding 4-lane shadow indexes...")
    builder = CandidateViewIndexBuilder(DEFAULT_OUT, "all-MiniLM-L6-v2")
    report = builder.build(pairs)
    report["gate"] = "gate-06-r4"
    report["structured_text_version"] = STRUCTURED_TEXT_VERSION
    report["raw_parity"] = parity
    report["r5_structured_count"] = structured_count
    report["raw_only_count"] = raw_only_count
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    report_path = DEFAULT_OUT / "index-build-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\nBuild complete:")
    print(f"  Total pairs: {report['total_pairs']}")
    print(f"  Unique keys: {report['unique_candidate_keys']}")
    print(f"  Integrity OK: {report['integrity']['ok']}")
    for lane, stat in report["lanes"].items():
        print(f"  Lane {lane}: {stat['view_count']} views")
    print(f"  Report: {report_path}")
    print("=" * 70)
    return 0 if report["integrity"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
