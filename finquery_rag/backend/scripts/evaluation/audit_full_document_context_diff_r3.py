"""Gate 02 R3: Audit full-document context diff (HTML / BBox changes).

Reads the R2 ``probe-page-structural-diff.json`` and classifies the pages
with table HTML changes and the pages with table BBox changes into
reconciliation categories:

  - ``benign_html_normalization`` : HTML string hash changed but table
    count, text block count, and bbox are all unchanged (pure
    normalization of whitespace / attribute order).
  - ``better_table_segmentation`` : new output has more tables than the
    old probe (improved segmentation).
  - ``cross_page_context_change`` : table count decreased (merge) or
    surrounding text block count shifted, indicating cross-page context
    restructuring rather than content loss.  Pages "missing in new" are
    also bucketed here: their HTML/BBox "change" is an artifact of the
    new output having nothing to compare, and their true missing-page
    status is owned by
    ``reconcile_probe_structural_diff_r3.py``.
  - ``bbox_geometry_shift``        : only the table bbox moved/refined,
    table count unchanged.
  - ``row_structure_change``       : same table count but both HTML and
    bbox changed, indicating the table's internal row layout shifted.
  - ``actual_regression``          : a page present in the new output
    whose tables vanished without the text blocks growing to absorb
    them (genuine structural loss).

Only ``actual_regression`` is adapter-blocking.  HTML string hash changes
alone never cause a Fail, and pages "missing in new" are never flagged as
``actual_regression`` here.

Reads ONLY the R2 structural diff.  No questions, gold, or governance.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R2_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2"
R3_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r3"

CLASSIFICATION_KEYS = (
    "benign_html_normalization",
    "better_table_segmentation",
    "cross_page_context_change",
    "bbox_geometry_shift",
    "row_structure_change",
    "actual_regression",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _classify_changed_page(rec: dict[str, Any]) -> str:
    old_tc = int(rec.get("old_table_count") or 0)
    new_tc = int(rec.get("new_table_count") or 0)
    old_tb = int(rec.get("old_text_block_count") or 0)
    new_tb = int(rec.get("new_text_block_count") or 0)
    html_changed = bool(rec.get("table_html_changed", False))
    bbox_changed = bool(rec.get("table_bbox_changed", False))
    page_present = bool(rec.get("page_present_in_new", True))

    tc_delta = new_tc - old_tc
    tb_delta = new_tb - old_tb

    # Pages "missing in new" (no tables and no text blocks in the new output)
    # are reconciled by reconcile_probe_structural_diff_r3.py, which owns the
    # true missing-page blocking decision.  Here they are never flagged as
    # actual_regression; the HTML/BBox "change" is only an artifact of the new
    # output having nothing to compare.  Bucket as a non-blocking context
    # change.
    if not page_present:
        return "cross_page_context_change"

    # Present pages: real regression only when tables vanished and the text
    # blocks did not grow to absorb them (genuine structural loss).
    if old_tc > 0 and new_tc == 0 and new_tb <= old_tb:
        return "actual_regression"

    # More tables in new output → improved segmentation.
    if tc_delta > 0:
        return "better_table_segmentation"

    # Fewer tables but content preserved → cross-page merge / context shift.
    if tc_delta < 0:
        return "cross_page_context_change"

    # Same table count.
    if html_changed and bbox_changed:
        return "row_structure_change"
    if bbox_changed:
        return "bbox_geometry_shift"
    if tb_delta != 0:
        return "cross_page_context_change"
    return "benign_html_normalization"


def _page_entry(rec: dict[str, Any]) -> dict[str, Any]:
    classification = _classify_changed_page(rec)
    return {
        "document_id": str(rec.get("document_id") or ""),
        "pdf_page": int(rec.get("pdf_page") or 0),
        "old_table_count": int(rec.get("old_table_count") or 0),
        "new_table_count": int(rec.get("new_table_count") or 0),
        "table_count_delta": int(rec.get("table_count_delta") or 0),
        "old_text_block_count": int(rec.get("old_text_block_count") or 0),
        "new_text_block_count": int(rec.get("new_text_block_count") or 0),
        "table_html_changed": bool(rec.get("table_html_changed", False)),
        "table_bbox_changed": bool(rec.get("table_bbox_changed", False)),
        "classification": classification,
        "adapter_blocking": classification == "actual_regression",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-out", type=Path, default=R2_OUT)
    parser.add_argument("--r3-out", type=Path, default=R3_OUT)
    args = parser.parse_args()

    diff_path = args.r2_out / "probe-page-structural-diff.json"
    if not diff_path.is_file():
        print("ERROR: R2 structural diff not found at", diff_path)
        return 1
    structural_diff = json.loads(diff_path.read_text(encoding="utf-8"))

    per_page = structural_diff.get("per_page", [])
    html_changed = [r for r in per_page if r.get("table_html_changed", False)]
    bbox_changed = [r for r in per_page if r.get("table_bbox_changed", False)]

    html_entries = [_page_entry(r) for r in html_changed]
    bbox_entries = [_page_entry(r) for r in bbox_changed]

    # Summary counts each unique changed page once (a page present in both
    # lists is counted only once, using its single classification).
    summary: Counter[str] = Counter()
    seen_keys: set[tuple[str, int]] = set()
    for entry in html_entries + bbox_entries:
        key = (entry["document_id"], entry["pdf_page"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        summary[entry["classification"]] += 1

    classification_summary = {key: summary.get(key, 0) for key in CLASSIFICATION_KEYS}

    actual_regression_count = classification_summary["actual_regression"]
    adapter_blocking = actual_regression_count > 0

    output = {
        "schema": "pdf-retrieval-v4/gate-02-r3/full-document-context-diff-audit/v1",
        "html_changed_pages": html_entries,
        "bbox_changed_pages": bbox_entries,
        "classification_summary": classification_summary,
        "actual_regression_count": actual_regression_count,
        "adapter_blocking": adapter_blocking,
    }
    _write_json(args.r3_out / "full-document-context-diff-audit.json", output)

    print(f"HTML changed pages: {len(html_entries)}")
    print(f"BBox changed pages: {len(bbox_entries)}")
    print(f"  actual_regression_count: {actual_regression_count}")
    print(f"  adapter_blocking: {adapter_blocking}")

    return 0 if not adapter_blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
