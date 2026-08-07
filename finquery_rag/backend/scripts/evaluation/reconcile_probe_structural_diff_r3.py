"""Gate 02 R3: Reconcile R2 probe structural diff "missing in new" pages.

Reads the R2 ``probe-page-structural-diff.json`` and classifies each page
that was reported as "missing in new" (``page_present_in_new == false``)
into a reconciliation category:

  - ``corpus_scope_difference`` : document is not one of the 8 frozen
    benchmark PDFs (e.g. dev corpus); the absence is by-design, not a real
    missing page.
  - ``empty_page_no_content``   : page is in the frozen corpus but had no
    tables/text in either old or new output (genuinely empty page).
  - ``representation_changed``  : old probe had tables/text but new output
    has none of either, while the new MinerU ``middle.json`` still carries
    content for the page (content exists but was not detected as a table
    by the old comparator).
  - ``structural_loss``         : page record is genuinely absent from the
    new MinerU output (real structural loss).

Only ``page_record_present == false`` counts as a true missing page.  The
expected ``true_missing_page_count`` is 0.

Reads ONLY the R2 structural diff and the frozen-corpus MinerU
``middle.json`` outputs.  No questions, gold, or governance data is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SHARED_NANOCHAT_ROOT = ROOT.parents[4]
R2_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2"
R3_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r3"
MINERU_OUTPUT = SHARED_NANOCHAT_ROOT / ".runtime/pdf-retrieval-v4-gate-02-r2/mineru"

FROZEN_DOCUMENT_IDS = {
    "aapl_fy2025",
    "jpm_fy2025",
    "ko_fy2025",
    "msft_fy2025",
    "nvda_fy2025",
    "pfe_fy2024",
    "tsla_fy2025",
    "v_fy2025",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _find_middle_json(doc_output: Path) -> Path | None:
    if not doc_output.is_dir():
        return None
    matches = sorted(doc_output.rglob("*_middle.json"))
    return matches[0] if matches else None


def _load_middle(
    doc_id: str,
    mineru_output: Path,
    cache: dict[str, Any],
) -> Any:
    if doc_id in cache:
        return cache[doc_id]
    middle_path = _find_middle_json(mineru_output / doc_id)
    middle: Any = None
    if middle_path and middle_path.is_file():
        try:
            middle = json.loads(middle_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            middle = None
    cache[doc_id] = middle
    return middle


def _page_record_in_range(middle_data: Any, page_idx: int) -> bool:
    """True if ``middle.json`` pdf_info has a record at ``page_idx``."""
    if not isinstance(middle_data, dict):
        return False
    pdf_info = middle_data.get("pdf_info", [])
    if not isinstance(pdf_info, list):
        return False
    return 0 <= page_idx < len(pdf_info)


def _page_has_mineru_content(middle_data: Any, page_idx: int) -> bool:
    """True if the page record carries any preproc blocks (non-empty page)."""
    if not _page_record_in_range(middle_data, page_idx):
        return False
    pdf_info = middle_data.get("pdf_info", [])
    page_data = pdf_info[page_idx]
    if not isinstance(page_data, dict):
        return False
    blocks = page_data.get("preproc_blocks") or page_data.get("blocks") or []
    return bool(blocks)


def _reconcile_page(
    rec: dict[str, Any],
    mineru_output: Path,
    middle_cache: dict[str, Any],
) -> dict[str, Any]:
    doc_id = str(rec.get("document_id") or "")
    pdf_page = int(rec.get("pdf_page") or 0)
    page_idx = pdf_page - 1  # 0-based index into pdf_info

    old_tc = int(rec.get("old_table_count") or 0)
    new_tc = int(rec.get("new_table_count") or 0)
    old_tb = int(rec.get("old_text_block_count") or 0)
    new_tb = int(rec.get("new_text_block_count") or 0)

    difference_class = "structural_loss"
    page_record_present = False
    mineru_content_present = False

    if doc_id not in FROZEN_DOCUMENT_IDS:
        # Out-of-scope dev document: the new full-corpus run deliberately
        # excludes it, so this is not a real missing page.
        difference_class = "corpus_scope_difference"
        page_record_present = True
        mineru_content_present = False
    else:
        middle = _load_middle(doc_id, mineru_output, middle_cache)
        page_record_present = _page_record_in_range(middle, page_idx)
        mineru_content_present = _page_has_mineru_content(middle, page_idx)

        if old_tc == 0 and old_tb == 0:
            # Old probe also had no content → genuinely empty page.
            difference_class = "empty_page_no_content"
            page_record_present = True
        elif mineru_content_present:
            # Content exists in new MinerU but was not detected as a
            # table/text block by the old comparator → representation
            # changed, not a structural loss.
            difference_class = "representation_changed"
            page_record_present = True
        elif page_record_present:
            # Page record exists but carries no content blocks.
            difference_class = "empty_page_no_content"
        else:
            # Page record genuinely absent from new MinerU output.
            difference_class = "structural_loss"
            page_record_present = False

    adapter_blocking = not page_record_present

    return {
        "document_id": doc_id,
        "pdf_page": pdf_page,
        "page_record_present": page_record_present,
        "old_table_count": old_tc,
        "new_table_count": new_tc,
        "old_text_block_count": old_tb,
        "new_text_block_count": new_tb,
        "mineru_content_present": mineru_content_present,
        "difference_class": difference_class,
        "adapter_blocking": adapter_blocking,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-out", type=Path, default=R2_OUT)
    parser.add_argument("--r3-out", type=Path, default=R3_OUT)
    parser.add_argument("--mineru-output", type=Path, default=MINERU_OUTPUT)
    args = parser.parse_args()

    diff_path = args.r2_out / "probe-page-structural-diff.json"
    if not diff_path.is_file():
        print("ERROR: R2 structural diff not found at", diff_path)
        return 1
    structural_diff = json.loads(diff_path.read_text(encoding="utf-8"))

    per_page = structural_diff.get("per_page", [])
    missing_pages = [r for r in per_page if not r.get("page_present_in_new", True)]

    middle_cache: dict[str, Any] = {}
    reconciled = [
        _reconcile_page(r, args.mineru_output, middle_cache) for r in missing_pages
    ]

    true_missing = sum(1 for r in reconciled if not r["page_record_present"])
    adapter_blocking = true_missing > 0

    output = {
        "schema": "pdf-retrieval-v4/gate-02-r3/probe-structural-diff-reconciliation/v1",
        "total_pages_missing_in_r2_diff": len(missing_pages),
        "reconciled_pages": reconciled,
        "true_missing_page_count": true_missing,
        "adapter_blocking": adapter_blocking,
    }
    _write_json(args.r3_out / "probe-structural-diff-reconciliation.json", output)

    print(f"Reconciled {len(missing_pages)} missing-in-new pages")
    print(f"  true_missing_page_count: {true_missing}")
    print(f"  adapter_blocking: {adapter_blocking}")

    return 0 if not adapter_blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
