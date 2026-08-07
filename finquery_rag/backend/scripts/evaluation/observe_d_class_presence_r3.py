"""Post-seal D-class / B-class structural presence observer for Gate 02 R3.

Runs AFTER the Prediction Seal.  Observes whether Gold target pages that
previously had no structure (D-class) or incomplete structure (B-class)
now have V4 structure in the R3 adapter predictions.

This script does NOT modify the adapter.  It only reads sealed predictions
and reports structural presence.

D-class: Oracle records where the R1 scoring found no table on the target
page (``table_recovery == false``) — the page previously had no structure.

B-class: Oracle records where the R1 scoring found a table but did not
fully recover the row/cell numeric (``table_recovery == true`` but
``numeric_exact_recovery == false`` or ``source_backtrace == false``).

Reads ONLY the sealed R3 predictions and the Gate 02 Oracle audit file.
No questions, gold, or governance data is read.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R3_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r3"
R1_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02"
SHARED_NANOCHAT_ROOT = ROOT.parents[4]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_oracle_records(r1_out: Path) -> list[dict[str, Any]]:
    """Load Oracle records from the Gate 02 scoring JSONL."""
    oracle_path = r1_out / "gate-02-oracle-source-audit.jsonl"
    records: list[dict[str, Any]] = []
    if oracle_path.is_file():
        for line in oracle_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_r3_predictions(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load R3 predictions and index by (document_id, pdf_page)."""
    index: dict[tuple[str, int], dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            page = json.loads(line)
            doc_id = str(page.get("document_id") or "")
            pdf_page = int(page.get("pdf_page") or 0)
            index[(doc_id, pdf_page)] = page
    return index


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify_record(record: dict[str, Any]) -> str:
    """Classify an Oracle record as D-class or B-class.

    D-class: no table found in R1 scoring (table_recovery == false).
    B-class: table found but row/cell numeric not fully recovered
             (numeric_exact_recovery == false or source_backtrace == false).
    Records that fully recovered are neither D nor B.
    """
    table_recovery = bool(record.get("table_recovery", False))
    if not table_recovery:
        return "D"
    numeric_exact = bool(record.get("numeric_exact_recovery", False))
    source_backtrace = bool(record.get("source_backtrace", False))
    if not numeric_exact or not source_backtrace:
        return "B"
    return "recovered"


# ---------------------------------------------------------------------------
# Presence checks
# ---------------------------------------------------------------------------


def _check_d_class_presence(
    record: dict[str, Any],
    r3_pages: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Check if a D-class target page now has page/table/row in R3."""
    doc_id = str(record.get("document_id") or "")
    pdf_page = int(record.get("pdf_page") or 0)

    page = r3_pages.get((doc_id, pdf_page))
    page_present = page is not None
    table_present = False
    row_present = False
    if page is not None:
        tables = page.get("tables") or []
        table_present = len(tables) > 0
        for table in tables:
            if table.get("rows"):
                row_present = True
                break

    return {
        "oracle_record_id": record.get("oracle_record_id"),
        "case_id": record.get("case_id"),
        "document_id": doc_id,
        "pdf_page": pdf_page,
        "expected_metric": record.get("expected_metric"),
        "page_present": page_present,
        "table_present": table_present,
        "row_present": row_present,
    }


def _check_b_class_structure(
    record: dict[str, Any],
    r3_pages: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Check if a B-class target page now has row/cell structure in R3."""
    doc_id = str(record.get("document_id") or "")
    pdf_page = int(record.get("pdf_page") or 0)

    page = r3_pages.get((doc_id, pdf_page))
    page_present = page is not None
    row_cell_exists = False
    if page is not None:
        for table in page.get("tables") or []:
            rows = table.get("rows") or []
            cells = table.get("cells") or []
            if rows and cells:
                # Check if any cell has parsed_numeric
                has_numeric = any(
                    cell.get("parsed_numeric") for cell in cells
                )
                if has_numeric:
                    row_cell_exists = True
                    break

    return {
        "oracle_record_id": record.get("oracle_record_id"),
        "case_id": record.get("case_id"),
        "document_id": doc_id,
        "pdf_page": pdf_page,
        "expected_metric": record.get("expected_metric"),
        "page_present": page_present,
        "row_cell_structure_exists": row_cell_exists,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-out", type=Path, default=R3_OUT)
    parser.add_argument("--r1-out", type=Path, default=R1_OUT)
    args = parser.parse_args()

    # 1. Verify the seal exists
    seal_path = args.r3_out / "adapter-prediction-seal.json"
    if not seal_path.is_file():
        print(f"ERROR: Seal not found at {seal_path}")
        print("Run seal_pdf_v4_gate_02_r3.py first.")
        return 1
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("sealed"):
        print("ERROR: Seal is not valid (sealed != true).")
        return 1

    # 2. Load Oracle records (post-seal, allowed)
    oracle_records = _load_oracle_records(args.r1_out)
    print(f"Loaded {len(oracle_records)} Oracle records (post-seal)")
    if not oracle_records:
        print("ERROR: No Oracle records found.")
        return 1

    # 3. Load R3 predictions
    predictions_path = args.r3_out / "adapter-predictions.jsonl.gz"
    if not predictions_path.is_file():
        print(f"ERROR: R3 predictions not found at {predictions_path}")
        return 1
    r3_pages = _load_r3_predictions(predictions_path)
    print(f"Loaded {len(r3_pages)} R3 prediction pages")

    # 4. Classify and check
    d_class_records: list[dict[str, Any]] = []
    b_class_records: list[dict[str, Any]] = []
    for record in oracle_records:
        classification = _classify_record(record)
        if classification == "D":
            d_class_records.append(record)
        elif classification == "B":
            b_class_records.append(record)

    d_class_per_record = [
        _check_d_class_presence(rec, r3_pages) for rec in d_class_records
    ]
    b_class_per_record = [
        _check_b_class_structure(rec, r3_pages) for rec in b_class_records
    ]

    d_class_total = len(d_class_records)
    d_class_page_present = sum(1 for r in d_class_per_record if r["page_present"])
    d_class_table_present = sum(1 for r in d_class_per_record if r["table_present"])
    d_class_row_present = sum(1 for r in d_class_per_record if r["row_present"])

    b_class_total = len(b_class_records)
    b_class_row_cell_exists = sum(
        1 for r in b_class_per_record if r["row_cell_structure_exists"]
    )

    result = {
        "schema": "pdf-retrieval-v4/gate-02-r3/d-class-structural-presence/v1",
        "d_class_total": d_class_total,
        "d_class_page_present": d_class_page_present,
        "d_class_table_present": d_class_table_present,
        "d_class_row_present": d_class_row_present,
        "b_class_total": b_class_total,
        "b_class_row_cell_exists": b_class_row_cell_exists,
        "per_record": d_class_per_record + b_class_per_record,
    }

    _write_json(args.r3_out / "d-class-structural-presence.json", result)

    print("\nD-class / B-class structural presence:")
    print(f"  D-class Page Present:  {d_class_page_present}/{d_class_total}")
    print(f"  D-class Table Present: {d_class_table_present}/{d_class_total}")
    print(f"  D-class Row Present:    {d_class_row_present}/{d_class_total}")
    print(f"  B-class Row/Cell structure exists: {b_class_row_cell_exists}/{b_class_total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
