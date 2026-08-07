"""Gate 02 R3.1: Benchmark Structural Presence Closure.

Runs AFTER the Gate 02 R3 Prediction Seal is verified.  Reads 33 Gold Source
records (16 D-class from Gate 08 R1.1 + 17 B-class unrecovered from Gate 08 R2)
and checks whether the R3 full-corpus adapter now provides structural coverage
for each record's target page.

Five-layer coverage check per record:
  L0 Page           — page record exists in R3 predictions
  L1 Table/Narrative — at least one table or text block exists on the page
  L2 Row             — at least one row exists in a table on the page
  L3 Cell            — at least one cell with text exists in a table
  L4 Candidate-compat — at least one cell with parsed_numeric or non-empty
                       resolved_text that could serve as evidence

This script does NOT modify the adapter or re-run it.  It only reads sealed
R3 predictions and the Gate 08 R1.1/R2 classification files.

Reads ONLY the sealed R3 predictions and the Gate 08 classification files.
No questions, gold answers, or governance data is read.
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
R1_1_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r1-1"
R2_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


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


def _load_d_class_records(r1_1_out: Path) -> list[dict[str, Any]]:
    """Load 16 D-class records from Gate 08 R1.1 classification."""
    path = r1_1_out / "gold-coverage-classification.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = (
        data.get("rows", data.get("records", data)) if isinstance(data, dict) else data
    )
    d_class = [
        r
        for r in rows
        if isinstance(r, dict) and r.get("coverage_class") == "structurally_absent"
    ]
    for r in d_class:
        r["benchmark_class"] = "D"
    return d_class


def _load_b_class_unrecovered(r1_1_out: Path, r2_out: Path) -> list[dict[str, Any]]:
    """Load 17 B-class unrecovered records, enriched with R1.1 metadata."""
    # Get document_id and pdf_page from R1.1 B-class records
    r11_path = r1_1_out / "gold-coverage-classification.json"
    r11_data = json.loads(r11_path.read_text(encoding="utf-8"))
    r11_rows = (
        r11_data.get("rows", r11_data.get("records", r11_data))
        if isinstance(r11_data, dict)
        else r11_data
    )
    b_r11_by_identity: dict[str, dict[str, Any]] = {}
    for r in r11_rows:
        if (
            isinstance(r, dict)
            and r.get("coverage_class") == "strict_mapped_not_retrieved"
        ):
            b_r11_by_identity[str(r.get("gold_source_identity") or "")] = r

    # Get unrecovered B-class from R2
    b_path = r2_out / "scoring/b-class-detail.json"
    b_data = json.loads(b_path.read_text(encoding="utf-8"))
    b_rows = (
        b_data.get("rows", b_data.get("records", b_data))
        if isinstance(b_data, dict)
        else b_data
    )
    b_unrecovered = [
        r for r in b_rows if isinstance(r, dict) and not r.get("recovered", True)
    ]

    # Enrich with R1.1 metadata
    enriched: list[dict[str, Any]] = []
    for r in b_unrecovered:
        identity = str(r.get("gold_source_identity") or "")
        r11_record = b_r11_by_identity.get(identity, {})
        enriched.append(
            {
                "gold_source_identity": identity,
                "case_id": r.get("case_id") or r11_record.get("case_id"),
                "document_id": r11_record.get("document_id"),
                "pdf_page": r11_record.get("pdf_page"),
                "gold_candidate_key": r.get("gold_candidate_key"),
                "recovered": r.get("recovered"),
                "rank_bucket": r.get("rank_bucket"),
                "rank_in_candidate_direct_pool": r.get("rank_in_candidate_direct_pool"),
                "benchmark_class": "B",
            }
        )
    return enriched


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------


def _check_coverage(
    record: dict[str, Any],
    predictions: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Check 5-layer structural coverage for a single record."""
    doc_id = str(record.get("document_id") or "")
    pdf_page = int(record.get("pdf_page") or 0)
    identity = str(record.get("gold_source_identity") or "")
    case_id = str(record.get("case_id") or "")
    benchmark_class = str(record.get("benchmark_class") or "")

    page_record = predictions.get((doc_id, pdf_page))

    # L0 Page
    page_present = page_record is not None

    # L1 Table / Narrative Block
    tables: list[dict[str, Any]] = []
    text_block_count = 0
    native_text_present = False
    if page_record:
        tables = page_record.get("tables", []) or []
        text_block_count = int(page_record.get("text_block_count", 0) or 0)
        native_text_present = bool(page_record.get("native_text_present"))

    table_present = len(tables) > 0
    narrative_present = text_block_count > 0 or native_text_present

    # L2 Row
    all_rows: list[dict[str, Any]] = []
    for table in tables:
        all_rows.extend(table.get("rows", []) or [])
    row_present = len(all_rows) > 0

    # L3 Cell
    all_cells: list[dict[str, Any]] = []
    for table in tables:
        all_cells.extend(table.get("cells", []) or [])
    cells_with_text = [
        c for c in all_cells if c.get("resolved_text") or c.get("raw_text")
    ]
    cell_present = len(cells_with_text) > 0

    # L4 Candidate-compatible Structure
    # A cell is candidate-compatible if it has parsed_numeric values or
    # non-empty resolved text that could serve as evidence.
    candidate_compatible_cells = [
        c
        for c in all_cells
        if c.get("parsed_numeric")
        or (c.get("resolved_text") and len(str(c["resolved_text"]).strip()) > 0)
    ]
    candidate_compatible_structure = len(candidate_compatible_cells) > 0

    # Determine best table/row/cell IDs (first table with most cells)
    best_table_id = None
    best_row_id = None
    best_cell_ids: list[str] = []
    if tables:
        best_table = max(tables, key=lambda t: len(t.get("cells", []) or []))
        best_table_id = str(best_table.get("table_fragment_id") or "")
        best_rows = best_table.get("rows", []) or []
        if best_rows:
            best_row_id = str(best_rows[0].get("row_id") or "")
        best_cells = best_table.get("cells", []) or []
        best_cell_ids = [
            str(c.get("cell_id") or "") for c in best_cells[:5] if c.get("cell_id")
        ]

    # Coverage level: highest layer reached
    coverage_level = "none"
    if page_present:
        coverage_level = "page"
        if table_present or narrative_present:
            coverage_level = "table" if table_present else "narrative"
            if row_present:
                coverage_level = "row"
                if cell_present:
                    coverage_level = "cell"
                    if candidate_compatible_structure:
                        coverage_level = "candidate_compatible"

    return {
        "case_id": case_id,
        "gold_source_identity": identity,
        "benchmark_class": benchmark_class,
        "document_id": doc_id,
        "pdf_page": pdf_page,
        "page_present": page_present,
        "table_present": table_present,
        "narrative_present": narrative_present,
        "row_present": row_present,
        "cell_present": cell_present,
        "candidate_compatible_structure": candidate_compatible_structure,
        "best_table_id": best_table_id,
        "best_row_id": best_row_id,
        "best_cell_ids": best_cell_ids,
        "coverage_level": coverage_level,
        "table_count": len(tables),
        "row_count": len(all_rows),
        "cell_count": len(all_cells),
        "cells_with_text": len(cells_with_text),
        "candidate_compatible_cell_count": len(candidate_compatible_cells),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-out", type=Path, default=R3_OUT)
    parser.add_argument("--r1-1-out", type=Path, default=R1_1_OUT)
    parser.add_argument("--r2-out", type=Path, default=R2_OUT)
    args = parser.parse_args()

    # 1. Verify the R3 seal exists and is valid
    seal_path = args.r3_out / "adapter-prediction-seal.json"
    if not seal_path.is_file():
        print(f"ERROR: Seal not found at {seal_path}")
        print("Run Gate 02 R3 first.")
        return 1
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("sealed"):
        print("ERROR: Seal is not valid (sealed != true).")
        return 1

    # 2. Load R3 predictions
    predictions_path = args.r3_out / "adapter-predictions.jsonl.gz"
    if not predictions_path.is_file():
        print(f"ERROR: R3 predictions not found at {predictions_path}")
        return 1
    predictions = _load_r3_predictions(predictions_path)
    print(f"Loaded {len(predictions)} R3 prediction pages")

    # 3. Load 33 Gold Source records (post-seal, allowed)
    d_class = _load_d_class_records(args.r1_1_out)
    b_class = _load_b_class_unrecovered(args.r1_1_out, args.r2_out)
    print(f"Loaded {len(d_class)} D-class records from Gate 08 R1.1")
    print(f"Loaded {len(b_class)} B-class unrecovered records from Gate 08 R2")

    # 4. Check coverage for each record
    d_results = [_check_coverage(r, predictions) for r in d_class]
    b_results = [_check_coverage(r, predictions) for r in b_class]

    # 5. Aggregate metrics
    def _count(results: list[dict[str, Any]], field: str) -> int:
        return sum(1 for r in results if r.get(field))

    d_metrics = {
        "total": len(d_results),
        "page_present": _count(d_results, "page_present"),
        "table_present": _count(d_results, "table_present"),
        "narrative_present": _count(d_results, "narrative_present"),
        "row_present": _count(d_results, "row_present"),
        "cell_present": _count(d_results, "cell_present"),
        "candidate_compatible_structure": _count(
            d_results, "candidate_compatible_structure"
        ),
    }
    b_metrics = {
        "total": len(b_results),
        "page_present": _count(b_results, "page_present"),
        "table_present": _count(b_results, "table_present"),
        "narrative_present": _count(b_results, "narrative_present"),
        "row_present": _count(b_results, "row_present"),
        "cell_present": _count(b_results, "cell_present"),
        "candidate_compatible_structure": _count(
            b_results, "candidate_compatible_structure"
        ),
    }

    # Newly structurally recoverable: records that were previously
    # structurally absent (D) or unrecovered (B) but now have row/cell
    newly_recoverable_d = sum(
        1 for r in d_results if r.get("row_present") or r.get("cell_present")
    )
    newly_recoverable_b = sum(
        1 for r in b_results if r.get("row_present") or r.get("cell_present")
    )

    # 6. Decision
    d_row = d_metrics["row_present"]
    b_row = b_metrics["row_present"]
    d_total = d_metrics["total"]
    b_total = b_metrics["total"]

    if d_row >= 12 and b_row >= 14:
        decision = "full_corpus_benchmark_structural_presence_closed"
        next_gate = "full_corpus_financial_semantic_graph"
        strength = "strong"
    elif d_row >= 8 and b_row >= 12:
        decision = "full_corpus_benchmark_structural_presence_closed"
        next_gate = "full_corpus_financial_semantic_graph"
        strength = "acceptable"
    else:
        decision = "full_corpus_structural_presence_insufficient"
        next_gate = "stop_and_classify_missing_evidence_shapes"
        strength = "insufficient"

    result = {
        "schema": "pdf-retrieval-v4/gate-02-r3.1/benchmark-structural-presence-closure/v1",
        "seal_verified": True,
        "d_class_metrics": d_metrics,
        "b_class_metrics": b_metrics,
        "newly_structurally_recoverable": {
            "d_class": newly_recoverable_d,
            "b_class": newly_recoverable_b,
            "total": newly_recoverable_d + newly_recoverable_b,
        },
        "d_class_records": d_results,
        "b_class_records": b_results,
        "strength": strength,
        "decision": decision,
        "next_gate": next_gate,
        "production_switch_allowed": False,
    }

    output_path = args.r3_out / "benchmark-structural-presence-closure.json"
    _write_json(output_path, result)

    print("\nBenchmark Structural Presence Closure:")
    print(f"  D-class ({d_total} records):")
    print(f"    Page Present:       {d_metrics['page_present']}/{d_total}")
    print(f"    Table Present:      {d_metrics['table_present']}/{d_total}")
    print(f"    Narrative Present:  {d_metrics['narrative_present']}/{d_total}")
    print(f"    Row Present:        {d_metrics['row_present']}/{d_total}")
    print(f"    Cell Present:       {d_metrics['cell_present']}/{d_total}")
    print(
        f"    Candidate-compat:   {d_metrics['candidate_compatible_structure']}/{d_total}"
    )
    print(f"  B-class ({b_total} records):")
    print(f"    Page Present:       {b_metrics['page_present']}/{b_total}")
    print(f"    Table Present:      {b_metrics['table_present']}/{b_total}")
    print(f"    Narrative Present:  {b_metrics['narrative_present']}/{b_total}")
    print(f"    Row Present:        {b_metrics['row_present']}/{b_total}")
    print(f"    Cell Present:       {b_metrics['cell_present']}/{b_total}")
    print(
        f"    Candidate-compat:   {b_metrics['candidate_compatible_structure']}/{b_total}"
    )
    print(
        f"  Newly Structurally Recoverable: {newly_recoverable_d + newly_recoverable_b}"
    )
    print(f"  Strength: {strength}")
    print(f"  Decision: {decision}")
    print(f"  Next gate: {next_gate}")

    return 0 if decision == "full_corpus_benchmark_structural_presence_closed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
