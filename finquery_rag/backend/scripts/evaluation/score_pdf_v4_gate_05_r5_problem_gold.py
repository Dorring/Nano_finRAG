#!/usr/bin/env python3
"""Gate 05 R5 — Score 33 Problem Gold.

Evaluates whether Grade-A Structured Candidate Views cover the 33 Problem
Gold records (17 B-class, 16 D-class) from Gate 03 R2.

Each Problem Gold record has ``matched_row_ids`` — we check if those
rows are covered by any structured view's ``row_ids``.

Table fragmentation handling: PDF extraction often splits one logical
table into multiple table fragments, creating duplicate rows with
different row_ids but the same label on the same page. The scoring
expands matched_row_ids with equivalent rows (same normalized label +
same document + same page) before checking coverage.

Usage:
    python3 scripts/evaluation/score_pdf_v4_gate_05_r5_problem_gold.py

Gates:
    Strong: B >= 15/17, D >= 14/16, Total >= 29/33
    Pass:   B >= 14/17, D >= 12/16, Total >= 26/33
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

OUTPUT_DIR = BACKEND_DIR / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5"
PROBLEM_GOLD_PATH = (
    BACKEND_DIR
    / "artifacts/evaluation/pdf-retrieval-v4-gate-03-r2/problem-gold-scoring.json"
)
SEMANTIC_ROWS_PATH = (
    BACKEND_DIR / "artifacts/evaluation/pdf-retrieval-v4-gate-03-r2/semantic-rows.jsonl"
)


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_label(label: str) -> str:
    """Normalize a row label for equivalence comparison."""
    s = label.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def build_row_equivalence_map() -> dict[str, set[str]]:
    """Build a row_id → set of equivalent row_ids map.

    Two rows are equivalent if they have the same normalized label
    on the same document + page. This handles table fragmentation
    where one logical table is split into multiple fragments.
    """
    # Group rows by (document_id, pdf_page, normalized_label)
    groups: dict[tuple[str, int, str], set[str]] = {}
    row_info: dict[str, tuple[str, int, str]] = {}  # row_id → (doc, page, norm_label)

    if SEMANTIC_ROWS_PATH.exists():
        for rec in read_jsonl(SEMANTIC_ROWS_PATH):
            rid = rec.get("row_id", "")
            if not rid:
                continue
            doc = rec.get("document_id", "")
            page = rec.get("pdf_page", 0)
            label = normalize_label(rec.get("raw_label", ""))
            key = (doc, page, label)
            groups.setdefault(key, set()).add(rid)
            row_info[rid] = key

    # Build equivalence map
    equiv_map: dict[str, set[str]] = {}
    for rid, key in row_info.items():
        equiv_map[rid] = groups.get(key, {rid})

    return equiv_map


def main() -> int:
    print("=" * 70)
    print("Gate 05 R5 — Score 33 Problem Gold")
    print("=" * 70)

    # Load problem gold scoring
    with open(PROBLEM_GOLD_PATH) as f:
        problem_gold = json.load(f)

    b_records = problem_gold.get("b_class", {}).get("records", [])
    d_records = problem_gold.get("d_class", {}).get("records", [])

    print(f"\nB-class records: {len(b_records)}")
    print(f"D-class records: {len(d_records)}")

    # Build row equivalence map from semantic rows
    equiv_map = build_row_equivalence_map()
    print(f"Row equivalence map: {len(equiv_map)} rows")

    # Load structured views
    structured_views = list(read_jsonl(OUTPUT_DIR / "structured-views.jsonl"))

    # Build lookup: row_id → list of structured views containing it
    views_by_row: dict[str, list[dict]] = {}
    for view in structured_views:
        for row_id in view.get("row_ids", []):
            views_by_row.setdefault(row_id, []).append(view)

    # Build lookup by metric_paths for fallback
    views_by_metric_page: dict[tuple[str, int, str], list[dict]] = {}
    for view in structured_views:
        doc = view.get("document_id", "")
        page = view.get("pdf_page", 0)
        for mp in view.get("metric_paths", []):
            nmp = normalize_label(mp)
            views_by_metric_page.setdefault((doc, page, nmp), []).append(view)

    print(f"Structured views: {len(structured_views)}")
    print(f"Views with row_ids: {sum(1 for v in structured_views if v.get('row_ids'))}")

    # Build row_id → (document_id, pdf_page, raw_label) lookup from semantic rows
    row_lookup: dict[str, dict] = {}
    if SEMANTIC_ROWS_PATH.exists():
        for rec in read_jsonl(SEMANTIC_ROWS_PATH):
            rid = rec.get("row_id", "")
            if rid:
                row_lookup[rid] = rec

    def score_records(records: list[dict], class_name: str) -> dict:
        """Score a set of records."""
        passed = 0
        failed = 0
        details: list[dict] = []

        for rec in records:
            record_id = rec.get("record_id") or ""
            document_id = rec.get("document_id") or ""
            matched_row_ids = rec.get("matched_row_ids") or []
            gold_metric_label = rec.get("gold_metric_label") or ""

            # Step 1: Expand matched_row_ids with equivalent rows
            expanded_row_ids: set[str] = set()
            for row_id in matched_row_ids:
                equivs = equiv_map.get(row_id, {row_id})
                expanded_row_ids.update(equivs)

            # Step 2: Check if any expanded row_id is covered by a structured view
            covered = False
            covering_views: list[str] = []
            match_method = "none"

            for row_id in expanded_row_ids:
                if row_id in views_by_row:
                    covered = True
                    match_method = "row_id_expanded"
                    for v in views_by_row[row_id]:
                        ck = v.get("candidate_key", "")
                        if ck not in covering_views:
                            covering_views.append(ck)

            # Step 3: Fallback — check by metric_path + document + page
            if not covered and matched_row_ids:
                # Get page and label from the first matched row
                for row_id in matched_row_ids:
                    row_rec = row_lookup.get(row_id)
                    if row_rec:
                        page = row_rec.get("pdf_page", 0)
                        label = normalize_label(row_rec.get("raw_label", ""))
                        if label:
                            key = (document_id, page, label)
                            if key in views_by_metric_page:
                                covered = True
                                match_method = "metric_path_fallback"
                                for v in views_by_metric_page[key]:
                                    ck = v.get("candidate_key", "")
                                    if ck not in covering_views:
                                        covering_views.append(ck)
                        break

            # Step 4: Fallback — check by gold_metric_label
            if not covered and gold_metric_label:
                # Try all pages for this document
                nmp = normalize_label(gold_metric_label)
                for key, views in views_by_metric_page.items():
                    if key[0] == document_id and key[2] == nmp:
                        covered = True
                        match_method = "gold_metric_label_fallback"
                        for v in views:
                            ck = v.get("candidate_key", "")
                            if ck not in covering_views:
                                covering_views.append(ck)
                        break

            if covered:
                passed += 1
                status = "covered"
            else:
                failed += 1
                status = "not_covered"

            details.append(
                {
                    "record_id": record_id,
                    "document_id": document_id,
                    "matched_row_ids": matched_row_ids[:3],
                    "expanded_row_count": len(expanded_row_ids),
                    "status": status,
                    "match_method": match_method if covered else None,
                    "covering_views": covering_views[:3],
                }
            )

        return {
            "total": len(records),
            "passed": passed,
            "failed": failed,
            "details": details,
        }

    # Score B-class
    b_result = score_records(b_records, "B-class")
    print(f"\nB-class: {b_result['passed']}/{b_result['total']}")

    # Score D-class
    d_result = score_records(d_records, "D-class")
    print(f"D-class: {d_result['passed']}/{d_result['total']}")

    total_passed = b_result["passed"] + d_result["passed"]
    total = b_result["total"] + d_result["total"]

    print(f"Total:  {total_passed}/{total}")

    # Gate evaluation
    b_threshold_strong = 15
    b_threshold_pass = 14
    d_threshold_strong = 14
    d_threshold_pass = 12
    total_threshold_strong = 29
    total_threshold_pass = 26

    if (
        b_result["passed"] >= b_threshold_strong
        and d_result["passed"] >= d_threshold_strong
        and total_passed >= total_threshold_strong
    ):
        gate_status = "strong_pass"
    elif (
        b_result["passed"] >= b_threshold_pass
        and d_result["passed"] >= d_threshold_pass
        and total_passed >= total_threshold_pass
    ):
        gate_status = "pass"
    else:
        gate_status = "fail"

    result = {
        "b_class": {
            "total": b_result["total"],
            "passed": b_result["passed"],
            "failed": b_result["failed"],
            "threshold_strong": b_threshold_strong,
            "threshold_pass": b_threshold_pass,
            "details": b_result["details"],
        },
        "d_class": {
            "total": d_result["total"],
            "passed": d_result["passed"],
            "failed": d_result["failed"],
            "threshold_strong": d_threshold_strong,
            "threshold_pass": d_threshold_pass,
            "details": d_result["details"],
        },
        "total": {
            "passed": total_passed,
            "total": total,
            "threshold_strong": total_threshold_strong,
            "threshold_pass": total_threshold_pass,
        },
        "gate_status": gate_status,
    }

    # Write result
    output_path = OUTPUT_DIR / "problem-gold-scoring.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nGate Status: {gate_status}")
    print(
        f"  Strong: B>={b_threshold_strong}/17, D>={d_threshold_strong}/16, Total>={total_threshold_strong}/33"
    )
    print(
        f"  Pass:   B>={b_threshold_pass}/17, D>={d_threshold_pass}/16, Total>={total_threshold_pass}/33"
    )

    print(f"\nResult written to: {output_path}")
    print("=" * 70)

    return 0 if gate_status != "fail" else 2


if __name__ == "__main__":
    sys.exit(main())
