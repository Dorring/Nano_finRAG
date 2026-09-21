#!/usr/bin/env python3
"""Gate 05 R5.1 — Problem-Gold Scoring Equivalence Closure.

Tightens the Problem-Gold scoring equivalence rules from R5.

R5 used loose label-based equivalence (same document + same page +
same normalized label), which over-expanded some records to 22 rows.
R5.1 replaces this with the frozen ``semantic_equivalence_group_id``
from Gate 02 R3.2 R1 ``ambiguity-closure.json`` — only rows that
were explicitly verified as duplicate physical fragments of the same
canonical semantic evidence are treated as equivalent.

Formal pass logic allows only:
  1. ``direct_row_match`` — matched_row_id directly in a structured view
  2. ``canonical_equivalent_match`` — matched_row_id → frozen equivalent
     group → canonical member in a structured view

Diagnostic-only (not counted for formal pass):
  - ``diagnostic_label_match`` — same label on same page (loose)
  - ``diagnostic_metric_fallback`` — metric_path + page match

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
AMBIGUITY_CLOSURE_PATH = (
    BACKEND_DIR
    / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r3/ambiguity-closure.json"
)


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_label(label: str) -> str:
    """Normalize a row label for diagnostic comparison only."""
    s = label.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def build_frozen_equivalence_map() -> dict[str, set[str]]:
    """Build row_id → set of equivalent row_ids from frozen R3.2 R1 data.

    Loads ``ambiguity-closure.json`` from Gate 02 R3 and extracts records
    with ``alignment_status == "equivalent_set"``. Each record's
    ``equivalent_row_ids`` list defines a group of physical rows that
    are duplicate fragments of the same canonical semantic evidence.

    Only rows in these explicitly verified groups are treated as
    equivalent — no loose label-based expansion.
    """
    equiv_map: dict[str, set[str]] = {}

    if not AMBIGUITY_CLOSURE_PATH.exists():
        print(
            f"  WARNING: ambiguity-closure.json not found at {AMBIGUITY_CLOSURE_PATH}"
        )
        return equiv_map

    data = json.loads(AMBIGUITY_CLOSURE_PATH.read_text(encoding="utf-8"))
    records = data.get("records") or []
    equiv_records = [
        r for r in records if r.get("alignment_status") == "equivalent_set"
    ]

    for rec in equiv_records:
        row_ids = rec.get("equivalent_row_ids") or rec.get("physical_row_ids") or []
        if not row_ids:
            sources = rec.get("physical_sources") or []
            row_ids = [s.get("row_id") for s in sources if s.get("row_id")]
        if not row_ids:
            row_evidence = rec.get("row_evidence") or []
            row_ids = [re_.get("row_id") for re_ in row_evidence if re_.get("row_id")]
        if len(row_ids) < 2:
            continue
        group = set(row_ids)
        for rid in group:
            equiv_map[rid] = group

    return equiv_map


def build_diagnostic_label_map() -> dict[str, set[str]]:
    """Build loose label-based equivalence map for diagnostic purposes only.

    This is NOT used for formal pass logic — only for diagnostic reporting
    to show which records would have passed under the old R5 rules.
    """
    groups: dict[tuple[str, int, str], set[str]] = {}
    row_info: dict[str, tuple[str, int, str]] = {}

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

    equiv_map: dict[str, set[str]] = {}
    for rid, key in row_info.items():
        equiv_map[rid] = groups.get(key, {rid})

    return equiv_map


def main() -> int:
    print("=" * 70)
    print("Gate 05 R5.1 — Problem-Gold Scoring Equivalence Closure")
    print("=" * 70)

    # Load problem gold scoring
    with open(PROBLEM_GOLD_PATH) as f:
        problem_gold = json.load(f)

    b_records = problem_gold.get("b_class", {}).get("records", [])
    d_records = problem_gold.get("d_class", {}).get("records", [])

    print(f"\nB-class records: {len(b_records)}")
    print(f"D-class records: {len(d_records)}")

    # Build frozen equivalence map from R3.2 R1 ambiguity-closure
    frozen_equiv = build_frozen_equivalence_map()
    print(f"Frozen equivalence map: {len(frozen_equiv)} rows in verified groups")

    # Build diagnostic label map (NOT used for formal pass)
    diagnostic_label_map = build_diagnostic_label_map()
    print(f"Diagnostic label map: {len(diagnostic_label_map)} rows (not used for pass)")

    # Load structured views
    structured_views = list(read_jsonl(OUTPUT_DIR / "structured-views.jsonl"))

    # Build lookup: row_id → list of structured views containing it
    views_by_row: dict[str, list[dict]] = {}
    for view in structured_views:
        for row_id in view.get("row_ids", []):
            views_by_row.setdefault(row_id, []).append(view)

    # Build diagnostic lookup by metric_paths + page (NOT used for formal pass)
    views_by_metric_page: dict[tuple[str, int, str], list[dict]] = {}
    for view in structured_views:
        doc = view.get("document_id", "")
        page = view.get("pdf_page", 0)
        for mp in view.get("metric_paths", []):
            nmp = normalize_label(mp)
            views_by_metric_page.setdefault((doc, page, nmp), []).append(view)

    print(f"Structured views: {len(structured_views)}")
    print(f"Views with row_ids: {sum(1 for v in structured_views if v.get('row_ids'))}")

    # Build row_id → semantic row info lookup
    row_lookup: dict[str, dict] = {}
    if SEMANTIC_ROWS_PATH.exists():
        for rec in read_jsonl(SEMANTIC_ROWS_PATH):
            rid = rec.get("row_id", "")
            if rid:
                row_lookup[rid] = rec

    def score_records(records: list[dict]) -> dict:
        """Score a set of records with strict frozen equivalence."""
        passed = 0
        failed = 0
        details: list[dict] = []

        for rec in records:
            record_id = rec.get("record_id") or ""
            document_id = rec.get("document_id") or ""
            matched_row_ids = rec.get("matched_row_ids") or []
            gold_metric_label = rec.get("gold_metric_label") or ""

            formal_match_method = None
            diagnostic_match_method = None
            covering_views: list[str] = []

            # === FORMAL PASS LOGIC ===

            # Step 1: Direct row_id match
            for row_id in matched_row_ids:
                if row_id in views_by_row:
                    formal_match_method = "direct_row_match"
                    for v in views_by_row[row_id]:
                        ck = v.get("candidate_key", "")
                        if ck not in covering_views:
                            covering_views.append(ck)
                    break

            # Step 2: Frozen equivalent group match
            if formal_match_method is None:
                for row_id in matched_row_ids:
                    equiv_group = frozen_equiv.get(row_id)
                    if not equiv_group:
                        continue
                    for equiv_row in equiv_group:
                        if equiv_row in views_by_row:
                            formal_match_method = "canonical_equivalent_match"
                            for v in views_by_row[equiv_row]:
                                ck = v.get("candidate_key", "")
                                if ck not in covering_views:
                                    covering_views.append(ck)
                            break
                    if formal_match_method is not None:
                        break

            # === DIAGNOSTIC-ONLY (not counted for formal pass) ===

            # Diagnostic: loose label match
            if formal_match_method is None:
                for row_id in matched_row_ids:
                    label_equivs = diagnostic_label_map.get(row_id, {row_id})
                    for equiv_row in label_equivs:
                        if equiv_row in views_by_row:
                            diagnostic_match_method = "diagnostic_label_match"
                            break
                    if diagnostic_match_method is not None:
                        break

            # Diagnostic: metric_path + page match
            if formal_match_method is None and diagnostic_match_method is None:
                for row_id in matched_row_ids:
                    row_rec = row_lookup.get(row_id)
                    if row_rec:
                        page = row_rec.get("pdf_page", 0)
                        label = normalize_label(row_rec.get("raw_label", ""))
                        if label:
                            key = (document_id, page, label)
                            if key in views_by_metric_page:
                                diagnostic_match_method = "diagnostic_metric_fallback"
                        break

            # Diagnostic: gold_metric_label fallback
            if formal_match_method is None and diagnostic_match_method is None:
                if gold_metric_label:
                    nmp = normalize_label(gold_metric_label)
                    for key in views_by_metric_page:
                        if key[0] == document_id and key[2] == nmp:
                            diagnostic_match_method = "diagnostic_metric_fallback"
                            break

            # Determine formal status
            covered = formal_match_method is not None
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
                    "status": status,
                    "formal_match_method": formal_match_method,
                    "diagnostic_match_method": diagnostic_match_method,
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
    b_result = score_records(b_records)
    print(f"\nB-class (strict): {b_result['passed']}/{b_result['total']}")

    # Score D-class
    d_result = score_records(d_records)
    print(f"D-class (strict): {d_result['passed']}/{d_result['total']}")

    total_passed = b_result["passed"] + d_result["passed"]
    total = b_result["total"] + d_result["total"]

    print(f"Total (strict):   {total_passed}/{total}")

    # Report match method breakdown
    from collections import Counter

    b_formal = Counter(
        d["formal_match_method"]
        for d in b_result["details"]
        if d["formal_match_method"]
    )
    d_formal = Counter(
        d["formal_match_method"]
        for d in d_result["details"]
        if d["formal_match_method"]
    )
    b_diag = Counter(
        d["diagnostic_match_method"]
        for d in b_result["details"]
        if d["diagnostic_match_method"]
    )
    d_diag = Counter(
        d["diagnostic_match_method"]
        for d in d_result["details"]
        if d["diagnostic_match_method"]
    )

    print("\n=== Match Method Breakdown ===")
    print(f"B-class formal:   {dict(b_formal)}")
    print(f"B-class diagnostic-only: {dict(b_diag)}")
    print(f"D-class formal:   {dict(d_formal)}")
    print(f"D-class diagnostic-only: {dict(d_diag)}")

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
        "scoring_version": "r5.1_strict_frozen_equivalence",
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
        "formal_match_methods": {
            "b_class": dict(b_formal),
            "d_class": dict(d_formal),
        },
        "diagnostic_match_methods": {
            "b_class": dict(b_diag),
            "d_class": dict(d_diag),
        },
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
