"""P1.6-A2B-19B: the shadow classifier, graded — and graded only where grading means something.

The denominator is not the corpus.  It is:

  * **906 data tables** -- the 592 page-layout scaffolds left before anything was
    classified, so no metric here is computed over 40% padding;
  * **the hard oracle**: the 42 primary statements, read one by one and all genuine, and
    the 229 dangerous negatives, tables whose own caption names a disaggregation and which
    must never be promoted.  These 271 decide whether the classifier passes;
  * **weak**: the rules' other negative verdicts.  They are the filing declaring the scope
    of its own facts, which is sound, but nobody has checked them one by one.  They are
    reported as `extended` and never as the gate.

The gate, in priority order:

    known dangerous tables promoted to PRIMARY      must be 0
    primary precision on the hard set               must be 1.0
    recall against 42                               must improve on the baseline's 5
    every filing independently                      not Microsoft alone

  python score_table_authority_classifier.py --oracle <dir>/table-authority-oracle.json \
      --shadow <dir>/shadow-classification.json --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

PRIMARY_FAMILIES = {"INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    shadow = json.loads(args.shadow.read_text(encoding="utf-8"))

    report = {"phase": "P1.6-A2B-19B", "mutation": "none", "documents": {}}
    totals = collections.Counter()
    per_document = {}

    print("=== shadow classifier against the hard oracle ===")
    print()
    for document_id in sorted(oracle["documents"]):
        predicted = {
            row["oracle_key"]: row
            for row in shadow["documents"].get(document_id, {}).get("tables", [])
        }
        hard = [t for t in oracle["documents"][document_id]["tables"]
                if t["oracle_role"] in ("PRIMARY", "DANGEROUS_NEGATIVE")]
        weak = [t for t in oracle["documents"][document_id]["tables"]
                if t["oracle_role"] in ("WEAK_NON_PRIMARY", "OPEN")]
        scaffolds = sum(
            1 for t in oracle["documents"][document_id]["tables"]
            if t["table_eligibility"] == "LAYOUT_SCAFFOLD"
        )

        cells = collections.Counter()
        dangerous_promoted = []
        false_negative = []
        unsupported = []
        for table in hard:
            row = predicted.get(table["oracle_key"])
            if row is None:
                cells["UNCLASSIFIED"] += 1
                continue
            is_primary = row["new_table_role"] == "PRIMARY_FINANCIAL_STATEMENT"
            if table["oracle_role"] == "PRIMARY":
                if is_primary:
                    cells["TP"] += 1
                    if row["status"] != "SOURCE_SUPPORTED":
                        unsupported.append(row)
                else:
                    cells["FN"] += 1
                    false_negative.append({"oracle_key": table["oracle_key"],
                                           "predicted": row["new_table_role"],
                                           "title": table["title"]})
            else:
                if is_primary:
                    cells["DANGEROUS_PROMOTED"] += 1
                    dangerous_promoted.append({
                        "oracle_key": table["oracle_key"],
                        "danger": table["danger_line"],
                        "score": row["score"],
                        "evidence": [e["code"] for e in row["evidence"]],
                    })
                else:
                    cells["TN"] += 1

        # The weak set: reported, never decisive.
        extended = collections.Counter()
        for table in weak:
            row = predicted.get(table["oracle_key"])
            if row is None:
                continue
            is_primary = row["new_table_role"] == "PRIMARY_FINANCIAL_STATEMENT"
            if table["oracle_role"] == "WEAK_NON_PRIMARY":
                extended["EXT_FP" if is_primary else "EXT_TN"] += 1
            else:
                # OPEN: the oracle did not decide these, so a promotion here is
                # neither counted for nor against -- it is listed for reading.
                extended["OPEN_PROMOTED" if is_primary else "OPEN_OTHER"] += 1

        denom = cells["TP"] + cells["DANGEROUS_PROMOTED"]
        precision = cells["TP"] / denom if denom else None

        print(f"--- {document_id}  data tables {len(hard) + len(weak)}  "
              f"(scaffolds excluded {scaffolds})")
        print(f"    hard  TP {cells['TP']:>3}  FN {cells['FN']:>3}  "
              f"TN {cells['TN']:>3}  dangerous-promoted {cells['DANGEROUS_PROMOTED']}")
        print(f"    recall {cells['TP']}/42-of-document   precision {precision}")
        print(f"    weak  {dict(extended)}")
        for row in dangerous_promoted[:6]:
            print(f"      !! PROMOTED {row['oracle_key']} score={row['score']} "
                  f"{str(row['danger'])[:44]!r} {row['evidence']}")
        for row in false_negative[:8]:
            print(f"      FN {row['oracle_key']:>20} predicted={row['predicted']:26} "
                  f"{str(row['title'])[:40]!r}")
        print()

        report["documents"][document_id] = {
            "cells": dict(cells), "extended": dict(extended),
            "precision": precision,
            "dangerous_promoted": dangerous_promoted,
            "false_negative": false_negative,
            "primary_without_source_support": unsupported,
            "scaffolds_excluded": scaffolds,
        }
        totals.update(cells)
        totals.update({f"EXT_{k}": v for k, v in extended.items()})
        per_document[document_id] = dict(cells)

    denom = totals["TP"] + totals["DANGEROUS_PROMOTED"]
    precision = totals["TP"] / denom if denom else None
    recall = totals["TP"] / 42
    baseline_recall = 5 / 42
    report.update({"totals": dict(totals), "per_document": per_document,
                   "precision": precision, "recall": recall,
                   "baseline_recall": baseline_recall})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "shadow-vs-hard-oracle.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print("=== hard oracle totals ===")
    print(f"  TP {totals['TP']}   FN {totals['FN']}   TN {totals['TN']}   "
          f"dangerous-promoted {totals['DANGEROUS_PROMOTED']}")
    print(f"  recall {totals['TP']}/42 = {recall:.3f}  (baseline {baseline_recall:.3f})")
    print(f"  precision {precision}")
    print()
    print("=== the gate ===")
    print(f"  dangerous tables promoted to PRIMARY : {totals['DANGEROUS_PROMOTED']}  "
          f"{'PASS' if totals['DANGEROUS_PROMOTED'] == 0 else 'FAIL'}")
    print(f"  hard-set primary precision           : {precision}  "
          f"{'PASS' if precision == 1.0 else 'FAIL'}")
    print(f"  recall improves on 5/42              : {totals['TP']}/42  "
          f"{'PASS' if totals['TP'] > 5 else 'FAIL'}")
    per_doc_ok = all(v.get("TP", 0) > 0 for v in per_document.values())
    print(f"  every filing improves independently  : "
          f"{'PASS' if per_doc_ok else 'FAIL'} "
          f"({ {d: c.get('TP', 0) for d, c in per_document.items()} })")
    print()
    print("=== weak / extended, reported separately ===")
    for name in ("EXT_EXT_FP", "EXT_EXT_TN", "EXT_OPEN_PROMOTED", "EXT_OPEN_OTHER"):
        if totals.get(name):
            print(f"  {name:22} {totals[name]}")
    print()
    print(f"  written to {args.out / 'shadow-vs-hard-oracle.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
