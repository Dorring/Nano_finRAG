"""P1.6-A2B-19B baseline: the current statement classifier, measured against the oracle.

Nothing is changed here.  The oracle from A2B-19A says which tables may speak for
the company; this says what the classifier currently claims, and puts the two side
by side.  That is the whole point of building the oracle first: without it, "the
classifier misses 120 primary statements" is a number from the same rules it is
criticising, and the false-positive side -- the side that matters, because
promoting a note to company-level authority is worse than missing a statement --
could not be counted at all.

The join is `(document_id, source_order)`: the index of the `<table>` in
`root.iter()`.  The parser records it on every block, and the oracle records it
independently, so neither has to reproduce the other's `table_id`.

  python score_table_authority_oracle.py --oracle <dir>/table-authority-oracle.json --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

CORPUS = Path("/disk/qh/nano-finrag/data/financial_corpus_v2")
PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"

DOCUMENTS = {
    "aapl_fy2025": ("AAPL", "SEC_320193_000032019325000079"),
    "jpm_fy2025": ("JPM", "SEC_19617_000162828026008131"),
    "ko_fy2025": ("KO", "SEC_21344_000162828026010047"),
    "msft_fy2025": ("MSFT", "SEC_789019_000095017025100235"),
    "nvda_fy2025": ("NVDA", "SEC_1045810_000104581025000023"),
    "pfe_fy2024": ("PFE", "SEC_78003_000007800325000054"),
    "tsla_fy2025": ("TSLA", "SEC_1318605_000162828026003952"),
    "v_fy2025": ("V", "SEC_1403161_000140316125000089"),
}

PRIMARY_FAMILIES = {"INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"}


def current_labels(document_id: str, ticker: str, accession: str) -> dict[int, dict]:
    """What the classifier says now, keyed by the table's index in document order."""
    from lxml import etree, html

    spec = importlib.util.spec_from_file_location("nf17a4", PARSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    raw = CORPUS / "raw/SEC" / ticker / accession / "primary.html"
    root = html.parse(
        str(raw),
        etree.HTMLParser(recover=True, no_network=True, huge_tree=True,
                         remove_comments=True),
    ).getroot()
    doc = {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"}
    blocks, _lookup, _prior = module.make_blocks(root, doc)
    return {
        block["source_order"]: {
            "section_type": block["section_type"],
            "table_id": block["table_id"],
        }
        for block in blocks
        if block["block_type"] == "TABLE"
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--adjudications", type=Path, default=None,
        help="the hand-read verdicts for tables the oracle left UNRESOLVED "
             "(scripts/evaluation/table_authority_adjudications.json)",
    )
    args = parser.parse_args(argv)

    adjudicated: dict[str, dict] = {}
    if args.adjudications and args.adjudications.is_file():
        for row in json.loads(args.adjudications.read_text(encoding="utf-8")
                              ).get("adjudications", []):
            adjudicated[row["oracle_key"]] = row

    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    report = {"phase": "P1.6-A2B-19B-baseline", "mutation": "none", "documents": {}}
    totals = collections.Counter()
    per_document = {}

    print("=== current classifier against the oracle ===")
    print()
    for document_id in sorted(oracle["documents"]):
        record = oracle["documents"][document_id]
        ticker, accession = DOCUMENTS[document_id]
        current = current_labels(document_id, ticker, accession)

        cells = collections.Counter()
        false_positive = []   # oracle says NON_PRIMARY, classifier says primary
        false_negative = []   # oracle says PRIMARY, classifier says otherwise
        silent = []           # classifier says primary, oracle could not decide
        unjoined = 0

        for table in record["tables"]:
            now = current.get(table["source_order"])
            if now is None:
                unjoined += 1
                continue
            says_primary = now["section_type"] in PRIMARY_FAMILIES
            is_primary = table["verdict"]["label"] == "PRIMARY_FINANCIAL_STATEMENT"
            is_negative = table["verdict"]["label"] == "NON_PRIMARY"
            if says_primary and is_primary:
                cells["TP"] += 1
            elif says_primary and is_negative:
                cells["FP"] += 1
                false_positive.append({
                    "oracle_key": table["oracle_key"],
                    "current": now["section_type"],
                    "oracle_reason": table["verdict"]["reason"],
                    "label_line": table["label_line"],
                    "body_head": table["body_head"][:120],
                })
            elif not says_primary and is_primary:
                cells["FN"] += 1
                false_negative.append({
                    "oracle_key": table["oracle_key"],
                    "current": now["section_type"],
                    "family": table["verdict"]["family"],
                    "title": table["title"],
                })
            elif is_negative:
                cells["TN"] += 1
            else:
                # The oracle did not decide this table.  Counting it as either a
                # success or a failure would be an assumption; it is counted
                # separately, and when the classifier *promotes* such a table that
                # is the number that matters, because nothing the filing states has
                # yet contradicted it.
                cells["ORACLE_SILENT"] += 1
                if says_primary:
                    read = adjudicated.get(table["oracle_key"])
                    if read is None:
                        cells["PROMOTED_WHILE_ORACLE_SILENT"] += 1
                        silent.append({
                            "oracle_key": table["oracle_key"],
                            "current": now["section_type"],
                            "oracle_reason": table["verdict"]["reason"],
                            "label_line": table["label_line"],
                            "body_head": table["body_head"][:120],
                        })
                    elif read["label"] == "PRIMARY_FINANCIAL_STATEMENT":
                        cells["TP_ADJUDICATED"] += 1
                    else:
                        cells["FP_ADJUDICATED"] += 1
                        false_positive.append({
                            "oracle_key": table["oracle_key"],
                            "current": now["section_type"],
                            "oracle_reason": f"adjudicated: {read['reason']}",
                            "label_line": table["label_line"],
                            "body_head": table["body_head"][:120],
                            "evidence": read["evidence"],
                        })

        true_positive = cells["TP"] + cells["TP_ADJUDICATED"]
        false_positive_total = cells["FP"] + cells["FP_ADJUDICATED"]
        decided = true_positive + false_positive_total
        precision = true_positive / decided if decided else None
        recall = true_positive / (true_positive + cells["FN"]) if (true_positive + cells["FN"]) else None

        print(f"--- {document_id}")
        print(f"    TP {cells['TP']:>3} (+{cells['TP_ADJUDICATED']} adjudicated)  "
              f"FP {cells['FP']:>3} (+{cells['FP_ADJUDICATED']} adjudicated)  "
              f"FN {cells['FN']:>3}  TN {cells['TN']:>4}  "
              f"still-silent {cells['PROMOTED_WHILE_ORACLE_SILENT']:>3}")
        print(f"    precision {precision if precision is None else round(precision, 3)}"
              f"   recall {recall if recall is None else round(recall, 3)}")
        # What the classifier calls primary, all of it, however the oracle answered.
        called = true_positive + false_positive_total + cells["PROMOTED_WHILE_ORACLE_SILENT"]
        print(f"    the classifier calls {called} table(s) primary: "
              f"{true_positive} true, {false_positive_total} not, "
              f"{cells['PROMOTED_WHILE_ORACLE_SILENT']} unjudged")
        for row in false_positive[:8]:
            print(f"      FP {row['oracle_key']:>20} current={row['current']:16} "
                  f"oracle={row['oracle_reason'][:34]:34} {str(row['label_line'])[:40]!r}")
        for row in silent[:8]:
            print(f"      ?? {row['oracle_key']:>20} current={row['current']:16} "
                  f"oracle={row['oracle_reason']:26} {str(row['label_line'])[:40]!r}")
        for row in false_negative[:8]:
            print(f"      FN {row['oracle_key']:>20} current={row['current']:16} "
                  f"family={row['family']:20} {str(row['title'])[:44]!r}")
        print()

        report["documents"][document_id] = {
            "cells": dict(cells), "precision": precision, "recall": recall,
            "unjoined": unjoined, "called_primary": called,
            "true_positive": true_positive, "false_positive": false_positive_total,
            "false_positive_detail": false_positive,
            "false_negative": false_negative,
            "promoted_while_oracle_silent": silent,
        }
        totals.update(cells)
        per_document[document_id] = dict(cells)

    true_positive = totals["TP"] + totals["TP_ADJUDICATED"]
    false_positive_total = totals["FP"] + totals["FP_ADJUDICATED"]
    decided = true_positive + false_positive_total
    precision = true_positive / decided if decided else None
    recall = true_positive / (true_positive + totals["FN"]) if (true_positive + totals["FN"]) else None
    report.update({"totals": dict(totals), "per_document": per_document,
                   "true_positive": true_positive,
                   "false_positive": false_positive_total,
                   "precision": precision, "recall": recall,
                   "adjudicated_keys": sorted(adjudicated)})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baseline-vs-oracle.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print("=== totals ===")
    print(f"  TP {totals['TP']} + {totals['TP_ADJUDICATED']} adjudicated   "
          f"FP {totals['FP']} + {totals['FP_ADJUDICATED']} adjudicated   "
          f"FN {totals['FN']}   TN {totals['TN']}")
    print(f"  precision {precision}   recall {recall}")
    print()
    print(f"  the classifier calls "
          f"{true_positive + false_positive_total + totals['PROMOTED_WHILE_ORACLE_SILENT']} "
          f"tables primary: {true_positive} true, {false_positive_total} not, "
          f"{totals['PROMOTED_WHILE_ORACLE_SILENT']} unjudged")
    print()
    print(f"  written to {args.out / 'baseline-vs-oracle.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
