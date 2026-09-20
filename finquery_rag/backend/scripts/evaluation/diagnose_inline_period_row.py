"""P1.6-A3-1c: Pfizer's equity statement -- a period that lives on the data row.

Diagnosis only.  **No binding change, no parser change.**  A3-1b2 removed this table
from the header-selection repair: its period does not come from a header block above it.
The rows read

    Balance, December 31, 2022 | value | value | ...

so the period expression is *on the row*, and the questions are the three that decide
whether that is a contract or a special case:

  1. SCOPE      does the row's period bind that row alone, or the rows that follow it?
                A statement of changes in equity has opening balances, movement rows and
                closing balances, so "a date appeared, inherit downward" would silently
                stamp one date across a section.
  2. DIRECTION  does it reach the row's own numeric cells -- and never the whole column?
  3. IDENTITY   can a bound fact name the method, the source cell the expression came
                from, and the numeric cell it landed on?

This answers them from the grid.  It does not decide the contract.

  python diagnose_inline_period_row.py --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"
BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"

TABLES = (
    ("pfe_fy2024", 24395, "EQUITY", "the case"),
    ("pfe_fy2024", 23679, "BALANCE_SHEET", "sibling control: binds today"),
)

_MONTH_DAY_YEAR = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\.?\s+\d{1,2},?\s+((?:19|20)\d{2})\b", re.I)
_VALUE_LIKE = re.compile(r"^\(?\$?\s*-?[\d,]+(?:\.\d+)?\s*\)?%?$")
_YEAR_ONLY = re.compile(r"^\s*(?:FY\s*)?(?:19|20)\d{2}\s*$", re.I)


def _value_like(text) -> bool:
    t = " ".join(str(text or "").split())
    if not t or _YEAR_ONLY.match(t):
        return False
    return bool(_VALUE_LIKE.match(t))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    spec = importlib.util.spec_from_file_location("nf17a4", PARSER)
    nf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nf)
    bspec = importlib.util.spec_from_file_location("store_builder", BUILDER)
    builder = importlib.util.module_from_spec(bspec)
    bspec.loader.exec_module(builder)

    from lxml import etree, html

    report = {"phase": "P1.6-A3-1c", "mutation": "none", "tables": {}}
    print("=== inline period on a data row ===")
    print()

    for document_id, order, family, role in TABLES:
        ticker, accession = builder.DOCUMENTS[document_id]
        raw = (Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")
               / ticker / accession / "primary.html")
        root = html.parse(str(raw), etree.HTMLParser(
            recover=True, no_network=True, huge_tree=True,
            remove_comments=True)).getroot()
        blocks, lookup, _prior = nf.make_blocks(
            root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
        block = next(b for b in blocks
                     if b["block_type"] == "TABLE" and b["source_order"] == order)
        grid = nf.grid_rows(nf.direct_rows(lookup[block["table_id"]]))

        rows = []
        for ri, row in enumerate(grid):
            seen: set[int] = set()
            period_cells, numeric_cells, labels = [], [], []
            for ci, cell in enumerate(row):
                if not cell or id(cell) in seen:
                    continue
                seen.add(id(cell))
                text = nf.ws(cell["raw_text"])
                if not text:
                    continue
                match = _MONTH_DAY_YEAR.search(text)
                if match:
                    period_cells.append({"column": ci, "text": text[:60],
                                         "expression": match.group(0)})
                elif _value_like(text):
                    numeric_cells.append({"column": ci, "raw": text[:24]})
                elif ci == 0:
                    labels.append({"column": ci, "text": text[:70]})
            if not (period_cells or numeric_cells or labels):
                continue
            if period_cells and numeric_cells:
                kind = "PERIOD_AND_VALUES"
            elif period_cells:
                kind = "PERIOD_ONLY"
            elif numeric_cells and labels:
                kind = "VALUES_ONLY"
            else:
                kind = "LABEL_ONLY"
            rows.append({
                "row": ri, "kind": kind,
                "period_cells": period_cells, "numeric_cells": numeric_cells[:3],
                "numeric_count": len(numeric_cells), "label": labels[0]["text"] if labels else None,
            })

        key = f"{document_id}#{order}"
        report["tables"][key] = {"family": family, "role": role,
                                 "grid_rows": len(grid), "rows": rows}

        kinds: dict[str, int] = {}
        for entry in rows:
            kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
        print(f"--- {key}  {family}  ({role})  {len(grid)} rows   {kinds}")
        for entry in rows:
            if entry["kind"] == "PERIOD_AND_VALUES":
                print(f"    row {entry['row']:<3} PERIOD_AND_VALUES  "
                      f"period={entry['period_cells'][0]['expression']!r} "
                      f"at col {entry['period_cells'][0]['column']}   "
                      f"numeric cells {entry['numeric_count']}")
                print(f"          label={entry['label']!r}")
                print(f"          values={[c['raw'] for c in entry['numeric_cells']]}")
            elif entry["kind"] == "PERIOD_ONLY" and entry["period_cells"]:
                print(f"    row {entry['row']:<3} PERIOD_ONLY         "
                      f"period={entry['period_cells'][0]['expression']!r}")
        print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "inline-period-row.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'inline-period-row.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
