"""P1.6-A3-1d: Coca-Cola's equity statement -- a source that gives a year and no date.

Diagnosis only.  **No binding change, no parser change.**  The statement's header is

    2025  colspan=3        2024  colspan=3

and the first thing not to do is turn that into `2025-12-31`.  What has to be settled
before a `YEAR_ONLY_PERIOD` method exists:

  1. SCOPE      the year cell spans three logical columns -- is that three columns
                sharing one binding, or one logical column replicated?  Read off the
                values under each covered column, not assumed from the colspan.
  2. SEMANTICS  what the year *means* here.  `duration` is only claimable if the
                filing's own words say so; a bare year with no `years ended` beside it
                is a year whose temporal kind the source does not state, and saying
                `duration` anyway would be a guess wearing a contract's clothes.
  3. MATCHING   `YEAR(2025)` against a slot's `FY2025`, and against `2025-12-31`.
  4. INHERITANCE which cells the year reaches, per cell, without crossing into the
                2024 group.

  python diagnose_year_only_period.py --out <dir>
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
    ("ko_fy2025", 12533, "EQUITY", "the case"),
    ("ko_fy2025", 10778, "INCOME_STATEMENT", "sibling: a full date beside it"),
)

_BARE_YEAR = re.compile(r"^\s*(?:FY\s*)?((?:19|20)\d{2})\s*$", re.I)
_VALUE_LIKE = re.compile(r"^\(?\$?\s*-?[\d,]+(?:\.\d+)?\s*\)?%?$")

#: What the source must say for the year to be a duration rather than an unresolved year.
_DURATION_PHRASE = re.compile(
    r"\byear(?:s)?\s+ended\b|\bfor\s+the\s+years?\b|\bfiscal\s+year\b|\bannual\b", re.I)


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

    report = {"phase": "P1.6-A3-1d", "mutation": "none", "tables": {}}
    print("=== a source that gives a year and no date ===")
    print()

    for document_id, order, family, role in TABLES:
        ticker, accession = builder.DOCUMENTS[document_id]
        raw = (Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")
               / ticker / accession / "primary.html")
        root = html.parse(str(raw), etree.HTMLParser(
            recover=True, no_network=True, huge_tree=True,
            remove_comments=True)).getroot()
        blocks, lookup, prior = nf.make_blocks(
            root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
        block = next(b for b in blocks
                     if b["block_type"] == "TABLE" and b["source_order"] == order)
        table = next(t for t in builder.parse_filing(ticker, accession, document_id)["tables"]
                     if t["table_id"] == block["table_id"])
        grid = nf.grid_rows(nf.direct_rows(lookup[block["table_id"]]))

        # --- 1. the year cells and the logical columns each covers -----------------
        groups = []
        for ri, row in enumerate(grid):
            seen: set[int] = set()
            for ci, cell in enumerate(row):
                if not cell or id(cell) in seen:
                    continue
                seen.add(id(cell))
                text = nf.ws(cell["raw_text"])
                match = _BARE_YEAR.match(text)
                if not match:
                    continue
                covers = [c for c in range(len(row))
                          if row[c] is not None and id(row[c]) == id(cell)]
                # The values under each covered column, to settle replica vs distinct.
                sequences = {}
                for column in covers:
                    values = []
                    for lower in grid[ri + 1:]:
                        if column < len(lower) and lower[column] and _VALUE_LIKE.match(
                                nf.ws(lower[column]["raw_text"])):
                            values.append(nf.ws(lower[column]["raw_text"]))
                        if len(values) >= 5:
                            break
                    sequences[column] = values
                groups.append({
                    "row": ri, "column": ci, "year": match.group(1),
                    "text": text, "colspan": cell.get("colspan"),
                    "rowspan": cell.get("rowspan"),
                    "covers_logical_columns": covers,
                    "values_under_each_column": sequences,
                    "replica": len({tuple(v) for v in sequences.values()}) == 1,
                })

        # --- 2. what the source says the year means --------------------------------
        labels = [nf.ws(c["raw_text"]) for row in grid for c in row
                  if c and c.get("column_index") == 0 if nf.ws(c["raw_text"])]
        evidence_text = nf.ws(" ".join(
            [table.get("table_title") or ""]
            + [str(t) for t in (table.get("column_headers") or [])]
            + [nf.ws(c["raw_text"]) for row in grid[:8] for c in row if c]
        ))
        duration_here = _DURATION_PHRASE.search(evidence_text)

        # The statement heading the parse associated with the block, which is where a
        # filer states `for the years ended`.
        heading = nf.ws(" ".join([str(block.get("text") or "")[:200]]))

        key = f"{document_id}#{order}"
        report["tables"][key] = {
            "family": family, "role": role,
            "groups": groups,
            "duration_phrase_in_table_evidence": duration_here.group(0) if duration_here else None,
            "evidence_text": evidence_text[:300],
            "row_labels_sample": [x for x in labels if x][:8],
        }

        print(f"--- {key}  {family}  ({role})   {len(grid)} rows")
        for group in groups:
            print(f"    year {group['year']}  at row {group['row']} col {group['column']}  "
                  f"colspan={group['colspan']} rowspan={group['rowspan']}")
            print(f"        covers logical columns {group['covers_logical_columns']}")
            print(f"        identical values under each covered column: {group['replica']}")
            for column, values in list(group["values_under_each_column"].items())[:3]:
                print(f"          col {column}: {values}")
        print(f"    duration phrase in the table's own evidence: "
              f"{duration_here.group(0) if duration_here else 'NONE'}")
        print(f"    evidence text: {evidence_text[:180]!r}")
        print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "year-only-period.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'year-only-period.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
