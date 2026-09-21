"""P1.6-A3-1a2: why the other five tables' period never binds.

Diagnosis only.  **No period binding change, no reconstruction rule change.**  A3-1a
reconstructed four of the nine zero-record primary statements by joining a month-day
fragment to the bare-year columns beside it, and left five untouched.  The question
here is not how to make those five succeed -- it is **which grid geometry breaks the
period provenance**, so that A3-1b can be one attributable repair or several rather
than one heuristic grown until it fits.

For each table this reports:

  1. the grid, and which rows `header_idx` selected as header;
  2. every row holding a year token or a month-day, whether it was selected, and the
     span of the cell it sits in;
  3. which logical columns each of those cells covers -- `grid_rows` shares one record
     across every position a colspan reaches, so this is read off, not inferred;
  4. what `col_headers` produced per column and what `period_binding` bound;
  5. a verdict, from the evidence above and nothing else.

The three shapes A3-1a did not settle, held as hypotheses and not pre-selected:

    A  HEADER_ROW_NOT_SELECTED   the year is in the table, in a row header_idx skipped
    B  SPAN_NOT_PROPAGATED       the year is in a spanned cell that did not reach the
                                 logical columns its data sits under
    C  FUSED_INTO_LONGER_TEXT    the year is inside a longer cell with no standalone
                                 year token to find

  python diagnose_header_geometry.py --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"
BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"

#: The five A3-1a could not reconstruct, with what its shadow output said about each.
REMAINING = (
    ("ko_fy2025", 12533, "EQUITY"),
    ("nvda_fy2025", 8766, "CASH_FLOW"),
    ("pfe_fy2024", 24395, "EQUITY"),
    ("tsla_fy2025", 10407, "CASH_FLOW"),
    ("v_fy2025", 9951, "BALANCE_SHEET"),
)

_BARE_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_MONTH_DAY = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\.?\s+\d{1,2}(?:st|nd|rd|th)?\b",
    re.I,
)


def diagnose(grid, idx, headers, binds) -> dict:
    """The grid's period evidence, and what became of it."""
    selected = set(idx)

    # Every grid position, grouped by the physical cell it came from: `grid_rows`
    # stores one record per cell and repeats it across the columns a colspan covers,
    # so collecting the positions that share a record recovers the span exactly.
    span_columns: dict[int, list[int]] = defaultdict(list)
    span_of: dict[int, dict] = {}
    for ri, row in enumerate(grid):
        for ci, cell in enumerate(row):
            if not cell:
                continue
            key = id(cell)
            span_columns[key].append(ci)
            span_of[key] = cell

    year_rows = []
    month_rows = []
    for ri, row in enumerate(grid):
        years, months = set(), set()
        for cell in row:
            if not cell:
                continue
            text = " ".join(str(cell.get("raw_text") or "").split())
            years.update(_BARE_YEAR.findall(text))
            if _MONTH_DAY.search(text):
                months.add(_MONTH_DAY.search(text).group(0))
        if years:
            year_rows.append((ri, sorted(years), ri in selected))
        if months:
            month_rows.append((ri, sorted(months), ri in selected))

    # Which cell each year actually sits in, and how far its span reaches.
    carriers = []
    for ri, row in enumerate(grid):
        for ci, cell in enumerate(row):
            if not cell:
                continue
            text = " ".join(str(cell.get("raw_text") or "").split())
            years = _BARE_YEAR.findall(text)
            month = _MONTH_DAY.search(text)
            if not years and not month:
                continue
            covers = sorted(span_columns[id(cell)])
            if any(entry["row"] == ri and entry["col"] == ci for entry in carriers):
                continue
            carriers.append({
                "row": ri,
                "col": ci,
                "text": text[:70],
                "years": years,
                "month_day": month.group(0) if month else None,
                "rowspan": cell.get("rowspan"),
                "colspan": cell.get("colspan"),
                "is_th": bool(cell.get("header")),
                "row_selected": ri in selected,
                "covers_columns": covers,
                "standalone_year": bool(re.fullmatch(r"\s*(?:FY\s*)?(?:19|20)\d{2}\s*",
                                                      text, re.I)),
            })

    bound = [
        {"column": c,
         "header": (headers[c] if c < len(headers) else "")[:80],
         "period_end": (binds[c] or {}).get("period_end") if c < len(binds) else None,
         "semantics": (binds[c] or {}).get("period_semantics") if c < len(binds) else None}
        for c in range(len(headers))
    ]
    dated = [b for b in bound if b["period_end"]]

    # The verdict, read off the evidence rather than chosen.
    data_columns = [b["column"] for b in bound
                    if b["column"] > 0 and not b["period_end"]]
    reachable = {entry["col"]: entry for entry in carriers}
    if not carriers:
        verdict = "NO_YEAR_TOKEN_IN_TABLE"
    elif not dated and any(not entry["row_selected"] and entry["standalone_year"]
                           for entry in carriers):
        verdict = "HEADER_ROW_NOT_SELECTED"
    elif not dated and any(entry["row_selected"] and entry["standalone_year"]
                           and (int(entry["colspan"] or 1) > 1
                                or int(entry["rowspan"] or 1) > 1)
                           for entry in carriers):
        verdict = "SPAN_NOT_PROPAGATED"
    elif not dated and not any(entry["standalone_year"] for entry in carriers):
        verdict = "FUSED_INTO_LONGER_TEXT"
    elif not dated:
        verdict = "UNEXPLAINED"
    else:
        verdict = "BINDS_SOMEWHERE"

    return {
        "grid_rows": len(grid),
        "grid_columns": max((len(r) for r in grid), default=0),
        "header_idx": idx,
        "year_rows": [{"row": r, "years": y, "selected": s} for r, y, s in year_rows],
        "month_day_rows": [{"row": r, "months": m, "selected": s}
                           for r, m, s in month_rows],
        "carriers": carriers,
        "bound_columns": bound,
        "dated_columns": len(dated),
        "undated_data_columns": data_columns,
        "verdict": verdict,
    }


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

    report = {"phase": "P1.6-A3-1a2", "mutation": "none", "tables": {}}
    print("=== header geometry diagnosis ===")
    print()

    for document_id, order, family in REMAINING:
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
        idx = nf.header_idx(grid)
        headers = nf.col_headers(grid, idx)
        binds = [nf.period_binding(h, "", "", family) for h in headers]
        result = diagnose(grid, idx, headers, binds)

        key = f"{document_id}#{order}"
        report["tables"][key] = {"family": family, **result}

        print(f"--- {key}  {family}")
        print(f"    grid {result['grid_rows']} rows x {result['grid_columns']} cols   "
              f"header_idx {idx}   dated columns {result['dated_columns']}")
        print(f"    YEAR rows:   "
              f"{[(y['row'], y['years'], 'sel' if y['selected'] else 'NOT-SEL') for y in result['year_rows']]}")
        print(f"    MONTH rows:  "
              f"{[(m['row'], m['months'][:1], 'sel' if m['selected'] else 'NOT-SEL') for m in result['month_day_rows']]}")
        for entry in result["carriers"][:5]:
            print(f"      row {entry['row']:<3} col {entry['col']:<3} "
                  f"rs={entry['rowspan']} cs={entry['colspan']} "
                  f"th={entry['is_th']} {'sel' if entry['row_selected'] else 'NOT-SEL'} "
                  f"covers={entry['covers_columns']}")
            print(f"          text={entry['text']!r}")
        print(f"    sample undated data column: "
              f"{result['bound_columns'][result['undated_data_columns'][0]] if result['undated_data_columns'] else None}")
        print(f"    VERDICT {result['verdict']}")
        print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "header-geometry.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'header-geometry.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
