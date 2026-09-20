"""P1.6-A3-1b: header-row selection, in shadow.

**Nothing applied.**  A3-1a2 traced two tables whose period never binds because a real
header row was never selected:

    v_fy2025#9951     row 1 holds 'September 30,' with colspan 9 and IS a header row; the
                      selector skipped it, and row 2's bare years were kept without it.
    pfe_fy2024#24395  the 2024 header row sits at index 37 and the second scan reads
                      grid[:32].

The obvious repair is to widen the constant, and that is the one thing not to do: it would
fix Pfizer and fail again on the next filing with an 80-row header block.  What replaces
the bound is a **structural test for what a data row is**, so the scan can cover the whole
grid because a data row cannot qualify -- not because a number was made large enough.

    a bare year or a month-day NAMES A TIME
    a number counts something

`December 31,` and `2024` are periods.  `10,270` is a value.  A row holding two or more
value-like numbers outside its label column is a data row, whatever else it says.

This reports the old and the new header set side by side for the two targets and controls,
and what each would bind, so the change is judged before it is made.

  python shadow_header_row_selection.py --out <dir>
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
    ("v_fy2025", 9951, "BALANCE_SHEET", "target"),
    ("pfe_fy2024", 24395, "EQUITY", "target"),
    ("ko_fy2025", 11388, "BALANCE_SHEET", "target"),
    ("aapl_fy2025", 5498, "BALANCE_SHEET", "control"),
    ("ko_fy2025", 10778, "INCOME_STATEMENT", "control"),
    ("msft_fy2025", 15431, "BALANCE_SHEET", "control"),
    ("tsla_fy2025", 7803, "BALANCE_SHEET", "control"),
)

#: A number that counts something.  A bare year is excluded: it names a time.
_VALUE_LIKE = re.compile(r"^\(?\$?\s*-?[\d,]+(?:\.\d+)?\s*\)?%?$")
_YEAR_ONLY = re.compile(r"^\s*(?:FY\s*)?(?:19|20)\d{2}\s*$", re.I)
_MONTH_DAY = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\.?\s+\d{1,2}(?:st|nd|rd|th)?\b", re.I)
_PERIOD_PHRASE = re.compile(
    r"\b(three|six|nine|twelve|3|6|9|12)[- ]months?\b|\byear(?:s)? ended\b"
    r"|\bas of\b|\bquarter(?:ly)?\b|\byear[- ]to[- ]date\b|\bytd\b", re.I)


def _value_like(text) -> bool:
    t = " ".join(str(text or "").split())
    if not t or _YEAR_ONLY.match(t):
        return False
    return bool(_VALUE_LIKE.match(t))


def header_idx_v2(nf, grid) -> list[int]:
    """Select header rows structurally, over the whole grid.

    The two properties that replaced the constants:

      * a row with two or more value-like numbers outside its label column is a data
        row and cannot be a header, so the scan needs no upper bound at all;
      * a month-day is a period token even without a year beside it, which is what
        `September 30,` is -- the row the old selector dropped for containing a digit.
    """
    out: list[int] = []
    for i, row in enumerate(grid):
        cells = [c for c in row if c]
        if not cells:
            continue
        if sum(1 for c in row[1:] if c and _value_like(c["raw_text"])) >= 2:
            continue  # a data row
        texts = [nf.ws(c["raw_text"]) for c in cells]
        row_text = nf.ws(" ".join(texts))
        if not row_text:
            continue
        if any(c["header"] for c in cells):
            out.append(i)
        elif _PERIOD_PHRASE.search(row_text) or _MONTH_DAY.search(row_text):
            out.append(i)
        elif nf.parse_date_text(row_text):
            out.append(i)
        elif len(set(nf.year_tokens(row_text))) >= 2:
            out.append(i)
        elif i < 4 and not any(nf.has_num(t) for t in texts):
            out.append(i)
    return out or ([0] if grid else [])


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

    report = {"phase": "P1.6-A3-1b", "mutation": "none", "tables": {}}
    print("=== header-row selection (shadow) ===")
    print()

    for document_id, order, family, role in TABLES:
        ticker, accession = builder.DOCUMENTS[document_id]
        raw = (Path("/disk/qh/nano-finrag/data/corpus" if False else
                    "/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")
               / ticker / accession / "primary.html")
        root = html.parse(str(raw), etree.HTMLParser(
            recover=True, no_network=True, huge_tree=True,
            remove_comments=True)).getroot()
        blocks, lookup, _prior = nf.make_blocks(
            root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
        block = next(b for b in blocks
                     if b["block_type"] == "TABLE" and b["source_order"] == order)
        grid = nf.grid_rows(nf.direct_rows(lookup[block["table_id"]]))

        old = nf.header_idx(grid)
        new = header_idx_v2(nf, grid)

        def dated(idx):
            headers = nf.col_headers(grid, idx)
            binds = [nf.period_binding(h, "", "", family) for h in headers]
            return sum(1 for b in binds if b["period_end"]), len(headers)

        old_dated, width = dated(old)
        new_dated, _ = dated(new)

        key = f"{document_id}#{order}"
        report["tables"][key] = {
            "family": family, "role": role, "grid_rows": len(grid),
            "old_header_idx": old, "new_header_idx": new,
            "added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)),
            "old_dated_columns": old_dated, "new_dated_columns": new_dated,
            "columns": width,
        }
        print(f"  [{role}] {key:<20} {family:<16} rows {len(grid):>3}")
        print(f"        old {old}  -> dated {old_dated}/{width}")
        print(f"        new {new}  -> dated {new_dated}/{width}")
        if set(new) - set(old):
            print(f"        added {sorted(set(new) - set(old))}   "
                  f"removed {sorted(set(old) - set(new))}")
        print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "header-selection-shadow.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'header-selection-shadow.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
