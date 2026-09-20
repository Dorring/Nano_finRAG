"""P1.6-A3-1a: reconstruct a period header that the grid split across columns.

**Shadow.**  Nothing is applied.  A3-0 proved the causal chain:

    the date is split across grid columns
      -> no column carries a complete date
      -> period_binding returns period_end None
      -> cell.normalized_period is None
      -> temporal kind is unknown / category / bucket
      -> emit_atomic_facts refuses every cell
      -> 0 atomic facts -> 0 store records

This reconstructs the split header and reports what it *would* bind, so the two
controls can be shown unchanged before anything downstream moves.

**The rule is geometric, not a string search.**  A month-day fragment applies to the
run of bare-year columns that follows it *within the same header row*, and every
column keeps its own year.  That is what makes `December 31, | 2025 | 2024` become
`December 31, 2025` and `December 31, 2024` rather than one year leaking into the
other.  A heuristic that looks left for the nearest month would cross header groups
and multi-year rows would collapse onto one date.

The contract is a **period expression**, not a date: `Years ended` is carried with
the date so the semantics stay `duration` rather than degrading to a point in time.

  python reconstruct_period_headers.py --out <dir>
"""

from __future__ import annotations

import argparse
import collections
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

#: A bare year, and nothing else -- the fragment a split header leaves behind.
_BARE_YEAR = re.compile(r"^\s*(?:FY\s*)?((?:19|20)\d{2})\s*$", re.I)

#: A month and a day, with or without the ordinal suffix: `December 31,`.
_MONTH_DAY = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.I,
)

#: A duration phrase, which must survive the join or the period becomes an instant.
_DURATION = re.compile(
    r"\b(?:three|six|nine|twelve|3|6|9|12)[- ]months?\b"
    r"|\byear(?:s)?\s+ended\b|\byear[- ]to[- ]date\b|\bytd\b|\bquarter(?:ly)?\b",
    re.I,
)


def _fragments(grid, idx) -> list[list[str]]:
    """Per header row, the raw text of each column's cell (empty where blank)."""
    width = max((len(row) for row in grid), default=0)
    out = []
    for i in idx:
        row = grid[i]
        out.append([
            " ".join(str(row[c]["raw_text"]).split()) if c < len(row) and row[c] else ""
            for c in range(width)
        ])
    return out


def reconstruct(grid, idx, current_headers: list[str]) -> list[dict]:
    """What each column's period header would be if the split were repaired.

    Returns one record per column with the current header, the reconstructed one,
    and the fragments the reconstruction consumed, so the join can be checked
    against the grid rather than trusted.
    """
    rows = _fragments(grid, idx)
    width = max((len(r) for r in rows), default=0)
    columns: list[dict] = [
        {"column": c, "current": current_headers[c] if c < len(current_headers) else "",
         "month_day": None, "year": None, "duration": None, "reconstructed": None}
        for c in range(width)
    ]

    for row in rows:
        pending_month_day = None
        for c in range(width):
            cell = row[c] if c < len(row) else ""
            if not cell:
                continue
            md = _MONTH_DAY.search(cell)
            if md:
                # A new month-day opens a new header group; it applies to the run of
                # bare-year columns that follows, and to no column before it.
                pending_month_day = md.group(0).rstrip(",") + ","
                if columns[c]["duration"] is None:
                    duration = _DURATION.search(cell)
                    if duration:
                        columns[c]["duration"] = duration.group(0)
                continue
            year = _BARE_YEAR.match(cell)
            if year:
                columns[c]["year"] = year.group(1)
                if pending_month_day:
                    columns[c]["month_day"] = pending_month_day
                continue
            duration = _DURATION.search(cell)
            if duration and pending_month_day:
                # `Years ended` on the same line as the date, before the years.
                for later in columns[c:]:
                    if later["duration"] is None and later["year"] is None:
                        later["duration"] = duration.group(0)
                continue

    for column in columns:
        parts = [part for part in (column["duration"], column["month_day"],
                                   column["year"]) if part]
        column["reconstructed"] = " ".join(parts) if parts else None
    return columns


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

    oracle = json.loads(Path(
        "/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a2b19a-oracle/"
        "table-authority-oracle.json").read_text(encoding="utf-8"))
    funnel = json.loads(Path(
        "/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a3-0-funnel/"
        "emission-funnel.json").read_text(encoding="utf-8"))

    # The nine zero-record primary statements A3-0 found, plus two controls.
    zero, controls = [], []
    for document_id, tables in funnel["documents"].items():
        for table in tables:
            if table["oracle_role"] != "PRIMARY":
                continue
            (zero if table["store_records"] == 0 else controls).append(table["oracle_key"])
    wanted = zero + controls[:2]

    from lxml import etree, html
    print("=== period header reconstruction (shadow) ===")
    print()
    report = {"phase": "P1.6-A3-1a", "mutation": "none", "tables": {}}
    summary = collections.Counter()

    by_document: dict[str, list[str]] = collections.defaultdict(list)
    for key in wanted:
        document_id, order = key.split("#")
        by_document[document_id].append(int(order))

    for document_id, orders in sorted(by_document.items()):
        ticker, accession = builder.DOCUMENTS[document_id]
        raw = (Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")
               / ticker / accession / "primary.html")
        root = html.parse(str(raw), etree.HTMLParser(
            recover=True, no_network=True, huge_tree=True,
            remove_comments=True)).getroot()
        blocks, lookup, prior = nf.make_blocks(
            root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
        by_order = {b["source_order"]: b for b in blocks if b["block_type"] == "TABLE"}

        for order in sorted(orders):
            block = by_order.get(order)
            if block is None:
                continue
            grid = nf.grid_rows(nf.direct_rows(lookup[block["table_id"]]))
            idx = nf.header_idx(grid)
            headers = nf.col_headers(grid, idx)
            columns = reconstruct(grid, idx, headers)

            improved, unchanged = [], []
            for column in columns:
                if column["reconstructed"] is None:
                    continue
                before = nf.parse_date_text(column["current"])
                after = nf.parse_date_text(column["reconstructed"])
                if before is None and after is not None:
                    improved.append(column)
                elif before is not None:
                    unchanged.append(column)

            key = f"{document_id}#{order}"
            report["tables"][key] = {
                "columns": columns,
                "newly_dated_columns": [c["column"] for c in improved],
                "already_dated_columns": [c["column"] for c in unchanged],
                "sample": [
                    {"column": c["column"], "current": c["current"][:70],
                     "reconstructed": c["reconstructed"],
                     "date_before": nf.parse_date_text(c["current"]),
                     "date_after": nf.parse_date_text(c["reconstructed"])}
                    for c in (improved or columns)[:3]
                ],
            }
            kind = "ZERO" if key in zero else "ctrl"
            summary[(kind, "gained" if improved else "none")] += 1
            print(f"  [{kind}] {key:<20} columns {len(columns):>3}  "
                  f"newly dated {len(improved):>3}  already dated {len(unchanged):>3}")
            for c in (improved or columns)[:2]:
                print(f"        col {c['column']:<3} {c['current'][:46]!r}")
                print(f"              -> {c['reconstructed']!r}")
            if kind == "ctrl" and not improved:
                print("        (control: nothing changed, as required)")
        print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "period-reconstruction-shadow.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print("=== summary ===")
    for (kind, outcome), count in sorted(summary.items()):
        print(f"  {kind:<5} {outcome:<6} {count}")
    print()
    print(f"  written to {args.out / 'period-reconstruction-shadow.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
