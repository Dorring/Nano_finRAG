"""P1.6-A3-W3-0: one statement, two granularities -- where the group propagation breaks.

Diagnosis only.  **No TemporalKind change.**

W3's producer reports Visa's balance sheet with DAY on some columns and YEAR on others.
If those columns share one source-declared period (`September 30,` over the year cells),
they must not differ merely because the grid reaches them by different geometry.  That is
a `PeriodBindingV2` group-propagation question, and answering it with a TemporalKind rule
-- "a YEAR column beside a point column is probably a point" -- would push inference back
into the layer that is supposed to report what the source said.

Per logical column this reports the raw header cells with their spans and the columns they
cover, the binding, and the kind, so the divergence can be read rather than inferred.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"
BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"
PB_SHADOW = _BACKEND_DIR / "scripts/evaluation/period_binding_shadow.py"
TK_SHADOW = _BACKEND_DIR / "scripts/evaluation/temporal_kind_shadow.py"

TABLE = ("v_fy2025", "V", "SEC_1403161_000140316125000089", 9951, "BALANCE_SHEET")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    nf = load("nf17a4", PARSER)
    builder = load("store_builder", BUILDER)
    pb = load("pb_shadow", PB_SHADOW)
    tk = load("tk_shadow", TK_SHADOW)

    from src.pdf_retrieval_v4 import temporal_axis_graph as tag
    from lxml import etree, html

    document_id, ticker, accession, order, family = TABLE
    root = html.parse(
        str(Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")
            / ticker / accession / "primary.html"),
        etree.HTMLParser(recover=True, no_network=True, huge_tree=True,
                         remove_comments=True)).getroot()
    blocks, lookup, _prior = nf.make_blocks(
        root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
    block = next(b for b in blocks
                 if b["block_type"] == "TABLE" and b["source_order"] == order)
    grid = nf.grid_rows(nf.direct_rows(lookup[block["table_id"]]))
    union, added = pb.extended_header_idx(nf, grid)
    bound = pb.bind_table(nf, grid, document_id, block["table_id"])

    width = max((len(r) for r in grid), default=0)

    # Which logical columns each header-row cell covers, read off the shared record.
    spans = []
    for i in union:
        seen: set[int] = set()
        for col in range(width):
            cell = grid[i][col] if i < len(grid) and col < len(grid[i]) else None
            if not cell or id(cell) in seen:
                continue
            seen.add(id(cell))
            covers = [c for c in range(len(grid[i]))
                      if grid[i][c] is not None and id(grid[i][c]) == id(cell)]
            spans.append({"header_row": i, "at_col": col,
                          "text": nf.ws(cell["raw_text"])[:50],
                          "colspan": cell.get("colspan"),
                          "rowspan": cell.get("rowspan"),
                          "in_added_row": i in added,
                          "covers": covers})

    columns = []
    for col in range(width):
        cells = bound["columns"].get(col)
        evidence, _ = tk.column_evidence(nf, grid, union)[col]
        kind = tk.classify(evidence, tag, cells)
        columns.append({
            "logical_col": col,
            "raw_header_texts": [t for s in spans
                                 if s["at_col"] <= col < s["at_col"] + len(s["covers"])
                                 and col in s["covers"] for t in [s["text"]]],
            "binding_status": cells.status.value if cells else None,
            "normalized_period": cells.normalized_period if cells else None,
            "granularity": cells.granularity.value if cells else None,
            "method": cells.method.value if cells else None,
            "source_cells": [(c.row, c.column) for c in cells.source_cells] if cells else [],
            "temporal_kind": kind.kind.value,
        })

    report = {"phase": "P1.6-A3-W3-0", "mutation": "none",
              "union_header_idx": union, "added_header_rows": sorted(added),
              "spans": spans, "columns": columns}

    print("=== Visa balance sheet: period group consistency ===")
    print()
    print(f"union header rows {union}   added {sorted(added)}")
    print()
    print("header cells and the logical columns each covers:")
    for span in spans:
        mark = "ADDED " if span["in_added_row"] else "      "
        print(f"  {mark}row {span['header_row']:<3} col {span['at_col']:<3} "
              f"cs={span['colspan']:<3} covers={span['covers']}")
        print(f"          {span['text']!r}")
    print()
    print("per logical column:")
    for entry in columns:
        print(f"  col {entry['logical_col']:<3} {str(entry['granularity']):<8} "
              f"{str(entry['method']):<28} {str(entry['normalized_period']):<12} "
              f"kind={entry['temporal_kind']:<8} src={entry['source_cells']}")
    print()

    dated = [c for c in columns if c["granularity"] == "DAY"]
    yearonly = [c for c in columns if c["granularity"] == "YEAR"]
    print(f"columns DAY {len(dated)}: {[c['logical_col'] for c in dated]}")
    print(f"columns YEAR {len(yearonly)}: {[c['logical_col'] for c in yearonly]}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "visa-period-group.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"\n  written to {args.out / 'visa-period-group.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
