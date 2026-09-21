"""P1.6-A3-W4-A3: the stockholders' equity statement -- where do its periods actually live?

Diagnosis only.  **No fix.**  W4-A2 left 474 oracle-PRIMARY cells the legacy stores and the
new producer refuses, across seven tables of which six are statements of stockholders'
equity.  That concentration is the point: if the equity statement's period geometry is not
the one the column binder assumes, then the question is not "how do we bind these columns"
but "what is the period of this cell, according to the source".

Three things have to be settled before any fix, and they are different questions:

  1. GEOMETRY   Are the periods in columns or in rows?  A stockholders' equity statement
                is conventionally row-major -- `Balances as of <date>` opens a section, the
                activity lines follow, the next `Balances as of` closes it -- while the
                columns are the equity components.  If so, a column has no period at all.
  2. ANCHORS    Which rows carry the dates, and does the legacy header selector treat those
                data rows as header rows?  It did on NVDA ord=7899 (`header_idx` returned
                0,1,2,3,4,12,20,28, and 4/12/20/28 are balance rows).
  3. TRUTH      For each lost cell, what period does the legacy axis assign, and what does
                the source say?  This is the one that matters: if the legacy is taking the
                *opening* balance of the section -- the year-ago date -- as the period of
                the activity inside it, then the 474 are not coverage the store would lose,
                they are facts the store is holding under a period off by a year, and the
                producer's refusal is the safer answer rather than a gap.

The third is answered by counting, not by argument: for every lost cell, is the legacy's
period the balance date *above* the cell, the one *below* it, or neither?

  python diagnose_equity_geometry.py --delta <artifact> --out <dir>
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

from src.pdf_retrieval_v4.html_semantic_adapter import _adapt_table  # noqa: E402
from src.pdf_retrieval_v4.metric_path_builder import build_metric_paths  # noqa: E402
from src.pdf_retrieval_v4.semantic_row_classifier import classify_table_rows  # noqa: E402
from src.pdf_retrieval_v4.temporal_axis_graph import build_axis_bindings  # noqa: E402
from src.pdf_retrieval_v4.typed_evidence_emitters import _get_numeric_value  # noqa: E402

PARSER = _BACKEND_DIR / "scripts/evaluation/run_nf_v2_17a4_parse.py"
BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"
CORPUS = Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")

_MONTH_WORD = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
               r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
               r"Dec(?:ember)?)")
FULL_DATE = re.compile(rf"\b{_MONTH_WORD}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+((?:19|20)\d{{2}})",
                       re.I)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    nf = load("nf17a4", PARSER)
    builder = load("store_builder", BUILDER)
    from lxml import etree, html

    report = json.loads(args.delta.read_text(encoding="utf-8"))
    lost = [e for e in report["producer_loss"] if e["legacy_blocker"] == ""]
    prim = [e for e in lost if e["table_role"] == "PRIMARY"]
    by_table = collections.defaultdict(list)
    for entry in prim:
        by_table[(entry["document_id"], entry["source_order"])].append(entry)

    out = {"phase": "P1.6-A3-W4-A3", "mutation": "none", "tables": {}}
    verdict = collections.Counter()

    for (document_id, order), cells in sorted(by_table.items(),
                                              key=lambda kv: -len(kv[1])):
        ticker, accession = builder.DOCUMENTS[document_id]
        parsed = builder.parse_filing(ticker, accession, document_id)
        table = next(t for t in parsed["tables"] if t.get("source_order") == order)
        adapted = _adapt_table(table, parsed)
        tid = adapted["table_fragment_id"]
        rows = classify_table_rows(adapted, tid, adapted["document_id"], 0)
        axes = {a.cell_id: a for a in build_axis_bindings(adapted["cells"], tid)}
        cells_by_id = {c["cell_id"]: c for c in adapted["cells"]}

        root = html.parse(str(CORPUS / ticker / accession / "primary.html"),
                          etree.HTMLParser(recover=True, no_network=True, huge_tree=True,
                                           remove_comments=True)).getroot()
        _b, lookup, _p = nf.make_blocks(
            root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
        block = next(b for b in _b
                     if b["block_type"] == "TABLE" and b["source_order"] == order)
        grid = nf.grid_rows(nf.direct_rows(lookup[block["table_id"]]))

        # 1. geometry: which rows carry a full date, and where is that date's text?
        anchors: list[tuple[int, str, int]] = []
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if not cell:
                    continue
                text = nf.ws(cell["raw_text"])
                if FULL_DATE.search(text):
                    anchors.append((i, text[:60], j))
                    break

        # 2. is the legacy header selector treating those data rows as headers?
        legacy_header_idx = nf.header_idx(grid)
        anchors_in_header = [i for i, _, _ in anchors if i in legacy_header_idx]

        # 3. truth: for each lost cell, which balance date does the legacy period name?
        tally = collections.Counter()
        detail = []
        for entry in cells:
            cell = cells_by_id.get(str(entry["cell_id"]))
            if cell is None:
                continue
            axis = axes.get(str(entry["cell_id"]))
            row_index = int(cell.get("row_index") or 0)
            above = [a for a in anchors if a[0] < row_index]
            below = [a for a in anchors if a[0] > row_index]
            # `normalized_period` is the field the store actually carries, so it decides
            # whether "no period" means the legacy found nothing or merely that
            # `period_start`/`period_end` are unset while a normalized string is present.
            legacy_normalized = getattr(axis, "normalized_period", None) if axis else None
            legacy_period = "|".join(str(x) for x in (
                legacy_normalized, axis.period_start if axis else None,
                axis.period_end if axis else None))
            legacy_years = set(re.findall(r"(?:19|20)\d{2}", legacy_period))

            def years_of(anchor):
                return set(FULL_DATE.search(anchor[1]).groups()) if anchor else set()

            above_years = years_of(above[-1]) if above else set()
            below_years = years_of(below[0]) if below else set()

            if not legacy_years:
                where = "legacy has no period at all"
            elif legacy_years & above_years and legacy_years & below_years:
                where = "ambiguous: matches both anchors"
            elif legacy_years & above_years:
                where = "matches the OPENING balance above"
            elif legacy_years & below_years:
                where = "matches the CLOSING balance below"
            else:
                where = "matches neither anchor"
            tally[where] += 1

            detail.append({
                "cell_id": entry["cell_id"], "row_index": row_index,
                "column_index": entry["column_index"],
                "row_label": cell.get("row_label"),
                "legacy_kind": entry["legacy_kind"],
                "legacy_normalized_period": legacy_normalized,
                "legacy_period_start": axis.period_start if axis else None,
                "legacy_period_end": axis.period_end if axis else None,
                "anchor_above": above[-1][1] if above else None,
                "anchor_below": below[0][1] if below else None,
                "verdict": where,
            })

        key = f"{document_id}#{order}"
        out["tables"][key] = {
            "rows": len(grid), "columns": max(len(r) for r in grid),
            "loss_cells": len(cells),
            "date_bearing_rows": [{"row": i, "text": t, "column": j}
                                  for i, t, j in anchors],
            "legacy_header_idx": legacy_header_idx,
            "date_rows_the_legacy_calls_headers": anchors_in_header,
            "verdicts": dict(tally),
            "cells": detail,
        }
        verdict.update(tally)

        print(f"--- {key}   {len(cells)} lost primary cells")
        print(f"    grid {len(grid)} rows x {max(len(r) for r in grid)} cols")
        print(f"    legacy header_idx {legacy_header_idx}")
        print(f"    date-bearing rows {[(i, t[:34]) for i, t, _ in anchors]}")
        print(f"    ... of which the legacy treats as HEADER rows: {anchors_in_header}")
        for name, count in tally.most_common():
            print(f"      {count:>5}  {name}")
        for entry in detail[:3]:
            print(f"      e.g. r{entry['row_index']} {str(entry['row_label'])[:30]!r} "
                  f"col {entry['column_index']} -> legacy "
                  f"normalized={entry['legacy_normalized_period']!r} "
                  f"{entry['legacy_period_start']}..{entry['legacy_period_end']} "
                  f"| above {str(entry['anchor_above'])[:34]!r}")
        with_normalized = sum(1 for e in detail if e["legacy_normalized_period"])
        print(f"    cells whose legacy axis carries a normalized_period: "
              f"{with_normalized} of {len(detail)}")
        print()

    print("=== every lost primary cell, by which balance its legacy period names ===")
    for name, count in verdict.most_common():
        print(f"  {count:>5}  {name}")
    out["verdicts"] = dict(verdict)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "equity-geometry.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"\n  written to {args.out / 'equity-geometry.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
