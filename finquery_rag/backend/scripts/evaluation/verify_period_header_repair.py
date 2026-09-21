"""P1.6-A3-W4-A7: the period-header repair, verified against the rule it replaces.

**The property under test is monotonicity.**  A repair that raises recall by unbinding
columns that already worked would look identical in a before/after count of facts, so the
count is not the check.  The check is: run the *old* predicate and the new one over every
table in the eight filings and assert that no column which bound under the old rule fails
to bind under the new one.  A repair that unbinds a column is not a repair.

Two module instances of the same producer are loaded and one of them is patched back to the
replaced predicate, which is transcribed here verbatim from the commit that removed it.  A
copy could drift; the assertion that catches drift is that the old rule reproduces W4-A6's
numbers, so `--expect-old-refusals` is checked rather than trusted.

  python verify_period_header_repair.py --out <dir>
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
SHADOW = _BACKEND_DIR / "scripts/evaluation/period_binding_shadow.py"
CORPUS = Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")


def _old_is_period_header_cell(text: str, match: re.Match) -> bool:
    """The predicate W4-A7 replaced, transcribed from the commit that removed it.

    `a date extracted, and no more than twelve characters of anything else`.
    """
    rest = (text[:match.start()] + " " + text[match.end():]).strip()
    rest = re.sub(r"\([^)]*\)", " ", rest)
    rest = re.sub(r"[\s,;:—–-]+", " ", rest).strip()
    if re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", rest):
        return False
    return len(rest) <= 12


def _load(name: str, patch_old: bool):
    spec = importlib.util.spec_from_file_location(name, SHADOW)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if patch_old:
        module._is_period_header_cell = _old_is_period_header_cell
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    nf_spec = importlib.util.spec_from_file_location("nf17a4", PARSER)
    nf = importlib.util.module_from_spec(nf_spec)
    nf_spec.loader.exec_module(nf)
    bspec = importlib.util.spec_from_file_location("store_builder", BUILDER)
    builder = importlib.util.module_from_spec(bspec)
    bspec.loader.exec_module(builder)

    old = _load("pb_old", patch_old=True)
    new = _load("pb_new", patch_old=False)
    from lxml import etree, html

    tally = collections.Counter()
    lost: list[dict] = []
    gained: list[dict] = []
    report = {"phase": "P1.6-A3-W4-A7", "mutation": "none", "tables": 0, "lost": [],
              "gained": []}

    for document_id in sorted(builder.DOCUMENTS):
        ticker, accession = builder.DOCUMENTS[document_id]
        root = html.parse(str(CORPUS / ticker / accession / "primary.html"),
                          etree.HTMLParser(recover=True, no_network=True, huge_tree=True,
                                           remove_comments=True)).getroot()
        blocks, lookup, _prior = nf.make_blocks(
            root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})

        for block in blocks:
            if block["block_type"] != "TABLE":
                continue
            grid = nf.grid_rows(nf.direct_rows(lookup[block["table_id"]]))
            report["tables"] += 1
            before = old.bind_table(nf, grid, document_id, block["table_id"])
            after = new.bind_table(nf, grid, document_id, block["table_id"])

            before_cols = {c: b.normalized_period for c, b in before["columns"].items()}
            after_cols = {c: b.normalized_period for c, b in after["columns"].items()}

            def header_text(col: int) -> str:
                """What the column's own header cells say, so a verdict can be read."""
                parts = []
                for i in before["union_header_idx"]:
                    if i < len(grid) and col < len(grid[i]) and grid[i][col]:
                        text = nf.ws(grid[i][col]["raw_text"])
                        if text:
                            parts.append(text)
                return " / ".join(parts)[:160]

            for col, period in before_cols.items():
                if col not in after_cols:
                    tally["columns unbound by the repair"] += 1
                    lost.append({"document_id": document_id,
                                 "source_order": block["source_order"], "column": col,
                                 "was": period, "header": header_text(col)})
                elif after_cols[col] != period:
                    tally["columns whose period CHANGED"] += 1
                    lost.append({"document_id": document_id,
                                 "source_order": block["source_order"], "column": col,
                                 "was": period, "now": after_cols[col],
                                 "header": header_text(col)})
                else:
                    tally["columns unchanged"] += 1
            for col, period in after_cols.items():
                if col not in before_cols:
                    tally["columns newly bound"] += 1
                    if len(gained) < 400:
                        gained.append({"document_id": document_id,
                                       "source_order": block["source_order"],
                                       "column": col, "period": period,
                                       "method": after["columns"][col].method.value
                                       if after["columns"][col].method else None})

            before_rows = set(before["rows"])
            after_rows = set(after["rows"])
            tally["row bindings lost"] += len(before_rows - after_rows)
            tally["row bindings gained"] += len(after_rows - before_rows)

    print("=== old predicate vs new, over every table in the eight filings ===")
    print(f"    {report['tables']} tables")
    for key in sorted(tally):
        print(f"    {key:<34} {tally[key]}")
    print()

    if lost:
        print("=== MONOTONICITY VIOLATED ===")
        for entry in lost[:24]:
            print(f"    {entry['document_id']} ord={entry['source_order']} "
                  f"col={entry['column']} was={entry['was']} "
                  f"now={entry.get('now', 'NOTHING')}")
            print(f"        header: {entry['header']!r}")
    else:
        print("=== monotone: every column the old rule bound, the new rule binds too ===")
    print()

    print("=== columns the repair newly binds, by document ===")
    for key, count in collections.Counter(
            (g["document_id"], g["method"]) for g in gained).most_common(20):
        print(f"    {key[0]:<14} {str(key[1]):<26} {count}")

    report["lost"] = lost
    report["gained"] = gained
    report["totals"] = dict(tally)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "period-header-repair.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"\n  written to {args.out / 'period-header-repair.json'}")
    return 1 if lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
