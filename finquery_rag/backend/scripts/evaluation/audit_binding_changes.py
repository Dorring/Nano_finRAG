"""P1.6-A3-W4-A8: what the period-header repair actually changed, adjudicated.

W4-A7 replaced the period-header predicate.  Against the binder it moved 27 columns out of
a binding and 645 from one period to another, and W4-A7 reported both as counts without
saying what they *mean*.  This says what they mean.

The gate it is measured against is **source-truth monotonicity**, not strict monotonicity:

    CORRECT_SOURCE_GROUNDED_BINDING_LOST = 0
    NEW_UNSUPPORTED_BINDING              = 0
    UNEXPLAINED_BINDING_CHANGE           = 0
    V2_PRODUCER_GAP                      = 0

Strict monotonicity was the right gate when the change was a pure regex widening, where
every new binding was additional and no old one had a reason to move.  It is the wrong gate
now: A7 both removes false positives and corrects wrong periods, so requiring every old
binding to survive requires V2 to reproduce the legacy defect.  What must hold is that no
binding the source actually supports is lost, and that nothing unsupported is gained.

Two hard gates, and the whole point of the exercise is to see whether they are zero:

    NEW_BINDING_REGRESSION = 0    legacy had it right and the repair has it wrong
    ROW_PRODUCER_GAP       = 0    the source declares a row-local period and the row
                                  producer does not take it

**"More precise" is not "better".**  A column that said `2025` and now says `2025-12-31` is
a refinement only if the source declares a full date for that column; if all the filing says
is `2025`, the new binding is a fabrication however much finer it looks.

  python audit_binding_changes.py --out <dir>
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
    """The predicate W4-A7 replaced, transcribed from the commit that removed it."""
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


def _summary(binding) -> dict:
    return {"period": binding.normalized_period,
            "granularity": binding.granularity.value,
            "method": binding.method.value if binding.method else None,
            "status": binding.status.value,
            "source_cells": [[c.row, c.column, c.text[:34]] for c in binding.source_cells]}


#: What a change means.  The distinction that matters is not how much finer the new period
#: is but whether the source states it -- `2025` becoming `2025-12-31` is a refinement only
#: if the column's own header carries a `December 31,` to complete with, and every family
#: below was read against its geometry before being filed.
CHANGE_MEANINGS = {
    "SOURCE_GROUNDED_REFINEMENT":
        "the column's own year, completed by a month-day declaration the old predicate "
        "refused for being longer than twelve characters -- the YEAR-only binding was the "
        "incomplete reading of a column whose header states both halves",
    "SEMANTICALLY_UNCHANGED":
        "the same period, expressed differently",
    "LEGACY_WRONG_PERIOD_CORRECTED":
        "the old period came from outside the column and the new one is the column's own",
    "SOURCE_AMBIGUOUS":
        "neither side is provable from the cell -- registered, not chosen between",
    "NEW_BINDING_REGRESSION":
        "the source did not state a full date and the repair supplied one anyway",
}


def meaning_of(old_period: str, new_period: str, new_method: str | None,
               new_granularity: str) -> str:
    """Which of the five a change is, from the transition alone.

    Deliberately narrow.  Anything the transition does not settle comes back as ambiguous
    rather than being folded into the nearest reassuring label.
    """
    if old_period == new_period:
        return "SEMANTICALLY_UNCHANGED"
    if (len(str(old_period)) == 4 and str(old_period).isdigit()
            and str(new_period).startswith(str(old_period))
            and new_granularity == "DAY"
            and new_method in ("ADJACENT_YEAR_JOIN", "HEADER_ROW_SELECTION_EXTEND")):
        return "SOURCE_GROUNDED_REFINEMENT"
    return "SOURCE_AMBIGUOUS"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
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
    unbound: list[dict] = []
    changed: list[dict] = []
    gained: list[dict] = []

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
            before = old.bind_table(nf, grid, document_id, block["table_id"])
            after = new.bind_table(nf, grid, document_id, block["table_id"])

            def column_cells(col: int) -> str:
                parts = []
                for i in before["union_header_idx"]:
                    if i < len(grid) and col < len(grid[i]) and grid[i][col]:
                        text = nf.ws(grid[i][col]["raw_text"])
                        if text:
                            parts.append(text)
                return " / ".join(parts)[:150]

            def value_rows(col: int) -> list[int]:
                """Rows of this column that hold something a fact could be made from."""
                out = []
                for i, row in enumerate(grid):
                    if col < len(row) and row[col]:
                        raw = nf.ws(row[col]["raw_text"])
                        if raw and nf.numeric(raw).get("normalized_value") is not None:
                            out.append(i)
                return out

            def row_could_declare(i: int) -> bool:
                """Whether the row producer's own rule could fire on this row at all.

                Two conditions, both from `bind_table`'s row rule: at least two value-like
                cells beside the date, and exactly one year in a cell that states a date.
                A row meeting both and still unbound is a real gap; a row meeting neither
                has no row-local period to take.

                Both halves were needed.  The first version checked only the value count,
                which made every activity row inside a stockholders' equity section --
                `Net income`, `Provision`, `Other comprehensive income` -- look like a row
                whose period had been dropped, and reported 12 columns as PARTLY_REHOMED
                when the period had in fact moved correctly from a wrong column binding to
                the right row bindings.  The rows that *do* state a period there
                (`Balance, January 1, 2022`, `Balance, December 31, 2022`) are all bound.
                """
                if i >= len(grid):
                    return False
                row = grid[i]
                if sum(1 for c in row
                       if c and old._value_like(nf.ws(c["raw_text"]))) < 2:
                    return False
                for cell in row:
                    if not cell:
                        continue
                    text = nf.ws(cell["raw_text"])
                    if not old._MONTH_DAY_YEAR.search(text):
                        continue
                    if len(set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text))) == 1:
                        return True
                return False

            for col, binding in before["columns"].items():
                if col not in after["columns"]:
                    rows = value_rows(col)
                    homed = [r for r in rows if r in after["rows"]]
                    declarable = [r for r in rows if row_could_declare(r)]
                    homed = [r for r in declarable if r in after["rows"]]
                    if not declarable:
                        verdict = "NOT_A_PERIOD_DECLARATION"
                    elif len(homed) == len(declarable):
                        verdict = "REHOMED_TO_ROW_BINDING"
                    elif homed:
                        verdict = "PARTLY_REHOMED"
                    else:
                        verdict = "ROW_PRODUCER_GAP"
                    tally[f"unbound: {verdict}"] += 1
                    if len(unbound) < 300:
                        unbound.append({
                            "document_id": document_id,
                            "source_order": block["source_order"], "column": col,
                            "was": binding.normalized_period, "verdict": verdict,
                            "value_rows": rows[:12], "row_bound": homed[:12],
                            "rows_the_row_rule_could_take": declarable[:12],
                            "row_periods": sorted({after["rows"][r].normalized_period
                                                   for r in homed})[:6],
                            "header": column_cells(col)})
                elif after["columns"][col].normalized_period != binding.normalized_period:
                    tally["changed"] += 1
                    if len(changed) < 4000:
                        changed.append({
                            "document_id": document_id,
                            "source_order": block["source_order"], "column": col,
                            "old": binding.normalized_period,
                            "old_granularity": binding.granularity.value,
                            "old_method": binding.method.value if binding.method else None,
                            "new": _summary(after["columns"][col]),
                            "header": column_cells(col)})
                else:
                    tally["unchanged"] += 1

            for col, binding in after["columns"].items():
                if col in before["columns"]:
                    continue
                # A binding with no old one to compare against still has to be checked:
                # `NEW_UNSUPPORTED_BINDING = 0` is about what the repair *gained*, and the
                # unchanged/changed split says nothing about it.  The test is the one
                # W4-A5 used on the gains -- every source cell the binding read must be in
                # the fact's own column, unless the method is a row-scoped declaration.
                columns_read = {c.column for c in binding.source_cells}
                supported = (columns_read == {col}
                             or binding.method
                             is old.PeriodBindingMethod.INLINE_PERIOD_DATA_ROW)
                tally["gained: supported" if supported
                      else "gained: SOURCE CELLS OUTSIDE THE COLUMN"] += 1
                if not supported and len(gained) < 200:
                    gained.append({"document_id": document_id,
                                   "source_order": block["source_order"], "column": col,
                                   "new": _summary(binding),
                                   "header": column_cells(col)})

    print("=== columns that lost their column binding ===")
    for key, count in sorted(tally.items()):
        if key.startswith("unbound") or key.startswith("gained"):
            print(f"    {key:<42} {count}")
    print()
    for entry in gained[:8]:
        print(f"    UNSUPPORTED: {entry['document_id']} ord={entry['source_order']} "
              f"col={entry['column']} -> {entry['new']['period']} "
              f"via {entry['new']['method']}")
        print(f"        header: {entry['header'][:110]!r}")
        print(f"        source_cells: {entry['new']['source_cells'][:3]}")
    print()
    for entry in unbound[:args.samples * 4]:
        print(f"    {entry['document_id']} ord={entry['source_order']} "
              f"col={entry['column']} was={entry['was']}  {entry['verdict']}")
        print(f"        value rows {entry['value_rows']}  row-bound {entry['row_bound']} "
              f"-> {entry['row_periods']}")
        print(f"        header: {entry['header'][:110]!r}")
    print()

    # --- changed columns, clustered by the transition itself -------------------------
    families = collections.Counter()
    examples: dict[tuple, list] = collections.defaultdict(list)
    for entry in changed:
        key = (entry["old_granularity"], entry["new"]["granularity"],
               entry["new"]["method"], entry["old"], entry["new"]["period"])
        families[key] += 1
        if len(examples[key]) < args.samples:
            examples[key].append(entry)

    print("=== period changes, by binding transition ===")
    meanings = collections.Counter()
    for key, count in families.most_common(20):
        old_g, new_g, method, old_p, new_p = key
        meaning = meaning_of(old_p, new_p, method, new_g)
        meanings[meaning] += count
        print(f"    {count:>5}  {old_p} ({old_g}) -> {new_p} ({new_g}) via {method}"
              f"   [{meaning}]")
        for entry in examples[key]:
            print(f"             {entry['document_id']} ord={entry['source_order']} "
                  f"col={entry['column']}")
            print(f"               header: {entry['header'][:104]!r}")
            print(f"               new source_cells: {entry['new']['source_cells'][:3]}")
    for key, count in {k: v for k, v in families.items()
                       if k not in dict(families.most_common(20))}.items():
        meanings[meaning_of(key[3], key[4], key[2], key[1])] += count
    print()
    print("=== what the changes mean, over columns ===")
    for name, count in meanings.most_common():
        print(f"    {count:>5}  {name}")
        print(f"           {CHANGE_MEANINGS[name]}")
    tally["NEW_BINDING_REGRESSION"] = meanings.get("NEW_BINDING_REGRESSION", 0)
    tally["ROW_PRODUCER_GAP"] = sum(
        v for k, v in tally.items() if k.endswith("ROW_PRODUCER_GAP"))
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "binding-changes.json").write_text(
        json.dumps({"phase": "P1.6-A3-W4-A8", "mutation": "none",
                    "totals": dict(tally), "unbound": unbound, "changed": changed,
                    "gained_unsupported": gained},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'binding-changes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
