"""P1.6-A3-W4-A6: audit the 1,649 losses that look column-local.

Diagnosis only.  **Narrow on purpose: this audits 1,649 cells, not 8,369.**

W4-A5 measured that 79.5% of the deficit carries a period the cell's own column does not
name, and the correction that follows from it is a boundary this file exists to respect:

    the column does not name this period   !=   this period is wrong

A legitimate period can come from a row, a row-group, or the table's own context, none of
which the column-scoped producer can see.  So the 6,405 are neither confirmed nor refuted
and are not touched here.  What is audited is the 1,649 that satisfy

    legacy period  ~=  the period the cell's own column explicitly names

because those are the only ones that can be a genuine V2 capability regression, and they are
the number W4-B has to be decided on.  "At most 6% of the store" is a budget; this is meant
to turn it into a fact.

Per **column** (one column can hold several of the 1,649 cells) it asks the producer why it
has no binding, using the producer's own predicates rather than a restatement of them:

    NO_HEADER_CELL       the column has no cell in any header row the producer reads, so
                         the period lives in a row or in the table's own context
    FULL_DATE_ACCEPTED   a full date the producer's rule 1 accepts, and yet unbound.
                         Expected to be empty; if it is not, that is a defect in the
                         producer rather than a question about the filing.
    BARE_YEAR            rule 3 should have bound it.  Also expected to be empty.
    FULL_DATE_REFUSED    a full date, refused by `_is_period_header_cell` -- recorded with
                         the rest-of-cell string, because that predicate has two very
                         different reasons for refusing and they mean opposite things
    MONTH_DAY_NO_YEAR    a month-day with the year elsewhere in the filing
    NO_DATE_IN_COLUMN    the lens found a period in the store's column_header but the
                         producer's own header rows hold no date expression at all

Each structural bucket is then read against four verdicts, and the verdict is *reported*
rather than assumed, so a bucket that does not fit any of them is visible as such:

    A  V2_PRODUCER_GAP                 the source states it plainly, V2 missed it
    B  LEGACY_FALSE_POSITIVE           the "column-local" period is prose or a range
    C  VALID_BUT_OUT_OF_SCOPE_GEOMETRY correct period, shape the column model cannot state
    D  SOURCE_AMBIGUOUS                cannot be proven either way -- do not delete

  python audit_valid_losses.py --delta <artifact> --store <store-v2.jsonl> --out <dir>
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
PB_SHADOW = _BACKEND_DIR / "scripts/evaluation/period_binding_shadow.py"
TRUTH = _BACKEND_DIR / "scripts/evaluation/diagnose_loss_period_truth.py"
CORPUS = Path("/disk/qh/nano-finrag/data/financial_corpus_v2/raw/SEC")

#: Structural bucket -> the verdict it supports, and why.  Kept beside each other so a
#: disagreement between the two is legible rather than buried in a branch.
VERDICTS = {
    "NO_HEADER_CELL": ("C", "the period is not in a header row; a row or the table states it"),
    "FULL_DATE_ACCEPTED": ("A", "the producer's own rule 1 accepts this cell"),
    "BARE_YEAR": ("A", "rule 3 binds a bare-year cell"),
    "FULL_DATE_REFUSED": ("B", "refused, so the date sits in prose, a range, or a second "
                               "year -- read `rest` to tell which"),
    "MONTH_DAY_NO_YEAR": ("C", "a month-day whose year is not in the column"),
    "NO_DATE_IN_COLUMN": ("D", "the lens found a period the producer's rows do not hold"),
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def diagnose_column(nf, pb, grid, union, col) -> tuple[str, str, str]:
    """`(bucket, cell text, rest-of-cell)` for one unbound column."""
    cells: list[tuple[int, str]] = []
    for i in union:
        if i < len(grid) and col < len(grid[i]) and grid[i][col]:
            text = nf.ws(grid[i][col]["raw_text"])
            if text:
                cells.append((i, text))
    if not cells:
        return "NO_HEADER_CELL", "", ""

    for _, text in cells:
        match = pb._MONTH_DAY_YEAR.search(text)
        if match and pb._is_period_header_cell(text, match):
            return "FULL_DATE_ACCEPTED", text, ""

    for _, text in cells:
        match = pb._MONTH_DAY_YEAR.search(text)
        if match:
            rest = text[:match.start()] + " " + text[match.end():]
            rest = re.sub(r"\([^)]*\)", " ", rest)
            rest = re.sub(r"[\s,;:—–-]+", " ", rest).strip()
            return "FULL_DATE_REFUSED", text, rest

    for _, text in cells:
        if pb._MONTH_DAY.search(text):
            return "MONTH_DAY_NO_YEAR", text, ""

    for _, text in cells:
        if pb._BARE_YEAR.match(text):
            return "BARE_YEAR", text, ""

    return "NO_DATE_IN_COLUMN", " / ".join(t for _, t in cells)[:120], ""


#: A cell that states what the period *is*, rather than merely containing a date.
_PERIOD_PHRASE = re.compile(
    r"\b(as of|at|ended|ending|on|through|for the (?:year|quarter|period|three|six|nine|"
    r"fiscal)|year(?:s)?\s+ended|quarter\s+ended|three\s+months\s+ended|fiscal\s+year|"
    r"period\s+ended)\s*$", re.I)
#: A second period expression in the same cell -- the cell declares more than one.
_SECOND_PERIOD = re.compile(r"\b(?:19|20)\d{2}\b.*\b(?:19|20)\d{2}\b")
_CAPTION = re.compile(r"\([^)]*\)")
_SEPARATORS = re.compile(r"[\s,;:—–*]+")


#: Hand-adjudicated, family by family, after reading every family in full.
#:
#: The regex that groups the families is not the adjudicator -- it only groups.  These
#: verdicts are read off the cells, each with the reason it was reached, and the ones that
#: are a judgement call say so.  A rule was tried first and it put 950 of 960 in one class,
#: which is the signature of a rule that has stopped discriminating: `Fair value at
#: December 31, 2025` and `...Performance Share Units vesting on March 25, 2026` have the
#: same shape -- clause, preposition, date -- and only one of them names a period.
#:
#:   (prefix the family's cells share, verdict, reason)
ADJUDICATIONS = (
    ("For the Year Ended", "A",
     "a period header in words -- `For the Year Ended September 30, 2025` is what a "
     "duration column looks like, and only the twelve-character prose rule refused it"),
    ("As of or for the year ended", "A",
     "same, in the long form JPMorganChase uses"),
    ("Fair value at", "A",
     "an as-of date naming the column's measurement point, in a fair-value rollforward"),
    ("Contractual rate in effect at", "A",
     "an as-of date"),
    ("Fair value purchase price allocation as of", "A",
     "an as-of date"),
    ("Central case assumptions at", "A",
     "an as-of date"),
    ("Change in unrealized gains", "A",
     "the same as-of date inside a longer clause; the date is the column's point"),
    ("Amounts Recognized as of Acquisition Date", "D",
     "the date sits inside a parenthetical aside -- `(as previously reported as of "
     "December 31, 2023)` -- so which period the column is reported at is not provable "
     "from the cell"),
    ("Summary of TSRU and PTU information as of", "D",
     "a note title carrying its as-of date; whether the column is reported at that date "
     "cannot be settled from the header alone"),
    ("50% of the net issued shares", "B",
     "a vesting clause describing a share tranche -- `vesting on March 25, 2026` -- not a "
     "period the column is reported at"),
    ("President and Chief Financial Officer", "B", "a signature block"),
    ("Chairman of the Board of Directors", "B", "a signature block"),
    ("Recovery of Erroneously Awarded", "B", "a policy description"),
)


def adjudicate(prefix: str) -> tuple[str, str]:
    for shared, verdict, reason in ADJUDICATIONS:
        if prefix.startswith(shared):
            return verdict, reason
    return "D", "not adjudicated -- no family matched, so it is not deleted"


def family_of(pb, bucket: str, text: str, rest: str) -> tuple[str, str, str]:
    """`(family, verdict, reason)` for one unbound column."""
    if bucket != "FULL_DATE_REFUSED":
        if bucket == "NO_DATE_IN_COLUMN":
            return (f"the column names no date at all: {text[:40]!r}",
                    *VERDICTS[bucket])
        if bucket == "MONTH_DAY_NO_YEAR":
            return f"month-day without a year: {text[:40]!r}", *VERDICTS[bucket]
        return bucket, *VERDICTS[bucket]

    match = pb._MONTH_DAY_YEAR.search(text)
    if match is None:  # pragma: no cover - diagnose_column returns this bucket only on a match
        return f"unreadable: {text[:40]!r}", *VERDICTS[bucket]
    prefix = _SEPARATORS.sub(" ", _CAPTION.sub(" ", text[:match.start()])).strip()
    verdict, reason = adjudicate(prefix)
    return f"{prefix[:46]!r}", verdict, reason


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)

    nf = load("nf17a4", PARSER)
    builder = load("store_builder", BUILDER)
    pb = load("pb_shadow", PB_SHADOW)
    truth = load("period_truth", TRUTH)
    from lxml import etree, html

    report = json.loads(args.delta.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in args.store.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    by_cell = {str(r.get("cell_id")): r for r in records if r.get("cell_id")}

    # --- the 1,649, selected by the W4-A5 lens and by nothing else -------------------
    plausibly = []
    for entry in report["producer_loss"]:
        if entry["legacy_blocker"] != "":
            continue
        record = by_cell.get(str(entry["cell_id"]))
        if record is None:
            continue
        if truth.classify(record) == truth.DECLARED:
            plausibly.append((entry, record))

    columns: dict[tuple, list] = collections.defaultdict(list)
    for entry, record in plausibly:
        columns[(entry["document_id"], entry["source_order"], entry["column_index"])].append(
            (entry, record))

    print(f"=== {len(plausibly)} plausibly-column-local losses, "
          f"across {len(columns)} columns ===")
    print()

    out = {"phase": "P1.6-A3-W4-A6", "mutation": "none", "cells": len(plausibly),
           "columns": len(columns), "buckets": {}, "columns_detail": []}
    buckets: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    detail: dict[str, list] = collections.defaultdict(list)
    #: The families inside a bucket, keyed by the text that separates them.  The verdict
    #: map above is written from expectation; this is written from the filings, and where
    #: the two disagree this is the one to believe.  `FULL_DATE_REFUSED` is the case that
    #: forced it: `As of or for the year ended December 31, 2024` and `Indenture, dated as
    #: of October 28, 2021` land in the same bucket and mean opposite things.
    families: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    #: The reason recorded for each verdict, kept so the summary states why rather than
    #: only how many.
    verdict_reason: dict[str, str] = {}

    by_table = collections.defaultdict(list)
    for key in columns:
        by_table[key[0]].append(key)

    for document_id, keys in sorted(by_table.items(),
                                    key=lambda kv: -sum(len(columns[k]) for k in kv[1])):
        ticker, accession = builder.DOCUMENTS[document_id]
        # Once per filing, not once per table.  The first version parsed the whole filing
        # inside the table loop, which is the same 40 MB of HTML re-read for every table
        # in it and is why a 1,649-cell audit took over ten minutes.
        parsed = builder.parse_filing(ticker, accession, document_id)
        by_order = {t.get("source_order"): t for t in parsed["tables"]}
        root = html.parse(str(CORPUS / ticker / accession / "primary.html"),
                          etree.HTMLParser(recover=True, no_network=True, huge_tree=True,
                                           remove_comments=True)).getroot()
        _b, lookup, _p = nf.make_blocks(
            root, {"document_id": document_id, "ticker": ticker, "role": "ANNUAL"})
        geometry: dict[int, tuple] = {}
        for order in {k[1] for k in keys}:
            table = by_order.get(order)
            if table is None or table["table_id"] not in lookup:
                continue
            grid = nf.grid_rows(nf.direct_rows(lookup[table["table_id"]]))
            geometry[order] = (table["table_id"], grid,
                               pb.extended_header_idx(nf, grid)[0])

        for order in sorted({k[1] for k in keys}):
            if order not in geometry:
                continue
            tid, grid, union = geometry[order]
            table_keys = [k for k in keys if k[1] == order]

            for key in table_keys:
                col = key[2]
                members = columns[key]
                entry, record = members[0]
                bucket, text, rest = diagnose_column(nf, pb, grid, union, col)
                family, verdict, why = family_of(pb, bucket, text, rest)
                buckets[bucket][verdict] += len(members)
                buckets[bucket]["columns"] += 1
                families[bucket][family] += len(members)
                verdict_reason.setdefault(verdict, why)
                detail[bucket].append({
                    "document_id": document_id, "source_order": order,
                    "table_fragment_id": tid, "column_index": col,
                    "cells": len(members), "table_role": entry["table_role"],
                    "statement_type": record.get("statement_type"),
                    "legacy_kind": entry["legacy_kind"],
                    "v2_reason": entry["v2_reason"],
                    "period": record.get("period_end") or record.get("period"),
                    "column_header": str(record.get("column_header"))[:120],
                    "producer_reads": text[:120], "rest": rest[:60],
                    "verdict": verdict,
                })

            role = collections.Counter(e["table_role"] for key in table_keys
                                       for e, _ in columns[key])
            print(f"--- {document_id}#{order}   {len(table_keys)} columns   {dict(role)}")
            for key in table_keys[:4]:
                entry, record = columns[key][0]
                bucket, text, rest = diagnose_column(nf, pb, grid, union, key[2])
                print(f"    col {key[2]:<4} {bucket:<20} x{len(columns[key]):<4} "
                      f"{str(record.get('period_end') or record.get('period')):<12} "
                      f"{text[:52]!r}")
                if rest:
                    print(f"          rest of cell: {rest[:64]!r}")
            print()

    print("=== structural buckets ===")
    for bucket, counts in sorted(buckets.items(), key=lambda kv: -kv[1]["columns"]):
        cells = sum(v for k, v in counts.items() if k != "columns")
        splits = ", ".join(f"{v}:{counts[v]}" for v in "ABCD" if counts.get(v))
        print(f"  {bucket:<20} columns {counts['columns']:>4}  cells {cells:>5}   {splits}")
        for sample in detail[bucket][:args.samples]:
            print(f"        e.g. {sample['document_id']} ord={sample['source_order']} "
                  f"col={sample['column_index']} period={sample['period']} "
                  f"role={sample['table_role']} kind={sample['legacy_kind']}")
            print(f"          producer read: {sample['producer_reads'][:96]!r}")
    print()

    print("=== families inside each bucket, adjudicated ===")
    for bucket, counts in sorted(families.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {bucket}")
        for family, count in counts.most_common(14):
            verdict, reason = adjudicate(family.strip("'")) if bucket == "FULL_DATE_REFUSED" \
                else VERDICTS[bucket]
            print(f"      {count:>5}  {verdict}  {family}")
            print(f"             {reason[:96]}")
    print()

    print("=== verdicts, over cells ===")
    verdict_cells = collections.Counter()
    for bucket, counts in buckets.items():
        for verdict in "ABCD":
            verdict_cells[verdict] += counts.get(verdict, 0)
    total = sum(verdict_cells.values())
    names = {"A": "V2_PRODUCER_GAP", "B": "LEGACY_FALSE_POSITIVE",
             "C": "VALID_BUT_OUT_OF_SCOPE_GEOMETRY", "D": "SOURCE_AMBIGUOUS"}
    for verdict in "ABCD":
        count = verdict_cells.get(verdict, 0)
        print(f"  {verdict}  {names[verdict]:<32} {count:>5}  "
              f"({100.0 * count / max(1, total):4.1f}%)")

    out["buckets"] = {b: dict(c) for b, c in buckets.items()}
    out["families"] = {b: dict(c) for b, c in families.items()}
    out["verdicts"] = {names[v]: verdict_cells.get(v, 0) for v in "ABCD"}
    out["columns_detail"] = {b: d for b, d in detail.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "valid-loss-audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"\n  written to {args.out / 'valid-loss-audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
