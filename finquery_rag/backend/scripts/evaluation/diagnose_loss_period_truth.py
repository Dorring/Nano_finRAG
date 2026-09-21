"""P1.6-A3-W4-A5: is the period the store would lose actually the cell's own?

Diagnosis only.

W4-A2 measured 8,369 cells the legacy stores and the new producer withholds.  W4-A3 showed
that for the 474 oracle-PRIMARY ones the legacy period is **not** the cell's period -- a
row-major equity statement where the column carries no period at all -- so "the store would
lose these" was the wrong description of them.  This asks the same question of the other
7,895, which is what decides whether W4-B's deficit is a cost or mostly a correction.

The lens generalises off the equity special case rather than reusing its anchors, because
the other tables are not equity statements.  The one thing every fact in Store V2 carries
is its own column header, so the question becomes:

    does the column this fact sits in declare a period, and is the fact's period that one?

    one year in the column, and it is the fact's     the column declares it -- plausible
    one year in the column, and it is not the fact's the fact contradicts its own column
    several years in the column                      the column declares no single period,
                                                     so the fact's period is a choice among
                                                     them rather than a declaration
    no year in the column                            the period came from outside the
                                                     column altogether

Only the first is a period the source states for that cell.  The other three are the legacy
picking a date from somewhere other than the cell's own column, which is exactly the equity
defect in a shape that does not need equity anchors to see.

  python diagnose_loss_period_truth.py --delta <artifact> --store <store-v2.jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_MONTH_WORD = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
               r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
               r"Dec(?:ember)?)")
MONTH_DAY_YEAR = re.compile(
    rf"\b{_MONTH_WORD}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}\b", re.I)

DECLARED = "the column names one period and it is the fact's"
CONTRADICTS = "the column names one period and it is NOT the fact's"
CHOICE = "the column names several periods -- the fact's is a choice among them"
OUTSIDE = "the column names no period -- it came from outside the column"

CAUSES = (DECLARED, CONTRADICTS, CHOICE, OUTSIDE)


def named_periods(header: str) -> set[str]:
    """Every distinct period the column header names.

    Counting *years* alone is not enough, and the first version of this did exactly that:
    `June 29, 2025 to August 2, 2025: / August 3, 2025 to August 30, 2025:` names one year
    and several periods, so it was scored as a column that declares a single period and the
    fact's -- a false positive on a weekly-reporting column.  A month-day-year counts as a
    period in its own right; a bare year only counts if no month-day already claimed it.
    """
    found = {m.group(0).casefold() for m in MONTH_DAY_YEAR.finditer(header)}
    found |= set(YEAR.findall(MONTH_DAY_YEAR.sub(" ", header)))
    return found


def classify(record: dict) -> str:
    periods = named_periods(str(record.get("column_header") or ""))
    claimed = YEAR.findall(str(record.get("period_end") or record.get("period") or ""))
    if not periods:
        return OUTSIDE
    if len(periods) > 1:
        return CHOICE
    if claimed and claimed[0] in periods.pop():
        return DECLARED
    return CONTRADICTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args(argv)

    report = json.loads(args.delta.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in args.store.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    by_cell = {str(r.get("cell_id")): r for r in records if r.get("cell_id")}

    # The regression direction: cells the legacy admitted and V2 withholds.
    lost = [e for e in report["producer_loss"] if e["legacy_blocker"] == ""]

    out = {"phase": "P1.6-A3-W4-A5", "mutation": "none", "total": len(lost),
           "by_role": {}, "samples": collections.defaultdict(list)}
    overall = collections.Counter()
    by_role: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    missing = 0

    for entry in lost:
        record = by_cell.get(str(entry["cell_id"]))
        if record is None:
            missing += 1
            continue
        verdict = classify(record)
        overall[verdict] += 1
        by_role[str(entry["table_role"])][verdict] += 1
        if len(out["samples"][verdict]) < args.samples:
            out["samples"][verdict].append({
                "document_id": entry["document_id"], "source_order": entry["source_order"],
                "column_index": entry["column_index"],
                "row_label": record.get("row_label"),
                "column_header": str(record.get("column_header"))[:110],
                "period_end": record.get("period_end"), "period": record.get("period"),
                "table_role": entry["table_role"],
            })

    print(f"=== {len(lost)} cells the legacy stores and V2 withholds ===")
    print(f"    {missing} not found in the store by cell_id")
    print()
    for name in CAUSES:
        count = overall.get(name, 0)
        share = 100.0 * count / max(1, len(lost) - missing)
        print(f"  {count:>6}  {share:5.1f}%  {name}")
        for sample in out["samples"][name]:
            print(f"            {sample['document_id']} ord={sample['source_order']} "
                  f"col={sample['column_index']} "
                  f"period={sample['period_end'] or sample['period']!r}")
            print(f"              row {str(sample['row_label'])[:44]!r}")
            print(f"              col {sample['column_header'][:88]!r}")
    print()

    print("=== by table role ===")
    for role in sorted(by_role, key=lambda r: -sum(by_role[r].values())):
        counts = by_role[role]
        total = sum(counts.values())
        declared = counts.get(DECLARED, 0)
        print(f"  {role:<20} {total:>6}   declared {declared:>6} "
              f"({100.0 * declared / max(1, total):4.1f}%)   "
              f"not-declared {total - declared:>6}")

    out["overall"] = dict(overall)
    out["by_role"] = {r: dict(c) for r, c in by_role.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "loss-period-truth.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"\n  written to {args.out / 'loss-period-truth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
