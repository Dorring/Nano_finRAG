"""P1.6-A3-W4-A: why the new period producer has no binding where the legacy axis has one.

Diagnosis only, and it reads the W4-A delta artifact rather than the corpus, so it costs
nothing to re-run after a fix.

W4-A measured two deltas and they came out nothing alike:

    legacy -> V1   the rule change      +14,802 / -0        one kind wide, as contracted
    V1     -> V2   the producer change  +0      / -18,104   all of it `NO_BINDING`

`+0` is the part that decides the phase.  The producer never binds a cell the legacy axis
missed, so V2 is not a wider path with a different failure mode -- it is strictly narrower,
and switching to it would lose facts the store holds today.  9,688 of those were admitted
by the legacy rule; 708 of them are in **oracle-PRIMARY** tables.

This classifies each lost cell by what its column header actually says, which is what
turns "the producer is narrower" into a list of named things to fix:

    PRIMARY            352  an abbreviated month name -- `Jan 26, 2025`.  The binder's
                            month pattern lists full names only, so NVIDIA's fiscal
                            calendar matches nothing at all.
                       332  `as of December 31` with the year in a sibling cell.  The
                            adjacent-year join only fires on a *bare* year cell
                            (`^2025$`); here the year travels with other text.
                        22  `Year ended December 31` in every column, no year present --
                            the same join, one step further out.
                         2  an empty header.  Correctly unbound.

    non-primary      2,907  a month-day *range* (`June 29, 2025 to August 2, 2025:`) or a
                            table of contents.  Refusing these is the producer being
                            right, and matches what `table_role` already says about them.

So the primary losses are almost entirely two mechanical gaps, and the non-primary losses
are mostly correct refusals.  The buckets are reported together rather than as one number
because "18,104 lost" reads as a catastrophe and "684 primary cells, two named causes"
does not.

  python diagnose_admission_losses.py --delta <path to emission-admission-shadow.json>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

ABBREVIATED_MONTH = re.compile(
    r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}\b")
FULL_MONTH_DAY = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|"
    r"December)\.?\s+\d{1,2}\b")
AS_OF_WITHOUT_YEAR = re.compile(
    r"\bas of\s+\w+\.?\s+\d{1,2}\s*(?!,?\s*(?:19|20)\d{2})", re.I)

CAUSES = (
    "abbreviated month name",
    "`as of <month day>` with no year in the same cell",
    "month-day present, still unbound (range, prose, or a year that travels)",
    "no month-day expression in the column header",
)


def cause_of(entry: dict) -> str:
    header = str(entry.get("column_header") or "")
    if ABBREVIATED_MONTH.search(header):
        return CAUSES[0]
    if AS_OF_WITHOUT_YEAR.search(header):
        return CAUSES[1]
    if FULL_MONTH_DAY.search(header):
        return CAUSES[2]
    return CAUSES[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args(argv)

    report = json.loads(args.delta.read_text(encoding="utf-8"))

    # The regression direction only: cells the legacy **admitted** that V2 withholds.  The
    # rest of `producer_loss` is cells the legacy never stored either, so nothing is lost
    # by withholding them and counting them here would inflate the damage.
    lost = [e for e in report["producer_loss"] if e["legacy_blocker"] == ""]
    print(f"=== cells the legacy stores and V2 would not: {len(lost)} ===")
    print()
    for role, count in collections.Counter(e["table_role"] for e in lost).most_common():
        print(f"    {str(role):<20} {count}")
    print()

    for role, _ in collections.Counter(e["table_role"] for e in lost).most_common():
        subset = [e for e in lost if e["table_role"] == role]
        print(f"=== {role}: {len(subset)} ===")
        for name in CAUSES:
            subset_of = [e for e in subset if cause_of(e) == name]
            if not subset_of:
                continue
            print(f"    {len(subset_of):>6}  {name}")
            for entry in subset_of[:args.samples]:
                print(f"              {entry['document_id']} ord={entry['source_order']} "
                      f"col={entry['column_index']} "
                      f"{str(entry['column_header'])[:60]!r}")
        print()
        if role == "PRIMARY":
            gap = sum(1 for e in subset if cause_of(e) in CAUSES[:2])
            print(f"    -> {gap} of {len(subset)} primary cells are the two mechanical "
                  f"gaps ({', '.join(CAUSES[:2])})")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
