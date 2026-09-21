"""P1.6-A3-W4-A: why the new period producer has no binding where the legacy axis has one.

Diagnosis only, and it reads the W4-A delta artifact rather than the corpus, so it costs
nothing to re-run after a fix.

The first version of this classified a lost cell by whether an abbreviated month name
appeared anywhere in its column header, and W4-A2 falsified it.  It predicted the
abbreviated-month widening would recover 352 of the 708 oracle-PRIMARY losses; it
recovered 234, and 118 of the cells it had blamed on abbreviations were still lost
afterwards:

    nvda ord=7899 col=27  'Retained / Earnings / ... / as of Jan 30 / ... / as of Jan 29'

`Jan 30` is abbreviated, and that is incidental -- the cell is lost because there is no
**year** in the column, exactly like the 332 it had filed under a different heading.  The
buckets overlapped and the first match won, so one cause was reported as two and the count
attached to the wrong one.

So this now asks the structural question instead of the lexical one:

    no month-day expression in the column header     nothing to bind
    a month-day with no year in the column header    the year is not here to join to
    both present, still unbound                      refused -- prose, a range, or a
                                                     second year beside it

Reading, not assuming: `abbreviated month` is deliberately not a bucket any more, because
after the widening both spellings take the same path and the distinction no longer
explains anything.

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

_MONTH_WORD = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
               r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
               r"Dec(?:ember)?)")
MONTH_DAY = re.compile(rf"\b{_MONTH_WORD}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?", re.I)
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

NO_DATE = "no month-day expression in the column header"
NO_YEAR = "a month-day with no year in this column's header"
BOTH_PRESENT = "month-day and year both present, still unbound"

CAUSES = (NO_DATE, NO_YEAR, BOTH_PRESENT)


def cause_of(entry: dict) -> str:
    """Which of the three structural situations this column's header is in.

    Deliberately structural.  The lexical version this replaces asked whether a month name
    was abbreviated, which is true of a cell whose actual problem is a missing year, and
    the resulting count was wrong by a third.
    """
    header = str(entry.get("column_header") or "")
    if not MONTH_DAY.search(header):
        return NO_DATE
    if not YEAR.search(header):
        return NO_YEAR
    return BOTH_PRESENT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--width", type=int, default=60)
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
                      f"{str(entry['column_header'])[:args.width]!r}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
