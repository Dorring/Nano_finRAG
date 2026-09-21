"""P1.6-A3-W4-B3: adjudicate every `period` value the B2 bridge moved.

W4-B2 was scoped to *persist* a fact's period identity.  It also changed the `period`
field on cells that were admitted both before and after, because the emitted fact now
takes its `normalized_period` from the binding rather than from the legacy axis.  That is
one line of the bridge and it deserves one account, so this is it.

Three directions, which mean three different things:

  RESPELLED  same precision, different string.  Mostly the legacy `FY2024` convention
             against the `2024` the source actually wrote; the rest are exhibit-index
             date columns, where neither value is a financial period.
  NARROWED   the old `period` stated a day the new one does not.  These are only
             acceptable where the day was never the cell's: the check is that the old
             `period` **contradicted the cell's own `period_end`**, which is the field
             the resolver reads and which never moved.
  WIDENED    the new `period` states a day the old one did not.  This is the direction
             that would be a fabrication if anything were, so it is bounded by a gate:
             every widened cell must sit in a table that cannot speak for the company.

The gate that makes the whole thing safe is stated first and checked on every moved cell:
`period_end` does not move.  `period` is a rendering; `period_end` is the fact's physical
period and the resolver's input.  A rendering may change only where the thing rendered
did not.

  python audit_period_field_change.py --old <store> --new <store> --out <dir>
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

_FULL_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: A table that may speak for the company.  A widened period inside one of these would
#: need reading by hand, because it would be the store asserting a day on a figure that
#: reaches production.
AUTHORITATIVE = "PRIMARY_FINANCIAL_STATEMENT"


def read_store(path: Path) -> dict[str, dict]:
    return {str(r["cell_id"]): r
            for r in (json.loads(line)
                      for line in path.read_text(encoding="utf-8").splitlines()
                      if line.strip())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", type=int, default=8)
    args = parser.parse_args(argv)

    old, new = read_store(args.old), read_store(args.new)
    unchanged = set(old) & set(new)
    moved = [c for c in unchanged if old[c].get("period") != new[c].get("period")]

    failures: list[str] = []
    report: dict = {"phase": "P1.6-A3-W4-B3-period-adjudication",
                    "unchanged": len(unchanged), "period_moved": len(moved)}

    print(f"unchanged cells {len(unchanged)}   period moved {len(moved)}")
    print()

    # --- the gate the rest of this rests on ------------------------------------------
    end_moved = [c for c in moved if old[c].get("period_end") != new[c].get("period_end")]
    print("=== the field the resolver reads ===")
    print(f"    period_end moved on those cells   {len(end_moved)}")
    if end_moved:
        failures.append(f"PERIOD_END_MOVED = {len(end_moved)}")
        for cell in end_moved[:5]:
            print(f"        {cell} {old[cell].get('period_end')!r} -> "
                  f"{new[cell].get('period_end')!r}")
    report["period_end_moved"] = len(end_moved)
    print()

    # --- direction and self-consistency ------------------------------------------------
    direction = collections.Counter()
    agreement = collections.Counter()
    contradicted_before = 0
    for cell in moved:
        before = str(old[cell].get("period") or "")
        after = str(new[cell].get("period") or "")
        end = str(old[cell].get("period_end") or "")
        if _FULL_DATE.match(after) and not _FULL_DATE.match(before):
            direction["WIDENED"] += 1
        elif _FULL_DATE.match(before) and not _FULL_DATE.match(after):
            direction["NARROWED"] += 1
            # The legacy path was `normalized_period or period_end`, so the two fields
            # could disagree.  Where they did, the record said two things about one cell.
            if before != end:
                contradicted_before += 1
        else:
            direction["RESPELLED"] += 1
        agreement[f"{'old agrees' if before == end else 'old differs'}"
                  f" -> {'new agrees' if after == end else 'new differs'}"] += 1

    print("=== direction ===")
    for name, count in direction.most_common():
        print(f"    {count:>6}  {name}")
    print()
    print("=== did `period` agree with the cell's own `period_end`? ===")
    for name, count in agreement.most_common():
        print(f"    {count:>6}  {name}")
    print()
    report["direction"] = dict(direction)
    report["agreement"] = dict(agreement)
    report["narrowed_that_contradicted_period_end"] = contradicted_before

    # A narrowing is only safe where the day was never the cell's to begin with.  If a
    # narrowed cell's own `period_end` is a full date, then a day *was* available in the
    # record and `period` stopped showing it -- that is a real loss and blocks.
    unsafe_narrow = [c for c in moved
                     if _FULL_DATE.match(str(old[c].get("period") or ""))
                     and not _FULL_DATE.match(str(new[c].get("period") or ""))
                     and _FULL_DATE.match(str(old[c].get("period_end") or ""))]
    print(f"narrowed cells whose own `period_end` held a full date   {len(unsafe_narrow)}")
    for cell in unsafe_narrow[:5]:
        print(f"    {cell} {old[cell].get('period')!r} -> {new[cell].get('period')!r}  "
              f"period_end {old[cell].get('period_end')!r}")
    if unsafe_narrow:
        failures.append(f"DAY_LOST_FROM_PERIOD = {len(unsafe_narrow)}")
    report["day_lost_from_period"] = len(unsafe_narrow)
    print()

    # --- widened cells, bounded by table authority -------------------------------------
    widened = [c for c in moved
               if _FULL_DATE.match(str(new[c].get("period") or ""))
               and not _FULL_DATE.match(str(old[c].get("period") or ""))]
    print(f"=== WIDENED: {len(widened)} cells where a day appears ===")
    roles = collections.Counter(str(new[c].get("table_role")) for c in widened)
    docs = collections.Counter(
        str(new[c].get("source", {}).get("document_id")) for c in widened)
    for name, count in roles.most_common():
        print(f"    {count:>6}  role {name}")
    for name, count in docs.most_common():
        print(f"    {count:>6}  in {name}")
    on_authoritative = [c for c in widened
                        if str(new[c].get("table_role")) == AUTHORITATIVE]
    if on_authoritative:
        print(f"    {len(on_authoritative)} of them sit in a table that may speak for "
              f"the company, and need reading:")
        for cell in on_authoritative[:10]:
            print(f"        {new[cell].get('row_label')!r} "
                  f"{str(new[cell].get('column_header'))[:90]!r}")
        failures.append(f"WIDENED_ON_AUTHORITATIVE_TABLE = {len(on_authoritative)}")
    report["widened"] = len(widened)
    report["widened_by_role"] = dict(roles)
    report["widened_by_document"] = dict(docs)
    report["widened_on_authoritative"] = len(on_authoritative)
    print()

    # --- the families, so they can be read as families ---------------------------------
    families: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    samples: dict[tuple, tuple] = {}
    for cell in moved:
        before = str(old[cell].get("period") or "")
        after = str(new[cell].get("period") or "")
        if _FULL_DATE.match(after) and not _FULL_DATE.match(before):
            kind = "WIDENED"
        elif _FULL_DATE.match(before) and not _FULL_DATE.match(after):
            kind = "NARROWED"
        else:
            kind = "RESPELLED"
        header = re.sub(r"\s+", " ", str(new[cell].get("column_header") or ""))[:60]
        key = (kind, str(new[cell].get("table_role")), header)
        families[key][f"{new[cell].get('source', {}).get('document_id')}"
                       f"#{new[cell].get('table_fragment_id')}"] += 1
        samples.setdefault(key, (before, after, new[cell].get("column_header")))

    for kind in ("NARROWED", "WIDENED", "RESPELLED"):
        rows = sorted(((k, v) for k, v in families.items() if k[0] == kind),
                      key=lambda kv: -sum(kv[1].values()))
        total = sum(sum(v.values()) for _, v in rows)
        print(f"=== {kind}: {total} cells in {len(rows)} table families ===")
        for (family_kind, role, header_key), tables in rows[:args.families]:
            before, after, header = samples[(family_kind, role, header_key)]
            print(f"    {sum(tables.values()):>5} cells  {len(tables):>3} tables  role={role}")
            print(f"          {before!r} -> {after!r}")
            print(f"          header {str(header)[:120]!r}")
        print()
    report["families"] = {f"{k[0]}/{k[1]}/{k[2]}": sum(v.values())
                          for k, v in families.items()}

    print(f"  failures: {failures or 'none'}")
    report["failures"] = failures
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "period-field-adjudication.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'period-field-adjudication.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
