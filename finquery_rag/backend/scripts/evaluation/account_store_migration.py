"""P1.6-A3-W4-B: what the rebuild changed, fact by fact and class by class.

A count delta is not an accounting.  The manifest says the migration should remove 749
cells under three headings; this says whether the rebuilt store did that, and refuses to
close if anything it added or removed falls outside a known class:

    UNCLASSIFIED_ADDED    -> blocks closure
    UNCLASSIFIED_REMOVED  -> blocks closure

Facts are matched on the cell they came from, which is the only identity a store record and
the audit share.

The three removal classes are kept apart throughout and never summed into one number.
`legacy false positives removed` is the only correction; the other two are facts withheld
because the source does not settle them, and calling all three "removed incorrect facts"
would be a claim none of them supports.

  python account_store_migration.py --old <store> --new <store> --audit <json> \
      --shadow <json> --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

#: The manifest's headings, spelled the same way so the two cannot drift apart.
REMOVAL_CLASSES = ("LEGACY_FALSE_POSITIVE", "SOURCE_AMBIGUOUS",
                   "VALID_BUT_OUT_OF_SCOPE_GEOMETRY")


def read_store(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        out[str(record.get("cell_id"))] = record
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    old = read_store(args.old)
    new = read_store(args.new)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    shadow = json.loads(args.shadow.read_text(encoding="utf-8"))

    # cell -> removal class, from the adjudicated audit.
    cell_class: dict[str, str] = {}
    for bucket, entries in audit["columns_detail"].items():
        for entry in entries:
            for cell_id in entry.get("cell_ids", []):
                cell_class[str(cell_id)] = entry["verdict"]

    # cell -> (method, status, kind) for everything the migration would ADD, from the
    # shadow run.  The store has no such fields -- persisting them is W5 -- so the only
    # place they exist is the run that decided them.
    added_provenance: dict[str, dict] = {}
    for name in ("rule_gain", "producer_gain"):
        for entry in shadow.get(name, []):
            if entry["v2_outcome"] == "WITHHOLD":
                continue
            added_provenance[str(entry["cell_id"])] = {
                "method": entry.get("binding_method"),
                "status": entry.get("binding_status"),
                "granularity": entry.get("binding_granularity"),
                "legacy_kind": entry.get("legacy_kind"),
                "table_role": entry.get("table_role"),
            }

    removed = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    unchanged = set(old) & set(new)

    report = {"phase": "P1.6-A3-W4-B-accounting", "mutation": "applied",
              "old_count": len(old), "new_count": len(new),
              "removed": len(removed), "added": len(added),
              "unchanged": len(unchanged), "failures": []}
    tally = collections.Counter()

    print(f"=== Store V2 ===")
    print(f"    before {len(old):>7}")
    print(f"    after  {len(new):>7}")
    print(f"    net    {len(new) - len(old):>+7}")
    print(f"    added {len(added)}   removed {len(removed)}   unchanged {len(unchanged)}")
    print()

    print("=== removed, by class ===")
    for cell_id in removed:
        verdict = cell_class.get(cell_id)
        tally[f"removed: {verdict or 'UNCLASSIFIED_REMOVED'}"] += 1
    for key in sorted(tally):
        if key.startswith("removed"):
            print(f"    {key:<52} {tally[key]}")
    report["removed_by_class"] = {
        k.split(": ", 1)[1]: v for k, v in tally.items() if k.startswith("removed")}
    unexplained_removed = tally.get("removed: UNCLASSIFIED_REMOVED", 0)
    if unexplained_removed:
        report["failures"].append(f"UNCLASSIFIED_REMOVED = {unexplained_removed}")
    print()

    print("=== added, by provenance ===")
    by = collections.defaultdict(collections.Counter)
    unclassified_added = 0
    for cell_id in added:
        prov = added_provenance.get(cell_id)
        if prov is None:
            unclassified_added += 1
            continue
        for field in ("method", "status", "granularity", "legacy_kind", "table_role"):
            by[field][str(prov.get(field))] += 1
    for field in ("method", "status", "granularity", "legacy_kind", "table_role"):
        print(f"    by {field}:")
        for value, count in by[field].most_common(6):
            print(f"        {count:>6}  {value}")
    print(f"    UNCLASSIFIED_ADDED  {unclassified_added}")
    if unclassified_added:
        report["failures"].append(f"UNCLASSIFIED_ADDED = {unclassified_added}")
    report["added_by"] = {f: dict(c) for f, c in by.items()}
    report["unclassified_added"] = unclassified_added
    report["unclassified_removed"] = unexplained_removed
    print()

    print("=== the three classes, kept apart ===")
    for name in REMOVAL_CLASSES:
        count = report["removed_by_class"].get(name, 0)
        print(f"    {name:<34} {count:>5}")
        report.setdefault("removal_readings", {})[name] = {
            "count": count,
            "reading": {
                "LEGACY_FALSE_POSITIVE":
                    "a legacy binding the source does not support -- the only correction",
                "SOURCE_AMBIGUOUS":
                    "the source does not settle the period either way; withheld, not deleted",
                "VALID_BUT_OUT_OF_SCOPE_GEOMETRY":
                    "a correct period the column model cannot express",
            }[name],
        }
    print()

    print(f"  failures: {report['failures'] or 'none'}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "store-migration-accounting.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'store-migration-accounting.json'}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
