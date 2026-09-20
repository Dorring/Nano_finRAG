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

#: The audit records a verdict letter; the manifest and every later summary use the name.
#: Mapping here rather than renaming in one place keeps a single spelling of each class
#: across the two artifacts, which is the whole point of keeping them apart.
VERDICT_NAMES = {"B": "LEGACY_FALSE_POSITIVE",
                 "C": "VALID_BUT_OUT_OF_SCOPE_GEOMETRY",
                 "D": "SOURCE_AMBIGUOUS",
                 "A": "V2_PRODUCER_GAP"}

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
                # The A3 kind, which is the one the admission decision actually routes on.
                # `legacy_kind` is kept beside it so an addition can be read against the
                # kind the old rule would have seen rather than only the new one.
                "new_kind": entry.get("new_kind"),
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
        VERDICT_NAMES.get(k.split(": ", 1)[1], k.split(": ", 1)[1]): v
        for k, v in tally.items() if k.startswith("removed")}
    unexplained_removed = tally.get("removed: UNCLASSIFIED_REMOVED", 0)
    if unexplained_removed:
        report["failures"].append(f"UNCLASSIFIED_REMOVED = {unexplained_removed}")
    print()

    print("=== added, by provenance ===")
    by = collections.defaultdict(collections.Counter)
    identity = collections.Counter()
    unclassified_added = 0
    for cell_id in added:
        prov = added_provenance.get(cell_id)
        if prov is None:
            unclassified_added += 1
            continue
        for field in ("method", "status", "granularity", "legacy_kind", "new_kind",
                      "table_role"):
            by[field][str(prov.get(field))] += 1
        # The joint pair, because the two marginal distributions cannot be added back
        # together: `3368 PARTIAL` and `3368 YEAR` are consistent with every PARTIAL
        # being a YEAR and with none of them being one, and the claim under test is
        # about the pair.
        identity[f"{prov.get('status')} / {prov.get('granularity')}"] += 1
    for field in ("method", "status", "granularity", "legacy_kind", "new_kind",
                  "table_role"):
        print(f"    by {field}:")
        for value, count in by[field].most_common(6):
            print(f"        {count:>6}  {value}")
    print("    by (status / granularity):")
    for value, count in identity.most_common(8):
        print(f"        {count:>6}  {value}")
    print(f"    UNCLASSIFIED_ADDED  {unclassified_added}")
    if unclassified_added:
        report["failures"].append(f"UNCLASSIFIED_ADDED = {unclassified_added}")
    report["added_by"] = {f: dict(c) for f, c in by.items()}
    report["added_by_identity"] = dict(identity)
    report["unclassified_added"] = unclassified_added
    report["unclassified_removed"] = unexplained_removed
    print()

    # --- unchanged has to mean unchanged, not merely "still there" --------------------
    #
    # A cell present in both stores under the same key proves nothing about its contents.
    # The B2 bridge added three fields to the record; if it also moved a fourth -- and
    # `period_end` is the one that would matter, because that is what the resolver reads
    # for its period match -- the delta accounting above would report the cell as
    # UNCHANGED and never look inside it.  So the intersection is compared field by field.
    print("=== unchanged, compared field by field ===")
    drifted = collections.Counter()
    drift_examples: dict[str, list[str]] = collections.defaultdict(list)
    for cell_id in unchanged:
        before, after = old[cell_id], new[cell_id]
        for field in sorted(set(before) | set(after)):
            if before.get(field) == after.get(field):
                continue
            drifted[field] += 1
            if len(drift_examples[field]) < 3:
                drift_examples[field].append(
                    f"{cell_id}: {before.get(field)!r} -> {after.get(field)!r}")
    if not drifted:
        print("    every field identical on all "
              f"{len(unchanged)} cells that survived the rebuild")
    for field, count in drifted.most_common():
        print(f"    {count:>6}  {field} changed")
        for example in drift_examples[field]:
            print(f"            {example}")
    report["unchanged_field_drift"] = {f: {"count": c, "examples": drift_examples[f]}
                                       for f, c in drifted.items()}
    # The three fields B2 introduced are the expected difference and are not drift.
    b2_fields = {"normalized_period", "period_binding_status", "period_granularity"}
    # `period` is the fourth, and it is not free: B2 makes the emitted fact take its
    # period from the binding rather than from the legacy axis, so `period` can move on a
    # cell whose identity did not.  The invariant that makes that safe is that `period`
    # is a *rendering* while `period_end` is the cell's own physical period and the field
    # the resolver matches on -- so a rendering may change only where the thing rendered
    # did not.  A `period_end` that moves on an otherwise-unchanged cell is a silent
    # mutation of the resolver's input, and blocks.
    period_moved = {c for c in unchanged
                    if old[c].get("period") != new[c].get("period")}
    period_end_moved = {c for c in period_moved
                        if old[c].get("period_end") != new[c].get("period_end")}
    if period_moved:
        print(f"    {len(period_moved):>6}  period changed (a rendering; see "
              f"audit_period_field_change.py for the adjudication)")
    if period_end_moved:
        print(f"    {len(period_end_moved):>6}  period_end changed ON THOSE SAME CELLS")
    report["period_rendering_moved"] = len(period_moved)
    report["period_end_moved_on_those_cells"] = len(period_end_moved)
    unexpected = {f: c for f, c in drifted.items()
                  if f not in b2_fields | {"period"}}
    if unexpected:
        report["failures"].append(f"UNCHANGED_FIELD_DRIFT = {unexpected}")
    if period_end_moved:
        report["failures"].append(
            f"PERIOD_END_MOVED_ON_UNCHANGED_CELLS = {len(period_end_moved)}")
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
