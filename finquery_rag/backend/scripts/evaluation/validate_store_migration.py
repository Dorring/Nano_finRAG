"""P1.6-A3-W4-B3: validate the migration at the system level, not the count level.

W4-B2 proved the bridge persists the gains losslessly.  This asks the different question:
did moving the store from the legacy admission to the A3 admission change any *answer*,
and if so is every change one the evidence supports?

Four checks, in the order they can fail:

  1. fact-level delta      ADDED / REMOVED / UNCHANGED, with `UNCLASSIFIED_ADDED = 0` and
                           `UNCLASSIFIED_REMOVED = 0` -- an addition or removal that falls
                           outside a known class blocks closure rather than being absorbed
                           into a neighbouring one
  2. the 47 slots          compared **slot by slot** against the pre-migration store under
                           the same authority, because two flips in opposite directions
                           leave the same distribution and hide each other
  3. the safety gates      wrong_scope / wrong_metric / value_mismatch /
                           dangerous_authority / scaffold_authority
  4. the legacy resolver   against the new YEAR facts.  The resolver is deliberately
                           untouched, so a YEAR fact that contributes nothing today is
                           registered debt and not a failure -- but a YEAR fact that has
                           been given a day, in the field the resolver reads, is a
                           fabricated date reaching production and blocks.

The additions are reconciled by the identity the shadow decided them on, not by a count:
`PARTIAL / YEAR` and `RESOLVED / DAY` are checked as a **joint** pair, because the two
marginal distributions cannot be added back together.

  python validate_store_migration.py --old <store> --new <store> --baseline-slots <json> \
      --new-slots <json> --accounting <json> --shadow <json> --out <dir>
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

EVAL = _BACKEND_DIR / "scripts/evaluation"
RESOLVER = EVAL / "resolve_store_v2_slot.py"
CROSS = EVAL / "resolve_cross_entity_v2.py"

RESOLVED = "RESOLVED_COMPANY_LEVEL"

#: The three removal classes, with the count the manifest froze before the migration ran.
#: Expected values rather than read from the accounting, so that the accounting reporting
#: a different number is a failure rather than a new baseline.
EXPECTED_REMOVALS = {"LEGACY_FALSE_POSITIVE": 691,
                     "SOURCE_AMBIGUOUS": 4188,
                     "VALID_BUT_OUT_OF_SCOPE_GEOMETRY": 2173}

#: And the additions, by the joint identity the shadow decided them on.
EXPECTED_ADDED = {("PARTIAL", "YEAR"): 3368, ("RESOLVED", "DAY"): 3018}

GATES = ("wrong_scope", "wrong_metric", "value_mismatch", "dangerous_authority",
         "scaffold_authority")

_YEAR_ONLY = re.compile(r"^\d{4}$")
_FULL_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _module(path: Path, name: str):
    """Load one of the eval scripts by path, the way the evaluator does."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_store(path: Path) -> list[dict]:
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--baseline-slots", type=Path, required=True)
    parser.add_argument("--new-slots", type=Path, required=True)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    failures: list[str] = []
    report: dict = {"phase": "P1.6-A3-W4-B3"}

    old_records = read_store(args.old)
    new_records = read_store(args.new)
    old = {str(r["cell_id"]): r for r in old_records}
    new = {str(r["cell_id"]): r for r in new_records}
    accounting = json.loads(args.accounting.read_text(encoding="utf-8"))

    added = set(new) - set(old)
    removed = set(old) - set(new)
    unchanged = set(old) & set(new)

    # --- 1. the delta, and whether every part of it is classified ----------------------
    print("=== 1. fact-level delta ===")
    print(f"    before {len(old)}   after {len(new)}   "
          f"added {len(added)}   removed {len(removed)}   unchanged {len(unchanged)}")
    report["delta"] = {"before": len(old), "after": len(new), "added": len(added),
                       "removed": len(removed), "unchanged": len(unchanged)}

    unclassified_added = accounting.get("unclassified_added")
    unclassified_removed = accounting.get("unclassified_removed")
    print(f"    UNCLASSIFIED_ADDED    {unclassified_added}")
    print(f"    UNCLASSIFIED_REMOVED  {unclassified_removed}")
    if unclassified_added or unclassified_removed:
        failures.append(f"unclassified delta: added={unclassified_added} "
                        f"removed={unclassified_removed}")
    report["unclassified"] = {"added": unclassified_added, "removed": unclassified_removed}

    print("    removals, kept apart:")
    for name, expected in EXPECTED_REMOVALS.items():
        got = accounting["removed_by_class"].get(name, 0)
        mark = "ok " if got == expected else "MOVED"
        print(f"        {mark} {name:<34} {got:>5}  (pre-registered {expected})")
        if got != expected:
            failures.append(f"{name} = {got}, pre-registered {expected}")
    report["removals"] = accounting["removed_by_class"]

    print("    additions, by the identity the shadow decided on:")
    for (status, granularity), expected in sorted(EXPECTED_ADDED.items()):
        got = accounting["added_by_identity"].get(f"{status} / {granularity}", 0)
        mark = "ok " if got == expected else "MOVED"
        print(f"        {mark} {status} / {granularity:<8} {got:>5}  "
              f"(pre-registered {expected})")
        if got != expected:
            failures.append(f"added {status}/{granularity} = {got}, "
                            f"pre-registered {expected}")
    report["additions_by_identity"] = accounting["added_by_identity"]

    # Every addition has to name a shadow decision.  `UNCLASSIFIED_ADDED = 0` already
    # says so; this says it again against the shadow itself, because the accounting and
    # the shadow are produced by different runs and agreement between them is the point.
    shadow = json.loads(args.shadow.read_text(encoding="utf-8"))
    decided = {str(e["cell_id"]) for name in ("rule_gain", "producer_gain")
               for e in shadow.get(name, []) if e["v2_outcome"] != "WITHHOLD"}
    orphans = sorted(added - decided)
    print(f"    additions with no shadow decision: {len(orphans)}")
    if orphans:
        failures.append(f"UNEXPLAINED_ADDITION = {len(orphans)}")
        for cell in orphans[:5]:
            print(f"        {cell} {new[cell].get('row_label')!r}")
    report["unexplained_additions"] = len(orphans)
    print()

    # Every addition must take its period from its **own column**.  This is the property
    # that separates a fact being admitted on evidence about itself from a fact being
    # admitted on evidence about its neighbour, and it is the one direction the store
    # cannot show after the fact -- the store keeps the identity, not the binding that
    # produced it, which is W5's job.  So it is checked from the shadow, where the binding
    # is still recorded, and a row-scoped declaration is allowed only because
    # `INLINE_PERIOD_DATA_ROW` says so in its own name.
    provenance = collections.Counter()
    stray = []
    for name in ("rule_gain", "producer_gain"):
        for entry in shadow.get(name, []):
            if entry["v2_outcome"] == "WITHHOLD":
                continue
            cells = entry.get("source_cells") or []
            if not cells:
                provenance["admitted with no source cell at all"] += 1
                stray.append(entry)
            elif {c["column"] for c in cells} == {entry["column_index"]}:
                provenance["every source cell in the fact's own column"] += 1
            elif entry.get("binding_method") == "INLINE_PERIOD_DATA_ROW":
                provenance["row-scoped declaration (INLINE_PERIOD_DATA_ROW)"] += 1
            else:
                provenance["source cells OUTSIDE the fact's own column"] += 1
                stray.append(entry)
    print("    where each addition's period came from:")
    for name, count in provenance.most_common():
        print(f"        {count:>6}  {name}")
    if stray:
        failures.append(f"ADDITION_BOUND_OUTSIDE_ITS_COLUMN = {len(stray)}")
        for entry in stray[:5]:
            print(f"        {entry['document_id']} col={entry['column_index']} "
                  f"method={entry['binding_method']} "
                  f"src={[(c['row'], c['column']) for c in entry.get('source_cells') or []]}")
    report["addition_provenance"] = dict(provenance)
    print()

    # --- 2. the 47 slots, slot by slot -------------------------------------------------
    baseline = json.loads(args.baseline_slots.read_text(encoding="utf-8"))
    migrated = json.loads(args.new_slots.read_text(encoding="utf-8"))
    base_by = {(r["case_id"], r["entity"]): r for r in baseline["slots"]}
    new_by = {(r["case_id"], r["entity"]): r for r in migrated["slots"]}

    print("=== 2. the 47 slots, before and after, under the same authority ===")
    if set(base_by) != set(new_by):
        failures.append("the two runs do not cover the same slots")
    transitions = collections.Counter()
    regressions, gains, value_moves = [], [], []
    for key in sorted(base_by):
        before, after = base_by[key], new_by[key]
        transitions[f"{before['new_status']} -> {after['new_status']}"] += 1
        if before["new_status"] == RESOLVED and after["new_status"] != RESOLVED:
            regressions.append((key, before, after))
        elif before["new_status"] != RESOLVED and after["new_status"] == RESOLVED:
            gains.append((key, before, after))
        elif before["new_status"] == RESOLVED == after["new_status"]:
            if str(before.get("new_value")) != str(after.get("new_value")):
                value_moves.append((key, before, after))

    for name, count in transitions.most_common():
        print(f"    {count:>3}  {name}")
    print()
    print(f"    existing resolved regression  {len(regressions)}")
    for key, before, after in regressions:
        print(f"        {key}  {before['new_value']} -> {after['new_status']}")
    if regressions:
        failures.append(f"existing resolved regression = {len(regressions)}")
    print(f"    resolved value changed         {len(value_moves)}")
    for key, before, after in value_moves:
        print(f"        {key}  {before['new_value']} -> {after['new_value']}")
    if value_moves:
        failures.append(f"resolved value changed = {len(value_moves)}")
    print(f"    newly resolved                 {len(gains)}")
    report["slots"] = {"transitions": dict(transitions),
                       "regressions": [list(k) for k, _, _ in regressions],
                       "value_moves": [list(k) for k, _, _ in value_moves],
                       "new_gains": [list(k) for k, _, _ in gains]}

    # Each new resolution has to have come from a fact the migration *added*, from a
    # table with authority, by a binding that names its source cells.  A slot that
    # resolves against something already there would mean the migration changed an
    # answer without changing the evidence, which is drift rather than a gain.
    if gains:
        resolver = _module(RESOLVER, "resolver")
        cross = _module(CROSS, "cross")
        print()
        print("    each newly resolved slot, and where it came from:")
        unverified = []
        for key, _before, after in gains:
            case_id, entity = key
            metric = after["metric"]
            period = cross.PERIOD.get(entity, "FY2025")
            out = resolver.resolve(new_records, entity, metric, period,
                                   resolver.AUTHORITY_TABLE_ROLE)
            cell = str(out.get("cell_id") or "")
            record = new.get(cell, {})
            verdict = []
            if cell not in added:
                verdict.append("NOT A NEWLY ADMITTED FACT")
            if str(out.get("table_role")) != "PRIMARY_FINANCIAL_STATEMENT":
                verdict.append(f"role {out.get('table_role')}")
            status = str(record.get("period_binding_status") or "")
            granularity = str(record.get("period_granularity") or "")
            if status not in ("RESOLVED", "PARTIAL"):
                verdict.append(f"binding status {status!r}")
            if granularity == "UNKNOWN" or not granularity:
                verdict.append(f"granularity {granularity!r}")
            mark = "ok " if not verdict else "?? "
            print(f"        {mark} {case_id:12} {entity[:18]:18} {metric[:22]:22} "
                  f"{after['new_value']!s:>10}")
            print(f"             cell {cell}  role={out.get('table_role')}  "
                  f"{status}/{granularity}  period={record.get('normalized_period')!r}")
            if verdict:
                unverified.append((key, verdict))
                print(f"             {'; '.join(verdict)}")
        if unverified:
            failures.append(f"unverified new resolutions = {len(unverified)}")
        report["new_gains_verified"] = len(gains) - len(unverified)
    print()

    # --- 3. the safety gates ------------------------------------------------------------
    print("=== 3. the safety gates ===")
    gate = migrated.get("gate", {})
    for name in GATES:
        value = gate.get(name, 0)
        print(f"    {name:22} {value}  {'PASS' if value == 0 else 'FAIL'}")
        if value:
            failures.append(f"{name} = {value}")
    report["gate"] = {name: gate.get(name, 0) for name in GATES}
    print()

    # --- 4. the legacy resolver against the new YEAR facts ------------------------------
    #
    # The resolver is untouched, so a YEAR fact it cannot use is registered debt.  What
    # it must never do is read a day that no source stated: the resolver matches on
    # `period_end`, so a fabricated date there is a fabricated date in production, even
    # though `[:4]` would make it match the same year.  The check is therefore on what
    # the store *claims*, which is where a fabrication would have to live.
    print("=== 4. YEAR facts against the field the resolver actually reads ===")
    by_identity = collections.defaultdict(collections.Counter)
    dated_year_facts = []
    promoted = []
    for record in new_records:
        status = str(record.get("period_binding_status") or "")
        granularity = str(record.get("period_granularity") or "")
        end = str(record.get("period_end") or "")
        by_identity[(status, granularity)]["total"] += 1
        if granularity == "YEAR":
            if _FULL_DATE.match(end):
                dated_year_facts.append(record)
                by_identity[(status, granularity)]["period_end is a full date"] += 1
            elif _YEAR_ONLY.match(end):
                by_identity[(status, granularity)]["period_end is a bare year"] += 1
            elif not end:
                by_identity[(status, granularity)]["period_end is empty"] += 1
            else:
                by_identity[(status, granularity)][f"period_end {end!r}"] += 1
        if status in ("CONFLICT", "UNRESOLVED"):
            promoted.append(record)

    for key in sorted(by_identity):
        print(f"    {key[0]} / {key[1]}:")
        for name, count in by_identity[key].most_common():
            print(f"        {count:>6}  {name}")
    print()
    print(f"    YEAR facts carrying a full date in `period_end`   {len(dated_year_facts)}")
    for record in dated_year_facts[:5]:
        print(f"        {record.get('row_label')!r} period_end="
              f"{record.get('period_end')!r}")
    if dated_year_facts:
        failures.append(f"YEAR fact with a fabricated day = {len(dated_year_facts)}")
    print(f"    CONFLICT or UNRESOLVED facts in the store        {len(promoted)}")
    if promoted:
        failures.append(f"CONFLICT/UNRESOLVED promoted into the store = {len(promoted)}")
    report["year_safety"] = {
        "dated_year_facts": len(dated_year_facts),
        "promoted_conflict_or_unresolved": len(promoted),
        "by_identity": {f"{k[0]}/{k[1]}": dict(v) for k, v in by_identity.items()},
    }

    # And the same question asked of the resolver's output rather than of the store.  The
    # resolver matches `str(period_end)[:4]` against the year a question asks for.  A YEAR
    # fact with a bare-year `period_end` is therefore *matchable*, and that is correct --
    # a fact whose period is the year 2025 is a legitimate candidate for an FY2025
    # question.  What would not be correct is the two fields naming different years, so
    # the resolver would match a fact on a year its own identity does not claim.
    matchable = [r for r in new_records
                 if str(r.get("period_granularity")) == "YEAR"
                 and _YEAR_ONLY.match(str(r.get("period_end") or ""))]
    disagreeing = [r for r in matchable
                   if str(r.get("normalized_period") or "")[:4]
                   != str(r.get("period_end") or "")[:4]]
    unmatchable = [r for r in new_records
                   if str(r.get("period_granularity")) == "YEAR"
                   and not r.get("period_end")]
    print(f"    YEAR facts the resolver may match on `period_end`   {len(matchable)}")
    print(f"        of those, whose `period_end` year disagrees with "
          f"`normalized_period`   {len(disagreeing)}")
    for record in disagreeing[:5]:
        print(f"            {record.get('row_label')!r} period_end="
              f"{record.get('period_end')!r} identity="
              f"{record.get('normalized_period')!r}")
    if disagreeing:
        failures.append(f"YEAR fact matchable on a year it does not claim = "
                        f"{len(disagreeing)}")
    print(f"    YEAR facts it cannot match at all (no `period_end`)  {len(unmatchable)}")
    print(f"        these are the registered downstream capability debt: the resolver is")
    print(f"        untouched by design, so a year-only fact with no `period_end` is a")
    print(f"        fact it has nothing to match on yet, not a fact it got wrong")
    report["year_safety"]["matchable"] = len(matchable)
    report["year_safety"]["matchable_year_disagrees"] = len(disagreeing)
    report["year_safety"]["unmatchable"] = len(unmatchable)
    print()

    print(f"  failures: {failures or 'none'}")
    report["failures"] = failures
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "migration-validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'migration-validation.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
