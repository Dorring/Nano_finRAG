"""P1.6-A3-W4-B: the pre-switch accounting, frozen before anything is switched.

The numbers below are read out of the W4-A7/W4-A8 artifacts, not restated, and the gates are
asserted rather than reported -- a manifest that agrees with itself is worth nothing.

What this freezes is the *prediction*: which cells the migration is expected to remove and
under which of three headings.  The rebuild then has to reproduce it, and a mismatch in
either direction is a finding:

    shadow predicts a removal but the store still holds it   -> persistence / schema
    the store drops more than the shadow predicted           -> migration wiring

The three headings are kept apart on purpose and must stay apart in every later summary.
`legacy false positives removed` and `source-ambiguous facts withheld` are different claims,
and neither is `corrected`.

  python freeze_migration_manifest.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

A8 = Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a3-w4a8-changes/"
          "binding-changes.json")
A7_AUDIT = Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a3-w4a7-audit/"
                "valid-loss-audit.json")
A7_SHADOW = Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-6-a3-w4a7-shadow/"
                 "emission-admission-shadow.json")

#: The gates W4-B was allowed to start on, each with where its number comes from.
GATES = (
    ("CORRECT_SOURCE_GROUNDED_BINDING_LOST", "binding-changes.json", "unbound"),
    ("NEW_UNSUPPORTED_BINDING", "binding-changes.json", "gained"),
    ("ROW_PRODUCER_GAP", "binding-changes.json", "unbound"),
    ("V2_PRODUCER_GAP", "valid-loss-audit.json", "verdicts"),
)

#: The three classes the migration is expected to remove, kept apart.
REMOVAL_CLASSES = (
    ("LEGACY_FALSE_POSITIVE", "LEGACY_FALSE_POSITIVE"),
    ("SOURCE_AMBIGUOUS", "SOURCE_AMBIGUOUS"),
    ("VALID_BUT_OUT_OF_SCOPE_GEOMETRY", "VALID_BUT_OUT_OF_SCOPE_GEOMETRY"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    changes = json.loads(A8.read_text(encoding="utf-8"))
    audit = json.loads(A7_AUDIT.read_text(encoding="utf-8"))
    shadow = json.loads(A7_SHADOW.read_text(encoding="utf-8"))
    totals = changes["totals"]

    manifest = {"phase": "P1.6-A3-W4-B-manifest", "mutation": "none",
                "gates": {}, "removals": {}, "before": {}, "failures": []}

    print("=== the gates W4-B was allowed to start on ===")
    for name, source, bucket in GATES:
        if source == "binding-changes.json":
            value = totals.get(f"unbound: {name}", 0) if name != "NEW_UNSUPPORTED_BINDING" \
                else totals.get("gained: SOURCE CELLS OUTSIDE THE COLUMN", 0)
        else:
            value = audit["verdicts"].get(name, 0)
        manifest["gates"][name] = value
        ok = value == 0
        print(f"    {name:<38} {value:>5}  {'ok' if ok else 'NOT MET'}")
        if not ok:
            manifest["failures"].append(f"{name} = {value}")

    print()
    print("=== the three classes the migration is expected to remove ===")
    for label, key in REMOVAL_CLASSES:
        value = audit["verdicts"].get(key, 0)
        manifest["removals"][label] = value
        print(f"    {label:<38} {value:>5}")
    unexplained = audit["verdicts"].get("V2_PRODUCER_GAP", 0)
    manifest["removals"]["V2_PRODUCER_GAP"] = unexplained
    print(f"    {'V2_PRODUCER_GAP':<38} {unexplained:>5}  "
          f"{'(must stay 0 after the switch)' if unexplained == 0 else 'NOT MET'}")
    print(f"    {'total':<38} {sum(manifest['removals'].values()):>5}")

    print()
    print("=== the binder before the switch ===")
    for key in ("unchanged_admitted", "producer_loss", "net_loss", "net_gain",
                "rule_gain", "unchanged_withheld", "rule_gain_then_producer_loss"):
        manifest["before"][key] = shadow["totals"].get(key, 0)
        print(f"    {key:<38} {shadow['totals'].get(key, 0):>7}")

    manifest["binding_changes"] = {
        "supported": totals.get("gained: supported", 0),
        "unsupported": totals.get("gained: SOURCE CELLS OUTSIDE THE COLUMN", 0),
        "not_a_period_declaration": totals.get("unbound: NOT_A_PERIOD_DECLARATION", 0),
        "rehomed_to_row_binding": totals.get("unbound: REHOMED_TO_ROW_BINDING", 0),
        "changed": totals.get("changed", 0),
    }
    manifest["note"] = (
        "The three removal classes are not corrections and must never be summed into one "
        "number. LEGACY_FALSE_POSITIVE is a legacy binding the source does not support; "
        "SOURCE_AMBIGUOUS is a fact whose period cannot be proved either way; "
        "VALID_BUT_OUT_OF_SCOPE_GEOMETRY is a correct period the column model cannot "
        "express. Only the first is a correction.")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "migration-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print()
    print(f"  failures: {manifest['failures'] or 'none'}")
    print(f"  written to {args.out / 'migration-manifest.json'}")
    return 1 if manifest["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
