"""P1.6-A2B-20: does `table_role` actually buy anything at the resolver?

One variable.  The store gains `table_role` and `table_eligibility`; the resolver
gains a switch for which field decides that a table may speak for the company.
Everything else -- entity, metric and period matching, the company-level column
rule, the refusals, the fixture, the retrieval, the Binder, production -- is
untouched, and the old behaviour is reproduced from the same store by asking for
the old rule.  Any difference between the two runs is attributable to this one
choice.

The outcome is not the count of resolved slots.  It is whether the 47 slots can be
re-decomposed **safely**:

    wrong-scope resolution          0
    wrong-metric resolution         0
    resolved value mismatch         0
    dangerous NON_PRIMARY used      0

Two transitions are pre-registered as improvements rather than regressions, so
that they cannot be explained away after the fact:

  * JPMorganChase's `Net income` slots move AMBIGUOUS -> RESOLVED, because the
    candidates were competing statements that `statement_type` could not order;
  * Coca-Cola's `Long-term debt` moves AMBIGUOUS -> NO_COMPANY_LEVEL_FACT, because
    the old ambiguity was a hedging note wrongly carrying `BALANCE_SHEET`.  Losing
    a resolution that came from a table without authority is the fix working.

  python evaluate_table_role_authority.py --store <dir>/store-v2.jsonl \
      --roles <dir>/table-roles.json --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

RESOLVER = _BACKEND_DIR / "scripts/evaluation/resolve_store_v2_slot.py"
CROSS = _BACKEND_DIR / "scripts/evaluation/resolve_cross_entity_v2.py"

RESOLVED = "RESOLVED_COMPANY_LEVEL"


def _resolver():
    spec = importlib.util.spec_from_file_location("resolver", RESOLVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_of(record: dict, metric: str) -> bool:
    """The resolver's own metric test, restated so a gate can use it."""
    import re

    caption = re.compile(r"\s*\((?:in|except|dollars in|amounts in)[^)]*\)\s*$", re.I)
    folded = caption.sub("", " ".join(str(record.get("row_label") or "").split()).casefold())
    return folded.strip() == " ".join(str(metric or "").split()).casefold()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    resolver = _resolver()

    cross_spec = importlib.util.spec_from_file_location("cross", CROSS)
    cross = importlib.util.module_from_spec(cross_spec)
    cross_spec.loader.exec_module(cross)

    records = [json.loads(line)
               for line in args.store.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    roles = json.loads(args.roles.read_text(encoding="utf-8"))
    # Keyed by cell, not by table: a table holds many rows and looking the resolving
    # row up by its table would compare the metric against whichever row happened to
    # come first, which reads as 33 wrong-metric resolutions that are not there.
    by_cell = {r["cell_id"]: r for r in records if r.get("cell_id")}

    print(f"=== table_role authority experiment ({len(records)} records) ===")
    print()

    results = {}
    for authority in (resolver.AUTHORITY_STATEMENT_TYPE, resolver.AUTHORITY_TABLE_ROLE):
        run = {}
        for case_id, (metric, entities) in sorted(cross.CASES.items()):
            for entity in entities:
                period = cross.PERIOD.get(entity, "FY2025")
                out = resolver.resolve(records, entity, metric, period, authority)
                run[(case_id, entity)] = {"metric": metric, "period": period, **out}
        results[authority] = run

    old = results[resolver.AUTHORITY_STATEMENT_TYPE]
    new = results[resolver.AUTHORITY_TABLE_ROLE]

    # --- the gates --------------------------------------------------------------
    gate = collections.Counter()
    gate_detail = collections.defaultdict(list)
    for (case_id, entity), slot in new.items():
        if slot["status"] != RESOLVED:
            continue
        record = by_cell.get(slot.get("cell_id"))
        role = roles.get(str(slot.get("table_fragment_id")), {})
        if str(slot.get("table_role")) != "PRIMARY_FINANCIAL_STATEMENT":
            gate["wrong_scope"] += 1
            gate_detail["wrong_scope"].append(f"{case_id}/{entity}")
        if not case_of(record or {}, slot["metric"]):
            gate["wrong_metric"] += 1
            gate_detail["wrong_metric"].append(f"{case_id}/{entity}")
        expected = cross.EXPECTED.get((case_id, entity))
        if expected is not None and str(slot.get("value_raw")) != expected:
            gate["value_mismatch"] += 1
            gate_detail["value_mismatch"].append(
                f"{case_id}/{entity}: {slot.get('value_raw')} != {expected}")
        if role.get("oracle_role") == "DANGEROUS_NEGATIVE":
            gate["dangerous_authority"] += 1
            gate_detail["dangerous_authority"].append(
                f"{case_id}/{entity} from {slot.get('table_fragment_id')}")
        if role.get("table_eligibility") == "LAYOUT_SCAFFOLD":
            gate["scaffold_authority"] += 1
            gate_detail["scaffold_authority"].append(f"{case_id}/{entity}")

    # The same checks under the old rule, so "the new one is clean" is a comparison
    # rather than an assertion.
    old_gate = collections.Counter()
    old_detail = collections.defaultdict(list)
    for (case_id, entity), slot in old.items():
        if slot["status"] != RESOLVED:
            continue
        role = roles.get(str(slot.get("table_fragment_id")), {})
        if str(slot.get("statement_type")) not in resolver._PRIMARY_STATEMENTS:
            old_gate["wrong_scope"] += 1
        if role.get("oracle_role") == "DANGEROUS_NEGATIVE":
            old_gate["dangerous_authority"] += 1
            old_detail["dangerous_authority"].append(
                f"{case_id}/{entity} from {slot.get('table_fragment_id')} "
                f"({slot.get('statement_type')})")

    # --- the transition matrix ---------------------------------------------------
    transitions = collections.Counter()
    rows = []
    for key in sorted(old):
        case_id, entity = key
        before, after = old[key], new[key]
        label, reason = _label(before, after, roles)
        transitions[label] += 1
        transitions[f"{before['status']} -> {after['status']}"] += 1
        rows.append({
            "case_id": case_id, "entity": entity, "metric": before["metric"],
            "old_status": before["status"], "new_status": after["status"],
            "old_value": before.get("value_raw"), "new_value": after.get("value_raw"),
            "old_table": before.get("table_fragment_id"),
            "new_table": after.get("table_fragment_id"),
            "new_statement_type": after.get("statement_type"),
            "label": label, "reason": reason,
        })

    print("=== transitions ===")
    for row in rows:
        if row["label"] in ("UNCHANGED",):
            continue
        print(f"  {row['case_id']:14} {row['entity'][:20]:20} {row['metric'][:24]:24}")
        print(f"      {row['old_status']:24} -> {row['new_status']:24}  {row['label']}")
        print(f"      old {str(row['old_value']):>10}   new {str(row['new_value']):>10}   "
              f"{row['reason']}")
    print()
    print("=== labels ===")
    for name, count in transitions.most_common(12):
        print(f"  {count:>3}  {name}")

    report = {
        "phase": "P1.6-A2B-20", "mutation": "authority: statement_type -> table_role",
        "records": len(records),
        "gate": dict(gate), "gate_detail": {k: v for k, v in gate_detail.items()},
        "baseline_gate": dict(old_gate),
        "baseline_gate_detail": {k: v for k, v in old_detail.items()},
        "transitions": dict(transitions), "slots": rows,
        "old_statuses": dict(collections.Counter(s["status"] for s in old.values())),
        "new_statuses": dict(collections.Counter(s["status"] for s in new.values())),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "authority-experiment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print()
    print("=== status distribution ===")
    print(f"  old (statement_type) {dict(collections.Counter(s['status'] for s in old.values()))}")
    print(f"  new (table_role)     {dict(collections.Counter(s['status'] for s in new.values()))}")
    print()
    print("=== the gate ===")
    for name, limit in (("wrong_scope", 0), ("wrong_metric", 0),
                        ("value_mismatch", 0), ("dangerous_authority", 0),
                        ("scaffold_authority", 0)):
        value = gate.get(name, 0)
        print(f"  {name:22} {value}  {'PASS' if value <= limit else 'FAIL'}")
        for detail in gate_detail.get(name, [])[:5]:
            print(f"      {detail}")
    print()
    print("=== the same checks under the old rule ===")
    for name in ("wrong_scope", "dangerous_authority"):
        print(f"  {name:22} {old_gate.get(name, 0)}")
        for detail in old_detail.get(name, [])[:5]:
            print(f"      {detail}")
    print()
    print(f"  written to {args.out / 'authority-experiment.json'}")
    return 0


def _label(before: dict, after: dict, roles: dict) -> tuple[str, str]:
    """Name the transition, and say why it moved."""

    was, now = before["status"], after["status"]
    if was == now:
        return "UNCHANGED", ""

    if now == RESOLVED:
        return "CAPABILITY_GAIN", (
            f"resolved from a table with authority; was {was.lower().replace('_', ' ')}"
        )

    if was == RESOLVED:
        role = roles.get(str(before.get("table_fragment_id")), {})
        if role.get("oracle_role") == "PRIMARY":
            return "REGRESSION", (
                f"lost a resolution that came from an oracle-verified primary statement "
                f"({before.get('value_raw')})"
            )
        return "SAFER_FAIL_CLOSED", (
            f"the old answer came from a table without authority "
            f"(statement_type={before.get('statement_type')}, "
            f"oracle_role={role.get('oracle_role')}); refusing it is correct"
        )

    return "REFUSAL_RETYPED", (
        f"both are refusals; the candidate set changed rather than the answer"
    )


if __name__ == "__main__":
    raise SystemExit(main())
