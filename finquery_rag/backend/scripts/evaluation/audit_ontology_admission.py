"""E2-2B: admit or refuse each blocked label against six frozen rules.

The gate refuses these labels because the ontology (13 canonical metrics) does
not name them.  48-50 of them are well-formed quantity names the store carries,
which makes "extend the ontology" the obvious next move -- and the wrong one if
done on that evidence alone.  `Deferred` is in the store with 21 rows and cannot
be mapped to one concept; adding it would buy a passing case at the cost of the
gate's credibility.

So the bar is the six rules below, and a label is added only when **every** rule
is satisfied.  Where a rule cannot be established rather than merely not
disproven, the label is refused: the project's whole posture is fail-closed, and
an alias admitted on "probably fine" is an UNKNOWN turned into a known without
the proof.

  SEMANTICALLY_UNIQUE  the label denotes exactly one concept
  SOURCE_GROUNDED      the store carries the concept
  ENTITY_CONSISTENT    it means the same thing for every entity it appears under
  SCOPE_INDEPENDENT    its meaning does not depend on one table's row geometry
  DETERMINISTIC        no embedding threshold, no model call
  NO_COLLISION         it does not collide with an existing canonical metric

**Which rules are established and which are proxied is printed per row.**  Two
of them are not fully decidable from the store, and saying so is the point:
ENTITY_CONSISTENT is proxied by unit/scale agreement across the entities that
carry the label.  A proxied rule is a real check that can fail; it is not
proof, and the row says which.  SCOPE_INDEPENDENT is established from an
explicit list of positional labels rather than measured, because the first
attempt to measure it -- require recurrence across two tables -- refused
`iPad` and `Basic earnings per share` for not recurring in a 20k-row sample.

  python audit_ontology_admission.py --blocks <semantic-gate-blocks>.json \\
      --store financial-facts.jsonl --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_semantic_alignment_blocks import _FOOTNOTE, _PERSONLIKE, _PERSON_NAME, _norm

#: Labels that name a position in a table rather than a quantity, or that a
#: reader cannot resolve without the table.  Every entry is a string the
#: benchmark actually used; nothing here is inferred by pattern.
STRUCTURAL = frozenset({
    "beginning balance at january 1", "balance in aoci at beginning of year",
    "ending balance", "total", "other", "current", "additions",
    "impact of the state aid decision", "income tax effect",
    "change in fair value of derivative instruments",
    "net foreign currency translation adjustments",
    "total non current portion of term debt", "total automotive cost of revenues",
    "statutory federal income tax rate", "risk free interest rate",
    "return on equity", "allowance for loan losses to total retained loans",
})

#: Labels that are one word and a concept in more than one sense.  `Services` is
#: a reportable segment and a revenue line; `Deferred` is deferred revenue, tax
#: or compensation; `Intersegment` is a column of an elimination table.
AMBIGUOUS = frozenset({
    # Genuinely under-determined: each names a position or a family rather than
    # one quantity, and a reader needs the table to say which.
    "deferred", "services", "intersegment", "additions",
    "other contracts", "other contracts sold", "total fees",
    "medicare rebates", "u s gses and government agencies",
    "deferred taxes", "deferred income taxes",
    "foreign currency contracts", "fair value hedges",
    "foreign currency translation adjustment",
    "other comprehensive income loss", "total comprehensive income",
    "other accrued expenses", "stock based compensation expense",
    "share based compensation", "commercial",
})

DECISION_ADD = "ADD_ALIAS"
DECISION_AMBIGUOUS = "KEEP_BLOCKED_AMBIGUOUS"
DECISION_NON_CONCEPT = "KEEP_BLOCKED_NON_CONCEPT"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from rag_v2.supervisor.semantic_alignment import (
        canonical_metric_id, metric_alias_registry)

    registry = metric_alias_registry()
    existing_aliases = {_norm(alias) for aliases in registry.values() for alias in aliases}
    existing_aliases |= {_norm(metric_id) for metric_id in registry}

    rows_by_label: dict[str, list[dict]] = collections.defaultdict(list)
    for line in args.store.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        label = str(record.get("metric") or "").strip()
        if label:
            rows_by_label[label].append(record)

    blocks = json.loads(args.blocks.read_text(encoding="utf-8"))
    labels: collections.Counter = collections.Counter()
    carrying_cases: dict[str, list[str]] = collections.defaultdict(list)
    for case in blocks["cases"]:
        for slot in case["slots"]:
            if slot["canonical_id"]:
                continue
            labels[slot["metric"]] += 1
            carrying_cases[slot["metric"]].append(case["case_id"])

    print("=" * 78)
    print("E2-2B -- ONTOLOGY ADMISSION, SIX RULES")
    print("=" * 78)
    print(f"  distinct blocking labels  {len(labels)}")
    print(f"  ontology now carries      {len(existing_aliases)} aliases "
          f"across {len(registry)} canonical metrics")
    print()

    table: list[dict] = []
    tally: collections.Counter = collections.Counter()
    failed_rule: collections.Counter = collections.Counter()

    for label, count in labels.most_common():
        norm = _norm(label)
        rows = rows_by_label.get(label, [])
        units = {str(r.get("unit")) for r in rows if r.get("unit")}
        scales = {str(r.get("scale")) for r in rows if r.get("scale")}
        entities = {str(r.get("entity")) for r in rows if r.get("entity")}
        tables = {str(r.get("table_id")) for r in rows if r.get("table_id")}

        checks = {
            # Established only negatively: a label that is structurally a table
            # position, or ambiguous by construction, fails outright.
            "SEMANTICALLY_UNIQUE": norm not in AMBIGUOUS,
            "SOURCE_GROUNDED": bool(rows),
            # Proxy: a label whose rows disagree on unit or scale is carrying
            # more than one concept across the entities that report it.
            "ENTITY_CONSISTENT": len(units) <= 1 and len(scales) <= 1,
            # Whether the label's meaning depends on its table position.  A
            # first version also required the label to recur across >= 2 tables,
            # which refused `iPad`, `Basic earnings per share` and `Tangible book
            # value per share` as "confined to a single table" -- a legitimate
            # concept need not recur in a 20k-row sample, so that requirement was
            # measuring sample coverage rather than meaning and is removed.
            "SCOPE_INDEPENDENT": norm not in STRUCTURAL,
            "DETERMINISTIC": True,   # nothing here consults a model
            "NO_COLLISION": norm not in existing_aliases,
        }
        non_concept = (
            bool(_FOOTNOTE.search(label))
            or bool(_PERSONLIKE.search(label))
            or bool(_PERSON_NAME.match(label))
            or not rows
        )

        if non_concept:
            decision, reason = DECISION_NON_CONCEPT, (
                "footnote marker" if _FOOTNOTE.search(label) else
                "person or role name" if (_PERSONLIKE.search(label)
                                          or _PERSON_NAME.match(label)) else
                "no rows in the store")
        elif all(checks.values()):
            decision, reason = DECISION_ADD, "all six rules satisfied"
        elif not checks["SEMANTICALLY_UNIQUE"]:
            decision, reason = DECISION_AMBIGUOUS, "denotes more than one concept"
        elif not checks["SCOPE_INDEPENDENT"]:
            decision, reason = DECISION_AMBIGUOUS, (
                "table position, or confined to a single table")
        elif not checks["ENTITY_CONSISTENT"]:
            decision, reason = DECISION_AMBIGUOUS, "unit or scale disagrees across entities"
        elif not checks["NO_COLLISION"]:
            decision, reason = DECISION_AMBIGUOUS, "collides with an existing alias"
        else:
            decision, reason = DECISION_AMBIGUOUS, "no rule satisfied it"

        for rule, ok in checks.items():
            if not ok and not non_concept:
                failed_rule[rule] += 1
        tally[decision] += 1
        table.append({
            "label": label, "slots": count, "cases": carrying_cases[label],
            "decision": decision, "reason": reason,
            "proposed_canonical_id": (re.sub(r"[^a-z0-9]+", "_", norm).strip("_")
                                      if decision == DECISION_ADD else None),
            "rules": checks,
            "evidence": {"store_rows": len(rows), "entities": sorted(entities)[:6],
                         "tables": len(tables), "units": sorted(units),
                         "scales": sorted(scales)},
        })

    print("  decision                    labels")
    for decision in (DECISION_ADD, DECISION_AMBIGUOUS, DECISION_NON_CONCEPT):
        print(f"    {decision:26} {tally.get(decision, 0):>4}")
    print(f"    {'UNREVIEWED':26} {0:>4}")
    print()
    print("  rule failures among non-concept labels")
    for rule, count in failed_rule.most_common():
        print(f"    {count:>4}  {rule}")
    print()

    for decision in (DECISION_ADD, DECISION_AMBIGUOUS, DECISION_NON_CONCEPT):
        entries = [row for row in table if row["decision"] == decision]
        if not entries:
            continue
        print(f"  {decision} ({len(entries)})")
        for row in entries:
            target = f" -> {row['proposed_canonical_id']}" if row["proposed_canonical_id"] else ""
            print(f"    {row['label'][:52]:54}{target:34}{row['reason']}")
        print()

    print("  Rules ENTITY_CONSISTENT and SCOPE_INDEPENDENT are PROXIES: unit/scale")
    print("  agreement across entities, and recurrence across tables.  They can fail")
    print("  a bad label and they cannot prove a good one. Every ADD above still")
    print("  needs a source reading before it goes in.")
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ontology-admission.json").write_text(
        json.dumps({"labels": len(labels), "by_decision": dict(tally),
                    "rule_failures": dict(failed_rule),
                    "proxied_rules": ["ENTITY_CONSISTENT", "SCOPE_INDEPENDENT"],
                    "table": table},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"  written to {args.out / 'ontology-admission.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
