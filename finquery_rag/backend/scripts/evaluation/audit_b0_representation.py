"""B0 -- is the representation loss real, and what would it actually buy?

A Go/No-Go gate for a fact-representation migration.  It changes nothing: no
store, no gold, no code under `src/`.  It answers four questions.

1. Of the blocked cases, how many are *really* caused by representation loss?
   Each is attributed individually.  The default is not "recoverable".
2. What is the minimal field set that would explain these failures?
3. Which of it can be rebuilt deterministically from the existing source truth?
4. What is the end-to-end ceiling if it were rebuilt -- not "how many clear the
   gate", but how many reach `released` through retrieval, binding, slot
   completeness and validation?

The classifier uses one piece of evidence rather than a judgement: every fact
carries its own row in `content`.  So at a shared `(entity, metric, period)`:

  all facts share one `content`  -> the values are COLUMNS of a single row
  the `content` strings differ   -> the values are DIFFERENT ROWS

Those need different fields and have different ceilings, and conflating them is
how a migration gets scoped at twice its real size.

  python audit_b0_representation.py --cases <cases.jsonl> --pinned <pinned.jsonl> \\
      --v2-fact-store <store.jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
import sys
import unicodedata
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _fold(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def _number(value: object) -> float | None:
    text = str(value or "").replace(",", "").replace("$", "").strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    if not text or text.endswith("%"):
        return None
    match = re.fullmatch(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    number = float(match.group(0))
    return -number if negative else number


def _has_total(values: list[float]) -> bool:
    """One value is the sum of a subset of the others."""

    for size in range(2, min(5, len(values)) + 1):
        for combination in itertools.combinations(values, size):
            rest = [v for v in values if v not in combination]
            for candidate in rest:
                if abs(sum(combination) - candidate) < 1e-6 * max(1.0, abs(candidate)):
                    return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--v2-fact-store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    records = _load(args.v2_fact_store)
    pinned = {row["case_id"]: row for row in _load(args.pinned)}

    by_coordinate: dict[tuple[str, str, str], list[dict]] = collections.defaultdict(list)
    by_logical: dict[tuple, set[str]] = collections.defaultdict(set)
    for record in records:
        key = (_fold(record.get("entity")), _fold(record.get("metric")),
               _fold(record.get("period")))
        by_coordinate[key].append(record)
        logical = key + (_fold(record.get("value")),)
        candidate = str(record.get("candidate_key") or "")
        if candidate:
            by_logical[logical].add(candidate)

    rows = [r for r in _load(args.cases)
            if r.get("comparable") and not r.get("must_refuse")]
    blocked = [r for r in rows if not r.get("released")]

    report: dict = {"cases": [], "totals": {}, "store": {}}
    for row in blocked:
        entry = pinned.get(row["case_id"]) or {}
        plan = entry.get("plan") or {}
        slots = plan.get("required_slots") or []

        stage = (
            "GATE" if row.get("gate_blocked")
            else "NOT_BOUND" if row.get("gold_in_pool") and not row.get("gold_bound")
            else "NOT_IN_POOL" if not row.get("gold_in_pool")
            else "VALIDATION" if not row.get("validation_passed")
            else "SLOTS" if row.get("missing_slot_ids")
            else "OTHER"
        )

        slot_detail = []
        for slot in slots:
            key = (_fold(slot.get("entity")), _fold(slot.get("metric")),
                   _fold(slot.get("period")) if slot.get("period") else "")
            facts = []
            for candidate_key, facts_at in by_coordinate.items():
                if candidate_key[0] != key[0] or candidate_key[1] != key[1]:
                    continue
                if key[2] and candidate_key[2] not in ("", key[2]):
                    continue
                facts.extend(facts_at)
            if not facts:
                continue
            values = [v for v in (_number(f.get("value")) for f in facts) if v is not None]
            distinct = sorted(set(values))
            contents = {_fold(f.get("content")) for f in facts}
            slot_detail.append({
                "slot_id": slot.get("slot_id"),
                "metric": slot.get("metric"),
                "entity": slot.get("entity"),
                "period": slot.get("period"),
                "facts": len(facts),
                "distinct_values": len(distinct),
                "one_row": len(contents) == 1,
                "fragments": len({f.get("table_fragment_id") for f in facts}),
                "has_total": _has_total(distinct) if len(distinct) > 1 else False,
                "values": distinct[:6],
            })

        # -- attribution -------------------------------------------------
        #
        # The test is not "is the record missing a dimension" -- every one of
        # these is.  It is "would restoring it let *this question* select an
        # answer", because a dimension the question cannot address buys nothing.
        multi = [s for s in slot_detail if s["distinct_values"] > 1]
        if not multi:
            verdict = ("NOT_ACTUALLY_BLOCKED_BY_B" if stage in ("OTHER", "VALIDATION",
                                                               "NOT_IN_POOL")
                       else "BINDER_SEMANTIC")
        elif all(s["one_row"] for s in multi):
            # Every competing value sits in one row: these are columns.  A
            # column is selectable only if something names it, and the question
            # asks for the row, not a column.  Restoring `column_identity` would
            # let the system *describe* the row; it would not tell it which
            # column the answer is.
            verdict = "SOURCE_AMBIGUOUS"
        elif any(s["has_total"] for s in multi):
            # Different rows, and the relation between them is arithmetic and
            # already in the record: one row is the sum of the others.  Nothing
            # outside the store is needed to know which row is the total, so
            # this is recoverable on the evidence available today.
            verdict = "RECOVERABLE_FROM_THE_RECORD"
        else:
            # Different rows, no arithmetic relation.  Which row is meant is
            # settled by the *statement or table* it sits in, and that is in the
            # source rather than in the record.  Recoverable only if the source
            # really carries it, which this audit cannot see -- so it is counted
            # as a conditional, never as a gain.
            verdict = "RECOVERABLE_IF_SOURCE_SELECTS_ONE"

        report["cases"].append({
            "case_id": row["case_id"], "stage": stage, "verdict": verdict,
            "question": row["question"], "slots": slot_detail,
        })

    verdicts = collections.Counter(c["verdict"] for c in report["cases"])
    stages = collections.Counter(c["stage"] for c in report["cases"])
    report["totals"] = {
        "blocked_comparable_answerable": len(blocked),
        "by_verdict": dict(verdicts),
        "by_stage": dict(stages),
        "unclassified": verdicts.get("UNCLASSIFIED", 0),
    }

    # -- the logical/physical duplicate measurement -----------------------
    duplicated = {k: v for k, v in by_logical.items() if len(v) > 1}
    report["store"] = {
        "records": len(records),
        "distinct_candidate_keys": len({str(r.get("candidate_key")) for r in records}),
        "logical_facts": len(by_logical),
        "logical_facts_with_several_physical_keys": len(duplicated),
        "physical_keys_above_logical": sum(len(v) - 1 for v in duplicated.values()),
    }

    print("=" * 78)
    print("B0 -- REPRESENTATION FEASIBILITY")
    print("=" * 78)
    print("  blocked comparable answerable  %d" % len(blocked))
    print("  UNCLASSIFIED                   %d" % report["totals"]["unclassified"])
    print()
    print("  -- by verdict")
    for key, count in verdicts.most_common():
        print("    %-32s %4d" % (key, count))
    print()
    print("  -- by stage")
    for key, count in stages.most_common():
        print("    %-32s %4d" % (key, count))
    print()
    print("  -- store identity")
    for key, value in report["store"].items():
        print("    %-44s %s" % (key, value))
    print()
    for case in report["cases"]:
        print("  %-26s %-10s %s" % (case["case_id"], case["stage"], case["verdict"]))
        for slot in case["slots"]:
            print("      %-46s n=%d distinct=%d one_row=%s frags=%d total=%s"
                  % (str(slot["metric"])[:46], slot["facts"], slot["distinct_values"],
                     slot["one_row"], slot["fragments"], slot["has_total"]))
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "b0-representation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8", newline="\n")
    print("  written to %s" % (args.out / "b0-representation.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
