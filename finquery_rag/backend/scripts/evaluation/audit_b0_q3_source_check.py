"""B0 Q3 -- can the source say which row a fact is?

The eight `RECOVERABLE_IF_SOURCE_SELECTS_ONE` cases are the whole of the Go
margin: if a row's enclosing statement or table cannot be recovered, they are
zero and the migration's ceiling drops from 60.7% to 55.4%.

The first place to look is not the HTML corpus but the record, because the store
already keeps each fact's own row in `content`.  For a fact whose row label is
shared with its siblings -- four "Cost of revenue" rows under one segment header
-- the question is whether `content` carries anything that differs per *row*
rather than per *fragment*.  If the first line is the same on all of them, the
header is a fragment attribute that has already been flattened across the rows
it was meant to separate, and nothing short of re-extraction recovers it.

  python audit_b0_q3_source_check.py --cases <cases.jsonl> --pinned <pinned.jsonl> \\
      --v2-fact-store <store.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--v2-fact-store", type=Path, required=True)
    args = parser.parse_args(argv)

    from rag_v2.supervisor.semantic_alignment import canonical_period_id
    from src.runtime.trusted_v2_production import StructuredFactStore

    store = StructuredFactStore(args.v2_fact_store)
    pinned = {row["case_id"]: row for row in _load(args.pinned)}
    rows = [r for r in _load(args.cases)
            if r.get("comparable") and not r.get("must_refuse")]

    tally: collections.Counter = collections.Counter()
    print("=" * 78)
    print("B0 Q3 -- is a row's enclosing context differentiable in the record?")
    print("=" * 78)
    for row in rows:
        if row.get("released"):
            continue
        entry = pinned.get(row["case_id"]) or {}
        for slot in (entry.get("plan") or {}).get("required_slots") or []:
            facts = list(store.facts_at_coordinate(
                slot.get("entity"), slot.get("metric"),
                canonical_period_id(slot.get("period"))))
            if not facts:
                continue
            values = {str(f.get("value")) for f in facts}
            if len(values) < 2:
                continue
            heads = collections.Counter(
                str(f.get("content") or "").split("\n")[0].strip()[:58]
                for f in facts
            )
            # Does the *first line* separate the rows from each other?  A header
            # that is identical on every row of a shared label separates
            # fragments, not rows.
            distinct = len(heads)
            verdict = ("ROW_CONTEXT_DIFFERS" if distinct == len(facts)
                       else "ONE_CONTEXT_FOR_ALL_ROWS" if distinct == 1
                       else "PARTIAL")
            tally[verdict] += 1
            tally["facts"] += len(facts)
            print("  %-26s %-26s n=%d distinct_contexts=%d  %s"
                  % (row["case_id"], str(slot.get("metric"))[:26], len(facts),
                     distinct, verdict))
            for head, count in heads.most_common(3):
                print("        x%-3d %s" % (count, head))
            break
    print()
    print("  %s" % dict(tally))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
