"""Does a blocked case's gold cite facts of the company the question names?

`tv2f01-s2-sum-007` asks for The Coca-Cola Company's operating income and its
gold cites `Operating income 13,426 | 12,536` -- from the table the filing
introduces as *"a summary of financial information for our equity method
investees"*, whose net operating revenues are 102,800 against Coca-Cola's own
47,941.  The gold attributes the investees' combined figures to Coca-Cola.

One such case is an anecdote.  This asks how many there are, because a case whose
gold names another filer's numbers is not a case the system failed, and counting
it as one misreports both the coverage and the work left.

Reported, never reclassified: moving a case out of the denominator changes the
headline, and that is a decision about the benchmark rather than about the
system.

  python audit_gold_entity_consistency.py --cases <cases.jsonl> --gold <gold.jsonl> \\
      --v2-fact-store <store.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

#: The company a question is about, from its own wording.
_ISSUER = re.compile(
    r"(?:According to|In the|What (?:was|is)|How much did|What percentage of|"
    r"What is the (?:sum|average|difference) of)\s+"
    r"([A-Z][\w&.,'’-]*(?:\s+[A-Z][\w&.,'’-]*){0,4}?)"
    r"(?:'s|’s|\s+annual|\s+report|\s+FY|\s+reported|\s+had|\s+in\s+FY)"
)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _fold(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--v2-fact-store", type=Path, required=True)
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args(argv)

    store = {str(r.get("candidate_key")): r for r in _load(args.v2_fact_store)}
    gold = {r["id"]: r for r in _load(args.gold)}
    rows = [r for r in _load(args.cases)
            if not r.get("must_refuse") and r.get("comparable")]

    flagged: list[tuple] = []
    tally: collections.Counter = collections.Counter()
    for row in rows:
        record = gold.get(row["case_id"]) or {}
        facts = [store.get(str(f)) for f in (record.get("fact_ids") or [])]
        facts = [f for f in facts if f]
        if not facts:
            continue
        match = _ISSUER.search(row["question"])
        issuer = match.group(1).strip() if match else ""
        named = _fold(issuer).split()[0] if issuer else ""
        entities = {_fold(f.get("entity")) for f in facts}
        if not named or not entities:
            tally["NO_ISSUER_PARSED"] += 1
            continue
        if any(named in entity for entity in entities):
            tally["CONSISTENT" if row["released"] else "CONSISTENT_BLOCKED"] += 1
        else:
            tally["MISMATCH_RELEASED" if row["released"] else "MISMATCH_BLOCKED"] += 1
            flagged.append((row["case_id"], row["released"], issuer,
                            sorted(entities), record.get("expected_value")))

    print("=" * 78)
    print("GOLD ENTITY CONSISTENCY")
    print("=" * 78)
    for key, value in tally.most_common():
        print("  %-24s %3d" % (key, value))
    print()
    for case_id, released, issuer, entities, expected in sorted(flagged):
        print("  %-26s released=%-5s question=%r gold-entity=%s expected=%r"
              % (case_id, released, issuer[:24], entities, expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
