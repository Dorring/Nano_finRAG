"""Which citations miss the gold, and whether they miss it by id or by fact.

Citation precision compares cited candidate ids against the benchmark's gold
fact ids.  When an answer is correct its cited facts are usually the gold facts
-- but not always the same *rows*: a value is often stated twice, once in the
statement and once in a note, and the two rows carry different ids and the same
quantity.

The difference matters.  A citation that names the same quantity by a different
row is a stricter-than-necessary metric; a citation that names a different
quantity is a real grounding failure.  This separates them by comparing the
(e, metric, period, value) of the cited fact against the gold's.

  python audit_citation_precision.py --cases <cases.jsonl> --gold <gold.jsonl> \\
      --fact-store <store.jsonl>
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

import score_nf_v3_final as scorer  # noqa: E402


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _coord(record: dict) -> tuple:
    return (str(record.get("entity") or "").casefold(),
            str(record.get("metric") or "").casefold(),
            str(record.get("period") or "").casefold(),
            str(record.get("value") or ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--fact-store", type=Path, required=True)
    args = parser.parse_args(argv)

    gold = {row["id"]: row for row in _load(args.gold)}
    aliases = scorer.load_alias_map(args.fact_store)
    records: dict[str, dict] = {}
    for record in _load(args.fact_store):
        candidate = str(record.get("candidate_id") or record.get("candidate_key") or "")
        if candidate:
            records[candidate] = record

    rows = [r for r in _load(args.cases)
            if r.get("comparable") and not r.get("must_refuse") and r.get("released")]
    tally: collections.Counter = collections.Counter()

    for row in rows:
        record = gold.get(row["case_id"]) or {}
        wanted = {scorer.resolve(f, aliases) for f in (record.get("fact_ids") or [])}
        wanted_coords = {_coord(records[w]) for w in wanted if w in records}
        cited = [scorer.resolve(c, aliases) for c in (row.get("citation_ids") or [])]
        for value in cited:
            if value in wanted:
                tally["CITED_GOLD_ROW"] += 1
            elif _coord(records.get(value, {})) in wanted_coords:
                tally["SAME_QUANTITY_OTHER_ROW"] += 1
                print("  %-26s same quantity, other row: %s vs gold"
                      % (row["case_id"], records.get(value, {}).get("metric")))
            else:
                tally["DIFFERENT_QUANTITY"] += 1
                print("  %-26s DIFFERENT quantity: cited %r %r %r = %r"
                      % (row["case_id"],
                         records.get(value, {}).get("entity"),
                         records.get(value, {}).get("metric"),
                         records.get(value, {}).get("period"),
                         records.get(value, {}).get("value")))
    total = sum(tally.values())
    print()
    print("  cited total %d" % total)
    for key, count in tally.most_common():
        print("    %-26s %4d  (%.1f%%)" % (key, count, 100.0 * count / max(1, total)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
