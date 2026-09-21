"""Why the gold fact never reached the pool, case by case.

`gold not in pool` is the one loss no later stage can recover from, and it has
several distinct causes that the count alone cannot separate: the slot asked for
a label the store spells differently, the slot named the wrong period, the
document scope excluded the fact, or the fact is not in the store at all.  Each
needs a different fix and one of them needs no fix.

The store record is printed beside the slot so the difference is readable rather
than inferred.

  python audit_retrieval_misses.py --cases <cases.jsonl> --pinned <pinned.jsonl> \\
      --v2-fact-store <store.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
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


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.replace("—", " ").replace("–", " ").split())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, action="append", required=True)
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--v2-fact-store", type=Path, required=True)
    args = parser.parse_args(argv)

    records: dict[str, dict] = {}
    for record in _load(args.v2_fact_store):
        for field in ("candidate_key", "fact_id", "evidence_id", "candidate_id"):
            key = record.get(field)
            if key:
                records.setdefault(str(key), record)
    pinned = {row["case_id"]: row for row in _load(args.pinned)}

    for path in args.cases:
        rows = [r for r in _load(path) if r.get("comparable") and not r.get("must_refuse")]
        misses = [r for r in rows if not r.get("gold_in_pool") and r.get("reached_retrieval", True)
                  and not r.get("gate_blocked")]
        print("=" * 78)
        print("%s   comparable=%d   gold-not-in-pool=%d" % (path, len(rows), len(misses)))
        print("=" * 78)

        verdicts: collections.Counter = collections.Counter()
        for row in misses:
            entry = pinned.get(row["case_id"]) or {}
            slots = (entry.get("plan") or {}).get("required_slots") or []
            gold_ids = row.get("gold_ids") or []
            facts = [records.get(g) for g in gold_ids]
            slot = slots[0] if slots else {}
            print("  %s  pool=%s" % (row["case_id"], row.get("pool_size")))
            print("     q    %s" % row["question"][:88])
            print("     slot entity=%r metric=%r period=%r"
                  % (slot.get("entity"), slot.get("metric"), slot.get("period")))
            for fact in facts:
                if fact is None:
                    verdicts["GOLD_NOT_IN_STORE"] += 1
                    print("     gold NOT IN STORE")
                    continue
                same_metric = _norm(fact.get("metric")) == _norm(slot.get("metric"))
                same_entity = _norm(fact.get("entity")) == _norm(slot.get("entity"))
                if not same_metric:
                    verdicts["SLOT_METRIC_DIFFERS"] += 1
                elif not same_entity:
                    verdicts["SLOT_ENTITY_DIFFERS"] += 1
                else:
                    verdicts["SAME_COORDINATE_NOT_RETRIEVED"] += 1
                print("     gold entity=%r metric=%r period=%r value=%r  [%s%s]"
                      % (fact.get("entity"), fact.get("metric"), fact.get("period"),
                         fact.get("value"),
                         "metric-ok" if same_metric else "METRIC-DIFFERS",
                         " entity-ok" if same_entity else " ENTITY-DIFFERS"))
        print()
        print("  %s" % dict(verdicts))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
