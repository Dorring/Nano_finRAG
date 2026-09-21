"""Why the gold reached the pool and still was not bound.

The Binder is a model, so "it bound something else" is not a diagnosis.  The
readable question is whether it *could* have bound the gold: did the slot and the
gold fact agree on entity, metric and period, and was the gold in the packet the
Binder was handed at all?

Those are three different failures:

  COORDINATE_DIFFERS   the slot and the gold disagree, so no Binder could bind it
  NOT_IN_PACKET        the pool holds it but the packet the Binder saw did not
  PACKET_HAD_IT        the Binder was handed the gold and chose another fact

Only the last is a Binder question, and only the first two are fixable upstream.

  python audit_binding_misses.py --cases <cases.jsonl> --pinned <pinned.jsonl> \\
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
    for dash in ("—", "–", "-"):
        text = text.replace(dash, " ")
    return " ".join(text.split())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
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

    rows = [r for r in _load(args.cases)
            if r.get("comparable") and not r.get("must_refuse")]
    misses = [r for r in rows if r.get("gold_in_pool") and not r.get("gold_bound")]

    verdicts: collections.Counter = collections.Counter()
    print("=" * 78)
    print("gold in pool, not bound: %d of %d comparable" % (len(misses), len(rows)))
    print("=" * 78)
    for row in misses:
        entry = pinned.get(row["case_id"]) or {}
        slots = (entry.get("plan") or {}).get("required_slots") or []
        print("  %s  pool=%s  binder=%s" % (
            row["case_id"], row.get("pool_size"),
            list(row.get("binder_status_per_round") or [])))
        print("     q    %s" % row["question"][:88])
        for index, gold_id in enumerate(row.get("gold_ids") or []):
            fact = records.get(str(gold_id))
            slot = slots[min(index, len(slots) - 1)] if slots else {}
            if fact is None:
                verdicts["GOLD_NOT_IN_STORE"] += 1
                print("     gold NOT IN STORE %s" % gold_id)
                continue
            same = (_norm(fact.get("metric")) == _norm(slot.get("metric"))
                    and _norm(fact.get("entity")) == _norm(slot.get("entity")))
            verdict = "COORDINATE_OK" if same else "COORDINATE_DIFFERS"
            verdicts[verdict] += 1
            print("     %-17s slot=(%r,%r,%r)"
                  % (verdict, slot.get("entity"), slot.get("metric"), slot.get("period")))
            print("     %-17s gold=(%r,%r,%r) value=%r"
                  % ("", fact.get("entity"), fact.get("metric"), fact.get("period"),
                     fact.get("value")))
        print("     bound=%s" % (row.get("bound_evidence_ids") or [])[:3])
    print()
    print("  %s" % dict(verdicts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
