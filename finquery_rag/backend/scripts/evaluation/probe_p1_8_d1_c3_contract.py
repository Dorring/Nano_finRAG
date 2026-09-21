#!/usr/bin/env python3
"""P1.8-D1-C3 probe: what does the authoring contract actually produce?

Run on 4090-qh, where the fact store lives.

    .venv/bin/python scripts/evaluation/probe_p1_8_d1_c3_contract.py

It answers three questions the C3 brief needs answered before any fixture is
regenerated:

1. Does `author_plan` -- the contract, invoked as a module rather than
   re-implemented -- derive the cross-entity slot order from `gold.fact_ids`?
2. Does the coordinate index the validator reads still resolve those slots, or
   is the full build's stop at `compare-001` a defect in the contract?
3. Does the answer depend on which gold is passed (V2 versus the pre-migration
   revision the host repo carries)?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path("/disk/qh/nano-finrag/finquery_rag/backend")
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(BACKEND / "scripts/evaluation") not in sys.path:
    sys.path.insert(0, str(BACKEND / "scripts/evaluation"))

import build_p1_2_plan_fixtures as B  # noqa: E402

V2 = Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-8-c-v2")
REPO = BACKEND / "benchmarks/tv2_canonical_v1"

GOLDS = {
    "v2   (a3d17211)": V2 / "gold-evidence-v1.jsonl",
    "pre  (3d2a0c5b)": REPO / "gold-evidence-v1.jsonl",
}
EVAL_SET = V2 / "canonical-eval-v1.jsonl"

CASES = [
    "tv2f01-s3-crossdiff-001",
    "tv2f01-s3-crossdiff-002",
    "tv2f01-s3-crossdiff-003",
    "tv2f01-s3-crossdiff-004",
    "tv2f01-s3-crossdiff-005",
    "tv2f01-s3-compare-001",
]


def main() -> int:
    eval_by_id = B._load_records(EVAL_SET)
    stores = [B.DEFAULT_FACT_STORE, B.DEFAULT_IXBRL_FACT_STORE]
    needed: list[str] = []
    for gold_path in GOLDS.values():
        needed.extend(B.operand_fact_ids(gold_path))
    facts = B.load_fact_coordinates(stores, needed)
    index = B.load_coordinate_index(stores)
    print(f"fact store records read: {len(facts)}   index keys: {len(index)}")
    print()

    for label, gold_path in GOLDS.items():
        gold_by_id = B._load_gold(gold_path)
        print("=" * 78)
        print(f"GOLD {label}  ({gold_path})")
        print("=" * 78)
        for case in CASES:
            record = eval_by_id[case]
            gold = gold_by_id[case]
            row = B.author_plan(record["question"], gold, record, facts)
            B._enrich_slots(row, gold, facts, index, case=case, entity_hint="")
            print(f"{case}")
            print(f"    q          : {row['question']}")
            print(f"    op/intent  : {row['plan']['operation']} / {row['plan']['intent']}")
            print(f"    gold facts : {gold.get('fact_ids')}")
            print(f"    gold values: {gold.get('values')}  expected={gold.get('expected_value')}")
            for slot in row["plan"]["required_slots"]:
                count = len(B._slot_candidates(slot, index))
                print(
                    f"      {slot['slot_id']} role={slot.get('role'):<11} "
                    f"entity={str(slot.get('entity')):<22} metric={str(slot.get('metric')):<20} "
                    f"period={slot.get('period')} candidates={count}"
                )
            print(
                "    sourced    : "
                + json.dumps(row.get("sourced_from"), sort_keys=True, ensure_ascii=False)
            )
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
