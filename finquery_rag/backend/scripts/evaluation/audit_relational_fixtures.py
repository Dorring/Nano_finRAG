"""P1.5-S2E step 3: check the migrated fixtures before any model runs.

Cross-entity comparison and ranking now carry an *operation*, which moves them
off the unchecked prose path and onto deterministic execution.  That is a
change to what the pipeline will do, so it is worth proving statically -- with
no retrieval, no binder and no GPU -- that every migrated plan can actually be
executed and resolved.

What this can check without running anything:

* the operation is expressible in the plan vocabulary and executable in the
  registry, and the two agree;
* the arity matches the operation -- a comparison ranks exactly two operands, a
  ranking at least two;
* every slot is addressable: unique ``slot_id``, and a slot that a relational
  result would name;
* the directory built from the plan resolves every slot the plan declares, so a
  result over those refs cannot come back unresolvable;
* nothing in the plan refers to an evidence id, which a relational result must
  never do.

What it cannot check, and says so rather than implying otherwise: whether a
*specific* case binds enough evidence to produce an ordering.  That is the
run's job.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_SCRIPTS = _BACKEND_DIR / "scripts/evaluation"
for _path in (str(_BACKEND_DIR), str(_SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_p1_2_dual_track_benchmark as runner  # noqa: E402

RELATIONAL = ("comparison", "ranking")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=runner.DEFAULT_FIXTURES)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    from rag_v2.contracts.plan import _OPERATION_VALUES, SupervisorPlan
    from src.finance.calculation_registry import CALCULATION_REGISTRY, supports
    from src.runtime.trusted_v2_calculation import build_relational_operand_directory

    rows = runner._load_jsonl(args.fixtures)
    migrated = [
        row for row in rows if row["plan"].get("operation") in RELATIONAL
    ]

    print(f"=== fixtures: {len(rows)} total, {len(migrated)} carrying a relational operation ===")
    print("  by operation:", dict(collections.Counter(
        row["plan"]["operation"] for row in migrated
    )))
    print("  by stratum :", dict(collections.Counter(
        row["stratum"] for row in migrated
    )))
    print()

    problems: list[str] = []
    for row in migrated:
        case = row["id"]
        plan = SupervisorPlan.from_dict(row["plan"])
        operation = plan.operation

        def fail(message: str) -> None:
            problems.append(f"{case}: {message}")

        # Expressible and executable, and the two agree.
        if operation not in _OPERATION_VALUES:
            fail(f"operation {operation!r} is not in the plan vocabulary")
        if not supports(operation):
            fail(f"operation {operation!r} has no registry entry")

        # Arity.
        slots = plan.required_slots
        if len(slots) < 2:
            fail(f"{operation} needs at least 2 slots, has {len(slots)}")
        if operation == "comparison" and len(slots) != 2:
            fail(f"comparison ranks exactly 2 operands, has {len(slots)}")

        # Addressability.
        slot_ids = [slot.slot_id for slot in slots]
        if len(set(slot_ids)) != len(slot_ids):
            fail(f"slot ids are not unique: {slot_ids}")
        if any(not slot_id for slot_id in slot_ids):
            fail("a slot has an empty slot_id")

        # The directory must resolve every slot the plan declares, or a result
        # over those refs comes back unresolvable and the release invariant
        # refuses a plan that was in fact answerable.
        directory = build_relational_operand_directory(plan)
        unresolved = [slot_id for slot_id in slot_ids if slot_id not in directory]
        if unresolved:
            fail(f"directory cannot resolve {unresolved}")

        # A relational result refers to semantic operands, never to evidence.
        for slot in slots:
            blob = json.dumps(slot.to_dict(), sort_keys=True)
            if "evidence" in blob or "chunk" in blob:
                fail(f"slot {slot.slot_id} mentions evidence")

        # Intention: these are multi-side questions, not calculations.
        if plan.intent.value != "MULTI_EVIDENCE":
            fail(f"intent is {plan.intent.value}, expected MULTI_EVIDENCE")

    print("=== static audit ===")
    if problems:
        for problem in problems:
            print(f"  FAIL  {problem}")
    else:
        print(f"  all {len(migrated)} migrated plans pass")
    print()

    print("=== the cases the exit gate names ===")
    for case in ("tv2f01-s3-compare-002", "tv2f01-s3-rank-002"):
        row = next((r for r in migrated if r["id"] == case), None)
        if row is None:
            print(f"  {case}: NOT MIGRATED")
            continue
        plan = SupervisorPlan.from_dict(row["plan"])
        directory = build_relational_operand_directory(plan)
        print(f"  {case}: operation={plan.operation} slots={len(plan.required_slots)}")
        print(f"      resolvable refs: {[ref.slot_id for ref in directory.refs]}")

    # Regression guard: nothing outside cross-entity may have moved.
    others = [
        row["id"]
        for row in rows
        if row["plan"].get("operation") in RELATIONAL
        and row["stratum"] != "cross_entity_comparison"
    ]
    print()
    print(f"  non-cross-entity cases carrying a relational operation: {others or 'none'}")

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "relational_fixture_audit.json").write_text(
            json.dumps(
                {
                    "fixture": str(args.fixtures),
                    "migrated": {row["id"]: row["plan"]["operation"] for row in migrated},
                    "problems": problems,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
