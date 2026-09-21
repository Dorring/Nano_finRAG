#!/usr/bin/env python3
"""P1.8 coverage sprint, step 1: where the 25 unreleased answerable cases stop.

The D1 seal leaves 52 of 77 answerable cases released and 25 not.  "25" is a
count, not a diagnosis.  This gives each of the 25 exactly one
`FIRST_FAILURE_STAGE` -- the *earliest* stage at which it stopped, because a
later stage cannot be blamed for a case that never reached it -- and the audit
is finished only when `UNATTRIBUTED = 0`.

    python scripts/evaluation/audit_p1_8_first_failure.py \
        --predictions <replay-predictions.jsonl> \
        --gold <gold-evidence-v1.jsonl>

Why this is not `audit_first_failure.py`
----------------------------------------

That audit reads the E1 attribution schema, which carries `gold_in_pool` and
`gold_bound` computed inside its own arms.  The sealed replay does not record
those, and inventing them here would put a fabricated field under a real stage
name.  So the ladder below is built only from signals the replay actually
emits -- its `reached` trace, its declared reason codes and the Binder's final
status -- and it says so, rather than borrowing a finer vocabulary than its
evidence supports.

**What that costs.**  `RETRIEVAL` and `BINDING` cannot be separated from these
artifacts: `reached.retrieval` means the stage ran, not that the gold was found.
Every case below therefore lands at `BINDING` or later, and a case whose gold
never reached the pool would be attributed one stage too late.  Separating them
needs a per-case pool membership measurement that the sealed run does not carry.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Mapping

#: A capability raised instead of answering: the component was handed something
#: it could not express, which is a different claim from "it decided not to answer".
ADAPTER_CODES = {"CAPABILITY_EXCEPTION", "COORDINATOR_EXCEPTION", "HARNESS_EXCEPTION"}

#: A calculation whose operands did not all arrive.  Binding returned something
#: and the operation still could not run, so the failure is a slot that came back
#: missing *after* binding rather than a binding that never happened.
INCOMPLETE_OPERAND_CODES = {"CALCULATION_INVALID", "INSUFFICIENT_OPERANDS", "MISSING_OPERAND"}

STAGES = (
    "SEMANTIC_ALIGNMENT",
    "RETRIEVAL",
    "BINDING",
    "SLOT_COMPLETENESS",
    "ADAPTER_CONTRACT",
    "GENERATION",
    "VALIDATION",
    "FINALIZATION",
)


def first_failure(row: Mapping[str, Any], *, ignore_gate: bool = False) -> tuple[str, str]:
    """(stage, detail).  The detail is why this stage and not a later one.

    `ignore_gate` drops the first rule.  It exists because the two views answer
    different questions and the difference is the whole gate debate: pipeline
    order says a gate-refused case failed at the gate, while the replay -- which
    *overrides* refusals so the rest of the chain can be measured -- shows what
    would block it next.  A coverage plan that only reads the first view would
    spend itself on the gate; one that only reads the second would never see it.
    """

    reasons = {str(r) for r in (row.get("reason_codes") or [])}
    reached = row.get("reached") or {}
    binder = str(row.get("binder_final_status") or "")

    # The gate is first in the pipeline.  A case it refused ran on only because
    # the replay overrides refusals so the rest of the chain can be measured; in
    # production it stops here, so this is its first failure.
    if row.get("align_overridden") and not ignore_gate:
        return "SEMANTIC_ALIGNMENT", f"gate refused ({row.get('computed_align_status')})"

    if reasons & ADAPTER_CODES:
        return "ADAPTER_CONTRACT", ",".join(sorted(reasons & ADAPTER_CODES))

    if not reached.get("binding"):
        detail = {
            "AMBIGUOUS": "binder ambiguous",
            "MISSING": "binder missing",
            "BOUND": "binding did not reach evidence",
        }.get(binder, f"binder={binder or 'none'}")
        return "BINDING", f"{detail}: {','.join(sorted(reasons)) or 'no reason code'}"

    if reasons & INCOMPLETE_OPERAND_CODES or row.get("binder_final_missing_slot_ids"):
        return "SLOT_COMPLETENESS", ",".join(sorted(reasons & INCOMPLETE_OPERAND_CODES)) or "missing slot after binding"

    if reached.get("validation") and str(row.get("validator_status")) == "FAIL":
        return "VALIDATION", ",".join(sorted(reasons))

    if not reached.get("generation"):
        return "GENERATION", "generation stage never ran"

    return "UNATTRIBUTED", "passed every recorded stage and did not release"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    def load(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    gold = {row["id"]: row for row in load(args.gold)}
    rows = load(args.predictions)
    answerable = [r for r in rows if gold[r["id"]].get("expected_outcome") == "ANSWER"]
    unreleased = [r for r in answerable if r.get("release_status") != "RELEASED"]

    buckets: collections.Counter = collections.Counter()
    attributed: list[dict] = []
    for row in unreleased:
        stage, detail = first_failure(row)
        buckets[stage] += 1
        attributed.append(
            {
                "case_id": row["id"],
                "stratum": row.get("stratum"),
                "route": row.get("route"),
                "operation": gold[row["id"]].get("operation"),
                "first_failure_stage": stage,
                "detail": detail,
                "status": row.get("status"),
                "reason_codes": row.get("reason_codes"),
                "binder_final_status": row.get("binder_final_status"),
                "validator_status": row.get("validator_status"),
                "question": row.get("question"),
            }
        )

    print("answerable %d   released %d   unreleased %d"
          % (len(answerable), len(answerable) - len(unreleased), len(unreleased)))
    print()
    for stage in STAGES:
        if buckets[stage]:
            print("    %-20s %3d" % (stage, buckets[stage]))
    unattributed = buckets["UNATTRIBUTED"]
    print("    %-20s %3d   <- must be 0" % ("UNATTRIBUTED", unattributed))
    print()
    # The same ladder with the gate rule dropped: what blocks each case once the
    # gate is out of the way.  The gate is overridden in this track, so these are
    # the stages the refused cases actually reached.
    gate_free: collections.Counter = collections.Counter()
    for row in unreleased:
        stage, _ = first_failure(row, ignore_gate=True)
        gate_free[stage] += 1
    print("  same cases, gate rule dropped (the replay overrides gate refusals):")
    for stage in STAGES:
        if gate_free[stage]:
            print("    %-20s %3d" % (stage, gate_free[stage]))
    print("    %-20s %3d   <- must be 0" % ("UNATTRIBUTED", gate_free["UNATTRIBUTED"]))
    print()
    for stage in STAGES:
        rows_at = [a for a in attributed if a["first_failure_stage"] == stage]
        if not rows_at:
            continue
        print(f"-- {stage}")
        for item in rows_at:
            print("   %-26s %-22s %-26s %s"
                  % (item["case_id"], item["stratum"], item["detail"][:26],
                     str(item["question"])[:44]))
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "answerable": len(answerable),
                    "released": len(answerable) - len(unreleased),
                    "unreleased": len(unreleased),
                    "by_stage": dict(buckets),
                    "by_stage_ignoring_gate": dict(gate_free),
                    "unattributed": unattributed,
                    "cases": attributed,
                    "retrieval_binding_caveat": (
                        "reached.retrieval records that the stage ran, not that the "
                        "gold reached the pool; RETRIEVAL cannot be separated from "
                        "BINDING on these artifacts"
                    ),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print("written to %s" % args.out)

    return 1 if unattributed else 0


if __name__ == "__main__":
    raise SystemExit(main())
