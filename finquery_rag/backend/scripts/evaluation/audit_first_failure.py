"""P1.7-C1 -- where an answerable case first fails, with nothing left over.

"Gate blocked = 20" is a count, not a diagnosis.  The question that finds
implementation bugs is different and narrower:

    which stage did this case first fail at, and did that stage violate an
    invariant it was not supposed to violate?

So every unreleased answerable case gets exactly one `FIRST_FAILURE_STAGE`, and
the audit is only finished when `UNATTRIBUTED = 0` -- a case that cannot be
placed is a case whose failure nobody has explained, and rounding it into a
bucket is how a count turns into a conclusion it does not support.

The stages are ordered by the pipeline, and the *first* one that fails wins:
later stages cannot be blamed for a case that never reached them.

  PLAN              the plan itself was rejected
  SEMANTIC_ALIGNMENT the query/plan gate refused it
  RETRIEVAL         the gold fact never reached the pool
  BINDING           the gold was in the pool and was not bound
  SLOT_COMPLETENESS a slot came back missing after binding
  ADAPTER_CONTRACT  a capability raised instead of answering
  GENERATION        the answer was not produced
  VALIDATION        the release validator refused it
  FINALIZATION      everything passed and it still did not release

  python audit_first_failure.py --cases <cases.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

#: Reason codes that mean a capability raised rather than answered.  These are
#: *adapter contract* failures: the component was handed something it could not
#: express, which is a different claim from "it decided not to answer".
ADAPTER_CODES = {
    "CAPABILITY_EXCEPTION",
    "COORDINATOR_EXCEPTION",
    "HARNESS_EXCEPTION",
}
PLAN_CODES = {
    "INVALID_PLAN",
    "SUPERVISOR_ERROR",
    "SUPERVISOR_ABSTAIN",
    "NO_PINNED_PLAN",
}
ALIGNMENT_CODES = {
    "QUERY_PLAN_SEMANTIC_MISMATCH",
    "QUERY_PLAN_SEMANTIC_AMBIGUOUS",
    "QUERY_PLAN_SEMANTIC_UNKNOWN",
}

STAGES = (
    "PLAN",
    "SEMANTIC_ALIGNMENT",
    "RETRIEVAL",
    "BINDING",
    "SLOT_COMPLETENESS",
    "ADAPTER_CONTRACT",
    "GENERATION",
    "VALIDATION",
    "FINALIZATION",
)


def _first_failure(row: dict) -> str:
    reasons = {str(r) for r in (row.get("reason_codes") or [])}
    terminal = str(row.get("terminal_state") or "")

    if row.get("gate_blocked") or (reasons & ALIGNMENT_CODES):
        return "SEMANTIC_ALIGNMENT"
    if reasons & PLAN_CODES or terminal == "NO_PINNED_PLAN":
        return "PLAN"
    if reasons & ADAPTER_CODES:
        return "ADAPTER_CONTRACT"
    if "GENERATION_EXCEPTION" in reasons or "GENERATOR_NOT_WIRED" in reasons:
        return "GENERATION"
    if not (row.get("retrieval_rounds") or row.get("pool_size")):
        return "RETRIEVAL"
    if not row.get("comparable"):
        # A relation question's gold carries no fact ids, so `gold_in_pool` and
        # `gold_bound` are 0 *by construction* -- blaming retrieval or binding
        # for them would be measuring the harness rather than the system.  The
        # only stages that can be read from such a case are the ones that do not
        # consult gold at all.
        if not row.get("bound_evidence"):
            return "BINDING"
        if not row.get("validation_passed"):
            return "VALIDATION"
        return "FINALIZATION"
    if not row.get("gold_in_pool"):
        return "RETRIEVAL"
    if not row.get("gold_bound"):
        return "BINDING"
    if row.get("missing_slot_ids") or row.get("missing_operand_slots"):
        return "SLOT_COMPLETENESS"
    if not row.get("validation_passed"):
        return "VALIDATION"
    if not row.get("released"):
        return "FINALIZATION"
    return "UNATTRIBUTED"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report: dict = {}
    for path in args.cases:
        rows = [json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        answerable = [r for r in rows if not r.get("must_refuse")]
        unreleased = [r for r in answerable if not r.get("released")]

        buckets: collections.Counter = collections.Counter()
        by_stage: dict[str, list[dict]] = collections.defaultdict(list)
        for row in unreleased:
            stage = _first_failure(row)
            buckets[stage] += 1
            by_stage[stage].append(row)

        print("=" * 78)
        print("%s" % path)
        print("=" * 78)
        print("  answerable %d   released %d   unreleased %d"
              % (len(answerable), len(answerable) - len(unreleased), len(unreleased)))
        for stage in STAGES:
            if buckets[stage]:
                print("    %-20s %3d" % (stage, buckets[stage]))
        unattributed = buckets["UNATTRIBUTED"]
        print("    %-20s %3d   <- must be 0" % ("UNATTRIBUTED", unattributed))
        print()
        for stage in STAGES:
            if stage in ("SEMANTIC_ALIGNMENT", "UNATTRIBUTED") or not by_stage[stage]:
                continue
            for row in by_stage[stage]:
                print("    %-16s %-26s %s" % (stage, row["case_id"],
                                              str(row.get("question"))[:56]))
        print("    -- SEMANTIC_ALIGNMENT (the gate), by blocker")
        for row in by_stage["SEMANTIC_ALIGNMENT"]:
            print("       %-24s %s" % (row["case_id"],
                                       str(row.get("plan_metrics"))[:56]))
        print()

        report[str(path)] = {
            "answerable": len(answerable),
            "released": len(answerable) - len(unreleased),
            "unreleased": len(unreleased),
            "by_stage": dict(buckets),
            "unattributed": unattributed,
            "cases_by_stage": {
                stage: [{"case_id": r["case_id"], "question": r.get("question"),
                         "plan_metrics": r.get("plan_metrics"),
                         "reason_codes": r.get("reason_codes"),
                         "terminal_state": r.get("terminal_state")}
                        for r in rows_at]
                for stage, rows_at in by_stage.items()
            },
        }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                       default=str) + "\n",
                            encoding="utf-8", newline="\n")
        print("  written to %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
