#!/usr/bin/env python3
"""C0.2 + C0.4: two attributions, and a recoverability label per case.

The two attributions answer different questions and are not averaged:

  A  PRODUCTION      the gate is first in the pipeline, so a gate-refused case
                     stops there.  This is the real FIRST_FAILURE_STAGE.
  B  GATE-BYPASS     what blocks each case once the gate is out of the way.  The
                     replay already overrides refusals to measure the rest of the
                     chain, so these are the stages those cases actually reached.
                     **Not a production first-failure count.**

Both use the pool trace, so `RETRIEVAL` is decided by whether a slot's gold fact
reached the Binder's pool -- not by `reached.retrieval`, which only says the
stage ran.

    python scripts/evaluation/attribute_p1_8_coverage.py \
        --pool-trace <pool-membership.jsonl> --gold <gold-evidence-v1.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Mapping

ADAPTER_CODES = {"CAPABILITY_EXCEPTION", "COORDINATOR_EXCEPTION", "HARNESS_EXCEPTION"}
BUDGET_CODES = {"BUDGET_EXHAUSTED", "NO_PROGRESS"}
INCOMPLETE_CODES = {"CALCULATION_INVALID", "INSUFFICIENT_OPERANDS", "MISSING_OPERAND"}

STAGES = (
    "SEMANTIC_ALIGNMENT",
    "TRUE_RETRIEVAL_MISS",
    "BINDING",
    "SLOT_COMPLETENESS",
    "ADAPTER_CONTRACT",
    "GENERATION",
    "VALIDATION",
    "FINALIZATION",
    "UNATTRIBUTED",
)

#: Sub-buckets for BINDING, then the recoverability label each implies.  The
#: labels are not merged: a bucket that can be recovered by a rule and one that
#: needs a representation change are different work with different risk.
BINDING_SUB = {
    "EVIDENCE_CONFLICT": "CAPABILITY_GAP",
    "BUDGET_EXHAUSTED": "DETERMINISTIC_RECOVERABLE",
    "SCHEMA_CONTRACT": "IMPLEMENTATION_BUG",
    "STRUCTURAL_DISAMBIGUATION": "CAPABILITY_GAP",
    "OTHER": "UNCLASSIFIED",
}


#: Three cases need a judgement the trace cannot make.  Each carries the
#: evidence that decides it rather than a rule, so the label is auditable and
#: not a silent special-case -- and the evidence is the store record or the
#: slot, not an opinion about the case.
JUDGED = {
    "tv2f01-s3-compare-001": (
        "POLICY_CORRECT_BLOCK",
        "gold is in the pool at rank 6 and 1 and the Binder returned BOUND; the "
        "refusal is `QUERY_EVIDENCE_SEMANTIC_MISMATCH` because Coca-Cola files "
        "the line as `Consolidated Net Income` where the slot says `Net income`. "
        "P1.7 measured this and declined the alias: `net_income` already carries "
        "归母净利润, and consolidated net income includes non-controlling "
        "interests, so the two are different numbers in any filing that has one.",
    ),
    "tv2f01-s1-nvda-025": (
        "SOURCE_AMBIGUOUS",
        "the stored value is `2024`, which is the *year column* of the row "
        "`| President and CEO | 2024 | 996,514 | 26,676,415 | ... |` -- a "
        "compensation table. The question asks for a person and the gold is a "
        "year, so the gold does not denote what the question names.",
    ),
    "tv2f01-s1-tsla-035": (
        "IMPLEMENTATION_BUG",
        "gold is in the pool at rank 1 and bound; the validator refuses with "
        "`SCV_UNIT_UNSUPPORTED`. The source row reads `Risk-free interest rate | "
        "3.95 % | 3.92 % | 3.90 %` while the record's `unit` is null, so the "
        "emitter dropped a unit the source states and the validator has nothing "
        "to read. Recoverable by parsing the unit already in the row.",
    ),
}


def stage_b(row: Mapping[str, Any]) -> tuple[str, str]:
    """Gate-bypass ladder: earliest stage that failed, retrieval decided by the pool."""

    reasons = {str(r) for r in (row.get("reason_codes") or [])}
    per_slot = row["per_slot"]
    absent = [s["required_slot_id"] for s in per_slot if not s["gold_in_final_pool"]]

    if absent:
        return "TRUE_RETRIEVAL_MISS", f"no authoritative gold match in pool for slot(s) {absent}"
    if reasons & ADAPTER_CODES:
        return "ADAPTER_CONTRACT", ",".join(sorted(reasons & ADAPTER_CODES))
    if not row.get("reached_binding"):
        return "BINDING", f"binder={row.get('binder_final_status')}: {','.join(sorted(reasons))}"
    if reasons & INCOMPLETE_CODES:
        return "SLOT_COMPLETENESS", ",".join(sorted(reasons & INCOMPLETE_CODES))
    if row.get("validator_status") == "FAIL":
        return "VALIDATION", ",".join(sorted(reasons))
    return "FINALIZATION", "passed every recorded stage and did not release"


def binding_sub(row: Mapping[str, Any]) -> str:
    reasons = {str(r) for r in (row.get("reason_codes") or [])}
    if reasons & BUDGET_CODES:
        return "BUDGET_EXHAUSTED"
    if reasons & ADAPTER_CODES:
        return "SCHEMA_CONTRACT"
    if "EVIDENCE_CONFLICT" in reasons or str(row.get("binder_final_status")) == "AMBIGUOUS":
        # The gold IS in the pool and the Binder still would not choose: the
        # coordinate holds several values and nothing in it says which is meant.
        return "STRUCTURAL_DISAMBIGUATION"
    return "OTHER"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-trace", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    trace = [
        json.loads(line)
        for line in args.pool_trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in args.gold.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    unreleased = [r for r in trace if r["release_status"] != "RELEASED"]
    production: collections.Counter = collections.Counter()
    bypass: collections.Counter = collections.Counter()
    sub: collections.Counter = collections.Counter()
    labels: collections.Counter = collections.Counter()
    cases = []

    for row in unreleased:
        stay, detail_b = stage_b(row)
        if row.get("gate_refused"):
            stage_a = "SEMANTIC_ALIGNMENT"
        else:
            stage_a = stay
        production[stage_a] += 1
        bypass[stay] += 1

        sub_bucket = binding_sub(row) if stay == "BINDING" else None
        if sub_bucket:
            sub[sub_bucket] += 1
            label = BINDING_SUB[sub_bucket]
        elif stay == "TRUE_RETRIEVAL_MISS":
            label = "CAPABILITY_GAP"
        elif stay == "ADAPTER_CONTRACT":
            label = "IMPLEMENTATION_BUG"
        elif stay == "SLOT_COMPLETENESS":
            label = "IMPLEMENTATION_BUG"
        else:
            label = "UNCLASSIFIED"
        judged = JUDGED.get(row["case_id"])
        if judged:
            label, judged_note = judged
        else:
            judged_note = None
        labels[label] += 1

        cases.append(
            {
                "case_id": row["case_id"],
                "stratum": row["stratum"],
                "operation": row["operation"],
                "A_production_stage": stage_a,
                "B_gate_bypass_stage": stay,
                "bypass_detail": detail_b,
                "binding_sub_bucket": sub_bucket,
                "recoverability": label,
                "recoverability_evidence": judged_note,
                "gold_pool_complete": row["gold_pool_complete"],
                "pool_ranks": [
                    s.get("final_pool_rank", s.get("matched_pool_ranks"))
                    for s in row["per_slot"]
                ],
                "match_classes": [s.get("match_class") for s in row["per_slot"]],
                "binder_final_status": row.get("binder_final_status"),
                "reason_codes": row.get("reason_codes"),
                "question": (gold.get(row["case_id"]) or {}).get("expected_value"),
            }
        )

    print("A. PRODUCTION FIRST_FAILURE_STAGE (gate ON -- the real one)")
    for stage in STAGES:
        if stage != "UNATTRIBUTED" and production[stage]:
            print(f"     {stage:<20} {production[stage]:>3}")
    print(f"     {'UNATTRIBUTED':<20} {production['UNATTRIBUTED']:>3}   <- must be 0")
    print()
    print("B. GATE-BYPASS downstream distribution (diagnostic, NOT production)")
    for stage in STAGES:
        if stage != "UNATTRIBUTED" and bypass[stage]:
            print(f"     {stage:<20} {bypass[stage]:>3}")
    print(f"     {'UNATTRIBUTED':<20} {bypass['UNATTRIBUTED']:>3}   <- must be 0")
    print()
    print("   BINDING sub-buckets (not merged):")
    for name in ("STRUCTURAL_DISAMBIGUATION", "EVIDENCE_CONFLICT", "BUDGET_EXHAUSTED", "SCHEMA_CONTRACT", "OTHER"):
        if sub[name]:
            print(f"     {name:<28} {sub[name]:>3}")
    print()
    print("   recoverability labels:")
    for name in ("DETERMINISTIC_RECOVERABLE", "IMPLEMENTATION_BUG", "CAPABILITY_GAP", "SOURCE_AMBIGUOUS", "POLICY_CORRECT_BLOCK", "UNCLASSIFIED"):
        print(f"     {name:<28} {labels[name]:>3}")
    print()
    print("   per case:")
    print(f"     {'case':<26} {'A':<18} {'B':<17} {'binding sub':<26} {'recoverability':<24} ranks")
    for case in sorted(cases, key=lambda c: (c["B_gate_bypass_stage"], c["case_id"])):
        print("     %-26s %-18s %-17s %-26s %-24s %s"
              % (case["case_id"], case["A_production_stage"], case["B_gate_bypass_stage"],
                 case["binding_sub_bucket"] or "-", case["recoverability"], case["pool_ranks"]))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "A_production": dict(production),
                    "B_gate_bypass": dict(bypass),
                    "binding_sub_buckets": dict(sub),
                    "recoverability": dict(labels),
                    "cases": cases,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\nwritten to {args.out}")

    unattributed = production["UNATTRIBUTED"] + bypass["UNATTRIBUTED"]
    return 1 if unattributed else 0


if __name__ == "__main__":
    raise SystemExit(main())
