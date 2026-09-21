"""E1-0: freeze the planner, and measure how much freezing costs.

The retrieval query in production is `"<metric> <period>"` per slot, and those
come from `SupervisorPlan.required_slots` -- produced by a live LLM with
`max_retries=0`, no cache and no seed.  So a two-arm retrieval experiment cannot
be run against the live planner: re-plan per arm and the arms differ in what was
asked for as well as in what was retrieved, and no difference between them is
attributable to either.

This writes the plan once and makes it a file.  Every arm replays it.

**The first sample is the frozen one.**  The planner is called `--samples` times
per question and the *first* response is what gets pinned, whatever the later
ones are.  Taking the sample that looks best would be cherry-picking a plan to
flatter a downstream number, and the stability measurement below exists to be
reported, not to be selected on.

**Stability is reported, not repaired.**  If plans are unstable that is a
finding about the pipeline and belongs in the record as capability debt.  The
metrics are exact-match rather than similarity scores:

  plan_exact      the whole serialized plan is identical across samples
  slot_count      the number of required slots is identical
  slot_identity   every slot, aligned by slot_id, agrees on entity, metric and
                  period -- the three fields the retrieval demand is built from

A slot that only exists in some samples is a failure of `slot_identity`, not a
partial match: `to_dict`/`from_dict` require unique slot ids and the demand
builder emits one lane per slot, so a differing slot set is a differing number
of retrieval lanes.

  .venv/bin/python scripts/evaluation/pin_supervisor_plans.py \\
      --out-dir /disk/qh/nano-finrag/artifacts/evaluation/e1-pinned-plans --samples 3
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

BENCH_DIR = _BACKEND_DIR / "benchmarks" / "tv2_canonical_v1"

#: Fields the retrieval demand is actually built from.  `slot_id` is excluded
#: because it is an identity, not a semantic: two samples can agree on what is
#: wanted and disagree on what to call it.
DEMAND_FIELDS = ("entity", "entity_id", "metric", "period")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _compare(sample: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, bool]:
    """Exact structural agreement between two plan dicts."""
    left = list(sample.get("required_slots") or [])
    right = list(reference.get("required_slots") or [])
    by_id = {slot.get("slot_id"): slot for slot in left}
    identity = True
    for slot in right:
        # A slot the sample does not have is a differing slot set, which is a
        # differing number of retrieval lanes -- not a partial agreement.
        other = by_id.get(slot.get("slot_id"))
        if other is None:
            identity = False
            break
        if any(other.get(name) != slot.get(name) for name in DEMAND_FIELDS):
            identity = False
            break
    return {
        "plan_exact": json.dumps(sample, sort_keys=True) == json.dumps(
            reference, sort_keys=True),
        "slot_count": len(left) == len(right),
        "slot_identity": identity and len(left) == len(right),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=BENCH_DIR / "canonical-eval-v1.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3,
                        help="Planner calls per question. The FIRST is pinned; the rest "
                             "measure stability and are never selected on.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    import run_p1_2_dual_track_benchmark as runner

    runner._load_deployment_env()

    from src.runtime.trusted_v2_production import _build_supervisor

    environ = dict(os.environ)
    supervisor = _build_supervisor(environ)

    questions = _load_jsonl(args.eval_set)
    if args.limit:
        questions = questions[: args.limit]

    print("=" * 78)
    print("E1-0 -- PIN THE PLANNER")
    print("=" * 78)
    print(f"  questions        {len(questions)}")
    print(f"  samples each     {args.samples}   (the FIRST is pinned)")
    print(f"  provider         {environ.get('V2_SUPERVISOR_PROVIDER', 'bailian')}")
    print(f"  model            {environ.get('V2_SUPERVISOR_MODEL', '<unset>')}")
    print(f"  temperature      {environ.get('V2_SUPERVISOR_TEMPERATURE', '0.0')} "
          f"(0.0 is not a determinism guarantee for a remote model)")
    print()
    print("  planning ...", flush=True)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    tally: collections.Counter = collections.Counter()
    per_stratum: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    failures: list[dict[str, Any]] = []

    for position, question in enumerate(questions, 1):
        case_id = question["id"]
        stratum = str(question.get("stratum"))
        samples: list[dict[str, Any]] = []
        for _ in range(max(1, args.samples)):
            try:
                run = supervisor.plan(question["question"])
            except Exception as exc:  # a planner failure is data, not a crash
                failures.append({"case_id": case_id,
                                 "error": f"{type(exc).__name__}: {exc}"})
                break
            # `plan()` returns a SupervisorRun, not a plan: `plan_valid` is the
            # validation verdict and `plan` is None when the provider failed.
            # Reading the run as though it were the plan yields an empty slot
            # list that looks like a planner returning nothing.
            if run.plan is None or not run.plan_valid:
                failures.append({"case_id": case_id,
                                 "error": run.error or "plan_invalid",
                                 "plan_valid": bool(run.plan_valid)})
                break
            samples.append(run.plan.to_dict())
        if not samples:
            tally["plan_failed"] += 1
            per_stratum[stratum]["plan_failed"] += 1
            rows.append({"case_id": case_id, "stratum": stratum, "plan": None,
                         "samples": [], "stability": None})
            continue

        reference = samples[0]
        verdicts = [_compare(sample, reference) for sample in samples[1:]]
        stability = {
            key: (sum(1 for v in verdicts if v[key]) / len(verdicts)) if verdicts else 1.0
            for key in ("plan_exact", "slot_count", "slot_identity")
        }
        tally["planned"] += 1
        tally["slots"] += len(reference.get("required_slots") or [])
        per_stratum[stratum]["planned"] += 1
        for key, value in stability.items():
            if value == 1.0:
                tally[f"stable_{key}"] += 1
                per_stratum[stratum][f"stable_{key}"] += 1

        rows.append({
            "case_id": case_id,
            "stratum": stratum,
            "question": question["question"],
            "plan": reference,
            "plan_dict_version": "supervisor-plan-to_dict/1",
            "stability": stability,
            "samples": samples if args.samples > 1 else [],
        })
        if position % 20 == 0 or position == len(questions):
            print(f"    [{position:>3}/{len(questions)}] "
                  f"{time.perf_counter() - started:.0f}s", flush=True)

    planned = tally["planned"]
    print()
    print("=" * 78)
    print("PLANNER STABILITY (the first sample is pinned regardless)")
    print("=" * 78)
    if args.samples < 2:
        print("    only one sample was taken; stability is not measured")
    else:
        for key in ("plan_exact", "slot_count", "slot_identity"):
            stable = tally[f"stable_{key}"]
            print(f"    {key:16} stable in {stable}/{planned}"
                  f"   ({stable / max(1, planned):.1%})")
    print()
    print(f"    required slots total   {tally['slots']}")
    print(f"    mean slots per case    {tally['slots'] / max(1, planned):.2f}")
    slot_histogram = collections.Counter(
        len(row["plan"].get("required_slots") or []) for row in rows if row.get("plan"))
    print(f"    slot count histogram   {dict(sorted(slot_histogram.items()))}")
    print()

    print("  per stratum, slot_count stability")
    for stratum in sorted(per_stratum):
        entry = per_stratum[stratum]
        print(f"    {stratum:26} planned {entry['planned']:>3}   "
              f"slot_count stable {entry['stable_slot_count']:>3}")
    print()
    if failures:
        print(f"  planner failures: {len(failures)}")
        for failure in failures[:5]:
            print(f"    {failure['case_id']}: {failure['error'][:110]}")
        print()

    print("  A low stability number is a finding about the pipeline, not a reason to")
    print("  re-plan: every arm replays the FIRST sample, so the arms are comparable")
    print("  even where the planner is not reproducible.")
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pinned = args.out_dir / "pinned-plans-v1.jsonl"
    with pinned.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "artifact": "pinned-plans-v1",
        "note": "The FIRST planner sample per case is pinned. Later samples measure "
                "stability and are never selected on.",
        "questions": len(questions),
        "samples_per_question": args.samples,
        "planned": planned,
        "plan_failed": tally["plan_failed"],
        "required_slots_total": tally["slots"],
        "mean_slots_per_case": round(tally["slots"] / max(1, planned), 4),
        "slot_count_histogram": {str(k): v for k, v in sorted(slot_histogram.items())},
        "stability": {
            key: {"stable": tally[f"stable_{key}"], "of": planned,
                  "rate": round(tally[f"stable_{key}"] / max(1, planned), 6)}
            for key in ("plan_exact", "slot_count", "slot_identity")},
        "per_stratum": {k: dict(v) for k, v in per_stratum.items()},
        "failures": failures,
        "provider": {k: environ.get(k) for k in
                     ("V2_SUPERVISOR_PROVIDER", "V2_SUPERVISOR_MODEL",
                      "V2_SUPERVISOR_TEMPERATURE", "V2_SUPERVISOR_BASE_URL")},
    }
    (args.out_dir / "pinned-plans-v1.manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"  written to {pinned}")
    print(f"  written to {args.out_dir / 'pinned-plans-v1.manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
