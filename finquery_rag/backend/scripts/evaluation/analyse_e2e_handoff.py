"""E2E handoff, stage 0: what the *existing* run already tells us, and what it can't.

The question is where the evidence dies between retrieval and release.  Most of
the funnel needs an instrumented run, but two stages are recoverable from the
committed 120-case prediction file as it stands, because each prediction carries
`evidence_ids` (the facts the Binder admitted) and `release_status`:

    gold admitted by the Binder   was the benchmark's fact among evidence_ids
    released                      did the runtime answer

which is enough for the one number that matters most:

    P(released | the gold fact was admitted)

If the gold is admitted and the runtime still refuses, the loss is downstream of
evidence acquisition and no retrieval change can reach it.  If the gold is
admitted rarely, the loss is upstream.  Those two worlds need different work and
this separates them before any experiment is built.

**What this cannot see, and says so rather than leaving it blank.**

  * the retrieval pool -- which candidates were *offered* to the Binder.  The
    adapter drops the pool before it leaves the process (only admitted evidence
    reaches the response), so "gold in retrieval top-20" is not in this file at
    all, in either direction.
  * per-candidate rejection reasons -- computed and discarded in-process.
  * the required-slot set the plan actually asked for -- the plan is an LLM
    sample and is not recorded here.

So this is a floor, not the funnel.  It exists because it costs nothing and it
can already rule out one of the two worlds.  The stages it cannot see are named
so the gaps are not mistaken for zeros.

  python analyse_e2e_handoff.py --predictions <jsonl> --gold <jsonl> \\
      --fact-store <jsonl> --out <dir>
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

#: Stages recoverable from the prediction file, in pipeline order.
OBSERVABLE = ("gold admitted by the Binder", "released")

#: Stages the file cannot answer.  Listed so a reader sees the gaps explicitly
#: rather than inferring zero from absence.
NOT_OBSERVABLE = (
    "gold in the retrieval pool (the pool never leaves the process)",
    "per-candidate Binder rejection reasons (computed, then discarded)",
    "the required-slot set the plan asked for (LLM sample, not recorded)",
    "validator failure reason codes (in-process trace only)",
)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--fact-store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    predictions = {row["id"]: row for row in _load(args.predictions)}
    gold = {row["id"]: row for row in _load(args.gold)}
    aliases = scorer.load_alias_map(args.fact_store)

    tally = collections.Counter()
    by_stratum: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    rows: list[dict] = []

    for case_id, prediction in predictions.items():
        record = gold.get(case_id)
        if record is None:
            continue
        stratum = str(prediction.get("stratum"))
        must_refuse = bool(record.get("expected_failure_reason"))
        if must_refuse:
            continue
        tally["answerable"] += 1
        by_stratum[stratum]["answerable"] += 1

        wanted = {scorer.resolve(str(f), aliases) for f in record.get("fact_ids") or []}
        admitted = {scorer.resolve(str(e), aliases)
                    for e in prediction.get("evidence_ids") or []}
        released = prediction.get("release_status") == "RELEASED"

        # The two stores have separate id namespaces and the evidence ids in
        # this file are all `v2fact`.  A case whose gold is `ixbrl:` cannot be
        # compared against them at all -- resolving both sides through the v2
        # alias map silently leaves them disjoint, and every such case would
        # then read as "gold never admitted", which is a fact about identifiers
        # and not about the pipeline.  Measured, not assumed: the cross-entity
        # stratum is reported separately rather than folded into a zero.
        comparable = bool(wanted) and all(
            str(f).startswith("v2fact:") for f in record.get("fact_ids") or [])

        tally["evidence_n"] += len(admitted)
        # Release is a fact about the runtime, not about identifiers, so it is
        # counted for every answerable case -- including the ones whose gold
        # cannot be compared.  Only the *admission* comparison is id-space
        # bound; folding a cross-entity release out of the release count would
        # understate the release rate for a reason that has nothing to do with
        # the release.
        if released:
            tally["released"] += 1
            by_stratum[stratum]["released"] += 1
        if not comparable:
            tally["not_comparable"] += 1
            by_stratum[stratum]["not_comparable"] += 1
            if released:
                tally["released_not_comparable"] += 1
            rows.append({
                "case_id": case_id, "stratum": stratum,
                "comparable": False, "released": released,
                "gold_ids": len(wanted), "admitted_evidence": len(admitted),
                "reason_codes": prediction.get("reason_codes"),
                "status": prediction.get("status"),
            })
            continue

        hit = len(wanted & admitted)
        any_admitted = hit > 0
        complete = bool(wanted) and wanted <= admitted

        if any_admitted:
            tally["gold_admitted"] += 1
            by_stratum[stratum]["gold_admitted"] += 1
        if complete:
            tally["gold_admitted_complete"] += 1
        if released and any_admitted:
            tally["released_and_admitted"] += 1
        if released and not any_admitted:
            tally["released_without_gold"] += 1
        if not released and any_admitted:
            tally["admitted_but_refused"] += 1

        rows.append({
            "case_id": case_id, "stratum": stratum,
            "gold_ids": len(wanted), "admitted_gold": hit,
            "admitted_evidence": len(admitted), "released": released,
            "reason_codes": prediction.get("reason_codes"),
            "status": prediction.get("status"),
        })

    answerable = tally["answerable"]
    comparable = answerable - tally["not_comparable"]
    admitted = tally["gold_admitted"]
    released = tally["released"]

    print("=" * 78)
    print("E2E HANDOFF -- STAGE 0, FROM THE EXISTING PREDICTION FILE")
    print("=" * 78)
    print(f"  answerable cases              {answerable}")
    print(f"    comparable in this id space {comparable}"
          f"   (the other {tally['not_comparable']} are cross-entity: gold is `ixbrl:`,")
    print("                                 evidence ids are `v2fact:`, no comparison")
    print("                                 exists and none is reported as a miss)")
    print()
    print(f"  gold admitted by the Binder   {admitted}"
          f"   ({admitted / max(1, comparable):.1%} of comparable)")
    print(f"  released                      {released}"
          f"   ({released / max(1, answerable):.1%} of answerable)")
    print(f"  released AND gold admitted    {tally['released_and_admitted']}")
    print(f"  released WITHOUT gold admitted{tally['released_without_gold']:>4}"
          f"   <- a correctness question, not a coverage one")
    print(f"  admitted but refused          {tally['admitted_but_refused']}"
          f"   <- the gold was in hand and the runtime still refused")
    print()
    print(f"  P(released | gold admitted)   "
          f"{tally['released_and_admitted'] / max(1, admitted):.3f}"
          f"   ({tally['released_and_admitted']}/{admitted})")
    print(f"  evidence items per case, mean "
          f"{tally['evidence_n'] / max(1, comparable):.2f}"
          f"   <- what actually reached the release stage")
    print()

    print("  per stratum")
    print(f"    {'stratum':26}{'answerable':>11}{'admitted':>10}{'released':>10}")
    for stratum in sorted(by_stratum):
        entry = by_stratum[stratum]
        print(f"    {stratum:26}{entry['answerable']:>11}{entry['gold_admitted']:>10}"
              f"{entry['released']:>10}")
    print()

    print("  --- not visible in this artifact, so not reported as zero ---")
    for item in NOT_OBSERVABLE:
        print(f"    {item}")
    print()

    # The decision this stage exists to make.
    print("  READING")
    if admitted <= 2:
        print("    The gold is almost never in the admitted set, so the loss is")
        print("    upstream of release and a retrieval-side experiment can reach it.")
        print("    But it is also not separable here from the plan and the Binder:")
        print("    this file shows WHAT was admitted, never what was offered.")
    elif tally["released_and_admitted"] < admitted:
        print(f"    Of {admitted} cases whose gold reached the Binder, "
              f"{tally['released_and_admitted']} released.")
        print("    The remainder died at or after binding, which no retrieval change")
        print("    can reach. Where exactly is the instrumented run's job.")
    else:
        print("    Every case whose gold was admitted released. The constraint is")
        print("    admission, not release.")
    print()

    report = {
        "analysis": "E2E-handoff-stage-0",
        "counts": dict(tally),
        "by_stratum": {k: dict(v) for k, v in by_stratum.items()},
        "observable_stages": list(OBSERVABLE),
        "not_observable": list(NOT_OBSERVABLE),
        "p_released_given_admitted": round(
            tally["released_and_admitted"] / max(1, admitted), 6),
        "cases": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "handoff-stage0.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"  written to {args.out / 'handoff-stage0.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
