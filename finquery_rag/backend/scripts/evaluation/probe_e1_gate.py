"""E1 probe 2: how many cases die at the semantic gate, before retrieval runs?

The first probe ran one factual case and it never retrieved anything: the trace
shows `terminal_state='PLAN'`, `reason_codes=['QUERY_PLAN_SEMANTIC_MISMATCH']`
and `retrieval_rounds` empty.  The production builder sets
`unknown_semantic_policy=STRICT_DIRECT_FACT`, so a question whose metric the
alignment gate does not recognise is refused at PLAN -- upstream of retrieval,
Binder and validator alike.

If that is common, an attribution experiment that starts the funnel at retrieval
would report empty funnels and attribute nothing, because for those cases no
retrieval was ever attempted.  The gate is a stage, and it needs its own line.

This runs N cases through the real in-process path and tallies where each one
stopped.  It does NOT pass `alignment_override` -- the point is to measure the
gate as it ships.  The builder's own comment notes that seam exists for a
benchmark that wants the chain *past* the gate, which is a different measurement
and a later decision.

  TRUSTED_V2_SPECIALIST_DEVICE=cpu .venv/bin/python probe_e1_gate.py --limit 30
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

BENCH_DIR = _BACKEND_DIR / "benchmarks" / "tv2_canonical_v1"


def _run_one(case, question, resources, factory=None):
    from src.runtime.query_lifecycle import QueryExecutionService
    from src.runtime.runtime_contract import FinancialQueryRequest
    from src.runtime.trusted_v2_production import build_trusted_v2_runtime_for_request

    request = FinancialQueryRequest(
        request_id=f"e1-gate-{case}",
        user_id="e1",
        session_id=f"__stateless__:e1-gate-{case}",
        original_query=question["question"],
        standalone_query=question["question"],
        query_as_resolved=True,
        conversation_metadata={},
        request_metadata={
            # NOT scoped to the case's document.  The published benchmark posts
            # only {"question", "session_id"}, and in v2 mode
            # `_resolve_query_document_names_for_user` returns [] for a None
            # request -- so the shipped run searched the whole index.  Passing
            # the case's own document here scopes retrieval to one filing, which
            # is a different experiment and was the first confound between this
            # harness and the baseline it is meant to reproduce.
            "document_names": [],
            "n_results": 5,
            "conversation_history": None,
            "memory_profile": None,
        },
    )
    runtime = build_trusted_v2_runtime_for_request(
        None, request, resources=resources, retriever_factory=factory)
    return asyncio.run(QueryExecutionService(runtime).execute(request))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=BENCH_DIR / "canonical-eval-v1.jsonl")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args(argv)

    import run_p1_2_dual_track_benchmark as runner

    runner._load_deployment_env()

    from src.runtime.trusted_v2_production import _cached_resources

    questions = [json.loads(line)
                 for line in args.eval_set.read_text(encoding="utf-8").splitlines()
                 if line.strip()][: args.limit]

    print("=" * 78)
    print("E1 PROBE 2 -- WHERE DOES EACH CASE STOP?")
    print("=" * 78)
    print(f"  cases  {len(questions)}   (production semantic policy, no override)")
    print(f"  device {os.environ.get('TRUSTED_V2_SPECIALIST_DEVICE', '<unset>')}")
    print()

    resources = _cached_resources(dict(os.environ))

    by_stage: collections.Counter = collections.Counter()
    by_reason: collections.Counter = collections.Counter()
    by_stratum: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    rows: list[dict] = []

    for position, question in enumerate(questions, 1):
        case = question["id"]
        try:
            result = _run_one(case, question, resources)
        except Exception as exc:
            by_stage["EXCEPTION"] += 1
            by_reason[f"{type(exc).__name__}"] += 1
            rows.append({"case_id": case, "stage": "EXCEPTION",
                         "error": f"{type(exc).__name__}: {exc}"})
            continue
        trace = (getattr(result, "debug_metadata", None) or {}).get("trace") or {}
        stage = str(trace.get("terminal_state") or result.status or "UNKNOWN")
        reasons = [str(r) for r in (trace.get("reason_codes") or result.reason_codes or [])]
        by_stage[stage] += 1
        for reason in reasons:
            by_reason[reason] += 1
        by_stratum[str(question.get("stratum"))][stage] += 1
        rows.append({
            "case_id": case, "stratum": question.get("stratum"), "stage": stage,
            "reason_codes": reasons,
            "retrieval_rounds": len(trace.get("retrieval_rounds") or []),
            "candidate_ids_per_round": [len(x) for x in (trace.get("candidate_ids_per_round") or [])],
            "bound_evidence_ids": len(trace.get("bound_evidence_ids") or []),
            "release_status": trace.get("release_status") or result.release_status,
        })
        if position % 10 == 0 or position == len(questions):
            print(f"    [{position:>3}/{len(questions)}]", flush=True)

    print()
    print("  terminal stage")
    for stage, count in by_stage.most_common():
        print(f"    {count:>4}  {stage}")
    print()
    print("  reason codes")
    for reason, count in by_reason.most_common():
        print(f"    {count:>4}  {reason}")
    print()
    print("  terminal stage by stratum")
    for stratum in sorted(by_stratum):
        print(f"    {stratum:26} {dict(by_stratum[stratum])}")
    print()
    ran_retrieval = sum(1 for row in rows if row.get("retrieval_rounds"))
    print(f"  cases that reached retrieval: {ran_retrieval}/{len(rows)}")
    print(f"  cases that reached release:   "
          f"{sum(1 for row in rows if row.get('release_status') == 'RELEASED')}/{len(rows)}")
    print()

    out = Path("/tmp/e1-gate-probe.json")
    out.write_text(json.dumps({"by_stage": dict(by_stage), "by_reason": dict(by_reason),
                               "cases": rows}, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
