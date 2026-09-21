"""P1.5-0b: at which hop does the gold fact disappear?

`pctshare-001` and `004` established that the bound-evidence gate was the defect
there, and the P1.4F fix closed them.  Cross-entity did not move: all nine of
its deterministic-ceiling-RESOLVABLE questions fail with *every* slot missing,
identically across three runs, while 157/157 of their gold candidates sit in
`candidate-metadata.sqlite`.

So the loss is between the index and the Binder's view -- and "retrieval is
bad" does not yet follow, because there are two hops in between:

    candidate index
      -> state.evidence_packets      retrieval: formulation, ranking, top-K
      -> _facts_for_binding output   view: eligibility, metric prefilter, scope
      -> a binding                   the model's choice

This records, per slot per run, a boolean at each hop plus the gold's rank
inside the packet, which separates the four:

    slot | index | topK | rank | view | bound

`index` is not re-measured here: it is a read-only fact about the sqlite index
(157/157) and asking it again per run would only invite the two to disagree.

The rank matters as much as the boolean.  `rank=43, K=40` is a top-K budget
problem, `rank=3` with an empty view is a filter problem, and `rank=None` is
query formulation.  A boolean alone cannot tell those apart, which is the whole
reason this is worth a second probe rather than a re-run of the last one.

Instrumentation is call-through: every wrapper returns exactly what it was
given, and nothing is monkeypatched that decides anything.  That is necessary
and not sufficient -- a traced run is only trusted once it reproduces the
untraced outcome distribution, so run both and compare before believing a row.

  python probe_evidence_reachability.py --cases tv2f01-s3-compare-006 ... --runs 3 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_SCRIPTS = _BACKEND_DIR / "scripts/evaluation"
for _path in (str(_BACKEND_DIR), str(_SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_p1_2_dual_track_benchmark as runner  # noqa: E402

#: One entry per `_facts_for_binding` call, in call order.
HOP: list[dict[str, Any]] = []

#: The identifier fields a candidate may carry.  A gold id is matched against
#: all of them rather than one: `_facts` promotes metadata and keeps whichever
#: of `fact_id`/`candidate_key`/`evidence_id` the packet happened to use, so
#: testing one spelling is how a probe reports a false negative -- the mistake
#: that made an earlier read of `pctshare-004` conclude the fact was absent.
_ID_KEYS = ("candidate_key", "candidate_id", "fact_id", "evidence_id")


def _ids(fact: Any) -> set[str]:
    if not isinstance(fact, dict):
        return set()
    return {str(fact[k]) for k in _ID_KEYS if fact.get(k)}


def _install() -> None:
    """Wrap the view builder, calling through unchanged."""

    from src.runtime.trusted_v2_binder import SemanticEvidenceEvaluationCapability

    original = SemanticEvidenceEvaluationCapability._facts_for_binding

    def patched(state: Any, plan: Any, facts: Any) -> Any:
        view = original(state, plan, facts)
        packet = tuple(getattr(state, "evidence_packets", None) or ())
        HOP.append(
            {
                "slots": [
                    {
                        "slot_id": slot.slot_id,
                        "metric": slot.metric,
                        "period": slot.period,
                        "entity": getattr(slot, "entity", None),
                    }
                    for slot in plan.required_slots
                ],
                # The packet as retrieval produced it, with positions kept so a
                # gold's rank is its index here.
                "packet": [sorted(_ids(f)) for f in packet],
                "pre_view": [sorted(_ids(f)) for f in facts],
                "view": [sorted(_ids(f)) for f in view],
            }
        )
        return view

    SemanticEvidenceEvaluationCapability._facts_for_binding = staticmethod(patched)


def _bound_by_slot(runtime: Any) -> dict[str, list[str]]:
    """What the Binder actually bound, per slot, read off the capability.

    Read after the fact rather than wrapped around `bind`: the binding is a
    returned value, and recording it needs no interception at all.
    """

    bound: dict[str, list[str]] = {}
    try:
        evaluator = runtime.coordinator.capabilities.evidence_evaluator
    except Exception:  # noqa: BLE001 - diagnostic only
        return bound
    run = getattr(evaluator, "last_run", None)
    binding = getattr(run, "binding", None)
    for slot_id, fact_ids in (getattr(binding, "slot_bindings", None) or {}).items():
        bound[str(slot_id)] = [str(item) for item in fact_ids]
    return bound


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--fixtures", type=Path, default=runner.DEFAULT_FIXTURES)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = _BACKEND_DIR / "benchmarks/tv2_canonical_v1"
    questions = {row["id"]: row for row in runner._load_jsonl(base / "canonical-eval-v1.jsonl")}
    fixtures = {row["id"]: row for row in runner._load_jsonl(args.fixtures)}
    gold = {row["id"]: row for row in runner._load_jsonl(base / "gold-evidence-v1.jsonl")}

    import os

    from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
    from src.runtime import FinancialQueryRequest, V2ExecutionRequest
    from src.runtime.trusted_v2_production import (
        _load_resources,
        build_trusted_v2_runtime_for_request as build_runtime,
    )

    _install()
    runner._load_deployment_env()
    resources = _load_resources(dict(os.environ))

    rows: list[dict[str, Any]] = []
    for index in range(1, args.runs + 1):
        for case in args.cases:
            question, fixture = questions[case], fixtures[case]
            HOP.clear()
            plan = runner._plan_from_fixture(fixture)
            computed = runner._alignment_for(question["question"], plan)
            override = None
            if not computed.allowed:
                override = dataclasses.replace(
                    computed,
                    status=computed.status.__class__.ALIGNED,
                    mismatches=(),
                    ambiguous_query_fields=(),
                )
            request = FinancialQueryRequest(
                request_id=f"reach-{case}-{index}",
                user_id="p1-5-0b",
                session_id=f"reach-{case}-{index}",
                original_query=question["question"],
                standalone_query=question["question"],
                query_as_resolved=True,
            )
            replay = dataclasses.replace(
                resources,
                supervisor=SupervisorService(
                    DeterministicFallbackProvider({question["question"]: plan})
                ),
            )
            started = time.perf_counter()
            runtime = None
            try:
                runtime = build_runtime(
                    None, request, resources=replay, alignment_override=override
                )
                outcome = asyncio.run(
                    runtime.coordinator.execute(
                        V2ExecutionRequest.from_financial_request(request)
                    )
                )
                error = None
            except Exception as exc:  # noqa: BLE001
                outcome, error = None, f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - started) * 1000.0

            hop = HOP[-1] if HOP else {"slots": [], "packet": [], "pre_view": [], "view": []}
            bound = _bound_by_slot(runtime) if runtime is not None else {}
            gold_ids = [str(x) for x in (gold.get(case, {}).get("fact_ids") or [])]

            rows.append(
                {
                    "run": index,
                    "case": case,
                    "error": error,
                    "latency_ms": round(latency_ms, 1),
                    "hop": hop,
                    "bound_by_slot": bound,
                    "gold_fact_ids": gold_ids,
                    "binder_final_status": (
                        None if outcome is None else getattr(outcome, "binder_final_status", None)
                    ),
                    "status": None if outcome is None else str(getattr(outcome, "status", "")),
                }
            )
            print(f"  run{index} {case}: packet={len(hop['packet'])} view={len(hop['view'])} bound={bound}", flush=True)

    (args.out_dir / "reachability.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nwrote {args.out_dir / 'reachability.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
