"""Run the real runtime on a handful of cases and print everything it said.

The E1 attribution harness records the fields it knew to ask for, and an
exception raised inside a capability is not one of them: `CAPABILITY_EXCEPTION`
and `GENERATION_EXCEPTION` reach the case row as a reason code with the detail
left behind in `runtime_metadata`.  A reason code without its detail is a
finding that cannot be acted on, so this prints the whole outcome for named
cases -- same pinned plan, same store, same everything, just not summarised.

  python probe_outcome_detail.py --pinned <pinned.jsonl> --v2-fact-store <store> \\
      tv2f01-s3-compare-002 tv2f01-s3-rank-004
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _path in (str(_BACKEND_DIR), str(_BACKEND_DIR / "scripts" / "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

BENCH_DIR = _BACKEND_DIR / "benchmarks" / "tv2_canonical_v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_ids", nargs="+")
    parser.add_argument("--eval-set", type=Path,
                        default=BENCH_DIR / "canonical-eval-v1.jsonl")
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tracebacks", action="store_true",
                        help="wrap the generation and evaluation capabilities so the "
                             "exception a reason code stands in for is printed. "
                             "Diagnostic only: the exception is re-raised unchanged, "
                             "so the run still fails exactly as it did.")
    parser.add_argument("--evidence", action="store_true",
                        help="print the bound-evidence semantic check when it "
                             "refuses, which is the one refusal the trace drops")
    parser.add_argument("--gold", type=Path, default=BENCH_DIR / "gold-evidence-v1.jsonl")
    parser.add_argument("--ixbrl-fact-store", type=Path, default=Path(
        "/disk/qh/nano-finrag/data/trusted-v2/fact-store/"
        "financial-facts-ixbrl-v1.jsonl"))
    parser.add_argument("--check-pool", action="store_true",
                        help="resolve this case's gold fact ids to candidate keys and "
                             "report whether each one is in the pool retrieval built. "
                             "`gold_in_pool` is 0 for a relation question by "
                             "construction, so the harness cannot answer this and the "
                             "question is exactly whether the plan's entities were "
                             "retrieved at all.")
    args = parser.parse_args(argv)

    import run_p1_2_dual_track_benchmark as runner

    runner._load_deployment_env()

    import run_e1_attribution as harness

    from src.runtime.query_lifecycle import QueryExecutionService
    from src.runtime.runtime_contract import FinancialQueryRequest
    from src.runtime.trusted_v2_production import (
        _cached_resources,
        build_trusted_v2_runtime_for_request,
    )

    def _load(path: Path) -> list[dict]:
        return [json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    questions = {row["id"]: row for row in _load(args.eval_set)}
    pinned = {row["case_id"]: row for row in _load(args.pinned)}
    plans = {row["question"]: row["plan"] for row in pinned.values()
             if row.get("plan")}

    resources = dataclasses.replace(
        _cached_resources(dict(os.environ)),
        supervisor=harness.make_replay_supervisor(plans))

    if args.tracebacks:
        import traceback

        from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability

        def _wrap(cls, name):
            original = getattr(cls, name)

            def wrapper(self, *call_args, **call_kwargs):
                try:
                    return original(self, *call_args, **call_kwargs)
                except Exception:
                    print("\n!!! %s.%s raised:" % (cls.__name__, name))
                    traceback.print_exc()
                    raise

            setattr(cls, name, wrapper)

        _wrap(TrustedV2GenerationCapability, "generate")

        for module_name, class_name in (
            ("src.runtime.trusted_v2_binder", "SemanticEvidenceEvaluationCapability"),
            ("src.runtime.trusted_v2_calculation", "DeterministicCalculationCapability"),
        ):
            module = __import__(module_name, fromlist=[class_name])
            capability_cls = getattr(module, class_name)
            for method in ("evaluate", "compute", "__call__"):
                if method in capability_cls.__dict__:
                    _wrap(capability_cls, method)

    if args.evidence:
        # The binder computes this and keeps it in its own round trace, which
        # never reaches the coordinator's -- so a case refused by the *second*
        # firewall records `QUERY_EVIDENCE_SEMANTIC_MISMATCH` and nothing about
        # which fact or which field caused it.  Wrapping the function is the
        # only way to see it without changing what production traces.
        import src.runtime.trusted_v2_binder as binder_module

        original_check = binder_module.align_bound_evidence_to_query

        def checking(*call_args, **call_kwargs):
            result = original_check(*call_args, **call_kwargs)
            if not result.allowed:
                print("\n!!! bound-evidence check refused:")
                print("    %s" % json.dumps(result.to_dict(), ensure_ascii=False,
                                             default=str)[:1200])
            return result

        binder_module.align_bound_evidence_to_query = checking

    report: dict = {}

    pool_resolver = None
    gold_by_id: dict = {}
    if args.check_pool:
        import run_nf_v3_retrieval_benchmark as bench

        from run_tv2_canonical_benchmark import FactStoreGroundingIndex

        gold_by_id = {row["id"]: row for row in _load(args.gold)}
        index_keys = {key for (_lane, key) in resources.index_reader._candidate_to_view}
        pool_resolver = bench.GoldResolver(
            FactStoreGroundingIndex(resources.fact_store.path),
            bench.load_ixbrl_candidate_keys(args.ixbrl_fact_store),
            index_keys)
    for case_id in args.case_ids:
        question = questions[case_id]
        request = FinancialQueryRequest(
            request_id=f"probe-{case_id}",
            user_id="probe",
            session_id=f"__stateless__:probe-{case_id}",
            original_query=question["question"],
            standalone_query=question["question"],
            query_as_resolved=True,
            conversation_metadata={},
            request_metadata={"document_names": [], "n_results": 5,
                              "conversation_history": None,
                              "memory_profile": None},
        )
        runtime = build_trusted_v2_runtime_for_request(
            None, request, resources=resources)
        result = asyncio.run(QueryExecutionService(runtime).execute(request))

        metadata = getattr(result, "debug_metadata", None) or {}
        trace = metadata.get("trace") or {}
        if pool_resolver is not None:
            gold = gold_by_id.get(case_id) or {}
            pool: list[str] = []
            for round_ids in (trace.get("candidate_ids_per_round") or []):
                pool.extend(str(item) for item in (round_ids or []))
            wanted = []
            for raw in (gold.get("fact_ids") or []):
                resolved = pool_resolver.resolve(
                    str(raw), document_id=question.get("document_id"),
                    metric=gold.get("metric"), period=gold.get("period"))
                wanted.append((str(raw), resolved.candidate_key, resolved.indexed))
            print("  -- gold vs pool  (pool=%d)" % len(set(pool)))
            for raw, key, indexed in wanted:
                print("     %-50s indexed=%-5s in_pool=%s"
                      % (raw[:50], indexed, key in set(pool) if key else None))
            print("     plan slots   %s"
                  % [s.get("slot_id") for s in
                     ((pinned.get(case_id) or {}).get("plan") or {}).get(
                         "required_slots") or []])
            print("     slot queries %s"
                  % (trace.get("source_branch_metadata", {}) or {}).get(
                      "slot_query_variants", "n/a"))
            report.setdefault(case_id, {})["gold_vs_pool"] = [
                {"gold": raw, "candidate_key": key, "indexed": indexed,
                 "in_pool": (key in set(pool)) if key else None}
                for raw, key, indexed in wanted]
        print("=" * 78)
        print(case_id, "|", question["question"][:90])
        print("=" * 78)
        print("  release_status  %s" % getattr(result, "release_status", None))
        print("  reason_codes    %s" % (getattr(result, "reason_codes", None),))
        for key in ("terminal_state", "retrieval_rounds", "binder_status_per_round",
                    "missing_slot_ids", "conflict_ids", "validation_passed"):
            print("  %-16s%s" % (key, str(trace.get(key))[:160]))
        print("  -- runtime_metadata")
        print(json.dumps(metadata, ensure_ascii=False, indent=2,
                         default=str)[:4000])
        report[case_id] = {"trace": trace, "metadata": metadata,
                           "reason_codes": getattr(result, "reason_codes", None)}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                       default=str) + "\n",
                            encoding="utf-8", newline="\n")
        print("written to %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
