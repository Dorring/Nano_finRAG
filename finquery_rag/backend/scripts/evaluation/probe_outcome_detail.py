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

    report: dict = {}
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
