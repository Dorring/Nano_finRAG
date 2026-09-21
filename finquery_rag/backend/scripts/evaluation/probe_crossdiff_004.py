"""Capture the exception behind one case's CAPABILITY_EXCEPTION, with its traceback.

The coordinator never sees the exception: ``_RetrievalTool.__call__`` catches it
from the retrieval port, appends it to ``capability_errors`` and re-raises, and
the bounded loop swallows it.  By the time a result exists the exception is a
boolean -- the list is non-empty -- so the outcome carries ``CAPABILITY_EXCEPTION``
and ``status=EXECUTION_ERROR`` with an empty ``error`` field, and the cause is
gone.

This runs the case with that one method wrapped, so the traceback is printed and
kept before it is swallowed.  It reuses the runner's own resource loading, plan
fixture and alignment override, so what it exercises is the real replay path and
not a reconstruction of it.

  python probe_crossdiff_004.py --case tv2f01-s3-crossdiff-004 --runs 10 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import traceback
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_SCRIPTS = _BACKEND_DIR / "scripts/evaluation"
for _path in (str(_BACKEND_DIR), str(_SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_p1_2_dual_track_benchmark as runner  # noqa: E402

#: Every retrieval-port exception seen, in order, with the traceback intact.
CAPTURED: list[dict[str, Any]] = []


def _install_probe() -> None:
    """Wrap the two places that catch a capability exception and swallow it.

    ``exceptions=0`` on the first pass ruled the retrieval port out: the
    coordinator fills ``capability_errors`` from two wrappers, and only one of
    them was instrumented.  This is the other -- the evidence evaluator, which
    is the binder -- and it is the one the ``binder_round_count=0`` shape points
    at anyway.
    """

    import src.runtime.trusted_v2_coordinator as coordinator

    retrieval_call = coordinator._RetrievalTool.__call__
    evaluator_evaluate = coordinator._EvaluatorAdapter.evaluate

    def _record(where: str, exc: BaseException) -> None:
        CAPTURED.append(
            {
                "where": where,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    def patched_retrieval(self: Any, query: str, state: Any) -> Any:
        try:
            return retrieval_call(self, query, state)
        except Exception as exc:  # noqa: BLE001 - the point is to record it
            _record("retrieval_port", exc)
            raise

    def patched_evaluate(self: Any, state: Any) -> Any:
        try:
            return evaluator_evaluate(self, state)
        except Exception as exc:  # noqa: BLE001 - the point is to record it
            _record("evidence_evaluator", exc)
            raise

    coordinator._RetrievalTool.__call__ = patched_retrieval
    coordinator._EvaluatorAdapter.evaluate = patched_evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--fixtures", type=Path, default=runner.DEFAULT_FIXTURES)
    parser.add_argument("--eval-set", type=Path, default=None)
    parser.add_argument("--gold-evidence", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
    from src.runtime import FinancialQueryRequest, V2ExecutionRequest
    from src.runtime.trusted_v2_production import (
        _load_resources,
        build_trusted_v2_runtime_for_request as build_runtime,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    questions = [row for row in runner._load_jsonl(
        args.eval_set or _BACKEND_DIR / "benchmarks/tv2_canonical_v1/canonical-eval-v1.jsonl"
    ) if row["id"] == args.case]
    if not questions:
        print(f"case not in the eval set: {args.case}", file=sys.stderr)
        return 2
    question = questions[0]

    fixtures = {row["id"]: row for row in runner._load_jsonl(args.fixtures) if row["id"] == args.case}
    if not fixtures:
        print(f"case has no plan fixture: {args.case}", file=sys.stderr)
        return 2

    _install_probe()

    import os

    runner._load_deployment_env()
    resources = _load_resources(dict(os.environ))

    rows: list[dict[str, Any]] = []
    for index in range(1, args.runs + 1):
        CAPTURED.clear()
        fixture = fixtures[args.case]
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
            request_id=f"probe-{args.case}-{index}",
            user_id="p1-4c-probe",
            session_id=f"probe-{args.case}-{index}",
            original_query=question["question"],
            standalone_query=question["question"],
            query_as_resolved=True,
        )

        replay_resources = dataclasses.replace(
            resources,
            supervisor=SupervisorService(
                DeterministicFallbackProvider({question["question"]: plan})
            ),
        )

        import time

        started = time.perf_counter()
        runtime = None
        try:
            runtime = build_runtime(
                None, request, resources=replay_resources, alignment_override=override
            )
            outcome = asyncio.run(
                runtime.coordinator.execute(V2ExecutionRequest.from_financial_request(request))
            )
            error = None
        except Exception as exc:  # noqa: BLE001
            outcome, error = None, f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000.0

        row = runner._replay_row(
            question=question,
            outcome=outcome,
            error=error,
            latency_ms=latency_ms,
            computed=computed,
            overridden=override is not None,
            runtime=runtime if outcome is not None else None,
        )
        row["run"] = index
        row["captured_exceptions"] = list(CAPTURED)
        rows.append(row)

        print(
            f"  run {index:>2}  status={str(row.get('status')):<16} "
            f"binder_rounds={row.get('binder_round_count')} "
            f"exceptions={len(CAPTURED)} {CAPTURED[0]['exception_type'] if CAPTURED else ''}",
            flush=True,
        )

    out = args.out_dir / f"probe-{args.case}.jsonl"
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print()
    summary = {}
    for row in rows:
        key = (str(row.get("status")), len(row.get("captured_exceptions") or []))
        summary[key] = summary.get(key, 0) + 1
    print("status x exception-count:", summary)

    kinds = {}
    for row in rows:
        for item in row.get("captured_exceptions") or []:
            kinds[item["exception_type"]] = kinds.get(item["exception_type"], 0) + 1
    print("captured exception types:", kinds)

    for row in rows:
        for item in row.get("captured_exceptions") or []:
            print()
            print(f"--- run {row['run']}  where={item['where']} ---")
            print(item["traceback"])
            break
        else:
            continue
        break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
