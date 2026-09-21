"""P1.4F-2: freeze what the Binder was shown and what it answered, for one case.

`pctshare-001` fails with the Binder reporting `BOUND` and the bound-evidence
semantic gate then discarding everything (`QUERY_EVIDENCE_SEMANTIC_MISMATCH`).
That is a statement about the *gate*, over a binding the Binder produced, and
the run's own output cannot say what either of them was looking at.

So this freezes four layers per run, in the order they are produced, plus the
parsed binding and the gate's verdict:

    A  RequiredSlot.to_dict()           what the plan asked for
    B  BinderRequest candidate facts     what the Binder was given to choose from
    C  build_binder_messages()           the exact messages that went out
    D  provider raw_response             the exact bytes that came back

Nothing is changed.  Every wrapper calls through unchanged, so a run under this
probe is the run it would have been without it.

  python probe_binder_trace.py --case tv2f01-s2-pctshare-001 --runs 10 --out-dir <dir>
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

LAYERS: dict[str, list[Any]] = {
    "A_slots": [],
    "B_candidates": [],
    "B_view": [],
    "C_messages": [],
    "D_raw": [],
}


def _install_probe() -> None:
    """Wrap the three production seams, calling through unchanged."""

    from rag_v2.evidence import binder_service
    from rag_v2.evidence import prompt as binder_prompt
    import src.runtime.trusted_v2_binder as runtime_binder

    service_bind = binder_service.SemanticBinderService.bind
    messages_fn = binder_prompt.build_binder_messages
    facts_for_binding = runtime_binder.SemanticEvidenceEvaluationCapability._facts_for_binding

    def patched_bind(self: Any, request: Any) -> Any:
        try:
            payload = request.to_dict()
        except Exception:  # noqa: BLE001 - diagnostic only
            payload = {"unserialisable": repr(request)}
        LAYERS["A_slots"].append(payload.get("required_slots"))
        LAYERS["B_candidates"].append(
            [
                {
                    k: fact.get(k)
                    for k in ("fact_id", "candidate_id", "metric", "period", "value", "entity", "unit", "scale")
                }
                for fact in (payload.get("facts") or payload.get("candidate_facts") or [])
                if isinstance(fact, dict)
            ]
        )
        run = service_bind(self, request)
        LAYERS["D_raw"].append(getattr(run, "raw_response", None))
        return run

    def patched_messages(request: Any) -> Any:
        messages = messages_fn(request)
        LAYERS["C_messages"].append(messages)
        return messages

    def patched_facts(state: Any, plan: Any, facts: Any) -> Any:
        view = facts_for_binding(state, plan, facts)
        LAYERS["B_view"].append(
            [
                {"metric": f.get("metric"), "period": f.get("period"), "value": f.get("value")}
                for f in view
                if isinstance(f, dict)
            ]
        )
        return view

    binder_service.SemanticBinderService.bind = patched_bind
    binder_prompt.build_binder_messages = patched_messages
    runtime_binder.SemanticEvidenceEvaluationCapability._facts_for_binding = staticmethod(patched_facts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--fixtures", type=Path, default=runner.DEFAULT_FIXTURES)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
    from src.runtime import FinancialQueryRequest, V2ExecutionRequest
    from src.runtime.trusted_v2_production import (
        _load_resources,
        build_trusted_v2_runtime_for_request as build_runtime,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = _BACKEND_DIR / "benchmarks/tv2_canonical_v1"
    question = next(
        row for row in runner._load_jsonl(base / "canonical-eval-v1.jsonl") if row["id"] == args.case
    )
    fixture = next(
        row for row in runner._load_jsonl(args.fixtures) if row["id"] == args.case
    )

    _install_probe()
    runner._load_deployment_env()
    import os

    resources = _load_resources(dict(os.environ))

    rows: list[dict[str, Any]] = []
    for index in range(1, args.runs + 1):
        for key in LAYERS:
            LAYERS[key] = []
        plan = runner._plan_from_fixture(fixture)
        computed = runner._alignment_for(question["question"], plan)
        override = None
        if not computed.allowed:
            override = dataclasses.replace(
                computed, status=computed.status.__class__.ALIGNED, mismatches=(), ambiguous_query_fields=()
            )
        request = FinancialQueryRequest(
            request_id=f"trace-{args.case}-{index}",
            user_id="p1-4f2-trace",
            session_id=f"trace-{args.case}-{index}",
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
        started = time.perf_counter()
        runtime = None
        try:
            runtime = build_runtime(None, request, resources=replay_resources, alignment_override=override)
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
        row["layers"] = {key: list(value) for key, value in LAYERS.items()}
        rows.append(row)
        print(
            f"  run {index:>2}: status={row.get('status')} "
            f"binder={row.get('binder_final_status')} "
            f"reasons={row.get('reason_codes')} "
            f"candidates={len(LAYERS['B_candidates'][0]) if LAYERS['B_candidates'] else '-'} "
            f"view={len(LAYERS.get('B_view', [[]])[0])}",
            flush=True,
        )

    out = args.out_dir / f"trace-{args.case}.jsonl"
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
