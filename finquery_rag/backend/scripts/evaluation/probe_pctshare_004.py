"""What the binder had to choose from on pctshare-004, and what it chose.

`004` fails with MISSING on `s1`, whose coordinate is `Impact of the State Aid
Decision` / Apple / FY2025 -- the *same* coordinate `003` binds successfully as
its `s2`, in the same run, under the same identity resolution.  So the identity
is no longer what blocks it, and there are two candidates left: the fact never
reached the packet, or it reached it and the binder did not pick it.

Those want opposite fixes, so this separates them before anything is changed.
It wraps the evidence evaluator and records, per round, the metrics actually
present in the packet the binder was handed, and the binding the provider
returned for it.

  python probe_pctshare_004.py --case tv2f01-s2-pctshare-004 --runs 3 --out-dir <dir>
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

PACKETS: list[dict[str, Any]] = []


def _install_probe() -> None:
    import src.runtime.trusted_v2_coordinator as coordinator

    inner = coordinator._EvaluatorAdapter.evaluate

    def patched(self: Any, state: Any) -> Any:
        packets = list(getattr(state, "evidence_packets", ()) or ())
        PACKETS.append(
            {
                "packet_size": len(packets),
                "metrics": sorted(
                    {
                        str((p or {}).get("metric"))
                        for p in packets
                        if isinstance(p, dict) and p.get("metric")
                    }
                )[:60],
                "state_aid_present": [
                    {
                        "fact_id": (p or {}).get("fact_id"),
                        "metric": (p or {}).get("metric"),
                        "period": (p or {}).get("period"),
                        "value": (p or {}).get("value"),
                        "entity": (p or {}).get("entity"),
                    }
                    for p in packets
                    if isinstance(p, dict)
                    and "state aid" in str(p.get("metric") or "").casefold()
                ],
            }
        )
        return inner(self, state)

    coordinator._EvaluatorAdapter.evaluate = patched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--runs", type=int, default=3)
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
        row for row in runner._load_jsonl(base / "canonical-eval-v1.jsonl")
        if row["id"] == args.case
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
        PACKETS.clear()
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
            user_id="p1-4e-probe",
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
        row["packets"] = list(PACKETS)
        rows.append(row)

        first = PACKETS[0] if PACKETS else {}
        print(
            f"  run {index}: status={row.get('status')} "
            f"packet={first.get('packet_size')} "
            f"state_aid_rows={len(first.get('state_aid_present') or [])} "
            f"missing={row.get('binder_final_missing_slot_ids')}",
            flush=True,
        )

    out = args.out_dir / f"probe-{args.case}.jsonl"
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print()
    for row in rows:
        for packet in row["packets"]:
            print(
                f"  run {row['run']}: packet_size={packet['packet_size']} "
                f"state_aid_rows={json.dumps(packet['state_aid_present'], ensure_ascii=False)}"
            )
            print(f"            metrics={json.dumps(packet['metrics'][:24], ensure_ascii=False)}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
