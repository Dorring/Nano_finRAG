#!/usr/bin/env python3
"""P2.0: runtime performance, measured on the sealed code rather than recalled.

Every number the README quotes for latency has to come from one run on one
machine, with the machine named. This does that and nothing else -- it changes no
behaviour, reads the benchmark it is given, and writes a JSON plus a markdown
table.

    .venv/bin/python scripts/evaluation/measure_runtime_performance.py \
        --eval-set <canonical-eval-v1.jsonl> --gold <gold-evidence-v1.jsonl> \
        --fixtures <plan-fixtures-v9.jsonl> --out-dir <dir>

What it measures, and what it deliberately keeps apart
------------------------------------------------------

  alignment    `align_query_to_plan`. Pure local, no model, no network.
  retrieval    `CandidateDirectR4Policy.retrieve`. Local index + fusion + the
               pool cut. Deterministic, so it is repeated and reported cold and
               warm rather than averaged into one number.
  end-to-end   the real replay over the whole benchmark: the same code path the
               seal was measured on.
  generation   whatever the runtime itself reports for the specialist call.

**Provider latency is not local latency.** The Binder and the generator call a
remote model, so the end-to-end figure contains network and inference time that
this machine does not own. `generation` is reported separately where the runtime
exposes it, and the remainder is labelled `harness_other` rather than presented
as harness compute -- it still contains the Binder's remote calls.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

BACKEND = Path("/disk/qh/nano-finrag/finquery_rag/backend")
for _p in (str(BACKEND), str(BACKEND / "scripts/evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarise(values: list[float]) -> dict[str, Any]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0}
    return {
        "n": len(clean),
        "p50_ms": round(percentile(clean, 0.50), 2),
        "p95_ms": round(percentile(clean, 0.95), 2),
        "p99_ms": round(percentile(clean, 0.99), 2) if len(clean) >= 20 else None,
        "mean_ms": round(statistics.fmean(clean), 2),
        "max_ms": round(max(clean), 2),
    }


def environment() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }
    try:
        import subprocess

        info["cpu_model"] = subprocess.run(
            ["grep", "-m1", "model name", "/proc/cpuinfo"],
            capture_output=True, text=True,
        ).stdout.split(":", 1)[-1].strip()
        info["ram_total_kb"] = int(
            [l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal")][0].split()[1]
        )
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip().splitlines()
        info["gpus"] = smi
    except Exception:  # noqa: BLE001 - the snapshot is best-effort
        pass
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--retrieval-reps", type=int, default=3)
    parser.add_argument("--skip-replay", action="store_true", help="local timings only")
    args = parser.parse_args(argv)

    import run_p1_2_dual_track_benchmark as runner

    runner._load_deployment_env()
    from rag_v2.contracts.plan import SupervisorPlan
    from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever
    from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader
    from src.runtime.trusted_v2_production import StructuredFactStore, _path_env
    from src.runtime.trusted_v2_r4 import (
        CandidateDirectR4Policy,
        R4RetrievalRequest,
        build_slot_retrieval_requests,
    )

    def load(path: Path) -> list[dict]:
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    environ = dict(os.environ)
    eval_rows = load(args.eval_set)
    gold = {r["id"]: r for r in load(args.gold)}
    fixtures = {r["id"]: r for r in load(args.fixtures)}
    answerable = [r for r in eval_rows if gold[r["id"]].get("expected_outcome") == "ANSWER"]

    result: dict[str, Any] = {"environment": environment()}
    result["environment"]["fact_store_path"] = environ.get("TRUSTED_V2_FACT_STORE_PATH")
    result["environment"]["index_dir"] = environ.get("TRUSTED_V2_R4_INDEX_DIR")
    result["benchmark"] = {
        "eval_set": str(args.eval_set),
        "fixtures": str(args.fixtures),
        "questions": len(eval_rows),
        "answerable": len(answerable),
    }

    print("environment:", json.dumps(result["environment"], indent=2, default=str)[:900])

    # --- alignment: pure local -------------------------------------------------
    from rag_v2.supervisor import UnknownSemanticPolicy, align_query_to_plan

    alignment_ms: list[float] = []
    for row in eval_rows:
        plan = SupervisorPlan.from_dict(fixtures[row["id"]]["plan"])
        started = time.perf_counter()
        align_query_to_plan(row["question"], plan, unknown_policy=UnknownSemanticPolicy.STRICT_DIRECT_FACT)
        alignment_ms.append((time.perf_counter() - started) * 1000.0)
    result["alignment"] = summarise(alignment_ms)
    print("alignment", result["alignment"])

    # --- retrieval: pure local, deterministic ----------------------------------
    reader = CandidateViewIndexReader(_path_env(environ, "TRUSTED_V2_R4_INDEX_DIR", directory=True))
    store = StructuredFactStore(_path_env(environ, "TRUSTED_V2_FACT_STORE_PATH", directory=False))
    policy = CandidateDirectR4Policy(CandidateDirectRetriever(reader), materializer=store.materialize)

    per_rep: list[dict[str, Any]] = []
    for rep in range(args.retrieval_reps):
        samples: list[float] = []
        for row in answerable:
            plan = SupervisorPlan.from_dict(fixtures[row["id"]]["plan"])
            requests = build_slot_retrieval_requests(plan)
            started = time.perf_counter()
            policy.retrieve(
                R4RetrievalRequest(
                    request_id=f"perf-{rep}-{row['id']}",
                    standalone_query=row["question"],
                    plan=plan,
                    reason_code="MISSING_SLOT",
                    target_slots=tuple(r.slot_id for r in requests),
                )
            )
            samples.append((time.perf_counter() - started) * 1000.0)
        per_rep.append({"rep": rep, **summarise(samples)})
        print(f"retrieval rep {rep}: {per_rep[-1]}")
    result["retrieval_per_rep"] = per_rep
    result["retrieval_warm"] = per_rep[-1]
    result["retrieval_cold"] = per_rep[0]

    # --- end-to-end, the real replay -------------------------------------------
    if not args.skip_replay:
        replay_dir = args.out_dir / "e2e"
        replay_dir.mkdir(parents=True, exist_ok=True)
        namespace = argparse.Namespace(
            eval_set=args.eval_set,
            gold_evidence=args.gold,
            fixtures=args.fixtures,
            fact_store=environ.get("TRUSTED_V2_FACT_STORE_PATH"),
            sessions_db=str(BACKEND / "sessions.db"),
            endpoint="unused-by-replay",
            out_dir=replay_dir,
            track="replay",
            limit=0,
            timeout_per_query=180.0,
        )
        print(f"replay: {len(eval_rows)} questions (this is the slow part)")
        rows, _summary = runner.run_replay(eval_rows, fixtures, namespace)
        (replay_dir / "replay-predictions.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        by_path: dict[str, list[float]] = {"all": [], "released": [], "refused": [], "calculation": []}
        generation_ms: list[float] = []
        output_tokens: list[int] = []
        for row, source in zip(rows, eval_rows):
            latency = row.get("latency_ms")
            if latency is None:
                continue
            by_path["all"].append(latency)
            if row.get("release_status") == "RELEASED":
                by_path["released"].append(latency)
            else:
                by_path["refused"].append(latency)
            if str(row.get("route") or "").upper().startswith("CALCULATION"):
                by_path["calculation"].append(latency)
            if row.get("generation_latency_ms") is not None:
                generation_ms.append(float(row["generation_latency_ms"]))
            tokens = (row.get("tokens") or {}).get("actual_output_tokens")
            if tokens:
                output_tokens.append(int(tokens))

        result["end_to_end"] = {name: summarise(values) for name, values in by_path.items()}
        result["generation"] = summarise(generation_ms)
        result["output_tokens"] = {
            "n": len(output_tokens),
            "mean": round(statistics.fmean(output_tokens), 1) if output_tokens else None,
            "total": sum(output_tokens) or None,
        }
        # The remainder is labelled, not claimed as harness compute: the Binder's
        # remote calls live inside it.
        total_p50 = result["end_to_end"]["all"].get("p50_ms")
        gen_p50 = result["generation"].get("p50_ms")
        result["harness_other"] = {
            "note": "end_to_end p50 minus generation p50; still contains the Binder's remote calls",
            "p50_ms": round(total_p50 - gen_p50, 2) if (total_p50 and gen_p50) else None,
        }
        for name in ("all", "released", "refused", "calculation"):
            print(f"e2e {name}: {result['end_to_end'][name]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "runtime-performance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nwritten to {args.out_dir / 'runtime-performance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
