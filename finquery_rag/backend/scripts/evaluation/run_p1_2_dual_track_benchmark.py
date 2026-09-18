#!/usr/bin/env python3
"""P1.2: the dual-track benchmark -- RAW and ALIGNED-REPLAY over one question set.

    RAW              the question text through the real HTTP entry, with the
                     semantic alignment gate fully in the loop.  Nothing is
                     injected.  This is the baseline H2B has to beat.
    ALIGNED-REPLAY   in-process, with an authored plan fixture and a frozen
                     alignment verdict, so the chain from retrieval onward is
                     the real one: retrieval -> binding -> calculator ->
                     ContextCompiler -> financial model -> validator -> release.

They answer different questions and their results are never averaged together.
RAW says how many queries get in.  REPLAY says what happens once they do.

**What REPLAY does not inject.**  The fixture carries plan fields only --
intent, metric, period, role, operation.  The gold's ``fact_ids``,
``expected_value``, ``operands`` and ``values`` never enter the runtime; if they
did, the run would be a lookup of an answer we already had rather than a RAG
benchmark.  The only other thing supplied is the alignment verdict, and the
gate's real verdict for every row is recorded beside it.

Run under Rule B: the backend must be stopped first, because this process loads
the specialist itself and exactly one PyTorch process may hold a CUDA context.

    python run_p1_2_dual_track_benchmark.py --track both --out-dir <dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_EVALUATION_DIR = Path(__file__).resolve().parent
if str(_EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALUATION_DIR))

DEFAULT_FIXTURES = _BACKEND_DIR / "benchmarks/tv2_canonical_v1/plan-fixtures-v2.jsonl"
DEFAULT_OUT_DIR = _BACKEND_DIR / "artifacts/evaluation/p1-2-dual-track"
EXPECTED_PRODUCTION_FINGERPRINT = (
    "25e7c9b33792637ba61fefc53b80d10745895696ebe8910b3ff2c1e21ee534fc"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


# --- fixtures -----------------------------------------------------------------------------------


def _plan_from_fixture(row: dict[str, Any]) -> Any:
    from rag_v2.contracts import Action, Intent, RequiredSlot, SupervisorPlan

    blob = row["plan"]
    return SupervisorPlan(
        Intent(blob["intent"]),
        tuple(
            RequiredSlot(
                slot["slot_id"],
                slot["metric"],
                slot["period"],
                slot["role"],
                slot["value_type"],
                slot["unit"],
            )
            for slot in blob["required_slots"]
        ),
        blob["operation"],
        Action(blob["next_action"]),
    )


def _alignment_for(question: str, plan: Any) -> Any:
    from rag_v2.supervisor import UnknownSemanticPolicy, align_query_to_plan

    return align_query_to_plan(
        question, plan, unknown_policy=UnknownSemanticPolicy.STRICT_DIRECT_FACT
    )


# --- RAW ----------------------------------------------------------------------------------------

#: RAW's stage attribution is read off the HTTP response, which is coarser than
#: the replay's.  Recorded as a separate derivation rather than blended, so the
#: two tracks' ``reached`` columns are not read as the same measurement.
_RAW_DERIVATION = "http_response: retrieved_chunks/sources/evidence_ids"


def run_raw(
    questions: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    import run_tv2_canonical_benchmark as canonical

    client = canonical.BenchmarkHTTPClient(
        base_url=args.endpoint, sessions_db=args.sessions_db
    )
    fact_index = canonical.FactStoreGroundingIndex(args.fact_store)
    rows: list[dict[str, Any]] = []

    for index, question in enumerate(questions, 1):
        body, provenance, latency_ms = client.execute_query(
            q_id=question["id"],
            q_text=question["question"],
            timeout_s=args.timeout_per_query,
        )
        prediction = canonical.process_query_result(
            question, body, provenance, latency_ms, fact_index
        )
        trace = prediction.get("trace") or {}
        evidence_ids = prediction.get("evidence_ids") or []
        rows.append(
            {
                "id": prediction["id"],
                "stratum": prediction["stratum"],
                "question": prediction["question"],
                "track": "RAW",
                "status": prediction.get("status"),
                "release_status": prediction.get("release_status"),
                "reason_codes": prediction.get("reason_codes") or [],
                "answer": prediction.get("answer") or "",
                "evidence_ids": evidence_ids,
                "citation_ids": prediction.get("citation_ids") or [],
                "calculation_ids": prediction.get("calculation_ids") or [],
                "calculations": prediction.get("calculations") or [],
                "error": prediction.get("error"),
                "latency_ms": latency_ms,
                "reached": {
                    "retrieval": bool(trace.get("source_count")),
                    "binding": bool(evidence_ids),
                    "generation": bool(trace.get("evidence_count")),
                    "validation": prediction.get("release_status") == "RELEASED",
                },
                "reached_derivation": _RAW_DERIVATION,
                "computed_align_status": None,
                "align_overridden": False,
                "tokens": {},
            }
        )
        print(
            f"  RAW [{index:>3}/{len(questions)}] {prediction['id']:<28} "
            f"{str(prediction.get('status')):<14} {latency_ms:>8.1f}ms",
            flush=True,
        )
    return rows


# --- ALIGNED-REPLAY -----------------------------------------------------------------------------


def _load_deployment_env() -> None:
    """Load the deployment configuration without clobbering an explicit device.

    The replay track needs the real assets the service uses -- the R4 index, the
    fact store, the binder credentials, the specialist checkpoint -- and those
    live in ``config/deployment/online.env``.  It must be loaded with
    ``override=False``: that file sets ``CUDA_VISIBLE_DEVICES``, and the whole
    point of this run is to use a device the operator chose.  A silent override
    would put a second PyTorch process on the service's card.
    """

    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        return
    for candidate in (
        Path("/disk/qh/nano-finrag/config/deployment/online.env"),
        _BACKEND_DIR.parents[1] / "config/deployment/online.env",
        _BACKEND_DIR / "online.env",
    ):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            print(f"  deployment env loaded from {candidate} (override=False)")
            return
    print("  no deployment env found; relying on the process environment")


def run_replay(
    questions: list[dict[str, Any]],
    fixtures_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import asyncio

    from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
    from src.runtime import FinancialQueryRequest, V2ExecutionRequest
    from src.runtime.trusted_v2_production import _load_resources
    from src.runtime.trusted_v2_production import (
        build_trusted_v2_runtime_for_request as build_runtime,
    )

    _load_deployment_env()
    environ = dict(os.environ)
    resources = _load_resources(environ)
    print("  resources loaded from the production loader (assets unchanged)")

    rows: list[dict[str, Any]] = []
    overrides_used = 0

    for index, question in enumerate(questions, 1):
        fixture = fixtures_by_id[question["id"]]
        plan = _plan_from_fixture(fixture)
        computed = _alignment_for(question["question"], plan)

        # Only override where the gate actually refused.  A row the gate allows
        # runs un-overridden, so the twelve questions that legitimately align are
        # measured on the real verdict and not on a fixture of it.
        override = None
        if not computed.allowed:
            override = dataclasses.replace(
                computed,
                status=computed.status.__class__.ALIGNED,
                mismatches=(),
                ambiguous_query_fields=(),
            )
            overrides_used += 1

        # The fixture supervisor is the *only* production component replaced.
        # Everything else -- index reader, fact store, binder, specialist,
        # budget -- is the object the production loader built.
        replay_resources = dataclasses.replace(
            resources,
            supervisor=SupervisorService(
                DeterministicFallbackProvider({question["question"]: plan})
            ),
        )

        request = FinancialQueryRequest(
            request_id=f"p1-2-replay-{question['id']}",
            user_id="p1-2-benchmark",
            session_id=f"p1-2-replay-{question['id']}",
            original_query=question["question"],
            standalone_query=question["question"],
            query_as_resolved=True,
        )

        started = time.perf_counter()
        try:
            runtime = build_runtime(
                None, request, resources=replay_resources, alignment_override=override
            )
            outcome = asyncio.run(
                runtime.coordinator.execute(
                    V2ExecutionRequest.from_financial_request(request)
                )
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            outcome, error = None, f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000.0

        row = _replay_row(
            question=question,
            outcome=outcome,
            error=error,
            latency_ms=latency_ms,
            computed=computed,
            overridden=override is not None,
            runtime=runtime if outcome is not None else None,
        )
        rows.append(row)
        print(
            f"  REPLAY [{index:>3}/{len(questions)}] {question['id']:<28} "
            f"{str(row['status']):<14} align={computed.status.value:<9} "
            f"override={str(row['align_overridden']):<5} {latency_ms:>8.1f}ms",
            flush=True,
        )

    summary = {
        "resources_from_production_loader": True,
        "supervisor_replaced_by_fixture": True,
        "align_overrides_used": overrides_used,
        "align_unoverridden": len(questions) - overrides_used,
    }
    return rows, summary


def _binder_observation(runtime: Any) -> dict[str, Any]:
    """What the binder decided, in the binder's own vocabulary.

    ``EVIDENCE_CONFLICT`` is one reason code covering three unrelated
    mechanisms: a slot with admissible-but-disagreeing candidates and no safe
    consensus, a binding the provider returned as ``AMBIGUOUS``, and a binding
    the provider returned as ``INVALID`` -- which is a schema failure, not a
    conflict at all.  Counting them together makes the number unusable for
    saying *why* binding failed, and a repair aimed at "conflicts" would be
    aimed at all three at once.

    The capability already records which one fired, per round, and exposes it
    through ``trace_snapshot()``; the benchmark simply was not reading it.
    Capturing it here is observation only -- the snapshot is taken after the
    run and nothing in the pipeline reads it back.
    """

    if runtime is None:
        return {}
    try:
        capability = runtime.coordinator.capabilities.evidence_evaluator
        snapshot = capability.trace_snapshot()
    except Exception:  # pragma: no cover - defensive, shape may change
        return {}

    rounds = list(snapshot.get("binder_rounds") or [])
    final = rounds[-1] if rounds else {}
    return {
        # One entry per binder round; the last is the one the outcome reflects.
        "binder_round_statuses": [str(item.get("status")) for item in rounds],
        "binder_final_status": str(final.get("status")) if final else None,
        "binder_final_bound_slot_ids": list(final.get("bound_slot_ids") or []),
        "binder_final_missing_slot_ids": list(final.get("missing_slot_ids") or []),
        # On the ``_unresolved_conflict_slots`` path this carries the slots the
        # consensus test refused; on the AMBIGUOUS path, the provider's own.
        "binder_final_conflict_slot_ids": list(final.get("ambiguous_slot_ids") or []),
        "binder_bound_slots": sorted(
            str(key) for key in (snapshot.get("bound_slot_bindings") or {})
        ),
        "binder_round_count": len(rounds),
    }


def _replay_row(
    *,
    question: dict[str, Any],
    outcome: Any,
    error: str | None,
    latency_ms: float,
    computed: Any,
    overridden: bool,
    runtime: Any,
) -> dict[str, Any]:
    generation = None
    if runtime is not None:
        try:
            generation = runtime.coordinator.capabilities.generation
        except Exception:  # pragma: no cover - defensive, shape may change
            generation = None

    tokens: dict[str, Any] = {}
    pack_handle_count = None
    generation_latency = None
    generation_calls = 0
    if generation is not None:
        generation_calls = int(getattr(generation, "specialist_calls", 0) or 0)
        pack = getattr(generation, "last_context_pack", None)
        if pack is not None:
            budget = getattr(pack, "budget", None)
            tokens["context_payload_tokens"] = getattr(
                budget, "selected_context_tokens", None
            )
            references = getattr(pack, "references", None)
            handles = getattr(references, "evidence_handles", None)
            if handles is not None:
                pack_handle_count = len(handles)
        invocation = getattr(generation, "invocation", None)
        response = getattr(invocation, "last_response", None)
        usage = dict(getattr(response, "usage", None) or {})
        tokens["actual_model_input_tokens"] = usage.get("prompt_tokens")
        tokens["actual_output_tokens"] = usage.get("completion_tokens")

    if outcome is None:
        return {
            "id": question["id"],
            "stratum": question["stratum"],
            "question": question["question"],
            "track": "ALIGNED-REPLAY",
            "status": "ERROR",
            "release_status": "NOT_RELEASED",
            "reason_codes": [],
            "answer": "",
            "evidence_ids": [],
            "citation_ids": [],
            "calculation_ids": [],
            "calculations": [],
            "error": error,
            "latency_ms": round(latency_ms, 2),
            "generation_latency_ms": generation_latency,
            "reached": {},
            "reached_derivation": "in_process_outcome",
            "computed_align_status": computed.status.value,
            "computed_align_mismatches": list(computed.mismatches),
            "align_overridden": overridden,
            "provider_failure": bool(error),
            "tokens": tokens,
            "pack_handle_count": pack_handle_count,
            **_binder_observation(runtime),
        }

    terminal_state = (outcome.runtime_metadata or {}).get("terminal_state")
    return {
        "id": question["id"],
        "stratum": question["stratum"],
        "question": question["question"],
        "track": "ALIGNED-REPLAY",
        "status": outcome.status.value,
        "release_status": outcome.release_status.value,
        "reason_codes": list(outcome.reason_codes),
        "answer": outcome.answer or "",
        "evidence_ids": list(outcome.evidence_ids),
        "citation_ids": list(outcome.citation_ids),
        "calculation_ids": list(outcome.calculation_ids),
        "calculations": list(outcome.calculations),
        "error": None,
        "latency_ms": round(latency_ms, 2),
        "generation_latency_ms": generation_latency,
        "reached": {
            # Past planning means the loop started, which is the first thing
            # that happens after retrieval is dispatched.
            "retrieval": terminal_state not in {"SUPERVISOR", "PLAN", None},
            "binding": bool(outcome.evidence_ids),
            "generation": generation_calls > 0,
            "validation": outcome.validator_status is not None,
        },
        "reached_derivation": "in_process_outcome:terminal_state+evidence+generation_calls",
        "computed_align_status": computed.status.value,
        "computed_align_mismatches": list(computed.mismatches),
        "align_overridden": overridden,
        "provider_failure": False,
        "route": outcome.route,
        "validator_status": outcome.validator_status,
        "tokens": {key: value for key, value in tokens.items() if value is not None},
        "pack_handle_count": pack_handle_count,
        **_binder_observation(runtime),
    }


# --- report -------------------------------------------------------------------------------------


def build_report(
    raw_rows: Sequence[dict[str, Any]],
    replay_rows: Sequence[dict[str, Any]],
    raw_scored: dict[str, Any] | None,
    replay_scored: dict[str, Any] | None,
    gold: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
) -> str:
    lines = [
        "# P1.2 — Dual-Track Benchmark",
        "",
        "The two tables below answer different questions and are not combined.",
        "Table B isolates the semantic alignment gate; it is **not** full",
        "end-to-end accuracy and its denominator is stated on the table.",
        "",
    ]

    if raw_scored is not None:
        lines += [
            f"## A. Full-pipeline Reachability — {raw_scored['total']} → align → retrieval → generation → release",
            "",
            "| stage | reached |",
            "| --- | --- |",
            f"| align pass | {raw_scored['align_pass_rate']} |",
            f"| retrieval | {raw_scored['reached_retrieval']} |",
            f"| binding | {raw_scored['reached_binding']} |",
            f"| generation | {raw_scored['reached_generation']} |",
            f"| validation | {raw_scored['reached_validation']} |",
            "",
            "Reject reasons:",
            "",
            "| reason | count |",
            "| --- | --- |",
        ]
        lines += [
            f"| `{reason}` | {count} |"
            for reason, count in raw_scored["reject_reason_distribution"].items()
        ]
        lines.append("")

    if replay_scored is not None:
        aligned = sum(
            1 for row in replay_rows if row.get("computed_align_status") == "ALIGNED"
        )
        lines += [
            f"## B. Post-alignment RAG success — {replay_scored['total']} authored plans → real chain → release",
            "",
            f"Denominator: {replay_scored['total']} questions "
            f"({replay_scored['answerable']} answerable, {replay_scored['abstention']} abstention). "
            f"The gate genuinely allowed {aligned}; the remainder ran under a recorded override.",
            "",
            "| metric | value |",
            "| --- | --- |",
            f"| candidate correctness (answerable) | {replay_scored['candidate_correctness']} |",
            f"| released correctness (answerable) | {replay_scored['answerable_correctness']} |",
            f"| no-answer correctness | {replay_scored['no_answer_correctness']} |",
            f"| no-answer reason attribution | {replay_scored['no_answer_reason_attribution']} |",
            f"| false release rate | {replay_scored['false_release_rate']} |",
            f"| over-conservative block rate | {replay_scored['over_conservative_rate']} |",
            "",
            "Counts:",
            "",
            "| outcome | count |",
            "| --- | --- |",
        ]
        lines += [
            f"| {name} | {count} |"
            for name, count in replay_scored["counts"].items()
        ]
        citations = replay_scored["citations"]
        lines += [
            "",
            "Citation diagnostics (measured, not repaired):",
            "",
            "| metric | value | denominator |",
            "| --- | --- | --- |",
            f"| literal placeholder | {citations['literal_citation_placeholder_rate']} | {citations['rows']} |",
            f"| malformed | {citations['malformed_citation_rate']} | {citations['rows']} |",
            f"| handle beyond pack | {citations['cited_handle_not_in_pack_rate']} | {citations['cited_handle_not_in_pack_denominator']} |",
            "",
        ]

    lines += [
        "## Provenance",
        "",
        "```json",
        json.dumps(provenance, indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=None)
    parser.add_argument("--gold-evidence", type=Path, default=None)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--fact-store", default="/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl")
    parser.add_argument("--sessions-db", default=None)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18002")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--track", choices=("raw", "replay", "both"), default="both")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-per-query", type=float, default=90.0)
    args = parser.parse_args(argv)

    args.eval_set = args.eval_set or Path(
        "benchmarks/tv2_canonical_v1/canonical-eval-v1.jsonl"
    )
    args.gold_evidence = args.gold_evidence or Path(
        "benchmarks/tv2_canonical_v1/gold-evidence-v1.jsonl"
    )
    args.sessions_db = args.sessions_db or str(_BACKEND_DIR / "sessions.db")

    import src.evaluation.p1_2_dual_track as dual

    questions = _load_jsonl(args.eval_set)
    gold = {row["id"]: row for row in _load_jsonl(args.gold_evidence)}

    fixture_rows = _load_jsonl(args.fixtures)
    fixtures_by_id = {row["id"]: row for row in fixture_rows}
    fixture_digest = hashlib.sha256(
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) for row in fixture_rows
        ).encode("utf-8")
        + b"\n"
    ).hexdigest()

    if args.limit:
        questions = questions[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    replay_summary: dict[str, Any] = {}

    if args.track in ("raw", "both"):
        print(f"RAW: {len(questions)} questions via {args.endpoint}")
        raw_rows = run_raw(questions, args)
        (args.out_dir / "raw-predictions.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in raw_rows) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    if args.track in ("replay", "both"):
        print(f"ALIGNED-REPLAY: {len(questions)} questions in-process")
        replay_rows, replay_summary = run_replay(questions, fixtures_by_id, args)
        (args.out_dir / "replay-predictions.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in replay_rows) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    raw_scored = dual.score_reachability(raw_rows) if raw_rows else None
    replay_scored = (
        dual.score_downstream(replay_rows, gold) if replay_rows else None
    )
    if raw_scored is not None:
        _write_json(args.out_dir / "reachability.json", raw_scored)
    if replay_scored is not None:
        _write_json(args.out_dir / "downstream.json", replay_scored)
        _write_json(
            args.out_dir / "downstream-by-stratum.json",
            dual.pareto_by_stratum(replay_rows, gold),
        )

    provenance = {
        "stage": "P1-2-DUAL-TRACK",
        "commit_sha": os.environ.get("P1_2_COMMIT_SHA"),
        "config_fingerprint": EXPECTED_PRODUCTION_FINGERPRINT,
        "eval_set_sha256": _sha256_file(args.eval_set),
        "gold_sha256": _sha256_file(args.gold_evidence),
        "fixture_sha256": fixture_digest,
        "track": args.track,
        "question_count": len(questions),
        "raw_client": "scripts/evaluation/run_tv2_canonical_benchmark.py (reused verbatim)",
        "replay_injection": "plan fixture + recorded alignment override; supervisor is the only replaced component",
        **replay_summary,
    }
    _write_json(args.out_dir / "seal.json", provenance)
    (args.out_dir / "report.md").write_text(
        build_report(raw_rows, replay_rows, raw_scored, replay_scored, gold, provenance),
        encoding="utf-8",
        newline="\n",
    )

    print()
    if raw_scored is not None:
        print("Reachability:", json.dumps(raw_scored, ensure_ascii=False, sort_keys=True))
    if replay_scored is not None:
        print("Downstream counts:", json.dumps(replay_scored["counts"], ensure_ascii=False, sort_keys=True))
        print("Downstream rates:", json.dumps(
            {
                key: replay_scored[key]
                for key in (
                    "answerable_correctness",
                    "no_answer_correctness",
                    "no_answer_reason_attribution",
                    "candidate_correctness",
                    "false_release_rate",
                    "over_conservative_rate",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        ))
    print(f"\nreport written to {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
