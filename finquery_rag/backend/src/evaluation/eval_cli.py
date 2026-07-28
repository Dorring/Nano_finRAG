"""CLI for offline FinQuery RAG evaluation fixtures.

Examples:
  python -m src.eval_cli score --cases eval/golden.jsonl --predictions eval/preds.jsonl
  python -m src.eval_cli replay-from-traces --db trace_log.db --tenant-id 1 --out eval/replay.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
from html import escape
import json
from pathlib import Path
import os
import sys

from .evaluation import (
    audit_evaluation_fixtures,
    build_failure_analysis_markdown,
    build_interview_report,
    compare_reports,
    diagnose_retrieval,
    diagnose_candidate_stages,
    evaluate_predictions,
    export_replay_cases_from_feedback,
    export_replay_cases_from_traces,
    load_jsonl_cases,
    load_jsonl_predictions,
    write_json_file,
)
from src.services.feedback import FeedbackStore
from src.services.trace import TraceLogger
from .eval_runner import run_jsonl_cases, run_jsonl_cases_http, validate_n_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FinQuery RAG offline evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score predictions against JSONL cases")
    score.add_argument("--cases", required=True, help="Golden/replay cases JSONL")
    score.add_argument("--predictions", required=True, help="Predictions JSONL")
    score.add_argument("--out", help="Optional report JSON output path")

    run = sub.add_parser("run", help="Run RAGEngine against JSONL cases")
    run.add_argument("--cases", required=True, help="Golden/replay cases JSONL")
    run.add_argument("--out", required=True, help="Predictions JSONL output path")
    run.add_argument("--user-id", type=int, required=True, help="Tenant/user id for scoped retrieval")
    run.add_argument("--n-results", type=int, default=5, help="Top-k chunks per query")

    run_http = sub.add_parser("run-http", help="Run JSONL cases through a running FinQuery HTTP backend")
    run_http.add_argument("--cases", required=True, help="Golden/replay cases JSONL")
    run_http.add_argument("--out", required=True, help="Predictions JSONL output path")
    run_http.add_argument("--api-base", default=os.getenv("FINQUERY_API_BASE", "http://127.0.0.1:8000"), help="FinQuery backend base URL")
    run_http.add_argument("--token", default=os.getenv("FINQUERY_TOKEN"), help="Bearer token; defaults to FINQUERY_TOKEN")
    run_http.add_argument("--n-results", type=int, default=5, help="Top-k chunks per query")
    run_http.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds")

    compare = sub.add_parser("compare", help="Compare baseline and candidate reports")
    compare.add_argument("--baseline", required=True, help="Baseline report JSON")
    compare.add_argument("--candidate", required=True, help="Candidate report JSON")
    compare.add_argument("--tolerance", type=float, default=0.0, help="Allowed negative metric delta")
    compare.add_argument("--out", help="Optional comparison JSON output path")

    gate = sub.add_parser("gate", help="Score predictions and enforce CI-friendly eval thresholds")
    gate.add_argument("--cases", required=True, help="Golden/replay cases JSONL")
    gate.add_argument("--predictions", required=True, help="Predictions JSONL")
    gate.add_argument("--baseline", help="Optional baseline report JSON for regression comparison")
    gate.add_argument("--tolerance", type=float, default=0.0, help="Allowed negative metric delta versus baseline")
    gate.add_argument("--min-pass-rate", type=float, default=1.0, help="Minimum required pass_rate")
    gate.add_argument("--max-missing", type=int, default=0, help="Maximum allowed missing predictions")
    gate.add_argument("--out", help="Optional candidate report JSON output path")
    gate.add_argument("--comparison-out", help="Optional comparison JSON output path when --baseline is provided")
    gate.add_argument("--junit-out", help="Optional JUnit XML output path for CI annotations")

    doctor = sub.add_parser("doctor", help="Run non-secret FinQuery runtime readiness checks")
    doctor.add_argument("--bm25-db", help="Override BM25 SQLite DB path")
    doctor.add_argument("--trace-db", help="Override TraceLogger SQLite DB path")
    doctor.add_argument("--feedback-db", help="Override feedback SQLite DB path")
    doctor.add_argument("--out", help="Optional health snapshot JSON output path")
    doctor.add_argument("--warn-only", action="store_true", help="Return 0 even when readiness is degraded")

    migration = sub.add_parser("migration-audit", help="Audit local stores for legacy unscoped index data")
    migration.add_argument("--bm25-db", help="Override BM25 SQLite DB path")
    migration.add_argument("--registry-db", help="Override document registry SQLite DB path")
    migration.add_argument("--chroma-path", help="Override Chroma directory path")
    migration.add_argument("--out", help="Optional migration audit JSON output path")
    migration.add_argument("--warn-only", action="store_true", help="Return 0 even when high-risk migration issues are found")

    preflight = sub.add_parser("preflight", help="Run deployment preflight checks without model calls")
    preflight.add_argument("--cases", default="eval/golden_smoke.jsonl", help="Golden/replay cases JSONL")
    preflight.add_argument("--predictions", default="eval/predictions_smoke.jsonl", help="Predictions JSONL")
    preflight.add_argument("--baseline", default="eval/baseline_smoke_report.json", help="Optional baseline report JSON; use empty string to skip")
    preflight.add_argument("--bm25-db", help="Override BM25 SQLite DB path")
    preflight.add_argument("--registry-db", help="Override document registry SQLite DB path")
    preflight.add_argument("--chroma-path", help="Override Chroma directory path")
    preflight.add_argument("--trace-db", help="Override TraceLogger SQLite DB path")
    preflight.add_argument("--feedback-db", help="Override feedback SQLite DB path")
    preflight.add_argument("--min-pass-rate", type=float, default=1.0)
    preflight.add_argument("--max-missing", type=int, default=0)
    preflight.add_argument("--tolerance", type=float, default=0.0, help="Allowed negative metric delta versus baseline")
    preflight.add_argument("--min-cases", type=int, default=1)
    preflight.add_argument("--required-tag", dest="required_tags", action="append", default=[])
    preflight.add_argument("--require-expected-intent", action="store_true")
    preflight.add_argument("--out", help="Optional preflight JSON output path")
    preflight.add_argument("--warn-only", action="store_true", help="Return 0 even when preflight sections fail")

    retrieval_diag = sub.add_parser("retrieval-diagnostics", help="Explain expected-source retrieval coverage")
    retrieval_diag.add_argument("--cases", required=True, help="Golden/replay cases JSONL")
    retrieval_diag.add_argument("--predictions", required=True, help="Predictions JSONL")
    retrieval_diag.add_argument("--k", dest="ks", type=int, action="append", help="Recall@K cutoff; repeatable")
    retrieval_diag.add_argument("--candidate-field", choices=["retrieved_chunks", "sources"], default="retrieved_chunks")
    retrieval_diag.add_argument("--worst-limit", type=int, default=10, help="Maximum worst cases to include")
    retrieval_diag.add_argument("--out", help="Optional diagnostics JSON output path")

    candidate_stage = sub.add_parser("candidate-stage-diagnostics", help="Classify candidate loss across BM25, dense, RRF, reranker, and final stages")
    candidate_stage.add_argument("--cases", required=True, help="Golden/replay cases JSONL")
    candidate_stage.add_argument("--predictions", required=True, help="Predictions JSONL with retrieval_debug candidate stages")
    candidate_stage.add_argument("--out", help="Optional diagnostics JSON output path")

    interview = sub.add_parser("interview-report", help="Build a compact interview/demo metrics report")
    interview.add_argument("--cases", required=True, help="Golden/replay cases JSONL")
    interview.add_argument("--predictions", required=True, help="Predictions JSONL")
    interview.add_argument("--k", dest="ks", type=int, action="append", help="Recall@K cutoff; repeatable")
    interview.add_argument("--candidate-field", choices=["retrieved_chunks", "sources"], default="retrieved_chunks")
    interview.add_argument("--worst-limit", type=int, default=5, help="Maximum weak cases to include")
    interview.add_argument("--out", help="Optional interview report JSON output path")

    failure_analysis = sub.add_parser("failure-analysis", help="Write a Markdown failure analysis report")
    failure_analysis.add_argument("--cases", required=True, help="Golden/replay cases JSONL")
    failure_analysis.add_argument("--predictions", required=True, help="Predictions JSONL")
    failure_analysis.add_argument("--out", required=True, help="Markdown output path")
    failure_analysis.add_argument("--limit", type=int, help="Maximum failed cases to include")

    retrieval_bundle = sub.add_parser("retrieval-eval-bundle", help="Write score, retrieval diagnostics, and interview reports together")
    retrieval_bundle.add_argument("--cases", required=True, help="Golden/replay cases JSONL")
    retrieval_bundle.add_argument("--predictions", required=True, help="Predictions JSONL")
    retrieval_bundle.add_argument("--k", dest="ks", type=int, action="append", help="Recall@K cutoff; repeatable")
    retrieval_bundle.add_argument("--candidate-field", choices=["retrieved_chunks", "sources"], default="retrieved_chunks")
    retrieval_bundle.add_argument("--out-dir", required=True, help="Directory for score.json, retrieval_diagnostics.json, interview_report.json, and manifest.json")

    audit = sub.add_parser("audit-fixtures", help="Audit evaluation fixture coverage and quality")
    audit.add_argument("--cases", required=True, help="Golden/replay cases JSONL")
    audit.add_argument("--min-cases", type=int, default=1, help="Minimum required case count")
    audit.add_argument("--required-tag", dest="required_tags", action="append", default=[], help="Tag that must appear at least once; repeatable")
    audit.add_argument("--require-expected-source", action="store_true", help="Fail cases without expected_sources")
    audit.add_argument("--require-expected-intent", action="store_true", help="Fail cases without expected_intent")
    audit.add_argument("--out", help="Optional audit JSON output path")

    traces = sub.add_parser("traces", help="Export tenant-scoped trace rows as JSONL")
    traces.add_argument(
        "--db",
        default=os.getenv("TRACE_DB_PATH", "trace_log.db"),
        help="TraceLogger SQLite DB",
    )
    traces.add_argument("--tenant-id", type=int, required=True)
    traces.add_argument("--limit", type=int, default=100)
    traces.add_argument("--offset", type=int, default=0)
    traces.add_argument("--created-after", type=float)
    traces.add_argument("--created-before", type=float)
    traces.add_argument("--error-only", action="store_true")
    traces.add_argument("--out", required=True, help="Output traces JSONL")

    traces_cleanup = sub.add_parser("traces-cleanup", help="Delete old trace rows")
    traces_cleanup.add_argument(
        "--db",
        default=os.getenv("TRACE_DB_PATH", "trace_log.db"),
        help="TraceLogger SQLite DB",
    )
    traces_cleanup.add_argument(
        "--ttl-seconds",
        type=int,
        default=int(os.getenv("TRACE_TTL_SECONDS", "0")),
        help="Delete traces older than this TTL. 0 deletes traces older than now.",
    )
    traces_cleanup.add_argument("--tenant-id", type=int, help="Optional tenant/user scope")
    traces_cleanup.add_argument("--out", help="Optional JSON cleanup report output path")

    replay = sub.add_parser("replay-from-traces", help="Export replay cases from trace DB")
    replay.add_argument(
        "--db",
        default=os.getenv("TRACE_DB_PATH", "trace_log.db"),
        help="TraceLogger SQLite DB",
    )
    replay.add_argument("--tenant-id", type=int, required=True)
    replay.add_argument("--limit", type=int, default=100)
    replay.add_argument("--offset", type=int, default=0)
    replay.add_argument("--created-after", type=float)
    replay.add_argument("--created-before", type=float)
    replay.add_argument("--error-only", action="store_true")
    replay.add_argument("--out", required=True, help="Output replay JSONL")

    feedback_replay = sub.add_parser("feedback-to-replay", help="Export feedback-linked trace replay cases")
    feedback_replay.add_argument(
        "--feedback-db",
        default=os.getenv("FEEDBACK_DB_PATH", "feedback.db"),
        help="Feedback SQLite DB",
    )
    feedback_replay.add_argument(
        "--trace-db",
        default=os.getenv("TRACE_DB_PATH", "trace_log.db"),
        help="TraceLogger SQLite DB",
    )
    feedback_replay.add_argument("--tenant-id", type=int, required=True)
    feedback_replay.add_argument("--rating", choices=["up", "down"], default="down")
    feedback_replay.add_argument("--limit", type=int, default=100)
    feedback_replay.add_argument("--offset", type=int, default=0)
    feedback_replay.add_argument("--out", required=True, help="Output replay JSONL")

    bm25_check = sub.add_parser("bm25-check", help="Check BM25/FTS5 index consistency")
    bm25_check.add_argument(
        "--db",
        default=os.getenv("BM25_DB_PATH", "rag_bm25.db"),
        help="BM25 SQLite DB",
    )
    bm25_check.add_argument("--user-id", type=int, help="Optional tenant/user scope")
    bm25_check.add_argument("--out", help="Optional JSON report output path")

    bm25_rebuild = sub.add_parser("bm25-rebuild", help="Rebuild BM25/FTS5 index from chunk_store")
    bm25_rebuild.add_argument(
        "--db",
        default=os.getenv("BM25_DB_PATH", "rag_bm25.db"),
        help="BM25 SQLite DB",
    )
    bm25_rebuild.add_argument("--user-id", type=int, help="Optional tenant/user scope")
    bm25_rebuild.add_argument("--out", help="Optional JSON report output path")

    args = parser.parse_args(argv)

    if args.command == "score":
        try:
            cases = load_jsonl_cases(args.cases)
            predictions = load_jsonl_predictions(args.predictions)
            report = evaluate_predictions(cases, predictions)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            write_json_file(args.out, report)
        print(payload)
        return 0

    if args.command == "run":
        user_id = _normalize_positive_int(args.user_id, "user-id")
        if isinstance(user_id, str):
            print(user_id, file=sys.stderr)
            return 2
        try:
            n_results = validate_n_results(args.n_results)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        # Import lazily because main initializes FastAPI globals and the OpenAI client.
        from src.main import get_rag_engine

        predictions = asyncio.run(run_jsonl_cases(
            args.cases,
            args.out,
            get_rag_engine(),
            user_id=user_id,
            n_results=n_results,
        ))
        print(f"wrote {len(predictions)} predictions to {args.out}")
        return 0

    if args.command == "run-http":
        try:
            n_results = validate_n_results(args.n_results)
            predictions = run_jsonl_cases_http(
                args.cases,
                args.out,
                api_base=args.api_base,
                token=args.token,
                n_results=n_results,
                timeout=args.timeout,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"wrote {len(predictions)} predictions to {args.out}")
        return 0

    if args.command == "compare":
        tolerance = _normalize_non_negative_float(args.tolerance, "tolerance")
        if isinstance(tolerance, str):
            print(tolerance, file=sys.stderr)
            return 2
        try:
            baseline = _load_json_object(args.baseline, "baseline")
            candidate = _load_json_object(args.candidate, "candidate")
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        try:
            comparison = compare_reports(baseline, candidate, regression_tolerance=tolerance)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        payload = json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            write_json_file(args.out, comparison)
        print(payload)
        if not comparison["passed"]:
            _print_compare_failure_summary(comparison)
        return 0 if comparison["passed"] else 1

    if args.command == "gate":
        tolerance = _normalize_non_negative_float(args.tolerance, "tolerance")
        min_pass_rate = _normalize_fraction(args.min_pass_rate, "min-pass-rate")
        max_missing = _normalize_non_negative_int(args.max_missing, "max-missing")
        for error in (tolerance, min_pass_rate, max_missing):
            if isinstance(error, str):
                print(error, file=sys.stderr)
                return 2
        try:
            cases = load_jsonl_cases(args.cases)
            predictions = load_jsonl_predictions(args.predictions)
            report = evaluate_predictions(cases, predictions)
            comparison = None
            if args.baseline:
                baseline = _load_json_object(args.baseline, "baseline")
                comparison = compare_reports(baseline, report, regression_tolerance=tolerance)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        gate_result = _build_gate_result(
            report,
            comparison=comparison,
            min_pass_rate=min_pass_rate,
            max_missing=max_missing,
        )
        if args.out:
            write_json_file(args.out, report)
        if args.comparison_out and comparison is not None:
            write_json_file(args.comparison_out, comparison)
        if args.junit_out:
            _write_gate_junit(args.junit_out, gate_result)

        payload = json.dumps(gate_result, ensure_ascii=False, indent=2, sort_keys=True)
        print(payload)
        if not gate_result["passed"]:
            _print_gate_failure_summary(gate_result)
        return 0 if gate_result["passed"] else 1

    if args.command == "doctor":
        from src.services.health import collect_health_snapshot

        snapshot = collect_health_snapshot(
            bm25_db_path=args.bm25_db,
            trace_db_path=args.trace_db,
            feedback_db_path=args.feedback_db,
        )
        payload = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            write_json_file(args.out, snapshot)
        print(payload)
        if not snapshot.get("ready"):
            _print_doctor_failure_summary(snapshot)
        return 0 if snapshot.get("ready") or args.warn_only else 1

    if args.command == "migration-audit":
        from src.services.migration_audit import audit_migration_readiness

        report = audit_migration_readiness(
            bm25_db_path=args.bm25_db,
            registry_db_path=args.registry_db,
            chroma_path=args.chroma_path,
        )
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            write_json_file(args.out, report)
        print(payload)
        if not report["passed"]:
            _print_migration_audit_failure_summary(report)
        return 0 if report["passed"] or args.warn_only else 1

    if args.command == "preflight":
        from src.services.preflight import build_preflight_report

        try:
            report = build_preflight_report(
                cases_path=args.cases,
                predictions_path=args.predictions,
                baseline_path=args.baseline or None,
                bm25_db_path=args.bm25_db,
                registry_db_path=args.registry_db,
                chroma_path=args.chroma_path,
                trace_db_path=args.trace_db,
                feedback_db_path=args.feedback_db,
                min_pass_rate=args.min_pass_rate,
                max_missing=args.max_missing,
                regression_tolerance=args.tolerance,
                min_cases=args.min_cases,
                required_tags=tuple(args.required_tags or ()),
                require_expected_intent=args.require_expected_intent,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            write_json_file(args.out, report)
        print(payload)
        if not report["passed"]:
            _print_preflight_failure_summary(report)
        return 0 if report["passed"] or args.warn_only else 1

    if args.command == "retrieval-diagnostics":
        try:
            cases = load_jsonl_cases(args.cases)
            predictions = load_jsonl_predictions(args.predictions)
            report = diagnose_retrieval(
                cases,
                predictions,
                ks=args.ks or (1, 3, 5),
                candidate_field=args.candidate_field,
                worst_limit=args.worst_limit,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            write_json_file(args.out, report)
        print(payload)
        return 0

    if args.command == "candidate-stage-diagnostics":
        try:
            report = diagnose_candidate_stages(
                load_jsonl_cases(args.cases),
                load_jsonl_predictions(args.predictions),
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if args.out:
            write_json_file(args.out, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "interview-report":
        try:
            cases = load_jsonl_cases(args.cases)
            predictions = load_jsonl_predictions(args.predictions)
            report = build_interview_report(
                cases,
                predictions,
                ks=args.ks or (1, 3, 5),
                candidate_field=args.candidate_field,
                worst_limit=args.worst_limit,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            write_json_file(args.out, report)
        print(payload)
        return 0

    if args.command == "failure-analysis":
        try:
            cases = load_jsonl_cases(args.cases)
            predictions = load_jsonl_predictions(args.predictions)
            markdown = build_failure_analysis_markdown(
                cases,
                predictions,
                limit=args.limit,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(markdown, encoding="utf-8", newline="\n")
        print(f"wrote failure analysis to {args.out}")
        return 0

    if args.command == "retrieval-eval-bundle":
        try:
            cases = load_jsonl_cases(args.cases)
            predictions = load_jsonl_predictions(args.predictions)
            ks = args.ks or (1, 3, 5)
            score_report = evaluate_predictions(cases, predictions)
            retrieval_report = diagnose_retrieval(
                cases,
                predictions,
                ks=ks,
                candidate_field=args.candidate_field,
            )
            interview_report = build_interview_report(
                cases,
                predictions,
                ks=ks,
                candidate_field=args.candidate_field,
            )
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            outputs = {
                "score": out_dir / "score.json",
                "retrieval_diagnostics": out_dir / "retrieval_diagnostics.json",
                "interview_report": out_dir / "interview_report.json",
                "manifest": out_dir / "manifest.json",
            }
            write_json_file(outputs["score"], score_report)
            write_json_file(outputs["retrieval_diagnostics"], retrieval_report)
            write_json_file(outputs["interview_report"], interview_report)
            manifest = {
                "cases": args.cases,
                "predictions": args.predictions,
                "candidate_field": args.candidate_field,
                "ks": list(ks),
                "outputs": {key: str(value) for key, value in outputs.items() if key != "manifest"},
            }
            write_json_file(outputs["manifest"], manifest)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        print(payload)
        return 0

    if args.command == "audit-fixtures":
        try:
            cases = load_jsonl_cases(args.cases)
            audit_report = audit_evaluation_fixtures(
                cases,
                min_cases=args.min_cases,
                required_tags=args.required_tags,
                require_expected_source=args.require_expected_source,
                require_expected_intent=args.require_expected_intent,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        payload = json.dumps(audit_report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            write_json_file(args.out, audit_report)
        print(payload)
        if not audit_report["passed"]:
            _print_fixture_audit_failure_summary(audit_report)
        return 0 if audit_report["passed"] else 1

    if args.command == "traces":
        tenant_id = _normalize_positive_int(args.tenant_id, "tenant-id")
        if isinstance(tenant_id, str):
            print(tenant_id, file=sys.stderr)
            return 2
        bounds = _normalize_trace_bounds(args.limit, args.offset, args.created_after, args.created_before)
        if bounds["error"]:
            print(bounds["error"], file=sys.stderr)
            return 2
        logger = TraceLogger(db_path=args.db, sample_rate=1.0, redact_content=True)
        count = logger.export_traces_jsonl(
            tenant_id=tenant_id,
            output_path=args.out,
            limit=bounds["limit"],
            offset=bounds["offset"],
            created_after=bounds["created_after"],
            created_before=bounds["created_before"],
            error_only=args.error_only,
        )
        print(f"exported {count} traces to {args.out}")
        return 0

    if args.command == "traces-cleanup":
        ttl_seconds = _normalize_non_negative_int(args.ttl_seconds, "ttl-seconds")
        if isinstance(ttl_seconds, str):
            print(ttl_seconds, file=sys.stderr)
            return 2
        tenant_id = None
        if args.tenant_id is not None:
            tenant_id = _normalize_positive_int(args.tenant_id, "tenant-id")
            if isinstance(tenant_id, str):
                print(tenant_id, file=sys.stderr)
                return 2
        logger = TraceLogger(db_path=args.db, sample_rate=1.0, redact_content=True)
        report = logger.cleanup_by_ttl(ttl_seconds, tenant_id=tenant_id)
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            write_json_file(args.out, report)
        print(payload)
        return 0

    if args.command == "replay-from-traces":
        tenant_id = _normalize_positive_int(args.tenant_id, "tenant-id")
        if isinstance(tenant_id, str):
            print(tenant_id, file=sys.stderr)
            return 2
        bounds = _normalize_trace_bounds(args.limit, args.offset, args.created_after, args.created_before)
        if bounds["error"]:
            print(bounds["error"], file=sys.stderr)
            return 2
        logger = TraceLogger(db_path=args.db, sample_rate=1.0, redact_content=True)
        traces = logger.query_traces(
            tenant_id=tenant_id,
            limit=bounds["limit"],
            offset=bounds["offset"],
            created_after=bounds["created_after"],
            created_before=bounds["created_before"],
            error_only=args.error_only,
        )
        try:
            cases = export_replay_cases_from_traces(traces, args.out)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"exported {len(cases)} replay cases to {args.out}")
        return 0

    if args.command == "feedback-to-replay":
        tenant_id = _normalize_positive_int(args.tenant_id, "tenant-id")
        if isinstance(tenant_id, str):
            print(tenant_id, file=sys.stderr)
            return 2
        limit = _normalize_limit(args.limit)
        offset = _normalize_non_negative_int(args.offset, "offset")
        for error in (limit, offset):
            if isinstance(error, str):
                print(error, file=sys.stderr)
                return 2
        feedback_store = FeedbackStore(db_path=args.feedback_db)
        trace_logger = TraceLogger(db_path=args.trace_db, sample_rate=1.0, redact_content=True)
        feedback_rows = feedback_store.list_for_tenant(
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
            rating=args.rating,
        )
        try:
            cases = export_replay_cases_from_feedback(
                feedback_rows,
                lambda trace_id: trace_logger.get_trace_for_tenant(tenant_id, trace_id),
                args.out,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"exported {len(cases)} feedback replay cases to {args.out}")
        return 0

    if args.command == "bm25-check":
        user_id = _normalize_optional_positive_int(args.user_id, "user-id")
        if isinstance(user_id, str):
            print(user_id, file=sys.stderr)
            return 2
        from src.services.retrieval import SqliteBM25Retriever

        retriever = SqliteBM25Retriever(db_path=args.db)
        report = retriever.integrity_report(user_id=user_id)
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            write_json_file(args.out, report)
        print(payload)
        return 0 if report["ok"] else 1

    if args.command == "bm25-rebuild":
        user_id = _normalize_optional_positive_int(args.user_id, "user-id")
        if isinstance(user_id, str):
            print(user_id, file=sys.stderr)
            return 2
        from src.services.retrieval import SqliteBM25Retriever

        retriever = SqliteBM25Retriever(db_path=args.db)
        report = retriever.rebuild_fts_index(user_id=user_id)
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            write_json_file(args.out, report)
        print(payload)
        return 0 if report["ok"] else 1

    return 2


MAX_TRACE_EXPORT_LIMIT = 1000


def _normalize_limit(value, *, default: int = 100, maximum: int = MAX_TRACE_EXPORT_LIMIT):
    try:
        limit = int(value if value is not None else default)
    except (TypeError, ValueError):
        return "limit must be an integer"
    if limit <= 0:
        return "limit must be >= 1"
    return min(limit, maximum)


def _normalize_non_negative_int(value, name: str):
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return f"{name} must be an integer"
    if parsed < 0:
        return f"{name} must be >= 0"
    return parsed


def _normalize_positive_int(value, name: str):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return f"{name} must be an integer"
    if parsed < 1:
        return f"{name} must be >= 1"
    return parsed


def _normalize_optional_positive_int(value, name: str):
    if value is None:
        return None
    return _normalize_positive_int(value, name)


def _normalize_non_negative_float(value, name: str):
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return f"{name} must be a number"
    if parsed < 0:
        return f"{name} must be >= 0"
    return parsed


def _load_json_object(path: str, label: str) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{label} report cannot be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} report must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} report must be a JSON object")
    return payload


def _normalize_fraction(value, name: str):
    parsed = _normalize_non_negative_float(value, name)
    if isinstance(parsed, str):
        return parsed
    if parsed > 1:
        return f"{name} must be <= 1"
    return parsed


def _build_gate_result(
    report: dict,
    *,
    comparison: dict | None,
    min_pass_rate: float,
    max_missing: int,
) -> dict:
    summary = report.get("summary") or {}
    pass_rate = float(summary.get("pass_rate") or 0.0)
    missing_predictions = int(summary.get("missing_predictions") or 0)
    pass_rate_ok = pass_rate >= min_pass_rate
    missing_ok = missing_predictions <= max_missing
    checks = [
        {
            "name": "min_pass_rate",
            "passed": pass_rate_ok,
            "actual": pass_rate,
            "expected": min_pass_rate,
            "message": (
                "pass_rate %.6f meets required %.6f"
                if pass_rate_ok
                else "pass_rate %.6f is below required %.6f"
            ) % (pass_rate, min_pass_rate),
        },
        {
            "name": "max_missing_predictions",
            "passed": missing_ok,
            "actual": missing_predictions,
            "expected": max_missing,
            "message": (
                "missing_predictions %d is within allowed %d"
                if missing_ok
                else "missing_predictions %d exceeds allowed %d"
            ) % (missing_predictions, max_missing),
        },
    ]
    if comparison is not None:
        comparison_ok = bool(comparison.get("passed"))
        checks.append({
            "name": "baseline_regression",
            "passed": comparison_ok,
            "actual": comparison.get("regressions", []),
            "expected": [],
            "message": (
                "candidate matches baseline within tolerance"
                if comparison_ok
                else "; ".join(comparison.get("failure_reasons") or ["candidate regressed versus baseline"])
            ),
        })

    failed_checks = [check for check in checks if not check["passed"]]
    return {
        "passed": not failed_checks,
        "summary": summary,
        "checks": checks,
        "failed_checks": failed_checks,
        "missing_case_ids": report.get("missing_case_ids", []),
        "warnings": report.get("warnings", []),
        "comparison": comparison,
    }


def _write_gate_junit(path: str | Path, gate_result: dict) -> None:
    checks = gate_result.get("checks") or []
    failures = [check for check in checks if not check.get("passed")]
    body = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<testsuite name="finquery-eval-gate" tests="%d" failures="%d">' % (len(checks), len(failures)),
    ]
    for check in checks:
        name = escape(str(check.get("name") or "check"), quote=True)
        body.append('  <testcase classname="finquery.eval" name="%s">' % name)
        if not check.get("passed"):
            message = escape(str(check.get("message") or "evaluation gate check failed"), quote=True)
            body.append('    <failure message="%s">%s</failure>' % (message, message))
        body.append('  </testcase>')
    body.append('</testsuite>')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(body) + "\n", encoding="utf-8")


def _normalize_trace_bounds(limit, offset, created_after, created_before) -> dict:
    normalized_limit = _normalize_limit(limit)
    if isinstance(normalized_limit, str):
        return {"error": normalized_limit}
    normalized_offset = _normalize_non_negative_int(offset, "offset")
    if isinstance(normalized_offset, str):
        return {"error": normalized_offset}
    if (
        created_after is not None
        and created_before is not None
        and float(created_after) > float(created_before)
    ):
        return {"error": "created-after must be <= created-before"}
    return {
        "error": None,
        "limit": normalized_limit,
        "offset": normalized_offset,
        "created_after": created_after,
        "created_before": created_before,
    }



def _print_preflight_failure_summary(report: dict) -> None:
    """Print a compact preflight failure summary to stderr."""
    print("FinQuery preflight failed:", file=sys.stderr)
    for section in report.get("summary", {}).get("failed_sections", [])[:10]:
        print(f"- {section}", file=sys.stderr)


def _print_migration_audit_failure_summary(report: dict) -> None:
    """Print a compact migration audit failure summary to stderr."""
    print("FinQuery migration audit detected high-risk legacy data:", file=sys.stderr)
    high_risks = [risk for risk in report.get("risks", []) if risk.get("severity") == "high"]
    for risk in high_risks[:10]:
        print(f"- {risk.get('store')}: {risk.get('message')}", file=sys.stderr)
    if len(high_risks) > 10:
        print(f"- ... {len(high_risks) - 10} more high-risk issues", file=sys.stderr)


def _print_fixture_audit_failure_summary(audit_report: dict) -> None:
    """Print a compact fixture audit failure summary to stderr."""
    print("FinQuery fixture audit failed:", file=sys.stderr)
    for issue in audit_report.get("errors", [])[:10]:
        case = issue.get("case_id") or "fixture"
        print(f"- {case}: {issue.get('message')}", file=sys.stderr)
    errors = audit_report.get("errors", [])
    if len(errors) > 10:
        print(f"- ... {len(errors) - 10} more errors", file=sys.stderr)


def _print_doctor_failure_summary(snapshot: dict) -> None:
    """Print a compact readiness failure summary to stderr."""
    print("FinQuery doctor detected degraded readiness:", file=sys.stderr)
    checks = snapshot.get("checks") or {}
    for name, check in checks.items():
        if not isinstance(check, dict) or check.get("ok", False):
            continue
        required = "required" if check.get("required", True) else "optional"
        error = check.get("error") or ", ".join(check.get("errors") or []) or "check failed"
        print(f"- {name} ({required}): {error}", file=sys.stderr)


def _print_gate_failure_summary(gate_result: dict) -> None:
    """Print a compact human-readable gate failure summary to stderr."""
    print("FinQuery eval gate failed:", file=sys.stderr)
    for check in gate_result.get("failed_checks", [])[:10]:
        print(f"- {check.get('name')}: {check.get('message')}", file=sys.stderr)
    missing = gate_result.get("missing_case_ids") or []
    if missing:
        print("Missing predictions: " + ", ".join(str(item) for item in missing[:10]), file=sys.stderr)
        if len(missing) > 10:
            print(f"- ... {len(missing) - 10} more missing predictions", file=sys.stderr)


def _print_compare_failure_summary(comparison: dict) -> None:
    """Print a compact human-readable compare failure summary to stderr."""
    print("FinQuery eval comparison failed:", file=sys.stderr)
    reasons = comparison.get("failure_reasons") or []
    if reasons:
        for reason in reasons[:10]:
            print(f"- {reason}", file=sys.stderr)
        if len(reasons) > 10:
            print(f"- ... {len(reasons) - 10} more failure reasons", file=sys.stderr)
    else:
        print("- no failure reason details available", file=sys.stderr)

    regressions = comparison.get("regression_details") or []
    if regressions:
        print("Metric regressions:", file=sys.stderr)
        for item in regressions[:10]:
            print(
                "- {metric}: {baseline:.6f} -> {candidate:.6f} (delta {delta:.6f}, tolerance {allowed_drop:.6f})".format(
                    metric=item.get("metric", "unknown"),
                    baseline=float(item.get("baseline") or 0.0),
                    candidate=float(item.get("candidate") or 0.0),
                    delta=float(item.get("delta") or 0.0),
                    allowed_drop=float(item.get("allowed_drop") or 0.0),
                ),
                file=sys.stderr,
            )

    case_failures = comparison.get("case_failure_details") or []
    if case_failures:
        print("Newly failed cases:", file=sys.stderr)
        for item in case_failures[:10]:
            tags = item.get("tags") or []
            suffix = f" tags={','.join(tags)}" if tags else ""
            print(f"- {item.get('id')}{suffix}", file=sys.stderr)



if __name__ == "__main__":
    raise SystemExit(main())
