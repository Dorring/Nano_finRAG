#!/usr/bin/env python3
"""Run the frozen TV2-07R1 canonical production-readiness evaluation.

The command performs a preflight first.  Without --execute, or when the
canonical set is absent/ineligible, it writes an explicit pending artifact and
does not invoke either runtime.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.tv2_07_r1_readiness import (
    build_tv2_07_r1_manifest,
    build_tv2_07_r1_preflight,
    finalize_r1_decision,
    load_tv2_07_r1_dataset,
    write_tv2_07_r1_artifacts,
    write_tv2_07_r1_pending_artifacts,
)
from src.evaluation.tv2_07_readiness import (
    TV2IntegratedEvaluationRunner,
    finalize_tv2_07_manifest,
    score_predictions,
)


def _load_callable(path: str) -> Callable[..., Any]:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("callable must use module:attribute syntax")
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    if not callable(value):
        raise TypeError(f"{path!r} is not callable")
    return value


def _json_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "production_runtime": "V1",
            "v2_authority": "OFF",
            "canary": "NOT_STARTED",
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("runtime config must contain a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        default=Path("tests/fixtures/tv2_07_canonical_readiness/questions.jsonl"),
        type=Path,
    )
    parser.add_argument(
        "--labels",
        default=Path("tests/fixtures/tv2_07_canonical_readiness/labels.jsonl"),
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=Path("artifacts/evaluation/tv2-07-production-readiness/canonical-r1"),
        type=Path,
    )
    parser.add_argument("--repo-path", default=Path("."), type=Path)
    parser.add_argument(
        "--corpus-freeze",
        default=Path(
            "artifacts/evaluation/nf-v2-17-financial-corpus-v2/"
            "financial-corpus-v2-freeze.json"
        ),
        type=Path,
    )
    parser.add_argument(
        "--index-config",
        default=Path(
            "artifacts/evaluation/nf-v2-17-financial-corpus-v2/index-config-v2.json"
        ),
        type=Path,
    )
    parser.add_argument(
        "--index-build",
        default=Path(
            "artifacts/evaluation/nf-v2-17-financial-corpus-v2/index-build-report.json"
        ),
        type=Path,
    )
    parser.add_argument(
        "--index-integrity",
        default=Path(
            "artifacts/evaluation/nf-v2-17-financial-corpus-v2/index-integrity.json"
        ),
        type=Path,
    )
    parser.add_argument(
        "--model-manifest",
        default=Path(
            "artifacts/runtime/nf-v2-21-local-specialist-integration/"
            "runtime-model-config.json"
        ),
        type=Path,
    )
    parser.add_argument(
        "--raw-corpus-manifest",
        default=Path(
            "artifacts/evaluation/nf-v2-17-financial-corpus-v2/"
            "raw-corpus-manifest-v2.jsonl"
        ),
        type=Path,
    )
    parser.add_argument(
        "--parsed-corpus-manifest",
        default=Path(
            "artifacts/evaluation/nf-v2-17-financial-corpus-v2/"
            "parsed-corpus-manifest-v2.jsonl"
        ),
        type=Path,
    )
    parser.add_argument(
        "--wiring-queries",
        default=Path("tests/fixtures/tv2_07_production_readiness/questions.jsonl"),
        type=Path,
    )
    parser.add_argument(
        "--wiring-labels",
        default=Path("tests/fixtures/tv2_07_production_readiness/labels.jsonl"),
        type=Path,
    )
    parser.add_argument("--min-cases", default=100, type=int)
    parser.add_argument("--verify-index-paths", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--v1-factory")
    parser.add_argument("--v2-factory")
    parser.add_argument("--request-factory")
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--timeout-seconds", default=120.0, type=float)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--quality-reviewed", action="store_true")
    parser.add_argument("--latency-reviewed", action="store_true")
    parser.add_argument("--qualitative-reviewed", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> int:
    preflight = build_tv2_07_r1_preflight(
        repo_path=args.repo_path,
        queries_path=args.queries,
        labels_path=args.labels,
        corpus_freeze_path=args.corpus_freeze,
        index_config_path=args.index_config,
        index_build_path=args.index_build,
        index_integrity_path=args.index_integrity,
        model_manifest_path=args.model_manifest,
        raw_corpus_manifest_path=args.raw_corpus_manifest,
        parsed_corpus_manifest_path=args.parsed_corpus_manifest,
        wiring_queries_path=args.wiring_queries,
        wiring_labels_path=args.wiring_labels,
        min_cases=args.min_cases,
        verify_index_paths=args.verify_index_paths,
    )
    if not args.execute or not preflight.ready_to_run:
        write_tv2_07_r1_pending_artifacts(
            args.output,
            preflight=preflight,
            repo_path=args.repo_path,
        )
        print("HOLD_FOR_QUALITY")
        return 0

    if not args.v1_factory or not args.v2_factory:
        raise ValueError("--execute requires --v1-factory and --v2-factory")
    queries, labels = load_tv2_07_r1_dataset(args.queries, args.labels)
    runtime_config = _json_config(args.runtime_config)
    runtime_config.update({
        "timeout_seconds": args.timeout_seconds,
        "production_runtime": "V1",
        "v2_authority": "OFF",
        "canary": "NOT_STARTED",
    })
    request_factory = (
        _load_callable(args.request_factory)
        if args.request_factory
        else None
    )
    runner = TV2IntegratedEvaluationRunner(
        _load_callable(args.v1_factory),
        _load_callable(args.v2_factory),
        timeout_seconds=args.timeout_seconds,
        request_factory=request_factory,
    )
    predictions = await runner.run(queries)
    scored, metrics = score_predictions(
        predictions,
        labels,
        evaluation_scope_complete=True,
        corpus_verified=True,
        canonical_model_verified=True,
    )
    metrics = finalize_r1_decision(
        metrics,
        preflight,
        quality_reviewed=args.quality_reviewed,
        latency_reviewed=args.latency_reviewed,
        qualitative_reviewed=args.qualitative_reviewed,
    )
    manifest = build_tv2_07_r1_manifest(
        repo_path=args.repo_path,
        queries=queries,
        labels=labels,
        preflight=preflight,
        runtime_config=runtime_config,
        random_seed=args.random_seed,
    )
    manifest = finalize_tv2_07_manifest(manifest, repo_path=args.repo_path)
    write_tv2_07_r1_artifacts(
        args.output,
        manifest=manifest,
        preflight=preflight,
        queries=queries,
        labels=labels,
        scored_cases=scored,
        metrics=metrics,
    )
    print(metrics.decision.value)
    return 2 if metrics.decision.value == "BLOCKED_FOR_SAFETY" else 0


def main() -> int:
    return asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
