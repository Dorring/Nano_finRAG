#!/usr/bin/env python3
"""Run the frozen TV2-07 readiness harness with injected runtime factories.

Factories are imported by dotted module path and must return a
FinancialQARuntime. The runner itself never passes labels to either factory.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.tv2_07_readiness import (
    TV2IntegratedEvaluationRunner,
    build_tv2_07_manifest,
    finalize_tv2_07_manifest,
    load_tv2_07_dataset,
    score_predictions,
    write_tv2_07_artifacts,
)


def _factory(path: str) -> Callable[[], Any]:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    if not callable(value):
        raise TypeError(f"factory {path!r} is not callable")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--v1-factory", required=True)
    parser.add_argument("--v2-factory", required=True)
    parser.add_argument("--repo-path", default=".", type=Path)
    parser.add_argument("--timeout-seconds", default=120.0, type=float)
    parser.add_argument("--corpus-sha")
    parser.add_argument("--model-checkpoint")
    parser.add_argument("--scope-complete", action="store_true")
    parser.add_argument("--corpus-verified", action="store_true")
    parser.add_argument("--canonical-model-verified", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> int:
    queries, labels = load_tv2_07_dataset(args.queries, args.labels)
    manifest = build_tv2_07_manifest(
        repo_path=args.repo_path,
        queries=queries,
        labels=labels,
        runtime_config={
            "timeout_seconds": args.timeout_seconds,
            "production_runtime": "V1",
            "v2_authority": "OFF",
        },
        corpus_hash=args.corpus_sha,
        model_checkpoint=args.model_checkpoint,
    )
    runner = TV2IntegratedEvaluationRunner(
        _factory(args.v1_factory),
        _factory(args.v2_factory),
        timeout_seconds=args.timeout_seconds,
    )
    predictions = await runner.run(queries)
    scored, metrics = score_predictions(
        predictions,
        labels,
        evaluation_scope_complete=args.scope_complete,
        corpus_verified=args.corpus_verified,
        canonical_model_verified=args.canonical_model_verified,
    )
    manifest = finalize_tv2_07_manifest(manifest, repo_path=args.repo_path)
    write_tv2_07_artifacts(
        args.output,
        manifest=manifest,
        queries=queries,
        labels=labels,
        scored_cases=scored,
        metrics=metrics,
        runtime_manifest={
            "production_runtime": "V1",
            "evaluation_runtime": "TrustedFinancialRuntimeV2",
            "v1_factory": args.v1_factory,
            "v2_factory": args.v2_factory,
            "gold_evidence_injection": False,
        },
    )
    print(metrics.decision.value)
    return 0 if metrics.decision.value != "BLOCKED_FOR_SAFETY" else 2


def main() -> int:
    return asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
