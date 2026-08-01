"""Validate and report the Financial RAG v1 document whitelist.

The default mode is contract-only: it validates the corpus whitelist and
emits zero stage violations without running retrieval.  An optional JSON
fixture can exercise the same filter contract for offline tests or adapters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.evaluation.benchmark_scope import (
    benchmark_document_ids,
    validate_scope_pipeline,
)


def _load_fixture(path: Path | None) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if path is None:
        return {}, []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("scope fixture must be a JSON object")
    stages = value.get("stages", {})
    citations = value.get("citations", [])
    if not isinstance(stages, dict) or not isinstance(citations, list):
        raise ValueError("scope fixture stages/citations have invalid types")
    return {
        str(stage): list(candidates)
        for stage, candidates in stages.items()
        if isinstance(candidates, list)
    }, citations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/financial_rag_v1/corpus.json"),
    )
    parser.add_argument("--fixture", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/evaluation/nf-eval-01/scope-integrity-report.json"),
    )
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    allowed = benchmark_document_ids(corpus)
    stages, citations = _load_fixture(args.fixture)
    report = validate_scope_pipeline(stages, allowed, citations=citations)
    report.update(
        {
            "benchmark_id": corpus.get("benchmark_id", "financial-rag-v1"),
            "scope_mode": "fixture" if args.fixture else "contract_only",
            "allowed_document_ids_hash": hashlib.sha256(
                json.dumps(sorted(allowed), separators=(",", ":")).encode()
            ).hexdigest(),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["scope_integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
