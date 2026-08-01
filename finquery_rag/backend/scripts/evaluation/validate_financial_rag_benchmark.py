"""Validate Financial RAG Benchmark v1 draft or reviewed QA files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.evaluation.benchmark_foundation import load_json, load_jsonl, validate_dataset
    from scripts.evaluation.draft_quality import quality_audit
except ModuleNotFoundError:  # direct script execution from backend
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.evaluation.benchmark_foundation import load_json, load_jsonl, validate_dataset
    from scripts.evaluation.draft_quality import quality_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("benchmarks/financial_rag_v1/corpus.json"))
    parser.add_argument("--questions", type=Path, default=Path("benchmarks/financial_rag_v1/data/questions.draft.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("benchmarks/financial_rag_v1/data/labels.draft.jsonl"))
    parser.add_argument("--review", type=Path, default=Path("benchmarks/financial_rag_v1/data/review-status.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/evaluation/nf-eval-01/draft-validation-report.json"))
    parser.add_argument("--reviewed", action="store_true", help="apply stricter reviewed/golden answer checks")
    args = parser.parse_args()
    result = validate_dataset(
        corpus=load_json(args.corpus),
        questions=load_jsonl(args.questions),
        labels=load_jsonl(args.labels),
        review_records=load_jsonl(args.review),
        draft=not args.reviewed,
    )
    result["draft_mode"] = not args.reviewed
    result["quality"] = quality_audit(
        questions=load_jsonl(args.questions),
        labels=load_jsonl(args.labels),
        reviews=load_jsonl(args.review),
    )
    result["golden_case_count"] = 0 if not args.reviewed else sum(1 for item in load_jsonl(args.review) if item.get("ready_for_golden"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["schema_valid"] and result["quality"]["quality_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
