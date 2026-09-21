"""Close the Phase 1.2 authoring contract without promoting any Draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluation.annotation_contract import (
    annotation_contract_report,
    build_annotation_worklist,
    close_annotation_contract,
)
from scripts.evaluation.benchmark_foundation import load_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("benchmarks/financial_rag_v1/data/questions.draft.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("benchmarks/financial_rag_v1/data/labels.draft.jsonl"))
    parser.add_argument("--review", type=Path, default=Path("benchmarks/financial_rag_v1/data/review-status.jsonl"))
    parser.add_argument("--worklist", type=Path, default=Path("benchmarks/financial_rag_v1/data/annotation-worklist.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/evaluation/nf-eval-01/annotation-contract-report.json"))
    args = parser.parse_args()

    questions, labels, reviews = close_annotation_contract(
        load_jsonl(args.questions),
        load_jsonl(args.labels),
        load_jsonl(args.review),
    )
    write_jsonl(args.questions, questions)
    write_jsonl(args.labels, labels)
    write_jsonl(args.review, reviews)
    write_jsonl(args.worklist, build_annotation_worklist(questions, labels, reviews))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = annotation_contract_report(questions, labels, reviews)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["contract_valid"] and report["golden_case_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
