"""Validate Financial RAG Benchmark v1 catalog and reviewed annotation labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.financial_benchmark import (
    taxonomy_counts,
    validate_annotation_cases,
    validate_document_catalog,
)


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    args = parser.parse_args()
    documents = json.loads(args.catalog.read_text(encoding="utf-8"))["documents"]
    cases = _jsonl(args.labels)
    issues = validate_document_catalog(documents)
    issues.extend(validate_annotation_cases(cases, allowed_document_ids={item["document_id"] for item in documents}))
    report = {
        "document_count": len(documents),
        "case_count": len(cases),
        "taxonomy_counts": taxonomy_counts(cases),
        "issues": [{"record_id": item.record_id, "message": item.message} for item in issues],
        "passed": not issues,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
