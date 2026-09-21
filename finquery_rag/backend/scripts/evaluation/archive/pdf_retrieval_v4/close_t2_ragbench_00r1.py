#!/usr/bin/env python3
"""T2-00R1: close the published empty-question anomaly contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ROWS = 23_088
SUBSETS = ("FinQA", "ConvFinQA", "TAT-DQA")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def metadata_paths(root: Path) -> list[tuple[str, str, Path]]:
    result = [
        ("FinQA", split, root / "data" / "FinQA" / split / "metadata.jsonl")
        for split in ("train", "dev", "test")
    ]
    result.append(("ConvFinQA", "turn_0", root / "data" / "ConvFinQA" / "turn_0.jsonl"))
    result.extend(
        ("TAT-DQA", split, root / "data" / "TAT-DQA" / split / "metadata.jsonl")
        for split in ("train", "dev", "test")
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--intake-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    intake = args.intake_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    commit = subprocess.check_output(
        ["git", "-C", str(dataset), "rev-parse", "HEAD"], text=True
    ).strip()
    intake_manifest = json.loads((intake / "dataset-manifest.json").read_text(encoding="utf-8"))
    intake_counts = json.loads((intake / "dataset-counts.json").read_text(encoding="utf-8"))
    if commit != intake_manifest["exact_commit_sha"]:
        raise RuntimeError("dataset_commit_mismatch")
    if intake_counts["actual_total_rows"] != EXPECTED_ROWS:
        raise RuntimeError("published_row_count_mismatch")
    if intake_counts["lfs_pointer_count"] != 0:
        raise RuntimeError("lfs_materialization_incomplete")

    rows: list[dict[str, Any]] = []
    subset_counts: Counter[str] = Counter()
    empty_rows: list[dict[str, Any]] = []
    missing_company = 0
    for subset, split, path in metadata_paths(dataset):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                subset_counts[subset] += 1
                company_name = row.get("company_name")
                question = row.get("question")
                if company_name is None or company_name == "":
                    missing_company += 1
                query = f"{company_name} : {question}"
                record = {
                    "subset": subset,
                    "split": row.get("split", split),
                    "id": row.get("id"),
                    "context_id": row.get("context_id"),
                    "question_raw_value": question,
                    "question_python_type": type(question).__name__,
                    "company_name": company_name,
                    "file_name": row.get("file_name"),
                    "context_present": bool(row.get("context")),
                    "retrieval_query": query,
                    "source_file": str(path.relative_to(dataset)).replace("\\", "/"),
                    "source_line": line_number,
                }
                rows.append(record)
                if question == "" or question is None:
                    empty_rows.append(record)

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"row_count_contract:{len(rows)}")
    query_contract = {
        "query_template": "f'{company_name} : {question}'",
        "python_semantics": "literal f-string; no fallback, strip, rewrite, or repair",
        "empty_question_behavior": "Company Name : ",
        "none_question_behavior": "Company Name : None",
        "query_count": len(rows),
        "company_name_missing_count": missing_company,
        "query_adapter_sha256": hashlib.sha256(
            b"retrieval_query = f\"{company_name} : {question}\""
        ).hexdigest(),
    }
    write_json(output / "empty-question-audit.json", {"count": len(empty_rows), "rows": empty_rows})
    write_json(
        output / "published-anomaly-contract.json",
        {
            "dataset_commit": commit,
            "published_rows": len(rows),
            "headline_denominator": len(rows),
            "empty_question_rows": len(empty_rows),
            "empty_question_ids": [row["id"] for row in empty_rows],
            "silent_exclusion_allowed": False,
            "question_repair_allowed": False,
            "published_raw_track_ready": True,
            "known_anomaly_status": "published_dataset_known_anomaly",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(output / "official-query-contract.json", query_contract)
    write_json(
        output / "acceptance.json",
        {
            "dataset_materialization_accepted": True,
            "published_raw_track_ready": True,
            "published_rows": len(rows),
            "headline_denominator": len(rows),
            "empty_question_rows": len(empty_rows),
            "empty_question_diagnostic_denominator": len(rows) - len(empty_rows),
            "silent_exclusion_allowed": False,
            "question_repair_allowed": False,
            "dataset_commit": commit,
            "subset_rows": dict(sorted(subset_counts.items())),
            "query_contract_sha256": sha256(output / "official-query-contract.json"),
            "decision": "published_raw_compatibility_closed",
            "next_gate": "t2_ragbench_01_standard_whole_context_retrieval",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

