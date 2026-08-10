#!/usr/bin/env python3
"""T2-04A official split freeze and structure-audit gate.

The first phase is deliberately fail-closed.  The published ConvFinQA
configuration has split=all rather than train/dev/test; this script records
that fact instead of silently assigning those rows to a split.  Structure
features are not generated when the official split contract is invalid.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Iterable

EXPECTED_ROWS = 23_088
DATASET_COMMIT = "adf7fe1541ac37351ce1142544d8e3b43010ed92"
REQUIRED_SPLITS = {"train", "dev", "test"}
VALID_SUBSETS = {"FinQA", "ConvFinQA", "TAT-DQA"}
METRIC_WORDS = {
    "assets", "cash", "cost", "debt", "expense", "flow", "income", "margin",
    "net", "profit", "revenue", "sales", "tax", "volume", "earnings",
    "liabilities", "equity", "transactions", "payments", "operating",
}


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_metric_tokens(value: str) -> tuple[str, ...]:
    tokens = []
    for token in normalize_text(value).split():
        if token in {"the", "of", "for", "a", "an"}:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.append(token)
    return tuple(sorted(tokens))


def extract_periods(value: str) -> list[str]:
    text = str(value)
    matches = re.findall(r"(?i)\b(?:fy\s*)?(?:19|20)\d{2}\b|\bq[1-4]\b", text)
    return sorted({normalize_text(match).replace(" ", "") for match in matches})


def extract_entities(question: str, company_name: str | None = None) -> list[str]:
    candidates = []
    if company_name:
        candidates.append(company_name)
    prefix = str(question).split(":", 1)[0].strip()
    if prefix:
        candidates.append(prefix)
    return sorted({normalize_text(item) for item in candidates if normalize_text(item)})


def extract_metric_terms(question: str) -> list[str]:
    tokens = normalize_text(question).split()
    terms = []
    for index, token in enumerate(tokens):
        if token in METRIC_WORDS:
            window = tokens[max(0, index - 2): min(len(tokens), index + 3)]
            terms.append(" ".join(window))
            terms.append(token)
    return sorted(set(terms))


def extract_operation_intent(question: str) -> str:
    text = normalize_text(question)
    if any(word in text for word in ("percentage change", "percent change", "increase", "decrease", "growth")):
        return "percentage_change"
    if "difference" in text or "net change" in text:
        return "difference"
    if "average" in text or "per transaction" in text:
        return "average"
    if "ratio" in text or "divided by" in text:
        return "ratio"
    if "percentage" in text or "percent" in text:
        return "percentage"
    if "sum" in text or "combined" in text or "total of" in text:
        return "sum"
    if "compare" in text or "compared" in text or "versus" in text:
        return "comparison"
    return "direct_fact"


def extract_query_structure(question: str, company_name: str | None = None) -> dict[str, Any]:
    periods = extract_periods(question)
    return {
        "entity": extract_entities(question, company_name),
        "metric_terms": extract_metric_terms(question),
        "periods": periods,
        "operation_intent": extract_operation_intent(question),
        "requires_multiple_periods": len(periods) > 1,
        "currency_terms": sorted(set(re.findall(r"(?i)\b(?:usd|dollars?|euros?|gbp|eur)\b", question))),
    }


def _metadata_paths(dataset_root: Path) -> Iterable[tuple[str, str | None, Path]]:
    for split in ("train", "dev", "test"):
        yield "FinQA", split, dataset_root / "data" / "FinQA" / split / "metadata.jsonl"
        yield "TAT-DQA", split, dataset_root / "data" / "TAT-DQA" / split / "metadata.jsonl"
    yield "ConvFinQA", "all", dataset_root / "data" / "ConvFinQA" / "turn_0.jsonl"


def load_published_rows(dataset_root: Path) -> list[dict[str, Any]]:
    rows = []
    for subset, expected_split, path in _metadata_paths(dataset_root):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                split = str(source.get("split", expected_split or ""))
                if split != expected_split:
                    raise RuntimeError(f"split_field_mismatch:{source.get('id')}:{split}:{expected_split}")
                rows.append(
                    {
                        "query_id": str(source["id"]),
                        "subset": subset,
                        "split": split,
                        "context_id": str(source["context_id"]),
                        "file_name": source.get("file_name"),
                        "question": source.get("question"),
                        "question_type": type(source.get("question")).__name__,
                        "company_name": source.get("company_name"),
                    }
                )
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"published_row_count:{len(rows)}")
    return rows


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_ids(path: Path, ids: list[str]) -> str:
    data = json.dumps(sorted(ids), ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    path.write_bytes(data)
    return sha256_bytes(data)


def git_commit(dataset_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(dataset_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_placeholder_gzip(path: Path, reason: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"status": "not_run", "reason": reason}) + "\n")


def run_audit(dataset_root: Path, prediction_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    actual_commit = git_commit(dataset_root)
    if actual_commit != DATASET_COMMIT:
        raise RuntimeError(f"dataset_commit:{actual_commit}")
    protocol = json.loads((prediction_root / "protocol.json").read_text())
    if protocol.get("dataset_commit") != DATASET_COMMIT:
        raise RuntimeError("prediction_dataset_commit")
    rows = load_published_rows(dataset_root)
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids = []
    for row in rows:
        if row["query_id"] in by_id:
            duplicate_ids.append(row["query_id"])
        by_id[row["query_id"]] = row

    split_ids = {split: [] for split in REQUIRED_SPLITS}
    all_ids: list[str] = []
    unknown_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] in REQUIRED_SPLITS:
            split_ids[row["split"]].append(row["query_id"])
        else:
            all_ids.append(row["query_id"])
            unknown_rows.append(row)

    overlaps = {
        "train_dev": len(set(split_ids["train"]) & set(split_ids["dev"])),
        "train_test": len(set(split_ids["train"]) & set(split_ids["test"])),
        "dev_test": len(set(split_ids["dev"]) & set(split_ids["test"])),
    }
    recognized_union = set().union(*(set(values) for values in split_ids.values()))
    all_union = recognized_union | set(all_ids)
    unknown_split_values = sorted({row["split"] for row in unknown_rows})
    missing_from_union = sorted(set(by_id) - all_union)
    split_contract_accepted = (
        not duplicate_ids
        and not missing_from_union
        and not any(overlaps.values())
        and not unknown_rows
        and len(all_union) == EXPECTED_ROWS
    )

    hashes = {}
    for split, ids in split_ids.items():
        hashes[split] = write_ids(output_root / f"{split}-query-ids.json", ids)
    hashes["all"] = write_ids(output_root / "all-query-ids.json", all_ids)
    hashes["unmapped"] = write_ids(output_root / "unmapped-query-ids.json", all_ids)
    split_manifest = {
        "gate": "T2-04A",
        "dataset_commit": actual_commit,
        "formal_denominator": EXPECTED_ROWS,
        "row_count": len(rows),
        "query_counts": {split: len(ids) for split, ids in split_ids.items()},
        "all_split_count": len(all_ids),
        "recognized_union_count": len(recognized_union),
        "actual_split_distribution": {
            f"{row['subset']}:{row['split']}": sum(
                candidate["subset"] == row["subset"] and candidate["split"] == row["split"]
                for candidate in rows
            )
            for row in rows
        },
        "unknown_split_values": unknown_split_values,
        "unknown_split_count": len(unknown_rows),
        "duplicate_query_ids": sorted(set(duplicate_ids)),
        "cross_split_overlap": overlaps,
        "missing_from_union": missing_from_union,
        "empty_question_count": sum(row["question"] == "" for row in rows),
        "empty_question_ids": sorted(row["query_id"] for row in rows if row["question"] == ""),
        "train_dev_test_union_covers_formal_denominator": len(recognized_union) == EXPECTED_ROWS,
        "split_contract_accepted": split_contract_accepted,
    }
    (output_root / "split-manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "split-hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    frozen_metrics = json.loads(
        (prediction_root / "weighted-metrics.json").read_text()
    )["baselines"]["bm25"]
    (output_root / "split-baseline-metrics.json").write_text(
        json.dumps(
            {
                "status": "not_run_split_contract_blocked",
                "frozen_whole_dataset_bm25": frozen_metrics,
                "formal_denominator": EXPECTED_ROWS,
                "retrieval_rerun": False,
                "reason": "ConvFinQA split=all cannot be assigned to train/dev/test",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    placeholder_reason = "split_contract_not_accepted"
    for name in (
        "query-structure.jsonl.gz",
        "context-structure.jsonl.gz",
        "candidate-structure-features.jsonl.gz",
    ):
        write_placeholder_gzip(output_root / name, placeholder_reason)
    for name in (
        "feature-seal.json",
        "extraction-coverage.json",
        "feature-separability.json",
        "single-feature-ranking.json",
        "subset-analysis.json",
        "query-type-analysis.json",
    ):
        (output_root / name).write_text(
            json.dumps({"status": "not_run", "reason": placeholder_reason}, indent=2) + "\n",
            encoding="utf-8",
        )

    decision = {
        "gate": "T2-04A",
        "formal_denominator": EXPECTED_ROWS,
        "dataset_commit": actual_commit,
        "train_queries": len(split_ids["train"]),
        "dev_queries": len(split_ids["dev"]),
        "test_queries": len(split_ids["test"]),
        "all_split_queries": len(all_ids),
        "split_overlap_count": sum(overlaps.values()),
        "split_contract_accepted": split_contract_accepted,
        "bm25_retrieval_rerun": False,
        "model_execution": False,
        "gold_reads_before_feature_seal": 0,
        "feature_extraction_started": False,
        "entity_extraction_coverage": None,
        "metric_extraction_coverage": None,
        "period_extraction_coverage": None,
        "table_structure_coverage": None,
        "structure_signal_supported": None,
        "strongest_structure_signals": [],
        "next_gate": "t2_04a_split_contract_review",
        "decision_reason": [
            "ConvFinQA published file exposes split=all, not train/dev/test",
            "3458 queries cannot be assigned without an unregistered split rule",
            "feature extraction and Gold separability analysis stopped fail-closed",
        ],
    }
    (output_root / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "# T2-04A\n\n"
        "The official split contract is blocked fail-closed. FinQA and TAT-DQA "
        "provide train/dev/test, but the published ConvFinQA turn_0 file contains "
        "3,458 rows with split=all. Those rows are preserved in all/unmapped lists "
        "and are not silently assigned to a split. No structure features or Gold "
        "separability analysis were run. The frozen whole-dataset BM25 baseline "
        "is preserved without retrieval rerun.\n",
        encoding="utf-8",
    )
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    decision = run_audit(
        args.dataset_root.resolve(),
        args.prediction_root.resolve(),
        args.output_root.resolve(),
    )
    return 0 if decision["split_contract_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

