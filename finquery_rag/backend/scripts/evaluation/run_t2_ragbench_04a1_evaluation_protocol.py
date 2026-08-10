#!/usr/bin/env python3
"""T2-04A.1: freeze the published T2-RAGBench evaluation protocol."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

EXPECTED_ROWS = 23_088
DATASET_COMMIT = "adf7fe1541ac37351ce1142544d8e3b43010ed92"
PRIMARY_SUBSETS = ("FinQA", "TAT-DQA")
NATIVE_SPLITS = ("train", "dev", "test")
KS = (1, 3, 5, 10, 20, 50, 100)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_ids(path: Path, ids: Iterable[str]) -> str:
    data = (
        json.dumps(sorted(str(value) for value in ids), ensure_ascii=False, indent=2)
        + "\n"
    ).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def dataset_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def metadata_paths(root: Path) -> list[tuple[str, str, Path]]:
    paths = [
        (subset, split, root / "data" / subset / split / "metadata.jsonl")
        for subset in ("FinQA", "TAT-DQA")
        for split in NATIVE_SPLITS
    ]
    paths.append(("ConvFinQA", "all", root / "data" / "ConvFinQA" / "turn_0.jsonl"))
    return paths


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for subset, expected_split, path in metadata_paths(root):
        if not path.exists():
            raise RuntimeError(f"missing_dataset_file:{path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                query_id = str(source["id"])
                split = str(source.get("split", expected_split))
                if split != expected_split:
                    raise RuntimeError(
                        f"split_field_mismatch:{query_id}:{split}:{expected_split}"
                    )
                rows.append(
                    {
                        "query_id": query_id,
                        "subset": subset,
                        "split": split,
                        "context_id": str(source["context_id"]),
                        "file_name": (
                            str(source["file_name"])
                            if source.get("file_name") is not None
                            else None
                        ),
                        "question": source.get("question"),
                        "question_python_type": type(
                            source.get("question")
                        ).__name__,
                    }
                )
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"published_row_count:{len(rows)}")
    return rows


def load_bm25(
    prediction_root: Path, expected_ids: set[str]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    seal_path = prediction_root / "prediction-seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("sealed") or seal.get("gold_scoring_reads_before_seal") != 0:
        raise RuntimeError("invalid_t2_01_prediction_seal")
    if seal.get("prediction_count") != EXPECTED_ROWS:
        raise RuntimeError("invalid_t2_01_prediction_count")
    prediction_path = prediction_root / "bm25-predictions.jsonl.gz"
    expected_hash = seal.get("output_sha256", {}).get("bm25")
    if not expected_hash or sha256_file(prediction_path) != expected_hash:
        raise RuntimeError("t2_01_bm25_mutated")
    predictions: dict[str, list[str]] = {}
    with gzip.open(prediction_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id in predictions:
                raise RuntimeError(f"duplicate_prediction:{query_id}")
            ranked = row.get("ranked_contexts") or []
            ids = [str(item["context_id"]) for item in ranked]
            ranks = [int(item["rank"]) for item in ranked]
            if len(ids) != 100 or len(ids) != len(set(ids)):
                raise RuntimeError(f"invalid_candidate_depth:{query_id}")
            if ranks != list(range(1, 101)):
                raise RuntimeError(f"invalid_candidate_ranks:{query_id}")
            predictions[query_id] = ids
    if set(predictions) != expected_ids:
        raise RuntimeError("prediction_identity_mismatch")
    return predictions, {
        "prediction_count": len(predictions),
        "candidate_depth": 100,
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_seal_sha256": sha256_file(seal_path),
        "gold_scoring_reads_before_seal": 0,
        "retrieval_rerun": False,
    }


def empty_stat() -> dict[str, Any]:
    return {
        "count": 0,
        "hits": {str(k): 0 for k in KS},
        "mrr": 0.0,
        "ndcg5": 0.0,
        "ndcg10": 0.0,
    }


def add_stat(stat: dict[str, Any], rank: int | None) -> None:
    stat["count"] += 1
    for cutoff in KS:
        if rank is not None and rank <= cutoff:
            stat["hits"][str(cutoff)] += 1
    if rank is not None and rank <= 5:
        stat["mrr"] += 1.0 / rank
        stat["ndcg5"] += 1.0 / math.log2(rank + 1.0)
    if rank is not None and rank <= 10:
        stat["ndcg10"] += 1.0 / math.log2(rank + 1.0)


def pct(value: float) -> float:
    return round(value * 100.0, 6)


def finalize(stat: dict[str, Any]) -> dict[str, Any]:
    total = stat["count"]
    mrr = stat["mrr"] / total if total else 0.0
    ndcg5 = stat["ndcg5"] / total if total else 0.0
    ndcg10 = stat["ndcg10"] / total if total else 0.0
    return {
        "count": total,
        "hits": {str(k): int(stat["hits"][str(k)]) for k in KS},
        "recall": {
            f"@{k}": (
                f"{stat['hits'][str(k)]}/{total}" if total else "0/0"
            )
            for k in KS
        },
        "recall_pct": {
            f"@{k}": pct(stat["hits"][str(k)] / total) if total else 0.0
            for k in KS
        },
        "mrr": mrr,
        "mrr_pct": pct(mrr),
        "mrr_at_5": mrr,
        "mrr_at_5_pct": pct(mrr),
        "ndcg_at_5": ndcg5,
        "ndcg_at_5_pct": pct(ndcg5),
        "ndcg_at_10": ndcg10,
        "ndcg_at_10_pct": pct(ndcg10),
    }


def score_ids(
    ids: Iterable[str],
    rows_by_id: dict[str, dict[str, Any]],
    predictions: dict[str, list[str]],
) -> dict[str, Any]:
    stat = empty_stat()
    for query_id in ids:
        ranked = predictions.get(query_id)
        if ranked is None:
            raise RuntimeError(f"missing_prediction:{query_id}")
        try:
            rank = ranked.index(rows_by_id[query_id]["context_id"]) + 1
        except ValueError:
            rank = None
        add_stat(stat, rank)
    return finalize(stat)


def partition(rows: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    groups = {
        "primary_train": [],
        "primary_dev": [],
        "primary_test": [],
        "convfinqa_transfer": [],
    }
    for row in rows:
        if row["subset"] in PRIMARY_SUBSETS:
            groups[f"primary_{row['split']}"].append(row["query_id"])
        elif row["subset"] == "ConvFinQA":
            if row["split"] != "all":
                raise RuntimeError(f"convfinqa_non_all:{row['query_id']}")
            groups["convfinqa_transfer"].append(row["query_id"])
    return {key: sorted(value) for key, value in groups.items()}


def pairwise(values: dict[str, set[str]]) -> dict[str, int]:
    names = ("train", "dev", "test")
    return {
        f"{left}_{right}": len(values[left] & values[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }


def overlap_audit(
    rows_by_id: dict[str, dict[str, Any]], groups: dict[str, list[str]]
) -> dict[str, Any]:
    dimensions = {}
    for label, field in (
        ("query", "query_id"),
        ("context", "context_id"),
        ("document", "file_name"),
    ):
        values = {}
        for split in NATIVE_SPLITS:
            values[split] = {
                str(rows_by_id[query_id][field])
                for query_id in groups[f"primary_{split}"]
                if rows_by_id[query_id][field] is not None
            }
        dimensions[label] = values
    return {
        "primary_query_overlap": pairwise(dimensions["query"]),
        "primary_context_id_overlap": pairwise(dimensions["context"]),
        "primary_file_document_overlap": pairwise(dimensions["document"]),
        "notes": {
            "query_overlap_required_zero": True,
            "context_overlap_report_only": True,
            "file_name_used_as_document_identity": True,
        },
    }


def split_contract(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    values: dict[str, set[str]] = {}
    for row in rows:
        key = f"{row['subset']}:{row['split']}"
        counts[key] = counts.get(key, 0) + 1
        values.setdefault(row["subset"], set()).add(row["split"])
    return {
        "counts": dict(sorted(counts.items())),
        "split_values": {
            key: sorted(value) for key, value in sorted(values.items())
        },
        "native_benchmark_splits": {
            "FinQA": list(NATIVE_SPLITS),
            "TAT-DQA": list(NATIVE_SPLITS),
            "ConvFinQA": ["all"],
        },
        "published_file_contract": {
            "FinQA": "data/FinQA/{train,dev,test}/metadata.jsonl",
            "TAT-DQA": "data/TAT-DQA/{train,dev,test}/metadata.jsonl",
            "ConvFinQA": "data/ConvFinQA/turn_0.jsonl",
        },
    }


def empty_audit(
    rows_by_id: dict[str, dict[str, Any]], groups: dict[str, list[str]]
) -> dict[str, Any]:
    empty = [row for row in rows_by_id.values() if row["question"] == ""]
    grouped = {}
    for group, ids in groups.items():
        selected = [
            rows_by_id[query_id]
            for query_id in ids
            if rows_by_id[query_id]["question"] == ""
        ]
        grouped[group] = {
            "count": len(selected),
            "rows": [
                {
                    "query_id": row["query_id"],
                    "subset": row["subset"],
                    "split": row["split"],
                    "context_id": row["context_id"],
                    "file_name": row["file_name"],
                    "question_raw_value": row["question"],
                    "question_python_type": row["question_python_type"],
                }
                for row in sorted(selected, key=lambda item: item["query_id"])
            ],
        }
    return {
        "published_denominator": EXPECTED_ROWS,
        "empty_question_count": len(empty),
        "empty_question_ids": sorted(row["query_id"] for row in empty),
        "null_question_count": sum(
            row["question"] is None for row in rows_by_id.values()
        ),
        "by_track_group": grouped,
        "silent_exclusion_allowed": False,
        "question_repair_allowed": False,
    }


def main_protocol(
    dataset_root: Path, prediction_root: Path, output_root: Path
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    if dataset_commit(dataset_root) != DATASET_COMMIT:
        raise RuntimeError("dataset_commit_mismatch")
    rows = load_rows(dataset_root)
    rows_by_id = {row["query_id"]: row for row in rows}
    if len(rows_by_id) != EXPECTED_ROWS:
        raise RuntimeError("duplicate_query_ids")
    predictions, integrity = load_bm25(prediction_root, set(rows_by_id))
    groups = partition(rows)
    expected = {
        "primary_train": 15314,
        "primary_dev": 2025,
        "primary_test": 2291,
        "convfinqa_transfer": 3458,
    }
    counts = {key: len(value) for key, value in groups.items()}
    if counts != expected:
        raise RuntimeError(f"track_counts:{counts}")
    split_hashes = {}
    for split in NATIVE_SPLITS:
        split_hashes[f"primary_{split}"] = write_ids(
            output_root / f"primary-{split}-query-ids.json",
            groups[f"primary_{split}"],
        )
    split_hashes["convfinqa_transfer"] = write_ids(
        output_root / "convfinqa-transfer-query-ids.json",
        groups["convfinqa_transfer"],
    )
    write_json(
        output_root / "split-hashes.json",
        {
            "dataset_commit": DATASET_COMMIT,
            "canonical_encoding": "sorted JSON array UTF-8 plus trailing newline",
            "files": {
                key: {
                    "path": (
                        f"primary-{key[8:]}-query-ids.json"
                        if key.startswith("primary_")
                        else "convfinqa-transfer-query-ids.json"
                    ),
                    "sha256": value,
                }
                for key, value in split_hashes.items()
            },
        },
    )
    write_json(
        output_root / "published-split-contract.json",
        {
            "gate": "T2-04A.1",
            "dataset_repo": "G4KMU/t2-ragbench",
            "dataset_commit": DATASET_COMMIT,
            "published_rows": EXPECTED_ROWS,
            "native_split_inventory": split_contract(rows),
            "native_subsets": ["FinQA", "TAT-DQA"],
            "secondary_published_track": {
                "subset": "ConvFinQA",
                "published_file": "data/ConvFinQA/turn_0.jsonl",
                "published_split_value": "all",
                "evaluation_label": "all / turn_0",
                "role": "secondary_transfer_evaluation",
            },
            "historical_whole_dataset_split_contract_accepted": False,
            "split_contract_accepted": False,
            "custom_split_created": False,
            "convfinqa_assigned_to_primary_split": False,
        },
    )
    primary_metrics = {
        group: score_ids(ids, rows_by_id, predictions)
        for group, ids in groups.items()
        if group.startswith("primary_")
    }
    finqa_test = [
        row["query_id"]
        for row in rows
        if row["subset"] == "FinQA" and row["split"] == "test"
    ]
    tatqa_test = [
        row["query_id"]
        for row in rows
        if row["subset"] == "TAT-DQA" and row["split"] == "test"
    ]
    primary_test_by_subset = {
        "FinQA_test": score_ids(finqa_test, rows_by_id, predictions),
        "TAT-DQA_test": score_ids(tatqa_test, rows_by_id, predictions),
        "combined_primary_test": score_ids(
            finqa_test + tatqa_test, rows_by_id, predictions
        ),
    }
    write_json(
        output_root / "primary-baseline-metrics.json",
        {
            "gate": "T2-04A.1",
            "retrieval_rerun": False,
            "prediction_artifact": "T2-01 frozen bm25-predictions.jsonl.gz",
            "primary_track": primary_metrics,
            "primary_test_by_subset": primary_test_by_subset,
        },
    )
    conv_metrics = score_ids(
        groups["convfinqa_transfer"], rows_by_id, predictions
    )
    write_json(
        output_root / "convfinqa-transfer-baseline.json",
        {
            "gate": "T2-04A.1",
            "evaluation_role": "secondary_transfer_evaluation",
            "published_split": "all / turn_0",
            "retrieval_rerun": False,
            "metrics": conv_metrics,
        },
    )
    weighted_path = prediction_root / "weighted-metrics.json"
    frozen_payload = json.loads(weighted_path.read_text(encoding="utf-8"))
    frozen = frozen_payload["baselines"]["bm25"]
    recomputed = score_ids(rows_by_id, rows_by_id, predictions)
    unchanged = (
        frozen_payload.get("denominator") == EXPECTED_ROWS
        and all(
            frozen["hits"][str(k)] == recomputed["hits"][str(k)] for k in KS
        )
        and all(
            abs(
                float(frozen["recall_pct"][f"@{k}"])
                - recomputed["recall_pct"][f"@{k}"]
            )
            < 1e-9
            for k in KS
        )
    )
    write_json(
        output_root / "whole-dataset-baseline-reference.json",
        {
            "gate": "T2-04A.1",
            "role": "diagnostic_only",
            "formal_denominator": EXPECTED_ROWS,
            "retrieval_rerun": False,
            "frozen_artifact": frozen,
            "recomputed_from_frozen_bm25_prediction": recomputed,
            "unchanged": unchanged,
            "weighted_metrics_sha256": sha256_file(weighted_path),
        },
    )
    overlap = overlap_audit(rows_by_id, groups)
    write_json(output_root / "identity-overlap-audit.json", overlap)
    empties = empty_audit(rows_by_id, groups)
    write_json(output_root / "empty-question-audit.json", empties)
    write_json(
        output_root / "input-integrity.json",
        {
            "dataset_commit": DATASET_COMMIT,
            "dataset_rows": len(rows),
            "prediction_integrity": integrity,
            "custom_split_created": False,
            "retrieval_rerun": False,
            "model_execution": False,
            "gold_identity": "context_id",
            "published_split_contract_sha256": sha256_file(
                output_root / "published-split-contract.json"
            ),
        },
    )
    query_overlap_count = sum(overlap["primary_query_overlap"].values())
    accepted = bool(
        unchanged
        and query_overlap_count == 0
        and empties["empty_question_count"] == 11
    )
    decision = {
        "gate": "T2-04A.1",
        "dataset_commit": DATASET_COMMIT,
        "formal_denominator": EXPECTED_ROWS,
        "published_rows": len(rows),
        "historical_split_contract_accepted": False,
        "split_contract_accepted": False,
        "primary_native_split_identity_complete": True,
        "primary_train_queries": len(groups["primary_train"]),
        "primary_dev_queries": len(groups["primary_dev"]),
        "primary_test_queries": len(groups["primary_test"]),
        "convfinqa_queries": len(groups["convfinqa_transfer"]),
        "custom_split_created": False,
        "query_overlap_count": query_overlap_count,
        "context_overlap_reported": True,
        "file_document_overlap_reported": True,
        "empty_questions_retained": empties["empty_question_count"] == 11,
        "bm25_prediction_loaded_not_rerun": True,
        "whole_dataset_bm25_unchanged": unchanged,
        "evaluation_protocol_accepted": accepted,
        "primary_track": "finqa_tatdqa_native_split",
        "whole_dataset_role": "diagnostic_only",
        "convfinqa_role": "secondary_transfer_evaluation",
        "next_gate": "t2_04a2_structure_signal_audit",
        "decision_reason": [
            "FinQA and TAT-DQA retain native train/dev/test identity.",
            "ConvFinQA remains published all / turn_0 and is not assigned to a custom primary split.",
            "The historical whole-dataset split_contract_accepted=false fact is preserved.",
            "All 23088 queries, including 11 empty-question rows, remain in the formal denominator.",
        ],
    }
    write_json(
        output_root / "evaluation-protocol.json",
        {
            "protocol": "T2-RAGBench Evaluation Protocol V1",
            "dataset_commit": DATASET_COMMIT,
            "published_queries": EXPECTED_ROWS,
            "whole_dataset": {
                "queries": EXPECTED_ROWS,
                "role": "diagnostic_only",
                "split_contract_accepted": False,
            },
            "primary_track": {
                "subsets": ["FinQA", "TAT-DQA"],
                "train_queries": len(groups["primary_train"]),
                "dev_queries": len(groups["primary_dev"]),
                "test_queries": len(groups["primary_test"]),
                "split_source": "native_t2_published_split",
            },
            "convfinqa_track": {
                "queries": len(groups["convfinqa_transfer"]),
                "published_split": "all / turn_0",
                "role": "secondary_transfer_evaluation",
            },
            "custom_split_created": False,
            "evaluation_protocol_accepted": accepted,
            "next_gate": decision["next_gate"],
        },
    )
    write_json(output_root / "decision.json", decision)
    (output_root / "README.md").write_text(
        "# T2-04A.1 Evaluation Protocol V1\n\n"
        "FinQA and TAT-DQA are the primary native-split track. ConvFinQA remains "
        "the published all / turn_0 secondary transfer track. No custom split, "
        "retrieval rerun, model execution, or question repair was performed. "
        "The whole 23088-query result remains diagnostic-only.\n",
        encoding="utf-8",
    )
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    decision = main_protocol(
        args.dataset_root.resolve(),
        args.prediction_root.resolve(),
        args.output_root.resolve(),
    )
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0 if decision["evaluation_protocol_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
