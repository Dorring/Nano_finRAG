#!/usr/bin/env python3
"""T2-00 read-only dataset intake audit.

The audit intentionally does not load a model, build an index, or score a
question.  It validates the pinned repository, materialized LFS files, and
the published QA metadata files before any retrieval experiment is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_TOTAL = 23_088
SUBSETS = ("FinQA", "ConvFinQA", "TAT-DQA")
REQUIRED_FIELDS = ("id", "context_id", "question", "context", "file_name")


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git(root: Path, *args: str) -> str:
    command = ["git", "-C", str(root), *args]
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError:
        # A materialized LFS worktree may not have git-lfs on PATH.  Status is
        # diagnostic only, so retain a deterministic marker instead of making
        # the dataset intake depend on the optional helper executable.
        if args[:2] == ("status", "--short"):
            return "lfs_status_unavailable_without_git_lfs"
        raise


def metadata_paths(root: Path) -> list[tuple[str, str, Path]]:
    paths: list[tuple[str, str, Path]] = []
    for split in ("train", "dev", "test"):
        paths.append(("FinQA", split, root / "data" / "FinQA" / split / "metadata.jsonl"))
    paths.append(("ConvFinQA", "turn_0", root / "data" / "ConvFinQA" / "turn_0.jsonl"))
    for split in ("train", "dev", "test"):
        paths.append(("TAT-DQA", split, root / "data" / "TAT-DQA" / split / "metadata.jsonl"))
    return paths


def audit_metadata(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    subset_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    unique_contexts: dict[str, set[str]] = defaultdict(set)
    unique_files: dict[str, set[str]] = defaultdict(set)
    question_counts: dict[str, Counter[str]] = defaultdict(Counter)
    context_id_counts: dict[str, Counter[str]] = defaultdict(Counter)
    missing: Counter[str] = Counter()
    parse_errors: list[dict[str, Any]] = []
    files_read: list[str] = []
    row_count = 0

    for subset, split, path in metadata_paths(root):
        if not path.exists():
            parse_errors.append({"file": str(path), "error": "missing_metadata_file"})
            continue
        files_read.append(str(path.relative_to(root)))
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append(
                        {"file": str(path.relative_to(root)), "line": line_number, "error": str(exc)}
                    )
                    continue
                row_count += 1
                subset_counts[subset] += 1
                split_counts[f"{subset}:{row.get('split', split)}"] += 1
                for field in REQUIRED_FIELDS:
                    value = row.get(field)
                    if value is None or value == "":
                        missing[field] += 1
                context_id = str(row.get("context_id") or "")
                file_name = str(row.get("file_name") or "")
                question = str(row.get("question") or "")
                if context_id:
                    unique_contexts[subset].add(context_id)
                    context_id_counts[subset][context_id] += 1
                if file_name:
                    unique_files[subset].add(file_name)
                if question:
                    question_counts[subset][question] += 1

    duplicate_questions = {
        subset: sum(count - 1 for count in counts.values() if count > 1)
        for subset, counts in question_counts.items()
    }
    duplicate_contexts = {
        subset: sum(count - 1 for count in counts.values() if count > 1)
        for subset, counts in context_id_counts.items()
    }
    counts = {
        "actual_total_rows": row_count,
        "expected_published_rows": EXPECTED_TOTAL,
        "rows_match_published_total": row_count == EXPECTED_TOTAL,
        "by_subset": dict(sorted(subset_counts.items())),
        "files_read": files_read,
        "parse_errors": parse_errors,
        "missing_fields": dict(sorted(missing.items())),
    }
    split_audit = {
        "split_distribution": dict(sorted(split_counts.items())),
        "row_count_by_subset": dict(sorted(subset_counts.items())),
    }
    identity = {
        "unique_context_ids_by_subset": {
            subset: len(values) for subset, values in sorted(unique_contexts.items())
        },
        "unique_context_ids_total": len(set().union(*unique_contexts.values()))
        if unique_contexts
        else 0,
        "unique_file_names_by_subset": {
            subset: len(values) for subset, values in sorted(unique_files.items())
        },
        "unique_file_names_total": len(set().union(*unique_files.values()))
        if unique_files
        else 0,
        "duplicate_question_count_by_subset": dict(sorted(duplicate_questions.items())),
        "duplicate_question_count_total": sum(duplicate_questions.values()),
        "duplicate_context_count_by_subset": dict(sorted(duplicate_contexts.items())),
        "duplicate_context_count_total": sum(duplicate_contexts.values()),
        "missing_context_id": missing["context_id"],
        "missing_question": missing["question"],
        "missing_context": missing["context"],
        "missing_file": missing["file_name"],
    }
    return counts, split_audit, identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    output = args.output_root.resolve()
    if not dataset.is_dir():
        raise SystemExit(f"dataset_root_missing:{dataset}")

    commit = git(dataset, "rev-parse", "HEAD")
    tree_files = git(dataset, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    working_status = git(dataset, "status", "--short")
    all_files = [p for p in dataset.rglob("*") if p.is_file() and ".git" not in p.parts]
    pointer_files: list[str] = []
    total_bytes = 0
    hashes: dict[str, dict[str, Any]] = {}
    for path in sorted(all_files):
        relative = str(path.relative_to(dataset)).replace("\\", "/")
        raw_head = path.read_bytes()[:128]
        is_pointer = raw_head.startswith(b"version https://git-lfs.github.com/spec/v1")
        if is_pointer:
            pointer_files.append(relative)
        size = path.stat().st_size
        total_bytes += size
        hashes[relative] = {"size": size, "sha256": sha256(path), "lfs_pointer": is_pointer}

    counts, split_audit, identity = audit_metadata(dataset)
    counts.update(
        {
            "repository_tree_file_count": len(tree_files),
            "materialized_file_count": len(all_files),
            "materialized_data_bytes": total_bytes,
            "lfs_pointer_count": len(pointer_files),
            "lfs_pointer_files_sample": pointer_files[:20],
            "working_tree_modified_file_count": len([line for line in working_status.splitlines() if line]),
        }
    )
    acceptance = {
        "dataset_commit_present": bool(commit),
        "actual_total_rows": counts["actual_total_rows"],
        "published_row_count_check": counts["rows_match_published_total"],
        "lfs_pointer_count": len(pointer_files),
        "lfs_materialization_complete": len(pointer_files) == 0,
        "metadata_parse_errors": len(counts["parse_errors"]),
        "missing_required_fields": counts["missing_fields"],
        "dataset_intake_accepted": bool(commit)
        and counts["actual_total_rows"] == EXPECTED_TOTAL
        and len(pointer_files) == 0
        and not counts["parse_errors"]
        and not counts["missing_fields"],
        "working_tree_note": "LFS materialization replaces pointer worktree contents; this is expected and not a dataset revision change.",
    }
    write_json(
        output / "dataset-manifest.json",
        {
            "repository": "https://huggingface.co/datasets/G4KMU/t2-ragbench",
            "repo_type": "dataset",
            "exact_commit_sha": commit,
            "license": "CC-BY-4.0",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "materialization": "git_clone_plus_user_space_git_lfs_pull",
            "dataset_root": str(dataset),
            "git_tree_file_count": len(tree_files),
            "lfs_pointer_count": len(pointer_files),
        },
    )
    write_json(output / "dataset-counts.json", counts)
    write_json(output / "split-audit.json", split_audit)
    write_json(output / "identity-audit.json", identity)
    write_json(output / "file-hashes.json", {"files": hashes})
    write_json(output / "acceptance.json", acceptance)
    return 0 if acceptance["dataset_intake_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

