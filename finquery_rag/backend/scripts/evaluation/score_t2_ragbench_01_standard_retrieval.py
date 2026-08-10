#!/usr/bin/env python3
"""Post-seal scorer for the T2-01 standard retrieval baselines."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_ROWS = 23_088
SUBSETS = ("FinQA", "ConvFinQA", "TAT-DQA")
BASELINES = ("bm25", "dense", "hybrid")
KS = (1, 3, 5, 10, 20, 50, 100)


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


def load_targets(root: Path) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for subset, default_split, path in metadata_paths(root):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                query_id = str(row["id"])
                if query_id in targets:
                    raise RuntimeError(f"duplicate_query_id:{query_id}")
                targets[query_id] = {
                    "subset": subset,
                    "split": row.get("split", default_split),
                    "gold_context_id": str(row["context_id"]),
                    "question_nonempty": bool(row.get("question")),
                }
    return targets


def pct(value: float) -> float:
    return round(value * 100.0, 6)


def empty_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "hits": {str(k): 0 for k in KS},
        "mrr_at_5_sum": 0.0,
        "ndcg_at_10_sum": 0.0,
    }


def add_stat(stat: dict[str, Any], rank: int | None) -> None:
    stat["total"] += 1
    for k in KS:
        if rank is not None and rank <= k:
            stat["hits"][str(k)] += 1
    if rank is not None and rank <= 5:
        stat["mrr_at_5_sum"] += 1.0 / rank
    if rank is not None and rank <= 10:
        stat["ndcg_at_10_sum"] += 1.0 / math.log2(rank + 1.0)


def finalize_stat(stat: dict[str, Any]) -> dict[str, Any]:
    total = stat["total"]
    return {
        "count": total,
        "hits": stat["hits"],
        "recall": {
            f"@{k}": f"{stat['hits'][str(k)]}/{total}" if total else "0/0"
            for k in KS
        },
        "recall_pct": {
            f"@{k}": pct(stat["hits"][str(k)] / total) if total else 0.0 for k in KS
        },
        "mrr_at_5": stat["mrr_at_5_sum"] / total if total else 0.0,
        "mrr_at_5_pct": pct(stat["mrr_at_5_sum"] / total) if total else 0.0,
        "ndcg_at_10": stat["ndcg_at_10_sum"] / total if total else 0.0,
        "ndcg_at_10_pct": pct(stat["ndcg_at_10_sum"] / total) if total else 0.0,
    }


def score_baseline(
    path: Path,
    targets: dict[str, dict[str, Any]],
    corpus_ids: dict[str, set[str]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {subset: empty_stats() for subset in SUBSETS}
    stats["overall"] = empty_stats()
    stats["nonempty_diagnostic"] = empty_stats()
    empty_stats_by_subset: dict[str, dict[str, Any]] = {subset: empty_stats() for subset in SUBSETS}
    seen: set[str] = set()
    invalid: Counter[str] = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id in seen or query_id not in targets:
                invalid["query_id"] += 1
                continue
            seen.add(query_id)
            target = targets[query_id]
            ranked = row.get("ranked_contexts") or []
            ids = [str(item["context_id"]) for item in ranked]
            if len(ids) != len(set(ids)):
                invalid["duplicate_candidate"] += 1
            if any(context_id not in corpus_ids[target["subset"]] for context_id in ids):
                invalid["candidate_outside_subset_corpus"] += 1
            try:
                rank = ids.index(target["gold_context_id"]) + 1
            except ValueError:
                rank = None
            add_stat(stats[target["subset"]], rank)
            add_stat(stats["overall"], rank)
            if target["question_nonempty"]:
                add_stat(stats["nonempty_diagnostic"], rank)
            else:
                add_stat(empty_stats_by_subset[target["subset"]], rank)

    invalid["missing_prediction"] = len(targets) - len(seen)
    return (
        {
            "by_subset": {subset: finalize_stat(stats[subset]) for subset in SUBSETS},
            "overall": finalize_stat(stats["overall"]),
            "nonempty_diagnostic": finalize_stat(stats["nonempty_diagnostic"]),
            "invalid": dict(sorted(invalid.items())),
            "prediction_count": len(seen),
        },
        {subset: finalize_stat(stat) for subset, stat in empty_stats_by_subset.items()},
        stats,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--closure-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    closure = args.closure_root.resolve()
    prediction_root = args.prediction_root.resolve()
    closure_acceptance_path = closure / "acceptance.json"
    closure_acceptance = json.loads(closure_acceptance_path.read_text(encoding="utf-8"))
    if not closure_acceptance.get("published_raw_track_ready"):
        raise RuntimeError("t2_00r1_not_closed")
    seal_path = prediction_root / "prediction-seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("sealed") or seal.get("gold_scoring_reads_before_seal") != 0:
        raise RuntimeError("prediction_seal_invalid")
    for name, expected_hash in seal["output_sha256"].items():
        path = prediction_root / f"{name}-predictions.jsonl.gz"
        if sha256(path) != expected_hash:
            raise RuntimeError(f"prediction_mutation:{name}")
    if seal.get("prediction_count") != EXPECTED_ROWS:
        raise RuntimeError("prediction_count_contract")

    targets = load_targets(dataset)
    if len(targets) != EXPECTED_ROWS:
        raise RuntimeError(f"target_count_contract:{len(targets)}")
    corpus_manifest = json.loads((prediction_root / "corpus-manifest.json").read_text(encoding="utf-8"))
    corpus_ids = {
        subset: {str(item["context_id"]) for item in payload["documents"]}
        for subset, payload in corpus_manifest["corpora"].items()
    }

    metrics: dict[str, Any] = {}
    empty_diagnostic: dict[str, Any] = {
        "published_denominator": EXPECTED_ROWS,
        "nonempty_diagnostic_denominator": sum(
            int(target["question_nonempty"]) for target in targets.values()
        ),
        "empty_question_rows": sum(not target["question_nonempty"] for target in targets.values()),
        "diagnostic_only": True,
        "by_baseline": {},
    }
    invalid: dict[str, Any] = {}
    for baseline in BASELINES:
        scored, empty_by_subset, _ = score_baseline(
            prediction_root / f"{baseline}-predictions.jsonl.gz", targets, corpus_ids
        )
        metrics[baseline] = scored
        empty_diagnostic["by_baseline"][baseline] = {
            "nonempty": scored["nonempty_diagnostic"],
            "empty_by_subset": empty_by_subset,
        }
        invalid[baseline] = scored["invalid"]

    for subset, filename in (("FinQA", "finqa-metrics.json"), ("ConvFinQA", "convfinqa-metrics.json"), ("TAT-DQA", "tat-dqa-metrics.json")):
        write_json(
            prediction_root / filename,
            {
                "gate": "t2_ragbench_01_standard_retrieval",
                "denominator": metrics["bm25"]["by_subset"][subset]["count"],
                "baselines": {baseline: metrics[baseline]["by_subset"][subset] for baseline in BASELINES},
            },
        )
    write_json(
        prediction_root / "weighted-metrics.json",
        {
            "weighting": "query_count",
            "denominator": EXPECTED_ROWS,
            "baselines": {baseline: metrics[baseline]["overall"] for baseline in BASELINES},
        },
    )
    write_json(prediction_root / "empty-question-diagnostic.json", empty_diagnostic)
    write_json(prediction_root / "identity-score-audit.json", {"by_baseline": invalid, "gold_unit": "context_id"})
    acceptance = {
        "decision": "first_valid_public_benchmark_measurement",
        "dataset_commit": closure_acceptance["dataset_commit"],
        "published_rows": EXPECTED_ROWS,
        "headline_denominator": EXPECTED_ROWS,
        "query_count_exact": all(metrics[baseline]["prediction_count"] == EXPECTED_ROWS for baseline in BASELINES),
        "gold_identity": "context_id",
        "gold_identity_errors": sum(sum(values.values()) for values in invalid.values()),
        "baselines_completed": list(BASELINES),
        "whole_context_unit": True,
        "pdf_parsing": 0,
        "chunking": 0,
        "cross_encoder": 0,
        "llm": 0,
        "parameter_scan": False,
        "query_rewrite": 0,
        "query_plan": 0,
        "next_gate": "t2_02_cross_encoder_review",
    }
    write_json(prediction_root / "acceptance.json", acceptance)
    write_json(
        prediction_root / "next-gate.json",
        {
            "decision": acceptance["decision"],
            "next_gate": acceptance["next_gate"],
            "hybrid_recall_at_5": metrics["hybrid"]["overall"]["recall"]["@5"],
            "hybrid_recall_at_100": metrics["hybrid"]["overall"]["recall"]["@100"],
            "policy": "Review measured candidate ceiling before opening Cross-Encoder; no threshold was pre-imposed.",
        },
    )
    return 0 if acceptance["gold_identity_errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

