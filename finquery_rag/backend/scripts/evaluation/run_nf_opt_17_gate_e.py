"""NF-OPT-17 Gate E: freeze one reranker contract and measure zero-shot dev performance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.nf_opt_17_gate_d import rank_triplet, render_structured_reranker_input

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/nf-opt-17-gate-e"
FROZEN_ISSUER_ORDER = ("Alphabet Inc.", "AMAZON COM INC", "Meta Platforms, Inc.", "NETFLIX INC")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    pairwise = Counter()
    pairwise_total = Counter()
    for row in rows:
        for kind, passed in row["ranking"]["pairwise"].items():
            pairwise_total[kind] += 1
            pairwise[kind] += int(passed)
    total_pairs = sum(pairwise_total.values())
    return {
        "query_count": count,
        "top1_count": sum(int(row["ranking"]["positive_top1"]) for row in rows),
        "top1_accuracy": sum(int(row["ranking"]["positive_top1"]) for row in rows) / count,
        "mrr": sum(float(row["ranking"]["reciprocal_rank"]) for row in rows) / count,
        "pairwise_correct_count": sum(pairwise.values()),
        "pairwise_count": total_pairs,
        "pairwise_accuracy": sum(pairwise.values()) / total_pairs,
        "pairwise_by_negative_type": {
            kind: {"correct": pairwise[kind], "count": pairwise_total[kind], "accuracy": pairwise[kind] / pairwise_total[kind]}
            for kind in sorted(pairwise_total)
        },
    }


def run(args: argparse.Namespace) -> int:
    annotations = _read_jsonl(args.annotations)
    candidates = {row["candidate_key"]: row for row in _read_jsonl(args.candidates)}
    if len(annotations) != 80:
        raise ValueError(f"expected 80 independent annotations, got {len(annotations)}")
    if set(row["issuer"] for row in annotations) != set(FROZEN_ISSUER_ORDER):
        raise ValueError("issuer set differs from the frozen development corpus")

    model_path = args.model.resolve()
    if not model_path.joinpath("config.json").is_file():
        raise ValueError("reranker snapshot is missing config.json")
    from FlagEmbedding import FlagReranker

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    started = time.perf_counter()
    reranker = FlagReranker(str(model_path), use_fp16=True, devices=args.device)
    rows: list[dict[str, Any]] = []
    issuer_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        candidate_refs = [annotation["positive_candidate"]]
        candidate_refs.extend(item["candidate"] for item in annotation["hard_negatives"])
        candidate_rows = [candidates[item["candidate_key"]] for item in candidate_refs]
        rendered = [render_structured_reranker_input(item) for item in candidate_rows]
        scores = reranker.compute_score(
            [[annotation["question"], value] for value in rendered],
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        if not isinstance(scores, list):
            scores = [scores]
        if len(scores) != 3 or not all(math.isfinite(float(value)) for value in scores):
            raise ValueError("reranker returned invalid triplet scores")
        negative_types = [item["negative_type"] for item in annotation["hard_negatives"]]
        record = {
            "annotation_id": annotation["annotation_id"],
            "query_id": annotation["query_id"],
            "issuer": annotation["issuer"],
            "candidate_keys": [item["candidate_key"] for item in candidate_rows],
            "input_sha256": [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in rendered],
            "scores": [round(float(value), 8) for value in scores],
            "ranking": rank_triplet(scores, negative_types),
        }
        rows.append(record)
        issuer_rows[str(annotation["issuer"])].append(record)
    elapsed = time.perf_counter() - started

    configuration = {
        "schema": "nf-opt-17/frozen-reranker-config/v1",
        "model_family": "BAAI/bge-reranker-v2-m3",
        "model_snapshot_path_committed": False,
        "model_config_sha256": _sha(model_path / "config.json"),
        "candidate_input_template": "issuer + xbrl_concept + period_end + period_kind + evidence_excerpt",
        "candidate_input_uses_expected_fields": False,
        "candidate_input_uses_frozen_benchmark_fields": False,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": args.device,
        "fp16": True,
        "offline": True,
        "seed": 20260805,
        "issuer_grouped_validation": True,
        "hyperparameter_scan_allowed": False,
        "training_performed": False,
    }
    result = {
        "schema": "nf-opt-17/zero-shot-development-baseline/v1",
        "overall": _metrics(rows),
        "by_issuer": {issuer: _metrics(issuer_rows[issuer]) for issuer in FROZEN_ISSUER_ORDER},
        "records": rows,
    }
    acceptance = {
        "schema": "nf-opt-17/gate-e/acceptance/v1",
        "annotation_count": len(annotations),
        "candidate_count": len(candidates),
        "annotation_input_sha256": _sha(args.annotations),
        "candidate_input_sha256": _sha(args.candidates),
        "model_scores_finite": True,
        "frozen_benchmark_question_or_label_reads": 0,
        "expected_field_reads": 0,
        "model_inference_calls": len(annotations),
        "model_training_steps": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "elapsed_seconds": round(elapsed, 3),
        "python_version": platform.python_version(),
        "decision": "frozen_reranker_contract_zero_shot_measured",
        "next_gate": "nf-opt-17-gate-f-single-pre_registered_training_run",
    }
    _write(args.out_dir / "frozen-reranker-configuration.json", configuration)
    _write(args.out_dir / "zero-shot-development-results.json", result)
    _write(args.out_dir / "nf-opt-17-gate-e-acceptance.json", acceptance)
    _write(args.out_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": acceptance["next_gate"], "production_switch_allowed": False})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
