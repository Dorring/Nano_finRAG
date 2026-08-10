#!/usr/bin/env python3
"""Gold-free batch invariance smoke for the T2-03 Qwen runtime."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from t2_ragbench_03_common import (
    BATCH_SIZE,
    EXPECTED_DATASET_COMMIT,
    INSTRUCTION,
    format_pair,
    frozen_bm25_top50,
    load_contexts,
    load_qwen_runtime,
    load_query_manifest,
    now,
    runtime_manifest,
    score_batch,
    write_json,
)


SMOKE_PAIRS = 20
SCORE_TOLERANCE = 1e-3


def score_in_batches(model, tokenizer, torch, pairs: list[str]) -> list[float]:
    scores: list[float] = []
    for offset in range(0, len(pairs), BATCH_SIZE):
        batch_scores, _ = score_batch(model, tokenizer, torch, pairs[offset : offset + BATCH_SIZE])
        scores.extend(batch_scores)
    return scores


def rank_order(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "gate": "T2-03R0-batch-smoke",
        "dataset_commit": EXPECTED_DATASET_COMMIT,
        "gold_reads": 0,
        "batch_size": BATCH_SIZE,
        "pairs": SMOKE_PAIRS,
        "score_tolerance": SCORE_TOLERANCE,
    }
    try:
        rows = load_query_manifest(args.prediction_root.resolve() / "query-manifest.jsonl.gz")
        selected = min(
            rows,
            key=lambda row: hashlib.sha256(str(row["query_id"]).encode("utf-8")).hexdigest(),
        )
        query_id = str(selected["query_id"])
        bm25 = frozen_bm25_top50(args.prediction_root.resolve() / "bm25-predictions.jsonl.gz", {query_id})
        contexts, _ = load_contexts(args.dataset_root.resolve())
        pairs = [
            format_pair(INSTRUCTION, str(selected["query"]), contexts[item["context_id"]])
            for item in bm25[query_id]["ranked_contexts"][:SMOKE_PAIRS]
        ]
        tokenizer, model, torch, transformers = load_qwen_runtime()
        started = now()
        batched = score_in_batches(model, tokenizer, torch, pairs)
        singles = []
        for pair in pairs:
            one, _ = score_batch(model, tokenizer, torch, [pair])
            singles.extend(one)
        elapsed = max(now() - started, 1e-9)
        max_abs_diff = max(abs(left - right) for left, right in zip(batched, singles))
        result.update(
            {
                "query_id": query_id,
                "runtime": runtime_manifest(tokenizer, torch, transformers, model),
                "batched_rank_order": rank_order(batched),
                "single_rank_order": rank_order(singles),
                "rank_order_parity": rank_order(batched) == rank_order(singles),
                "max_abs_score_diff": max_abs_diff,
                "finite": all(torch.isfinite(torch.tensor(value)) for value in batched + singles),
                "elapsed_seconds": elapsed,
                "decision": "batch_invariance_passed"
                if max_abs_diff <= SCORE_TOLERANCE
                and rank_order(batched) == rank_order(singles)
                else "t2_03_batch_invariance_blocked",
            }
        )
    except Exception as exc:  # pragma: no cover - runtime-only failures
        result.update({"decision": "t2_03_batch_invariance_blocked", "error": f"{type(exc).__name__}: {exc}"})
    write_json(output_root / "batch-smoke.json", result)
    return 0 if result["decision"] == "batch_invariance_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

