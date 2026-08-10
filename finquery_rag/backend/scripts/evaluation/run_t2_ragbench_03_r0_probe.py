#!/usr/bin/env python3
"""T2-03R0 Qwen4B runtime and frozen candidate contract probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

from t2_ragbench_03_common import (
    CANDIDATE_DEPTH,
    EXPECTED_DATASET_COMMIT,
    EXPECTED_MODEL_ID,
    INSTRUCTION,
    MAX_LENGTH,
    MODEL_REVISION,
    MODEL_SNAPSHOT,
    SYSTEM_PREFIX,
    SYSTEM_SUFFIX,
    frozen_bm25_top50,
    format_pair,
    load_contexts,
    load_qwen_runtime,
    load_query_manifest,
    model_file_manifest,
    now,
    runtime_manifest,
    score_batch,
    sha256_text,
    write_json,
)


EXPECTED_PROBE_QUERIES = 256
TRUNCATION_BLOCK_PERCENT = 50.0


def select_probe_queries(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], bool]:
    ordered = sorted(rows, key=lambda row: hashlib.sha256(str(row["query_id"]).encode("utf-8")).hexdigest())
    selected = ordered[:EXPECTED_PROBE_QUERIES]
    empty_rows = [row for row in rows if str(row.get("query", "")).endswith(": ")]
    injected = False
    if empty_rows and not any(str(row.get("query", "")).endswith(": ") for row in selected):
        selected[-1] = min(empty_rows, key=lambda row: hashlib.sha256(str(row["query_id"]).encode("utf-8")).hexdigest())
        injected = True
    selected.sort(key=lambda row: hashlib.sha256(str(row["query_id"]).encode("utf-8")).hexdigest())
    return selected, injected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    prediction_root = args.prediction_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    probe_path = output_root / "r0-runtime-probe.json"
    try:
        protocol = json.loads((prediction_root / "protocol.json").read_text(encoding="utf-8"))
        if protocol.get("dataset_commit") != EXPECTED_DATASET_COMMIT:
            raise RuntimeError("dataset_commit_contract")
        if protocol.get("published_rows") != 23088 or protocol.get("gold_unit") != "context_id":
            raise RuntimeError("baseline_contract")
        rows = load_query_manifest(prediction_root / "query-manifest.jsonl.gz")
        selected, empty_injected = select_probe_queries(rows)
        query_ids = {str(row["query_id"]) for row in selected}
        bm25 = frozen_bm25_top50(prediction_root / "bm25-predictions.jsonl.gz", query_ids)
        contexts, context_counts = load_contexts(dataset_root)
        for query_id, row in bm25.items():
            if row["subset"] != next(item["subset"] for item in selected if str(item["query_id"]) == query_id):
                raise RuntimeError(f"subset_contract:{query_id}")
            if any(item["context_id"] not in contexts for item in row["ranked_contexts"]):
                raise RuntimeError(f"candidate_context_missing:{query_id}")
        input_identity = [
            {
                "query_id": str(row["query_id"]),
                "candidate_ids": [item["context_id"] for item in bm25[str(row["query_id"])] ["ranked_contexts"]],
            }
            for row in selected
        ]
        write_json(
            output_root / "r0-input-manifest.json",
            {
                "selection_method": "sha256(query_id) ordering, first 256, deterministic empty-question inclusion",
                "query_count": len(selected),
                "candidate_depth": CANDIDATE_DEPTH,
                "empty_question_injected": empty_injected,
                "subset_counts": {subset: sum(row["subset"] == subset for row in selected) for subset in context_counts},
                "query_ids_sha256": sha256_text(json.dumps([str(row["query_id"]) for row in selected], separators=(",", ":"))),
                "candidate_identity_sha256": sha256_text(json.dumps(input_identity, separators=(",", ":"))),
            },
        )
        tokenizer, model, torch, transformers = load_qwen_runtime()
        write_json(output_root / "model-manifest.json", model_file_manifest(MODEL_SNAPSHOT))
        write_json(
            output_root / "instruction-contract.json",
            {
                "instruction": INSTRUCTION,
                "instruction_sha256": sha256_text(INSTRUCTION),
                "system_prefix_sha256": sha256_text(SYSTEM_PREFIX),
                "system_suffix_sha256": sha256_text(SYSTEM_SUFFIX),
                "query_document_format": "<Instruct> + <Query> + <Document>",
                "per_query_instruction": False,
            },
        )
        write_json(output_root / "runtime-contract.json", runtime_manifest(tokenizer, torch, transformers, model))
        torch.cuda.reset_peak_memory_stats(0)
        started = now()
        pair_count = 0
        truncated_count = 0
        nonfinite_count = 0
        token_counts_before: list[int] = []
        token_counts_after: list[int] = []
        runtime_errors: list[str] = []
        for query in selected:
            query_id = str(query["query_id"])
            bm25_row = bm25[query_id]
            pairs = [
                format_pair(INSTRUCTION, str(query["query"]), contexts[item["context_id"]])
                for item in bm25_row["ranked_contexts"]
            ]
            try:
                scores, token_meta = score_batch(model, tokenizer, torch, pairs)
            except Exception as exc:  # pragma: no cover - exercised only on a blocked runtime
                runtime_errors.append(f"{query_id}:{type(exc).__name__}:{exc}")
                break
            pair_count += len(scores)
            for score, meta in zip(scores, token_meta):
                if not torch.isfinite(torch.tensor(score)):
                    nonfinite_count += 1
                token_counts_before.append(meta["token_count_before_truncation"])
                token_counts_after.append(meta["token_count_after_truncation"])
                truncated_count += int(meta["truncated"])
        elapsed = max(now() - started, 1e-9)
        peak_vram = torch.cuda.max_memory_reserved(0) / (1024 * 1024)
        truncation_percent = 100.0 * truncated_count / pair_count if pair_count else 0.0
        decision = "reranker_runtime_probe_passed"
        if runtime_errors or pair_count != len(selected) * CANDIDATE_DEPTH or nonfinite_count:
            decision = "t2_03_runtime_blocked"
        if truncation_percent > TRUNCATION_BLOCK_PERCENT:
            decision = "reranker_input_contract_blocked"
        probe = {
            "gate": "T2-03R0",
            "decision": decision,
            "model_id": EXPECTED_MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_commit": EXPECTED_DATASET_COMMIT,
            "query_count": len(selected),
            "pairs_processed": pair_count,
            "expected_pairs": len(selected) * CANDIDATE_DEPTH,
            "candidate_depth": CANDIDATE_DEPTH,
            "pairs_per_second": pair_count / elapsed if pair_count else 0.0,
            "queries_per_second": len(selected) / elapsed if selected else 0.0,
            "elapsed_seconds": elapsed,
            "peak_vram_mb": peak_vram,
            "median_input_tokens_before": sorted(token_counts_before)[len(token_counts_before) // 2] if token_counts_before else 0,
            "p95_input_tokens_before": sorted(token_counts_before)[max(0, int(0.95 * len(token_counts_before)) - 1)] if token_counts_before else 0,
            "max_input_tokens_before": max(token_counts_before) if token_counts_before else 0,
            "median_input_tokens_after": sorted(token_counts_after)[len(token_counts_after) // 2] if token_counts_after else 0,
            "truncated_pair_count": truncated_count,
            "truncated_pair_percent": truncation_percent,
            "non_finite_score_count": nonfinite_count,
            "runtime_errors": runtime_errors,
            "gold_reads_before_seal": 0,
            "candidate_identity_mutation": 0,
            "instruction_sha256": sha256_text(INSTRUCTION),
            "max_length": MAX_LENGTH,
            "dtype": "bfloat16",
            "batch_size": 1,
            "empty_question_path": "official query manifest f-string result; no semantic repair",
        }
        write_json(probe_path, probe)
        return 0 if decision == "reranker_runtime_probe_passed" else 2
    except Exception as exc:  # pragma: no cover - runtime/environment failures
        write_json(
            probe_path,
            {
                "gate": "T2-03R0",
                "decision": "t2_03_runtime_blocked",
                "model_id": EXPECTED_MODEL_ID,
                "model_revision": MODEL_REVISION,
                "dataset_commit": EXPECTED_DATASET_COMMIT,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "gold_reads_before_seal": 0,
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

