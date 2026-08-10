from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from t2_ragbench_03_common import (
    CANDIDATE_DEPTH,
    EXPECTED_MODEL_ID,
    INSTRUCTION,
    MAX_LENGTH,
    MODEL_REVISION,
    MODEL_SNAPSHOT,
    format_pair,
    frozen_bm25_top50,
    load_contexts,
    load_qwen_runtime,
    score_batch,
    sha256_text,
)


BASE_COMMIT = "3352217"
METHOD_HASH = "93a7dccd72a1c3d19effe58053942504adfb9b9ca2c45ed32525edfcadd4006e"
FEATURE_SEAL = "98204451ca98046fb7bed2338ad346f511b10f90195b6e7d78f084c52131641d"
PRIMARY_COUNT = 2291
FORMAL_PAIRS = PRIMARY_COUNT * CANDIDATE_DEPTH
PROBE_COUNT = 256
MIN_FREE_MB = 20000
KS = (1, 3, 5, 10, 20, 50)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_obj(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for row in rows:
                compressed.write(
                    (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                        "utf-8"
                    )
                )


def load_frozen_contracts(
    t203_root: Path,
    method_root: Path,
    retrieval_root: Path,
) -> dict[str, Any]:
    runtime = read_json(t203_root / "runtime-contract.json")
    reference = read_json(t203_root / "runtime-acceleration" / "reference-runtime-contract.json")
    instruction = read_json(t203_root / "instruction-contract.json")
    model = read_json(t203_root / "model-manifest.json")
    r0 = read_json(t203_root / "r0-runtime-probe.json")
    if runtime["model_id"] != EXPECTED_MODEL_ID or runtime["model_revision"] != MODEL_REVISION:
        raise RuntimeError("qwen_runtime_contract_mismatch")
    if reference["model_class"] != "Qwen3ForCausalLM" or reference["model_loader"] != "AutoModelForCausalLM":
        raise RuntimeError("qwen_reference_backend_contract_mismatch")
    if reference["full_vocabulary_logits_computed"] is not True:
        raise RuntimeError("qwen_full_logits_contract_mismatch")
    if runtime["batch_size"] != 1 or runtime["max_length"] != MAX_LENGTH:
        raise RuntimeError("qwen_batch_or_length_contract_mismatch")
    if runtime["dtype"] != "bfloat16" or runtime["attention_implementation"] != "flash_attention_2":
        raise RuntimeError("qwen_dtype_attention_contract_mismatch")
    if instruction["instruction"] != INSTRUCTION:
        raise RuntimeError("qwen_instruction_mismatch")
    if instruction["instruction_sha256"] != sha256_text(INSTRUCTION):
        raise RuntimeError("qwen_instruction_hash_mismatch")
    if model["model_id"] != EXPECTED_MODEL_ID or model["revision"] != MODEL_REVISION:
        raise RuntimeError("qwen_model_manifest_mismatch")
    if not MODEL_SNAPSHOT.exists():
        raise RuntimeError("qwen_snapshot_missing")
    selected = read_json(method_root / "selected-method.json")
    manifest = read_json(method_root / "selected-method-manifest.json")
    if sha256_obj(manifest) != METHOD_HASH:
        raise RuntimeError("pcr_method_hash_mismatch")
    if selected["feature_seal"] != FEATURE_SEAL or selected["feature"] != "required_period_coverage":
        raise RuntimeError("pcr_method_contract_mismatch")
    prediction_seal = read_json(retrieval_root / "prediction-seal.json")
    bm25_path = retrieval_root / "bm25-predictions.jsonl.gz"
    if sha256_file(bm25_path) != prediction_seal["output_sha256"]["bm25"]:
        raise RuntimeError("bm25_prediction_mutation")
    if prediction_seal["gold_scoring_reads_before_seal"] != 0:
        raise RuntimeError("bm25_gold_preseal_contract")
    if prediction_seal["prediction_count"] != 23088:
        raise RuntimeError("bm25_prediction_count_contract")
    if r0["decision"] != "reranker_runtime_probe_passed" or r0["pairs_processed"] != 12800:
        raise RuntimeError("qwen_r0_not_passed")
    return {
        "runtime": runtime,
        "reference_runtime": reference,
        "instruction": instruction,
        "model": model,
        "r0": r0,
        "selected_method": selected,
        "method_manifest": manifest,
        "bm25_prediction_sha": prediction_seal["output_sha256"]["bm25"],
    }


def query_manifest_rows(retrieval_root: Path) -> list[dict[str, Any]]:
    rows = read_gz(retrieval_root / "query-manifest.jsonl.gz")
    if len(rows) != 23088 or len({str(row["query_id"]) for row in rows}) != 23088:
        raise RuntimeError("query_manifest_contract")
    return rows


def primary_ids(protocol_root: Path) -> set[str]:
    ids = {str(value) for value in read_json(protocol_root / "primary-test-query-ids.json")}
    if len(ids) != PRIMARY_COUNT:
        raise RuntimeError("primary_test_query_count")
    return ids


def select_probe_queries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(str(row["query_id"]).encode("utf-8")).hexdigest(),
    )
    selected = ordered[:PROBE_COUNT]
    empty_rows = [row for row in rows if str(row.get("query", "")).endswith(": ")]
    injected = False
    if empty_rows and not any(str(row.get("query", "")).endswith(": ") for row in selected):
        selected[-1] = min(
            empty_rows,
            key=lambda row: hashlib.sha256(str(row["query_id"]).encode("utf-8")).hexdigest(),
        )
        injected = True
    selected.sort(
        key=lambda row: hashlib.sha256(str(row["query_id"]).encode("utf-8")).hexdigest()
    )
    return selected, injected


def prepare_input_rows(
    rows: list[dict[str, Any]],
    candidate_map: dict[str, dict[str, Any]],
    query_ids: set[str],
) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        query_id = str(row["query_id"])
        if query_id not in query_ids:
            continue
        ranked = candidate_map[query_id]["ranked_contexts"]
        prepared.append(
            {
                "query_id": query_id,
                "subset": row["subset"],
                "split": row["split"],
                "query": row["query"],
                "candidate_contexts": [
                    {
                        "context_id": item["context_id"],
                        "original_bm25_rank": item["original_bm25_rank"],
                    }
                    for item in ranked
                ],
            }
        )
    if len(prepared) != len(query_ids):
        raise RuntimeError("prepared_query_count")
    if any(len(row["candidate_contexts"]) != CANDIDATE_DEPTH for row in prepared):
        raise RuntimeError("prepared_candidate_depth")
    return prepared


def input_hash(rows: list[dict[str, Any]]) -> str:
    identity = [
        {
            "query_id": row["query_id"],
            "candidate_ids": [item["context_id"] for item in row["candidate_contexts"]],
        }
        for row in rows
    ]
    return sha256_text(json.dumps(identity, separators=(",", ":")))


def shard_ranges(query_count: int, gpu_indices: list[int]) -> list[dict[str, int]]:
    ranges = []
    for shard_id in range(len(gpu_indices)):
        start = (shard_id * query_count) // len(gpu_indices)
        end = ((shard_id + 1) * query_count) // len(gpu_indices)
        ranges.append({"shard_id": shard_id, "gpu_index": gpu_indices[shard_id], "start": start, "end": end})
    if ranges[0]["start"] != 0 or ranges[-1]["end"] != query_count:
        raise RuntimeError("shard_boundary_contract")
    for left, right in zip(ranges, ranges[1:]):
        if left["end"] != right["start"]:
            raise RuntimeError("shard_gap_or_overlap")
    return ranges


def gpu_capacity() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    devices = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        index, name, total, used, free, utilization = fields
        devices.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mb": int(total),
                "memory_used_mb": int(used),
                "memory_free_mb": int(free),
                "utilization_gpu_percent": int(utilization),
            }
        )
    preferred = [1, 5, 6]
    usable = [
        device["index"]
        for device in devices
        if device["memory_free_mb"] >= MIN_FREE_MB and device["utilization_gpu_percent"] < 50
    ]
    usable = [index for index in preferred if index in usable] + [
        index for index in usable if index not in preferred
    ]
    if not usable:
        raise RuntimeError("no_usable_gpu")
    return {
        "available_gpu_count": len(devices),
        "gpus": devices,
        "selection_policy": {
            "preferred_indices": preferred,
            "min_free_mb": MIN_FREE_MB,
            "max_utilization_percent": 49,
        },
        "usable_gpu_indices": usable,
        "selected_gpu_indices": usable[: min(3, len(usable))],
        "processes_not_terminated": True,
    }


def run_worker(
    script: Path,
    prepared_path: Path,
    dataset_root: Path,
    output_path: Path,
    stats_path: Path,
    start: int,
    end: int,
    gpu_index: int,
    label: str,
) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["T2_QWEN_BATCH_SIZE"] = "1"
    env["T2_QWEN_ATTN"] = "flash_attention_2"
    env_lib = str(Path(sys.executable).resolve().parent.parent / "lib")
    env["LD_LIBRARY_PATH"] = env_lib + ":/mnt/disk/mxf/anaconda3/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        str(script),
        "--mode",
        "worker",
        "--prepared-input",
        str(prepared_path),
        "--dataset-root",
        str(dataset_root),
        "--output",
        str(output_path),
        "--stats",
        str(stats_path),
        "--start",
        str(start),
        "--end",
        str(end),
        "--gpu-label",
        str(gpu_index),
        "--label",
        label,
    ]
    return subprocess.call(command, env=env)


def launch_workers(
    script: Path,
    prepared_path: Path,
    dataset_root: Path,
    output_root: Path,
    ranges: list[dict[str, int]],
    label: str,
) -> list[dict[str, Any]]:
    processes = []
    started = time.perf_counter()
    for shard in ranges:
        output_path = output_root / f"predictions-shard-{shard['shard_id']}.jsonl.gz"
        stats_path = output_root / f"shard-{shard['shard_id']}-stats.json"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard["gpu_index"])
        env["T2_QWEN_BATCH_SIZE"] = "1"
        env["T2_QWEN_ATTN"] = "flash_attention_2"
        env_lib = str(Path(sys.executable).resolve().parent.parent / "lib")
        env["LD_LIBRARY_PATH"] = env_lib + ":/mnt/disk/mxf/anaconda3/lib:" + env.get("LD_LIBRARY_PATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            str(script),
            "--mode",
            "worker",
            "--prepared-input",
            str(prepared_path),
            "--dataset-root",
            str(dataset_root),
            "--output",
            str(output_path),
            "--stats",
            str(stats_path),
            "--start",
            str(shard["start"]),
            "--end",
            str(shard["end"]),
            "--gpu-label",
            str(shard["gpu_index"]),
            "--label",
            label,
        ]
        log_path = output_root / f"shard-{shard['shard_id']}.log"
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((process, log))
    statuses = []
    for process, log in processes:
        statuses.append(process.wait())
        log.close()
    if any(status != 0 for status in statuses):
        raise RuntimeError(f"worker_failed:{statuses}")
    elapsed = max(time.perf_counter() - started, 1e-9)
    stats = [read_json(output_root / f"shard-{shard['shard_id']}-stats.json") for shard in ranges]
    for item in stats:
        item["wall_clock_elapsed_seconds"] = elapsed
    return stats


def merge_shards(
    rows: list[dict[str, Any]],
    output_root: Path,
    ranges: list[dict[str, int]],
) -> tuple[list[dict[str, Any]], str]:
    merged: dict[str, dict[str, Any]] = {}
    for shard in ranges:
        path = output_root / f"predictions-shard-{shard['shard_id']}.jsonl.gz"
        for row in read_gz(path):
            query_id = str(row["query_id"])
            if query_id in merged:
                raise RuntimeError(f"duplicate_merged_query:{query_id}")
            merged[query_id] = row
    expected_ids = [str(row["query_id"]) for row in rows]
    if set(merged) != set(expected_ids):
        raise RuntimeError("merged_query_identity")
    ordered = [merged[query_id] for query_id in expected_ids]
    pair_count = sum(len(row["ranked_contexts"]) for row in ordered)
    if len(ordered) != PRIMARY_COUNT or pair_count != FORMAL_PAIRS:
        raise RuntimeError("merged_completeness")
    path = output_root / "predictions.jsonl.gz"
    write_gz(path, ordered)
    return ordered, sha256_file(path)


def rank_from_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    output = {}
    for row in rows:
        ranked = sorted(
            row["ranked_contexts"],
            key=lambda item: (
                -float(item["reranker_score"]),
                int(item["original_bm25_rank"]),
                str(item["context_id"]),
            ),
        )
        output[str(row["query_id"])] = [str(item["context_id"]) for item in ranked]
    return output


def score(order: dict[str, list[str]], qids: list[str], gold: dict[str, str]) -> dict[str, Any]:
    hits = {str(k): 0 for k in KS}
    mrr = ndcg5 = ndcg10 = 0.0
    for query_id in qids:
        try:
            rank = order[query_id].index(gold[query_id]) + 1
        except ValueError:
            rank = None
        for cutoff in KS:
            hits[str(cutoff)] += int(rank is not None and rank <= cutoff)
        if rank is not None and rank <= 5:
            mrr += 1.0 / rank
            ndcg5 += 1.0 / math.log2(rank + 1.0)
        if rank is not None and rank <= 10:
            ndcg10 += 1.0 / math.log2(rank + 1.0)
    count = len(qids)
    return {
        "count": count,
        "hits": hits,
        "recall_pct": {f"@{k}": round(100.0 * hits[str(k)] / count, 6) for k in KS},
        "recall": {f"@{k}": f"{hits[str(k)]}/{count}" for k in KS},
        "mrr_at_5_pct": round(100.0 * mrr / count, 6),
        "ndcg_at_5_pct": round(100.0 * ndcg5 / count, 6),
        "ndcg_at_10_pct": round(100.0 * ndcg10 / count, 6),
    }


def movement(
    baseline: dict[str, list[str]],
    candidate: dict[str, list[str]],
    gold: dict[str, str],
) -> dict[str, Any]:
    rescued = damaged = unchanged = 0
    for query_id, gold_id in gold.items():
        before = gold_id in baseline[query_id][:5]
        after = gold_id in candidate[query_id][:5]
        if after and not before:
            rescued += 1
        elif before and not after:
            damaged += 1
        else:
            unchanged += 1
    total = rescued + damaged
    return {
        "rescued_at_5": rescued,
        "damaged_at_5": damaged,
        "net_top5_gain": rescued - damaged,
        "unchanged_at_5": unchanged,
        "rescue_precision": round(rescued / total, 6) if total else None,
    }


def load_gold(dataset_root: Path, qids: set[str]) -> dict[str, str]:
    gold = {}
    for subset, split in (("FinQA", "test"), ("TAT-DQA", "test")):
        with (dataset_root / "data" / subset / split / "metadata.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                query_id = str(row["id"])
                if query_id in qids:
                    gold[query_id] = str(row["context_id"])
    if set(gold) != qids:
        raise RuntimeError("gold_identity_mismatch")
    return gold


def load_a2_query_structures(retrieval_root: Path, qids: set[str]) -> dict[str, dict[str, Any]]:
    path = Path(__file__).with_name("run_t2_ragbench_04a2_structure_signal_audit.py")
    spec = importlib.util.spec_from_file_location("t2_04a2_query", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("a2_query_extractor_missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = {}
    for row in read_gz(retrieval_root / "query-manifest.jsonl.gz"):
        query_id = str(row["query_id"])
        if query_id in qids:
            output[query_id] = module.extract_query(row.get("query", ""), row.get("company_name"))
    if set(output) != qids:
        raise RuntimeError("query_structure_identity")
    return output


def rank_compression(
    baseline: dict[str, list[str]],
    qwen: dict[str, list[str]],
    gold: dict[str, str],
) -> dict[str, Any]:
    rank_pairs = []
    cohorts = {
        "6_10": {"total": 0, "promoted_into_top5": 0},
        "11_20": {"total": 0, "promoted_into_top5": 0},
        "21_50": {"total": 0, "promoted_into_top5": 0},
    }
    for query_id, gold_id in gold.items():
        before = baseline[query_id].index(gold_id) + 1 if gold_id in baseline[query_id] else None
        after = qwen[query_id].index(gold_id) + 1 if gold_id in qwen[query_id] else None
        if before is None or after is None:
            continue
        rank_pairs.append((before, after))
        if 6 <= before <= 10:
            cohort = cohorts["6_10"]
        elif 11 <= before <= 20:
            cohort = cohorts["11_20"]
        elif 21 <= before <= 50:
            cohort = cohorts["21_50"]
        else:
            cohort = None
        if cohort is not None:
            cohort["total"] += 1
            cohort["promoted_into_top5"] += int(after <= 5)
    deltas = [before - after for before, after in rank_pairs]
    promoted = sum(delta > 0 for delta in deltas)
    demoted = sum(delta < 0 for delta in deltas)
    unchanged = sum(delta == 0 for delta in deltas)
    return {
        "gold_present_in_bm25_top50": len(rank_pairs),
        "mean_rank_delta_bm25_minus_qwen": round(statistics.mean(deltas), 6) if deltas else 0.0,
        "median_rank_delta_bm25_minus_qwen": statistics.median(deltas) if deltas else 0.0,
        "promoted_count": promoted,
        "demoted_count": demoted,
        "unchanged_count": unchanged,
        "cohorts": cohorts,
    }


def run_worker_mode(args: argparse.Namespace) -> int:
    prepared = read_gz(args.prepared_input)
    selected = prepared[args.start : args.end]
    contexts, _counts = load_contexts(args.dataset_root)
    tokenizer, model, torch, _transformers = load_qwen_runtime()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    output_rows = []
    pair_count = 0
    nonfinite = 0
    truncated = 0
    token_before = []
    token_after = []
    errors = []
    for query in selected:
        ranked = []
        for candidate in query["candidate_contexts"]:
            context = contexts.get(candidate["context_id"])
            if context is None:
                errors.append(f"missing_context:{query['query_id']}:{candidate['context_id']}")
                continue
            pair = format_pair(INSTRUCTION, query["query"], context)
            try:
                scores, metadata = score_batch(model, tokenizer, torch, [pair])
                value = float(scores[0])
                meta = metadata[0]
            except Exception as exc:  # pragma: no cover - runtime-only branch
                errors.append(f"{query['query_id']}:{type(exc).__name__}:{exc}")
                continue
            pair_count += 1
            nonfinite += int(not math.isfinite(value))
            truncated += int(meta["truncated"])
            token_before.append(int(meta["token_count_before_truncation"]))
            token_after.append(int(meta["token_count_after_truncation"]))
            ranked.append(
                {
                    "context_id": candidate["context_id"],
                    "original_bm25_rank": candidate["original_bm25_rank"],
                    "reranker_score": value,
                    "token_count_before_truncation": meta["token_count_before_truncation"],
                    "token_count_after_truncation": meta["token_count_after_truncation"],
                    "truncated": meta["truncated"],
                }
            )
        output_rows.append({"query_id": query["query_id"], "subset": query["subset"], "ranked_contexts": ranked})
    elapsed = max(time.perf_counter() - started, 1e-9)
    complete = not errors and len(output_rows) == len(selected) and pair_count == len(selected) * CANDIDATE_DEPTH
    write_gz(args.output, output_rows)
    stats = {
        "label": args.label,
        "gpu_index": args.gpu_label,
        "start": args.start,
        "end": args.end,
        "query_count": len(selected),
        "pair_count": pair_count,
        "expected_pair_count": len(selected) * CANDIDATE_DEPTH,
        "elapsed_seconds": elapsed,
        "pairs_per_second": pair_count / elapsed if pair_count else 0.0,
        "queries_per_second": len(selected) / elapsed if selected else 0.0,
        "peak_vram_mb": float(torch.cuda.max_memory_reserved(0) / (1024 * 1024)),
        "truncated_pair_count": truncated,
        "non_finite_score_count": nonfinite,
        "token_count_before_median": statistics.median(token_before) if token_before else 0,
        "token_count_before_p95": sorted(token_before)[max(0, int(0.95 * len(token_before)) - 1)] if token_before else 0,
        "token_count_after_median": statistics.median(token_after) if token_after else 0,
        "errors": errors,
        "complete": complete,
        "prediction_sha256": sha256_file(args.output),
    }
    write_json(args.stats, stats)
    return 0 if complete else 2


def compare_probe(reference_rows: list[dict[str, Any]], multi_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference = {row["query_id"]: row for row in reference_rows}
    multi = {row["query_id"]: row for row in multi_rows}
    if set(reference) != set(multi):
        raise RuntimeError("probe_query_identity")
    max_diff = 0.0
    diffs = []
    top1 = top5 = top10 = top50 = 0
    pair_count = 0
    for query_id in reference:
        ref_items = {item["context_id"]: item for item in reference[query_id]["ranked_contexts"]}
        multi_items = {item["context_id"]: item for item in multi[query_id]["ranked_contexts"]}
        if set(ref_items) != set(multi_items):
            raise RuntimeError(f"probe_candidate_identity:{query_id}")
        for context_id in ref_items:
            diff = abs(float(ref_items[context_id]["reranker_score"]) - float(multi_items[context_id]["reranker_score"]))
            diffs.append(diff)
            max_diff = max(max_diff, diff)
        pair_count += len(ref_items)
        def ordering(items: dict[str, dict[str, Any]]) -> list[str]:
            return [
                item["context_id"]
                for item in sorted(
                    items.values(),
                    key=lambda item: (
                        -float(item["reranker_score"]),
                        int(item["original_bm25_rank"]),
                        str(item["context_id"]),
                    ),
                )
            ]
        ref_order = ordering(ref_items)
        multi_order = ordering(multi_items)
        top1 += int(ref_order[:1] == multi_order[:1])
        top5 += int(ref_order[:5] == multi_order[:5])
        top10 += int(ref_order[:10] == multi_order[:10])
        top50 += int(ref_order == multi_order)
    count = len(reference)
    return {
        "probe_queries": count,
        "probe_pairs": pair_count,
        "score_max_abs_diff": max_diff,
        "score_mean_abs_diff": statistics.mean(diffs) if diffs else 0.0,
        "top1_agreement": top1 / count,
        "top5_ordered_agreement": top5 / count,
        "top10_ordered_agreement": top10 / count,
        "top50_ordered_agreement": top50 / count,
        "runtime_equivalence": "exact_ranking" if top50 == count else "material_difference",
        "multi_gpu_runtime_accepted": top50 == count,
        "gold_reads": 0,
    }


def main_coordinator(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    contracts = load_frozen_contracts(args.t203_root, args.method_root, args.retrieval_root)
    capacity = gpu_capacity()
    write_json(output_root / "gpu-capacity.json", capacity)
    selected_gpus = capacity["selected_gpu_indices"]
    if not selected_gpus:
        raise RuntimeError("no_selected_gpu")
    rows = query_manifest_rows(args.retrieval_root)
    primary_query_ids = primary_ids(args.protocol_root)
    candidate_map = frozen_bm25_top50(args.retrieval_root / "bm25-predictions.jsonl.gz", primary_query_ids)
    contexts, _ = load_contexts(args.dataset_root)
    for query_id, item in candidate_map.items():
        if any(candidate["context_id"] not in contexts for candidate in item["ranked_contexts"]):
            raise RuntimeError(f"candidate_context_missing:{query_id}")
    prepared = prepare_input_rows(rows, candidate_map, primary_query_ids)
    if len(prepared) != PRIMARY_COUNT:
        raise RuntimeError("primary_prepared_count")
    write_gz(output_root / "prepared-input.jsonl.gz", prepared)
    prepared_hash = sha256_file(output_root / "prepared-input.jsonl.gz")
    ranges = shard_ranges(len(prepared), selected_gpus)
    write_json(
        output_root / "shard-manifest.json",
        {
            "query_count": PRIMARY_COUNT,
            "candidate_depth": CANDIDATE_DEPTH,
            "formal_pairs": FORMAL_PAIRS,
            "gpu_indices": selected_gpus,
            "shards": ranges,
            "sharding": "deterministic contiguous query-level ranges",
            "union_complete": True,
            "intersection_empty": True,
            "prepared_input_sha256": prepared_hash,
        },
    )
    write_json(
        output_root / "frozen-contract.json",
        {
            "gate": "T2-05",
            "base_commit": BASE_COMMIT,
            "evaluation_role": "pre_frozen_strong_cross_encoder_calibration",
            "fresh_blind_test": False,
            "model_id": EXPECTED_MODEL_ID,
            "model_revision": MODEL_REVISION,
            "instruction": INSTRUCTION,
            "instruction_sha256": contracts["instruction"]["instruction_sha256"],
            "candidate_depth": CANDIDATE_DEPTH,
            "candidate_source": "frozen T2-01 BM25 Top50",
            "batch_size": 1,
            "max_length": MAX_LENGTH,
            "dtype": "bfloat16",
            "scoring": "last-position full-vocabulary causal-lm logits; log_softmax(no, yes) yes probability",
            "method_hash": METHOD_HASH,
            "feature_seal": FEATURE_SEAL,
            "qwen_contract_frozen_before_test_unlock": True,
            "gold_reads_before_prediction_seal": 0,
        },
    )

    probe_rows, empty_injected = select_probe_queries(rows)
    probe_ids = {str(row["query_id"]) for row in probe_rows}
    probe_candidates = frozen_bm25_top50(args.retrieval_root / "bm25-predictions.jsonl.gz", probe_ids)
    probe_prepared = prepare_input_rows(probe_rows, probe_candidates, probe_ids)
    probe_path = output_root / "probe-input.jsonl.gz"
    write_gz(probe_path, probe_prepared)
    r0_manifest = read_json(args.t203_root / "r0-input-manifest.json")
    probe_identity = input_hash(probe_prepared)
    query_id_hash = sha256_text(json.dumps([row["query_id"] for row in probe_prepared], separators=(",", ":")))
    if query_id_hash != r0_manifest["query_ids_sha256"] or probe_identity != r0_manifest["candidate_identity_sha256"]:
        raise RuntimeError("r0_probe_sample_mutation")
    if empty_injected != r0_manifest["empty_question_injected"]:
        raise RuntimeError("r0_empty_question_selection_mutation")
    probe_ranges = shard_ranges(len(probe_prepared), selected_gpus)

    # Recreate the single-GPU reference ordering because the historical R0 artifact
    # contains the contract and timing but no score/order payload.
    reference_gpu = selected_gpus[-1]
    reference_range = [{"shard_id": 0, "gpu_index": reference_gpu, "start": 0, "end": len(probe_prepared)}]
    reference_stats = launch_workers(
        Path(__file__), probe_path, args.dataset_root, output_root, reference_range, "r0-reference"
    )
    reference_probe_rows = read_gz(output_root / "predictions-shard-0.jsonl.gz")
    reference_probe_path = output_root / "r0-reference-predictions.jsonl.gz"
    write_gz(reference_probe_path, reference_probe_rows)
    reference_stats[0]["prediction_sha256"] = sha256_file(reference_probe_path)
    write_json(output_root / "r0-reference-stats.json", reference_stats[0])
    for path in (output_root / "predictions-shard-0.jsonl.gz", output_root / "shard-0-stats.json", output_root / "shard-0.log"):
        if path.exists():
            path.unlink()

    multi_stats = launch_workers(
        Path(__file__), probe_path, args.dataset_root, output_root, probe_ranges, "r0-multi-gpu"
    )
    multi_probe_rows = []
    for shard in probe_ranges:
        multi_probe_rows.extend(read_gz(output_root / f"predictions-shard-{shard['shard_id']}.jsonl.gz"))
    multi_probe_rows.sort(key=lambda row: [item["query_id"] for item in probe_prepared].index(row["query_id"]))
    equivalence = compare_probe(reference_probe_rows, multi_probe_rows)
    equivalence.update(
        {
            "reference_backend": "transformers_reference_single_gpu",
            "multi_gpu_backend": "transformers_reference_independent_replicas",
            "historical_r0_ordering_payload_available": False,
            "reference_recreated_from_frozen_t2_03_contract": True,
            "reference_stats": reference_stats[0],
            "multi_stats": multi_stats,
            "probe_query_selection_hash": query_id_hash,
            "probe_candidate_identity_hash": probe_identity,
        }
    )
    write_json(output_root / "runtime-equivalence.json", equivalence)
    if not equivalence["multi_gpu_runtime_accepted"]:
        write_json(
            output_root / "decision.json",
            {
                "gate": "T2-05",
                "base_commit": BASE_COMMIT,
                "decision": "multi_gpu_runtime_blocked",
                "multi_gpu_runtime_accepted": False,
                "runtime_equivalence": equivalence["runtime_equivalence"],
                "gold_reads": 0,
                "next_gate": "t2_05_single_gpu_reference",
            },
        )
        return {"decision": "multi_gpu_runtime_blocked"}

    # Formal one-shot scoring.
    ranges = shard_ranges(PRIMARY_COUNT, selected_gpus)
    full_stats = launch_workers(
        Path(__file__), output_root / "prepared-input.jsonl.gz", args.dataset_root, output_root, ranges, "formal"
    )
    merged_rows, prediction_sha = merge_shards(prepared, output_root, ranges)
    prediction_manifest = {
        "gate": "T2-05",
        "evaluation_role": "pre_frozen_strong_cross_encoder_calibration",
        "fresh_blind_test": False,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": MODEL_REVISION,
        "method_hash": METHOD_HASH,
        "feature_seal": FEATURE_SEAL,
        "candidate_depth": CANDIDATE_DEPTH,
        "query_count": PRIMARY_COUNT,
        "pair_count": FORMAL_PAIRS,
        "prepared_input_sha256": prepared_hash,
        "prediction_sha256": prediction_sha,
        "shards": full_stats,
        "gold_reads_before_seal": 0,
        "candidate_mutation": 0,
        "batch_size": 1,
    }
    write_json(output_root / "prediction-manifest.json", prediction_manifest)
    prediction_seal = {
        "gate": "T2-05",
        "sealed": True,
        "prediction_count": PRIMARY_COUNT,
        "pair_count": FORMAL_PAIRS,
        "prediction_sha256": prediction_sha,
        "method_hash": METHOD_HASH,
        "feature_seal": FEATURE_SEAL,
        "gold_reads_before_seal": 0,
        "candidate_mutation": 0,
        "recall_at_50_invariant_pending_score": True,
    }
    write_json(output_root / "prediction-seal.json", prediction_seal)

    # Gold is unlocked only after the Qwen prediction seal.
    gold = load_gold(args.dataset_root, primary_query_ids)
    qids = [row["query_id"] for row in prepared]
    bm25_order = {query_id: [item["context_id"] for item in candidate_map[query_id]["ranked_contexts"]] for query_id in qids}
    qwen_order = rank_from_prediction_rows(merged_rows)
    pcr_order = {}
    for row in read_gz(args.pcr_predictions):
        query_id = str(row["query_id"])
        if query_id not in primary_query_ids:
            continue
        ranked = sorted(
            row["ranked_contexts"],
            key=lambda item: int(item["post_rank"]),
        )
        pcr_order[query_id] = [str(item["context_id"]) for item in ranked]
    if set(pcr_order) != set(primary_query_ids):
        raise RuntimeError("pcr_prediction_identity")
    metrics = {
        "bm25": score(bm25_order, qids, gold),
        "pcr_v1": score(pcr_order, qids, gold),
        "qwen3_reranker_4b": score(qwen_order, qids, gold),
    }
    if metrics["qwen3_reranker_4b"]["hits"]["50"] != metrics["bm25"]["hits"]["50"]:
        raise RuntimeError("qwen_r50_invariant")
    write_json(output_root / "metrics.json", metrics)
    write_json(
        output_root / "method-comparison.json",
        {
            "metrics": metrics,
            "qwen_gain_over_bm25_pp": round(
                metrics["qwen3_reranker_4b"]["recall_pct"]["@5"] - metrics["bm25"]["recall_pct"]["@5"], 6
            ),
            "qwen_gain_over_pcr_pp": round(
                metrics["qwen3_reranker_4b"]["recall_pct"]["@5"] - metrics["pcr_v1"]["recall_pct"]["@5"], 6
            ),
        },
    )
    write_json(
        output_root / "rank-movement.json",
        {
            "vs_bm25": movement(bm25_order, qwen_order, gold),
            "vs_pcr_v1": movement(pcr_order, qwen_order, gold),
        },
    )
    write_json(output_root / "rank-compression.json", rank_compression(bm25_order, qwen_order, gold))

    # Reuse the A.2 query parser solely for frozen period/type diagnostics.
    query_structures = load_a2_query_structures(args.retrieval_root, primary_query_ids)
    def cohort(ids: list[str]) -> dict[str, Any]:
        return {
            "query_count": len(ids),
            "bm25": score(bm25_order, ids, gold),
            "pcr_v1": score(pcr_order, ids, gold),
            "qwen": score(qwen_order, ids, gold),
            "movement_vs_bm25": movement(
                {qid: bm25_order[qid] for qid in ids},
                {qid: qwen_order[qid] for qid in ids},
                {qid: gold[qid] for qid in ids},
            ),
        }
    period_ids = [qid for qid in qids if query_structures[qid]["periods"]]
    no_period_ids = [qid for qid in qids if not query_structures[qid]["periods"]]
    write_json(
        output_root / "period-cohort.json",
        {
            "period_required": cohort(period_ids),
            "no_period_requirement": {
                **cohort(no_period_ids),
                "qwen_ranking_identical_to_bm25": all(bm25_order[qid] == qwen_order[qid] for qid in no_period_ids),
                "pcr_ranking_identical_to_bm25": all(bm25_order[qid] == pcr_order[qid] for qid in no_period_ids),
            },
        },
    )
    types = defaultdict(list)
    for query_id in qids:
        types[query_structures[query_id]["operation_intent"]].append(query_id)
    query_type_output = {}
    for operation, ids in sorted(types.items()):
        query_type_output[operation] = {
            "query_count": len(ids),
            "bm25_r_at_5_pct": score(bm25_order, ids, gold)["recall_pct"]["@5"],
            "pcr_r_at_5_pct": score(pcr_order, ids, gold)["recall_pct"]["@5"],
            "qwen_r_at_5_pct": score(qwen_order, ids, gold)["recall_pct"]["@5"],
            "movement_vs_bm25": movement(
                {qid: bm25_order[qid] for qid in ids},
                {qid: qwen_order[qid] for qid in ids},
                {qid: gold[qid] for qid in ids},
            ),
        }
    write_json(output_root / "query-type-analysis.json", query_type_output)
    subset_output = {}
    for subset in ("FinQA", "TAT-DQA"):
        ids = [qid for qid in qids if next(row["subset"] for row in prepared if row["query_id"] == qid) == subset]
        subset_output[subset] = {
            "query_count": len(ids),
            "bm25": score(bm25_order, ids, gold),
            "pcr_v1": score(pcr_order, ids, gold),
            "qwen": score(qwen_order, ids, gold),
        }
    write_json(output_root / "subset-analysis.json", subset_output)
    tat_gain = subset_output["TAT-DQA"]["qwen"]["recall_pct"]["@5"] - subset_output["TAT-DQA"]["bm25"]["recall_pct"]["@5"]
    write_json(
        output_root / "tatdqa-bottleneck-analysis.json",
        {
            "bm25_r_at_5_pct": subset_output["TAT-DQA"]["bm25"]["recall_pct"]["@5"],
            "pcr_r_at_5_pct": subset_output["TAT-DQA"]["pcr_v1"]["recall_pct"]["@5"],
            "qwen_r_at_5_pct": subset_output["TAT-DQA"]["qwen"]["recall_pct"]["@5"],
            "qwen_gain_over_bm25_pp": round(tat_gain, 6),
            "material_gain_threshold_pp": 3.0,
            "tatdqa_bottleneck": (
                "ranking_semantic_interaction" if tat_gain >= 3.0 else "candidate_representation_or_table_structure"
            ),
        },
    )
    qwen_gain = metrics["qwen3_reranker_4b"]["recall_pct"]["@5"] - metrics["bm25"]["recall_pct"]["@5"]
    if qwen_gain >= 3.0:
        effective = "strong"
    elif qwen_gain > 0:
        effective = "marginal"
    else:
        effective = "false"
    decision = {
        "gate": "T2-05",
        "base_commit": BASE_COMMIT,
        "evaluation_role": "pre_frozen_strong_cross_encoder_calibration",
        "fresh_blind_test": False,
        "qwen_contract_frozen_before_test_unlock": True,
        "primary_test_queries": PRIMARY_COUNT,
        "candidate_depth": CANDIDATE_DEPTH,
        "formal_pairs": FORMAL_PAIRS,
        "model": EXPECTED_MODEL_ID,
        "model_revision": MODEL_REVISION,
        "bm25_recall_at_5": metrics["bm25"]["recall_pct"]["@5"] / 100.0,
        "pcr_v1_recall_at_5": metrics["pcr_v1"]["recall_pct"]["@5"] / 100.0,
        "qwen_recall_at_5": metrics["qwen3_reranker_4b"]["recall_pct"]["@5"] / 100.0,
        "qwen_gain_over_bm25_pp": round(qwen_gain, 6),
        "qwen_gain_over_pcr_pp": round(
            metrics["qwen3_reranker_4b"]["recall_pct"]["@5"] - metrics["pcr_v1"]["recall_pct"]["@5"], 6
        ),
        "finqa_qwen_recall_at_5": subset_output["FinQA"]["qwen"]["recall_pct"]["@5"] / 100.0,
        "tatdqa_qwen_recall_at_5": subset_output["TAT-DQA"]["qwen"]["recall_pct"]["@5"] / 100.0,
        "candidate_mutation": 0,
        "recall_at_50_invariant": True,
        "qwen_prediction_preseal_gold_reads": 0,
        "strong_cross_encoder_effective": effective,
        "tatdqa_ranking_bottleneck_confirmed": tat_gain >= 3.0,
        "next_gate": "external_track_final_review",
        "gpu_indices": selected_gpus,
        "formal_runtime": full_stats,
        "runtime_equivalence": equivalence,
        "prediction_sha256": prediction_sha,
    }
    write_json(output_root / "decision.json", decision)
    (output_root / "README.md").write_text(
        "# T2-05 Pre-Frozen Qwen3-4B Strong Cross-Encoder Calibration\n\n"
        "Qwen3-Reranker-4B was applied once to frozen BM25 Top50 candidates. "
        "The model and scoring contract were frozen before the Primary Test was unlocked.\n",
        encoding="utf-8",
    )
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("coordinator", "worker"), default="coordinator")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--method-root", type=Path)
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument("--t203-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--pcr-predictions", type=Path)
    parser.add_argument("--prepared-input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--gpu-label", type=int)
    parser.add_argument("--label", default="worker")
    args = parser.parse_args()
    if args.mode == "worker":
        return run_worker_mode(args)
    if not all(
        value is not None
        for value in (
            args.retrieval_root,
            args.feature_root,
            args.method_root,
            args.protocol_root,
            args.t203_root,
            args.output_root,
            args.pcr_predictions,
        )
    ):
        parser.error("coordinator paths required")
    decision = main_coordinator(args)
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
