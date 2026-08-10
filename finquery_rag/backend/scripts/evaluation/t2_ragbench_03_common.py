#!/usr/bin/env python3
"""Shared frozen T2-03 Qwen3 reranking contracts."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable


EXPECTED_ROWS = 23_088
EXPECTED_DATASET_COMMIT = "adf7fe1541ac37351ce1142544d8e3b43010ed92"
EXPECTED_BM25_PREDICTION_ROOT_NAME = "t2-ragbench-01-standard-retrieval"
EXPECTED_MODEL_ID = "Qwen/Qwen3-Reranker-4B"
MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
MODEL_SNAPSHOT = Path(
    "/home/mxf/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B"
) / "snapshots" / MODEL_REVISION
INSTRUCTION = (
    "Given a financial question, determine whether the candidate financial report context "
    "contains the evidence needed to answer the question. Consider both narrative text and tables. "
    "Rank contexts by direct evidential relevance to the question."
)
MAX_LENGTH = 8192
CANDIDATE_DEPTH = 50
BATCH_SIZE = 1
DTYPE_NAME = "bfloat16"
DEVICE = "cuda:0"
SYSTEM_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SYSTEM_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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


def load_contexts(dataset_root: Path) -> tuple[dict[str, str], dict[str, int]]:
    contexts: dict[str, str] = {}
    subset_counts: dict[str, set[str]] = {"FinQA": set(), "ConvFinQA": set(), "TAT-DQA": set()}
    for subset, _, path in metadata_paths(dataset_root):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                context_id = str(source["context_id"])
                context = str(source.get("context") or "")
                previous = contexts.get(context_id)
                if previous is not None and previous != context:
                    raise RuntimeError(f"context_id_content_conflict:{context_id}")
                contexts[context_id] = context
                subset_counts[subset].add(context_id)
    counts = {subset: len(values) for subset, values in subset_counts.items()}
    if sum(counts.values()) != 7318:
        raise RuntimeError(f"context_count_contract:{counts}")
    return contexts, counts


def load_query_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id in seen:
                raise RuntimeError(f"duplicate_query_manifest:{query_id}")
            seen.add(query_id)
            rows.append(row)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"query_count_contract:{len(rows)}")
    return rows


def iter_bm25_predictions(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def frozen_bm25_top50(prediction_path: Path, expected_query_ids: set[str] | None = None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in iter_bm25_predictions(prediction_path):
        query_id = str(row["query_id"])
        if expected_query_ids is not None and query_id not in expected_query_ids:
            continue
        ranked = row.get("ranked_contexts") or []
        if len(ranked) < CANDIDATE_DEPTH:
            raise RuntimeError(f"bm25_candidate_depth:{query_id}:{len(ranked)}")
        top50 = ranked[:CANDIDATE_DEPTH]
        ids = [str(item["context_id"]) for item in top50]
        ranks = [int(item["rank"]) for item in top50]
        if ranks != list(range(1, CANDIDATE_DEPTH + 1)):
            raise RuntimeError(f"bm25_rank_contract:{query_id}")
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"bm25_duplicate_candidate:{query_id}")
        result[query_id] = {
            "query_id": query_id,
            "subset": row["subset"],
            "ranked_contexts": [
                {
                    "context_id": str(item["context_id"]),
                    "original_bm25_rank": int(item["rank"]),
                }
                for item in top50
            ],
        }
    if expected_query_ids is not None and set(result) != expected_query_ids:
        raise RuntimeError("bm25_sample_identity_contract")
    return result


def format_pair(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def prepare_batch(tokenizer: Any, pairs: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prefix_tokens = tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SYSTEM_SUFFIX, add_special_tokens=False)
    available = MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens)
    if available <= 0:
        raise RuntimeError("invalid_max_length_budget")
    raw = tokenizer(
        pairs,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        add_special_tokens=True,
    )
    raw_ids = raw["input_ids"]
    token_meta: list[dict[str, Any]] = []
    truncated_ids: list[list[int]] = []
    for ids in raw_ids:
        before = len(ids) + len(prefix_tokens) + len(suffix_tokens)
        clipped = ids[:available]
        after = len(clipped) + len(prefix_tokens) + len(suffix_tokens)
        truncated_ids.append(prefix_tokens + clipped + suffix_tokens)
        token_meta.append(
            {
                "token_count_before_truncation": before,
                "token_count_after_truncation": after,
                "truncated": before > MAX_LENGTH,
            }
        )
    encoded = tokenizer.pad(
        {"input_ids": truncated_ids},
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
        max_length=MAX_LENGTH,
    )
    return encoded, token_meta


def runtime_manifest(tokenizer: Any, torch: Any, transformers: Any, model: Any) -> dict[str, Any]:
    gpu = torch.cuda.get_device_properties(0)
    return {
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": transformers.__version__,
        "sentence_transformers_version": None,
        "sentence_transformers_used": False,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_model": gpu.name,
        "gpu_index": 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_map": os.environ.get("T2_QWEN_DEVICE_MAP", "single_gpu"),
        "attention_implementation": os.environ.get("T2_QWEN_ATTN", "eager"),
        "dtype": DTYPE_NAME,
        "device": DEVICE,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "token_true_id": 9693,
        "token_false_id": 2152,
        "tokenizer_padding_side": tokenizer.padding_side,
        "model_max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
    }


def model_file_manifest(snapshot: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        files[str(path.relative_to(snapshot))] = {"size": path.stat().st_size, "sha256": sha256(path)}
    return {"model_id": EXPECTED_MODEL_ID, "revision": MODEL_REVISION, "snapshot": str(snapshot), "files": files}


def score_batch(model: Any, tokenizer: Any, torch: Any, pairs: list[str]) -> tuple[list[float], list[dict[str, Any]]]:
    inputs, token_meta = prepare_batch(tokenizer, pairs)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits[:, -1, :]
        yes_logits = logits[:, 9693]
        no_logits = logits[:, 2152]
        scores = torch.log_softmax(torch.stack([no_logits, yes_logits], dim=1), dim=1)[:, 1].exp()
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("non_finite_score")
    return [float(value) for value in scores.detach().cpu().tolist()], token_meta


def load_qwen_runtime() -> tuple[Any, Any, Any, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("cuda_unavailable")
    if os.environ.get("T2_QWEN_SDPA_MEMORY_EFFICIENT") == "1":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_SNAPSHOT),
        revision=MODEL_REVISION,
        local_files_only=True,
        padding_side="left",
    )
    load_kwargs = {
        "revision": MODEL_REVISION,
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if os.environ.get("T2_QWEN_DEVICE_MAP") == "auto":
        load_kwargs["device_map"] = "auto"
    if os.environ.get("T2_QWEN_ATTN"):
        load_kwargs["attn_implementation"] = os.environ["T2_QWEN_ATTN"]
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_SNAPSHOT), **load_kwargs)
    if os.environ.get("T2_QWEN_DEVICE_MAP") != "auto":
        model.to(DEVICE)
    model.eval()
    if tokenizer.convert_tokens_to_ids("yes") != 9693 or tokenizer.convert_tokens_to_ids("no") != 2152:
        raise RuntimeError("yes_no_token_contract")
    return tokenizer, model, torch, transformers


def now() -> float:
    return time.perf_counter()

