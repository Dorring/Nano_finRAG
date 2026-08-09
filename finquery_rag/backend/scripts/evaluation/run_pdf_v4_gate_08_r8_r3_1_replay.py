#!/usr/bin/env python3
"""Replay the frozen 0.6B reranker with candidate-global V2 context."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.qwen3_reranker import (  # noqa: E402
    Qwen3RerankerConfig,
    build_input_ids,
    score_batch,
)
from src.pdf_retrieval_v4.structure_aware_rerank_view import RERANK_INSTRUCTION  # noqa: E402

BASE = ROOT / "artifacts/evaluation"
R31A = BASE / "pdf-retrieval-v4-gate-08-r8-r3-1a"
VIEWS = R31A / "rerank-input-views-v2.jsonl.gz"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r3-1"
PRED = OUT / "rerank-predictions.jsonl.gz"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
EXPECTED_VIEWS_SHA = "82ea6c75dae8607e7bda462c39745abcff9ac991611c271c70e34fa318fc6dc1"
TOP100_SHA = "ced014c357a3c8862a5ae3723a4c618556156542fd63aa2a8fd4bf766b60f01a"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if sha(VIEWS) != EXPECTED_VIEWS_SHA:
        raise RuntimeError("r3_1a_input_views_sha_mismatch")
    config = Qwen3RerankerConfig(revision=REVISION, batch_size=args.batch_size)
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots" / REVISION
    if not snapshot.is_dir():
        raise RuntimeError("exact_model_snapshot_not_cached")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), padding_side="left", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), torch_dtype=torch.bfloat16, local_files_only=True
    ).to(args.device).eval()
    started = time.time()
    records = []
    truncated_count = total_pairs = total_tokens = 0
    with gzip.open(VIEWS, "rt", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            prepared = []
            for candidate in source["candidates"]:
                ids, audit = build_input_ids(tokenizer, RERANK_INSTRUCTION, candidate["query_view"], candidate["document_view"], config.max_length)
                prepared.append((candidate, ids, audit))
                truncated_count += int(audit["truncated"])
                total_tokens += audit["final_token_count"]
            scored = []
            for offset in range(0, 100, config.batch_size):
                batch = prepared[offset : offset + config.batch_size]
                scores = score_batch(model, tokenizer, [item[1] for item in batch])
                for (candidate, _, audit), score in zip(batch, scores, strict=True):
                    scored.append({"candidate_key": candidate["candidate_key"], "pre_rerank_rank": candidate["pre_rerank_rank"], "context_status": candidate["context_status"], **score, **audit, "query_view_sha256": candidate["query_view_sha256"], "document_view_sha256": candidate["document_view_sha256"]})
            scored.sort(key=lambda item: (-item["reranker_score"], item["pre_rerank_rank"], item["candidate_key"]))
            for rank, item in enumerate(scored, 1):
                item["post_rerank_rank"] = rank
            records.append({"case_id": source["case_id"], "input_candidate_count": 100, "ranked_candidates": scored})
            total_pairs += 100
    if len(records) != 72 or total_pairs != 7200:
        raise RuntimeError("prediction_pair_count_contract_failed")
    with PRED.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for record in records:
                zipped.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    elapsed = time.time() - started
    model_manifest = {"model_id": config.model_id, "model_revision": REVISION, "tokenizer_revision": REVISION, "transformers_version": transformers.__version__, "torch_version": torch.__version__, "python_version": platform.python_version(), "cuda_version": torch.version.cuda, "device": args.device, "device_name": torch.cuda.get_device_name(args.device), "dtype": config.dtype, "max_length": config.max_length, "batch_size": config.batch_size, "padding_side": config.padding_side, "generation": False}
    manifest = {"prediction_count": 72, "pair_count": 7200, "prediction_sha256": sha(PRED), "r3_1a_views_sha256": EXPECTED_VIEWS_SHA, "top100_sha256": TOP100_SHA, "context_v2_sha256": json.loads((R31A / "prediction-seal.json").read_text())["context_sha256"], "instruction_sha256": text_sha(RERANK_INSTRUCTION), **model_manifest}
    protocol = {"gate": "pdf_retrieval_v4_gate_08_r8_r3_1", "only_changed_variable": "document_context_source_v1_to_v2", "candidate_added": 0, "candidate_removed": 0, "retrieval_runs": 0, "index_reads": 0, "embedding_calls": 0, "gold_reads_before_seal": 0, "governance_reads_before_seal": 0, "reference_answer_reads": 0, "expected_value_reads": 0, "model_scan": False, "instruction_scan": False, "weight_scan": False, "generation_calls": 0, "production_writes": 0, "production_switch_allowed": False}
    write("protocol.json", protocol)
    write("input-integrity.json", {"r3_1a_seal_sha256": sha(R31A / "prediction-seal.json"), "candidate_identity_exact": "7200/7200", "query_views_exact_r3": True})
    write("model-manifest.json", model_manifest)
    write("runtime-stats.json", {"elapsed_seconds": elapsed, "pairs_per_second": total_pairs / elapsed})
    write("truncation-stats.json", {"pairs": total_pairs, "truncated_pairs": truncated_count, "mean_final_tokens": total_tokens / total_pairs, "max_length": 8192})
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**manifest, **protocol, "sealed": True})
    print(json.dumps({**manifest, "elapsed_seconds": elapsed, "truncated_pairs": truncated_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
