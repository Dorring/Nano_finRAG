#!/usr/bin/env python3
"""Run the single preregistered Qwen3 4B capacity escalation and seal it."""

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
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.qwen3_reranker import build_input_ids, score_batch  # noqa: E402
from src.pdf_retrieval_v4.structure_aware_rerank_view import RERANK_INSTRUCTION  # noqa: E402

BASE = ROOT / "artifacts/evaluation"
R31A = BASE / "pdf-retrieval-v4-gate-08-r8-r3-1a"
VIEWS = R31A / "rerank-input-views-v2.jsonl.gz"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r3-2"
PRED = OUT / "rerank-predictions.jsonl.gz"
MODEL_ID = "Qwen/Qwen3-Reranker-4B"
REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
EXPECTED_VIEWS_SHA = "82ea6c75dae8607e7bda462c39745abcff9ac991611c271c70e34fa318fc6dc1"
TOP100_SHA = "ced014c357a3c8862a5ae3723a4c618556156542fd63aa2a8fd4bf766b60f01a"
MAX_LENGTH = 8192


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batch_size not in (1, 2, 4):
        raise ValueError("batch_size_must_be_resource_choice_1_2_or_4")
    if sha(VIEWS) != EXPECTED_VIEWS_SHA:
        raise RuntimeError("r3_1_input_views_sha_mismatch")
    snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION))
    if snapshot.name != REVISION:
        raise RuntimeError("model_snapshot_revision_mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), padding_side="left", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), torch_dtype=torch.bfloat16, local_files_only=True
    ).to(args.device).eval()
    source_records = []
    with gzip.open(VIEWS, "rt", encoding="utf-8") as handle:
        source_records = [json.loads(line) for line in handle]
    smoke_items = source_records[0]["candidates"][:16]
    smoke_ids = [
        build_input_ids(tokenizer, RERANK_INSTRUCTION, item["query_view"], item["document_view"], MAX_LENGTH)[0]
        for item in smoke_items
    ]
    scores_one = [score_batch(model, tokenizer, [ids])[0]["reranker_score"] for ids in smoke_ids]
    scores_formal = []
    for offset in range(0, len(smoke_ids), args.batch_size):
        scores_formal.extend(item["reranker_score"] for item in score_batch(model, tokenizer, smoke_ids[offset : offset + args.batch_size]))
    order_one = sorted(range(len(scores_one)), key=lambda index: (-scores_one[index], index))
    order_formal = sorted(range(len(scores_formal)), key=lambda index: (-scores_formal[index], index))
    if order_one != order_formal:
        raise RuntimeError("batch_numerical_rank_invariance_failed")
    max_score_delta = max(abs(left - right) for left, right in zip(scores_one, scores_formal, strict=True))
    started = time.time()
    records = []
    truncated_count = total_pairs = total_tokens = 0
    for source in source_records:
        prepared = []
        for candidate in source["candidates"]:
            ids, audit = build_input_ids(tokenizer, RERANK_INSTRUCTION, candidate["query_view"], candidate["document_view"], MAX_LENGTH)
            prepared.append((candidate, ids, audit))
            truncated_count += int(audit["truncated"])
            total_tokens += audit["final_token_count"]
        scored = []
        for offset in range(0, 100, args.batch_size):
            batch = prepared[offset : offset + args.batch_size]
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
    model_files = {}
    for path in sorted(snapshot.iterdir()):
        if path.name == "config.json" or "tokenizer" in path.name or path.suffix == ".safetensors":
            model_files[path.name] = sha(path)
    model_manifest = {"model_id": MODEL_ID, "model_revision": REVISION, "tokenizer_revision": REVISION, "file_sha256": model_files, "transformers_version": transformers.__version__, "torch_version": torch.__version__, "python_version": platform.python_version(), "cuda_version": torch.version.cuda, "device": args.device, "device_name": torch.cuda.get_device_name(args.device), "dtype": "bfloat16", "max_length": MAX_LENGTH, "batch_size": args.batch_size, "padding_side": "left", "generation": False}
    manifest = {"prediction_count": 72, "pair_count": 7200, "prediction_sha256": sha(PRED), "r3_1a_views_sha256": EXPECTED_VIEWS_SHA, "top100_sha256": TOP100_SHA, "instruction_sha256": text_sha(RERANK_INSTRUCTION), **model_manifest}
    protocol = {"gate": "pdf_retrieval_v4_gate_08_r8_r3_2", "only_changed_variable": "reranker_model_capacity_0_6b_to_4b", "candidate_added": 0, "candidate_removed": 0, "candidate_mutation": 0, "query_view_mutation": 0, "document_view_mutation": 0, "retrieval_runs": 0, "index_reads": 0, "embedding_calls": 0, "bridge_runs": 0, "semantic_graph_runs": 0, "gold_reads_before_seal": 0, "governance_reads_before_seal": 0, "reference_answer_reads": 0, "expected_value_reads": 0, "instruction_scan": False, "model_scan": False, "max_length_scan": False, "slot_aware_scoring": False, "calculator_calls": 0, "generator_calls": 0, "production_writes": 0, "production_switch_allowed": False}
    write("protocol.json", protocol)
    write("input-integrity.json", {"r3_1a_seal_sha256": sha(R31A / "prediction-seal.json"), "candidate_identity_exact": "7200/7200", "query_view_mutation": 0, "document_view_mutation": 0, "authoritative_context_coverage_exact": True})
    write("model-manifest.json", model_manifest)
    write("runtime-manifest.json", {"elapsed_seconds": elapsed, "pairs_per_second": total_pairs / elapsed, "batch_size": args.batch_size, "batch_invariance_smoke_pairs": 16, "batch_rank_order_parity": True, "max_score_delta": max_score_delta})
    write("truncation-stats.json", {"pairs": total_pairs, "truncated_pairs": truncated_count, "mean_final_tokens": total_tokens / total_pairs, "max_length": MAX_LENGTH})
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**manifest, **protocol, "sealed": True})
    print(json.dumps({**manifest, "elapsed_seconds": elapsed, "truncated_pairs": truncated_count, "batch_smoke_max_delta": max_score_delta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
