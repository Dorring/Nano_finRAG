#!/usr/bin/env python3
"""Run the single preregistered Qwen3 0.6B cross-encoder prediction and seal it."""

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
from huggingface_hub import HfApi
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
P0 = BASE / "pdf-retrieval-v4-gate-08-r8-r3-p0"
VIEWS = P0 / "rerank-input-views.jsonl.gz"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r3"
PRED = OUT / "rerank-predictions.jsonl.gz"
EXPECTED_VIEWS_SHA = "3227ad35c0937813bb260f0708aae0bb129f44e8649dfc8bb4d34428621fbfc6"


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
        raise RuntimeError("p0_input_views_sha_mismatch")
    if tuple(map(int, transformers.__version__.split(".")[:2])) < (4, 51):
        raise RuntimeError("transformers_4_51_or_newer_required")
    revision = HfApi().model_info(Qwen3RerankerConfig.model_id).sha
    config = Qwen3RerankerConfig(revision=revision, batch_size=args.batch_size)
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=revision, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=revision, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    if tokenizer.encode("yes", add_special_tokens=False) != [tokenizer.convert_tokens_to_ids("yes")]:
        raise RuntimeError("yes_token_contract_failed")
    if tokenizer.encode("no", add_special_tokens=False) != [tokenizer.convert_tokens_to_ids("no")]:
        raise RuntimeError("no_token_contract_failed")
    started = time.time()
    prediction_records = []
    truncated_count = total_pairs = total_tokens = 0
    with gzip.open(VIEWS, "rt", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            prepared = []
            for candidate in source["candidates"]:
                ids, token_audit = build_input_ids(
                    tokenizer, RERANK_INSTRUCTION, candidate["query_view"],
                    candidate["document_view"], config.max_length,
                )
                prepared.append((candidate, ids, token_audit))
                truncated_count += int(token_audit["truncated"])
                total_tokens += token_audit["final_token_count"]
            scored = []
            for offset in range(0, len(prepared), config.batch_size):
                batch = prepared[offset : offset + config.batch_size]
                scores = score_batch(model, tokenizer, [item[1] for item in batch])
                for (candidate, _, audit), score in zip(batch, scores, strict=True):
                    scored.append({
                        "candidate_key": candidate["candidate_key"],
                        "pre_rerank_rank": candidate["pre_rerank_rank"],
                        **score,
                        **audit,
                        "query_view_sha256": candidate["query_view_sha256"],
                        "document_view_sha256": candidate["document_view_sha256"],
                    })
            scored.sort(key=lambda item: (-item["reranker_score"], item["pre_rerank_rank"], item["candidate_key"]))
            for rank, item in enumerate(scored, 1):
                item["post_rerank_rank"] = rank
            prediction_records.append({"case_id": source["case_id"], "input_candidate_count": len(scored), "ranked_candidates": scored})
            total_pairs += len(scored)
    if len(prediction_records) != 72 or total_pairs != 7200:
        raise RuntimeError("prediction_pair_count_contract_failed")
    with PRED.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for record in prediction_records:
                zipped.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    elapsed = time.time() - started
    model_manifest = {
        "model_id": config.model_id, "model_revision": revision,
        "tokenizer_revision": revision, "transformers_version": transformers.__version__,
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "cuda_version": torch.version.cuda, "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device), "dtype": config.dtype,
        "max_length": config.max_length, "batch_size": config.batch_size,
        "padding_side": config.padding_side, "generation": False,
    }
    serializer_source = ROOT / "src/pdf_retrieval_v4/structure_aware_rerank_view.py"
    reranker_source = ROOT / "src/pdf_retrieval_v4/qwen3_reranker.py"
    manifest = {
        "prediction_file": PRED.name, "prediction_count": 72, "pair_count": 7200,
        "prediction_sha256": sha(PRED), "p0_input_views_sha256": EXPECTED_VIEWS_SHA,
        "top100_sha256": json.loads((P0 / "input-seal.json").read_text())["top100_sha256"],
        "serializer_sha256": sha(serializer_source), "reranker_source_sha256": sha(reranker_source),
        "instruction_sha256": text_sha(RERANK_INSTRUCTION), **model_manifest,
    }
    protocol = {
        "gate": "pdf_retrieval_v4_gate_08_r8_r3", "candidate_added": 0,
        "candidate_removed": 0, "retrieval_runs": 0, "index_reads": 0,
        "embedding_calls": 0, "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0, "reference_answer_reads": 0,
        "expected_value_reads": 0, "model_scan": False, "instruction_scan": False,
        "max_length_scan": False, "weight_scan": False, "query_rewrite": False,
        "generation_calls": 0, "calculator_calls": 0, "production_writes": 0,
        "production_switch_allowed": False,
    }
    write("protocol.json", protocol)
    write("input-integrity.json", {"p0_input_seal_sha256": sha(P0 / "input-seal.json"), "candidate_identity_exact": "7200/7200", "input_views_sha256": EXPECTED_VIEWS_SHA})
    write("model-manifest.json", model_manifest)
    write("serializer-manifest.json", json.loads((P0 / "serializer-manifest.json").read_text()))
    write("runtime-stats.json", {"elapsed_seconds": elapsed, "pairs_per_second": total_pairs / elapsed, "batch_size": config.batch_size})
    write("truncation-stats.json", {"pairs": total_pairs, "truncated_pairs": truncated_count, "mean_final_tokens": total_tokens / total_pairs, "max_length": config.max_length})
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**manifest, **protocol, "sealed": True})
    print(json.dumps({**manifest, "elapsed_seconds": elapsed, "truncated_pairs": truncated_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
