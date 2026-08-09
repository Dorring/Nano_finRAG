#!/usr/bin/env python3
"""Run focused slot queries, reuse exact main scores, and seal fixed Top5 composition."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.qwen3_reranker import build_input_ids, score_batch  # noqa: E402
from src.pdf_retrieval_v4.slot_aware_neural_composition import compose_slot_aware_top5  # noqa: E402
from src.pdf_retrieval_v4.structure_aware_rerank_view import RERANK_INSTRUCTION, sha256_text  # noqa: E402

BASE = ROOT / "artifacts/evaluation"
P0 = BASE / "pdf-retrieval-v4-gate-08-r8-r3-3-p0"
VIEWS = P0 / "queryplan-rerank-input-views.jsonl.gz"
R32 = BASE / "pdf-retrieval-v4-gate-08-r8-r3-2"
MAIN_SOURCE = R32 / "rerank-predictions.jsonl.gz"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r3-3"
MAIN_OUT = OUT / "main_rerank_predictions.jsonl.gz"
SLOT_OUT = OUT / "slot_rerank_predictions.jsonl.gz"
FINAL_OUT = OUT / "slot_aware_top5_predictions.jsonl.gz"
MODEL_ID = "Qwen/Qwen3-Reranker-4B"
REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
MAX_LENGTH = 8192
EXPECTED_VIEWS_SHA = "91bafe5612fab14d1229c877c9dd1bc290b815a8d73f7f014427ce916cdf1705"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def dump_gzip(path: Path, records: list[dict]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for record in records:
                zipped.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if sha(VIEWS) != EXPECTED_VIEWS_SHA:
        raise RuntimeError("r3_3_p0_views_sha_mismatch")
    p0_seal = json.loads((P0 / "input-seal.json").read_text())
    if p0_seal["main_query_changed_cases"] != 0 or p0_seal["main_query_reused_cases"] != 72:
        raise RuntimeError("main_score_reuse_contract_changed")
    OUT.mkdir(parents=True, exist_ok=True)
    source_views = {item["case_id"]: item for item in load_gzip(VIEWS)}
    main_source = {item["case_id"]: item for item in load_gzip(MAIN_SOURCE)}
    main_records = [{**record, "score_reused": True, "score_reuse_source_sha256": sha(MAIN_SOURCE)} for record in main_source.values()]
    snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots" / REVISION
    if not snapshot.is_dir():
        raise RuntimeError("exact_4b_snapshot_not_cached")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), padding_side="left", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), torch_dtype=torch.bfloat16, local_files_only=True).to("cuda:0").eval()
    started = time.time()
    slot_records = []
    slot_by_case: dict[str, dict[str, list[dict]]] = {}
    pairs = truncated = 0
    for case_id, source in source_views.items():
        slot_by_case[case_id] = {}
        for slot in source["slot_query_views"]:
            scored = []
            for candidate in source["candidates"]:
                ids, audit = build_input_ids(tokenizer, RERANK_INSTRUCTION, slot["query_view"], candidate["document_view"], MAX_LENGTH)
                score = score_batch(model, tokenizer, [ids])[0]
                scored.append({"candidate_key": candidate["candidate_key"], "pre_rerank_rank": candidate["pre_rerank_rank"], "slot_score": score["reranker_score"], "yes_logit": score["yes_logit"], "no_logit": score["no_logit"], "truncated": audit["truncated"], "query_view_sha256": slot["query_view_sha256"], "document_view_sha256": candidate["document_view_sha256"]})
                pairs += 1
                truncated += int(audit["truncated"])
            scored.sort(key=lambda item: (-item["slot_score"], item["pre_rerank_rank"], item["candidate_key"]))
            for rank, item in enumerate(scored, 1):
                item["slot_rank"] = rank
            slot_by_case[case_id][slot["slot_id"]] = scored
            slot_records.append({"case_id": case_id, "slot_id": slot["slot_id"], "query_view_sha256": slot["query_view_sha256"], "ranked_candidates": scored})
    if pairs != 3600 or len(slot_records) != 36:
        raise RuntimeError("slot_pair_count_contract_failed")
    final_records = []
    for case_id, main_record in main_source.items():
        main = main_record["ranked_candidates"]
        if slot_by_case[case_id]:
            selected = compose_slot_aware_top5(main, slot_by_case[case_id])
        else:
            selected = [{**item, "selection_source": "main", "final_rank": rank} for rank, item in enumerate(main[:5], 1)]
        final_records.append({"case_id": case_id, "is_multi_slot": bool(slot_by_case[case_id]), "candidates": selected})
    dump_gzip(MAIN_OUT, main_records)
    dump_gzip(SLOT_OUT, slot_records)
    dump_gzip(FINAL_OUT, final_records)
    elapsed = time.time() - started
    r32_manifest = json.loads((R32 / "model-manifest.json").read_text())
    manifest = {"cases": 72, "main_pair_count": 7200, "main_scores_reused": 7200, "main_scores_recomputed": 0, "slot_count": 36, "slot_pair_count": 3600, "final_top5_cases": 72, "candidate_added": 0, "candidate_removed": 0, "model_id": MODEL_ID, "model_revision": REVISION, "model_manifest_exact_r3_2": r32_manifest, "max_length": MAX_LENGTH, "dtype": "bfloat16", "batch_size": 1, "instruction_sha256": sha256_text(RERANK_INSTRUCTION), "p0_views_sha256": EXPECTED_VIEWS_SHA, "main_source_prediction_sha256": sha(MAIN_SOURCE), "main_prediction_sha256": sha(MAIN_OUT), "slot_prediction_sha256": sha(SLOT_OUT), "final_prediction_sha256": sha(FINAL_OUT)}
    protocol = {"gate": "pdf_retrieval_v4_gate_08_r8_r3_3", "gold_reads_before_seal": 0, "governance_reads_before_seal": 0, "reference_answer_reads": 0, "expected_value_reads": 0, "retrieval_runs": 0, "embedding_calls": 0, "index_reads": 0, "bridge_runs": 0, "candidate_added": 0, "candidate_removed": 0, "model_scan": False, "prompt_scan": False, "slot_quota_scan": False, "model_8b": False, "slot_top_n": 1, "final_top_k": 5, "production_writes": 0, "production_switch_allowed": False}
    write("protocol.json", protocol)
    write("input-integrity.json", {"candidate_identity_exact": "7200/7200", "main_score_reuse_exact": True, "p0_input_seal_sha256": sha(P0 / "input-seal.json")})
    write("runtime-manifest.json", {"elapsed_seconds": elapsed, "slot_pairs_per_second": pairs / elapsed, "truncated_pairs": truncated})
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**manifest, **protocol, "sealed": True})
    print(json.dumps({**manifest, "elapsed_seconds": elapsed, "truncated_pairs": truncated}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
