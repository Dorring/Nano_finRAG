"""NF-OPT-17 Gate F: one pre-registered issuer-disjoint reranker training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.evaluation.nf_opt_17_gate_d import rank_triplet, render_structured_reranker_input

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/nf-opt-17-gate-f"
TRAIN_ISSUERS = ("Alphabet Inc.", "AMAZON COM INC", "Meta Platforms, Inc.")
VALIDATION_ISSUER = "NETFLIX INC"
SEED = 20260805


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _triplet(annotation: dict[str, Any], candidates: dict[str, dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    refs = [annotation["positive_candidate"]]
    refs.extend(item["candidate"] for item in annotation["hard_negatives"])
    rendered = [render_structured_reranker_input(candidates[item["candidate_key"]]) for item in refs]
    kinds = [item["negative_type"] for item in annotation["hard_negatives"]]
    return str(annotation["question"]), rendered, kinds


def _encode(tokenizer: Any, questions: list[str], documents: list[str], max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        questions,
        documents,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


@torch.no_grad()
def _evaluate(
    model: Any,
    tokenizer: Any,
    annotations: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    *,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    records = []
    negative_correct = Counter()
    negative_total = Counter()
    for annotation in annotations:
        question, rendered, kinds = _triplet(annotation, candidates)
        inputs = _encode(tokenizer, [question] * 3, rendered, max_length, device)
        scores = model(**inputs).logits.reshape(-1).float().cpu().tolist()
        if len(scores) != 3 or not all(math.isfinite(value) for value in scores):
            raise ValueError("invalid validation scores")
        ranking = rank_triplet(scores, kinds)
        for kind, passed in ranking["pairwise"].items():
            negative_total[kind] += 1
            negative_correct[kind] += int(passed)
        records.append(
            {
                "annotation_id": annotation["annotation_id"],
                "query_id": annotation["query_id"],
                "candidate_keys": [
                    annotation["positive_candidate"]["candidate_key"],
                    *[item["candidate"]["candidate_key"] for item in annotation["hard_negatives"]],
                ],
                "scores": [round(value, 8) for value in scores],
                "ranking": ranking,
            }
        )
    count = len(records)
    pair_count = sum(negative_total.values())
    return {
        "query_count": count,
        "top1_count": sum(int(row["ranking"]["positive_top1"]) for row in records),
        "top1_accuracy": sum(int(row["ranking"]["positive_top1"]) for row in records) / count,
        "mrr": sum(float(row["ranking"]["reciprocal_rank"]) for row in records) / count,
        "pairwise_correct_count": sum(negative_correct.values()),
        "pairwise_count": pair_count,
        "pairwise_accuracy": sum(negative_correct.values()) / pair_count,
        "pairwise_by_negative_type": {
            kind: {
                "correct": negative_correct[kind],
                "count": negative_total[kind],
                "accuracy": negative_correct[kind] / negative_total[kind],
            }
            for kind in sorted(negative_total)
        },
        "records": records,
    }


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _set_seed()
    annotations = _read_jsonl(args.annotations)
    candidates = {row["candidate_key"]: row for row in _read_jsonl(args.candidates)}
    train = [row for row in annotations if row["issuer"] in TRAIN_ISSUERS]
    validation = [row for row in annotations if row["issuer"] == VALIDATION_ISSUER]
    if len(train) != 60 or len(validation) != 20:
        raise ValueError("frozen issuer split must contain 60 train and 20 validation queries")
    if {row["issuer"] for row in train} & {row["issuer"] for row in validation}:
        raise ValueError("issuer leakage across train and validation")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        raise ValueError("model does not expose a classification head")
    for parameter in classifier.parameters():
        parameter.requires_grad = True
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    generator = random.Random(SEED)
    optimizer_steps = 0
    micro_steps = 0
    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(args.epochs):
        order = list(range(len(train)))
        generator.shuffle(order)
        for start in range(0, len(order), args.batch_queries):
            batch = [train[index] for index in order[start : start + args.batch_queries]]
            questions: list[str] = []
            documents: list[str] = []
            for annotation in batch:
                question, rendered, _kinds = _triplet(annotation, candidates)
                questions.extend([question] * 3)
                documents.extend(rendered)
            inputs = _encode(tokenizer, questions, documents, args.max_length, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16):
                logits = model(**inputs).logits.reshape(len(batch), 3)
                targets = torch.zeros(len(batch), dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, targets) / args.gradient_accumulation
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * args.gradient_accumulation)
            micro_steps += 1
            if micro_steps % args.gradient_accumulation == 0 or start + args.batch_queries >= len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
    elapsed = time.perf_counter() - started
    validation_result = _evaluate(
        model,
        tokenizer,
        validation,
        candidates,
        max_length=args.max_length,
        device=device,
    )
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(args.checkpoint)
    checkpoint_files = sorted(path for path in args.checkpoint.iterdir() if path.is_file())
    checkpoint_manifest = {path.name: _sha(path) for path in checkpoint_files}

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["by_issuer"][VALIDATION_ISSUER]
    top1_gain = validation_result["top1_accuracy"] - baseline["top1_accuracy"]
    pairwise_gain = validation_result["pairwise_accuracy"] - baseline["pairwise_accuracy"]
    type_regressions = [
        kind
        for kind, metric in validation_result["pairwise_by_negative_type"].items()
        if metric["accuracy"] < baseline["pairwise_by_negative_type"][kind]["accuracy"]
    ]
    gate_passed = top1_gain >= 0.10 and pairwise_gain >= 0.05 and not type_regressions
    decision = "hard_negative_reranker_training_gate_passed" if gate_passed else "hard_negative_reranker_training_gain_insufficient"
    next_gate = "single_frozen_benchmark_shadow_transfer" if gate_passed else "stop_hard_negative_reranker_training"
    configuration = {
        "schema": "nf-opt-17/pre-registered-training-config/v1",
        "train_issuers": list(TRAIN_ISSUERS),
        "validation_issuer": VALIDATION_ISSUER,
        "train_query_count": len(train),
        "validation_query_count": len(validation),
        "objective": "three_way_positive_first_cross_entropy",
        "trainable_parameter_scope": "classification_head_only",
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable_parameters),
        "epochs": args.epochs,
        "batch_queries": args.batch_queries,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "max_length": args.max_length,
        "seed": SEED,
        "fp16": args.fp16,
        "hyperparameter_scan_performed": False,
        "frozen_benchmark_read_allowed": False,
    }
    report = {
        "schema": "nf-opt-17/issuer-disjoint-training-result/v1",
        "baseline_validation": baseline,
        "trained_validation": validation_result,
        "top1_gain": top1_gain,
        "pairwise_gain": pairwise_gain,
        "negative_type_regressions": type_regressions,
        "training_mean_loss": sum(losses) / len(losses),
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": elapsed,
        "checkpoint_runtime_only": True,
        "checkpoint_manifest": checkpoint_manifest,
    }
    acceptance = {
        "schema": "nf-opt-17/gate-f/acceptance/v1",
        "annotation_input_sha256": _sha(args.annotations),
        "candidate_input_sha256": _sha(args.candidates),
        "issuer_overlap_count": 0,
        "frozen_benchmark_question_or_label_reads": 0,
        "expected_field_reads": 0,
        "hyperparameter_scan_count": 0,
        "training_run_count": 1,
        "optimizer_steps": optimizer_steps,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "training_gate_passed": gate_passed,
        "decision": decision,
        "next_gate": next_gate,
    }
    _write(args.out_dir / "pre-registered-training-configuration.json", configuration)
    _write(args.out_dir / "issuer-disjoint-training-result.json", report)
    _write(args.out_dir / "runtime-checkpoint-manifest.json", {"runtime_only": True, "files": checkpoint_manifest})
    _write(args.out_dir / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    _write(args.out_dir / "nf-opt-17-gate-f-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-queries", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
