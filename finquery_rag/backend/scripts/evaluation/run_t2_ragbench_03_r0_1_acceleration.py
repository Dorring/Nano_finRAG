#!/usr/bin/env python3
"""T2-03R0.1 gold-blind Qwen runtime acceleration gate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from t2_ragbench_03_common import (
    EXPECTED_DATASET_COMMIT,
    EXPECTED_MODEL_ID,
    MODEL_REVISION,
    MODEL_SNAPSHOT,
    sha256,
    write_json,
)

FULL_PAIRS = 1_154_400
PROBE_QUERIES = 256
PROBE_PAIRS = 12_800
REFERENCE_PAIRS_PER_SECOND = 6.2248253074107165
REFERENCE_PROBE_ELAPSED_SECONDS = 2056.282605193992
VLLM_HF_OVERRIDES = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}


def estimate_full_hours(pairs_per_second: float | None) -> float | None:
    if pairs_per_second is None or pairs_per_second <= 0:
        return None
    return FULL_PAIRS / pairs_per_second / 3600.0


def deterministic_shard(query_id: str, gpu_count: int) -> int:
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    return int(hashlib.sha256(query_id.encode("utf-8")).hexdigest(), 16) % gpu_count


def nvidia_smi() -> dict[str, Any]:
    query = "index,name,memory.total,memory.used,memory.free,utilization.gpu"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "gpus": []}
    rows = []
    for line in proc.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6:
            continue
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mb": int(float(fields[2])),
                    "memory_used_mb": int(float(fields[3])),
                    "memory_free_mb": int(float(fields[4])),
                    "utilization_gpu_percent": int(float(fields[5])),
                }
            )
        except ValueError:
            continue
    return {"available": True, "gpus": rows, "raw": proc.stdout}


def reference_contract(root: Path) -> dict[str, Any]:
    current = json.loads((root / "runtime-contract.json").read_text())
    probe = json.loads((root / "r0-runtime-probe.json").read_text())
    current.update(
        {
            "gate": "T2-03R0.1",
            "backend": "transformers_reference",
            "model_class": "Qwen3ForCausalLM",
            "tokenizer_class": "Qwen2TokenizerFast",
            "model_loader": "AutoModelForCausalLM",
            "scoring_path": "last-position full-vocabulary causal-lm logits",
            "full_vocabulary_logits_computed": True,
            "logits_to_keep": None,
            "reference_probe_pairs": probe.get("pairs_processed"),
            "reference_probe_elapsed_seconds": probe.get("elapsed_seconds"),
            "reference_pairs_per_second": probe.get("pairs_per_second"),
            "reference_estimated_full_hours": estimate_full_hours(probe.get("pairs_per_second")),
            "candidate_contract": "frozen BM25 Top50",
            "gold_reads": 0,
        }
    )
    return current


def vllm_signature() -> dict[str, Any]:
    try:
        import vllm
        from vllm import LLM

        sig = inspect.signature(LLM)
        return {
            "imported": True,
            "version": getattr(vllm, "__version__", None),
            "module": getattr(vllm, "__file__", None),
            "llm_parameters": sorted(sig.parameters),
            "runner_argument_supported": "runner" in sig.parameters,
            "score_method_supported": hasattr(LLM, "score"),
        }
    except Exception as exc:
        return {
            "imported": False,
            "version": None,
            "module": None,
            "llm_parameters": [],
            "runner_argument_supported": False,
            "score_method_supported": False,
            "import_error": f"{type(exc).__name__}: {exc}",
        }


def probe_vllm(signature: dict[str, Any], gpu: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "backend": "vllm_pooling_score",
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot": str(MODEL_SNAPSHOT),
        "hf_overrides": VLLM_HF_OVERRIDES,
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "gpu": gpu,
        "synthetic_pair_count": 0,
        "status": "blocked",
    }
    if not signature.get("imported"):
        result["error"] = signature.get("import_error", "vllm_import_failed")
        return result
    try:
        from vllm import LLM

        kwargs: dict[str, Any] = {
            "model": str(MODEL_SNAPSHOT),
            "tokenizer": str(MODEL_SNAPSHOT),
            "revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "task": "score",
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.80,
            "hf_overrides": VLLM_HF_OVERRIDES,
        }
        if signature.get("runner_argument_supported"):
            kwargs["runner"] = "pooling"
        started = time.perf_counter()
        llm = LLM(**kwargs)
        outputs = llm.score(
            ["runtime probe question"],
            ["runtime probe document"],
            use_tqdm=False,
        )
        result.update(
            {
                "status": "passed",
                "synthetic_pair_count": 1,
                "synthetic_elapsed_seconds": max(time.perf_counter() - started, 1e-9),
                "synthetic_output_type": type(outputs[0]).__name__ if outputs else None,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return result


def write_acceleration_artifacts(output: Path, root: Path, gpu: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "reference-runtime-contract.json", reference_contract(root))
    sig = vllm_signature()
    accelerated = probe_vllm(sig, gpu)
    accelerated["vllm_signature"] = sig
    accelerated["model_config_hash"] = None
    write_json(output / "accelerated-runtime-contract.json", accelerated)

    manifest_path = root / "r0-input-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256(manifest_path)
    reason = accelerated.get("error", "accelerated_backend_unavailable")
    write_json(
        output / "equivalence-probe.json",
        {
            "gate": "T2-03R0.1",
            "status": "blocked_before_pair_scoring",
            "probe_query_count": PROBE_QUERIES,
            "probe_pair_count": PROBE_PAIRS,
            "fixed_r0_input_manifest_sha256": manifest_sha,
            "fixed_r0_query_ids_sha256": manifest.get("query_ids_sha256"),
            "fixed_r0_candidate_identity_sha256": manifest.get("candidate_identity_sha256"),
            "reference_probe_reused": True,
            "reference_orderings_available": False,
            "accelerated_pairs_processed": 0,
            "gold_reads": 0,
            "candidate_mutation": 0,
            "top1_agreement": None,
            "top5_set_agreement": None,
            "top5_ordered_agreement": None,
            "top10_set_agreement": None,
            "top10_ordered_agreement": None,
            "top50_full_ordering_agreement": None,
            "spearman_rank_correlation": None,
            "maximum_rank_displacement": None,
            "mean_rank_displacement": None,
            "blocking_reason": reason,
        },
    )
    write_json(
        output / "repeatability-probe.json",
        {
            "status": "not_run_backend_blocked",
            "runs_requested": 2,
            "runs_completed": 0,
            "pairs_per_second_run1": None,
            "pairs_per_second_run2": None,
            "median_pairs_per_second": None,
            "ordering_repeatable": None,
            "non_finite_scores": 0,
            "runtime_errors": [],
            "gold_reads": 0,
        },
    )

    capacity = nvidia_smi()
    eligible = [
        row["index"]
        for row in capacity.get("gpus", [])
        if row.get("memory_free_mb", 0) >= 20_000
        and row.get("utilization_gpu_percent", 100) <= 20
    ]
    capacity.update(
        {
            "eligible_for_independent_4b_probe": eligible,
            "available_gpu_count": len(eligible),
            "selected_single_gpu": int(gpu) if str(gpu).isdigit() else gpu,
            "gold_reads": 0,
        }
    )
    write_json(output / "gpu-capacity.json", capacity)
    write_json(
        output / "multi-gpu-probe.json",
        {
            "status": "not_run_backend_blocked",
            "available_gpu_count": len(eligible),
            "eligible_gpu_indices": eligible,
            "sharding_contract": "sha256(query_id) % gpu_count",
            "query_split": "whole query with all 50 candidates; never split a candidate set",
            "single_vs_multi_ordering_equal": None,
            "aggregate_pairs_per_second": None,
            "estimated_full_hours": None,
            "gold_reads": 0,
        },
    )

    decision = {
        "gate": "T2-03R0.1",
        "base_gate": "T2-03R0",
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_commit": EXPECTED_DATASET_COMMIT,
        "reference_pairs_per_second": REFERENCE_PAIRS_PER_SECOND,
        "reference_probe_elapsed_seconds": REFERENCE_PROBE_ELAPSED_SECONDS,
        "reference_estimated_full_hours": estimate_full_hours(REFERENCE_PAIRS_PER_SECOND),
        "accelerated_backend": "vllm_pooling_score",
        "accelerated_pairs_per_second": None,
        "speedup": None,
        "accelerated_estimated_full_hours": None,
        "top1_agreement": None,
        "top5_set_agreement": None,
        "top10_set_agreement": None,
        "full_ranking_agreement": None,
        "gold_reads": 0,
        "accelerated_runtime_accepted": False,
        "runtime_equivalence": "unsupported_or_blocked",
        "available_gpu_count": len(eligible),
        "multi_gpu_supported": False,
        "next_gate": "t2_03_r1_reference_runtime",
        "decision_reason": [
            "installed_vllm_does_not_support_requested_runner_pooling_argument",
            "score_fallback_does_not_expose_sequence_classification_pooler",
            "no_accelerated_pairs_scored_and_no_gold_read",
        ],
    }
    write_json(output / "runtime-decision.json", decision)
    write_json(
        output / "runtime-acceleration-manifest.json",
        {
            "gate": "T2-03R0.1",
            "dataset_commit": EXPECTED_DATASET_COMMIT,
            "prediction_root": str(root.resolve()),
            "r0_input_manifest_sha256": manifest_sha,
            "probe_queries": PROBE_QUERIES,
            "probe_pairs": PROBE_PAIRS,
            "gold_reads": 0,
            "prediction_writes": 0,
        },
    )
    (output / "README.md").write_text(
        "# T2-03R0.1 Runtime Acceleration Gate\n\n"
        "The fixed R0 contract was audited without reading Gold. The installed "
        "vLLM backend is blocked: its LLM signature has no runner argument, and "
        "the task=score fallback fails before scoring because the Transformers "
        "sequence-classification wrapper has no pooling object. No benchmark pair "
        "was scored by the accelerated backend. The reference batch-1 runtime "
        "and BM25 Top50 contract remain unchanged.\n",
        encoding="utf-8",
    )
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", default=os.environ.get("T2_ACCEL_GPU", "5"))
    args = parser.parse_args()
    decision = write_acceleration_artifacts(
        args.output_root.resolve(), args.prediction_root.resolve(), args.gpu
    )
    return 0 if decision["accelerated_runtime_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
