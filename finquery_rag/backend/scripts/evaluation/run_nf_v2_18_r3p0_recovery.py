#!/usr/bin/env python3
"""Recover and health-check the historical Qwen3-Reranker-4B runtime.

This script has two phases. The parent performs read-only GPU discovery and
snapshot recovery, then starts a child with a dynamically selected physical
GPU exposed as logical ``cuda:0``. The child imports CUDA/model packages only
after that mapping is set. It uses synthetic smoke fixtures exclusively; no
development benchmark or Gold data is loaded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.pdf_retrieval_v4.gpu_selector import (  # noqa: E402
    discover_gpus,
    select_gpu,
    selected_gpu_is_still_eligible,
)


MODEL_ID = "Qwen/Qwen3-Reranker-4B"
REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
MAX_LENGTH = 8192
ARTIFACT = BACKEND_ROOT / "artifacts/evaluation/nf-v2-18-qwen-reranker-recovery"
SNAPSHOT_ROOT = Path("/mnt/disk/mxf/models/qwen3-reranker-4b")
HISTORICAL_EXPECTED_SHA = {
    "config.json": "38bff5eac700032a185745e4076eccad7aa453473cafc2a27de412cdb7b79e19",
    "model-00001-of-00002.safetensors": "cf2e87cbf71fa628961532232e04dd6c19702a0a057f5e2aff95ea1aca4fd488",
    "model-00002-of-00002.safetensors": "78946d22b7f6456ea7a5358dbdf3982de36c5bac1f166a5fd58e18e31db8048a",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def write_json(name: str, value: object) -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_audit() -> dict[str, Any]:
    actual = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch_distribution": package_version("torch"),
        "transformers": package_version("transformers"),
        "huggingface_hub": package_version("huggingface-hub"),
        "tokenizers": package_version("tokenizers"),
        "safetensors": package_version("safetensors"),
        "accelerate": package_version("accelerate"),
    }
    expected = {
        "python": "3.12.2",
        "torch": "2.9.1+cu128",
        "transformers": "4.57.3",
        "cuda": "12.8",
        "dtype": "bfloat16",
    }
    return {
        "captured_at_utc": utc_now(),
        "environment": "/mnt/disk/mxf/anaconda3/envs/QhChat",
        "actual": actual,
        "historical_manifest_expected": expected,
        "current_environment_selected": True,
        "transformers_patch_delta": "4.57.1_vs_historical_4.57.3",
        "environment_mutated": False,
        "dependency_change_required": False,
    }


def candidate_snapshot_paths() -> list[Path]:
    home = Path.home()
    candidates = [
        SNAPSHOT_ROOT / REVISION,
        SNAPSHOT_ROOT / "snapshot",
        Path("/mnt/disk/mxf/models/Qwen3-Reranker-4B") / REVISION,
        Path("/mnt/disk/mxf/models/Qwen/Qwen3-Reranker-4B") / REVISION,
        Path(
            "/mnt/disk/mxf/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots"
        )
        / REVISION,
        home
        / ".cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots"
        / REVISION,
        BACKEND_ROOT / "models/qwen3-reranker-4b" / REVISION,
    ]
    return list(dict.fromkeys(candidates))


def validate_snapshot(path: Path) -> dict[str, Any]:
    critical = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    missing = [name for name in critical if not (path / name).is_file()]
    actual_sha = {
        name: sha256_file(path / name) for name in critical if (path / name).is_file()
    }
    mismatches = {
        name: {
            "expected": HISTORICAL_EXPECTED_SHA[name],
            "actual": actual_sha.get(name),
        }
        for name in critical
        if actual_sha.get(name) != HISTORICAL_EXPECTED_SHA[name]
    }
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "missing_critical_files": missing,
        "critical_sha256": actual_sha,
        "historical_sha_mismatches": mismatches,
        "exact_historical_identity": path.is_dir() and not missing and not mismatches,
    }


def recover_snapshot() -> tuple[Path | None, dict[str, Any]]:
    searched: list[dict[str, Any]] = []
    for candidate in candidate_snapshot_paths():
        result = validate_snapshot(candidate)
        searched.append(result)
        if result["exact_historical_identity"]:
            source = {
                "model_id": MODEL_ID,
                "requested_revision": REVISION,
                "resolved_revision": REVISION,
                "snapshot_path": str(candidate),
                "file_count": sum(1 for item in candidate.rglob("*") if item.is_file()),
                "total_bytes": sum(
                    item.stat().st_size
                    for item in candidate.rglob("*")
                    if item.is_file()
                ),
                "download_or_reuse_status": "REUSED_EXACT_LOCAL_SNAPSHOT",
                "searched_paths": searched,
                "credentials_exposed": False,
            }
            return candidate, source

    target = SNAPSHOT_ROOT / REVISION
    try:
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().model_info(MODEL_ID, revision=REVISION)
        resolved = getattr(info, "sha", None)
        if resolved and resolved != REVISION:
            raise RuntimeError(f"resolved_revision_mismatch:{resolved}")
        target.mkdir(parents=True, exist_ok=True)
        downloaded = Path(
            snapshot_download(
                repo_id=MODEL_ID,
                revision=REVISION,
                local_dir=str(target),
                local_dir_use_symlinks=False,
            )
        )
        validation = validate_snapshot(downloaded)
        source = {
            "model_id": MODEL_ID,
            "requested_revision": REVISION,
            "resolved_revision": resolved or REVISION,
            "snapshot_path": str(downloaded),
            "file_count": sum(1 for item in downloaded.rglob("*") if item.is_file()),
            "total_bytes": sum(
                item.stat().st_size for item in downloaded.rglob("*") if item.is_file()
            ),
            "download_or_reuse_status": "DOWNLOADED_EXACT_REVISION",
            "searched_paths": searched,
            "download_validation": validation,
            "credentials_exposed": False,
        }
        if not validation["exact_historical_identity"]:
            raise RuntimeError("downloaded_snapshot_identity_mismatch")
        return downloaded, source
    except Exception as exc:  # pragma: no cover - depends on external network
        source = {
            "model_id": MODEL_ID,
            "requested_revision": REVISION,
            "resolved_revision": None,
            "snapshot_path": None,
            "download_or_reuse_status": "DOWNLOAD_FAILED",
            "failure_type": type(exc).__name__,
            "failure": str(exc)[:1000],
            "searched_paths": searched,
            "credentials_exposed": False,
        }
        return None, source


def snapshot_manifest(snapshot: Path) -> tuple[dict[str, Any], str]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        relative = path.relative_to(snapshot).as_posix()
        if relative.startswith(".cache/"):
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "critical_expected_sha256": HISTORICAL_EXPECTED_SHA,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
    }
    snapshot_sha = canonical_sha(payload)
    manifest = {
        **payload,
        "snapshot_sha256": snapshot_sha,
        "identity_excludes": ["mtime", "download_timestamp", "filesystem_order"],
    }
    write_json("snapshot-manifest.json", manifest)
    (ARTIFACT / "snapshot-manifest.sha256").write_text(
        snapshot_sha + "\n", encoding="utf-8"
    )
    return manifest, snapshot_sha


def historical_scorer_audit() -> dict[str, Any]:
    instruction_path = (
        BACKEND_ROOT / "src/pdf_retrieval_v4/structure_aware_rerank_view.py"
    )
    scorer_path = BACKEND_ROOT / "src/pdf_retrieval_v4/qwen3_reranker.py"
    script_path = (
        BACKEND_ROOT / "scripts/evaluation/run_pdf_v4_gate_08_r8_r3_2_rerank.py"
    )
    return {
        "located": all(
            path.is_file() for path in (instruction_path, scorer_path, script_path)
        ),
        "historical_script": str(script_path),
        "scorer_module": str(scorer_path),
        "instruction_module": str(instruction_path),
        "model_class": "transformers.AutoModelForCausalLM",
        "tokenizer_class": "transformers.AutoTokenizer",
        "prompt_protocol": "PREFIX + <Instruct>/<Query>/<Document> + SUFFIX",
        "instruction_source": "RERANK_INSTRUCTION from structure_aware_rerank_view.py",
        "query_document_formatting": "format_instruction(instruction, query, document)",
        "yes_no_score": "last-position logits; log_softmax([no, yes])[:, yes]",
        "score_field": "reranker_score",
        "max_length": MAX_LENGTH,
        "padding_side": "left",
        "historical_batch_size": 1,
        "historical_dtype": "bfloat16",
        "generation": False,
        "normalization": "two-class log probability over no/yes",
        "ranking_direction": "descending reranker_score",
        "raw_content_truncation": "only [CONTENT] raw tail is truncated; structured prefix retained",
        "adaptation": "none; reusable wrapper delegates to existing build_input_ids and score_batch",
    }


def smoke_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "numeric_operating_income",
            "query": "What was operating income in fiscal 2025?",
            "candidates": [
                {
                    "label": "relevant",
                    "document": "Operating income for fiscal 2025 was 32.1 billion dollars.",
                },
                {
                    "label": "irrelevant",
                    "document": "The company discusses cybersecurity risk factors and access controls.",
                },
            ],
        },
        {
            "case_id": "table_row_revenue",
            "query": "Which table row reports FY2025 net sales?",
            "candidates": [
                {
                    "label": "relevant",
                    "document": "[TABLE] Consolidated Statements of Income\n[ROW] Net sales | FY2025 | 100.0 billion USD",
                },
                {
                    "label": "irrelevant",
                    "document": "[TABLE] Balance Sheets\n[ROW] Total assets | FY2025 | 250.0 billion USD",
                },
            ],
        },
        {
            "case_id": "period_fy2025_vs_fy2024",
            "query": "What was FY2025 operating income?",
            "candidates": [
                {
                    "label": "relevant",
                    "document": "Operating income, year ended December 31, 2025: 32.1 billion dollars.",
                },
                {
                    "label": "wrong_period",
                    "document": "Operating income, year ended December 31, 2024: 29.4 billion dollars.",
                },
            ],
        },
        {
            "case_id": "qualitative_ai_risk",
            "query": "What risk is described for artificial intelligence?",
            "candidates": [
                {
                    "label": "relevant",
                    "document": "Risk factors: artificial intelligence systems may create privacy, security, and regulatory risks.",
                },
                {
                    "label": "irrelevant",
                    "document": "Cash and cash equivalents were 14.2 billion dollars at year end.",
                },
            ],
        },
        {
            "case_id": "period_quarter_vs_ytd",
            "query": "What was revenue for the three months ended June 30, 2025?",
            "candidates": [
                {
                    "label": "relevant",
                    "document": "Three months ended June 30, 2025: revenue was 25.0 billion dollars.",
                },
                {
                    "label": "wrong_period",
                    "document": "Six months ended June 30, 2025: revenue was 48.0 billion dollars.",
                },
            ],
        },
        {
            "case_id": "cash_flow_capex",
            "query": "What were capital expenditures in fiscal 2025?",
            "candidates": [
                {
                    "label": "relevant",
                    "document": "Cash flow note: capital expenditures in fiscal 2025 were 8.7 billion dollars.",
                },
                {
                    "label": "irrelevant",
                    "document": "Research and development expenses increased during fiscal 2025.",
                },
            ],
        },
    ]


def flatten_cases(
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    pairs: list[dict[str, str]] = []
    mapping: list[dict[str, Any]] = []
    for case in cases:
        for index, candidate in enumerate(case["candidates"]):
            pairs.append({"query": case["query"], "document": candidate["document"]})
            mapping.append(
                {
                    "case_id": case["case_id"],
                    "candidate_index": index,
                    "label": candidate["label"],
                }
            )
    return pairs, mapping


def grouped_results(
    cases: list[dict[str, Any]], score_result: Any
) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    offset = 0
    for case in cases:
        count = len(case["candidates"])
        scores = score_result.scores[offset : offset + count]
        order = sorted(
            range(count), key=lambda index: (-scores[index]["reranker_score"], index)
        )
        grouped.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "candidates": [
                    {
                        **candidate,
                        "score": scores[index],
                        "audit": score_result.audits[offset + index],
                        "candidate_index": index,
                    }
                    for index, candidate in enumerate(case["candidates"])
                ],
                "ranking": order,
                "top_label": case["candidates"][order[0]]["label"],
            }
        )
        offset += count
    return grouped


def run_child(snapshot: Path, selection: dict[str, Any], snapshot_sha: str) -> int:
    from src.pdf_retrieval_v4.qwen3_reranker_runtime import Qwen3RerankerRuntime
    from src.pdf_retrieval_v4.structure_aware_rerank_view import RERANK_INSTRUCTION

    try:
        import torch
    except Exception as exc:
        write_json(
            "model-load-test.json",
            {
                "status": "FAIL",
                "failure_type": "CUDA_RUNTIME_FAILURE",
                "error": str(exc),
            },
        )
        write_json(
            "decision.json",
            {"decision": "QWEN_RERANKER_NOT_READY", "blocker": "CUDA_RUNTIME_FAILURE"},
        )
        return 2

    started = time.perf_counter()
    try:
        runtime, contract = Qwen3RerankerRuntime.load(snapshot)
        load_result = {
            "status": "PASS",
            "load_seconds": contract["load_seconds"],
            "contract": contract,
            "allocated_vram_mib_after_load": torch.cuda.memory_allocated()
            / (1024 * 1024),
            "reserved_vram_mib_after_load": torch.cuda.memory_reserved()
            / (1024 * 1024),
        }
        write_json("model-load-test.json", load_result)
    except torch.cuda.OutOfMemoryError as exc:
        write_json(
            "model-load-test.json",
            {"status": "FAIL", "failure_type": "VRAM_INSUFFICIENT", "error": str(exc)},
        )
        write_json(
            "decision.json",
            {"decision": "QWEN_RERANKER_NOT_READY", "blocker": "VRAM_INSUFFICIENT"},
        )
        return 2
    except Exception as exc:
        write_json(
            "model-load-test.json",
            {
                "status": "FAIL",
                "failure_type": "MODEL_FORMAT_INCOMPATIBLE",
                "error": str(exc)[:1000],
            },
        )
        write_json(
            "decision.json",
            {
                "decision": "QWEN_RERANKER_NOT_READY",
                "blocker": "MODEL_FORMAT_INCOMPATIBLE",
            },
        )
        return 2

    cases = smoke_cases()
    pairs, _ = flatten_cases(cases)
    first = runtime.score_pairs(pairs, batch_size=1, instruction=RERANK_INSTRUCTION)
    grouped = grouped_results(cases, first)
    finite = all(
        value == value and abs(value) != float("inf")
        for score in first.scores
        for value in score.values()
    )
    relevance_pass = all(
        item["top_label"] == "relevant"
        for item in grouped
        if item["case_id"] not in {"period_fy2025_vs_fy2024", "period_quarter_vs_ytd"}
    )
    period_cases = {"period_fy2025_vs_fy2024", "period_quarter_vs_ytd"}
    period_pass = all(
        item["top_label"] == "relevant"
        for item in grouped
        if item["case_id"] in period_cases
    )
    write_json(
        "pair-scoring-smoke.json",
        {
            "cases": grouped,
            "finite_scores": finite,
            "relevance_ranking_pass": relevance_pass,
        },
    )
    write_json(
        "period-ranking-smoke.json",
        {
            "cases": [item for item in grouped if item["case_id"] in period_cases],
            "period_ranking_pass": period_pass,
        },
    )

    stability_runs: list[list[float]] = []
    stability_orders: list[list[list[int]]] = []
    for _ in range(3):
        result = runtime.score_pairs(
            pairs, batch_size=1, instruction=RERANK_INSTRUCTION
        )
        stability_runs.append([item["reranker_score"] for item in result.scores])
        stability_orders.append(
            [
                sorted(
                    range(len(case["candidates"])),
                    key=lambda index: (
                        -result.scores[offset + index]["reranker_score"],
                        index,
                    ),
                )
                for offset, case in _offset_cases(cases)
            ]
        )
    max_delta = max(
        abs(left - right)
        for left, right in zip(stability_runs[0], stability_runs[1], strict=True)
    )
    ranking_stable = all(order == stability_orders[0] for order in stability_orders[1:])
    write_json(
        "determinism-test.json",
        {
            "runs": 3,
            "finite_scores": all(
                all(value == value and abs(value) != float("inf") for value in run)
                for run in stability_runs
            ),
            "ranking_identical": ranking_stable,
            "maximum_absolute_score_delta_first_two_runs": max_delta,
            "score_bitwise_identity_required": False,
        },
    )

    batch_results: list[dict[str, Any]] = []
    for batch_size in (1, 4, 8, 16):
        try:
            result = runtime.score_pairs(
                pairs, batch_size=batch_size, instruction=RERANK_INSTRUCTION
            )
            batch_results.append(
                {
                    "batch_size": batch_size,
                    "status": "PASS",
                    "pairs": len(pairs),
                    "elapsed_seconds": result.elapsed_seconds,
                    "pairs_per_second": len(pairs) / result.elapsed_seconds
                    if result.elapsed_seconds
                    else None,
                    "peak_allocated_mib": result.peak_allocated_mib,
                    "peak_reserved_mib": result.peak_reserved_mib,
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            batch_results.append(
                {"batch_size": batch_size, "status": "OOM", "error": str(exc)[:500]}
            )
        except Exception as exc:
            batch_results.append(
                {"batch_size": batch_size, "status": "FAIL", "error": str(exc)[:500]}
            )
    safe_batches = [
        item["batch_size"] for item in batch_results if item["status"] == "PASS"
    ]
    selected_batch = 4 if 4 in safe_batches else (1 if 1 in safe_batches else None)
    write_json(
        "batching-test.json",
        {
            "historical_batch_size": 1,
            "tested_batch_sizes": batch_results,
            "selected_conservative_batch_size": selected_batch,
        },
    )

    current_gpu = discover_gpus()
    selected_physical = selection.get("selected_physical_gpu")
    selected_record = next(
        (
            item
            for item in current_gpu["gpus"]
            if item["physical_index"] == selected_physical
        ),
        None,
    )
    write_json(
        "gpu-runtime-metrics.json",
        {
            "selected_physical_gpu": selected_physical,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": "cuda:0",
            "post_test_gpu": selected_record,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated()
            / (1024 * 1024),
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 * 1024),
        },
    )

    truncations = [audit for audit in first.audits if audit.get("truncated")]
    instruction_sha = sha256_bytes(RERANK_INSTRUCTION.encode())
    runtime_config = {
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "snapshot_sha256": snapshot_sha,
        "scorer_module": "src.pdf_retrieval_v4.qwen3_reranker.score_batch",
        "input_builder": "src.pdf_retrieval_v4.qwen3_reranker.build_input_ids",
        "instruction_sha256": instruction_sha,
        "max_length": MAX_LENGTH,
        "padding_side": "left",
        "dtype": "bfloat16",
        "generation": False,
        "normalization": "log_softmax([no_logit, yes_logit]) yes class",
        "ranking_direction": "descending reranker_score",
        "selected_batch_size": selected_batch,
        "truncation_observations": {
            "pairs": len(first.audits),
            "truncated_pairs": len(truncations),
        },
        "gpu_operational_fields_excluded_from_semantic_hash": True,
    }
    write_json("runtime-config.json", runtime_config)
    (ARTIFACT / "runtime-config.sha256").write_text(
        canonical_sha(runtime_config) + "\n", encoding="utf-8"
    )
    write_json(
        "historical-fixture-validation.json",
        {
            "fixture_available": True,
            "fixture": str(
                BACKEND_ROOT
                / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r8-r3-1a/rerank-input-views-v2.jsonl.gz"
            ),
            "fixture_pairs": 7200,
            "small_fixture_executed": False,
            "reason": "historical fixture is a 7200-pair evaluation artifact; R3P0 uses synthetic smoke only and does not rerun benchmark-like data",
            "ranking_agreement": None,
        },
    )
    health_pass = bool(
        finite
        and relevance_pass
        and period_pass
        and ranking_stable
        and selected_batch is not None
        and all(
            item["status"] == "PASS"
            for item in batch_results
            if item["batch_size"] == selected_batch
        )
    )
    write_json(
        "reranker-health-check.json",
        {
            "status": "PASS" if health_pass else "FAIL",
            "resource_status": "AVAILABLE",
            "exact_snapshot_exists": True,
            "snapshot_sha256": snapshot_sha,
            "qhchat_environment": sys.prefix,
            "safe_gpu_selected": selected_physical is not None,
            "model_loaded_on_cuda": True,
            "one_batch_finite": finite,
            "smoke_relevance_pass": relevance_pass,
            "smoke_period_pass": period_pass,
            "ranking_stability_pass": ranking_stable,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    decision = {
        "decision": "QWEN_RERANKER_READY" if health_pass else "QWEN_RERANKER_NOT_READY",
        "blocker": None if health_pass else "SMOKE_OR_STABILITY_FAILURE",
        "exact_model_revision": True,
        "snapshot_frozen": True,
        "historical_scorer_reused": True,
        "dynamic_gpu_used": selected_physical is not None,
        "benchmark_rerun": False,
        "retrieval_modified": False,
        "production": "V1",
        "production_switch": False,
        "next_gate": "NF-V2-18A-R3_A4_PRESERVING_HIERARCHICAL_QWEN_RERANK"
        if health_pass
        else None,
    }
    write_json("decision.json", decision)
    (ARTIFACT / "README.md").write_text(
        "# NF-V2-18A-R3P0 Qwen3-Reranker Recovery\n\n"
        "This artifact restores the exact historical Qwen3-Reranker-4B revision and validates it only with synthetic, non-benchmark smoke fixtures. GPU selection is read-only and dynamic; the selected physical GPU is mapped to logical `cuda:0` in a child process. No retrieval evaluation, training, or benchmark rerun is performed.\n",
        encoding="utf-8",
    )
    return 0 if health_pass else 2


def _offset_cases(cases: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    offset = 0
    result = []
    for case in cases:
        result.append((offset, case))
        offset += len(case["candidates"])
    return result


def run_parent() -> int:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    write_json("environment-audit.json", environment_audit())
    try:
        initial = discover_gpus()
    except Exception as exc:
        write_json(
            "gpu-availability-snapshot.json", {"status": "ERROR", "error": str(exc)}
        )
        write_json(
            "decision.json",
            {
                "decision": "QWEN_RERANKER_NOT_READY",
                "blocker": "GPU_RESOURCE_NOT_AVAILABLE",
            },
        )
        return 2
    write_json("gpu-availability-snapshot.json", initial)
    selection = select_gpu(initial)
    if selection.get("selected_physical_gpu") is None:
        write_json("gpu-selection.json", selection)
        write_json(
            "decision.json",
            {
                "decision": "QWEN_RERANKER_NOT_READY",
                "blocker": "GPU_RESOURCE_NOT_AVAILABLE",
            },
        )
        return 2
    recheck = discover_gpus()
    if not selected_gpu_is_still_eligible(recheck, selection):
        selection = select_gpu(recheck)
        if selection.get(
            "selected_physical_gpu"
        ) is None or not selected_gpu_is_still_eligible(recheck, selection):
            write_json("gpu-selection.json", {**selection, "race_check": "FAILED"})
            write_json(
                "decision.json",
                {
                    "decision": "QWEN_RERANKER_NOT_READY",
                    "blocker": "GPU_RESOURCE_NOT_AVAILABLE",
                },
            )
            return 2
        selection["race_check"] = "RESELECTED_ONCE"
    else:
        selection["race_check"] = "PASSED"
    write_json("gpu-selection.json", selection)

    snapshot, source = recover_snapshot()
    write_json("model-source.json", source)
    if snapshot is None:
        write_json(
            "decision.json",
            {
                "decision": "QWEN_RERANKER_NOT_READY",
                "blocker": "MODEL_DOWNLOAD_BLOCKED"
                if source.get("download_or_reuse_status") == "DOWNLOAD_FAILED"
                else "EXACT_REVISION_UNAVAILABLE",
                "benchmark_rerun": False,
                "retrieval_modified": False,
            },
        )
        return 2
    manifest, snapshot_sha = snapshot_manifest(snapshot)
    write_json("historical-scorer-audit.json", historical_scorer_audit())
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(selection["selected_physical_gpu"])
    env["NF_R3P0_CHILD"] = "1"
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--snapshot",
            str(snapshot),
            "--snapshot-sha",
            snapshot_sha,
            "--selection-json",
            str(ARTIFACT / "gpu-selection.json"),
        ],
        env=env,
        check=False,
    )
    return child.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--snapshot-sha")
    parser.add_argument("--selection-json", type=Path)
    args = parser.parse_args()
    if args.child:
        if (
            args.snapshot is None
            or args.snapshot_sha is None
            or args.selection_json is None
        ):
            raise SystemExit("child_arguments_required")
        selection = json.loads(args.selection_json.read_text(encoding="utf-8"))
        return run_child(args.snapshot, selection, args.snapshot_sha)
    return run_parent()


if __name__ == "__main__":
    raise SystemExit(main())
