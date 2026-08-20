"""Reusable local Qwen3-Reranker-4B runtime.

The scoring implementation deliberately delegates to the historical
``qwen3_reranker`` protocol in this repository.  This wrapper adds only
snapshot validation, ordered batching, and runtime diagnostics; it does not
change prompts, logits, normalization, or ranking direction.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-Reranker-4B"
MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
MAX_LENGTH = 8192
DTYPE_NAME = "bfloat16"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ScoreBatchResult:
    scores: list[dict[str, float]]
    audits: list[dict[str, Any]]
    elapsed_seconds: float
    batch_size: int
    peak_allocated_mib: float
    peak_reserved_mib: float


class Qwen3RerankerRuntime:
    """Historical yes/no-logit scorer loaded from an immutable local snapshot."""

    def __init__(
        self,
        snapshot: Path,
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        transformers_version: str,
    ):
        self.snapshot = snapshot
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch_module
        self.transformers_version = transformers_version

    @classmethod
    def load(cls, snapshot: Path) -> tuple["Qwen3RerankerRuntime", dict[str, Any]]:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("cuda_unavailable_in_selected_runtime")
        required = [
            snapshot / "config.json",
            snapshot / "tokenizer.json",
            snapshot / "tokenizer_config.json",
        ]
        shards = sorted(snapshot.glob("*.safetensors"))
        missing = [str(path) for path in required if not path.is_file()]
        if not shards:
            missing.append("*.safetensors")
        if missing:
            raise RuntimeError(f"snapshot_required_files_missing:{','.join(missing)}")

        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), padding_side="left", local_files_only=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = (
            AutoModelForCausalLM.from_pretrained(
                str(snapshot),
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            .to("cuda:0")
            .eval()
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"model_loaded_on_non_cuda:{device}")
        contract = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "dtype": DTYPE_NAME,
            "max_length": MAX_LENGTH,
            "padding_side": "left",
            "generation": False,
            "yes_token_id": int(tokenizer.convert_tokens_to_ids("yes")),
            "no_token_id": int(tokenizer.convert_tokens_to_ids("no")),
            "pad_token_id": int(tokenizer.pad_token_id),
            "load_seconds": load_seconds,
        }
        return cls(
            snapshot, tokenizer, model, torch, transformers.__version__
        ), contract

    def score_pairs(
        self,
        pairs: list[dict[str, str]],
        *,
        batch_size: int,
        instruction: str,
    ) -> ScoreBatchResult:
        from src.pdf_retrieval_v4.qwen3_reranker import build_input_ids, score_batch

        prepared: list[tuple[list[int], dict[str, Any]]] = []
        for pair in pairs:
            ids, audit = build_input_ids(
                self.tokenizer,
                instruction,
                pair["query"],
                pair["document"],
                MAX_LENGTH,
            )
            prepared.append((ids, audit))

        self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        scores: list[dict[str, float]] = []
        audits: list[dict[str, Any]] = []
        for offset in range(0, len(prepared), batch_size):
            batch = prepared[offset : offset + batch_size]
            batch_scores = score_batch(
                self.model, self.tokenizer, [item[0] for item in batch]
            )
            scores.extend(batch_scores)
            audits.extend(item[1] for item in batch)
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return ScoreBatchResult(
            scores=scores,
            audits=audits,
            elapsed_seconds=elapsed,
            batch_size=batch_size,
            peak_allocated_mib=self.torch.cuda.max_memory_allocated() / (1024 * 1024),
            peak_reserved_mib=self.torch.cuda.max_memory_reserved() / (1024 * 1024),
        )
