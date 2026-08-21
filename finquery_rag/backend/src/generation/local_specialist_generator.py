"""Local Financial Specialist Generator (NF-V2-21).

Integrates the frozen, fresh-holdout-sealed NanoFinance 2.08B Grounded
Specialist Generator (Step-156, checkpoint SHA:
3bda9f032d7bfb29a3bdf7e0eeeee930a57a05e899e11e67e108483ca920894a)
into the financial RAG runtime under the FinancialGenerationViewV1 semantic contract.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import torch

NANOCHAT_REPO = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat")
if str(NANOCHAT_REPO) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_REPO))

extra_site = "/mnt/disk/mxf/anaconda3/lib/python3.12/site-packages"
if extra_site not in sys.path:
    sys.path.append(extra_site)

try:
    from nanochat.checkpoint_manager import build_model
    from nanochat.engine import Engine
except ImportError:
    build_model = None
    Engine = None

EXPECTED_CHECKPOINT_PATH = Path(
    "/home/mxf/.cache/nanochat/chatsft_checkpoints/d24_grounded_specialist_v3_lr5e6/model_000156.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "3bda9f032d7bfb29a3bdf7e0eeeee930a57a05e899e11e67e108483ca920894a"
)
EXPECTED_VIEW_SHA = (
    "943decf288dffb99ffa6f196abc44e0a5bdb226350cede40e0a160c4bd61f6e4"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class LocalSpecialistUnavailableError(Exception):
    """Raised when the Local Financial Specialist cannot be loaded or verified."""
    pass


class LocalSpecialistGenerator:
    """Local Financial Specialist Generator Model Service.

    Rootless, single-tenant, deterministic generation service wrapping
    Step-156 checkpoint.
    """

    ROLE = "LOCAL_FINANCIAL_SPECIALIST_GENERATOR"
    CONTRACT_VERSION = "FinancialGenerationViewV1"

    def __init__(
        self,
        checkpoint_path: Path | str = EXPECTED_CHECKPOINT_PATH,
        device: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._device_str = device or self._detect_device()
        self.device = torch.device(self._device_str)

        self.model = None
        self.tokenizer = None
        self.engine = None
        self.checkpoint_sha256 = None
        self._model_loaded = False
        self._load_duration_seconds = 0.0

        self.bos_token_id = None
        self.user_start_id = None
        self.user_end_id = None
        self.assistant_start_id = None

    def _detect_device(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        # If CUDA_VISIBLE_DEVICES is set, default to cuda:0
        return "cuda:0"

    def load(self) -> None:
        """Load and verify the Step-156 specialist checkpoint fail-fast."""
        if not self.checkpoint_path.exists():
            raise LocalSpecialistUnavailableError(
                f"Checkpoint file not found at {self.checkpoint_path}"
            )

        # 1. Verify Checkpoint SHA256
        actual_sha = sha256_file(self.checkpoint_path)
        if actual_sha != EXPECTED_CHECKPOINT_SHA256:
            raise LocalSpecialistUnavailableError(
                f"Checkpoint SHA256 mismatch! Expected {EXPECTED_CHECKPOINT_SHA256}, got {actual_sha}"
            )
        self.checkpoint_sha256 = actual_sha

        # 2. Build Model & Tokenizer
        if build_model is None or Engine is None:
            raise LocalSpecialistUnavailableError(
                "nanochat package not importable. Cannot build model."
            )

        t0 = time.perf_counter()
        ckpt_dir = str(self.checkpoint_path.parent)
        self.model, self.tokenizer, _ = build_model(
            ckpt_dir, 156, self.device, phase="eval"
        )
        self.model.eval()
        self.engine = Engine(self.model, self.tokenizer)
        self._load_duration_seconds = time.perf_counter() - t0

        # Cache special token IDs
        self.bos_token_id = self.tokenizer.get_bos_token_id()
        self.user_start_id = self.tokenizer.encode_special("<|user_start|>")
        self.user_end_id = self.tokenizer.encode_special("<|user_end|>")
        self.assistant_start_id = self.tokenizer.encode_special("<|assistant_start|>")

        self._model_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    def render_prompt(
        self,
        question: str,
        evidence_items: list[dict[str, Any]],
        calculation_result: dict[str, Any] | None = None,
    ) -> str:
        """Render prompt adhering strictly to FinancialGenerationViewV1."""
        lines = [f"[QUESTION]\n{question.strip()}\n", "[VERIFIED EVIDENCE]\n"]

        for i, ev in enumerate(evidence_items, start=1):
            cite_id = ev.get("citation_id", f"E{i}")
            if not re.match(r"^E\d+$", cite_id):
                cite_id = f"E{i}"

            metric = ev.get("metric") or ev.get("normalized_metric") or "Metric"
            period = ev.get("period") or "Period"
            value = str(ev.get("value", "")).strip()
            unit = ev.get("unit") or "not specified"
            currency = ev.get("currency") or "not specified"
            scale = ev.get("scale") or "1"
            scope = ev.get("scope") or metric
            source_doc = ev.get("document_id") or "filing"
            page = ev.get("page") or 1

            lines.append(f"[{cite_id}]")
            lines.append(f"Metric: {metric}")
            lines.append(f"Period: {period}")
            lines.append(f"Scope: {scope}")
            lines.append(f"Value: {value}")
            lines.append(f"Unit: {unit}")
            lines.append(f"Currency: {currency}")
            lines.append(f"Scale: {scale}")
            lines.append(f"Source: {source_doc}:{page}")

            if "source_text" in ev and ev["source_text"]:
                lines.append(f"Evidence: {ev['source_text']}")
            lines.append("")

        if calculation_result:
            c1_val = str(calculation_result.get("value", "")).strip()
            c1_unit = calculation_result.get("unit", "")
            c1_op = calculation_result.get("operation", "calculated_metric")
            lines.append("[VERIFIED CALCULATION]\n")
            lines.append("[C1]")
            lines.append(f"Operation: {c1_op}")
            lines.append(f"Value: {c1_val} {c1_unit}".strip())
            lines.append("")

        lines.append("[ANSWER RULES]")
        lines.append("1. Use only the verified evidence and calculation above.")
        lines.append("2. Do not introduce outside financial knowledge.")
        lines.append("3. Preserve supplied numbers, periods, units, currencies and scales exactly.")
        lines.append("4. Do not recalculate canonical calculation results.")
        lines.append("5. Cite factual claims using the supplied [E#] / [C#] IDs.")
        lines.append("6. If required evidence is missing, explicitly state that the provided evidence is insufficient.")
        lines.append("7. Answer concisely.")

        return "\n".join(lines)

    def generate(
        self,
        question: str,
        evidence_items: list[dict[str, Any]],
        calculation_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate response using greedy evaluation decoding."""
        if not self._model_loaded:
            raise LocalSpecialistUnavailableError(
                "LocalSpecialistGenerator is not loaded. Call load() first."
            )

        rendered_input = self.render_prompt(question, evidence_items, calculation_result)

        prompt_tokens = (
            [self.bos_token_id, self.user_start_id]
            + self.tokenizer.encode(rendered_input)
            + [self.user_end_id, self.assistant_start_id]
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            gen_tokens, _ = self.engine.generate_batch(
                tokens=prompt_tokens,
                num_samples=1,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        latency = time.perf_counter() - t0

        new_tokens = gen_tokens[0][len(prompt_tokens):]
        raw_output = self.tokenizer.decode(new_tokens)

        finish_reason = "length" if len(new_tokens) >= self.max_new_tokens else "stop"

        return {
            "raw_output": raw_output,
            "latency_seconds": round(latency, 4),
            "tokens_generated": len(new_tokens),
            "finish_reason": finish_reason,
            "rendered_input_length": len(prompt_tokens),
            "checkpoint_sha256_prefix": self.checkpoint_sha256[:16] if self.checkpoint_sha256 else "",
            "role": self.ROLE,
        }

    def get_health_status(self) -> dict[str, Any]:
        """Expose diagnostic health and telemetry metadata."""
        vram_alloc = 0.0
        vram_reserved = 0.0
        if torch.cuda.is_available() and self.device.type == "cuda":
            vram_alloc = round(torch.cuda.memory_allocated(self.device) / (1024 * 1024), 2)
            vram_reserved = round(torch.cuda.memory_reserved(self.device) / (1024 * 1024), 2)

        return {
            "model_loaded": self._model_loaded,
            "role": self.ROLE,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": str(self.device),
            "precision": "bfloat16" if torch.cuda.is_available() else "float32",
            "generation_contract_version": self.CONTRACT_VERSION,
            "contract_view_sha256": EXPECTED_VIEW_SHA,
            "load_duration_seconds": round(self._load_duration_seconds, 2),
            "vram_allocated_mb": vram_alloc,
            "vram_reserved_mb": vram_reserved,
        }
