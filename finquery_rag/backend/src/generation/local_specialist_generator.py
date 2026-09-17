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
import sys
import time
from typing import Any

import torch

from src.generation.specialist_prompt import render_specialist_prompt


def _resolve_nanochat_repo() -> Path:
    """Resolve the NanoChat source root without coupling V2 to one host path.

    ``NANOCHAT_REPO`` is the explicit deployment override; the source tree
    containing this module is the safe local default. No unrelated Python
    environment is appended to the interpreter path.
    """
    configured = os.getenv("NANOCHAT_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


NANOCHAT_REPO = _resolve_nanochat_repo()
if str(NANOCHAT_REPO) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_REPO))

# Keep the backend interpreter isolated.  Appending an unrelated Anaconda
# site-packages directory here made imports process-global and allowed binary
# extensions built against NumPy 1.x (for example numexpr/bottleneck) to
# shadow the backend venv's NumPy 2.x dependencies.  The canonical backend
# environment already contains the NanoChat runtime dependencies; if it does
# not, startup should fail fast instead of silently mixing environments.

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
EXPECTED_VIEW_SHA = "943decf288dffb99ffa6f196abc44e0a5bdb226350cede40e0a160c4bd61f6e4"


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
        """Render prompt adhering strictly to FinancialGenerationViewV1.

        H2A-3B0.  The renderer itself now lives in
        ``src/generation/specialist_prompt.py``, so that what a model is asked
        can be asserted on without importing ``torch`` -- this module is
        excluded from test collection wherever torch is absent, and a prompt
        assertion that silently never runs is worse than no assertion.

        This method is kept, and delegates, because callers and the frozen
        generation contract reach the prompt through it.  Its behaviour is the
        renderer's; the one intentional change is that a missing page is no
        longer rendered as page 1.
        """

        return render_specialist_prompt(
            question,
            evidence_items,
            calculation_result,
        )

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

        rendered_input = self.render_prompt(
            question, evidence_items, calculation_result
        )

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

        new_tokens = gen_tokens[0][len(prompt_tokens) :]
        raw_output = self.tokenizer.decode(new_tokens)

        finish_reason = "length" if len(new_tokens) >= self.max_new_tokens else "stop"

        return {
            "raw_output": raw_output,
            "latency_seconds": round(latency, 4),
            "tokens_generated": len(new_tokens),
            "finish_reason": finish_reason,
            "rendered_input_length": len(prompt_tokens),
            "checkpoint_sha256_prefix": self.checkpoint_sha256[:16]
            if self.checkpoint_sha256
            else "",
            "role": self.ROLE,
        }

    def get_health_status(self) -> dict[str, Any]:
        """Expose diagnostic health and telemetry metadata."""
        vram_alloc = 0.0
        vram_reserved = 0.0
        if torch.cuda.is_available() and self.device.type == "cuda":
            vram_alloc = round(
                torch.cuda.memory_allocated(self.device) / (1024 * 1024), 2
            )
            vram_reserved = round(
                torch.cuda.memory_reserved(self.device) / (1024 * 1024), 2
            )

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
