#!/usr/bin/env python3
"""P1/P1.1: the real Step-156 checkpoint through the produced binding.

An isolated, single-process verification.  Run under Rule B -- the backend
service must be stopped first, so that exactly one PyTorch process holds a CUDA
context on this host.

**P1.1 changed what this script is allowed to build.**  P1 constructed its own
token counter, which proved the tokenizer was exact and proved nothing about the
wiring.  This version reads the counter off the binding the capability actually
built, so a green run means the *contract* carries the capability and not merely
that it can be constructed in a script.

What it establishes, in the order the brief asks for it:

  1. the checkpoint loads and really infers;
  2. the tokenizer matches the model -- vocab 65000, *and* the binding's own
     counter and the provider independently agree about the same string;
  3. `ModelProviderV1` is sufficient: the capability reaches the model through
     `ModelBindingV1` with no change to the Harness core;
  4. B3 context is fed correctly -- the prompt is byte-identical to the frozen
     `BASELINE_V2`, measured through the real binding rather than the recording
     double the baseline tests use;
  5. the binding declares a stable `model_id` and a real exact counter, and
     **no token bound is configured** -- measured, not enforced.

Nothing here writes to the corpus, the indexes or the sealed artifacts.

    python run_p1_real_specialist_binding.py --backend <path> --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_BACKEND = "/disk/qh/nano-finrag/finquery_rag/backend"
DEFAULT_REPO = "/disk/qh/nano-finrag"
DEFAULT_CHECKPOINT = (
    "/disk/qh/nano-finrag/models/grounded-specialist-156/model_000156.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "3bda9f032d7bfb29a3bdf7e0eeeee930a57a05e899e11e67e108483ca920894a"
)
EXPECTED_VOCAB = 65000
EXPECTED_MODEL_ID = "nano-finance-2.08b-step156"

#: The provider wraps a rendered prompt in four tokens before the model sees it
#: (`LocalSpecialistGenerator.generate`).  They are a *provider* fact about
#: presentation, not part of the payload a counter measures -- which is why the
#: counter excludes them and this constant exists only to reconcile the two
#: numbers.  If they ever disagree, this is the first place to look.
CHAT_FRAME_TOKENS = 4


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(args.backend).resolve()))
    sys.path.insert(0, str(Path(args.repo).resolve()))

    import torch

    print("=== environment ===")
    print(f"  python          : {sys.version.split()[0]}")
    print(f"  torch           : {torch.__version__}")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    print(f"  device count    : {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        print("  !! no CUDA device visible; refusing to run the real checkpoint")
        return 2

    # --- the tokenizer, before any model is loaded ----------------------------
    from nanochat.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    vocab = tokenizer.get_vocab_size()
    print()
    print("=== tokenizer (the checkpoint's own) ===")
    print(f"  vocab_size      : {vocab}")
    print(f"  expected        : {EXPECTED_VOCAB}")
    print(f"  match           : {vocab == EXPECTED_VOCAB}")
    if vocab != EXPECTED_VOCAB:
        print("  !! tokenizer does not match the checkpoint's model config")
        return 2

    # --- the checkpoint -------------------------------------------------------
    from src.generation.local_specialist_generator import LocalSpecialistGenerator

    checkpoint = Path(args.checkpoint)
    print()
    print("=== checkpoint ===")
    print(f"  path            : {checkpoint}")
    actual = _sha256_file(checkpoint)
    print(f"  sha256          : {actual}")
    print(f"  matches sealed  : {actual == EXPECTED_CHECKPOINT_SHA256}")

    specialist = LocalSpecialistGenerator(
        checkpoint_path=checkpoint,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    started = time.perf_counter()
    specialist.load()
    print(f"  load            : {time.perf_counter() - started:.2f}s on {specialist.device}")

    # --- the binding, built exactly the way production builds it -------------
    from src.runtime.trusted_v2_generation import TrustedV2GenerationCapability
    from tests.harness.b3_legacy_context_baseline import (
        BASELINE_V2,
        SCENARIOS,
        build_state,
    )

    capability = TrustedV2GenerationCapability(model_backend=specialist)
    binding = capability.binding
    assert binding is not None

    counter = binding.exact_token_counter
    print()
    print("=== the binding production builds ===")
    print(f"  provider        : {type(binding.provider).__name__}")
    print(f"  provider_id     : {binding.provider_id}")
    print(f"  renderer        : {binding.renderer_id}")
    print(f"  model_id        : {binding.model_id!r}")
    print(f"  model_id stable : {binding.model_id == EXPECTED_MODEL_ID}")
    print(f"  exact counter   : {None if counter is None else counter.counter_id!r}")
    print(f"  token bound     : {capability.context_compiler.budget.max_input_tokens!r}"
          "   (None = measured, not enforced)")

    # --- one real invocation per frozen scenario ------------------------------
    rows: list[dict] = []
    print()
    print("=== real invocations through the produced binding ===")
    header = (
        f"  {'scenario':<30} {'prompt==base':<13} {'in(tok)':<9} "
        f"{'out(tok)':<9} {'cross-check':<12} {'latency':<9} finish"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for scenario in sorted(SCENARIOS):
        state = build_state(scenario)
        started = time.perf_counter()
        result = capability.generate(state)
        elapsed = time.perf_counter() - started

        invocation = capability.invocation
        request = invocation.last_request
        response = invocation.last_response
        pack = capability.last_context_pack
        assert request is not None and response is not None and pack is not None

        prompt = request.prompt
        identical = prompt == BASELINE_V2[scenario]["prompt"]

        usage = dict(response.usage or {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        # The binding's counter and the provider must agree about the same
        # string.  The provider counts the framed prompt, so the frame is
        # removed here rather than added to the counter -- one of the two has to
        # be reconciled, and it is the one that knows about the frame.
        own_count = None if counter is None else counter.count(prompt)
        cross_check = (
            "n/a"
            if (own_count is None or prompt_tokens is None)
            else ("AGREE" if own_count == prompt_tokens - CHAT_FRAME_TOKENS else "DISAGREE")
        )

        rows.append(
            {
                "scenario": scenario,
                "route": result.route,
                "route_target": capability.last_decision.target.value,
                "prompt_matches_baseline": identical,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "evidence_selected": pack.budget.evidence_selected,
                "pack_token_counter_id": pack.budget.token_counter_id,
                "pack_max_input_tokens": pack.budget.max_input_tokens,
                "pack_selected_context_tokens": pack.budget.selected_context_tokens,
                "binding_counter_count_of_prompt": own_count,
                "provider_prompt_tokens": prompt_tokens,
                "provider_completion_tokens": completion_tokens,
                "cross_check": cross_check,
                "latency_seconds": round(elapsed, 3),
                "finish_status": response.finish_status,
                "candidate_answer": result.candidate_answer,
                "answer_sha256": hashlib.sha256(
                    result.candidate_answer.encode("utf-8")
                ).hexdigest(),
            }
        )

        def _n(value: int | None) -> str:
            return "-" if value is None else str(value)

        print(
            f"  {scenario:<30} {str(identical):<13} {_n(prompt_tokens):<9} "
            f"{_n(completion_tokens):<9} {cross_check:<12} {elapsed:>7.2f}s  "
            f"{response.finish_status}"
        )

    # --- the answers themselves ----------------------------------------------
    print()
    print("=== the model's answers (first 220 chars) ===")
    for row in rows:
        print()
        print(f"  [{row['scenario']}]  sha256={row['answer_sha256'][:16]}…")
        for line in row["candidate_answer"].strip().splitlines()[:6]:
            print(f"      {line[:200]}")

    # --- verdict --------------------------------------------------------------
    all_identical = all(row["prompt_matches_baseline"] for row in rows)
    all_agree = all(row["cross_check"] == "AGREE" for row in rows)
    all_inferred = all(row["provider_completion_tokens"] not in (None, 0) for row in rows)
    all_specialist = all(row["route_target"] == "LOCAL_SPECIALIST" for row in rows)
    no_bound = all(row["pack_max_input_tokens"] is None for row in rows)
    counter_recorded = all(
        row["pack_token_counter_id"] == (None if counter is None else counter.counter_id)
        for row in rows
    ) and all(row["pack_selected_context_tokens"] is not None for row in rows)

    print()
    print("=== verdict ===")
    print(f"  every scenario reached the specialist        : {all_specialist}")
    print(f"  every prompt byte-identical to BASELINE_V2   : {all_identical}")
    print(f"  counter and provider agree on the tokenizer  : {all_agree}")
    print(f"  every invocation produced real tokens        : {all_inferred}")
    print(f"  binding declares the stable model_id         : {binding.model_id == EXPECTED_MODEL_ID}")
    print(f"  binding declares an exact counter            : {counter is not None}")
    print(f"  packs record the counter, no bound configured: {counter_recorded and no_bound}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "binding": {
                        "provider": type(binding.provider).__name__,
                        "provider_id": binding.provider_id,
                        "renderer": binding.renderer_id,
                        "model_id": binding.model_id,
                        "counter_id": None if counter is None else counter.counter_id,
                        "max_input_tokens": capability.context_compiler.budget.max_input_tokens,
                    },
                    "scenarios": rows,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"  wrote {args.json_out}")

    clean = (
        all_specialist
        and all_identical
        and all_agree
        and all_inferred
        and binding.model_id == EXPECTED_MODEL_ID
        and counter is not None
        and counter_recorded
        and no_bound
    )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
