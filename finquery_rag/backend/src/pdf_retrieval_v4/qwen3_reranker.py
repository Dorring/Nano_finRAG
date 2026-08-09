"""Frozen Qwen3 reranker configuration and official yes/no score protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Qwen3RerankerConfig:
    model_id: str = "Qwen/Qwen3-Reranker-0.6B"
    revision: str = ""
    max_length: int = 8192
    batch_size: int = 8
    dtype: str = "bfloat16"
    padding_side: str = "left"
    generation: bool = False


SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    "provided. Note that the answer can only be yes or no."
)

PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_instruction(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def build_input_ids(
    tokenizer: Any, instruction: str, query: str, document: str, max_length: int
) -> tuple[list[int], dict[str, Any]]:
    """Keep protocol/query/structured evidence and truncate only raw content head."""
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
    marker = "\n[CONTENT]\n"
    structured, raw = document.split(marker, 1) if marker in document else (document, "")
    fixed_text = format_instruction(instruction, query, structured + marker)
    fixed_tokens = tokenizer.encode(fixed_text, add_special_tokens=False)
    raw_tokens = tokenizer.encode(raw, add_special_tokens=False)
    original_count = len(prefix_tokens) + len(fixed_tokens) + len(raw_tokens) + len(suffix_tokens)
    available = max_length - len(prefix_tokens) - len(fixed_tokens) - len(suffix_tokens)
    if available < 0:
        raise ValueError("fixed_rerank_view_exceeds_max_length")
    final_raw = raw_tokens[:available]
    ids = prefix_tokens + fixed_tokens + final_raw + suffix_tokens
    return ids, {
        "original_token_count": original_count,
        "final_token_count": len(ids),
        "truncated": len(final_raw) < len(raw_tokens),
        "raw_token_count": len(raw_tokens),
        "retained_raw_token_count": len(final_raw),
    }


@torch.no_grad()
def score_batch(model: Any, tokenizer: Any, input_ids: list[list[int]]) -> list[dict[str, float]]:
    inputs = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    logits = model(**inputs).logits[:, -1, :]
    yes = logits[:, tokenizer.convert_tokens_to_ids("yes")].float()
    no = logits[:, tokenizer.convert_tokens_to_ids("no")].float()
    log_probability = torch.log_softmax(torch.stack([no, yes], dim=1), dim=1)[:, 1]
    return [
        {"yes_logit": float(y), "no_logit": float(n), "reranker_score": float(score)}
        for y, n, score in zip(yes.cpu(), no.cpu(), log_probability.cpu(), strict=True)
    ]
