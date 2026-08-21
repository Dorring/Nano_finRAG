#!/usr/bin/env python3
"""Evaluate external general baseline models (Qwen2.5-1.5B-Instruct, Qwen3-1.7B, Qwen3.5-2B)
on the exact same 200-sample financial evaluation dataset (finance_eval_small_val.jsonl).
"""

from __future__ import annotations

import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve()
BACKEND = HERE.parents[2]
REPO = BACKEND.parents[1]
ART = BACKEND / "artifacts/evaluation/nf-v2-19c-financial-model-capability"

VAL_SET_PATH = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finance-data-process/data/processed/sft/finance_eval_small_val.jsonl")

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
extra_site = "/mnt/disk/mxf/anaconda3/lib/python3.12/site-packages"
if extra_site not in sys.path:
    sys.path.append(extra_site)

from nanochat.finance_eval import evaluate_records, primary_metric  # noqa: E402


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonlines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonlines(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_thinking_trace(text: str) -> str:
    """Remove <think>...</think> thinking tags if present in model output."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def main():
    ART.mkdir(parents=True, exist_ok=True)
    val_records = read_jsonlines(VAL_SET_PATH)
    print(f"Loaded {len(val_records)} validation records from {VAL_SET_PATH}")

    external_models = [
        {
            "id": "Qwen2.5-1.5B-Instruct",
            "name": "Qwen/Qwen2.5-1.5B-Instruct",
            "snapshot": "/mnt/disk/mxf/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "params": "1.54B",
        },
        {
            "id": "Qwen3-1.7B",
            "name": "Qwen/Qwen3-1.7B",
            "snapshot": "/mnt/disk/mxf/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "params": "1.72B",
        },
        {
            "id": "Qwen3.5-2B",
            "name": "Qwen/Qwen3.5-2B",
            "snapshot": "/mnt/disk/mxf/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc",
            "params": "2.0B",
        },
    ]

    results = {}
    per_task_results = {}

    for info in external_models:
        mid = info["id"]
        mname = info["name"]
        msnap = info["snapshot"]
        pred_cache_file = ART / f"predictions_ext_{mid}.jsonl"

        print("\n==========================================")
        print(f"Loading {mname} ({msnap}) on cuda:3...")
        print("==========================================")

        tokenizer = AutoTokenizer.from_pretrained(msnap, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            msnap,
            torch_dtype=torch.bfloat16,
            device_map="cuda:3",
            trust_remote_code=True,
        )
        model.eval()

        scored_records = []
        print(f"Evaluating {mid} on {len(val_records)} validation samples...")
        for i, rec in enumerate(val_records, 1):
            user_text = rec["messages"][0]["content"]
            messages = [{"role": "user", "content": user_text}]

            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:3")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_tokens = out[0][inputs["input_ids"].shape[1]:]
            ans_raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            ans_clean = clean_thinking_trace(ans_raw)

            scored_records.append({
                **rec,
                "prediction": ans_clean,
                "raw_prediction": ans_raw,
            })
            if i % 25 == 0 or i == len(val_records):
                print(f"  [{mid}] Generated {i}/{len(val_records)} samples...", flush=True)

        write_jsonlines(pred_cache_file, scored_records)

        # Run evaluation
        eval_report = evaluate_records(scored_records)
        macro_score = float(eval_report.get("macro_primary_score") or 0.0)
        task_scores = eval_report.get("tasks", {})

        print(f"-> {mid} Financial Macro: {macro_score * 100:.2f}%")
        results[mid] = {
            "model_id": mid,
            "model_name": mname,
            "parameters": info["params"],
            "macro_score": round(macro_score, 4),
            "macro_score_pct": round(macro_score * 100.0, 2),
            "eval_report": eval_report,
        }

        for tname, tinfo in task_scores.items():
            if tname not in per_task_results:
                per_task_results[tname] = {}
            pm = primary_metric(tname)
            per_task_results[tname][mid] = round(float(tinfo.get(pm, tinfo.get("exact_match", 0.0))), 4)

        # Release GPU 3 memory
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Combined Summary
    summary = {
        "benchmark": "finance_eval_small_val (200 samples, 8 financial NLP sources)",
        "internal_nano_finance_models": {
            "NanoFinance_General_Base_2.08B": "1.34%",
            "NanoFinance_Financial_CPT_2.08B": "1.21%",
            "NanoFinance_Financial_SFT_2.08B": "19.78%",
            "NanoFinance_Grounding_Alignment_2.08B": "18.36%",
        },
        "external_general_baseline_models": {
            k: f"{v['macro_score_pct']}%" for k, v in results.items()
        },
        "task_breakdown_external": per_task_results,
        "full_results": results,
    }
    write_json(ART / "external-baselines-comparison.json", summary)

    print("\n========================================================")
    print("ALL EXTERNAL BASELINES EVALUATED SUCCESSFULLY")
    print("========================================================")
    for k, v in results.items():
        print(f"  * {k} ({v['parameters']}): {v['macro_score_pct']}%")
    print("========================================================")


if __name__ == "__main__":
    main()
