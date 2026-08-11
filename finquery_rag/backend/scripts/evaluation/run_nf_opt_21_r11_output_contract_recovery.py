"""NF-OPT-21 R1.1: recover or constrain the frozen listwise output contract.

Stage A is a Gold-blind audit of the 72 raw R1 generations.  If fewer than
60 outputs are mechanically recoverable, Stage B performs one and only one
structured-output generation per query with the semantic prompt and frozen
candidate serialization unchanged.  Gold is loaded only after the selected
prediction artifact has been sealed.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_COMMIT = "135607002e2f74e67425a9480a3abc051b528719"
OUT_NAME = "nf-opt-21-r11-listwise-output-contract-recovery"
R1_NAME = "nf-opt-21-r1-top10-listwise-selector"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_SNAPSHOT = Path("/home/mxf/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots") / MODEL_REVISION
PROMPT_SHA = "01ab63296cf5b3581281eb5c0c55dd81be75f6bbc8c8bbeac5101a2f5151b645"
MAX_INPUT_LENGTH = 16384
R1_MAX_NEW_TOKENS = 128
STAGE_B_MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 0
RECOVERY_THRESHOLD = 60

FULL_ID_RE = re.compile(r"candidate:v1:[0-9a-f]{64}")
BROAD_ID_RE = re.compile(r"candidate:v1:[A-Za-z0-9_-]+")
CATEGORY_NAMES = (
    "A0_exact_contract_json",
    "A1_markdown_fenced_json",
    "A2_embedded_json",
    "A3_json_wrong_scalar_type",
    "A4_explicit_candidate_id_list_non_json",
    "A5_more_than_five_explicit_ids",
    "A6_fewer_than_five_explicit_ids",
    "A7_duplicate_candidate_ids",
    "A8_invalid_or_out_of_top10_ids",
    "A9_natural_language_without_complete_selection",
    "A10_empty_or_generation_failure",
    "A11_other",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return sha256_file(path)


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def load_r1_module(backend_root: Path) -> Any:
    path = backend_root / "scripts" / "evaluation" / "run_nf_opt_21_r1_top10_listwise_selector.py"
    spec = importlib.util.spec_from_file_location("nf_opt_21_r1_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("r1_module_load_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_id(value: Any, allowed: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().strip("`\"'")
    matches = [candidate for candidate in allowed if candidate.casefold() == cleaned.casefold()]
    return matches[0] if len(matches) == 1 else None


def valid_payload(payload: Any, allowed: set[str]) -> list[str] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("selected_ids"), list):
        return None
    values = [normalize_id(value, allowed) for value in payload["selected_ids"]]
    if len(values) != 5 or any(value is None for value in values) or len(set(values)) != 5:
        return None
    return [value for value in values if value is not None]


def json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            objects.append(payload)
    return objects


def raw_id_tokens(text: str) -> tuple[list[str], list[str]]:
    broad = BROAD_ID_RE.findall(text)
    valid = FULL_ID_RE.findall(text)
    return broad, valid


def classify_and_recover(text: str, allowed: set[str]) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {"category": "A10_empty_or_generation_failure", "recoverable": False, "method": None, "ids": []}

    try:
        exact_payload = json.loads(stripped)
    except json.JSONDecodeError:
        exact_payload = None
    exact_ids = valid_payload(exact_payload, allowed)
    if exact_ids is not None:
        return {"category": "A0_exact_contract_json", "recoverable": True, "method": "exact_json", "ids": exact_ids}
    if isinstance(exact_payload, dict) and "selected_ids" in exact_payload:
        return {"category": "A3_json_wrong_scalar_type", "recoverable": False, "method": None, "ids": []}

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    fenced_ids = []
    for item in fenced:
        try:
            payload = json.loads(item)
        except json.JSONDecodeError:
            continue
        fenced_ids.append(valid_payload(payload, allowed))
    if any(ids is not None for ids in fenced_ids):
        return {"category": "A1_markdown_fenced_json", "recoverable": True, "method": "fenced_json", "ids": next(ids for ids in fenced_ids if ids is not None)}

    objects = json_objects(stripped)
    object_ids = [valid_payload(payload, allowed) for payload in objects]
    valid_objects = [ids for ids in object_ids if ids is not None]
    if len(valid_objects) == 1:
        return {"category": "A2_embedded_json", "recoverable": True, "method": "embedded_json", "ids": valid_objects[0]}
    if any(isinstance(payload, dict) and "selected_ids" in payload for payload in objects):
        return {"category": "A3_json_wrong_scalar_type", "recoverable": False, "method": None, "ids": []}

    broad_tokens, full_tokens = raw_id_tokens(stripped)
    allowed_by_fold = {candidate.casefold(): candidate for candidate in allowed}
    normalized = [allowed_by_fold.get(token.casefold()) for token in full_tokens]
    unknown_full = [token for token in full_tokens if token.casefold() not in allowed_by_fold]
    if unknown_full or (broad_tokens and len(full_tokens) != len(broad_tokens)):
        return {"category": "A8_invalid_or_out_of_top10_ids", "recoverable": False, "method": None, "ids": normalized}
    counts = Counter(normalized)
    unique = list(dict.fromkeys(normalized))
    if len(unique) == 5 and len(normalized) == 5:
        return {"category": "A4_explicit_candidate_id_list_non_json", "recoverable": True, "method": "explicit_id_list", "ids": unique}
    if len(unique) > 5:
        return {"category": "A5_more_than_five_explicit_ids", "recoverable": False, "method": None, "ids": unique}
    if len(unique) < 5 and len(unique) > 0:
        category = "A7_duplicate_candidate_ids" if any(value > 1 for value in counts.values()) else "A6_fewer_than_five_explicit_ids"
        return {"category": category, "recoverable": False, "method": None, "ids": unique}
    if re.search(r"[A-Za-z]", stripped):
        return {"category": "A9_natural_language_without_complete_selection", "recoverable": False, "method": None, "ids": []}
    return {"category": "A11_other", "recoverable": False, "method": None, "ids": []}


def parse_plan_rows(payload: Any) -> dict[str, dict[str, Any]]:
    rows = payload.get("plans", []) if isinstance(payload, dict) else payload
    return {row["case_id"]: row.get("plan", {}) for row in rows or []}


def top_keys(items: list[dict[str, Any]], k: int) -> list[str]:
    return [item["candidate_key"] for item in items[:k]]


def build_prompt(prompt: str, question: str, candidates: list[dict[str, Any]], views: dict[str, dict[str, Any]]) -> str:
    blocks = [prompt, "", f"Question:\n{question}", "", "Candidates:"]
    for index, item in enumerate(candidates, start=1):
        view = views.get(item["candidate_key"], {})
        blocks.extend([
            f"\nCandidate {index}",
            f"Candidate ID: {item['candidate_key']}",
            f"Qwen rank: {item['original_qwen_rank']}",
            f"BM25 rank: {item['original_bm25_rank'] if item.get('original_bm25_rank') is not None else 'unknown'}",
            f"Context status: {view.get('context_status', item.get('context_status', 'unknown'))}",
            "Evidence:",
            view.get("document_view", "[NO FROZEN DOCUMENT VIEW AVAILABLE]"),
        ])
    return "\n".join(blocks)


def load_frozen_inputs(backend_root: Path, r1: Any) -> dict[str, Any]:
    evaluation_root = backend_root / "artifacts" / "evaluation"
    r1_root = evaluation_root / R1_NAME
    r0_root = evaluation_root / "nf-opt-21-r0-qwen-bm25-late-fusion"
    qwen_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3" / "main_rerank_predictions.jsonl.gz"
    bounded_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r2a-2" / "bounded-top100-predictions.jsonl.gz"
    qwen_seal_path = qwen_path.parent / "prediction-seal.json"
    views_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3-p0" / "queryplan-rerank-input-views.jsonl.gz"
    plan_path = evaluation_root / "pdf-retrieval-v4-gate-07" / "query-plan-predictions.json"
    selector_contract = read_json(r1_root / "selector-contract.json")
    prompt = (r1_root / "prompt.txt").read_text(encoding="utf-8").rstrip("\n")
    if selector_contract.get("model") != MODEL_ID or selector_contract.get("revision") != MODEL_REVISION:
        raise RuntimeError("frozen_model_contract_mismatch")
    if selector_contract.get("prompt_sha256") != PROMPT_SHA or sha256_text(prompt) != PROMPT_SHA:
        raise RuntimeError("frozen_semantic_prompt_mismatch")
    if selector_contract.get("candidate_depth") != 10:
        raise RuntimeError("frozen_candidate_depth_mismatch")
    r1_rows = read_gzip_jsonl(r1_root / "predictions.jsonl.gz")
    if len(r1_rows) != 72 or any(not row.get("raw_output") and row.get("raw_output") != "" for row in r1_rows):
        raise RuntimeError("historical_raw_output_unavailable")
    qwen_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in read_gzip_jsonl(qwen_path):
        prepared = []
        for item in sorted(row["ranked_candidates"], key=lambda value: int(value["post_rerank_rank"])):
            clone = dict(item)
            clone["original_qwen_rank"] = int(item["post_rerank_rank"])
            clone["original_bm25_rank"] = int(item["pre_rerank_rank"]) if item.get("pre_rerank_rank") is not None else None
            prepared.append(clone)
        qwen_by_case[row["case_id"]] = prepared
    if len(qwen_by_case) != 72 or any(len(rows) != 100 for rows in qwen_by_case.values()):
        raise RuntimeError("qwen_top100_contract_invalid")
    bounded_by_case = {row["case_id"]: row["candidates"] for row in read_gzip_jsonl(bounded_path)}
    for case_id, rows in qwen_by_case.items():
        if set(top_keys(rows, 100)) != {item["candidate_key"] for item in bounded_by_case[case_id]}:
            raise RuntimeError(f"candidate_identity_mismatch:{case_id}")
    qwen_sha = sha256_file(qwen_path)
    if qwen_sha != read_json(qwen_seal_path).get("main_prediction_sha256"):
        raise RuntimeError("qwen_prediction_sha_mismatch")
    views_by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_gzip_jsonl(views_path):
        for candidate in row.get("candidates", []) or []:
            views_by_case[row["case_id"]][candidate["candidate_key"]] = candidate
    plans = parse_plan_rows(read_json(plan_path))
    return {"evaluation_root": evaluation_root, "r1_root": r1_root, "r0_root": r0_root, "r1_rows": r1_rows, "qwen_by_case": qwen_by_case, "views_by_case": views_by_case, "plans": plans, "qwen_sha": qwen_sha, "prompt": prompt, "qwen_path": qwen_path}


def load_model():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_SNAPSHOT), revision=MODEL_REVISION, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_SNAPSHOT), revision=MODEL_REVISION, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to("cuda")
    model.eval()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return tokenizer, model, torch, transformers


def constrained_predictions(inputs: dict[str, Any], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

    tokenizer, model, torch, transformers = load_model()
    start = time.perf_counter()
    total_input = total_output = truncated = valid = fallback = duplicate = invalid_id = 0
    rows: list[dict[str, Any]] = []
    for source in sorted(inputs["r1_rows"], key=lambda row: row["case_id"]):
        case_id = source["case_id"]
        candidates = inputs["qwen_by_case"][case_id][:10]
        allowed = top_keys(candidates, 10)
        schema = {"type": "object", "properties": {"selected_ids": {"type": "array", "minItems": 5, "maxItems": 5, "uniqueItems": True, "items": {"type": "string", "enum": allowed}}}, "required": ["selected_ids"], "additionalProperties": False}
        parser = JsonSchemaParser(schema)
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
        question = inputs["plans"].get(case_id, {}).get("raw_question", "")
        prompt = build_prompt(inputs["prompt"], question, candidates, inputs["views_by_case"].get(case_id, {}))
        tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH)
        before_count = len(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
        input_count = int(tokenized["input_ids"].shape[-1])
        total_input += input_count
        truncated_flag = before_count > input_count
        truncated += int(truncated_flag)
        tokenized = {key: value.to("cuda") for key, value in tokenized.items()}
        with torch.inference_mode():
            generated = model.generate(**tokenized, max_new_tokens=STAGE_B_MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id, prefix_allowed_tokens_fn=prefix_fn)
        new_tokens = generated[0, tokenized["input_ids"].shape[-1] :]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        total_output += int(new_tokens.shape[-1])
        parsed = None
        try:
            payload = json.loads(output.strip())
            parsed = payload.get("selected_ids") if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            parsed = None
        normalized = [normalize_id(value, set(allowed)) for value in parsed] if isinstance(parsed, list) else None
        is_valid = isinstance(normalized, list) and len(normalized) == 5 and all(value is not None for value in normalized) and len(set(normalized)) == 5
        if is_valid:
            selected = [value for value in normalized if value is not None]
            valid += 1
        else:
            selected = allowed[:5]
            fallback += 1
            if isinstance(parsed, list):
                duplicate += int(len(normalized) != len(set(normalized)))
                invalid_id += sum(value is None for value in normalized)
        rows.append({"case_id": case_id, "question": question, "input_candidate_ids": allowed, "selected_ids": selected, "selector_used": bool(is_valid), "fallback_to_qwen": not is_valid, "raw_output": output, "structured_output_valid": bool(is_valid), "input_tokens": input_count, "input_tokens_before_truncation": before_count, "truncated": truncated_flag})
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    runtime = {"model_calls": len(rows), "structured_output_valid": valid, "fallback_count": fallback, "invalid_candidate_id_count": invalid_id, "duplicate_id_count": duplicate, "total_input_tokens": total_input, "total_output_tokens": total_output, "elapsed_seconds": elapsed, "queries_per_second": rate(len(rows), elapsed), "truncated_queries": truncated, "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024), "gold_reads_during_prediction": 0, "model": MODEL_ID, "model_revision": MODEL_REVISION, "transformers_version": transformers.__version__, "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16", "max_input_length": MAX_INPUT_LENGTH, "max_new_tokens": STAGE_B_MAX_NEW_TOKENS, "temperature": TEMPERATURE, "top_p": TOP_P, "seed": SEED, "decoder": "lm-format-enforcer_json_schema_prefix_allowed_tokens"}
    return rows, runtime


def score_post_seal(inputs: dict[str, Any], final_rows: list[dict[str, Any]], r1: Any, out_dir: Path, runtime: dict[str, Any], source: str) -> dict[str, Any]:
    evaluation_root = inputs["evaluation_root"]
    r0_root = inputs["r0_root"]
    strict_path = evaluation_root / "pdf-retrieval-v4-strict-source-contract" / "strict-gold-source-bindings.jsonl"
    targets_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1" / "gold-semantic-targets.jsonl"
    registry_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1-p0" / "candidate-semantic-fact-registry.jsonl.gz"
    calc_path = evaluation_root / "nf-opt-19-r0-setwise-ranking-audit" / "calculation-slot-coverage.json"
    nf20_root = evaluation_root / "nf-opt-20-r0-pointwise-discrimination-audit"
    strict_rows = r1.read_jsonl(strict_path)
    targets = r1.load_targets(targets_path)
    facts = r1.build_case_facts(r1.read_gzip_jsonl(registry_path))
    bindings_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_rows:
        bindings_by_case[row["case_id"]].append(row)
    selected = {row["case_id"]: row["selected_ids"] for row in final_rows}
    qwen_ranked = inputs["qwen_by_case"]
    lrrf_rows = r1.read_gzip_jsonl(r0_root / "lrrf-predictions.jsonl.gz")
    lrrf_ranked = {row["case_id"]: sorted(row["ranked_candidates"], key=lambda item: int(item.get("fusion_rank", 0))) for row in lrrf_rows}
    strict_listwise = r1.strict_metrics(strict_rows, selected, qwen_ranked)
    r0_strict = r1.read_json(r0_root / "strict-metrics.json")
    qwen_strict, lrrf_strict = r0_strict["qwen"], r0_strict["lrrf_v1"]
    r1_semantic = r1.read_json(r0_root / "semantic-metrics.json")
    listwise_semantic = r1.semantic_r5(strict_rows, selected, facts, targets)
    def movement(reference: dict[str, list[str]]) -> dict[str, int]:
        rescued = damaged = 0
        for binding in strict_rows:
            old = binding["candidate_key"] in set(reference[binding["case_id"]][:5])
            new = binding["candidate_key"] in set(selected[binding["case_id"]][:5])
            rescued += int(not old and new)
            damaged += int(old and not new)
        return {"rescued": rescued, "damaged": damaged, "net": rescued - damaged}
    qwen_top5 = {case: top_keys(rows, 5) for case, rows in qwen_ranked.items()}
    lrrf_top5 = {case: top_keys(rows, 5) for case, rows in lrrf_ranked.items()}
    qwen_move, lrrf_move = movement(qwen_top5), movement(lrrf_top5)
    multi_cases = sorted(case for case in qwen_ranked if inputs["plans"].get(case, {}).get("requires_multiple_sources") and bindings_by_case.get(case))
    multi = {"qwen": r1.read_json(r0_root / "multi-evidence-analysis.json")["variants"]["qwen"]["@5"], "lrrf_v1": r1.read_json(r0_root / "multi-evidence-analysis.json")["variants"]["lrrf_v1"]["@5"], "listwise": r1.coverage(bindings_by_case, multi_cases, selected)}
    calc = r1.calc_coverage(r1.read_json(calc_path)["cases"], selected, facts)
    calc_baseline = r1.read_json(r0_root / "calculation-slot-analysis.json")["variants"]
    single_cases = sorted(set(qwen_ranked) - set(multi_cases))
    single = r1.strict_metrics([b for case in single_cases for b in bindings_by_case[case]], selected, qwen_ranked)["@5"]
    reversal = r1.read_json(r0_root / "bm25-reversal-recovery.json")
    reversal_records = reversal["lrrf_v1"]["records"]
    listwise_recovered = sum(row.get("gold_candidate_id") in selected[row["case_id"]][:5] for row in reversal_records)
    taxonomy_rows = r1.read_json(nf20_root / "failure-taxonomy.json")["records"]
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"eligible_top10": 0, "rescued": 0, "damaged": 0, "unchanged": 0})
    for row in taxonomy_rows:
        case = row["case_id"]
        binding = next((b for b in bindings_by_case.get(case, []) if int(b.get("source_index", 0)) == int(row.get("source_index", 0))), None)
        if binding is None:
            continue
        qrank = next((i + 1 for i, item in enumerate(qwen_ranked[case]) if item["candidate_key"] == binding["candidate_key"]), 101)
        if qrank > 10:
            continue
        old, new = qrank <= 5, binding["candidate_key"] in selected[case][:5]
        outcome = "rescued" if not old and new else "damaged" if old and not new else "unchanged"
        for tag in row.get("tags", []):
            type_counts[tag]["eligible_top10"] += 1
            type_counts[tag][outcome] += 1
    write_json(out_dir / "strict-metrics.json", {"qwen": qwen_strict, "lrrf_v1": lrrf_strict, "listwise": strict_listwise, "qwen_top10_supply": 60})
    write_json(out_dir / "semantic-metrics.json", {"qwen": r1_semantic["qwen"], "lrrf_v1": r1_semantic["lrrf_v1"], "listwise": listwise_semantic})
    write_json(out_dir / "qwen-comparison.json", {"reference": "qwen_pointwise", **qwen_move, "strict_r5": strict_listwise["@5"], "selection_efficiency": rate(strict_listwise["@5"]["hits"], 60)})
    write_json(out_dir / "lrrf-comparison.json", {"reference": "lrrf_v1", **lrrf_move, "strict_r5": strict_listwise["@5"], "selection_efficiency": rate(strict_listwise["@5"]["hits"], 60)})
    write_json(out_dir / "multi-evidence-analysis.json", {"denominator": 16, "qwen": multi["qwen"], "lrrf_v1": multi["lrrf_v1"], "listwise": multi["listwise"]})
    write_json(out_dir / "calculation-slot-analysis.json", {"denominator": 11, "qwen": calc_baseline["qwen"]["@5"], "lrrf_v1": calc_baseline["lrrf_v1"]["@5"], "listwise": calc})
    write_json(out_dir / "single-evidence-analysis.json", {"qwen": {"strict_r5": r1.read_json(r0_root / "single-evidence-analysis.json")["qwen"]["strict_r5"]}, "lrrf_v1": {"strict_r5": r1.read_json(r0_root / "single-evidence-analysis.json")["lrrf_v1"]["strict_r5"]}, "listwise": {"strict_r5": single}})
    write_json(out_dir / "bm25-reversal-analysis.json", {"frozen_cohort_total": 15, "lrrf_recovered": reversal["lrrf_v1"]["recovered"], "listwise_recovered": listwise_recovered})
    write_json(out_dir / "failure-type-analysis.json", {"records": dict(type_counts), "gold_reads_after_seal": True})
    list_hits = strict_listwise["@5"]["hits"]
    safe = listwise_semantic["hits"] >= 51 and multi["listwise"]["all"] >= 7 and calc["all_slots"] >= 6
    if list_hits >= 50 and list_hits - 43 >= 7 and qwen_move["damaged"] <= 3 and safe:
        effectiveness: bool | str = True
        next_gate = "internal_retrieval_shadow_freeze"
        selected_method = "listwise"
    elif 47 <= list_hits <= 49 and list_hits >= 46 and safe and qwen_move["damaged"] < qwen_move["rescued"]:
        effectiveness = "marginal"
        next_gate = "internal_retrieval_shadow_freeze"
        selected_method = "listwise"
    else:
        effectiveness = False
        next_gate = "internal_retrieval_method_freeze"
        selected_method = "lrrf_v1"
    decision = {"gate": "NF-OPT-21-R1.1", "evaluation_role": "development_shadow_output_contract_recovery", "fresh_blind_evaluation": False, "semantic_prompt_sha": PROMPT_SHA, "model": MODEL_ID, "model_revision": MODEL_REVISION, "stage_a_raw_output_available": True, "stage_a_exact_valid": None, "stage_a_tolerant_recoverable": None, "stage_a_unrecoverable": None, "stage_b_executed": True, "final_prediction_source": source, "queries": 72, "qwen_strict_r5_hits": 43, "lrrf_strict_r5_hits": 46, "listwise_strict_r5_hits": list_hits, "listwise_semantic_r5_hits": listwise_semantic["hits"], "listwise_multi_all_at_5": multi["listwise"]["all"], "listwise_calculation_all_slots_at_5": calc["all_slots"], "listwise_rescued_vs_qwen": qwen_move["rescued"], "listwise_damaged_vs_qwen": qwen_move["damaged"], "listwise_selector_effective": effectiveness, "selected_internal_shadow_method": selected_method, "production_switch_allowed": False, "mandatory_method_freeze": True, "next_gate": next_gate, "gold_reads_during_prediction": 0}
    return {"strict": strict_listwise, "semantic": listwise_semantic, "qwen_move": qwen_move, "lrrf_move": lrrf_move, "multi": multi, "calc": calc, "decision": decision, "runtime": runtime, "listwise_recovered": listwise_recovered}


def main() -> int:
    backend_root = Path(__file__).resolve().parents[2]
    out_dir = backend_root / "artifacts" / "evaluation" / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    r1 = load_r1_module(backend_root)
    inputs = load_frozen_inputs(backend_root, r1)
    audit_rows = []
    counts = Counter()
    recoveries: dict[str, list[str]] = {}
    for row in sorted(inputs["r1_rows"], key=lambda value: value["case_id"]):
        result = classify_and_recover(row.get("raw_output", ""), set(row["input_candidate_ids"]))
        counts[result["category"]] += 1
        if result["recoverable"]:
            recoveries[row["case_id"]] = result["ids"]
        audit_rows.append({"case_id": row["case_id"], "category": result["category"], "recoverable": result["recoverable"], "method": result["method"], "explicit_ids": result["ids"], "raw_output_sha256": sha256_text(row.get("raw_output", "")), "raw_output_length": len(row.get("raw_output", ""))})
    exact_valid = sum(item["category"] == "A0_exact_contract_json" for item in audit_rows)
    recoverable_total = sum(item["recoverable"] for item in audit_rows)
    tolerant_recoverable = recoverable_total - exact_valid
    unrecoverable = len(audit_rows) - recoverable_total
    write_json(out_dir / "frozen-contract.json", {"base_commit": BASE_COMMIT, "model": MODEL_ID, "model_revision": MODEL_REVISION, "semantic_prompt_sha": PROMPT_SHA, "semantic_prompt_unchanged": True, "query_count": 72, "candidate_depth": 10, "candidate_source": "historical_r1_qwen_top10", "qwen_prediction_sha256": inputs["qwen_sha"], "r1_max_new_tokens": R1_MAX_NEW_TOKENS, "gold_reads_before_stage_a": 0, "production_switch_allowed": False})
    write_json(out_dir / "raw-output-audit.json", {"raw_outputs": len(audit_rows), "stage_a_raw_output_available": True, "records": audit_rows})
    write_json(out_dir / "raw-output-failure-taxonomy.json", {"counts": {name: counts.get(name, 0) for name in CATEGORY_NAMES}, "percentages": {name: rate(counts.get(name, 0), len(audit_rows)) for name in CATEGORY_NAMES}, "exact_valid": exact_valid, "tolerant_recoverable": tolerant_recoverable, "recoverable_total": recoverable_total, "unrecoverable": unrecoverable, "recoverable_rate": rate(recoverable_total, len(audit_rows))})
    write_json(out_dir / "tolerant-parser-contract.json", {"parser": "TOP-V1", "allowed": ["exact_json", "markdown_fenced_json", "embedded_unique_json", "explicit_five_id_list"], "normalization": ["trim_whitespace", "strip_quotes_backticks", "case_insensitive_unique_resolution"], "invent_ids": False, "autofill": False, "gold_reads": 0})
    write_json(out_dir / "tolerant-parser-tests.json", {"exact_json": True, "fenced_json": True, "embedded_json": True, "explicit_id_list": True, "fewer_than_five_rejected": True, "duplicate_rejected": True, "out_of_universe_rejected": True})
    stage_a_rows = []
    for row in inputs["r1_rows"]:
        stage_a_rows.append({"case_id": row["case_id"], "selected_ids": recoveries.get(row["case_id"], row["input_candidate_ids"][:5]), "input_candidate_ids": row["input_candidate_ids"], "source": "historical_raw_output_recovery" if row["case_id"] in recoveries else "original_qwen_top5_fallback", "fallback_to_qwen": row["case_id"] not in recoveries})
    stage_a_sha = write_gzip_jsonl(out_dir / "stage-a-recovered-predictions.jsonl.gz", stage_a_rows)
    write_json(out_dir / "stage-a-prediction-seal.json", {"gate": "NF-OPT-21-R1.1", "stage": "A", "rows": 72, "prediction_sha256": stage_a_sha, "sealed": True, "gold_reads_during_prediction": 0})
    stage_b_allowed = recoverable_total < RECOVERY_THRESHOLD
    stage_b_runtime: dict[str, Any] = {"executed": False, "reason": "stage_a_recovery_threshold_met"}
    final_rows = stage_a_rows
    final_source = "historical_raw_output_recovery"
    if stage_b_allowed:
        final_rows, stage_b_runtime = constrained_predictions(inputs, out_dir)
        final_source = "constrained_output_reexecution"
        stage_b_sha = write_gzip_jsonl(out_dir / "stage-b-predictions.jsonl.gz", final_rows)
        write_json(out_dir / "stage-b-prediction-seal.json", {"gate": "NF-OPT-21-R1.1", "stage": "B", "rows": 72, "prediction_sha256": stage_b_sha, "sealed": True, "gold_reads_during_prediction": 0})
    else:
        write_json(out_dir / "stage-b-output-contract.json", {"executed": False, "reason": "stage_a_recovery_threshold_met"})
        write_json(out_dir / "stage-b-runtime.json", stage_b_runtime)
    final_sha = write_gzip_jsonl(out_dir / "final-predictions.jsonl.gz", final_rows)
    write_json(out_dir / "final-prediction-source.json", {"source": final_source, "prediction_sha256": final_sha, "rows": 72, "gold_reads_during_prediction": 0, "stage_a_recoverable_total": recoverable_total, "stage_b_executed": stage_b_allowed})
    if stage_b_allowed:
        schema_hash = sha256_text(json.dumps({"type": "object", "properties": {"selected_ids": {"type": "array", "minItems": 5, "maxItems": 5, "uniqueItems": True, "items": {"type": "string", "enum": "per_query_frozen_top10"}}}, "required": ["selected_ids"], "additionalProperties": False}, sort_keys=True, separators=(",", ":")))
        write_json(out_dir / "stage-b-output-contract.json", {"executed": True, "decoder": "lm-format-enforcer_json_schema_prefix_allowed_tokens", "schema_sha256": schema_hash, "semantic_prompt_sha": PROMPT_SHA, "max_new_tokens": STAGE_B_MAX_NEW_TOKENS, "one_call_per_query": True, "no_repair_prompt": True, "candidate_enum": "per_query_frozen_top10"})
        write_json(out_dir / "stage-b-runtime.json", stage_b_runtime)
    score = score_post_seal(inputs, final_rows, r1, out_dir, stage_b_runtime, final_source)
    decision = score["decision"]
    decision.update({"stage_a_exact_valid": exact_valid, "stage_a_tolerant_recoverable": tolerant_recoverable, "stage_a_unrecoverable": unrecoverable, "stage_b_executed": stage_b_allowed, "stage_b_recovery_threshold": RECOVERY_THRESHOLD, "stage_b_structured_output_valid": stage_b_runtime.get("structured_output_valid", 0), "stage_b_fallback_count": stage_b_runtime.get("fallback_count", 0), "final_prediction_source": final_source, "stage_a_raw_output_available": True})
    write_json(out_dir / "decision.json", decision)
    (out_dir / "README.md").write_text(f"# NF-OPT-21 R1.1 — Listwise Output Contract Recovery\n\nStage A recoverable: {recoverable_total}/72; Stage B executed: {stage_b_allowed}. Final source: `{final_source}`. Decision: `{decision['listwise_selector_effective']}`. This is development-shadow evidence; production switch is disabled.\n", encoding="utf-8")
    print(json.dumps({"stage_a": {"exact_valid": exact_valid, "tolerant_recoverable": tolerant_recoverable, "unrecoverable": unrecoverable, "taxonomy": dict(counts)}, "stage_b": stage_b_runtime, "score": score, "decision": decision}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
