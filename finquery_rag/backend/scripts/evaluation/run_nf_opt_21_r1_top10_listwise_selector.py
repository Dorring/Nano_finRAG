"""NF-OPT-21 R1: frozen Top10 listwise evidence selector shadow test.

The model prediction stage is deliberately Gold-blind. It consumes only the
sealed Qwen Top10, persisted candidate views, and the original question. All
strict/semantic and historical-cohort diagnostics are loaded after sealing.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


BASE_COMMIT = "45ecd7e"
OUT_NAME = "nf-opt-21-r1-top10-listwise-selector"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_SNAPSHOT = Path("/home/mxf/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots") / MODEL_REVISION
MAX_INPUT_LENGTH = 16384
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 0

PROMPT = """You are selecting evidence for a financial question.

Given one question and 10 retrieved evidence candidates,
select exactly 5 candidates that together provide the most
direct and trustworthy evidence needed to answer the question.

Prioritize:
- exact metric / financial concept match
- correct reporting period
- correct financial statement / table / row
- direct evidence over generic semantic similarity
- complementary evidence when the question requires
  multiple periods, operands, or facts

Do not infer facts not shown in the candidates.
Do not select candidates only because their wording is similar.
Use the provided Qwen and BM25 ranks only as supporting signals,
not as ground truth.

Return exactly 5 unique candidate IDs,
ordered from most to least useful.

Output JSON only:
{"selected_ids":["...","...","...","...","..."]}"""


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def top_keys(items: list[dict[str, Any]], k: int) -> list[str]:
    return [item["candidate_key"] for item in items[:k]]


def rank_map(items: list[dict[str, Any]]) -> dict[str, int]:
    return {item["candidate_key"]: index + 1 for index, item in enumerate(items)}


def parse_plan_rows(payload: Any) -> dict[str, dict[str, Any]]:
    rows = payload.get("plans", []) if isinstance(payload, dict) else payload
    return {row["case_id"]: row.get("plan", {}) for row in rows or []}


def build_case_facts(registry_rows: list[dict[str, Any]]) -> dict[str, dict[str, set[str]]]:
    facts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for record in registry_rows:
        ids = set(record.get("semantic_fact_ids", []))
        for fact in record.get("semantic_facts", []) or []:
            if fact.get("semantic_fact_id"):
                ids.add(fact["semantic_fact_id"])
        for occurrence in record.get("occurrences", []) or []:
            if occurrence.get("case_id"):
                facts[occurrence["case_id"]][record["candidate_key"]].update(ids)
    return facts


def semantic_union(case_id: str, ranked: list[dict[str, Any]], k: int, facts: dict[str, dict[str, set[str]]]) -> set[str]:
    values: set[str] = set()
    for key in top_keys(ranked, k):
        values.update(facts.get(case_id, {}).get(key, set()))
    return values


def load_targets(path: Path) -> dict[str, dict[str, Any]]:
    return {row["binding_id"]: row for row in read_jsonl(path)}


def target_ids(binding: dict[str, Any], targets: dict[str, dict[str, Any]]) -> set[str]:
    target = targets.get(binding.get("binding_id"), {})
    values = set(target.get("matching_semantic_fact_ids") or [])
    if target.get("gold_semantic_fact_id"):
        values.add(target["gold_semantic_fact_id"])
    return values


def strict_metrics(bindings: list[dict[str, Any]], selected: dict[str, list[str]], qwen_ranked: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for k in (1, 3, 5):
        hits = sum(binding["candidate_key"] in set(selected[binding["case_id"]][:k]) for binding in bindings)
        output[f"@{k}"] = {"hits": hits, "total": len(bindings), "rate": rate(hits, len(bindings))}
    hits10 = sum(binding["candidate_key"] in set(top_keys(qwen_ranked[binding["case_id"]], 10)) for binding in bindings)
    output["@10"] = {"hits": hits10, "total": len(bindings), "rate": rate(hits10, len(bindings)), "source": "original_qwen_top10_candidate_supply_invariant"}
    return output


def semantic_r5(bindings: list[dict[str, Any]], selected: dict[str, list[str]], facts: dict[str, dict[str, set[str]]], targets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hits = 0
    for binding in bindings:
        case_id = binding["case_id"]
        ids = target_ids(binding, targets)
        chosen = selected[case_id][:5]
        union = set().union(*(facts.get(case_id, {}).get(key, set()) for key in chosen))
        hits += int(binding["candidate_key"] in chosen or bool(ids & union))
    return {"hits": hits, "total": len(bindings), "rate": rate(hits, len(bindings))}


def coverage(bindings_by_case: dict[str, list[dict[str, Any]]], case_ids: list[str], selected: dict[str, list[str]]) -> dict[str, Any]:
    any_count = all_count = 0
    ratios: list[float] = []
    for case_id in case_ids:
        bindings = bindings_by_case.get(case_id, [])
        keys = set(selected[case_id][:5])
        represented = sum(binding["candidate_key"] in keys for binding in bindings)
        any_count += int(represented > 0)
        all_count += int(bool(bindings) and represented == len(bindings))
        ratios.append(rate(represented, len(bindings)) or 0.0)
    return {"queries": len(case_ids), "any": any_count, "all": all_count, "any_rate": rate(any_count, len(case_ids)), "all_rate": rate(all_count, len(case_ids)), "average_required_source_coverage": statistics.mean(ratios) if ratios else None}


def calc_coverage(calc_rows: list[dict[str, Any]], selected: dict[str, list[str]], facts: dict[str, dict[str, set[str]]]) -> dict[str, Any]:
    any_count = all_count = represented_total = required_total = 0
    records: list[dict[str, Any]] = []
    for row in calc_rows:
        required = list(row.get("required_semantic_fact_ids") or [])
        union = set().union(*(facts.get(row["case_id"], {}).get(key, set()) for key in selected[row["case_id"]][:5]))
        represented = sum(value in union for value in required)
        any_count += int(represented > 0)
        all_count += int(bool(required) and represented == len(required))
        represented_total += represented
        required_total += len(required)
        records.append({"case_id": row["case_id"], "required_slots": len(required), "represented_slots": represented, "all_slots": bool(required) and represented == len(required)})
    return {"queries": len(calc_rows), "any_slots": any_count, "all_slots": all_count, "any_slots_rate": rate(any_count, len(calc_rows)), "all_slots_rate": rate(all_count, len(calc_rows)), "represented_slots": represented_total, "required_slots": required_total, "average_slot_coverage": rate(represented_total, required_total), "records": records}


def parse_selection(text: str, allowed: set[str]) -> tuple[list[str] | None, bool, dict[str, int]]:
    stripped = text.strip()
    repaired = False
    payload: Any = None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                repaired = True
            except json.JSONDecodeError:
                payload = None
    stats = {"invalid_candidate_id": 0, "duplicate_id": 0}
    if not isinstance(payload, dict) or not isinstance(payload.get("selected_ids"), list):
        return None, repaired, stats
    values = payload["selected_ids"]
    if len(values) != len(set(values)):
        stats["duplicate_id"] = 1
    if any(value not in allowed for value in values):
        stats["invalid_candidate_id"] = 1
    if len(values) != 5 or len(set(values)) != 5 or any(value not in allowed for value in values):
        return None, repaired, stats
    return [str(value) for value in values], repaired, stats


def build_prompt(question: str, candidates: list[dict[str, Any]], views: dict[str, dict[str, Any]]) -> str:
    blocks = [PROMPT, "", f"Question:\n{question}", "", "Candidates:"]
    for index, item in enumerate(candidates, start=1):
        view = views.get(item["candidate_key"], {})
        blocks.extend([
            f"\nCandidate {index}",
            f"Candidate ID: {item['candidate_key']}",
            f"Qwen rank: {item['original_qwen_rank']}",
            f"BM25 rank: {item['original_bm25_rank'] if item['original_bm25_rank'] is not None else 'unknown'}",
            f"Context status: {view.get('context_status', item.get('context_status', 'unknown'))}",
            "Evidence:",
            view.get("document_view", "[NO FROZEN DOCUMENT VIEW AVAILABLE]"),
        ])
    return "\n".join(blocks)


def load_model() -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("cuda_unavailable_for_listwise_selector")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_SNAPSHOT), revision=MODEL_REVISION, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_SNAPSHOT), revision=MODEL_REVISION, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to("cuda")
    model.eval()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    config = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_index_visible": 0,
        "dtype": "bfloat16",
        "device": "cuda",
        "max_input_length": MAX_INPUT_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "decoding": "greedy_do_sample_false",
        "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
    }
    return tokenizer, model, torch, config


def generate_predictions(
    query_rows: list[dict[str, Any]],
    qwen_by_case: dict[str, list[dict[str, Any]]],
    plans: dict[str, dict[str, Any]],
    views_by_case: dict[str, dict[str, dict[str, Any]]],
    tokenizer: Any,
    model: Any,
    torch: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total_input_tokens = total_output_tokens = truncated = format_repairs = fallback = invalid_ids = duplicate_ids = 0
    start = time.perf_counter()
    for row in sorted(query_rows, key=lambda item: item["case_id"]):
        case_id = row["case_id"]
        candidates = qwen_by_case[case_id][:10]
        question = plans.get(case_id, {}).get("raw_question", "")
        prompt = build_prompt(question, candidates, views_by_case.get(case_id, {}))
        tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH)
        before_count = len(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
        input_count = int(tokenized["input_ids"].shape[-1])
        total_input_tokens += input_count
        truncated_flag = before_count > input_count
        truncated += int(truncated_flag)
        tokenized = {key: value.to("cuda") for key, value in tokenized.items()}
        with torch.inference_mode():
            generated = model.generate(**tokenized, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        new_tokens = generated[0, tokenized["input_ids"].shape[-1] :]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        total_output_tokens += int(new_tokens.shape[-1])
        allowed = {item["candidate_key"] for item in candidates}
        selected, repaired, parse_stats = parse_selection(output_text, allowed)
        format_repairs += int(repaired)
        invalid_ids += parse_stats["invalid_candidate_id"]
        duplicate_ids += parse_stats["duplicate_id"]
        fallback_used = selected is None
        if fallback_used:
            selected = top_keys(candidates, 5)
            fallback += 1
        results.append({
            "case_id": case_id,
            "question": question,
            "input_candidate_ids": top_keys(candidates, 10),
            "selected_ids": selected,
            "selector_used": not fallback_used,
            "fallback_to_qwen": fallback_used,
            "format_repaired": repaired,
            "raw_output": output_text,
            "input_tokens": input_count,
            "input_tokens_before_truncation": before_count,
            "truncated": truncated_flag,
        })
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return results, {
        "model_calls": len(results),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "elapsed_seconds": elapsed,
        "queries_per_second": rate(len(results), elapsed),
        "truncated_queries": truncated,
        "format_repairs": format_repairs,
        "fallback_to_qwen": fallback,
        "invalid_candidate_id_count": invalid_ids,
        "duplicate_id_count": duplicate_ids,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }


def main() -> int:
    backend_root = Path(__file__).resolve().parents[2]
    evaluation_root = backend_root / "artifacts" / "evaluation"
    out_dir = evaluation_root / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    r0_root = evaluation_root / "nf-opt-21-r0-qwen-bm25-late-fusion"
    qwen_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3" / "main_rerank_predictions.jsonl.gz"
    bounded_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r2a-2" / "bounded-top100-predictions.jsonl.gz"
    qwen_seal_path = qwen_path.parent / "prediction-seal.json"
    views_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-r3-3-p0" / "queryplan-rerank-input-views.jsonl.gz"
    plan_path = evaluation_root / "pdf-retrieval-v4-gate-07" / "query-plan-predictions.json"
    strict_path = evaluation_root / "pdf-retrieval-v4-strict-source-contract" / "strict-gold-source-bindings.jsonl"
    targets_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1" / "gold-semantic-targets.jsonl"
    registry_path = evaluation_root / "pdf-retrieval-v4-gate-08-r8-se1-p0" / "candidate-semantic-fact-registry.jsonl.gz"
    calc_path = evaluation_root / "nf-opt-19-r0-setwise-ranking-audit" / "calculation-slot-coverage.json"
    nf20_root = evaluation_root / "nf-opt-20-r0-pointwise-discrimination-audit"

    contract = read_json(r0_root / "frozen-input-contract.json")
    if contract.get("candidate_identity_mismatch") != 0 or contract.get("top10_candidate_set_invariant") is not True:
        raise RuntimeError("nf_opt_21_r0_frozen_input_contract_invalid")
    qwen_rows = read_gzip_jsonl(qwen_path)
    qwen_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in qwen_rows:
        prepared: list[dict[str, Any]] = []
        for item in sorted(row["ranked_candidates"], key=lambda value: int(value["post_rerank_rank"])):
            clone = dict(item)
            clone["original_qwen_rank"] = int(item["post_rerank_rank"])
            clone["original_bm25_rank"] = int(item["pre_rerank_rank"]) if item.get("pre_rerank_rank") is not None else None
            prepared.append(clone)
        qwen_by_case[row["case_id"]] = prepared
    if len(qwen_by_case) != 72 or any(len(rows) != 100 for rows in qwen_by_case.values()):
        raise RuntimeError("qwen_top100_query_contract_invalid")
    bounded_by_case = {row["case_id"]: row["candidates"] for row in read_gzip_jsonl(bounded_path)}
    for case_id, rows in qwen_by_case.items():
        qwen_ids = set(top_keys(rows, 100))
        bounded_ids = {item["candidate_key"] for item in bounded_by_case[case_id]}
        if qwen_ids != bounded_ids:
            raise RuntimeError(f"frozen_candidate_identity_mismatch:{case_id}")
    qwen_sha = sha256_file(qwen_path)
    if qwen_sha != read_json(qwen_seal_path).get("main_prediction_sha256"):
        raise RuntimeError("qwen_sha_mismatch")

    views_by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_gzip_jsonl(views_path):
        for candidate in row.get("candidates", []) or []:
            views_by_case[row["case_id"]][candidate["candidate_key"]] = candidate
    plans = parse_plan_rows(read_json(plan_path))
    query_rows = [{"case_id": case_id, "question": plans.get(case_id, {}).get("raw_question", "")} for case_id in sorted(qwen_by_case)]
    model_snapshot_files = {}
    for filename in ("config.json", "tokenizer_config.json", "generation_config.json", "tokenizer.json"):
        path = MODEL_SNAPSHOT / filename
        model_snapshot_files[filename] = {"sha256": sha256_file(path), "bytes": path.stat().st_size} if path.exists() else None
    selector_contract = {
        "gate": "NF-OPT-21-R1",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot": str(MODEL_SNAPSHOT),
        "dtype": "bfloat16",
        "max_input_length": MAX_INPUT_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "decoding": "greedy_do_sample_false",
        "prompt_sha256": sha256_text(PROMPT),
        "candidate_depth": 10,
        "one_call_per_query": True,
        "fallback": "original_qwen_top5",
        "gold_reads_during_prediction": 0,
        "model_snapshot_files": model_snapshot_files,
    }
    write_json(out_dir / "selector-contract.json", selector_contract)
    (out_dir / "prompt.txt").write_text(PROMPT + "\n", encoding="utf-8")
    (out_dir / "prompt-sha256.txt").write_text(sha256_text(PROMPT) + "\n", encoding="utf-8")
    write_json(out_dir / "input-manifest.json", {
        "gate": "NF-OPT-21-R1",
        "qwen_prediction_sha256": qwen_sha,
        "qwen_query_count": len(qwen_by_case),
        "candidate_depth": 10,
        "candidate_ids_from": "frozen_qwen_top10",
        "retrieval_rerun": False,
        "candidate_generation": False,
        "training": False,
        "gold_reads_during_prediction": 0,
    })

    tokenizer, model, torch, runtime_contract = load_model()
    prediction_rows, runtime_metrics = generate_predictions(query_rows, qwen_by_case, plans, views_by_case, tokenizer, model, torch)
    prediction_path = out_dir / "predictions.jsonl.gz"
    prediction_sha = write_gzip_jsonl(prediction_path, prediction_rows)
    if len(prediction_rows) != 72 or any(len(row["selected_ids"]) != 5 for row in prediction_rows):
        raise RuntimeError("prediction_completeness_invalid")
    write_json(out_dir / "runtime-metrics.json", {**runtime_contract, **runtime_metrics, "model_calls": 72, "gold_reads_during_prediction": 0})
    manifest = {"gate": "NF-OPT-21-R1", "rows": len(prediction_rows), "prediction_sha256": prediction_sha, "gold_reads_during_prediction": 0, "candidate_depth": 10, "selected_count": 5, "sealed": True}
    write_json(out_dir / "prediction-seal.json", manifest)

    # Post-seal only: Gold and historical diagnostic artifacts are loaded here.
    strict_rows = read_jsonl(strict_path)
    targets = load_targets(targets_path)
    facts = build_case_facts(read_gzip_jsonl(registry_path))
    bindings_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_rows:
        bindings_by_case[row["case_id"]].append(row)
    selected = {row["case_id"]: row["selected_ids"] for row in prediction_rows}
    qwen_ranked = qwen_by_case
    lrrf_rows = read_gzip_jsonl(r0_root / "lrrf-predictions.jsonl.gz")
    lrrf_ranked = {row["case_id"]: sorted(row["ranked_candidates"], key=lambda item: int(item.get("fusion_rank", 0))) for row in lrrf_rows}
    strict_listwise = strict_metrics(strict_rows, selected, qwen_ranked)
    r0_strict = read_json(r0_root / "strict-metrics.json")
    qwen_strict = r0_strict["qwen"]
    lrrf_strict = r0_strict["lrrf_v1"]
    write_json(out_dir / "strict-metrics.json", {"qwen": qwen_strict, "lrrf_v1": lrrf_strict, "listwise": strict_listwise, "qwen_top10_supply": 60})
    r0_semantic = read_json(r0_root / "semantic-metrics.json")
    qwen_semantic = r0_semantic["qwen"]["@5"]
    lrrf_semantic = r0_semantic["lrrf_v1"]["@5"]
    listwise_semantic = semantic_r5(strict_rows, selected, facts, targets)
    write_json(out_dir / "semantic-metrics.json", {"qwen": qwen_semantic, "lrrf_v1": lrrf_semantic, "listwise": listwise_semantic})

    def selected_movement(reference: dict[str, list[str]]) -> dict[str, Any]:
        rescued = damaged = 0
        for binding in strict_rows:
            case_id = binding["case_id"]
            key = binding["candidate_key"]
            old_hit = key in set(reference[case_id][:5])
            new_hit = key in set(selected[case_id][:5])
            rescued += int(not old_hit and new_hit)
            damaged += int(old_hit and not new_hit)
        return {"rescued": rescued, "damaged": damaged, "net": rescued - damaged}

    qwen_top5 = {case_id: top_keys(rows, 5) for case_id, rows in qwen_ranked.items()}
    lrrf_top5 = {case_id: top_keys(rows, 5) for case_id, rows in lrrf_ranked.items()}
    qwen_move = selected_movement(qwen_top5)
    lrrf_move = selected_movement(lrrf_top5)
    write_json(out_dir / "qwen-comparison.json", {"reference": "qwen_pointwise", **qwen_move, "strict_r5": qwen_strict["@5"], "listwise_strict_r5": strict_listwise["@5"], "selection_efficiency": rate(strict_listwise["@5"]["hits"], 60)})
    write_json(out_dir / "lrrf-comparison.json", {"reference": "lrrf_v1", **lrrf_move, "strict_r5": lrrf_strict["@5"], "listwise_strict_r5": strict_listwise["@5"], "selection_efficiency": rate(strict_listwise["@5"]["hits"], 60)})

    multi_cases = sorted(case_id for case_id in qwen_by_case if plans.get(case_id, {}).get("requires_multiple_sources") and bindings_by_case.get(case_id))
    single_cases = sorted(set(qwen_by_case) - set(multi_cases))
    r0_multi = read_json(r0_root / "multi-evidence-analysis.json")
    multi_out = {"qwen": {"@5": r0_multi["variants"]["qwen"]["@5"]}, "lrrf_v1": {"@5": r0_multi["variants"]["lrrf_v1"]["@5"]}, "listwise": {"@5": coverage(bindings_by_case, multi_cases, selected)}}
    write_json(out_dir / "multi-evidence-analysis.json", {"denominator": 16, "variants": multi_out})
    calc_rows = read_json(calc_path)["cases"]
    calc_selected = calc_coverage(calc_rows, selected, facts)
    r0_calc = read_json(r0_root / "calculation-slot-analysis.json")["variants"]
    write_json(out_dir / "calculation-slot-analysis.json", {"denominator": 11, "qwen": r0_calc["qwen"]["@5"], "lrrf_v1": r0_calc["lrrf_v1"]["@5"], "listwise": calc_selected})
    single_listwise = strict_metrics([binding for case_id in single_cases for binding in bindings_by_case[case_id]], selected, qwen_ranked)["@5"]
    r0_single = read_json(r0_root / "single-evidence-analysis.json")
    write_json(out_dir / "single-evidence-analysis.json", {"qwen": {"query_count": 56, "strict_r5": r0_single["qwen"]["strict_r5"]}, "lrrf_v1": {"query_count": 56, "strict_r5": r0_single["lrrf_v1"]["strict_r5"]}, "listwise": {"query_count": 56, "strict_r5": single_listwise}})

    reversal = read_json(r0_root / "bm25-reversal-recovery.json")
    reversal_records = reversal["lrrf_v1"]["records"]
    listwise_recovered = sum(bool(row.get("gold_candidate_id")) and row["gold_candidate_id"] in selected[row["case_id"]][:5] for row in reversal_records)
    write_json(out_dir / "bm25-reversal-analysis.json", {"frozen_cohort_total": 15, "lrrf_recovered": reversal["lrrf_v1"]["recovered"], "listwise_recovered": listwise_recovered, "records": reversal_records})

    taxonomy_rows = read_json(nf20_root / "failure-taxonomy.json")["records"]
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"eligible_top10": 0, "rescued": 0, "damaged": 0, "unchanged": 0})
    for row in taxonomy_rows:
        case_id = row["case_id"]
        binding = next((b for b in bindings_by_case.get(case_id, []) if int(b.get("source_index", 0)) == int(row.get("source_index", 0))), None)
        if binding is None:
            continue
        qrank = rank_map(qwen_ranked[case_id]).get(binding["candidate_key"], 101)
        if qrank > 10:
            continue
        old_hit = qrank <= 5
        new_hit = binding["candidate_key"] in selected[case_id][:5]
        outcome = "rescued" if not old_hit and new_hit else "damaged" if old_hit and not new_hit else "unchanged"
        for tag in row.get("tags", []):
            type_counts[tag]["eligible_top10"] += 1
            type_counts[tag][outcome] += 1
    write_json(out_dir / "failure-type-analysis.json", {"records": dict(type_counts), "gold_reads_after_seal": True})

    near_clear = {"near_boundary": {"eligible": 0, "rescued": 0, "damaged": 0}, "clear_loss": {"eligible": 0, "rescued": 0, "damaged": 0}}
    for row in taxonomy_rows:
        case_id = row["case_id"]
        binding = next((b for b in bindings_by_case.get(case_id, []) if int(b.get("source_index", 0)) == int(row.get("source_index", 0))), None)
        if binding is None:
            continue
        old = rank_map(qwen_ranked[case_id]).get(binding["candidate_key"], 101)
        new = binding["candidate_key"] in selected[case_id][:5]
        cohort = row.get("cohort")
        if cohort in near_clear and old > 5 and old <= 10:
            near_clear[cohort]["eligible"] += 1
            near_clear[cohort]["rescued"] += int(new)
            near_clear[cohort]["damaged"] += int(old <= 5 and not new)
    write_json(out_dir / "cohort-analysis.json", {"near_boundary": near_clear["near_boundary"], "clear_loss": near_clear["clear_loss"], "frozen_source": "NF-OPT-20 R0 taxonomy"})

    validity = {
        "queries": len(prediction_rows),
        "first_pass_valid": sum(not row["format_repaired"] and row["selector_used"] for row in prediction_rows),
        "format_repairs": runtime_metrics["format_repairs"],
        "fallback_to_qwen": runtime_metrics["fallback_to_qwen"],
        "invalid_candidate_id_count": runtime_metrics["invalid_candidate_id_count"],
        "duplicate_id_count": runtime_metrics["duplicate_id_count"],
        "final_invalid": 0,
        "gold_reads_during_prediction": 0,
    }
    write_json(out_dir / "output-validity.json", validity)

    list_hits = strict_listwise["@5"]["hits"]
    safe = listwise_semantic["hits"] >= 51 and multi_out["listwise"]["@5"]["all"] >= 7 and calc_selected["all_slots"] >= 6
    if list_hits >= 50 and list_hits - 43 >= 7 and qwen_move["damaged"] <= 3 and safe:
        effectiveness: bool | str = True
        next_gate = "internal_retrieval_shadow_freeze"
    elif 47 <= list_hits <= 49 and list_hits >= 46 and safe and qwen_move["damaged"] < qwen_move["rescued"]:
        effectiveness = "marginal"
        next_gate = "internal_retrieval_shadow_freeze"
    else:
        effectiveness = False
        next_gate = "internal_retrieval_method_freeze"
    decision = {
        "gate": "NF-OPT-21-R1",
        "base_commit": BASE_COMMIT,
        "evaluation_role": "development_shadow_listwise_selection",
        "fresh_blind_evaluation": False,
        "retrieval_rerun": False,
        "candidate_generation": False,
        "training": False,
        "qwen_pointwise_rerun": False,
        "queries": 72,
        "strict_sources": 80,
        "qwen_top10_supply": 60,
        "qwen_strict_r5_hits": 43,
        "lrrf_strict_r5_hits": 46,
        "listwise_strict_r5_hits": list_hits,
        "listwise_rescued_vs_qwen": qwen_move["rescued"],
        "listwise_damaged_vs_qwen": qwen_move["damaged"],
        "listwise_net_vs_qwen": qwen_move["net"],
        "listwise_rescued_vs_lrrf": lrrf_move["rescued"],
        "listwise_damaged_vs_lrrf": lrrf_move["damaged"],
        "listwise_net_vs_lrrf": lrrf_move["net"],
        "selection_efficiency": rate(list_hits, 60),
        "semantic_r5_hits": listwise_semantic["hits"],
        "multi_evidence_all_at_5": multi_out["listwise"]["@5"]["all"],
        "calculation_all_slots_at_5": calc_selected["all_slots"],
        "model_calls": runtime_metrics["model_calls"],
        "format_repairs": runtime_metrics["format_repairs"],
        "fallback_to_qwen": runtime_metrics["fallback_to_qwen"],
        "listwise_selector_effective": effectiveness,
        "production_switch_allowed": False,
        "prediction_sha256": prediction_sha,
        "prompt_sha256": sha256_text(PROMPT),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "next_gate": next_gate,
    }
    write_json(out_dir / "decision.json", decision)
    (out_dir / "README.md").write_text(f"# NF-OPT-21 R1 — Frozen Top10 Listwise Evidence Selector\n\nDevelopment-shadow, one-shot listwise selection over the frozen Qwen Top10. Gold is loaded only after prediction sealing.\n\nDecision: `{effectiveness}`; next gate: `{next_gate}`.\n", encoding="utf-8")
    print(json.dumps({"artifact_dir": str(out_dir), "decision": decision, "runtime": runtime_metrics, "validity": validity}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
