"""NF-OPT-23 R1: query-side financial requirement serialization.

The SADA-V1 Top100, Statement-Aware V1 candidate bytes, Qwen3-4B contract,
instruction, and scoring path are frozen.  The only experimental variable is
the deterministic query serialization derived from the pre-existing
question-only query-plan runtime.  Gold-dependent work is isolated after the
prediction seal.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


BASE_COMMIT = "b6a017e38bba49bd1c52145441556f5100dc5204"
OUT_NAME = "nf-opt-23-r1-query-requirement-serialization"
MODEL_ID = "Qwen/Qwen3-Reranker-4B"
REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
MAX_LENGTH = 8192
TOP100_SADA_REL = "nf-opt-24-r0-deep-supply-top100-admission/sada-v1-top100-predictions.jsonl.gz"
TOP100_SEAL_REL = "nf-opt-24-r0-deep-supply-top100-admission/sada-v1-prediction-seal.json"
TOP100_CURVE_REL = "nf-opt-24-r0-deep-supply-top100-admission/strict-recall-curve.json"
NF23_REL = "nf-opt-23-r0-statement-aware-evidence-unit"
QUERY_PLAN_REL = "pdf-retrieval-v4-gate-07/query-plan-predictions.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    return sha256_file(path)


def clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def percentile(values: list[int | float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_nf23(backend_root: Path) -> Any:
    return import_module(backend_root / "scripts/evaluation/run_nf_opt_23_r0_statement_aware_evidence_unit.py", "nf23_r0_contract")


def load_nf21(backend_root: Path) -> Any:
    return import_module(backend_root / "scripts/evaluation/run_nf_opt_21_r0_qwen_bm25_late_fusion.py", "nf21_metrics")


def extract_original_question(main_view: str) -> str:
    marker = "[QUESTION]"
    if marker not in main_view:
        raise RuntimeError("frozen query view has no [QUESTION] section")
    body = main_view.split(marker, 1)[1].lstrip("\n")
    if "\n\n[QUERY PLAN]" in body:
        body = body.split("\n\n[QUERY PLAN]", 1)[0]
    return body.rstrip("\n")


def requirement_from_plan(case_id: str, plan: dict[str, Any], original_question: str) -> dict[str, Any]:
    if clean(plan.get("raw_question")) != original_question:
        raise RuntimeError(f"original question mismatch for {case_id}")
    targets = [clean(value) for value in plan.get("metric_phrases", []) if clean(value)]
    periods = [clean(value) for value in plan.get("periods", []) if clean(value)]
    operation = clean(plan.get("operation")) or None
    slots: list[dict[str, str]] = []
    for slot in plan.get("operand_slots", []) or []:
        record: dict[str, str] = {}
        target = clean(slot.get("raw_metric_phrase"))
        period = clean(slot.get("period"))
        role = clean(slot.get("role"))
        if target:
            record["target"] = target
        if period:
            record["period"] = period
        if role:
            record["role"] = role
        if record:
            slots.append(record)
    return {
        "original_question": original_question,
        "target_terms": targets,
        "explicit_periods": periods,
        "operation": operation,
        "required_slots": slots,
    }


def serialize_requirement(requirement: dict[str, Any]) -> str:
    targets = requirement["target_terms"]
    periods = requirement["explicit_periods"]
    operation = requirement["operation"]
    slots = requirement["required_slots"]
    lines = ["[Question]", requirement["original_question"], "", "[Financial Evidence Requirements]"]
    lines.extend(["Target:", " | ".join(targets) if targets else "not explicitly resolved", ""])
    lines.append("Periods:")
    if periods:
        lines.extend(periods)
    else:
        lines.append("none explicitly resolved")
    lines.extend(["", "Operation:", operation or "not explicitly resolved", "", "Required Evidence Slots:"])
    if slots:
        for index, slot in enumerate(slots, 1):
            parts = [slot.get(key, "") for key in ("target", "period", "role") if slot.get(key)]
            lines.append(f"{index}. {' | '.join(parts)}")
    else:
        lines.append("no explicit multi-slot requirement resolved")
    return "\n".join(lines)


def requirement_category(requirement: dict[str, Any]) -> str:
    target = bool(requirement["target_terms"])
    period = bool(requirement["explicit_periods"])
    operation_or_slot = bool(requirement["operation"] or requirement["required_slots"])
    if target and period and operation_or_slot:
        return "target_period_operation_or_slot"
    if target and period:
        return "target_period"
    if period:
        return "period_only"
    if target:
        return "target_only"
    if requirement["operation"]:
        return "operation_only"
    return "original_question_only"


def build_requirements(qviews: dict[str, dict[str, Any]], backend_root: Path, out: Path) -> dict[str, Any]:
    plan_payload = read_json(backend_root / "artifacts/evaluation" / QUERY_PLAN_REL)
    plan_rows = {row["case_id"]: row.get("plan", {}) for row in plan_payload.get("plans", [])}
    if len(plan_rows) != 72:
        raise RuntimeError("frozen query-plan runtime must contain exactly 72 cases")
    requirements: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for case_id in sorted(qviews):
        question = extract_original_question(qviews[case_id]["main_query_view"])
        plan = plan_rows.get(case_id)
        if plan is None:
            raise RuntimeError(f"missing frozen query plan {case_id}")
        requirement = requirement_from_plan(case_id, plan, question)
        requirements[case_id] = requirement
        category = requirement_category(requirement)
        audit_rows.append({
            "case_id": case_id,
            "task_type": plan.get("task_type"),
            "query_type": "calculation" if plan.get("task_type") == "calculation_multi_operand" else "multi_evidence" if plan.get("requires_multiple_sources") else "direct_fact" if plan.get("task_type") in {"table_single_fact", "general_single_fact", "single_metric_multi_period", "multi_metric_comparison"} else "unresolved",
            "target_terms": bool(requirement["target_terms"]),
            "explicit_periods": bool(requirement["explicit_periods"]),
            "operation": bool(requirement["operation"]),
            "required_slots": bool(requirement["required_slots"]),
            "category": category,
        })
    allowed = {"original_question", "target_terms", "explicit_periods", "operation", "required_slots"}
    for requirement in requirements.values():
        if set(requirement) != allowed:
            raise RuntimeError("query requirement contains a prohibited field")
    write_json(out / "query-requirements.json", requirements)
    requirement_sha = sha256_file(out / "query-requirements.json")
    (out / "query-requirements.sha256").write_text(requirement_sha + "\n", encoding="utf-8")
    counts = {
        "queries": len(audit_rows),
        "target_terms": sum(row["target_terms"] for row in audit_rows),
        "explicit_periods": sum(row["explicit_periods"] for row in audit_rows),
        "operation": sum(row["operation"] for row in audit_rows),
        "required_slots": sum(row["required_slots"] for row in audit_rows),
        "at_least_one_structured_requirement": sum(any(row[key] for key in ("target_terms", "explicit_periods", "operation", "required_slots")) for row in audit_rows),
        "fully_populated": sum(all(row[key] for key in ("target_terms", "explicit_periods", "operation", "required_slots")) for row in audit_rows),
        "partially_populated": sum(any(row[key] for key in ("target_terms", "explicit_periods", "operation", "required_slots")) and not all(row[key] for key in ("target_terms", "explicit_periods", "operation", "required_slots")) for row in audit_rows),
        "original_question_only": sum(row["category"] == "original_question_only" for row in audit_rows),
        "by_category": {category: sum(row["category"] == category for row in audit_rows) for category in ("target_period_operation_or_slot", "target_period", "period_only", "target_only", "operation_only", "original_question_only")},
        "by_query_type": {query_type: sum(row["query_type"] == query_type for row in audit_rows) for query_type in ("direct_fact", "multi_evidence", "calculation", "unresolved")},
        "gold_reads": 0,
    }
    write_json(out / "query-requirement-audit.json", {**counts, "rows": audit_rows})
    write_json(out / "query-runtime-audit.json", {
        "components": [
            {"component": "pdf-retrieval-v4-gate-07/query-plan-predictions.json", "existing_before_gate": True, "input": "original question only", "output_fields": ["raw_question", "metric_phrases", "periods", "operation", "operand_slots", "task_type"], "gold_access": False, "candidate_access": False, "deterministic": True, "selected_for_R1": True},
            {"component": "src.retrieval_v3.query_router.route_question", "existing_before_gate": True, "input": "question text", "output_fields": ["metric_phrases", "periods", "operation", "requires_multiple_sources"], "gold_access": False, "candidate_access": False, "deterministic": True, "selected_for_R1": True},
        ],
        "query_plan_source_sha256": sha256_file(backend_root / "artifacts/evaluation" / QUERY_PLAN_REL),
        "gold_reads": 0,
    })
    write_json(out / "query-requirement-contract.json", {
        "gate": "NF-OPT-23-R1",
        "method": "Query Requirement Serialization V1",
        "allowed_fields": sorted(allowed),
        "template": "fixed_question_and_financial_evidence_requirements",
        "query_only": True,
        "candidate_independent": True,
        "gold_independent": True,
        "fail_closed": True,
        "no_question_id_lookup": True,
        "query_requirements_sha256": requirement_sha,
        "gold_reads_during_generation": 0,
    })
    return {"requirements": requirements, "audit": counts, "sha256": requirement_sha}


def load_inputs(backend_root: Path, nf23: Any, out: Path) -> dict[str, Any]:
    root = backend_root / "artifacts/evaluation"
    inputs = nf23.load_baseline_inputs(root)
    nf24 = import_module(backend_root / "scripts/evaluation/run_nf_opt_24_r0_deep_supply_admission.py", "nf24_contract")
    sada_path = root / TOP100_SADA_REL
    seal_path = root / TOP100_SEAL_REL
    curve_path = root / TOP100_CURVE_REL
    sada_sha = sha256_file(sada_path)
    seal = read_json(seal_path)
    if seal.get("prediction_sha256") != sada_sha or seal.get("queries") != 72 or seal.get("top100_candidates_per_query") != 100:
        raise RuntimeError("sealed SADA Top100 contract mismatch")
    curve = read_json(curve_path)
    if curve.get("sada", {}).get("@100", {}).get("hits") != 78:
        raise RuntimeError("frozen SADA Top100 strict supply is not 78/80")
    sada_rows = read_gzip_jsonl(sada_path)
    if len(sada_rows) != 72:
        raise RuntimeError("expected 72 SADA rows")
    sada = {row["case_id"]: sorted(row["ranked_candidates"], key=lambda item: int(item["post_rerank_rank"])) for row in sada_rows}
    if set(sada) != set(inputs["qviews"]):
        raise RuntimeError("SADA query identity mismatch")
    deep_inputs = nf24.load_contract_inputs(backend_root, nf23)
    deep_units, deep_unit_meta = nf24.build_units(backend_root, nf23, deep_inputs)
    deep_keys = {item["candidate_key"] for values in deep_inputs["pool"].values() for item in values}
    mismatches = 0
    for case_id, ranked in sada.items():
        if len(ranked) != 100 or [int(item["post_rerank_rank"]) for item in ranked] != list(range(1, 101)):
            raise RuntimeError(f"SADA rank/count mismatch {case_id}")
        if not {item["candidate_key"] for item in ranked} <= deep_keys:
            mismatches += 1
    if mismatches:
        raise RuntimeError(f"SADA candidate mismatch count={mismatches}")
    write_json(out / "frozen-sada-contract.json", {
        "gate": "NF-OPT-23-R1", "source": TOP100_SADA_REL, "prediction_sha256": sada_sha, "prediction_seal_sha256": sha256_file(seal_path), "candidate_budget": 100, "queries": 72, "strict_top100_supply": "78/80", "candidate_mismatch": 0, "admission_rerun": False, "gold_reads_before_experimental_prediction_seal": 0,
    })
    write_json(out / "frozen-candidate-contract.json", {
        "candidate_source": "frozen_sada_v1_top100", "candidate_budget": 100, "queries": 72, "pairs": 7200, "candidate_identity_unchanged": True, "sada_prediction_sha256": sada_sha, "sada_top100_strict_supply": "78/80", "candidate_mismatch": 0, "statement_aware_candidate_bytes_frozen": True, "gold_reads": 0,
    })
    return {"inputs": inputs, "sada": sada, "sada_sha": sada_sha, "curve": curve, "deep_inputs": deep_inputs, "units": deep_units, "unit_meta": deep_unit_meta}


def build_candidate_rows(backend_root: Path, nf23: Any, loaded: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = backend_root / "artifacts/evaluation"
    units = loaded["units"]
    expected_manifest = nf23.read_gzip_jsonl(root / "nf-opt-24-r0-deep-supply-top100-admission/serialization-manifest.jsonl.gz")
    expected = {(row["case_id"], row["candidate_key"]): row["serialization_sha256"] for row in expected_manifest}
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    mismatch = 0
    for case_id in sorted(loaded["sada"]):
        candidates: list[dict[str, Any]] = []
        for item in loaded["sada"][case_id]:
            key = item["candidate_key"]
            unit = units.get(key)
            if unit is None or expected.get((case_id, key)) != unit["serialization_sha256"] or item.get("serialization_sha256") != unit["serialization_sha256"]:
                mismatch += 1
            candidate = {
                "candidate_key": key,
                "original_sada_rank": int(item["post_rerank_rank"]),
                "original_deep_rank": int(item.get("original_deep_rank", 10**9)),
                "statement_serialization": unit["serialization"] if unit else "",
                "statement_serialization_sha256": unit["serialization_sha256"] if unit else None,
            }
            candidates.append(candidate)
            manifest_rows.append({"case_id": case_id, "candidate_key": key, "original_sada_rank": candidate["original_sada_rank"], "original_deep_rank": candidate["original_deep_rank"], "serialization_sha256": candidate["statement_serialization_sha256"]})
        rows.append({"case_id": case_id, "candidates": candidates})
    if mismatch:
        raise RuntimeError(f"Statement-Aware candidate serialization mismatch count={mismatch}")
    serialization_sha = write_gzip_jsonl(out / "serialization-manifest.jsonl.gz", manifest_rows)
    write_json(out / "serialization-seal.json", {"gate": "NF-OPT-23-R1", "records": len(manifest_rows), "queries": 72, "pairs": 7200, "serialization_sha256": serialization_sha, "candidate_serialization_match": True, "nf23_manifest_overlap_checked": len(expected_manifest), "mismatches": 0, "gold_reads": 0, "sealed": True})
    example_keys = sorted({item["candidate_key"] for row in loaded["sada"].values() for item in row}, key=lambda item: hashlib.sha256(item.encode("utf-8")).hexdigest())[:20]
    write_json(out / "representation-examples.json", {"source": "frozen_nf_opt_23_r0_statement_aware_units", "selection": "sha256(candidate_id) ascending", "sample_size": len(example_keys), "examples": [{"candidate_id": key, "statement_aware_serialization_sha256": units[key]["serialization_sha256"]} for key in example_keys]})
    return rows, {"serialization_sha256": serialization_sha, "mismatches": 0, "candidate_count": len(manifest_rows)}


def frozen_reranker_contract(backend_root: Path, nf23: Any, out: Path) -> dict[str, Any]:
    contract = nf23.load_internal_contract(backend_root)
    if contract["model_id"] != MODEL_ID or contract["revision"] != REVISION or contract["max_length"] != MAX_LENGTH:
        raise RuntimeError("frozen Qwen3-4B contract mismatch")
    write_json(out / "frozen-reranker-contract.json", {"gate": "NF-OPT-23-R1", "model": contract["model_id"], "model_revision": contract["revision"], "dtype": contract["dtype"], "max_length": contract["max_length"], "batch_size": contract["batch_size"], "scoring": contract["scoring"], "instruction_sha256": contract["instruction_sha256"], "instruction_unchanged": True, "query_candidate_contract": "same Statement-Aware V1 candidate bytes; query-only variable", "retrieval_rerun": False, "admission_rerun": False})
    return contract


def score_experiment(rows: list[dict[str, Any]], instruction: str, revision: str, max_length: int, out_path: Path) -> dict[str, Any]:
    import importlib.machinery
    import types

    if "sklearn" not in sys.modules:
        sklearn_stub = types.ModuleType("sklearn")
        metrics_stub = types.ModuleType("sklearn.metrics")
        sklearn_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
        metrics_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)
        metrics_stub.roc_curve = lambda *args, **kwargs: ([], [], [])
        sklearn_stub.metrics = metrics_stub
        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.metrics"] = metrics_stub
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.pdf_retrieval_v4.qwen3_reranker import build_input_ids, score_batch  # type: ignore

    snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots" / revision
    if not snapshot.is_dir() or not torch.cuda.is_available():
        raise RuntimeError("exact_4b_snapshot_not_cached_or_cuda_unavailable")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), padding_side="left", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), torch_dtype=torch.bfloat16, local_files_only=True).to("cuda:0").eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    output_rows: list[dict[str, Any]] = []
    token_lengths: list[int] = []
    original_lengths: list[int] = []
    pairs = 0
    truncated = 0
    nonfinite = 0
    with torch.no_grad():
        for source in sorted(rows, key=lambda item: item["case_id"]):
            requirement = source["query_requirement"]
            ranked: list[dict[str, Any]] = []
            for candidate in source["candidates"]:
                ids, audit = build_input_ids(tokenizer, instruction, requirement, candidate["statement_serialization"], max_length)
                score = score_batch(model, tokenizer, [ids])[0]
                value = float(score["reranker_score"])
                nonfinite += int(not math.isfinite(value))
                truncated += int(bool(audit["truncated"]))
                token_lengths.append(int(audit["final_token_count"]))
                original_lengths.append(int(audit["original_token_count"]))
                ranked.append({"candidate_key": candidate["candidate_key"], "qwen_statement_score": value, "yes_logit": float(score["yes_logit"]), "no_logit": float(score["no_logit"]), "original_sada_rank": candidate["original_sada_rank"], "original_deep_rank": candidate["original_deep_rank"], "serialization_sha256": candidate["statement_serialization_sha256"], "query_requirement_sha256": sha256_text(requirement), "truncated": bool(audit["truncated"]), "final_token_count": int(audit["final_token_count"]), "original_token_count": int(audit["original_token_count"])})
                pairs += 1
            ranked.sort(key=lambda item: (-item["qwen_statement_score"], item["original_sada_rank"], item["original_deep_rank"], item["candidate_key"]))
            for rank, item in enumerate(ranked, 1):
                item["post_rerank_rank"] = rank
            output_rows.append({"case_id": source["case_id"], "input_candidate_count": len(ranked), "ranked_candidates": ranked})
    elapsed = max(time.time() - started, 1e-9)
    prediction_sha = write_gzip_jsonl(out_path, output_rows)
    return {"model_execution": True, "model": MODEL_ID, "model_revision": revision, "pairs": pairs, "queries": len(output_rows), "elapsed_seconds": elapsed, "pairs_per_second": pairs / elapsed, "peak_vram_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)), "truncated": truncated, "oom": 0, "nonfinite": nonfinite, "token_p50": statistics.median(token_lengths), "token_p90": percentile(token_lengths, .90), "token_p95": percentile(token_lengths, .95), "token_p99": percentile(token_lengths, .99), "token_max": max(token_lengths), "dtype": "bfloat16", "batch_size": 1, "max_length": max_length, "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "prediction_sha256": prediction_sha, "gold_reads_before_prediction_seal": 0}


def rank_map(items: list[dict[str, Any]]) -> dict[str, int]:
    return {item["candidate_key"]: index + 1 for index, item in enumerate(items)}


def prepare_runtime_rows(candidate_rows: list[dict[str, Any]], requirements: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"case_id": row["case_id"], "query_requirement": serialize_requirement(requirements[row["case_id"]]), "candidates": row["candidates"]} for row in candidate_rows]


def load_failure_records(root: Path) -> list[dict[str, Any]]:
    payload = read_json(root / "nf-opt-20-r0-pointwise-discrimination-audit/failure-taxonomy.json")
    return payload.get("records", []) if isinstance(payload, dict) else payload


def postseal(backend_root: Path, out: Path, loaded: dict[str, Any], requirements_data: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    root = backend_root / "artifacts/evaluation"
    nf21 = load_nf21(backend_root)
    strict_rows = read_jsonl(root / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl")
    registry = read_gzip_jsonl(root / "pdf-retrieval-v4-gate-08-r8-se1-p0/candidate-semantic-fact-registry.jsonl.gz")
    targets = nf21.load_targets(root / "pdf-retrieval-v4-gate-08-r8-se1/gold-semantic-targets.jsonl")
    facts = nf21.build_case_facts(registry)
    calc_rows = read_json(root / "nf-opt-19-r0-setwise-ranking-audit/calculation-slot-coverage.json")["cases"]
    prediction_rows = read_gzip_jsonl(out / "predictions.jsonl.gz")
    experiment = {row["case_id"]: sorted(row["ranked_candidates"], key=lambda item: int(item["post_rerank_rank"])) for row in prediction_rows}
    baseline = {case: sorted(rows, key=lambda item: int(item["post_rerank_rank"])) for case, rows in loaded["sada"].items()}
    strict = {"sada_original_query": nf21.strict_metrics(strict_rows, baseline), "query_requirement": nf21.strict_metrics(strict_rows, experiment)}
    semantic = {"sada_original_query": nf21.semantic_metrics(strict_rows, baseline, facts, targets), "query_requirement": nf21.semantic_metrics(strict_rows, experiment, facts, targets)}
    write_json(out / "strict-metrics.json", {"strict_sources": len(strict_rows), **strict})
    write_json(out / "semantic-metrics.json", semantic)
    movement_rows: list[dict[str, Any]] = []
    rescued = damaged = 0
    for binding in strict_rows:
        case_id, key = binding["case_id"], binding["candidate_key"]
        before = rank_map(baseline[case_id]).get(key, 10**9)
        after = rank_map(experiment[case_id]).get(key, 10**9)
        outcome = "unchanged"
        if before > 5 and after <= 5:
            rescued += 1
            outcome = "rescued"
        elif before <= 5 and after > 5:
            damaged += 1
            outcome = "damaged"
        movement_rows.append({"case_id": case_id, "source_index": binding.get("source_index", 0), "candidate_key": key, "baseline_rank": before, "experimental_rank": after, "rank_delta": before - after, "outcome": outcome})
    write_json(out / "rank-movement.json", {"baseline": "SADA-SA-OriginalQuery", "rescued": rescued, "damaged": damaged, "net": rescued - damaged, "rows": movement_rows})
    baseline_rank = {case: rank_map(rows) for case, rows in baseline.items()}
    experiment_rank = {case: rank_map(rows) for case, rows in experiment.items()}
    top10_sources = [row for row in movement_rows if 6 <= row["baseline_rank"] <= 10]
    write_json(out / "top10-to-top5-analysis.json", {
        "baseline_rank6_10_sources": len(top10_sources),
        "promoted_to_top5": sum(row["experimental_rank"] <= 5 for row in top10_sources),
        "remain_rank6_10": sum(6 <= row["experimental_rank"] <= 10 for row in top10_sources),
        "worsened_below10": sum(row["experimental_rank"] > 10 for row in top10_sources),
        "rows": top10_sources,
        "deeper_rank_movement": {band: {"count": sum(6 <= row["baseline_rank"] <= 10 for row in movement_rows) if band == "6-10" else sum(11 <= row["baseline_rank"] <= 20 for row in movement_rows) if band == "11-20" else sum(21 <= row["baseline_rank"] <= 50 for row in movement_rows) if band == "21-50" else sum(51 <= row["baseline_rank"] <= 100 for row in movement_rows), "promoted_to_top5": sum(row["experimental_rank"] <= 5 and ((6 <= row["baseline_rank"] <= 10) if band == "6-10" else (11 <= row["baseline_rank"] <= 20) if band == "11-20" else (21 <= row["baseline_rank"] <= 50) if band == "21-50" else (51 <= row["baseline_rank"] <= 100)) for row in movement_rows)} for band in ("6-10", "11-20", "21-50", "51-100")},
    })
    requirements = requirements_data["requirements"]
    categories = defaultdict(list)
    for case_id, requirement in requirements.items():
        categories[requirement_category(requirement)].append(case_id)
    cohort_output: dict[str, Any] = {}
    for category, case_ids in sorted(categories.items()):
        bindings = [row for row in strict_rows if row["case_id"] in set(case_ids)]
        cohort_output[category] = {"queries": len(case_ids), "bindings": len(bindings), "baseline_r5": nf21.strict_metrics(bindings, baseline)["@5"], "experimental_r5": nf21.strict_metrics(bindings, experiment)["@5"], "rank_improved": sum(baseline_rank[row["case_id"]].get(row["candidate_key"], 10**9) > experiment_rank[row["case_id"]].get(row["candidate_key"], 10**9) for row in bindings), "rank_worsened": sum(baseline_rank[row["case_id"]].get(row["candidate_key"], 10**9) < experiment_rank[row["case_id"]].get(row["candidate_key"], 10**9) for row in bindings)}
    write_json(out / "requirement-coverage-cohort.json", cohort_output)
    bindings_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_rows:
        bindings_by_case[row["case_id"]].append(row)
    multi_records = read_json(root / "nf-opt-19-r0-setwise-ranking-audit/diversity-ceiling.json")["records"]
    multi_cases = [row["case_id"] for row in multi_records if row.get("multi_evidence")]
    multi = {name: {f"@{k}": nf21.coverage(bindings_by_case, multi_cases, ranked, k) for k in (5, 10)} for name, ranked in (("sada_original_query", baseline), ("query_requirement", experiment))}
    write_json(out / "multi-evidence-analysis.json", {"denominator": len(multi_cases), "variants": multi})
    calc = {name: {f"@{k}": nf21.calc_coverage(calc_rows, ranked, facts, k) for k in (5, 10)} for name, ranked in (("sada_original_query", baseline), ("query_requirement", experiment))}
    write_json(out / "calculation-slot-analysis.json", {"denominator": len(calc_rows), "variants": calc})
    failure_records = load_failure_records(root)
    failure_outputs: dict[str, Any] = {}
    for tag in ("wrong_period", "same_metric_wrong_row", "multi_slot_competition", "calculation_operand_competition"):
        ids = {(row.get("case_id"), row.get("source_index", 0)) for row in failure_records if tag in (row.get("tags") or [])}
        rows = [row for row in movement_rows if (row["case_id"], row["source_index"]) in ids]
        failure_outputs[tag] = {"cases": len(rows), "rescued": sum(row["outcome"] == "rescued" for row in rows), "improved": sum(row["rank_delta"] > 0 for row in rows), "worsened": sum(row["rank_delta"] < 0 for row in rows), "unchanged": sum(row["rank_delta"] == 0 for row in rows)}
    write_json(out / "period-cohort-analysis.json", failure_outputs["wrong_period"])
    write_json(out / "same-metric-row-analysis.json", failure_outputs["same_metric_wrong_row"])
    write_json(out / "multi-slot-analysis.json", {**failure_outputs["multi_slot_competition"], "required_slots_extracted": {"true": sum(bool(requirements[row["case_id"]]["required_slots"]) for row in movement_rows if (row["case_id"], row["source_index"]) in {(x.get("case_id"), x.get("source_index", 0)) for x in failure_records if "multi_slot_competition" in (x.get("tags") or [])}), "false": 0}})
    write_json(out / "calculation-operand-analysis.json", failure_outputs["calculation_operand_competition"])
    near_ids = {(row.get("case_id"), row.get("source_index", 0)) for row in failure_records if row.get("cohort") == "near_boundary"}
    clear_ids = {(row.get("case_id"), row.get("source_index", 0)) for row in failure_records if row.get("cohort") == "clear_loss"}
    def summarize(ids: set[tuple[Any, Any]]) -> dict[str, Any]:
        rows = [row for row in movement_rows if (row["case_id"], row["source_index"]) in ids]
        return {"total": len(rows), "baseline_r5": sum(row["baseline_rank"] <= 5 for row in rows), "experimental_r5": sum(row["experimental_rank"] <= 5 for row in rows), "rescued": sum(row["outcome"] == "rescued" for row in rows), "damaged": sum(row["outcome"] == "damaged" for row in rows), "mean_rank_delta": statistics.mean([row["rank_delta"] for row in rows]) if rows else None}
    write_json(out / "near-boundary-clear-loss-analysis.json", {"near_boundary": summarize(near_ids), "clear_loss": summarize(clear_ids)})
    no_answer_path = root / "pdf-retrieval-v4-gate-07/no-answer-boundary-audit.json"
    no_answer_cases = [row["case_id"] for row in read_json(no_answer_path).get("records", [])]
    no_answer_rows = []
    for case_id in no_answer_cases:
        before = baseline[case_id]
        after = experiment[case_id]
        before_scores = [float(item.get("qwen_statement_score", 0.0)) for item in before]
        after_scores = [float(item.get("qwen_statement_score", 0.0)) for item in after]
        no_answer_rows.append({"case_id": case_id, "baseline_top1_score": before_scores[0], "experimental_top1_score": after_scores[0], "baseline_top1_minus_top5": before_scores[0] - before_scores[4], "experimental_top1_minus_top5": after_scores[0] - after_scores[4], "baseline_top1_minus_mean_top10": before_scores[0] - statistics.mean(before_scores[:10]), "experimental_top1_minus_mean_top10": after_scores[0] - statistics.mean(after_scores[:10])})
    write_json(out / "no-answer-analysis.json", {"queries": len(no_answer_rows), "threshold_modified": False, "records": no_answer_rows})
    strict_experiment = strict["query_requirement"]["@5"]["hits"]
    strict_r10 = strict["query_requirement"]["@10"]["hits"]
    strict_r20 = strict["query_requirement"]["@20"]["hits"]
    strict_r50 = strict["query_requirement"]["@50"]["hits"]
    strict_r100 = strict["query_requirement"]["@100"]["hits"]
    semantic_r5 = semantic["query_requirement"]["@5"]["hits"]
    multi_all5 = multi["query_requirement"]["@5"]["all"]
    calc_all5 = calc["query_requirement"]["@5"]["all_slots"]
    if strict_experiment >= 53 and rescued - damaged >= 7 and damaged <= 4 and strict_r10 >= 60 and strict_r20 >= 68 and strict_r50 >= 72 and strict_r100 == 78 and semantic_r5 >= semantic["sada_original_query"]["@5"]["hits"] and multi_all5 >= multi["sada_original_query"]["@5"]["all"] and calc_all5 >= calc["sada_original_query"]["@5"]["all_slots"]:
        effectiveness, next_gate = True, "internal_retrieval_shadow_freeze_review"
    elif 50 <= strict_experiment <= 52 and rescued > damaged and strict_r10 >= 59 and strict_r100 == 78 and semantic_r5 >= semantic["sada_original_query"]["@5"]["hits"] and multi_all5 >= multi["sada_original_query"]["@5"]["all"] and calc_all5 >= calc["sada_original_query"]["@5"]["all_slots"]:
        effectiveness, next_gate = "marginal", "reranker_instruction_calibration_r0"
    else:
        effectiveness, next_gate = False, "reranker_instruction_calibration_r0"
    decision = {"gate": "NF-OPT-23-R1", "evaluation_role": "development_shadow_query_requirement_ablation", "fresh_blind_evaluation": False, "retrieval_rerun": False, "admission_rerun": False, "training": False, "sada_top100_hits": 78, "model": MODEL_ID, "model_revision": REVISION, "statement_aware_contract_unchanged": True, "reranker_instruction_unchanged": True, "query_requirement_signal_sufficient": requirements_data["audit"]["at_least_one_structured_requirement"] >= 60, "queries_with_target_terms": requirements_data["audit"]["target_terms"], "queries_with_periods": requirements_data["audit"]["explicit_periods"], "queries_with_operation": requirements_data["audit"]["operation"], "queries_with_required_slots": requirements_data["audit"]["required_slots"], "baseline_r5_hits": 46, "baseline_r10_hits": 60, "baseline_r20_hits": 69, "baseline_r50_hits": 73, "baseline_r100_hits": 78, "query_requirement_r5_hits": strict_experiment, "query_requirement_r10_hits": strict_r10, "query_requirement_r20_hits": strict_r20, "query_requirement_r50_hits": strict_r50, "query_requirement_r100_hits": strict_r100, "rescued": rescued, "damaged": damaged, "net": rescued - damaged, "top10_to_top5_promoted": sum(row["experimental_rank"] <= 5 for row in top10_sources), "semantic_r5_hits": semantic_r5, "multi_evidence_all_at_5": multi_all5, "calculation_all_slots_at_5": calc_all5, "query_requirement_serialization_effective": effectiveness, "selected_internal_shadow_method": "sada_statement_aware_query_requirement_v1" if effectiveness is not False else "sada_statement_aware_v1", "production_switch_allowed": False, "gold_reads_before_experimental_prediction_seal": 0, "next_gate": next_gate}
    write_json(out / "decision.json", decision)
    return {"strict": strict, "semantic": semantic, "movement": {"rescued": rescued, "damaged": damaged, "net": rescued - damaged}, "multi": multi, "calc": calc, "decision": decision, "runtime": runtime}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postseal", action="store_true")
    args = parser.parse_args()
    backend_root = Path(__file__).resolve().parents[2]
    out = backend_root / "artifacts/evaluation" / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    nf23 = load_nf23(backend_root)
    loaded = load_inputs(backend_root, nf23, out)
    contract = frozen_reranker_contract(backend_root, nf23, out)
    requirements_data = build_requirements(loaded["inputs"]["qviews"], backend_root, out)
    if requirements_data["audit"]["at_least_one_structured_requirement"] < 60:
        write_json(out / "query-requirement-audit.json", {**read_json(out / "query-requirement-audit.json"), "query_requirement_signal_sufficient": False})
    candidate_rows, serialization_meta = build_candidate_rows(backend_root, nf23, loaded, out)
    runtime_path = out / "runtime-metrics.json"
    prediction_path = out / "predictions.jsonl.gz"
    if args.postseal:
        runtime = read_json(runtime_path)
    else:
        runtime_rows = prepare_runtime_rows(candidate_rows, requirements_data["requirements"])
        write_json(out / "input-manifest.json", {"queries": 72, "pairs": 7200, "candidate_budget": 100, "query_requirement_sha256": requirements_data["sha256"], "serialization_sha256": serialization_meta["serialization_sha256"], "candidate_identity_mismatch": 0, "gold_reads": 0})
        runtime = score_experiment(runtime_rows, contract["instruction"], contract["revision"], contract["max_length"], prediction_path)
        write_json(runtime_path, {**runtime, "query_requirement_sha256": requirements_data["sha256"], "candidate_serialization_sha256": serialization_meta["serialization_sha256"]})
        prediction_rows = read_gzip_jsonl(prediction_path)
        if len(prediction_rows) != 72 or sum(len(row["ranked_candidates"]) for row in prediction_rows) != 7200:
            raise RuntimeError("experimental prediction completeness contract failed")
        for row in prediction_rows:
            if len(row["ranked_candidates"]) != 100 or len({item["candidate_key"] for item in row["ranked_candidates"]}) != 100 or any(not math.isfinite(float(item["qwen_statement_score"])) for item in row["ranked_candidates"]):
                raise RuntimeError(f"experimental candidate/nonfinite contract failed {row['case_id']}")
        prediction_sha = sha256_file(prediction_path)
        write_json(out / "prediction-seal.json", {"gate": "NF-OPT-23-R1", "prediction_sha256": prediction_sha, "queries": 72, "pairs": 7200, "candidate_identity_mismatch": 0, "candidate_budget": 100, "query_requirement_sha256": requirements_data["sha256"], "statement_aware_serialization_sha256": serialization_meta["serialization_sha256"], "gold_reads_before_prediction_seal": 0, "sealed": True})
    if not prediction_path.exists():
        raise RuntimeError("prediction seal missing")
    result = postseal(backend_root, out, loaded, requirements_data, runtime)
    write_json(out / "runtime-metrics.json", {**runtime, "query_requirement_sha256": requirements_data["sha256"], "candidate_serialization_sha256": serialization_meta["serialization_sha256"]})
    (out / "README.md").write_text("# NF-OPT-23 R1\n\nQuery Requirement Serialization V1 is a query-only, deterministic ablation over the frozen SADA-V1 Top100 and Statement-Aware Evidence Unit V1. The Qwen3-4B model, revision, instruction, scoring, candidate identity, and candidate bytes are frozen. Gold is loaded only after prediction-seal.json. No retrieval, admission rerun, query-template sweep, or production switch is performed.\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "runtime": runtime, "strict": result["strict"], "decision": result["decision"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
