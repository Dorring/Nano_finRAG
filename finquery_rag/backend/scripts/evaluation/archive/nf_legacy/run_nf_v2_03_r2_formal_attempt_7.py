#!/usr/bin/env python3
"""NF-V2-03 R2 formal Attempt 7 with the frozen Binder Prompt R2."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402


OUT = ROOT / "artifacts/evaluation/nf-v2-03-r2-semantic-selection"
FORMAL_OUT = OUT / "formal-attempt-7"
SYNTHETIC = OUT / "synthetic-semantic-suite.json"
MODEL = "qwen3.7-plus"
BASE_COMMIT = "ccf7668ba6a363dfd9b0987c62b02082b3821fd1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_false_binding(scored: dict) -> tuple[int, int]:
    direct_bindable = set(scored["direct_bindable_ids"])
    direct_rows = [row for row in scored["strict_rows"] if row["intent"] == "DIRECT_FACT"]
    false_bindable = sum(int(row["question_id"] in direct_bindable and row["status"] == "BOUND" and not row["strict_complete"]) for row in direct_rows)
    return false_bindable, scored["false_binding_queries"] - false_bindable


def main() -> int:
    synthetic = read_json(SYNTHETIC)
    if synthetic.get("pass") is not True:
        decision = {"gate": "NF-V2-03-R2", "formal_attempt_7": "not_run", "reason": "synthetic_semantic_suite_failed", "production_default": "V1", "production_switch_allowed": False}
        write_json(OUT / "decision.json", decision)
        print(json.dumps(decision, sort_keys=True))
        return 3
    prompt_path = OUT / "binder-prompt-r2.txt"
    prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    prompt = prompt_path.read_text(encoding="utf-8")
    if (OUT / "binder-prompt-r2.sha256").read_text(encoding="utf-8").strip() != prompt_hash:
        raise RuntimeError("Prompt R2 hash mismatch")
    config = r1d.legacy.load_config()
    frozen = r1d.load_r1c_frozen_inputs()
    FORMAL_OUT.mkdir(parents=True, exist_ok=True)
    write_json(FORMAL_OUT / "config.json", {
        "gate": "NF-V2-03-R2",
        "base_commit": BASE_COMMIT,
        "provider": "Alibaba Bailian",
        "provider_role": "evidence_binder",
        "model": MODEL,
        "model_role": "strong_general_llm",
        "thinking": False,
        "temperature": 0.0,
        "max_retries": 0,
        "http_timeout_seconds": 180,
        "prompt_sha256": prompt_hash,
        "provider_schema_contract_sha256": r1d.provider_schema_contract_sha(),
        "binder_fact_view_sha256": r1d.sha256_file(r1d.R1C_OUT / "binder-fact-view-contract.json"),
        "supervisor_prediction_sha256": frozen["plan_sha256"],
        "financial_fact_supply_sha256": frozen["facts_sha256"],
        "top20_order_sha256": frozen["top20_order_sha256"],
        "supply_ceiling": {"direct": "27/56", "calculation": "6/11", "multi_evidence": "0/5"},
        "gold_reads_before_prediction_seal": 0,
        "production_default": "V1",
        "production_switch_allowed": False,
    })
    predictions, runtime = r1d.run_formal({**config, "base_url": os.getenv("V2_SUPERVISOR_BASE_URL", "").strip()}, frozen, system_prompt=prompt)
    prediction_path = FORMAL_OUT / "predictions.jsonl.gz"
    r1d.write_jsonl_gz(prediction_path, predictions)
    prediction_sha = sha256_file(prediction_path)
    seal = {
        "gate": "NF-V2-03-R2",
        "sealed": True,
        "predictions_written": len(predictions),
        "questions_expected": 72,
        "provider_model_calls": sum(int(not row["skipped_no_fact_supply"]) for row in predictions),
        "max_calls_per_query": 1,
        "gold_reads_before_prediction_seal": 0,
        "prediction_sha256": prediction_sha,
        "prompt_sha256": prompt_hash,
        "sealed_before_gold": True,
    }
    write_json(FORMAL_OUT / "prediction-seal.json", seal)
    if sha256_file(prediction_path) != prediction_sha:
        raise RuntimeError("Attempt 7 prediction seal verification failed")

    # Gold is opened only after the prediction seal.
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in r1d.LABELS.read_text(encoding="utf-8").splitlines()) if row}
    scored = r1d.score_supply_conditioned(frozen, predictions, labels)
    eligible = [row for row in predictions if not row["skipped_no_fact_supply"]]
    structural = {
        "questions": 72,
        "model_required_queries": len(eligible),
        "provider_responses": sum(int(row["provider_response_success"]) for row in eligible),
        "structured_output": sum(int(row["structured_output_success"]) for row in eligible),
        "dto_valid": sum(int(row["dto_valid"]) for row in eligible),
        "adapter_valid": sum(int(row["adapter_valid"]) for row in eligible),
        "binding_validator_pass": sum(int(row["binding_validator_pass"]) for row in predictions),
        "unknown_slots": sum(row["unknown_slot"] for row in predictions),
        "unknown_facts": sum(row["unknown_fact"] for row in predictions),
        "duplicate_handles": sum(row["duplicate_handle"] for row in predictions),
        "status_violations": sum(row["status_violation"] for row in predictions),
        "cardinality_violations": sum(row["cardinality_violation"] for row in predictions),
        "calculation_leakage": sum(row["calculation_leakage"] for row in predictions),
        "gold_reads_before_prediction_seal": 0,
    }
    write_json(FORMAL_OUT / "structural-metrics.json", structural)
    write_json(FORMAL_OUT / "direct-semantic-metrics.json", {"supply_ceiling": "27/56", **scored["direct"]})
    write_json(FORMAL_OUT / "calculation-semantic-metrics.json", {"supply_ceiling": "6/11", **scored["calculation"]})
    write_json(FORMAL_OUT / "false-binding-analysis.json", {"false_binding_queries": scored["false_binding_queries"], "false_binding_slots": scored["false_binding_slots"], "hard_safety_target": 0, "direct_false_binding_on_bindable": split_false_binding(scored)[0], "false_binding_on_unbindable": split_false_binding(scored)[1]})
    metadata_rows = [row["metadata"] for row in predictions if row.get("metadata")]
    latencies = [float(row.get("latency_ms") or 0.0) for row in metadata_rows]
    write_json(FORMAL_OUT / "latency-token-cost.json", {
        "provider_calls": len(eligible),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in metadata_rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in metadata_rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in metadata_rows),
        "average_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": r1d.percentile(latencies),
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "formal_wall_time_ms": runtime["formal_wall_time_ms"],
        "estimated_cost": "not_configured",
    })
    false_bindable, false_unbindable = split_false_binding(scored)
    direct_quality = scored["direct"]["success_given_bindable_percent"]
    calc_quality = scored["calculation"]["success_given_bindable_percent"]
    structural_healthy = bool(
        structural["dto_valid"] >= 0.98 * len(eligible)
        and structural["adapter_valid"] >= 0.98 * len(eligible)
        and structural["binding_validator_pass"] >= 0.98 * 72
        and sum(structural[key] for key in ("unknown_slots", "unknown_facts", "duplicate_handles", "status_violations", "cardinality_violations", "calculation_leakage")) == 0
    )
    strong = bool(structural_healthy and scored["direct"]["strict_correct_given_bindable"] >= 24 and scored["calculation"]["strict_correct_given_bindable"] >= 5 and scored["false_binding_queries"] <= 1)
    partial = bool(not strong and structural_healthy and scored["direct"]["strict_correct_given_bindable"] >= 20 and scored["calculation"]["strict_correct_given_bindable"] >= 4 and scored["false_binding_queries"] <= 3)
    decision = {
        "gate": "NF-V2-03-R2",
        "base_commit": BASE_COMMIT,
        "binder_model": MODEL,
        "formal_attempt_7": "executed",
        "formal_run_complete": True,
        "gold_reads_before_prediction_seal": 0,
        "prediction_seal": "pass",
        "structural_healthy": structural_healthy,
        "provider_responses": f"{structural['provider_responses']}/{len(eligible)}",
        "dto_valid": f"{structural['dto_valid']}/{len(eligible)}",
        "adapter_valid": f"{structural['adapter_valid']}/{len(eligible)}",
        "binding_validator_pass": f"{structural['binding_validator_pass']}/72",
        "direct_supply_ceiling": "27/56",
        "direct_strict_complete": f"{scored['direct']['strict_complete']}/56",
        "direct_success_given_bindable": direct_quality,
        "calculation_supply_ceiling": "6/11",
        "calculation_all_operand_strict": f"{scored['calculation']['strict_complete']}/11",
        "calculation_success_given_bindable": calc_quality,
        "multi_evidence_supply_ceiling": "0/5",
        "multi_evidence_semantic_evaluation": "not_evaluable",
        "false_binding_queries": scored["false_binding_queries"],
        "false_binding_on_bindable": false_bindable,
        "false_binding_on_unbindable": false_unbindable,
        "binder_semantic_selection_effective": True if strong else ("partial" if partial else False),
        "binder_semantic_policy_frozen": strong,
        "dominant_failure": "none" if strong else ("binder_semantic_selection" if not partial else "partial_semantic_selection"),
        "next_gate": "v2_04_missing_evidence_supply_repair" if strong else "v2_03_r2_failure_review",
        "production_default": "V1",
        "production_switch_allowed": False,
    }
    attempt6 = read_json(r1d.ROOT / "artifacts/evaluation/nf-v2-03-r1d-supply-conditioned-binder/formal-attempt-6/decision.json")
    write_json(FORMAL_OUT / "attempt6-attempt7-ablation.json", {
        "attempt6_prompt": {"direct": "12/27", "calculation": "0/6", "false_binding": 11},
        "attempt7_prompt_r2": {"direct": f"{scored['direct']['strict_correct_given_bindable']}/27", "calculation": f"{scored['calculation']['strict_correct_given_bindable']}/6", "false_binding": scored["false_binding_queries"]},
        "attempt6_decision_reference": {"binder_semantic_selection_effective": attempt6.get("binder_semantic_selection_effective")},
        "prompt_sha256": prompt_hash,
    })
    write_json(FORMAL_OUT / "decision.json", decision)
    write_json(OUT / "decision.json", decision)
    write_json(FORMAL_OUT / "README.md", {"gate": "NF-V2-03-R2", "description": "Formal Attempt 7 using one frozen, generalizable Binder Prompt R2. Provider schema, DTO, fact view, supply, and evaluator remain frozen.", "decision": decision})
    print(json.dumps({"gate": "NF-V2-03-R2", "provider_responses": structural["provider_responses"], "dto": structural["dto_valid"], "validator": structural["binding_validator_pass"], "direct": scored["direct"]["strict_correct_given_bindable"], "calc": scored["calculation"]["strict_correct_given_bindable"], "false_binding": scored["false_binding_queries"], "decision": decision["binder_semantic_selection_effective"], "next_gate": decision["next_gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
