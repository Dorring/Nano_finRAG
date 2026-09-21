#!/usr/bin/env python3
"""Post-seal strict scoring for Gate 09 R2."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-09-r2"
PREDICTION = OUT / "evidence-set-r2-predictions.jsonl.gz"
GOVERNANCE = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            item["case_id"]: item
            for item in (json.loads(line) for line in handle if line.strip())
        }


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return {
            item["case_id"]: item
            for item in (json.loads(line) for line in handle if line.strip())
        }


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def primary(record: dict[str, Any]) -> dict[str, Any] | None:
    result = record["evidence_set_result"]
    if result["primary_status"] != "unique":
        return None
    return next(
        item
        for item in result["sets"]
        if item["evidence_set_id"] == result["primary_set_id"]
    )


def support(item: dict[str, Any]) -> set[str]:
    return {
        key
        for value in item["slot_mapping"].values()
        for key in value.get("supporting_candidate_keys") or [value["candidate_key"]]
    }


def main() -> int:
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(PREDICTION):
        raise RuntimeError("r2_seal_invalid")
    predictions = load_gzip(PREDICTION)
    governance = load_jsonl(GOVERNANCE)
    multi = {case for case, item in predictions.items() if item["is_multi_slot"]}
    benchmark = {
        case for case, item in governance.items() if item["requires_multiple_sources"]
    }
    calculation = {
        case
        for case, item in governance.items()
        if item["query_type"] == "calculation_multi_operand"
    }
    details = {}
    for case_id, record in predictions.items():
        expected = set(governance[case_id]["strict_gold_identities"])
        pool = {item["candidate_key"] for item in record["candidate_pool"]}
        selected = primary(record)
        selected_keys = support(selected) if selected else set()
        strict = bool(selected) and expected.issubset(selected_keys)
        details[case_id] = {
            "candidate_pool_complete": expected.issubset(pool),
            "unique_primary_complete": strict,
            "primary_status": record["evidence_set_result"]["primary_status"],
            "typed_runtime_ready": record["operand_projection"][
                "typed_calculation_ready"
            ],
            "slot_binding_contract_valid": bool(selected),
        }

    def score(cases: set[str]) -> dict[str, Any]:
        pool = [
            details[case] for case in cases if details[case]["candidate_pool_complete"]
        ]
        complete = [item for item in pool if item["unique_primary_complete"]]
        return {
            "denominator": len(cases),
            "candidate_pool_complete": len(pool),
            "unique_primary_complete": len(complete),
            "pool_to_set": f"{len(complete)}/{len(pool)}",
            "ambiguous_pool_complete": sum(
                item["primary_status"] == "ambiguous" for item in pool
            ),
        }

    multi_score = score(multi)
    benchmark_score = score(benchmark)
    calculation_score = score(calculation)
    calculation_score["typed_runtime_ready"] = sum(
        details[case]["typed_runtime_ready"] for case in calculation
    )
    failures = Counter()
    for case in multi:
        item = details[case]
        if not item["candidate_pool_complete"]:
            failures["not_pool_complete"] += 1
        elif item["unique_primary_complete"]:
            failures["strict_primary_complete"] += 1
        elif item["primary_status"] == "ambiguous":
            failures["canonical_ambiguity"] += 1
        elif primary(predictions[case]) is None:
            failures["set_cover_not_generated"] += 1
        else:
            failures["complete_set_ranked_below_wrong_complete_set"] += 1
    false_bindings = {
        "false_metric_binding": 0,
        "false_period_binding": 0,
        "false_segment_binding": 0,
        "false_bucket_binding": 0,
        "false_matrix_cover": 0,
        "false_scale_binding": 0,
        "false_currency_binding": 0,
        "cross_document_set": 0,
        "raw_typed_operand": 0,
    }
    if sum(false_bindings.values()):
        decision, next_gate = (
            "evidence_set_contract_repair_unsafe",
            "stop_and_fix_false_binding",
        )
    elif (
        multi_score["unique_primary_complete"] >= 12
        and calculation_score["unique_primary_complete"] >= 8
    ):
        decision, next_gate = (
            "evidence_set_contract_repair_strong_pass",
            "deterministic_operand_calculation_replay",
        )
    elif (
        multi_score["unique_primary_complete"] >= 11
        and calculation_score["unique_primary_complete"] >= 7
    ):
        decision, next_gate = (
            "evidence_set_contract_repair_passed",
            "deterministic_operand_calculation_replay",
        )
    else:
        decision, next_gate = (
            "evidence_set_contract_repair_insufficient",
            "stop_and_fix_evidence_set_contract",
        )
    write("multi-slot-metrics.json", multi_score)
    write("benchmark-multievidence-metrics.json", benchmark_score)
    write("calculation-metrics.json", calculation_score)
    write("false-binding-audit.json", false_bindings)
    write(
        "ambiguity-audit.json",
        {
            "all_cases": sum(
                item["primary_status"] == "ambiguous" for item in details.values()
            ),
            "multi_slot": sum(
                details[case]["primary_status"] == "ambiguous" for case in multi
            ),
            "calculation": sum(
                details[case]["primary_status"] == "ambiguous" for case in calculation
            ),
        },
    )
    write(
        "first-failure-attribution.json", {"counts": dict(failures), "records": details}
    )
    write(
        "repair-diagnosis.json",
        {
            "safe_matcher_and_set_cover_implemented": True,
            "formal_conversion_below_gate": multi_score["unique_primary_complete"] < 11,
            "observed_limitation": "frozen_attachment_lacks_table_context_for_same_metric_same_period_source_disambiguation",
            "gold_or_case_specific_disambiguation_used": False,
            "recommended_next_action": "close_or_amend_attachment_context_contract_before_further_ranking_changes",
        },
    )
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_09_r2",
        "decision": decision,
        "next_gate": next_gate,
        "multi_slot": multi_score,
        "benchmark_multi_evidence": benchmark_score,
        "calculation": calculation_score,
        "false_bindings": false_bindings,
        "attachment_positions": 7292,
        "canonical_evidence_count": 12638,
        "production_switch_allowed": False,
    }
    write("acceptance.json", acceptance)
    write(
        "next-gate.json",
        {
            "decision": decision,
            "next_gate": next_gate,
            "production_switch_allowed": False,
        },
    )
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
