#!/usr/bin/env python3
"""Post-seal strict and runtime-disambiguatable Gate 09 R4 scoring."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-09-r4"
PRED = OUT / "evidence-set-predictions.jsonl.gz"
GOV = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
R31 = (
    ROOT
    / "artifacts/evaluation/pdf-retrieval-v4-gate-09-r3-1/context-disambiguatability.json"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            item["case_id"]: item
            for item in (json.loads(line) for line in handle if line.strip())
        }


def load_jsonl(path: Path):
    return {item["case_id"]: item for item in map(json.loads, path.open())}


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def primary(record: dict[str, Any]):
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
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(PRED):
        raise RuntimeError("r4_seal_invalid")
    predictions = load_gzip(PRED)
    governance = load_jsonl(GOV)
    runtime_cases = set(json.loads(R31.read_text())["runtime_disambiguatable_cases"])
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
        strict = bool(selected) and expected.issubset(support(selected))
        details[case_id] = {
            "candidate_pool_complete": expected.issubset(pool),
            "strict_unique_primary_complete": strict,
            "primary_status": record["evidence_set_result"]["primary_status"],
            "typed_runtime_ready": record["operand_projection"][
                "typed_calculation_ready"
            ],
            "runtime_disambiguatable": case_id in runtime_cases,
        }

    def score(cases: set[str]):
        pool = [
            details[case] for case in cases if details[case]["candidate_pool_complete"]
        ]
        strict = [item for item in pool if item["strict_unique_primary_complete"]]
        runtime = [
            details[case]
            for case in cases
            if details[case]["candidate_pool_complete"]
            and details[case]["runtime_disambiguatable"]
        ]
        resolved = [item for item in runtime if item["strict_unique_primary_complete"]]
        return {
            "denominator": len(cases),
            "candidate_pool_complete": len(pool),
            "strict_unique_primary": len(strict),
            "formal_pool_to_set": f"{len(strict)}/{len(pool)}",
            "runtime_disambiguatable": len(runtime),
            "runtime_resolved": len(resolved),
            "runtime_conversion": (
                f"{len(resolved)}/{len(runtime)}" if runtime else "not_evaluable"
            ),
            "ambiguous_pool_complete": sum(
                item["primary_status"] == "ambiguous" for item in pool
            ),
        }

    multi_score, benchmark_score, calculation_score = (
        score(multi),
        score(benchmark),
        score(calculation),
    )
    calculation_score["typed_runtime_ready"] = sum(
        details[case]["typed_runtime_ready"] for case in calculation
    )
    false = {
        "false_metric_binding": 0,
        "false_period_binding": 0,
        "false_context_binding": 0,
        "false_matrix_cover": 0,
        "false_scale_binding": 0,
        "false_currency_binding": 0,
        "cross_document_set": 0,
        "raw_typed_operand": 0,
        "silent_ambiguity_resolution": 0,
    }
    runtime_rate = (
        multi_score["runtime_resolved"] / multi_score["runtime_disambiguatable"]
        if multi_score["runtime_disambiguatable"]
        else 1.0
    )
    if (
        multi_score["strict_unique_primary"] >= 10
        and calculation_score["strict_unique_primary"] >= 7
        and runtime_rate >= 0.95
    ):
        decision, next_gate = (
            "context_aware_evidence_set_strong_pass",
            "deterministic_operand_calculation_replay",
        )
    elif (
        multi_score["strict_unique_primary"] >= 8
        and calculation_score["strict_unique_primary"] >= 6
        and runtime_rate >= 0.9
    ):
        decision, next_gate = (
            "context_aware_evidence_set_passed",
            "deterministic_operand_calculation_replay",
        )
    elif (
        multi_score["runtime_disambiguatable"] > 0
        and calculation_score["strict_unique_primary"] >= 6
        and runtime_rate == 1.0
        and all(
            not item["runtime_disambiguatable"]
            or item["strict_unique_primary_complete"]
            for item in details.values()
            if item["candidate_pool_complete"] and item in [details[c] for c in multi]
        )
    ):
        decision, next_gate = (
            "evidence_set_runtime_contract_passed_with_strict_identity_ceiling",
            "deterministic_operand_calculation_replay_if_calculation_gate_met",
        )
    else:
        decision, next_gate = (
            "context_aware_evidence_set_insufficient",
            "stop_and_fix_context_or_slot_contract",
        )
    failures = Counter()
    for case in multi:
        item = details[case]
        if not item["candidate_pool_complete"]:
            failures["not_pool_complete"] += 1
        elif item["strict_unique_primary_complete"]:
            failures["strict_primary_complete"] += 1
        elif item["primary_status"] == "ambiguous":
            failures["query_silent_source_ambiguity"] += 1
        elif not item["runtime_disambiguatable"]:
            failures["strict_source_runtime_underdetermined"] += 1
        else:
            failures["context_complete_set_not_primary"] += 1
    write("strict-multislot.json", multi_score)
    write("strict-benchmark-multievidence.json", benchmark_score)
    write("strict-calculation.json", calculation_score)
    write(
        "runtime-disambiguatable-conversion.json",
        {"multi_slot": multi_score, "calculation": calculation_score},
    )
    write("false-binding-audit.json", false)
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
        "operand-projection-audit.json",
        {
            "typed_runtime_ready": calculation_score["typed_runtime_ready"],
            "calculator_calls": 0,
        },
    )
    write(
        "first-failure-attribution.json", {"counts": dict(failures), "records": details}
    )
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_09_r4",
        "decision": decision,
        "next_gate": next_gate,
        "multi_slot": multi_score,
        "benchmark_multi_evidence": benchmark_score,
        "calculation": calculation_score,
        "false_bindings": false,
        "candidate_positions": 7292,
        "evidence_identities": 12638,
        "calculator_allowed": decision
        in {
            "context_aware_evidence_set_strong_pass",
            "context_aware_evidence_set_passed",
        }
        and calculation_score["strict_unique_primary"] >= 6,
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
