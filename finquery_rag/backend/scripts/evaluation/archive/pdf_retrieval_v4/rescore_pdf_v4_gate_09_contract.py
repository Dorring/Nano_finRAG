#!/usr/bin/env python3
"""Gate 09 R1 post-seal evaluation-contract closure."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GATE09 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-09"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-09-r1"
PREDICTIONS = GATE09 / "evidence-set-predictions.jsonl.gz"
SEAL = GATE09 / "prediction-seal.json"
ATTACHMENTS = GATE09 / "candidate-evidence-attachment.jsonl.gz"
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


def supporting_keys(evidence_set: dict[str, Any]) -> set[str]:
    return {
        key
        for value in evidence_set["slot_mapping"].values()
        for key in value.get("supporting_candidate_keys") or [value["candidate_key"]]
    }


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    seal = json.loads(SEAL.read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(PREDICTIONS):
        raise RuntimeError("original_gate09_seal_invalid")
    predictions = load_gzip(PREDICTIONS)
    governance = load_jsonl(GOVERNANCE)
    multi_cases = {
        case_id for case_id, item in predictions.items() if item["is_multi_slot"]
    }
    benchmark_multi = {
        case_id
        for case_id, item in governance.items()
        if item["requires_multiple_sources"]
    }
    calculation = {
        case_id
        for case_id, item in governance.items()
        if item["query_type"] == "calculation_multi_operand"
    }
    if (len(multi_cases), len(benchmark_multi), len(calculation)) != (18, 16, 11):
        raise RuntimeError("governance_denominator_drift")
    details = []
    ambiguity = Counter()
    for case_id, record in predictions.items():
        expected = set(governance[case_id]["strict_gold_identities"])
        pool_keys = {item["candidate_key"] for item in record["candidate_pool"]}
        result = record["evidence_set_result"]
        sets_by_id = {
            item["evidence_set_id"]: item for item in result.get("sets") or []
        }
        primary_ids = result.get("primary_set_ids") or []
        primaries = [sets_by_id[item] for item in primary_ids if item in sets_by_id]
        ambiguous = bool(result.get("ambiguous_primary"))
        if ambiguous:
            ambiguity["all_cases_ambiguous"] += 1
            ambiguity[
                "multi_slot_ambiguous"
                if case_id in multi_cases
                else "single_slot_ambiguous"
            ] += 1
            if case_id in calculation:
                ambiguity["calculation_ambiguous"] += 1
            if case_id in benchmark_multi:
                ambiguity["benchmark_multievidence_ambiguous"] += 1
        strict = (
            not ambiguous
            and len(primaries) == 1
            and expected.issubset(supporting_keys(primaries[0]))
        )
        any_co = any(expected.issubset(supporting_keys(item)) for item in primaries)
        union = (
            expected.issubset(
                set().union(*(supporting_keys(item) for item in primaries))
            )
            if primaries
            else False
        )
        set_exists = any(
            expected.issubset(supporting_keys(item))
            for item in result.get("sets") or []
        )
        canonical = record["canonical_evidence"]
        attached = {
            key
            for item in canonical
            for key in item.get("supporting_candidate_keys") or [item["candidate_key"]]
        }
        typed_attached = {
            key
            for item in canonical
            if item["evidence_type"] != "raw_candidate"
            for key in item.get("supporting_candidate_keys") or [item["candidate_key"]]
        }
        compatible = {
            key
            for matches in result.get("slot_matches", {}).values()
            for item in matches
            for key in item.get("supporting_candidate_keys") or [item["candidate_key"]]
        }
        pool_complete = expected.issubset(pool_keys)
        if not pool_complete:
            first_failure = "not_pool_complete"
        elif not expected.issubset(attached):
            first_failure = "gold_candidate_without_semantic_attachment"
        elif not expected.issubset(typed_attached):
            first_failure = "gold_attachment_without_compatible_evidence"
        elif not expected.issubset(compatible):
            first_failure = "metric_slot_mismatch"
        elif not set_exists:
            first_failure = "gold_complete_set_not_generated"
        elif strict:
            first_failure = "strict_primary_complete"
        elif any_co and ambiguous:
            first_failure = "gold_complete_set_co_primary_ambiguous"
        else:
            first_failure = "gold_complete_set_generated_but_not_primary"
        details.append(
            {
                "case_id": case_id,
                "candidate_pool_gold_complete": pool_complete,
                "strict_primary_gold_complete": strict,
                "any_co_primary_gold_complete": any_co,
                "union_of_co_primary_gold_complete": union,
                "union_diagnostic_only": True,
                "gold_complete_set_exists": set_exists,
                "ambiguous_primary": ambiguous,
                "first_failure_stage": first_failure,
            }
        )
    by_case = {item["case_id"]: item for item in details}

    def metrics(cases: set[str]) -> dict[str, Any]:
        pool = [
            by_case[case]
            for case in cases
            if by_case[case]["candidate_pool_gold_complete"]
        ]
        strict = [item for item in pool if item["strict_primary_gold_complete"]]
        return {
            "denominator": len(cases),
            "candidate_pool_complete": len(pool),
            "strict_unique_primary_complete": len(strict),
            "pool_to_set": f"{len(strict)}/{len(pool)}",
            "any_co_primary_complete": sum(
                item["any_co_primary_gold_complete"] for item in pool
            ),
            "union_of_co_primary_complete_diagnostic": sum(
                item["union_of_co_primary_gold_complete"] for item in pool
            ),
        }

    multi_metrics = metrics(multi_cases)
    benchmark_metrics = metrics(benchmark_multi)
    calc_metrics = metrics(calculation)
    typed_presence = sum(
        predictions[case]["evidence_set_result"]["status"] == "complete"
        for case in calculation
    )
    calculation_metrics = {
        **calc_metrics,
        "c0_typed_numeric_presence_ready": f"{typed_presence}/11",
        "c1_unique_primary_operand_complete": sum(
            not by_case[case]["ambiguous_primary"]
            and predictions[case]["evidence_set_result"].get("planner_complete")
            for case in calculation
        ),
        "c3_strict_calculation_evidence_complete": calc_metrics[
            "strict_unique_primary_complete"
        ],
    }
    pool_complete_multi = [
        item
        for item in details
        if item["case_id"] in multi_cases and item["candidate_pool_gold_complete"]
    ]
    stage_counts = Counter(item["first_failure_stage"] for item in pool_complete_multi)
    write(
        "evaluation-contract.json",
        {
            "strict_single_primary_is_formal": True,
            "any_co_primary_is_diagnostic": True,
            "union_of_co_primary_is_diagnostic": True,
            "calculation_ready_renamed": "typed_numeric_presence_ready",
        },
    )
    write(
        "primary-set-scoring.json",
        {"queryplan_multi_slot": multi_metrics, "records": details},
    )
    write(
        "ambiguity-audit.json",
        {
            **dict(ambiguity),
            "pool_complete_multi_slot_ambiguous": sum(
                item["ambiguous_primary"] for item in pool_complete_multi
            ),
        },
    )
    write("calculation-readiness-closure.json", calculation_metrics)
    write("benchmark-multievidence.json", benchmark_metrics)
    write(
        "stage-failure-localization.json",
        {
            "scope": "12_pool_complete_multi_slot_cases",
            "counts": dict(stage_counts),
            "records": pool_complete_multi,
        },
    )
    write(
        "set-existence-vs-primary.json",
        {
            "gold_complete_set_exists": sum(
                item["gold_complete_set_exists"] for item in pool_complete_multi
            ),
            "unique_primary_gold_complete": sum(
                item["strict_primary_gold_complete"] for item in pool_complete_multi
            ),
            "denominator": 12,
        },
    )
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_09_r1",
        "decision": "evidence_set_failure_contract_closed",
        "next_gate": "deterministic_evidence_set_contract_repair",
        "original_prediction_sha256": sha(PREDICTIONS),
        "original_attachment_sha256": sha(ATTACHMENTS),
        "original_gate09_immutable": True,
        "multi_slot": multi_metrics,
        "benchmark_multi_evidence": benchmark_metrics,
        "calculation": calculation_metrics,
        "first_failure_exhaustive": len(pool_complete_multi)
        == sum(stage_counts.values()),
        "retrieval_runs": 0,
        "slot_matcher_runs": 0,
        "set_generator_runs": 0,
        "production_switch_allowed": False,
    }
    write("acceptance.json", acceptance)
    write(
        "next-gate.json",
        {
            "decision": acceptance["decision"],
            "next_gate": acceptance["next_gate"],
            "production_switch_allowed": False,
        },
    )
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
