#!/usr/bin/env python3
"""Unified post-seal R8 scoring over the canonical 80 source bindings."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
CONTRACT = BASE / "pdf-retrieval-v4-strict-source-contract"
SIDECAR = CONTRACT / "strict-gold-source-bindings.jsonl"
R0_DIR = BASE / "pdf-retrieval-v4-gate-08-r8-r0"
R0 = R0_DIR / "candidate-depth-predictions.jsonl.gz"
R1_DIR = BASE / "pdf-retrieval-v4-gate-08-r8-r1"
R1 = R1_DIR / "candidate-top50-predictions.jsonl.gz"
R7 = BASE / "pdf-retrieval-v4-gate-08-r7/field-family-predictions.jsonl.gz"
GATE08 = BASE / "pdf-retrieval-v4-gate-08/retrieval-predictions.jsonl.gz"
GOV = BASE / "pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r1-1b"
DEPTHS = (5, 10, 20, 40, 50)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            item["case_id"]: item
            for item in (json.loads(line) for line in handle if line.strip())
            if item.get("case_id")
        }


def keys(items: list[dict[str, Any]]) -> set[str]:
    return {str(item["candidate_key"]) for item in items}


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    contract = json.loads((CONTRACT / "acceptance.json").read_text())
    if contract["decision"] != "strict_gold_source_binding_contract_closed" or contract["sidecar_sha256"] != sha(SIDECAR):
        raise RuntimeError("strict_source_binding_contract_not_closed")
    for directory, prediction in ((R0_DIR, R0), (R1_DIR, R1)):
        seal = json.loads((directory / "prediction-seal.json").read_text())
        if not seal.get("sealed") or seal["prediction_sha256"] != sha(prediction):
            raise RuntimeError(f"sealed_prediction_invalid:{prediction}")
    bindings = [json.loads(line) for line in SIDECAR.open(encoding="utf-8")]
    if len(bindings) != 80 or len({item["binding_id"] for item in bindings}) != 80:
        raise RuntimeError("strict_source_binding_count_invalid")
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in bindings:
        by_case[binding["case_id"]].append(binding)
    governance = {item["case_id"]: item for item in map(json.loads, GOV.open(encoding="utf-8"))}
    r0, r1, r7, original = map(load_gzip, (R0, R1, R7, GATE08))

    def score_pools(pools: dict[str, set[str]]) -> tuple[int, int, int]:
        recalled = sum(binding["candidate_key"] in pools[binding["case_id"]] for binding in bindings)
        multi = calculation = 0
        for case_id, expected_bindings in by_case.items():
            complete = all(item["candidate_key"] in pools[case_id] for item in expected_bindings)
            if governance[case_id]["requires_multiple_sources"]:
                multi += complete
            if governance[case_id]["query_type"] == "calculation_multi_operand":
                calculation += complete
        return recalled, multi, calculation

    r0_metrics = {}
    r1_metrics = {}
    for depth in DEPTHS:
        r0_pools = {case_id: keys(record[f"candidate_pool_{depth}"]) for case_id, record in r0.items()}
        r1_pools = {case_id: keys(record["bounded_candidate_ranking"][:depth]) for case_id, record in r1.items()}
        for target, pools in ((r0_metrics, r0_pools), (r1_metrics, r1_pools)):
            recall, multi, calculation = score_pools(pools)
            target[f"strict_source_binding_recall_at_{depth}"] = f"{recall}/80"
            target[f"benchmark_multi_evidence_complete_at_{depth}"] = f"{multi}/16"
            target[f"calculation_complete_at_{depth}"] = f"{calculation}/11"
    union_pools = {case_id: keys(record["r7_full_pool"]) for case_id, record in r7.items()}
    bounded_pools = {case_id: keys(record["bounded_candidate_ranking"]) for case_id, record in r1.items()}
    raw50_pools = {case_id: keys(record["raw_full_rrf_candidates"][:50]) for case_id, record in original.items()}
    union_bindings = {item["binding_id"] for item in bindings if item["candidate_key"] in union_pools[item["case_id"]]}
    bounded_bindings = {item["binding_id"] for item in bindings if item["candidate_key"] in bounded_pools[item["case_id"]]}
    raw_bindings = {item["binding_id"] for item in bindings if item["candidate_key"] in raw50_pools[item["case_id"]]}
    raw_retained = raw_bindings & bounded_bindings
    gross_loss = union_bindings - bounded_bindings
    synergy = bounded_bindings - union_bindings
    bounded_count = len(bounded_bindings)
    conversion = bounded_count / len(union_bindings)
    raw_regression = raw_bindings - raw_retained
    multi50 = int(r1_metrics["benchmark_multi_evidence_complete_at_50"].split("/")[0])
    calc50 = int(r1_metrics["calculation_complete_at_50"].split("/")[0])
    if bounded_count >= 57 and conversion >= 0.95 and len(raw_regression) <= 1:
        decision = "bounded_candidate_selection_strong_pass"
    elif bounded_count >= 54 and conversion >= 0.90 and len(raw_regression) <= 1:
        decision = "bounded_candidate_selection_passed"
    elif len(raw_regression) > 1:
        decision = "bounded_candidate_raw_regression_blocked"
    else:
        decision = "bounded_candidate_selection_meaningful"
    next_gate = "support_count_invariant_candidate_fusion" if bounded_count >= 54 else "bounded_selector_mechanism_audit"
    metrics = {
        "scoring_unit": "strict_gold_source_binding",
        "denominator": 80,
        "r8_r0": r0_metrics,
        "r8_r1": r1_metrics,
        "unbounded_presence": f"{len(union_bindings)}/80",
        "bounded_top50": f"{bounded_count}/80",
        "union_to_top50_conversion": f"{bounded_count}/{len(union_bindings)}",
        "union_to_top50_conversion_rate": conversion,
        "gross_loss": len(gross_loss),
        "selector_synergy": len(synergy),
        "net_gap": len(gross_loss) - len(synergy),
        "production_raw_own_recall_at_50": f"{len(raw_bindings)}/80",
        "raw_retained": f"{len(raw_retained)}/{len(raw_bindings)}",
        "raw_regression": len(raw_regression),
        "multi_evidence_complete_at_50": f"{multi50}/16",
        "calculation_complete_at_50": f"{calc50}/11",
    }
    write("unified-binding-metrics.json", metrics)
    write("raw-binding-retention.json", {"raw_binding_ids": sorted(raw_bindings), "retained_binding_ids": sorted(raw_retained), "regressed_binding_ids": sorted(raw_regression)})
    write("selector-binding-transition.json", {"gross_loss_binding_ids": sorted(gross_loss), "synergy_binding_ids": sorted(synergy)})
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_08_r8_r1_1b",
        "decision": decision,
        "next_gate": next_gate,
        "strict_source_contract_sha256": sha(SIDECAR),
        "metrics": metrics,
        "prediction_reruns": 0,
        "fusion_reruns": 0,
        "retrieval_runs": 0,
        "index_reads": 0,
        "embedding_calls": 0,
        "production_switch_allowed": False,
    }
    write("acceptance.json", acceptance)
    write("next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
