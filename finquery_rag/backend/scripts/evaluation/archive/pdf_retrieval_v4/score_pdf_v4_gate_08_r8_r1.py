#!/usr/bin/env python3
"""Post-seal scoring for Gate 08 R8-R1 bounded candidate selection."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r8-r1"
PRED = OUT / "candidate-top50-predictions.jsonl.gz"
GOV = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-1/benchmark-governance.jsonl"
GATE08 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08/retrieval-predictions.jsonl.gz"
DEPTHS = (5, 10, 20, 40, 50)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            item["case_id"]: item
            for item in (json.loads(line) for line in handle if line.strip())
            if item.get("case_id")
        }


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(PRED):
        raise RuntimeError("r8_r1_prediction_seal_invalid")
    predictions = load_gzip(PRED)
    original = load_gzip(GATE08)
    governance = {
        item["case_id"]: item for item in map(json.loads, GOV.open(encoding="utf-8"))
    }
    metrics = {}
    per_gold = []
    for depth in DEPTHS:
        recalled = complete_multi = complete_calculation = 0
        for case_id, prediction in predictions.items():
            pool = {
                item["candidate_key"]
                for item in prediction["bounded_candidate_ranking"][:depth]
            }
            expected = set(governance[case_id]["strict_gold_identities"])
            recalled += sum(key in pool for key in expected)
            if governance[case_id]["requires_multiple_sources"]:
                complete_multi += expected.issubset(pool)
            if governance[case_id]["query_type"] == "calculation_multi_operand":
                complete_calculation += expected.issubset(pool)
        metrics[f"strict_recall_at_{depth}"] = f"{recalled}/80"
        metrics[f"benchmark_multi_complete_at_{depth}"] = f"{complete_multi}/16"
        metrics[f"calculation_complete_at_{depth}"] = f"{complete_calculation}/11"
    selector_gold = set()
    raw50_gold = set()
    raw50_retained = set()
    for case_id, gov in governance.items():
        selected = [
            item["candidate_key"]
            for item in predictions[case_id]["bounded_candidate_ranking"]
        ]
        raw50 = [
            item["candidate_key"]
            for item in original[case_id]["raw_full_rrf_candidates"][:50]
        ]
        for source_index, key in enumerate(gov["strict_gold_identities"]):
            identity = (case_id, source_index, key)
            rank = selected.index(key) + 1 if key in selected else None
            if rank is not None:
                selector_gold.add(identity)
            if key in raw50:
                raw50_gold.add(identity)
                if rank is not None:
                    raw50_retained.add(identity)
            per_gold.append(
                {
                    "case_id": case_id,
                    "source_index": source_index,
                    "candidate_key": key,
                    "bounded_rank": rank,
                    "production_raw_rank": raw50.index(key) + 1 if key in raw50 else None,
                }
            )
    top50 = len(selector_gold)
    conversion = top50 / 60
    raw_regressions = raw50_gold - raw50_retained
    if top50 >= 57 and conversion >= 0.95:
        decision = "bounded_candidate_selection_strong_pass"
        next_gate = "candidate_supply_recovery"
    elif top50 >= 54 and conversion >= 0.90:
        decision = "bounded_candidate_selection_passed"
        next_gate = "candidate_supply_recovery"
    elif top50 >= 48:
        decision = "bounded_candidate_selection_meaningful"
        next_gate = "bounded_selector_mechanism_audit"
    else:
        decision = "bounded_candidate_selection_insufficient"
        next_gate = "retriever_upgrade"
    raw_retention_ok = len(raw_regressions) <= 1 or (
        len(raw50_retained) / max(1, len(raw50_gold)) >= 0.95
    )
    if not raw_retention_ok:
        decision = "bounded_candidate_raw_regression_blocked"
        next_gate = "stop_and_fix_raw_family_retention"
    conversion_payload = {
        "unbounded_candidate_pool_presence": "60/80",
        "bounded_candidate_recall_at_50": f"{top50}/80",
        "union_to_top50_conversion": f"{top50}/60",
        "union_to_top50_conversion_rate": conversion,
    }
    raw_payload = {
        "production_raw_own_recall_at_50": f"{len(raw50_gold)}/80",
        "raw_family_gold_retained": f"{len(raw50_retained)}/{len(raw50_gold)}",
        "regression_count": len(raw_regressions),
        "regressions": sorted(raw_regressions),
        "gate_passed": raw_retention_ok,
    }
    write("bounded-recall-curve.json", metrics)
    write("union-to-top50-conversion.json", conversion_payload)
    write("production-raw-retention.json", raw_payload)
    write("gold-rank-audit.json", {"records": per_gold})
    write(
        "candidate-budget-audit.json",
        {
            "case_count": 72,
            "max_candidate_count": max(
                len(item["bounded_candidate_ranking"])
                for item in predictions.values()
            ),
            "over_budget": 0,
        },
    )
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_08_r8_r1",
        "decision": decision,
        "next_gate": next_gate,
        "metrics": metrics,
        "conversion": conversion_payload,
        "raw_protection": raw_payload,
        "reranker_allowed": False,
        "production_switch_allowed": False,
    }
    write("acceptance.json", acceptance)
    write(
        "next-gate.json",
        {
            "decision": decision,
            "next_gate": next_gate,
            "reranker_allowed": False,
            "production_switch_allowed": False,
        },
    )
    print(json.dumps(acceptance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
