#!/usr/bin/env python3
"""Zero-retrieval Gate 08 R5.1 diagnostic contract closure."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.r5_rank_contract import (  # noqa: E402
    classify_rank_migration,
    recovered_to_cutoff,
)

R5 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r5"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r5-1"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
UNIVERSE = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5/universe-scoring.json"
R3_FAILURE = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3/first-failure-attribution-corrected.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    pred_path = R5 / "retrieval-predictions.jsonl.gz"
    seal_path = R5 / "prediction-seal.json"
    seal = json.loads(seal_path.read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(pred_path):
        raise RuntimeError("original_r5_seal_invalid")
    with gzip.open(pred_path, "rt", encoding="utf-8") as handle:
        predictions = {item["case_id"]: item for item in (json.loads(line) for line in handle if line.strip())}
    gold = []
    with LABELS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for index, source in enumerate(record.get("expected_sources") or []):
                if source.get("candidate_key"):
                    gold.append((record["case_id"], index, source["candidate_key"]))
    details = json.loads(UNIVERSE.read_text())["details"]
    grade_a = {
        identity
        for identity, detail in zip(gold, details, strict=True)
        if detail.get("new_status") == "mapped"
    }
    metrics = {}
    ranks_by_variant = {}
    for variant in [f"s{i}" for i in range(5)]:
        ranks = {}
        for identity in grade_a:
            items = predictions[identity[0]]["structured_family_rankings"][variant]
            keys = [item["candidate_key"] for item in items]
            ranks[identity] = keys.index(identity[2]) + 1 if identity[2] in keys else None
        ranks_by_variant[variant] = ranks
        metrics[variant] = {
            "recall_at_10": f"{sum(rank is not None and rank <= 10 for rank in ranks.values())}/68",
            "recall_at_20": f"{sum(rank is not None and rank <= 20 for rank in ranks.values())}/68",
            "recall_at_40": f"{sum(rank is not None and rank <= 40 for rank in ranks.values())}/68",
            "recall_at_50": f"{sum(rank is not None and rank <= 50 for rank in ranks.values())}/68",
            "presence_any": f"{sum(rank is not None for rank in ranks.values())}/68",
        }
    migrations = []
    for identity in sorted(grade_a):
        old_rank = ranks_by_variant["s0"][identity]
        new_rank = ranks_by_variant["s4"][identity]
        migrations.append({"case_id": identity[0], "source_index": identity[1], "candidate_key": identity[2], "old_rank": old_rank, "new_rank": new_rank, "category": classify_rank_migration(old_rank, new_rank)})
    frozen = json.loads(R3_FAILURE.read_text())["failure_details"]
    frozen12 = {(item["case_id"], item["source_index"], item["candidate_key"]) for item in frozen if item["first_failure_stage"] == "structured_bm25_and_dense_top50_miss"}
    recovery = {"frozen_r3_top50_misses": len(frozen12), "recovered_to_top40": 0, "recovered_to_top50": 0, "recovered_beyond_top50_only": 0, "still_absent": 0, "records": []}
    for identity in sorted(frozen12):
        rank = ranks_by_variant["s4"].get(identity)
        if recovered_to_cutoff(None, rank, 40):
            category = "recovered_to_top40"
        elif recovered_to_cutoff(None, rank, 50):
            category = "recovered_to_top50"
        elif rank is not None:
            category = "recovered_beyond_top50_only"
        else:
            category = "still_absent"
        recovery[category] += 1
        recovery["records"].append({"case_id": identity[0], "source_index": identity[1], "candidate_key": identity[2], "new_rank": rank, "category": category})
    full_hits = {identity for identity in grade_a if identity[2] in {item["candidate_key"] for item in predictions[identity[0]]["r5_full_pool"]}}
    full_metric = {"full_grade_a_recall": f"{len(full_hits)}/68", "full_inside_universe_miss": 68 - len(full_hits)}
    closure = {"gate": "pdf_retrieval_v4_gate_08_r5_1", "original_r5_prediction_sha256": sha(pred_path), "original_r5_seal_sha256": sha(seal_path), "original_r5_immutable": True, "bm25_searches": 0, "dense_searches": 0, "embedding_calls": 0, "index_reads": 0, "index_builds": 0, "prediction_reruns": 0, "decision": "field_aware_diagnostic_contract_closed", "next_gate": "slot_aware_candidate_composition"}
    write("rank-contract-closure.json", closure)
    write("structured-cutoff-metrics.json", metrics)
    write("corrected-rank-migration.json", {"summary": dict(Counter(item["category"] for item in migrations)), "records": migrations})
    write("corrected-top50-miss-recovery.json", recovery)
    write("full-grade-a-conversion.json", full_metric)
    write("acceptance.json", closure)
    print(json.dumps({"cutoffs": metrics, "full": full_metric, "top50_recovery": {key: value for key, value in recovery.items() if key != "records"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
