#!/usr/bin/env python3
"""Build and seal the Gate 08 R8-R2A.2 bounded Top100 rerank input."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.bounded_rerank_input_selector import (  # noqa: E402
    RERANK_INPUT_BUDGET,
    SLOT_COMPOSITION_HORIZON,
    SLOT_MIN_BUDGET,
    build_priority_ranking,
    select_multi_slot_top100,
    select_single_slot_top100,
)

BASE = ROOT / "artifacts/evaluation"
R2A = BASE / "pdf-retrieval-v4-gate-08-r8-r2a"
R2A_PRED = R2A / "deep-supply-predictions.jsonl.gz"
SIDECAR = BASE / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r2a-2"
PRED = OUT / "bounded-top100-predictions.jsonl.gz"
EXPECTED_R2A_SHA = "63dd2f91f078d6101e564c06d174e5772be11b82ba91e8a2c7416d9512dc6ee9"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if sha(R2A_PRED) != EXPECTED_R2A_SHA:
        raise RuntimeError("r2a_prediction_sha_mismatch")
    source_seal = json.loads((R2A / "prediction-seal.json").read_text())
    if not source_seal.get("sealed") or source_seal.get("prediction_sha256") != EXPECTED_R2A_SHA:
        raise RuntimeError("r2a_prediction_not_sealed")
    OUT.mkdir(parents=True, exist_ok=True)
    selector_source = ROOT / "src/pdf_retrieval_v4/bounded_rerank_input_selector.py"
    support_source = ROOT / "src/pdf_retrieval_v4/support_invariant_candidate_selector.py"
    records = []
    input_union_shortfall = 0
    for source in load_predictions(R2A_PRED):
        main_priority = build_priority_ranking(source["raw_family_v2"], source["structured_family_v2"])
        slot_priority = {
            slot_id: build_priority_ranking(trace["candidate_raw_fused"], trace["structured_family_v2"])
            for slot_id, trace in source["slot_deep_supply"].items()
        }
        if slot_priority:
            selected, composition = select_multi_slot_top100(slot_priority, main_priority)
        else:
            selected = select_single_slot_top100(main_priority)
            composition = {}
        deep_keys = set(source["deep_supply_candidate_keys"])
        selected_keys = [item["candidate_key"] for item in selected]
        if not set(selected_keys) <= deep_keys:
            raise RuntimeError(f"candidate_outside_deep_supply:{source['case_id']}")
        if len(selected_keys) != len(set(selected_keys)) or len(selected_keys) > RERANK_INPUT_BUDGET:
            raise RuntimeError(f"invalid_candidate_budget:{source['case_id']}")
        if [item["final_candidate_rank"] for item in selected] != list(range(1, len(selected) + 1)):
            raise RuntimeError(f"non_contiguous_candidate_rank:{source['case_id']}")
        input_union_shortfall += len(selected) < RERANK_INPUT_BUDGET
        records.append({
            "case_id": source["case_id"],
            "query_plan_id": source.get("query_plan_id"),
            "candidate_budget": RERANK_INPUT_BUDGET,
            "is_multi_slot": bool(slot_priority),
            "candidates": selected,
            "main_priority_ranking": main_priority,
            "slot_priority_rankings": slot_priority,
            "composition_audit": composition,
            "deep_supply_candidate_count": len(deep_keys),
        })
    if len(records) != 72:
        raise RuntimeError("prediction_count_must_equal_72")
    with PRED.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for record in records:
                zipped.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    input_hashes = {
        "r2a_prediction": EXPECTED_R2A_SHA,
        "r2a_prediction_seal": sha(R2A / "prediction-seal.json"),
        "r2a_input_integrity": sha(R2A / "input-integrity.json"),
        "strict_source_contract": sha(SIDECAR),
        "top100_selector": sha(selector_source),
        "support_invariant_selector": sha(support_source),
    }
    protocol = {
        "gate": "pdf_retrieval_v4_gate_08_r8_r2a_2",
        "schema": "pdf-retrieval-v4/bounded-rerank-input/v1",
        "retrieval_supply_horizon": 200,
        "slot_composition_horizon": SLOT_COMPOSITION_HORIZON,
        "candidate_budget": RERANK_INPUT_BUDGET,
        "slot_min_budget": SLOT_MIN_BUDGET,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "reference_answer_reads": 0,
        "expected_value_reads": 0,
        "bm25_searches": 0,
        "dense_searches": 0,
        "embedding_calls": 0,
        "index_reads": 0,
        "index_builds": 0,
        "bridge_runs": 0,
        "semantic_graph_runs": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "parameter_scan": False,
        "quota_scan": False,
        "topk_scan": False,
        "weight_scan": False,
        "production_writes": 0,
        "production_switch_allowed": False,
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "input-integrity.json", {"input_hashes": input_hashes, "all_inputs_exact": True})
    prediction_hash = sha(PRED)
    manifest = {
        "prediction_file": PRED.name,
        "prediction_count": len(records),
        "prediction_sha256": prediction_hash,
        "candidate_count_le_100": "72/72",
        "input_union_shortfall_cases": input_union_shortfall,
        "input_hashes": input_hashes,
    }
    write_json(OUT / "prediction-manifest.json", manifest)
    seal = {
        **protocol,
        **manifest,
        "sealed": True,
        "reranker_allowed": False,
    }
    write_json(OUT / "prediction-seal.json", seal)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
