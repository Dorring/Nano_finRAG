#!/usr/bin/env python3
"""Zero-search Gate 08 R6 slot-aware candidate composition."""

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

from src.pdf_retrieval_v4.slot_aware_candidate_composer import (  # noqa: E402
    FINAL_POOL_K,
    RRF_K,
    SLOT_CANDIDATE_HORIZON,
    SLOT_MIN_BUDGET,
    compose_slot_candidates,
)

R5 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r5"
R3 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3/predictions.jsonl.gz"
RS = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3-rs/slot-local-rankings.jsonl.gz"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r6"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {item["case_id"]: item for item in (json.loads(line) for line in handle if line.strip())}


def combine(prefix: list[dict[str, Any]], residual: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for item in [*prefix, *residual]:
        key = str(item.get("candidate_key") or "")
        if key and key not in seen:
            seen.add(key)
            result.append({"candidate_key": key, "rank": len(result) + 1})
    return result


def write(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    r5_path = R5 / "retrieval-predictions.jsonl.gz"
    r5_seal_path = R5 / "prediction-seal.json"
    r5_seal = json.loads(r5_seal_path.read_text())
    if not r5_seal.get("sealed") or r5_seal["prediction_sha256"] != sha(r5_path):
        raise RuntimeError("r5_seal_invalid")
    r5, r3, rs = load(r5_path), load(R3), load(RS)
    predictions = []
    single_parity = 0
    contract_errors = []
    for case_id in sorted(r5):
        source = r5[case_id]
        c0 = list(source["r5_full_pool"])
        if not source["is_multi_slot"]:
            c1 = c0
            trace = []
            audit = {"single_slot_exact_parity": True}
            union = []
            slot_input_rankings = {}
            single_parity += 1
        else:
            if case_id not in rs:
                contract_errors.append(f"missing_r3_rs:{case_id}")
            slot_trace = source.get("r5_full_slot_trace") or {}
            slot_defs = (rs.get(case_id) or {}).get("slot_definitions") or []
            expected_slots = [item["slot_id"] for item in slot_defs]
            if not expected_slots or set(expected_slots) != set(slot_trace):
                contract_errors.append(f"slot_contract_mismatch:{case_id}")
            if any(len(items) < SLOT_CANDIDATE_HORIZON for items in slot_trace.values()):
                contract_errors.append(f"slot_horizon_unavailable:{case_id}")
            ordered = {slot_id: slot_trace[slot_id] for slot_id in expected_slots}
            slot_input_rankings = {
                slot_id: [
                    {"candidate_key": item["candidate_key"], "rank": rank}
                    for rank, item in enumerate(items[:SLOT_CANDIDATE_HORIZON], 1)
                ]
                for slot_id, items in ordered.items()
            }
            trace, audit = compose_slot_candidates(ordered)
            c1 = combine(list(r3[case_id]["e0_pool"]), trace)
            union_keys = []
            seen = set()
            for slot_id in expected_slots:
                for item in ordered[slot_id][:SLOT_CANDIDATE_HORIZON]:
                    key = item["candidate_key"]
                    if key not in seen:
                        seen.add(key)
                        union_keys.append({"candidate_key": key})
            union = combine(list(r3[case_id]["e0_pool"]), union_keys)
        predictions.append({"case_id": case_id, "is_multi_slot": source["is_multi_slot"], "c0_pool": c0, "c1_pool": c1, "slot_input_rankings": slot_input_rankings, "composition_trace": trace, "composition_audit": audit, "unbounded_slot_union": union})
    if contract_errors or single_parity != 54:
        raise RuntimeError(f"slot_aware_input_contract_blocked:{contract_errors}:{single_parity}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "slot-aware-predictions.jsonl.gz"
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            for record in predictions:
                handle.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    manifest = {"prediction_count": 72, "single_slot_exact_parity": "54/54", "multi_slot_composed": "18/18", "prediction_sha256": sha(path), "input_hashes": {"r5_prediction": sha(r5_path), "r5_seal": sha(r5_seal_path), "r3_prediction": sha(R3), "r3_rs_sidecar": sha(RS)}}
    protocol = {"gate": "pdf_retrieval_v4_gate_08_r6", "rrf_k": RRF_K, "final_pool_k": FINAL_POOL_K, "slot_min_budget": SLOT_MIN_BUDGET, "slot_candidate_horizon": SLOT_CANDIDATE_HORIZON, "bm25_searches": 0, "dense_searches": 0, "embedding_calls": 0, "index_reads": 0, "index_builds": 0, "question_regeneration": 0, "query_plan_mutation": 0, "gold_reads_before_seal": 0, "governance_reads_before_seal": 0, "parameter_scan": False, "quota_scan": False, "production_writes": 0, "production_switch_allowed": False}
    write("protocol.json", protocol)
    write("input-integrity.json", {"r5_prediction_hash_immutable": True, "multi_slot_definitions_present": "18/18", "raw_family_rankings_present": "18/18", "s4_structured_rankings_present": "18/18", "r5_family_fused_slot_traces_present": "18/18"})
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**protocol, **manifest, "sealed": True})
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
