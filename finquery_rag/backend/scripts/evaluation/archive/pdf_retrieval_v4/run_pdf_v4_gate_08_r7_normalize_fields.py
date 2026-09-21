#!/usr/bin/env python3
"""Run zero-search Gate 08 R7 hierarchical field normalization."""

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

from src.pdf_retrieval_v4.field_family_normalizer import (  # noqa: E402
    RRF_K,
    fuse_flat_h0,
    fuse_hierarchical_structured,
)
from src.pdf_retrieval_v4.lane_preserving_fusion import (  # noqa: E402
    fuse_multi_slot_families,
    fuse_single_slot_families,
)
from src.pdf_retrieval_v4.slot_aware_candidate_composer import (  # noqa: E402
    FINAL_POOL_K,
    SLOT_CANDIDATE_HORIZON,
    SLOT_MIN_BUDGET,
    compose_slot_candidates,
)

R5_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r5"
R6_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r6"
R3 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3/predictions.jsonl.gz"
RS = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3-rs/slot-local-rankings.jsonl.gz"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r7"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {item["case_id"]: item for item in (json.loads(line) for line in handle if line.strip())}


def sequence(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("candidate_key") or "") for item in items]


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


def slot_pipeline(
    case_id: str,
    r5: dict[str, Any],
    r3: dict[str, Any],
    rs: dict[str, Any],
    structured_by_slot: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    slots = [item["slot_id"] for item in rs[case_id]["slot_definitions"]]
    family_input = {
        slot_id: {
            "raw": rs[case_id]["slot_family_rankings"][slot_id]["raw"],
            "structured": {"fused": structured_by_slot[slot_id]},
        }
        for slot_id in slots
    }
    _, family_traces = fuse_multi_slot_families(family_input)
    ordered_traces = {slot_id: family_traces[slot_id] for slot_id in slots}
    composition, composition_audit = compose_slot_candidates(ordered_traces)
    return combine(r3[case_id]["e0_pool"], composition), family_traces, {
        "trace": composition,
        "audit": composition_audit,
    }


def main() -> int:
    r5_path = R5_DIR / "retrieval-predictions.jsonl.gz"
    r5_seal_path = R5_DIR / "prediction-seal.json"
    r6_path = R6_DIR / "slot-aware-predictions.jsonl.gz"
    r6_seal_path = R6_DIR / "prediction-seal.json"
    for prediction, seal_path in ((r5_path, r5_seal_path), (r6_path, r6_seal_path)):
        seal = json.loads(seal_path.read_text())
        if not seal.get("sealed") or seal["prediction_sha256"] != sha(prediction):
            raise RuntimeError(f"invalid_input_seal:{prediction}")
    r5, r6, r3, rs = load(r5_path), load(r6_path), load(R3), load(RS)
    predictions = []
    structured_parity = 0
    full_parity = 0
    preflight = {"single_cases": 0, "multi_cases": 0, "multi_slots": 0, "missing_lanes": []}
    for case_id in sorted(r5):
        source = r5[case_id]
        main_lanes = source["lane_hits"].get("main") or {}
        required = {
            "candidate_structured_bm25",
            "candidate_structured_dense",
            "structured_metric_bm25",
            "structured_axis_bm25",
            "structured_context_bm25",
            "structured_evidence_bm25",
        }
        if set(main_lanes) != required:
            preflight["missing_lanes"].append(f"main:{case_id}")
        field_family, h1_main = fuse_hierarchical_structured(main_lanes)
        h0_main = fuse_flat_h0(main_lanes)
        if sequence(h0_main) != sequence(source["structured_family_rankings"]["s4"]):
            raise RuntimeError(f"h0_structured_parity_blocked:{case_id}")
        structured_parity += 1
        slot_field_family = {}
        slot_h1 = {}
        family_trace = {}
        composition_trace = {}
        if source["is_multi_slot"]:
            preflight["multi_cases"] += 1
            h0_slots = {}
            for slot_id, variants in source["slot_structured_family_rankings"].items():
                lane_hits = source["lane_hits"].get(slot_id) or {}
                if set(lane_hits) != required:
                    preflight["missing_lanes"].append(f"slot:{case_id}:{slot_id}")
                slot_field_family[slot_id], slot_h1[slot_id] = fuse_hierarchical_structured(lane_hits)
                h0_slots[slot_id] = fuse_flat_h0(lane_hits)
                if sequence(h0_slots[slot_id]) != sequence(variants["s4"]):
                    raise RuntimeError(f"h0_slot_parity_blocked:{case_id}:{slot_id}")
                preflight["multi_slots"] += 1
            h0_full, _, _ = slot_pipeline(case_id, source, r3, rs, h0_slots)
            h1_full, family_trace, composition_trace = slot_pipeline(case_id, source, r3, rs, slot_h1)
        else:
            preflight["single_cases"] += 1
            h0_residual = fuse_single_slot_families(r3[case_id]["candidate_raw"]["fused"], h0_main)
            h1_residual = fuse_single_slot_families(r3[case_id]["candidate_raw"]["fused"], h1_main)
            h0_full = combine(r3[case_id]["e0_pool"], h0_residual)
            h1_full = combine(r3[case_id]["e0_pool"], h1_residual)
            family_trace = {"main": h1_residual}
        if sequence(h0_full) != sequence(r6[case_id]["c1_pool"]):
            raise RuntimeError(f"h0_full_parity_blocked:{case_id}")
        full_parity += 1
        predictions.append({"case_id": case_id, "is_multi_slot": source["is_multi_slot"], "field_family_ranking": field_family, "structured_h0": h0_main, "structured_h1": h1_main, "slot_field_family_rankings": slot_field_family, "slot_structured_h1": slot_h1, "family_fusion_trace": family_trace, "slot_composition_trace": composition_trace, "h0_full_pool": h0_full, "r7_full_pool": h1_full})
    if preflight != {"single_cases": 54, "multi_cases": 18, "multi_slots": 36, "missing_lanes": []}:
        raise RuntimeError(f"field_normalization_input_contract_blocked:{preflight}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "field-family-predictions.jsonl.gz"
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            for record in predictions:
                handle.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    source_files = {
        "field_normalizer": ROOT / "src/pdf_retrieval_v4/field_family_normalizer.py",
        "candidate_rrf": ROOT / "src/pdf_retrieval_v4/candidate_rrf.py",
        "r4_family_fusion": ROOT / "src/pdf_retrieval_v4/lane_preserving_fusion.py",
        "r6_slot_composer": ROOT / "src/pdf_retrieval_v4/slot_aware_candidate_composer.py",
    }
    manifest = {"prediction_count": 72, "prediction_sha256": sha(path), "input_hashes": {"r5_prediction": sha(r5_path), "r5_seal": sha(r5_seal_path), "r3_rs": sha(RS), "r6_prediction": sha(r6_path), "r6_seal": sha(r6_seal_path)}, "source_hashes": {name: sha(path) for name, path in source_files.items()}}
    protocol = {"gate": "pdf_retrieval_v4_gate_08_r7", "primary": "h1_hierarchical_field_family", "rrf_k": RRF_K, "final_pool_k": FINAL_POOL_K, "slot_candidate_horizon": SLOT_CANDIDATE_HORIZON, "slot_min_budget": SLOT_MIN_BUDGET, "bm25_searches": 0, "dense_searches": 0, "embedding_calls": 0, "index_reads": 0, "index_builds": 0, "query_rebuilds": 0, "query_plan_mutations": 0, "gold_reads_before_seal": 0, "governance_reads_before_seal": 0, "parameter_scan": False, "weight_scan": False, "topk_scan": False, "architecture_scan": False, "production_writes": 0, "production_switch_allowed": False}
    write("protocol.json", protocol)
    write("input-integrity.json", {"preflight": preflight, "r5_prediction_immutable": True, "r6_prediction_immutable": True})
    write("baseline-parity.json", {"h0_structured_sequence_exact": f"{structured_parity}/72", "h0_full_sequence_exact": f"{full_parity}/72"})
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**protocol, **manifest, "sealed": True})
    print(json.dumps({"preflight": preflight, "structured_parity": structured_parity, "full_parity": full_parity, "prediction_sha256": sha(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
