#!/usr/bin/env python3
"""Build sealed, bounded Gate 08 R8-R1 candidate Top50 rankings."""

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

from src.pdf_retrieval_v4.bounded_candidate_selector import (  # noqa: E402
    CANDIDATE_BUDGET,
    RRF_K,
    SLOT_CANDIDATE_HORIZON,
    SLOT_MIN_BUDGET,
    build_raw_family,
    build_structured_family,
    select_multi_slot_top50,
    select_single_slot_top50,
)

BASE = ROOT / "artifacts/evaluation"
PATHS = {
    "gate08": BASE / "pdf-retrieval-v4-gate-08/retrieval-predictions.jsonl.gz",
    "r3": BASE / "pdf-retrieval-v4-gate-08-r3/predictions.jsonl.gz",
    "r3_rs": BASE / "pdf-retrieval-v4-gate-08-r3-rs/slot-local-rankings.jsonl.gz",
    "r5": BASE / "pdf-retrieval-v4-gate-08-r5/retrieval-predictions.jsonl.gz",
    "r7": BASE / "pdf-retrieval-v4-gate-08-r7/field-family-predictions.jsonl.gz",
}
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            item["case_id"]: item
            for item in (json.loads(line) for line in handle if line.strip())
            if item.get("case_id")
        }


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def explicit_existing(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        key = item.get("original_candidate_identity") or item.get("candidate_key")
        rank = item.get("structured_rank")
        if not key or rank is None:
            raise RuntimeError("existing_structured_rank_provenance_missing")
        result.append({"candidate_key": key, "rank": rank})
    return result


def trace(
    selected: list[dict[str, Any]],
    raw_family: list[dict[str, Any]],
    structured_family: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = {item["candidate_key"]: item for item in raw_family}
    structured = {item["candidate_key"]: item for item in structured_family}
    output = []
    for item in selected:
        key = item["candidate_key"]
        raw_item = raw.get(key, {})
        structured_item = structured.get(key, {})
        raw_lanes = raw_item.get("lane_ranks", {})
        structured_lanes = structured_item.get("lane_ranks", {})
        output.append(
            {
                **item,
                "production_raw_rank": raw_lanes.get("production_raw"),
                "candidate_raw_rank": raw_lanes.get("candidate_raw"),
                "structured_h1_rank": structured_lanes.get("structured_h1"),
                "metric_rank": structured_lanes.get("structured_metric"),
                "existing_structured_rank": structured_lanes.get("existing_structured"),
                "raw_family_rank": raw_item.get("rank"),
                "structured_family_rank": structured_item.get("rank"),
            }
        )
    return output


def main() -> int:
    data = {name: load(path) for name, path in PATHS.items()}
    case_ids = sorted(data["r7"])
    full_inputs = ("gate08", "r3", "r5", "r7")
    if len(case_ids) != 72 or any(
        not set(case_ids).issubset(data[name]) for name in full_inputs
    ):
        raise RuntimeError("candidate_top50_case_coverage_blocked")
    predictions = []
    audit = {
        "cases": 72,
        "single_slot": 0,
        "multi_slot": 0,
        "multi_slots": 0,
        "production_raw_rankable": 0,
        "existing_structured_rankable": 0,
        "missing_slot_inputs": [],
    }
    for case_id in case_ids:
        original = data["gate08"][case_id]
        r3 = data["r3"][case_id]
        r5 = data["r5"][case_id]
        r7 = data["r7"][case_id]
        production_raw = original["raw_full_rrf_candidates"]
        if any(item.get("stage_rank") is None for item in production_raw):
            raise RuntimeError(f"production_raw_rank_provenance_missing:{case_id}")
        audit["production_raw_rankable"] += 1
        existing = explicit_existing(original["structured_strict_source_pool"])
        audit["existing_structured_rankable"] += 1
        raw_family = build_raw_family(production_raw, r3["candidate_raw"]["fused"])
        metric = r5["lane_hits"]["main"]["structured_metric_bm25"]
        structured_family = build_structured_family(
            r7["structured_h1"], metric, existing
        )
        main_ranking = select_single_slot_top50(raw_family, structured_family)
        slot_trace: dict[str, Any] = {}
        composition_audit: dict[str, Any] = {}
        if r7["is_multi_slot"]:
            audit["multi_slot"] += 1
            sidecar = data["r3_rs"].get(case_id)
            if not sidecar:
                raise RuntimeError(f"missing_slot_sidecar:{case_id}")
            slot_rankings = {}
            for slot in sidecar["slot_definitions"]:
                slot_id = slot["slot_id"]
                try:
                    slot_raw = sidecar["slot_family_rankings"][slot_id]["raw"]["fused"]
                    slot_h1 = r7["slot_structured_h1"][slot_id]
                    slot_metric = r5["lane_hits"][slot_id]["structured_metric_bm25"]
                except KeyError:
                    audit["missing_slot_inputs"].append(f"{case_id}:{slot_id}")
                    continue
                slot_structured = build_structured_family(slot_h1, slot_metric)
                slot_ranking = select_single_slot_top50(slot_raw, slot_structured)
                slot_rankings[slot_id] = slot_ranking
                slot_trace[slot_id] = {
                    "raw_family": slot_raw,
                    "structured_family": slot_structured,
                    "candidate_ranking": slot_ranking,
                }
                audit["multi_slots"] += 1
            if len(slot_rankings) != len(sidecar["slot_definitions"]):
                raise RuntimeError(f"candidate_top50_slot_input_blocked:{case_id}")
            selected, composition_audit = select_multi_slot_top50(
                slot_rankings, main_ranking
            )
        else:
            audit["single_slot"] += 1
            selected = main_ranking
        if len(selected) > CANDIDATE_BUDGET:
            raise RuntimeError(f"candidate_budget_exceeded:{case_id}")
        predictions.append(
            {
                "case_id": case_id,
                "is_multi_slot": r7["is_multi_slot"],
                "raw_family": raw_family,
                "structured_family": structured_family,
                "main_candidate_ranking": main_ranking,
                "slot_rankings": slot_trace,
                "slot_composition_audit": composition_audit,
                "bounded_candidate_ranking": trace(
                    selected, raw_family, structured_family
                ),
            }
        )
    expected_audit = {
        "cases": 72,
        "single_slot": 54,
        "multi_slot": 18,
        "multi_slots": 36,
        "production_raw_rankable": 72,
        "existing_structured_rankable": 72,
        "missing_slot_inputs": [],
    }
    if audit != expected_audit:
        raise RuntimeError(f"candidate_top50_input_contract_blocked:{audit}")
    OUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUT / "candidate-top50-predictions.jsonl.gz"
    with prediction_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            for record in predictions:
                handle.write(
                    (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
                )
    protocol = {
        "gate": "pdf_retrieval_v4_gate_08_r8_r1",
        "selector": "family_normalized_rrf",
        "rrf_k": RRF_K,
        "candidate_budget": CANDIDATE_BUDGET,
        "slot_min_budget": SLOT_MIN_BUDGET,
        "slot_candidate_horizon": SLOT_CANDIDATE_HORIZON,
        "bm25_searches": 0,
        "dense_searches": 0,
        "embedding_calls": 0,
        "index_reads": 0,
        "index_builds": 0,
        "bridge_changes": 0,
        "query_changes": 0,
        "query_plan_changes": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "gold_reads_before_seal": 0,
        "parameter_scan": False,
        "quota_scan": False,
        "weight_scan": False,
        "topk_scan": False,
        "production_switch_allowed": False,
    }
    manifest = {
        "prediction_count": 72,
        "prediction_sha256": sha(prediction_path),
        "input_hashes": {name: sha(path) for name, path in PATHS.items()},
        "selector_source_sha256": sha(
            ROOT / "src/pdf_retrieval_v4/bounded_candidate_selector.py"
        ),
    }
    write("protocol.json", protocol)
    write("ranking-provenance-audit.json", audit)
    write("input-integrity.json", {"inputs": manifest["input_hashes"], "r7_artifacts_immutable": True})
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**protocol, **manifest, "sealed": True})
    print(json.dumps({"audit": audit, "prediction_sha256": sha(prediction_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
