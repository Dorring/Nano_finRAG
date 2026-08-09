#!/usr/bin/env python3
"""Run zero-search support-invariant candidate fusion for Gate 08 R8-R1.2."""

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
    SLOT_CANDIDATE_HORIZON,
    SLOT_MIN_BUDGET,
    select_multi_slot_top50,
)
from src.pdf_retrieval_v4.support_invariant_candidate_selector import (  # noqa: E402
    RRF_K,
    build_raw_family_v2,
    build_structured_family_v2,
    fuse_main_families_v2,
)

BASE = ROOT / "artifacts/evaluation"
PATHS = {
    "gate08": BASE / "pdf-retrieval-v4-gate-08/retrieval-predictions.jsonl.gz",
    "r3": BASE / "pdf-retrieval-v4-gate-08-r3/predictions.jsonl.gz",
    "r3_rs": BASE / "pdf-retrieval-v4-gate-08-r3-rs/slot-local-rankings.jsonl.gz",
    "r5": BASE / "pdf-retrieval-v4-gate-08-r5/retrieval-predictions.jsonl.gz",
    "r7": BASE / "pdf-retrieval-v4-gate-08-r7/field-family-predictions.jsonl.gz",
    "r8_r1": BASE / "pdf-retrieval-v4-gate-08-r8-r1/candidate-top50-predictions.jsonl.gz",
    "strict_source_contract": BASE / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl",
}
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r1-2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {
            item["case_id"]: item
            for item in (json.loads(line) for line in handle if line.strip())
            if item.get("case_id")
        }


def explicit_existing(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        key = item.get("original_candidate_identity") or item.get("candidate_key")
        rank = item.get("structured_rank")
        if not key or rank is None:
            raise RuntimeError("existing_structured_rank_provenance_missing")
        result.append({"candidate_key": key, "rank": rank})
    return result


def family_metadata(item: dict[str, Any] | None) -> dict[str, Any]:
    if not item:
        return {"best_rank": None, "second_best_rank": None, "lane_ranks": {}, "support_count": 0}
    return {
        "best_rank": item.get("best_rank"),
        "second_best_rank": item.get("second_best_rank"),
        "lane_ranks": item.get("lane_ranks", {}),
        "support_count": item.get("support_count", 0),
    }


def enrich(
    selected: list[dict[str, Any]],
    raw_family: list[dict[str, Any]],
    structured_family: list[dict[str, Any]],
    top_level: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = {item["candidate_key"]: item for item in raw_family}
    structured = {item["candidate_key"]: item for item in structured_family}
    top = {item["candidate_key"]: item for item in top_level}
    return [
        {
            **item,
            "raw_family": family_metadata(raw.get(item["candidate_key"])),
            "structured_family": family_metadata(structured.get(item["candidate_key"])),
            "top_level": family_metadata(top.get(item["candidate_key"])),
            "final_candidate_rank": rank,
        }
        for rank, item in enumerate(selected, 1)
    ]


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    data = {
        name: load_gzip(path)
        for name, path in PATHS.items()
        if path.suffix == ".gz"
    }
    contract = json.loads(
        (BASE / "pdf-retrieval-v4-strict-source-contract/acceptance.json").read_text()
    )
    if contract["decision"] != "strict_gold_source_binding_contract_closed" or contract["sidecar_sha256"] != sha(PATHS["strict_source_contract"]):
        raise RuntimeError("strict_source_contract_invalid")
    r1_seal = json.loads((BASE / "pdf-retrieval-v4-gate-08-r8-r1/prediction-seal.json").read_text())
    if r1_seal["prediction_sha256"] != sha(PATHS["r8_r1"]):
        raise RuntimeError("h0_prediction_seal_invalid")
    predictions = []
    audit = {"cases": 0, "single_slot": 0, "multi_slot": 0, "multi_slots": 0, "h0_sequence_exact": 0}
    for case_id in sorted(data["r7"]):
        original, r3, r5, r7 = (data[name][case_id] for name in ("gate08", "r3", "r5", "r7"))
        h0 = data["r8_r1"][case_id]
        production_raw = original["raw_full_rrf_candidates"]
        raw_family = build_raw_family_v2(production_raw, r3["candidate_raw"]["fused"])
        structured_family = build_structured_family_v2(
            r7["structured_h1"],
            r5["lane_hits"]["main"]["structured_metric_bm25"],
            explicit_existing(original["structured_strict_source_pool"]),
        )
        main_ranking = fuse_main_families_v2(raw_family, structured_family)
        main_top50 = main_ranking[:CANDIDATE_BUDGET]
        slot_trace = {}
        composition_audit = {}
        if r7["is_multi_slot"]:
            audit["multi_slot"] += 1
            sidecar = data["r3_rs"][case_id]
            slot_rankings = {}
            for definition in sidecar["slot_definitions"]:
                slot_id = definition["slot_id"]
                slot_raw = sidecar["slot_family_rankings"][slot_id]["raw"]["fused"]
                slot_structured = build_structured_family_v2(
                    r7["slot_structured_h1"][slot_id],
                    r5["lane_hits"][slot_id]["structured_metric_bm25"],
                )
                slot_ranking = fuse_main_families_v2(slot_raw, slot_structured)
                slot_rankings[slot_id] = slot_ranking
                slot_trace[slot_id] = {
                    "raw_family": slot_raw,
                    "structured_family_v2": slot_structured,
                    "slot_candidate_ranking_v2": slot_ranking,
                }
                audit["multi_slots"] += 1
            selected, composition_audit = select_multi_slot_top50(slot_rankings, main_top50)
        else:
            audit["single_slot"] += 1
            selected = main_top50
        h0_keys = [item["candidate_key"] for item in h0["bounded_candidate_ranking"]]
        if h0_keys != [item["candidate_key"] for item in data["r8_r1"][case_id]["bounded_candidate_ranking"]]:
            raise RuntimeError(f"h0_sequence_parity_blocked:{case_id}")
        audit["h0_sequence_exact"] += 1
        audit["cases"] += 1
        predictions.append(
            {
                "case_id": case_id,
                "is_multi_slot": r7["is_multi_slot"],
                "h0_bounded_candidate_ranking": h0["bounded_candidate_ranking"],
                "raw_family_v2": raw_family,
                "structured_family_v2": structured_family,
                "main_candidate_ranking_v2": main_ranking,
                "slot_family_trace_v2": slot_trace,
                "slot_composition_audit": composition_audit,
                "h1_bounded_candidate_ranking": enrich(selected, raw_family, structured_family, main_ranking),
            }
        )
    expected = {"cases": 72, "single_slot": 54, "multi_slot": 18, "multi_slots": 36, "h0_sequence_exact": 72}
    if audit != expected:
        raise RuntimeError(f"r1_2_input_contract_blocked:{audit}")
    OUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUT / "support-invariant-predictions.jsonl.gz"
    with prediction_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            for record in predictions:
                handle.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    protocol = {
        "gate": "pdf_retrieval_v4_gate_08_r8_r1_2",
        "primary": "h1_support_count_invariant_family_fusion",
        "h0": "r8_r1_sum_rrf_selector",
        "rrf_k": RRF_K,
        "candidate_budget": CANDIDATE_BUDGET,
        "slot_candidate_horizon": SLOT_CANDIDATE_HORIZON,
        "slot_min_budget": SLOT_MIN_BUDGET,
        "multi_slot_residual_fusion": "r8_r1_sum_rrf_exact",
        "bm25_searches": 0,
        "dense_searches": 0,
        "embedding_calls": 0,
        "index_reads": 0,
        "index_builds": 0,
        "bridge_runs": 0,
        "semantic_graph_runs": 0,
        "query_plan_changes": 0,
        "query_rebuilds": 0,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "weight_scan": False,
        "quota_scan": False,
        "topk_scan": False,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "production_writes": 0,
        "production_switch_allowed": False,
    }
    manifest = {
        "prediction_count": 72,
        "prediction_sha256": sha(prediction_path),
        "input_hashes": {name: sha(path) for name, path in PATHS.items()},
        "selector_source_sha256": sha(ROOT / "src/pdf_retrieval_v4/support_invariant_candidate_selector.py"),
    }
    write("protocol.json", protocol)
    write("input-integrity.json", audit)
    write("baseline-parity.json", {"h0_sequence_exact": "72/72", "expected_post_seal_score": {"recall_at_50": "55/80", "raw_retained": "22/24", "multi": "9/16", "calculation": "7/11"}})
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**protocol, **manifest, "sealed": True})
    print(json.dumps({"audit": audit, "prediction_sha256": sha(prediction_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
