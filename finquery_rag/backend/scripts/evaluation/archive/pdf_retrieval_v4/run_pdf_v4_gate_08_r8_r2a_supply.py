#!/usr/bin/env python3
"""Replay frozen candidate retrieval at the sole Top200 supply horizon."""

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
from src.pdf_retrieval_v4.candidate_field_index import CandidateFieldIndexReader  # noqa: E402
from src.pdf_retrieval_v4.candidate_field_query import build_field_queries  # noqa: E402
from src.pdf_retrieval_v4.candidate_query_builder import build_all_queries  # noqa: E402
from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader  # noqa: E402
from src.pdf_retrieval_v4.deep_candidate_supply import (  # noqa: E402
    RRF_K,
    SUPPLY_LANE_K,
    retrieve_deep_supply,
)
from src.pdf_retrieval_v4.serialization import query_plan_from_dict  # noqa: E402
from src.pdf_retrieval_v4.support_invariant_candidate_selector import (  # noqa: E402
    build_raw_family_v2,
    build_structured_family_v2,
    fuse_main_families_v2,
)

BASE = ROOT / "artifacts/evaluation"
PLANS = BASE / "pdf-retrieval-v4-gate-07/query-plan-predictions.json"
GENERAL = BASE / "pdf-retrieval-v4-gate-06-r4/candidate-indexes"
FIELDS = BASE / "pdf-retrieval-v4-gate-08-r5/field-indexes"
GATE08 = BASE / "pdf-retrieval-v4-gate-08/retrieval-predictions.jsonl.gz"
R12 = BASE / "pdf-retrieval-v4-gate-08-r8-r1-2/support-invariant-predictions.jsonl.gz"
CONTRACT = BASE / "pdf-retrieval-v4-strict-source-contract/strict-gold-source-bindings.jsonl"
OUT = BASE / "pdf-retrieval-v4-gate-08-r8-r2a"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def load_gzip(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {item["case_id"]: item for item in (json.loads(line) for line in handle if line.strip()) if item.get("case_id")}


def existing_structured(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_key": item["original_candidate_identity"],
            "rank": item["structured_rank"],
        }
        for item in items
        if item.get("original_candidate_identity") and item.get("structured_rank") is not None
    ]


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    r12_seal = json.loads((R12.parent / "prediction-seal.json").read_text())
    contract = json.loads((CONTRACT.parent / "acceptance.json").read_text())
    if r12_seal["prediction_sha256"] != sha(R12) or contract["sidecar_sha256"] != sha(CONTRACT):
        raise RuntimeError("r2a_frozen_input_seal_invalid")
    original = load_gzip(GATE08)
    plans_payload = json.loads(PLANS.read_text())
    plan_items = plans_payload.get("plans") or plans_payload.get("cases") or []
    plans = [(str(item["case_id"]), query_plan_from_dict(item["plan"])) for item in plan_items]
    if len(plans) != 72:
        raise RuntimeError("r2a_query_plan_count_invalid")
    general_reader = CandidateViewIndexReader(GENERAL, rrf_k=RRF_K)
    field_reader = CandidateFieldIndexReader(FIELDS)
    predictions = []
    totals = {"bm25_searches": 0, "dense_searches": 0, "field_searches": 0, "logical_queries": 0}
    for number, (case_id, plan) in enumerate(plans, 1):
        queries = build_all_queries(plan)
        scope = set(plan.document_scope)
        main_query = queries["raw_question"][0]
        main_lanes, candidate_raw, structured_h1, counts = retrieve_deep_supply(
            general_reader,
            field_reader,
            general_query=main_query,
            field_queries=build_field_queries(plan),
            document_scope=scope,
        )
        for key, value in counts.items():
            totals[key] += value
        totals["logical_queries"] += 1
        production_raw = original[case_id]["raw_full_rrf_candidates"]
        raw_family = build_raw_family_v2(production_raw, candidate_raw)
        structured_family = build_structured_family_v2(
            structured_h1,
            main_lanes["structured_metric_bm25"],
            existing_structured(original[case_id]["structured_strict_source_pool"]),
        )
        main_ranking = fuse_main_families_v2(raw_family, structured_family)
        deep_keys = {item["candidate_key"] for items in main_lanes.values() for item in items}
        deep_keys.update(item["candidate_key"] for item in main_ranking)
        multi = len(plan.operand_slots) >= 2
        slot_trace = {}
        composition_audit = {}
        if multi:
            slot_rankings = {}
            for slot in plan.operand_slots:
                query = (queries.get("slots", {}).get(slot.slot_id) or [""])[0]
                lanes, slot_candidate_raw, slot_h1, slot_counts = retrieve_deep_supply(
                    general_reader,
                    field_reader,
                    general_query=query,
                    field_queries=build_field_queries(plan, slot),
                    document_scope=scope,
                )
                for key, value in slot_counts.items():
                    totals[key] += value
                totals["logical_queries"] += 1
                slot_structured = build_structured_family_v2(
                    slot_h1, lanes["structured_metric_bm25"]
                )
                slot_ranking = fuse_main_families_v2(slot_candidate_raw, slot_structured)
                slot_rankings[slot.slot_id] = slot_ranking
                deep_keys.update(item["candidate_key"] for items in lanes.values() for item in items)
                deep_keys.update(item["candidate_key"] for item in slot_ranking)
                slot_trace[slot.slot_id] = {
                    "lane_hits": lanes,
                    "candidate_raw_fused": slot_candidate_raw,
                    "structured_h1": slot_h1,
                    "structured_family_v2": slot_structured,
                    "slot_candidate_ranking_v2": slot_ranking,
                }
            selected, composition_audit = select_multi_slot_top50(
                slot_rankings, main_ranking[:SLOT_CANDIDATE_HORIZON]
            )
        else:
            selected = main_ranking[:CANDIDATE_BUDGET]
        if len(selected) != CANDIDATE_BUDGET:
            raise RuntimeError(f"r2a_candidate_budget_invalid:{case_id}:{len(selected)}")
        predictions.append(
            {
                "case_id": case_id,
                "query_plan_id": plan.plan_id,
                "is_multi_slot": multi,
                "main_lane_hits": main_lanes,
                "candidate_raw_fused": candidate_raw,
                "structured_h1": structured_h1,
                "raw_family_v2": raw_family,
                "structured_family_v2": structured_family,
                "deep_main_ranking": main_ranking,
                "slot_deep_supply": slot_trace,
                "slot_composition_audit": composition_audit,
                "deep_supply_candidate_keys": sorted(deep_keys),
                "bounded_candidate_top50": selected,
            }
        )
        if number % 12 == 0:
            print(f"[{number}/72] {case_id}")
    unique_computations = len(general_reader._query_vector_cache)
    logical_embedding_requests = totals["dense_searches"]
    general_reader.close()
    OUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUT / "deep-supply-predictions.jsonl.gz"
    with prediction_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            for record in predictions:
                handle.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    protocol = {
        "gate": "pdf_retrieval_v4_gate_08_r8_r2a",
        "retrieval_supply_horizon": SUPPLY_LANE_K,
        "candidate_budget": CANDIDATE_BUDGET,
        "slot_composition_horizon": SLOT_CANDIDATE_HORIZON,
        "slot_min_budget": SLOT_MIN_BUDGET,
        "rrf_k": RRF_K,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_revision": EMBEDDING_REVISION,
        "bridge_runs": 0,
        "semantic_graph_runs": 0,
        "index_builds": 0,
        "parameter_scan": False,
        "topk_scan": False,
        "weight_scan": False,
        "model_scan": False,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "production_writes": 0,
        "production_switch_allowed": False,
    }
    search_counts = {
        **totals,
        "total_searches": totals["bm25_searches"] + totals["dense_searches"],
        "logical_embedding_requests": logical_embedding_requests,
        "unique_embedding_computations": unique_computations,
        "embedding_cache_hits": logical_embedding_requests - unique_computations,
    }
    input_hashes = {
        "r8_r1_2_prediction": sha(R12),
        "strict_source_contract": sha(CONTRACT),
        "query_plans": sha(PLANS),
        "candidate_index": tree_sha(GENERAL),
        "field_index": tree_sha(FIELDS),
        "query_builder": sha(ROOT / "src/pdf_retrieval_v4/candidate_query_builder.py"),
        "field_query_builder": sha(ROOT / "src/pdf_retrieval_v4/candidate_field_query.py"),
        "support_invariant_selector": sha(ROOT / "src/pdf_retrieval_v4/support_invariant_candidate_selector.py"),
    }
    manifest = {"prediction_count": 72, "prediction_sha256": sha(prediction_path), "input_hashes": input_hashes}
    write("protocol.json", protocol)
    write("input-integrity.json", {"input_hashes": input_hashes, "candidate_count": 38319, "grade_a_count": 19500, "raw_production_ranking_immutable": True})
    write("search-counts.json", search_counts)
    write("embedding-cache.json", {key: search_counts[key] for key in ("logical_embedding_requests", "unique_embedding_computations", "embedding_cache_hits")})
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**protocol, **manifest, **search_counts, "sealed": True})
    print(json.dumps({"prediction_sha256": sha(prediction_path), "search_counts": search_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
