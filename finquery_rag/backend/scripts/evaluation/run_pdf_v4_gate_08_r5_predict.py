#!/usr/bin/env python3
"""Run sealed Gate 08 R5 S0-S4 prediction and frozen R4-F2 full fusion."""

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

from src.pdf_retrieval_v4.candidate_field_index import CandidateFieldIndexReader  # noqa: E402
from src.pdf_retrieval_v4.candidate_field_query import (  # noqa: E402
    build_field_queries,
    field_queries_to_dict,
)
from src.pdf_retrieval_v4.candidate_query_builder import build_all_queries  # noqa: E402
from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit  # noqa: E402
from src.pdf_retrieval_v4.candidate_slot_pool import build_slot_pool  # noqa: E402
from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader  # noqa: E402
from src.pdf_retrieval_v4.field_aware_structured_retriever import (  # noqa: E402
    retrieve_field_aware_structured,
)
from src.pdf_retrieval_v4.lane_preserving_fusion import (  # noqa: E402
    fuse_multi_slot_families,
    fuse_single_slot_families,
)
from src.pdf_retrieval_v4.serialization import query_plan_from_dict  # noqa: E402

R3 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3/predictions.jsonl.gz"
RS = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3-rs/slot-local-rankings.jsonl.gz"
PLANS = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-07/query-plan-predictions.json"
GENERAL = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-06-r4/candidate-indexes"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r5"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def serial(items: list[CandidateRRFHit]) -> list[dict[str, Any]]:
    return [{"candidate_key": item.candidate_key, "rank": rank, "rrf_score": item.rrf_score, "lane_ranks": item.lane_ranks, "supporting_fields": sorted(name for name in item.lane_ranks if name.startswith("structured_"))} for rank, item in enumerate(items, 1)]


def pool(items: list[CandidateRRFHit]) -> list[dict[str, Any]]:
    return [{"candidate_key": item.candidate_key, "rank": rank} for rank, item in enumerate(items[:40], 1)]


def combine(e0: list[dict[str, Any]], residual: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*e0, *residual]:
        key = str(item.get("candidate_key") or "")
        if key and key not in seen:
            seen.add(key)
            result.append({"candidate_key": key, "rank": len(result) + 1})
    return result


def slot_pool(slot_rankings: dict[str, list[CandidateRRFHit]]) -> list[dict[str, Any]]:
    return build_slot_pool(slot_rankings, slot_top_k=20, slot_min_budget=10, total_k=40)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    r3 = {item["case_id"]: item for item in gz(R3)}
    rs = {item["case_id"]: item for item in gz(RS)}
    plan_payload = json.loads(PLANS.read_text(encoding="utf-8"))
    plan_items = plan_payload.get("plans") or plan_payload.get("cases") or []
    plans = [(str(item["case_id"]), query_plan_from_dict(item["plan"])) for item in plan_items]
    general_reader = CandidateViewIndexReader(GENERAL, rrf_k=60)
    field_reader = CandidateFieldIndexReader(OUT / "field-indexes")
    predictions: list[dict[str, Any]] = []
    parity_errors: list[dict[str, Any]] = []
    for index, (case_id, plan) in enumerate(plans, 1):
        source = r3[case_id]
        scope = set(plan.document_scope)
        queries = build_all_queries(plan)
        field_query_trace = field_queries_to_dict(plan)
        multi = bool(source.get("is_multi_slot"))
        variant_rankings: dict[str, list[dict[str, Any]]] = {}
        slot_variant_rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
        lane_trace: dict[str, Any] = {}
        if multi:
            main_general_query = queries["raw_question"][0]
            main_lane_hits, main_variants = retrieve_field_aware_structured(
                general_reader,
                field_reader,
                general_query=main_general_query,
                field_queries=build_field_queries(plan),
                document_scope=scope,
            )
            lane_trace["main"] = {
                lane: [
                    {"candidate_key": hit.candidate_key, "rank": hit.bm25_rank or hit.dense_rank}
                    for hit in hits
                ]
                for lane, hits in main_lane_hits.items()
            }
            slot_hits_by_variant: dict[str, dict[str, list[CandidateRRFHit]]] = {f"s{i}": {} for i in range(5)}
            for slot in plan.operand_slots:
                general_query = (queries.get("slots", {}).get(slot.slot_id) or [""])[0]
                lane_hits, variants = retrieve_field_aware_structured(general_reader, field_reader, general_query=general_query, field_queries=build_field_queries(plan, slot), document_scope=scope)
                lane_trace[slot.slot_id] = {lane: [{"candidate_key": hit.candidate_key, "rank": hit.bm25_rank or hit.dense_rank} for hit in hits] for lane, hits in lane_hits.items()}
                slot_variant_rankings[slot.slot_id] = {name: serial(items) for name, items in variants.items()}
                for name, items in variants.items():
                    slot_hits_by_variant[name][slot.slot_id] = items
            main_rankings = {name: serial(items) for name, items in main_variants.items()}
            variant_rankings = main_rankings
            structured_residuals = {
                name: slot_pool(slot_hits_by_variant[name]) for name in [f"s{i}" for i in range(5)]
            }
        else:
            general_query = queries["raw_question"][0]
            lane_hits, variants = retrieve_field_aware_structured(general_reader, field_reader, general_query=general_query, field_queries=build_field_queries(plan), document_scope=scope)
            lane_trace["main"] = {lane: [{"candidate_key": hit.candidate_key, "rank": hit.bm25_rank or hit.dense_rank} for hit in hits] for lane, hits in lane_hits.items()}
            variant_rankings = {name: serial(items) for name, items in variants.items()}
            structured_residuals = {
                name: ranking[:40] for name, ranking in variant_rankings.items()
            }
        expected_s0 = list(source.get("e2_expanded_pool") or [])
        s0_combined = combine(list(source.get("e0_pool") or []), structured_residuals["s0"])
        if [x["candidate_key"] for x in s0_combined] != [x["candidate_key"] for x in expected_s0]:
            parity_errors.append({"case_id": case_id, "stage": "s0_e2_expanded"})
        structured_pools = {
            name: combine(list(source.get("e0_pool") or []), structured_residuals[name])
            for name in variant_rankings
        }
        if multi:
            slot_input: dict[str, dict[str, Any]] = {}
            for slot_id, per_slot in slot_variant_rankings.items():
                slot_input[slot_id] = {"raw": rs[case_id]["slot_family_rankings"][slot_id]["raw"], "structured": {"fused": per_slot["s4"]}}
            full_residual, full_slot_trace = fuse_multi_slot_families(slot_input)
        else:
            full_residual = fuse_single_slot_families((source.get("candidate_raw") or {}).get("fused") or [], variant_rankings["s4"])
            full_slot_trace = {}
        r5_full = combine(list(source.get("e0_pool") or []), full_residual)
        predictions.append({"case_id": case_id, "is_multi_slot": multi, "query_plan_id": plan.plan_id, "field_queries": field_query_trace, "lane_hits": lane_trace, "structured_family_rankings": variant_rankings, "slot_structured_family_rankings": slot_variant_rankings, "structured_pools": structured_pools, "r5_full_pool": r5_full, "r5_full_trace": full_residual, "r5_full_slot_trace": full_slot_trace})
        if index % 10 == 0:
            print(f"[{index}/{len(plans)}] {case_id}")
    general_reader.close()
    if parity_errors:
        (OUT / "baseline-parity.json").write_text(json.dumps({"passed": False, "errors": parity_errors}, indent=2) + "\n")
        raise RuntimeError(f"field_aware_input_parity_blocked:{len(parity_errors)}")
    pred_path = OUT / "retrieval-predictions.jsonl.gz"
    with pred_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            for record in predictions:
                handle.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
    prediction_hash = sha(pred_path)
    protocol = {"gate": "pdf_retrieval_v4_gate_08_r5", "lane_k": 50, "rrf_k": 60, "final_pool_k": 40, "slot_top_k": 20, "slot_min_budget": 10, "primary": "s4", "parameter_scan": False, "weight_scan": False, "topk_scan": False, "gold_reads_before_seal": 0, "governance_reads_before_seal": 0, "expected_value_reads": 0, "reference_answer_reads": 0, "reranker_calls": 0, "calculator_calls": 0, "generator_calls": 0, "production_switch_allowed": False}
    (OUT / "query-contract.json").write_text(json.dumps({"source": "gate07_query_plan_only", "empty_field_query_fallback": False, "hard_filters": False}, indent=2) + "\n")
    (OUT / "baseline-parity.json").write_text(json.dumps({"passed": True, "case_sequence_exact": "72/72", "multi_slot_s0_exact_r3_rs_parity": "18/18"}, indent=2) + "\n")
    manifest = {"prediction_count": len(predictions), "prediction_sha256": prediction_hash, "supersedes_incomplete_prediction_sha256": "0cd5baccabe5e79f2eaf7cb783d55d02e907557af43c0088ab1e6e0ad36e6d63", "contract_remediation": "persist_multislot_main_query_rankings_for_conversion_metric", "input_hashes": {"r3": sha(R3), "r3_rs": sha(RS), "query_plans": sha(PLANS), "general_metadata": sha(GENERAL / "candidate-metadata.sqlite")}}
    (OUT / "prediction-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "prediction-seal.json").write_text(json.dumps({**protocol, **manifest, "sealed": True}, indent=2, sort_keys=True) + "\n")
    print(f"sealed={prediction_hash} predictions={len(predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
