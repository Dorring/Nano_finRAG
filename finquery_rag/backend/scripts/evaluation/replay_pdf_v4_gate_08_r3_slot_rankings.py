#!/usr/bin/env python3
"""Replay and seal the missing Gate 08 R3 multi-slot family rankings.

Only the 18 multi-slot Gate 07 plans are replayed. The original Gate 08 R3
prediction and seal are immutable inputs. No gold, governance, labels, expected
values, or reference answers are read.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_pdf_v4_gate_08_r3_predict import (  # noqa: E402
    RAW_LANES,
    STRUCTURED_LANES,
    _build_combined_pool,
    _compute_allowed_keys,
    _lane_hit_to_dict,
    _search_lanes,
    load_plans,
)
from src.pdf_retrieval_v4.candidate_query_builder import build_all_queries  # noqa: E402
from src.pdf_retrieval_v4.candidate_rrf import (  # noqa: E402
    CandidateRRFHit,
    fuse_candidate_hits,
)
from src.pdf_retrieval_v4.candidate_slot_pool import build_slot_pool  # noqa: E402
from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader  # noqa: E402

R3_COMMIT = "f9e9cdb"
R3_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3"
RS_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3-rs"
GATE07_PLANS = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-07/query-plan-predictions.json"
)
R4_INDEX_DIR = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-06-r4/candidate-indexes"
)
R4_INDEX_SEAL = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-06-r4/index-seal.json"

RRF_K = 60
LANE_K = 50
FINAL_POOL_K = 40
SLOT_TOP_K = 20
SLOT_MIN_BUDGET = 10
SLOT_TOTAL_K = 40

QUERY_CONTRACT_FILES = (
    "src/pdf_retrieval_v4/candidate_query_builder.py",
    "src/pdf_retrieval_v4/candidate_rrf.py",
    "src/pdf_retrieval_v4/candidate_slot_pool.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_gzip(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            for record in records:
                line = json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write((line + "\n").encode("utf-8"))
    return _sha256(path)


def _load_r3_predictions(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            result[str(record["case_id"])] = record
    return result


def _serialize_rrf(hits: list[CandidateRRFHit]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_key": hit.candidate_key,
            "rank": rank,
            "rrf_score": hit.rrf_score,
            "lane_ranks": dict(hit.lane_ranks),
            "supporting_view_ids": dict(hit.supporting_view_ids),
        }
        for rank, hit in enumerate(hits, 1)
    ]


def _pool_with_rank(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item, pool_rank=rank) for rank, item in enumerate(pool, 1)]


def _candidate_sequence(pool: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("candidate_key") or "") for item in pool]


def _git_blob_sha(commit: str, path: str) -> str:
    content = subprocess.check_output(
        ["git", "show", f"{commit}:finquery_rag/backend/{path}"],
        cwd=ROOT,
    )
    return hashlib.sha256(content).hexdigest()


def query_contract_parity() -> dict[str, Any]:
    command = ["git", "diff", "--exit-code", R3_COMMIT, "--", *QUERY_CONTRACT_FILES]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    files: list[dict[str, Any]] = []
    for relative in QUERY_CONTRACT_FILES:
        current_sha = _sha256(ROOT / relative)
        r3_sha = _git_blob_sha(R3_COMMIT, relative)
        files.append(
            {
                "path": relative,
                "r3_commit_sha256": r3_sha,
                "current_sha256": current_sha,
                "exact": current_sha == r3_sha,
            }
        )
    return {
        "r3_commit": R3_COMMIT,
        "semantic_diff_empty": result.returncode == 0,
        "git_diff": result.stdout,
        "files": files,
        "all_source_hashes_exact": all(item["exact"] for item in files),
    }


def _reconstruct_combined(
    e0_pool: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use the exact R3 combined-pool function with E0 as the frozen prefix."""
    return _build_combined_pool(e0_pool, [], candidate_pool)


def _replay_case(
    case_id: str,
    plan: Any,
    r3_prediction: dict[str, Any],
    reader: CandidateViewIndexReader,
) -> tuple[dict[str, Any], dict[str, bool], int]:
    queries = build_all_queries(plan)
    slot_queries = queries.get("slots") or {}
    if len(slot_queries) <= 1:
        raise ValueError(f"not_multi_slot:{case_id}")

    scope = set(plan.document_scope)
    raw_allowed = _compute_allowed_keys(reader, RAW_LANES, scope)
    structured_allowed = _compute_allowed_keys(reader, STRUCTURED_LANES, scope)

    raw_slot_hits: dict[str, list[CandidateRRFHit]] = {}
    structured_slot_hits: dict[str, list[CandidateRRFHit]] = {}
    early_slot_hits: dict[str, list[CandidateRRFHit]] = {}
    slot_definitions: list[dict[str, Any]] = []
    slot_rankings: dict[str, dict[str, Any]] = {}

    for slot_order, (slot_id, query_list) in enumerate(slot_queries.items()):
        if not query_list:
            raise ValueError(f"empty_slot_query:{case_id}:{slot_id}")
        query_text = str(query_list[0])
        raw_lane_hits = _search_lanes(
            reader,
            query_text,
            RAW_LANES,
            raw_allowed,
            LANE_K,
        )
        structured_lane_hits = _search_lanes(
            reader,
            query_text,
            STRUCTURED_LANES,
            structured_allowed,
            LANE_K,
        )
        raw_fused = fuse_candidate_hits(raw_lane_hits, rrf_k=RRF_K)
        structured_fused = fuse_candidate_hits(structured_lane_hits, rrf_k=RRF_K)
        early_fused = fuse_candidate_hits(
            {**raw_lane_hits, **structured_lane_hits},
            rrf_k=RRF_K,
        )

        raw_slot_hits[slot_id] = raw_fused
        structured_slot_hits[slot_id] = structured_fused
        early_slot_hits[slot_id] = early_fused
        slot_definitions.append(
            {
                "slot_id": slot_id,
                "slot_order": slot_order,
                "query_text": query_text,
                "query_sha256": _text_sha256(query_text),
                "query_plan_id": plan.plan_id,
                "document_scope": list(plan.document_scope),
                "task_type": plan.task_type,
                "operation": plan.operation,
            }
        )
        slot_rankings[slot_id] = {
            "raw": {
                "bm25": [
                    _lane_hit_to_dict(hit)
                    for hit in raw_lane_hits.get("candidate_raw_bm25", [])
                ],
                "dense": [
                    _lane_hit_to_dict(hit)
                    for hit in raw_lane_hits.get("candidate_raw_dense", [])
                ],
                "fused": _serialize_rrf(raw_fused),
            },
            "structured": {
                "bm25": [
                    _lane_hit_to_dict(hit)
                    for hit in structured_lane_hits.get(
                        "candidate_structured_bm25", []
                    )
                ],
                "dense": [
                    _lane_hit_to_dict(hit)
                    for hit in structured_lane_hits.get(
                        "candidate_structured_dense", []
                    )
                ],
                "fused": _serialize_rrf(structured_fused),
            },
            "early_cross_family_fused": _serialize_rrf(early_fused),
        }

    raw_pool = build_slot_pool(
        raw_slot_hits,
        slot_top_k=SLOT_TOP_K,
        slot_min_budget=SLOT_MIN_BUDGET,
        total_k=SLOT_TOTAL_K,
    )
    structured_pool = build_slot_pool(
        structured_slot_hits,
        slot_top_k=SLOT_TOP_K,
        slot_min_budget=SLOT_MIN_BUDGET,
        total_k=SLOT_TOTAL_K,
    )
    early_pool = build_slot_pool(
        early_slot_hits,
        slot_top_k=SLOT_TOP_K,
        slot_min_budget=SLOT_MIN_BUDGET,
        total_k=SLOT_TOTAL_K,
    )

    e0_pool = list(r3_prediction.get("e0_pool") or [])
    reconstructed = {
        "e1": _reconstruct_combined(e0_pool, raw_pool),
        "e2_expanded": _reconstruct_combined(e0_pool, structured_pool),
        "e3_expanded": _reconstruct_combined(e0_pool, early_pool),
    }
    parity = {
        "e1": _candidate_sequence(reconstructed["e1"])
        == _candidate_sequence(r3_prediction.get("e1_pool") or []),
        "e2_expanded": _candidate_sequence(reconstructed["e2_expanded"])
        == _candidate_sequence(r3_prediction.get("e2_expanded_pool") or []),
        "e3_expanded": _candidate_sequence(reconstructed["e3_expanded"])
        == _candidate_sequence(r3_prediction.get("e3_expanded_pool") or []),
    }
    record = {
        "case_id": case_id,
        "query_plan_id": plan.plan_id,
        "is_multi_slot": True,
        "slot_definitions": slot_definitions,
        "slot_family_rankings": slot_rankings,
        "reconstructed_candidate_pools": {
            "e1": _pool_with_rank(raw_pool),
            "e2_expanded": _pool_with_rank(structured_pool),
            "e3_expanded": _pool_with_rank(early_pool),
        },
        "replay_parity": parity,
    }
    return record, parity, len(slot_definitions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=GATE07_PLANS)
    parser.add_argument(
        "--r3-predictions",
        type=Path,
        default=R3_DIR / "predictions.jsonl.gz",
    )
    parser.add_argument(
        "--r3-seal",
        type=Path,
        default=R3_DIR / "prediction-seal.json",
    )
    parser.add_argument("--indexes", type=Path, default=R4_INDEX_DIR)
    parser.add_argument("--index-seal", type=Path, default=R4_INDEX_SEAL)
    parser.add_argument("--out-dir", type=Path, default=RS_DIR)
    args = parser.parse_args()

    original_prediction_hash_before = _sha256(args.r3_predictions)
    original_seal_hash_before = _sha256(args.r3_seal)
    r3_seal = _load_json(args.r3_seal)
    if original_prediction_hash_before != r3_seal.get("prediction_hash"):
        raise RuntimeError("original_r3_prediction_seal_hash_mismatch")

    query_parity = query_contract_parity()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "query-contract-parity.json", query_parity)
    if not query_parity["semantic_diff_empty"] or not query_parity[
        "all_source_hashes_exact"
    ]:
        raise RuntimeError("slot_reseal_query_contract_blocked")

    plans = load_plans(args.plans)
    r3_predictions = _load_r3_predictions(args.r3_predictions)
    multi_plans = [
        (case_id, plan)
        for case_id, plan in plans
        if len((build_all_queries(plan).get("slots") or {})) > 1
    ]
    r3_multi_cases = {
        case_id
        for case_id, prediction in r3_predictions.items()
        if prediction.get("is_multi_slot")
    }
    plan_multi_cases = {case_id for case_id, _ in multi_plans}
    if plan_multi_cases != r3_multi_cases:
        raise RuntimeError("multi_slot_case_set_drift")

    retrieval_config = {
        "lane_k": LANE_K,
        "rrf_k": RRF_K,
        "final_pool_k": FINAL_POOL_K,
        "slot_top_k": SLOT_TOP_K,
        "slot_min_budget": SLOT_MIN_BUDGET,
        "slot_total_k": SLOT_TOTAL_K,
        "bm25_weight": 1.0,
        "dense_weight": 1.0,
        "embedding_model": "all-MiniLM-L6-v2",
        "lanes": [*RAW_LANES, *STRUCTURED_LANES],
    }
    input_integrity = {
        "gate07_query_plan_sha256": _sha256(args.plans),
        "original_r3_prediction_sha256": original_prediction_hash_before,
        "original_r3_seal_sha256": original_seal_hash_before,
        "gate06_r4_index_seal_sha256": _sha256(args.index_seal),
        "candidate_metadata_sha256": _sha256(
            args.indexes / "candidate-metadata.sqlite"
        ),
        "candidate_query_builder_sha256": _sha256(
            ROOT / QUERY_CONTRACT_FILES[0]
        ),
        "candidate_rrf_sha256": _sha256(ROOT / QUERY_CONTRACT_FILES[1]),
        "candidate_slot_pool_sha256": _sha256(ROOT / QUERY_CONTRACT_FILES[2]),
        "retrieval_config_sha256": _payload_sha256(retrieval_config),
        "original_r3_prediction_immutable_before_replay": True,
        "multi_slot_case_count": len(multi_plans),
        "multi_slot_case_set_exact": plan_multi_cases == r3_multi_cases,
    }
    _write_json(args.out_dir / "input-integrity.json", input_integrity)

    protocol = {
        "schema": "pdf-retrieval-v4/gate-08-r3-rs/protocol/v1",
        "gate": "pdf_retrieval_v4_gate_08_r3_rs",
        "phase": "slot_local_family_ranking_reseal",
        "evaluation_type": "trace_replay_not_recall_experiment",
        "replay_scope": "18_multi_slot_cases_only",
        "retrieval_config": retrieval_config,
        "allowed_inputs": [
            "gate07_query_plans",
            "original_r3_sealed_prediction",
            "gate06_r4_expanded_candidate_index",
        ],
        "forbidden_inputs": [
            "gold",
            "governance",
            "labels",
            "expected_value",
            "reference_answer",
        ],
        "parameter_scan": False,
        "quota_scan": False,
        "index_builds": 0,
        "index_mutations": 0,
        "bridge_mutations": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
    }
    _write_json(args.out_dir / "protocol.json", protocol)

    reader = CandidateViewIndexReader(args.indexes, rrf_k=RRF_K)
    records: list[dict[str, Any]] = []
    parity_by_case: list[dict[str, Any]] = []
    total_slots = 0
    try:
        for case_id, plan in multi_plans:
            record, parity, slot_count = _replay_case(
                case_id,
                plan,
                r3_predictions[case_id],
                reader,
            )
            records.append(record)
            total_slots += slot_count
            parity_by_case.append({"case_id": case_id, **parity})
    finally:
        reader.close()

    records.sort(key=lambda record: str(record["case_id"]))
    sidecar_path = args.out_dir / "slot-local-rankings.jsonl.gz"
    sidecar_hash = _write_jsonl_gzip(sidecar_path, records)

    parity_counts = {
        family: sum(1 for item in parity_by_case if item[family])
        for family in ("e1", "e2_expanded", "e3_expanded")
    }
    parity_passed = all(count == len(records) for count in parity_counts.values())
    slot_parity = {
        "case_count": len(records),
        "parity_counts": parity_counts,
        "required": len(records),
        "all_exact": parity_passed,
        "cases": sorted(parity_by_case, key=lambda item: item["case_id"]),
    }
    _write_json(args.out_dir / "slot-replay-parity.json", slot_parity)

    original_prediction_hash_after = _sha256(args.r3_predictions)
    original_seal_hash_after = _sha256(args.r3_seal)
    original_immutable = (
        original_prediction_hash_after == original_prediction_hash_before
        and original_seal_hash_after == original_seal_hash_before
    )
    search_accounting = {
        "multi_slot_cases": len(records),
        "total_slots": total_slots,
        "raw_bm25_searches": total_slots,
        "raw_dense_searches": total_slots,
        "structured_bm25_searches": total_slots,
        "structured_dense_searches": total_slots,
        "total_lane_searches": total_slots * 4,
        "logical_query_embedding_requests": total_slots * 2,
        "unique_query_embeddings_with_reader_cache": len(
            {
                slot["query_sha256"]
                for record in records
                for slot in record["slot_definitions"]
            }
        ),
        "index_reads": total_slots * 4,
        "index_builds": 0,
    }

    manifest = {
        "schema": "pdf-retrieval-v4/gate-08-r3-rs/manifest/v1",
        "record_count": len(records),
        "slot_count": total_slots,
        "slot_sidecar_sha256": sidecar_hash,
        "search_accounting": search_accounting,
        "input_hashes": input_integrity,
        "parity": parity_counts,
    }
    _write_json(args.out_dir / "prediction-manifest.json", manifest)

    sealed = parity_passed and original_immutable and len(records) == 18
    seal = {
        "schema": "pdf-retrieval-v4/gate-08-r3-rs/seal/v1",
        "sealed": sealed,
        "replay_scope": "18_multi_slot_cases_only",
        "original_r3_prediction_immutable": original_immutable,
        "original_r3_prediction_sha256": original_prediction_hash_before,
        "original_r3_seal_sha256": original_seal_hash_before,
        "slot_sidecar_records": len(records),
        "slot_sidecar_sha256": sidecar_hash,
        "query_plan_sha256": input_integrity["gate07_query_plan_sha256"],
        "query_builder_sha256": input_integrity[
            "candidate_query_builder_sha256"
        ],
        "gate06_index_manifest_sha256": input_integrity[
            "gate06_r4_index_seal_sha256"
        ],
        "retrieval_config_sha256": input_integrity["retrieval_config_sha256"],
        "search_accounting": search_accounting,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "reference_answer_reads_before_seal": 0,
        "expected_value_reads_before_seal": 0,
        "parameter_scan": False,
        "quota_scan": False,
        "index_builds": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
    }
    seal_path = args.out_dir / "prediction-seal.json"
    _write_json(seal_path, seal)
    seal_hash = _sha256(seal_path)

    composite = {
        "schema": "pdf-retrieval-v4/gate-08-r4/composite-input/v1",
        "original_r3": {
            "prediction_sha256": original_prediction_hash_before,
            "seal_sha256": original_seal_hash_before,
        },
        "slot_local_sidecar": {
            "prediction_sha256": sidecar_hash,
            "seal_sha256": seal_hash,
        },
        "coverage": {
            "single_slot_cases": len(plans) - len(records),
            "multi_slot_cases": len(records),
            "total_cases": len(plans),
        },
        "slot_replay_parity": {
            "e1": f"{parity_counts['e1']}/{len(records)}",
            "e2_expanded": f"{parity_counts['e2_expanded']}/{len(records)}",
            "e3_expanded": f"{parity_counts['e3_expanded']}/{len(records)}",
        },
        "sealed": sealed,
    }
    _write_json(args.out_dir / "r4-composite-input-manifest.json", composite)

    decision = (
        "slot_local_family_rankings_resealed"
        if sealed
        else "slot_local_retrieval_replay_parity_blocked"
    )
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_08_r3_rs",
        "decision": decision,
        "next_gate": "lane_preserving_fusion" if sealed else "stop_and_fix_replay_parity",
        "gate_passed": sealed,
        "original_r3_prediction_immutable": original_immutable,
        "slot_definitions_persisted": f"{len(records)}/{len(records)}",
        "raw_family_rankings_present": f"{len(records)}/{len(records)}",
        "structured_family_rankings_present": f"{len(records)}/{len(records)}",
        "slot_replay_parity": composite["slot_replay_parity"],
        "search_accounting": search_accounting,
        "gold_reads": 0,
        "governance_reads": 0,
        "parameter_scan": False,
        "quota_scan": False,
        "production_switch_allowed": False,
    }
    _write_json(args.out_dir / "acceptance.json", acceptance)
    _write_json(
        args.out_dir / "next-gate.json",
        {
            "current_gate": "pdf_retrieval_v4_gate_08_r3_rs",
            "decision": decision,
            "next_gate": acceptance["next_gate"],
            "r4_input": "r4-composite-input-manifest.json" if sealed else None,
            "production_switch_allowed": False,
        },
    )

    print(f"records={len(records)} slots={total_slots}")
    print(f"E1 parity={parity_counts['e1']}/{len(records)}")
    print(f"E2 parity={parity_counts['e2_expanded']}/{len(records)}")
    print(f"E3 parity={parity_counts['e3_expanded']}/{len(records)}")
    print(f"sidecar_sha256={sidecar_hash}")
    print(f"decision={decision}")
    return 0 if sealed else 1


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    raise SystemExit(main())
