#!/usr/bin/env python3
"""Gate 08 R3-B: Coverage-only Retrieval Replay Prediction.

Runs ONE prediction per question that simultaneously acquires hits from
ALL experiment groups, ensuring identical Question/QueryPlan/Query text/Runtime.

Experiment groups:
  E0:          Production Raw + Existing Structured (from Gate 08)
  E1:          E0 + Candidate Raw (R4 raw lanes)
  E2-Legacy:   E0 + Legacy Structured (R2 structured lanes, 628)
  E2-Control:  E0 + Control Structured (S-Control, 464, gate06-r4-v1)
  E2-Expanded: E0 + Expanded Structured (R4 structured lanes, 19500)
  E3-Legacy:   E0 + 4-lane RRF (R4 raw + R2 structured)
  E3-Control:  E0 + 4-lane RRF (R4 raw + S-Control structured)
  E3-Expanded: E0 + 4-lane RRF (R4 raw + R4 structured)

Fixed budgets (must match Gate 08 R2):
  lane_k=50, rrf_k=60, final_pool_k=40, candidate_pool_k=40
  slot_top_k=20, slot_min_budget=10, total_k=40
  BM25 weight=1, Dense weight=1

Usage:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/evaluation/run_pdf_v4_gate_08_r3_predict.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.candidate_query_builder import build_all_queries  # noqa: E402
from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit, fuse_candidate_hits  # noqa: E402
from src.pdf_retrieval_v4.candidate_slot_pool import build_slot_pool  # noqa: E402
from src.pdf_retrieval_v4.candidate_view_index import (  # noqa: E402
    CandidateSearchHit,
    CandidateViewIndexReader,
)
from src.pdf_retrieval_v4.query_plan_models import QueryPlan  # noqa: E402
from src.pdf_retrieval_v4.serialization import query_plan_from_dict  # noqa: E402

R3_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3"
GATE07_PLANS = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-07/query-plan-predictions.json"
)
GATE08_RAW_PARITY = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08/raw-parity.json"
)
GATE08_PREDICTIONS = (
    ROOT
    / "artifacts/evaluation/pdf-retrieval-v4-gate-08/retrieval-predictions.jsonl.gz"
)

R4_INDEX_DIR = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-06-r4/candidate-indexes"
)
R2_INDEX_DIR = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2/candidate-indexes"
)
CONTROL_INDEX_DIR = R3_DIR / "control-indexes"

BUDGETS = {
    "rrf_k": 60,
    "lane_k": 50,
    "final_pool_k": 40,
    "candidate_pool_k": 40,
}
SLOT_TOP_K = 20
SLOT_MIN_BUDGET = 10
SLOT_TOTAL_K = 40

RAW_LANES = ["candidate_raw_bm25", "candidate_raw_dense"]
STRUCTURED_LANES = ["candidate_structured_bm25", "candidate_structured_dense"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _lane_hit_to_dict(hit: CandidateSearchHit) -> dict[str, Any]:
    rank = hit.bm25_rank if hit.bm25_rank is not None else hit.dense_rank
    score = hit.bm25_score if hit.bm25_score is not None else hit.dense_score
    return {
        "candidate_key": hit.candidate_key,
        "view_id": hit.view_id,
        "rank": rank,
        "score": score,
    }


def _rrf_hit_to_dict(hit: CandidateRRFHit, rank: int) -> dict[str, Any]:
    return {
        "candidate_key": hit.candidate_key,
        "rrf_score": hit.rrf_score,
        "rank": rank,
        "lane_ranks": dict(hit.lane_ranks),
        "supporting_view_ids": dict(hit.supporting_view_ids),
    }


def _serialize_lane_hits(
    hits: dict[str, list[CandidateSearchHit]],
) -> dict[str, list[dict[str, Any]]]:
    return {lane: [_lane_hit_to_dict(h) for h in hs] for lane, hs in hits.items()}


def _serialize_rrf(rrf: list[CandidateRRFHit]) -> list[dict[str, Any]]:
    return [_rrf_hit_to_dict(h, r) for r, h in enumerate(rrf, 1)]


def _rrf_pool(rrf_hits: list[CandidateRRFHit], pool_k: int) -> list[dict[str, Any]]:
    return [
        {
            "candidate_key": hit.candidate_key,
            "rrf_score": hit.rrf_score,
            "rank": rank,
            "lane_ranks": dict(hit.lane_ranks),
        }
        for rank, hit in enumerate(rrf_hits[:pool_k], 1)
    ]


def _slot_pool_to_list(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_key": item["candidate_key"],
            "slot_id": item["slot_id"],
            "slot_rank": item["slot_rank"],
            "supporting_slots": item["supporting_slots"],
        }
        for item in pool
    ]


# ---------------------------------------------------------------------------
# Combined pool builder (matches Gate 08 R2 pattern)
# ---------------------------------------------------------------------------


def _build_combined_pool(
    raw_full_pool: list[dict[str, Any]],
    structured_records: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build combined pool: raw + structured residual + candidate residual.

    Order: raw_full_pool (source=raw), structured residual (source=structured),
    candidate residual (source=candidate).  De-dup by candidate_key, keeping
    first occurrence.
    """
    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    rank = 1
    for item in raw_full_pool:
        key = str(item.get("candidate_key") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        combined.append({"candidate_key": key, "source": "raw", "rank": rank})
        rank += 1
    for item in structured_records:
        key = str(item.get("candidate_key") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        combined.append({"candidate_key": key, "source": "structured", "rank": rank})
        rank += 1
    for item in candidate_pool:
        key = str(item.get("candidate_key") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        combined.append({"candidate_key": key, "source": "candidate", "rank": rank})
        rank += 1
    return combined


# ---------------------------------------------------------------------------
# Lane search and RRF helpers
# ---------------------------------------------------------------------------


def _search_lanes(
    reader: CandidateViewIndexReader,
    query: str,
    lanes: list[str],
    allowed_keys: dict[str, set[str] | None],
    lane_k: int,
) -> dict[str, list[CandidateSearchHit]]:
    """Search specified lanes from a reader."""
    lane_hits: dict[str, list[CandidateSearchHit]] = {}
    for lane in lanes:
        hits = reader.search(
            lane,
            query,
            allowed_candidate_keys=allowed_keys.get(lane),
            k=lane_k,
        )
        lane_hits[lane] = hits
    return lane_hits


def _compute_allowed_keys(
    reader: CandidateViewIndexReader,
    lanes: list[str],
    document_scope: set[str],
) -> dict[str, set[str] | None]:
    """Compute allowed candidate keys per lane for document scope."""
    result: dict[str, set[str] | None] = {}
    for lane in lanes:
        if document_scope:
            result[lane] = reader.candidate_keys_for_documents(lane, document_scope)
        else:
            result[lane] = None
    return result


def _build_slot_pools_for_lanes(
    readers_config: list[
        tuple[CandidateViewIndexReader, list[str], dict[str, set[str] | None]]
    ],
    slot_queries: dict[str, list[str]],
    rrf_k: int,
    lane_k: int,
) -> dict[str, list[CandidateRRFHit]]:
    """For each slot, search all configured lanes and RRF fuse."""
    slot_rrf: dict[str, list[CandidateRRFHit]] = {}
    for slot_id, query_list in slot_queries.items():
        if not query_list:
            continue
        slot_query = query_list[0]
        all_lane_hits: dict[str, list[CandidateSearchHit]] = {}
        for reader, lanes, allowed_keys in readers_config:
            lane_hits = _search_lanes(reader, slot_query, lanes, allowed_keys, lane_k)
            all_lane_hits.update(lane_hits)
        slot_rrf[slot_id] = fuse_candidate_hits(all_lane_hits, rrf_k=rrf_k)
    return slot_rrf


def _build_pool(
    rrf_hits: list[CandidateRRFHit],
    is_multi_slot: bool,
    slot_queries: dict[str, list[str]],
    readers_config: list[
        tuple[CandidateViewIndexReader, list[str], dict[str, set[str] | None]]
    ],
    rrf_k: int,
    lane_k: int,
    pool_k: int,
) -> list[dict[str, Any]]:
    """Build final pool: slot pool for multi-slot, RRF pool otherwise."""
    if is_multi_slot and slot_queries:
        slot_rrf = _build_slot_pools_for_lanes(
            readers_config, slot_queries, rrf_k, lane_k
        )
        return build_slot_pool(
            slot_rrf,
            slot_top_k=SLOT_TOP_K,
            slot_min_budget=SLOT_MIN_BUDGET,
            total_k=SLOT_TOTAL_K,
        )
    return _rrf_pool(rrf_hits, pool_k)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_case(
    case_id: str,
    plan: QueryPlan,
    raw_case: dict[str, Any],
    gate08_prediction: dict[str, Any],
    r4_reader: CandidateViewIndexReader,
    r2_reader: CandidateViewIndexReader,
    control_reader: CandidateViewIndexReader,
) -> dict[str, Any]:
    """Run all experiment groups for a single case."""
    scope = set(plan.document_scope)
    lane_k = BUDGETS["lane_k"]
    rrf_k = BUDGETS["rrf_k"]
    pool_k = BUDGETS["final_pool_k"]

    # Build queries
    queries = build_all_queries(plan)
    raw_query = queries["raw_question"][0] if queries["raw_question"] else ""
    slot_queries = queries.get("slots", {})
    is_multi_slot = len(slot_queries) > 1

    # Compute allowed keys for each index
    r4_raw_allowed = _compute_allowed_keys(r4_reader, RAW_LANES, scope)
    r4_struct_allowed = _compute_allowed_keys(r4_reader, STRUCTURED_LANES, scope)
    r2_struct_allowed = _compute_allowed_keys(r2_reader, STRUCTURED_LANES, scope)
    control_struct_allowed = _compute_allowed_keys(
        control_reader, STRUCTURED_LANES, scope
    )

    # Search main query lanes
    r4_raw_hits = _search_lanes(r4_reader, raw_query, RAW_LANES, r4_raw_allowed, lane_k)
    r4_struct_hits = _search_lanes(
        r4_reader, raw_query, STRUCTURED_LANES, r4_struct_allowed, lane_k
    )
    r2_struct_hits = _search_lanes(
        r2_reader, raw_query, STRUCTURED_LANES, r2_struct_allowed, lane_k
    )
    control_struct_hits = _search_lanes(
        control_reader, raw_query, STRUCTURED_LANES, control_struct_allowed, lane_k
    )

    # RRF fuse for each experiment group (main query)
    e1_rrf = fuse_candidate_hits(r4_raw_hits, rrf_k=rrf_k)
    e2_legacy_rrf = fuse_candidate_hits(r2_struct_hits, rrf_k=rrf_k)
    e2_control_rrf = fuse_candidate_hits(control_struct_hits, rrf_k=rrf_k)
    e2_expanded_rrf = fuse_candidate_hits(r4_struct_hits, rrf_k=rrf_k)

    e3_legacy_lane_hits = {**r4_raw_hits, **r2_struct_hits}
    e3_legacy_rrf = fuse_candidate_hits(e3_legacy_lane_hits, rrf_k=rrf_k)
    e3_control_lane_hits = {**r4_raw_hits, **control_struct_hits}
    e3_control_rrf = fuse_candidate_hits(e3_control_lane_hits, rrf_k=rrf_k)
    e3_expanded_lane_hits = {**r4_raw_hits, **r4_struct_hits}
    e3_expanded_rrf = fuse_candidate_hits(e3_expanded_lane_hits, rrf_k=rrf_k)

    # Reader configs for slot pools
    e1_readers = [(r4_reader, RAW_LANES, r4_raw_allowed)]
    e2_legacy_readers = [(r2_reader, STRUCTURED_LANES, r2_struct_allowed)]
    e2_control_readers = [(control_reader, STRUCTURED_LANES, control_struct_allowed)]
    e2_expanded_readers = [(r4_reader, STRUCTURED_LANES, r4_struct_allowed)]
    e3_legacy_readers = [
        (r4_reader, RAW_LANES, r4_raw_allowed),
        (r2_reader, STRUCTURED_LANES, r2_struct_allowed),
    ]
    e3_control_readers = [
        (r4_reader, RAW_LANES, r4_raw_allowed),
        (control_reader, STRUCTURED_LANES, control_struct_allowed),
    ]
    e3_expanded_readers = [
        (r4_reader, RAW_LANES, r4_raw_allowed),
        (r4_reader, STRUCTURED_LANES, r4_struct_allowed),
    ]

    # Build pools
    e1_pool = _build_pool(
        e1_rrf, is_multi_slot, slot_queries, e1_readers, rrf_k, lane_k, pool_k
    )
    e2_legacy_pool = _build_pool(
        e2_legacy_rrf,
        is_multi_slot,
        slot_queries,
        e2_legacy_readers,
        rrf_k,
        lane_k,
        pool_k,
    )
    e2_control_pool = _build_pool(
        e2_control_rrf,
        is_multi_slot,
        slot_queries,
        e2_control_readers,
        rrf_k,
        lane_k,
        pool_k,
    )
    e2_expanded_pool = _build_pool(
        e2_expanded_rrf,
        is_multi_slot,
        slot_queries,
        e2_expanded_readers,
        rrf_k,
        lane_k,
        pool_k,
    )
    e3_legacy_pool = _build_pool(
        e3_legacy_rrf,
        is_multi_slot,
        slot_queries,
        e3_legacy_readers,
        rrf_k,
        lane_k,
        pool_k,
    )
    e3_control_pool = _build_pool(
        e3_control_rrf,
        is_multi_slot,
        slot_queries,
        e3_control_readers,
        rrf_k,
        lane_k,
        pool_k,
    )
    e3_expanded_pool = _build_pool(
        e3_expanded_rrf,
        is_multi_slot,
        slot_queries,
        e3_expanded_readers,
        rrf_k,
        lane_k,
        pool_k,
    )

    # E0: raw_full_pool + structured_records (from Gate 08)
    raw_full_pool = raw_case.get("raw_full_rrf_candidates") or []
    structured_records = [
        {
            "candidate_key": str(
                item.get("original_candidate_identity")
                or item.get("mapped_candidate_identity")
                or ""
            )
        }
        for item in gate08_prediction.get("structured_strict_source_pool") or []
        if item.get("original_candidate_identity")
        or item.get("mapped_candidate_identity")
    ]

    e0_pool = _build_combined_pool(raw_full_pool, structured_records, [])
    e1_combined = _build_combined_pool(raw_full_pool, structured_records, e1_pool)
    e2_legacy_combined = _build_combined_pool(
        raw_full_pool, structured_records, e2_legacy_pool
    )
    e2_control_combined = _build_combined_pool(
        raw_full_pool, structured_records, e2_control_pool
    )
    e2_expanded_combined = _build_combined_pool(
        raw_full_pool, structured_records, e2_expanded_pool
    )
    e3_legacy_combined = _build_combined_pool(
        raw_full_pool, structured_records, e3_legacy_pool
    )
    e3_control_combined = _build_combined_pool(
        raw_full_pool, structured_records, e3_control_pool
    )
    e3_expanded_combined = _build_combined_pool(
        raw_full_pool, structured_records, e3_expanded_pool
    )

    return {
        "case_id": case_id,
        "query_plan_id": plan.plan_id,
        "task_type": plan.task_type,
        "document_scope": list(plan.document_scope),
        "is_multi_slot": is_multi_slot,
        "candidate_raw": {
            "bm25": [
                _lane_hit_to_dict(h) for h in r4_raw_hits.get("candidate_raw_bm25", [])
            ],
            "dense": [
                _lane_hit_to_dict(h) for h in r4_raw_hits.get("candidate_raw_dense", [])
            ],
            "fused": _serialize_rrf(e1_rrf),
        },
        "structured_legacy": {
            "bm25": [
                _lane_hit_to_dict(h)
                for h in r2_struct_hits.get("candidate_structured_bm25", [])
            ],
            "dense": [
                _lane_hit_to_dict(h)
                for h in r2_struct_hits.get("candidate_structured_dense", [])
            ],
            "fused": _serialize_rrf(e2_legacy_rrf),
        },
        "structured_control": {
            "bm25": [
                _lane_hit_to_dict(h)
                for h in control_struct_hits.get("candidate_structured_bm25", [])
            ],
            "dense": [
                _lane_hit_to_dict(h)
                for h in control_struct_hits.get("candidate_structured_dense", [])
            ],
            "fused": _serialize_rrf(e2_control_rrf),
        },
        "structured_expanded": {
            "bm25": [
                _lane_hit_to_dict(h)
                for h in r4_struct_hits.get("candidate_structured_bm25", [])
            ],
            "dense": [
                _lane_hit_to_dict(h)
                for h in r4_struct_hits.get("candidate_structured_dense", [])
            ],
            "fused": _serialize_rrf(e2_expanded_rrf),
        },
        "e3_legacy_fused": _serialize_rrf(e3_legacy_rrf),
        "e3_control_fused": _serialize_rrf(e3_control_rrf),
        "e3_expanded_fused": _serialize_rrf(e3_expanded_rrf),
        "e0_pool": e0_pool,
        "e1_pool": e1_combined,
        "e2_legacy_pool": e2_legacy_combined,
        "e2_control_pool": e2_control_combined,
        "e2_expanded_pool": e2_expanded_combined,
        "e3_legacy_pool": e3_legacy_combined,
        "e3_control_pool": e3_control_combined,
        "e3_expanded_pool": e3_expanded_combined,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "answer_generation_calls": 0,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_plans(path: Path) -> list[tuple[str, QueryPlan]]:
    """Load Gate 07 query plan predictions."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("plans") or data.get("cases") or []
    return [
        (str(item["case_id"]), query_plan_from_dict(item["plan"])) for item in items
    ]


def load_raw_parity(path: Path) -> dict[str, dict[str, Any]]:
    """Load Gate 08 raw parity keyed by case_id."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("raw_cases") or []
    return {str(c["case_id"]): c for c in cases}


def load_gate08_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Load Gate 08 original predictions keyed by case_id."""
    result: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("stream") == "header":
                continue
            result[str(rec["case_id"])] = rec
    return result


def write_jsonl_gzip(path: Path, records: list[dict[str, Any]]) -> str:
    """Write records as gzip jsonl, return sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            for record in records:
                handle.write(
                    (
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )
    return _sha(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=GATE07_PLANS)
    parser.add_argument("--raw-parity", type=Path, default=GATE08_RAW_PARITY)
    parser.add_argument("--gate08-predictions", type=Path, default=GATE08_PREDICTIONS)
    parser.add_argument("--r4-indexes", type=Path, default=R4_INDEX_DIR)
    parser.add_argument("--r2-indexes", type=Path, default=R2_INDEX_DIR)
    parser.add_argument("--control-indexes", type=Path, default=CONTROL_INDEX_DIR)
    parser.add_argument("--out-dir", type=Path, default=R3_DIR)
    parser.add_argument("--code-commit", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("Gate 08 R3-B: Coverage-only Retrieval Replay Prediction")
    print("=" * 70)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    print("\nLoading inputs...")
    plans = load_plans(args.plans)
    print(f"  Plans: {len(plans)} cases")
    raw_cases = load_raw_parity(args.raw_parity)
    print(f"  Raw parity: {len(raw_cases)} cases")
    gate08_preds = load_gate08_predictions(args.gate08_predictions)
    print(f"  Gate 08 predictions: {len(gate08_preds)} cases")

    # Create readers
    print("\nCreating index readers...")
    r4_reader = CandidateViewIndexReader(args.r4_indexes, rrf_k=BUDGETS["rrf_k"])
    print(f"  R4 reader: {args.r4_indexes.name}")
    r2_reader = CandidateViewIndexReader(args.r2_indexes, rrf_k=BUDGETS["rrf_k"])
    print(f"  R2 reader: {args.r2_indexes.name}")
    control_reader = CandidateViewIndexReader(
        args.control_indexes, rrf_k=BUDGETS["rrf_k"]
    )
    print(f"  Control reader: {args.control_indexes.name}")

    # Run predictions
    print("\nRunning predictions...")
    predictions: list[dict[str, Any]] = []
    for idx, (case_id, plan) in enumerate(plans, 1):
        raw_case = raw_cases.get(case_id, {})
        gate08_pred = gate08_preds.get(case_id, {})
        pred = predict_case(
            case_id,
            plan,
            raw_case,
            gate08_pred,
            r4_reader,
            r2_reader,
            control_reader,
        )
        predictions.append(pred)
        if idx % 10 == 0 or idx == len(plans):
            print(f"  [{idx}/{len(plans)}] {case_id}")

    # Close readers
    r4_reader.close()
    r2_reader.close()
    control_reader.close()

    # Write predictions
    print("\nWriting predictions...")
    pred_path = args.out_dir / "predictions.jsonl.gz"
    pred_hash = write_jsonl_gzip(pred_path, predictions)
    print(f"  Predictions: {pred_path}")
    print(f"  Prediction hash: {pred_hash[:16]}...")
    print(f"  Record count: {len(predictions)}")

    # Write protocol
    code_commit = args.code_commit or _commit()
    protocol = {
        "schema": "pdf-retrieval-v4/gate-08-r3/prediction/v1",
        "gate": "pdf_retrieval_v4_gate_08_r3",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "implementation_commit": code_commit,
        "artifact_seal_commit": code_commit,
        "budgets": BUDGETS,
        "embedding_model": "all-MiniLM-L6-v2",
        "rrf": {"k": BUDGETS["rrf_k"], "lane_weights": "all_1.0"},
        "slot_pool": {
            "slot_top_k": SLOT_TOP_K,
            "slot_min_budget": SLOT_MIN_BUDGET,
            "total_k": SLOT_TOTAL_K,
        },
        "input_hashes": {
            "plans": _sha(args.plans),
            "raw_parity": _sha(args.raw_parity),
            "gate08_predictions": _sha(args.gate08_predictions),
            "r4_indexes": _sha(args.r4_indexes / "candidate-metadata.sqlite"),
            "r2_indexes": _sha(args.r2_indexes / "candidate-metadata.sqlite"),
            "control_indexes": _sha(args.control_indexes / "candidate-metadata.sqlite"),
        },
        "indexes": {
            "r4_expanded": str(args.r4_indexes),
            "r2_legacy": str(args.r2_indexes),
            "control": str(args.control_indexes),
        },
        "experiment_groups": [
            "e0",
            "e1",
            "e2_legacy",
            "e2_control",
            "e2_expanded",
            "e3_legacy",
            "e3_control",
            "e3_expanded",
        ],
        "prediction_inputs": [
            "question_text_in_sealed_plan",
            "document_scope",
            "gate_07_query_plan",
            "gate_06_r4_candidate_indexes",
            "gate_08_r2_candidate_indexes",
            "gate_08_r3_control_indexes",
            "gate_08_raw_replay",
            "gate_08_sealed_structured_pool",
        ],
        "forbidden_inputs": [
            "labels.golden.jsonl",
            "benchmark-governance.jsonl",
            "evidence-family-map.json",
            "expected_value",
            "reference_answer",
        ],
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "parameter_scan": False,
        "per_query_tuning": False,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
    }
    protocol_path = args.out_dir / "prediction-protocol.json"
    _write(protocol_path, protocol)
    print(f"  Protocol: {protocol_path}")

    # Write manifest
    manifest = {
        "schema": "pdf-retrieval-v4/gate-08-r3/predictions/v1",
        "record_count": len(predictions),
        "prediction_sha256": pred_hash,
        "budgets": BUDGETS,
        "slot_pool": {
            "slot_top_k": SLOT_TOP_K,
            "slot_min_budget": SLOT_MIN_BUDGET,
            "total_k": SLOT_TOTAL_K,
        },
        "rrf": {"k": BUDGETS["rrf_k"], "lane_weights": "all_1.0"},
        "embedding_model": "all-MiniLM-L6-v2",
        "indexes": {
            "r4_expanded": str(args.r4_indexes),
            "r2_legacy": str(args.r2_indexes),
            "control": str(args.control_indexes),
        },
        "code_commit": code_commit,
        "experiment_groups": [
            "e0",
            "e1",
            "e2_legacy",
            "e2_control",
            "e2_expanded",
            "e3_legacy",
            "e3_control",
            "e3_expanded",
        ],
    }
    manifest_path = args.out_dir / "prediction-manifest.json"
    _write(manifest_path, manifest)
    print(f"  Manifest: {manifest_path}")

    # Write seal
    seal = {
        "prediction_count": len(predictions),
        "prediction_hash": pred_hash,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "reference_answer_reads_before_seal": 0,
        "expected_value_reads_before_seal": 0,
        "parameter_scan": False,
        "per_query_tuning": False,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
        "sealed": True,
    }
    seal_path = args.out_dir / "prediction-seal.json"
    _write(seal_path, seal)
    print(f"  Seal: {seal_path}")

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETE")
    print(f"  prediction_count = {len(predictions)}")
    print(f"  prediction_hash = {pred_hash[:16]}...")
    print("  gold_reads_before_seal = 0")
    print("  parameter_scan = false")
    print("  reranker_calls = 0")
    print("  production_switch_allowed = false")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
