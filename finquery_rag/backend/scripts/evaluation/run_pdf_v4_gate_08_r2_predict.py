"""Gate 08 R2: Run candidate-aligned direct retrieval and seal results.

Runs the CandidateDirectRetriever over sealed Gate 07 query plans against
the Gate 08 R2 candidate-aligned indexes, then builds a combined pool
fusing:

  - Raw Full Pool (from frozen Gate 08 raw-parity replay)
  - Gate 08 Structured Pool (from sealed Gate 08 predictions)
  - Candidate Direct Residual (R2 direct pool minus raw/structured)

This script intentionally has no imports from benchmark governance or
labels.  It reads only sealed Query Plans, the R2 candidate indexes, the
frozen raw-parity replay, and the sealed Gate 08 predictions (for the
structured pool only).  No Gold/labels/expected_value/reference_answer
is read before the seal is written.
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

from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever  # noqa: E402
from src.pdf_retrieval_v4.candidate_rrf import CandidateRRFHit  # noqa: E402
from src.pdf_retrieval_v4.candidate_view_index import (  # noqa: E402
    CandidateSearchHit,
    CandidateViewIndexReader,
)
from src.pdf_retrieval_v4.query_plan_models import QueryPlan  # noqa: E402
from src.pdf_retrieval_v4.serialization import query_plan_from_dict  # noqa: E402

DEFAULT_PLANS = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-07/query-plan-predictions.json"
DEFAULT_INDEXES = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2/candidate-indexes"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2"
DEFAULT_RAW = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08/raw-parity.json"
DEFAULT_GATE08_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08"

BUDGETS = {
    "rrf_k": 60,
    "lane_k": 50,
    "final_pool_k": 40,
    "candidate_pool_k": 40,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_jsonl_gzip(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as handle:
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
    return sha(path)


def load_predictions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if value.get("stream") != "header":
                    records.append(value)
    return records


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


def _slot_pool_to_list(hits: list[CandidateRRFHit]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_key": hit.candidate_key,
            "rrf_score": hit.rrf_score,
            "rank": rank,
            "lane_ranks": dict(hit.lane_ranks),
            "supporting_view_ids": dict(hit.supporting_view_ids),
        }
        for rank, hit in enumerate(hits, 1)
    ]


def _build_combined_pool(
    raw_full_pool: list[dict[str, Any]],
    structured_pool: list[dict[str, Any]],
    candidate_direct_pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build combined pool: raw + structured residual + candidate_direct residual.

    Order: raw_full_pool (source=raw), structured residual (source=structured),
    candidate_direct residual (source=candidate_direct).  De-dup by
    candidate_key, keeping first occurrence.
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
    for item in structured_pool:
        key = str(item.get("candidate_key") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        combined.append(
            {"candidate_key": key, "source": "structured", "rank": rank}
        )
        rank += 1
    for item in candidate_direct_pool:
        key = str(item.get("candidate_key") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        combined.append(
            {"candidate_key": key, "source": "candidate_direct", "rank": rank}
        )
        rank += 1
    return combined


def predict_case(
    retriever: CandidateDirectRetriever,
    plan: QueryPlan,
    raw_case: dict[str, Any],
    gate08_prediction: dict[str, Any],
) -> dict[str, Any]:
    scope = set(plan.document_scope)
    result = retriever.retrieve(plan, document_scope=scope)

    candidate_direct_pool = result["candidate_direct_pool"]
    lane_hits = result["lane_hits"]
    rrf_hits = result["rrf_hits"]
    slot_pools = result["slot_pools"]

    lane_hits_serialized: dict[str, list[dict[str, Any]]] = {
        lane: [_lane_hit_to_dict(hit) for hit in hits]
        for lane, hits in lane_hits.items()
    }
    rrf_hits_serialized = [
        _rrf_hit_to_dict(hit, rank)
        for rank, hit in enumerate(rrf_hits, 1)
    ]
    slot_pools_serialized: dict[str, list[dict[str, Any]]] = {
        slot_id: _slot_pool_to_list(hits)
        for slot_id, hits in slot_pools.items()
    }

    # Extract structured pool from sealed Gate 08 predictions.
    structured_records = [
        {"candidate_key": str(item.get("original_candidate_identity") or "")}
        for item in gate08_prediction.get("structured_strict_source_pool") or []
        if item.get("original_candidate_identity")
    ]

    raw_full_pool = raw_case.get("raw_full_rrf_candidates") or []
    combined_pool = _build_combined_pool(
        raw_full_pool, structured_records, candidate_direct_pool
    )

    is_multi_slot = len(slot_pools) > 1

    return {
        "case_id": raw_case["case_id"],
        "query_plan_id": plan.plan_id,
        "task_type": plan.task_type,
        "document_scope": list(plan.document_scope),
        "candidate_direct_pool": candidate_direct_pool,
        "slot_pools": slot_pools_serialized,
        "lane_hits": lane_hits_serialized,
        "rrf_hits": rrf_hits_serialized,
        "combined_pool": combined_pool,
        "raw_candidate_hash": raw_case.get("raw_candidate_hash"),
        "is_multi_slot": is_multi_slot,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "answer_generation_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-parity", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--gate08-out", type=Path, default=DEFAULT_GATE08_OUT)
    parser.add_argument("--code-commit", default=None)
    args = parser.parse_args()

    plans_payload = json.loads(args.plans.read_text(encoding="utf-8"))
    raw_payload = json.loads(args.raw_parity.read_text(encoding="utf-8"))
    raw_by_case = {str(item["case_id"]): item for item in raw_payload["raw_cases"]}
    plans: list[tuple[str, QueryPlan]] = [
        (str(item["case_id"]), query_plan_from_dict(item["plan"]))
        for item in plans_payload["plans"]
    ]

    # Load Gate 08 sealed predictions for the structured pool.
    gate08_predictions_path = args.gate08_out / "retrieval-predictions.jsonl.gz"
    gate08_predictions_list = load_predictions(gate08_predictions_path)
    gate08_predictions = {
        str(item["case_id"]): item for item in gate08_predictions_list
    }

    code_commit = args.code_commit or commit()

    protocol = {
        "schema": "pdf-retrieval-v4/gate-08-r2/prediction/v1",
        "gate": "pdf_retrieval_v4_gate_08_r2",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "implementation_commit": code_commit,
        "artifact_seal_commit": code_commit,
        "budgets": BUDGETS,
        "embedding_model": "all-MiniLM-L6-v2",
        "rrf": {"k": 60, "lane_weights": "all_1.0"},
        "input_hashes": {
            "plans": sha(args.plans),
            "raw_parity": sha(args.raw_parity),
            "gate08_predictions": sha(gate08_predictions_path),
        },
        "prediction_inputs": [
            "question_text_in_sealed_plan",
            "document_scope",
            "gate_07_query_plan",
            "gate_08_r2_candidate_indexes",
            "gate_08_raw_replay",
            "gate_08_sealed_structured_pool",
        ],
        "forbidden_inputs": [
            "labels.golden.jsonl",
            "benchmark-governance.jsonl",
            "evidence-family-map.json",
            "expected_value",
            "reference_answer",
            "original_final_hit_identity",
        ],
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "reference_answer_reads_before_seal": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
        "parameter_scan": False,
        "per_query_oracle": False,
        "soft_continuation_expansion": False,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write(args.out_dir / "gate-08-r2-protocol.json", protocol)
    write(
        args.out_dir / "gate-08-r2-input-integrity.json",
        {
            "plans_sha256": sha(args.plans),
            "raw_parity_sha256": sha(args.raw_parity),
            "gate08_predictions_sha256": sha(gate08_predictions_path),
            "indexes_dir": str(args.indexes_dir),
            "prediction_count": len(plans),
        },
    )

    records: list[dict[str, Any]] = [
        {
            "stream": "header",
            "schema": "pdf-retrieval-v4/gate-08-r2/predictions/v1",
            "record_count": len(plans),
        }
    ]

    with CandidateViewIndexReader(
        args.indexes_dir, rrf_k=BUDGETS["rrf_k"]
    ) as reader:
        retriever = CandidateDirectRetriever(
            reader, rrf_k=BUDGETS["rrf_k"], lane_k=BUDGETS["lane_k"]
        )
        for case_id, plan in plans:
            if case_id not in raw_by_case:
                raise RuntimeError(f"missing_raw_case:{case_id}")
            if case_id not in gate08_predictions:
                raise RuntimeError(f"missing_gate08_prediction:{case_id}")
            records.append(
                predict_case(
                    retriever,
                    plan,
                    raw_by_case[case_id],
                    gate08_predictions[case_id],
                )
            )

    prediction_path = args.out_dir / "predictions.jsonl.gz"
    prediction_hash = write_jsonl_gzip(prediction_path, records)

    write(
        args.out_dir / "prediction-manifest.json",
        {
            "record_count": len(plans),
            "gzip_record_count_including_header": len(records),
            "prediction_sha256": prediction_hash,
            "final_pool_k": BUDGETS["final_pool_k"],
            "lane_k": BUDGETS["lane_k"],
            "rrf_k": BUDGETS["rrf_k"],
        },
    )

    write(
        args.out_dir / "prediction-seal.json",
        {
            "prediction_count": len(plans),
            "gold_reads_before_seal": 0,
            "governance_reads_before_seal": 0,
            "reference_answer_reads_before_seal": 0,
            "index_reads_before_seal": len(plans),
            "retrieval_runs": len(plans),
            "reranker_calls": 0,
            "calculator_calls": 0,
            "answer_generation_calls": 0,
            "parameter_scan": False,
            "per_query_oracle": False,
            "prediction_hash": prediction_hash,
            "protocol_hash": sha(args.out_dir / "gate-08-r2-protocol.json"),
            "sealed": True,
        },
    )

    print("Gate 08 R2 candidate direct retrieval prediction complete.")
    print(f"  Prediction count: {len(plans)}")
    print(f"  Prediction hash:  {prediction_hash}")
    print(f"  Output:           {prediction_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
