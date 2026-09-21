"""Gate 08A: freeze and verify the authoritative Raw production replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-0/production-stage-replay.json"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    expected = {
        "bm25_source_recall_at_200": 37,
        "dense_source_recall_at_200": 14,
        "rrf_source_recall_at_40": 20,
    }
    metrics = replay.get("metrics", {})
    metric_parity = {key: metrics.get(key) == value for key, value in expected.items()}
    raw_cases: list[dict[str, Any]] = []
    for case in replay.get("cases", []):
        stages = case.get("stages") or {}
        raw = list(stages.get("rrf_full") or stages.get("reranker_input") or [])
        raw_cases.append({
            "case_id": str(case.get("case_id")),
            "raw_full_rrf_candidate_count": len(raw),
            "raw_full_rrf_candidates": raw,
            "raw_rrf_at_40": list(stages.get("rrf") or [])[:40],
            "raw_candidate_ids": [str(item.get("candidate_key")) for item in raw],
            "raw_candidate_ranks": {str(item.get("candidate_key")): item.get("stage_rank") for item in raw},
            "raw_candidate_scores": {str(item.get("candidate_key")): item.get("score") for item in raw},
            "raw_candidate_hash": hashlib.sha256(json.dumps(
                [{"candidate_key": item.get("candidate_key"), "stage_rank": item.get("stage_rank"), "score": item.get("score")} for item in raw],
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest(),
        })
    protocol = {
        "schema": "pdf-retrieval-v4/gate-08a/raw-parity/v1",
        "gate": "pdf_retrieval_v4_gate_08a",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "implementation_commit": args.code_commit,
        "artifact_seal_commit": args.code_commit,
        "replay_sha256": sha(args.replay),
        "authoritative_metrics": expected | {"raw_full_pool_recall": "score_after_prediction_seal"},
        "prediction_reads": ["frozen_gate_0_production_stage_replay"],
        "forbidden_reads": ["labels.golden.jsonl", "benchmark-governance.jsonl", "gold_source", "expected_value", "reference_answer"],
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
        "parameter_scan": False,
        "per_query_oracle": False,
    }
    write(args.out_dir / "gate-08a-protocol.json", protocol)
    write(args.out_dir / "raw-parity.json", {
        "case_count": len(raw_cases),
        "metric_parity": metric_parity,
        "authoritative_metrics": expected,
        "raw_cases": raw_cases,
        "raw_candidate_order_frozen": True,
        "raw_score_frozen": True,
        "gate_passed": len(raw_cases) == 72 and all(metric_parity.values()),
    })
    write(args.out_dir / "gate-08a-input-integrity.json", {"replay_sha256": sha(args.replay), "case_count": len(raw_cases)})
    return 0 if len(raw_cases) == 72 and all(metric_parity.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
