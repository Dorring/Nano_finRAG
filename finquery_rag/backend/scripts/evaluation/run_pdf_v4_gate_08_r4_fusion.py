#!/usr/bin/env python3
"""Run zero-search Gate 08 R4 F0/F1/F2 fusion predictions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.lane_preserving_fusion import (  # noqa: E402
    fuse_multi_slot_families,
    fuse_single_slot_families,
)

R3_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3"
RS_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3-rs"
R4_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r4"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_gz(path: Path, records: list[dict[str, Any]]) -> str:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            for record in records:
                line = json.dumps(record, sort_keys=True, separators=(",", ":"))
                handle.write((line + "\n").encode())
    return _sha(path)


def _combine(e0: list[dict[str, Any]], residual: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*e0, *residual]:
        key = str(item.get("candidate_key") or "")
        if key and key not in seen:
            seen.add(key)
            result.append({"candidate_key": key, "rank": len(result) + 1})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=R4_DIR)
    args = parser.parse_args()
    composite_path = RS_DIR / "r4-composite-input-manifest.json"
    composite = _json(composite_path)
    if not composite.get("sealed"):
        raise RuntimeError("r4_composite_input_not_sealed")
    r3_path = R3_DIR / "predictions.jsonl.gz"
    sidecar_path = RS_DIR / "slot-local-rankings.jsonl.gz"
    if composite["original_r3"]["prediction_sha256"] != _sha(r3_path):
        raise RuntimeError("original_r3_hash_mismatch")
    if composite["slot_local_sidecar"]["prediction_sha256"] != _sha(sidecar_path):
        raise RuntimeError("slot_sidecar_hash_mismatch")

    r3 = {item["case_id"]: item for item in _load_gz(r3_path)}
    sidecar = {item["case_id"]: item for item in _load_gz(sidecar_path)}
    predictions: list[dict[str, Any]] = []
    for case_id in sorted(r3):
        source = r3[case_id]
        e0 = list(source.get("e0_pool") or [])
        f0 = list(source.get("e3_expanded_pool") or [])
        if source.get("is_multi_slot"):
            if case_id not in sidecar:
                raise RuntimeError(f"missing_sidecar:{case_id}")
            f2_residual, slot_traces = fuse_multi_slot_families(
                sidecar[case_id]["slot_family_rankings"]
            )
            f1 = f0
            f1_trace: list[dict[str, Any]] = []
            f2 = _combine(e0, f2_residual)
        else:
            family_trace = fuse_single_slot_families(
                (source.get("candidate_raw") or {}).get("fused") or [],
                (source.get("structured_expanded") or {}).get("fused") or [],
            )
            f1 = _combine(e0, family_trace)
            f2 = f1
            f1_trace = family_trace
            slot_traces = {}
            f2_residual = family_trace
        predictions.append(
            {
                "case_id": case_id,
                "is_multi_slot": bool(source.get("is_multi_slot")),
                "f0_pool": f0,
                "f1_pool": f1,
                "f2_pool": f2,
                "f1_trace": f1_trace,
                "f2_trace": f2_residual,
                "slot_local_traces": slot_traces,
                "e0_prefix_exact": [x["candidate_key"] for x in f2[: len(e0)]]
                == [x["candidate_key"] for x in e0],
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / "fusion-predictions.jsonl.gz"
    pred_hash = _write_gz(pred_path, predictions)
    protocol = {
        "gate": "pdf_retrieval_v4_gate_08_r4",
        "frozen_input": "gate_08_r3_r4_composite_seal",
        "rrf_k": 60,
        "final_pool_k": 40,
        "slot_top_k": 20,
        "structured_protected_k": 20,
        "multi_slot_structured_opportunity_k": 1,
        "parameter_scan": False,
        "quota_scan": False,
        "bm25_searches": 0,
        "dense_searches": 0,
        "embedding_calls": 0,
        "index_reads": 0,
        "index_builds": 0,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
    }
    _write(args.out_dir / "protocol.json", protocol)
    manifest = {
        "prediction_count": len(predictions),
        "prediction_sha256": pred_hash,
        "composite_input_sha256": _sha(composite_path),
    }
    _write(args.out_dir / "prediction-manifest.json", manifest)
    seal = {
        **protocol,
        **manifest,
        "sealed": True,
        "raw_e0_prefix_exact_cases": sum(x["e0_prefix_exact"] for x in predictions),
    }
    _write(args.out_dir / "prediction-seal.json", seal)
    print(f"prediction_count={len(predictions)} hash={pred_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
