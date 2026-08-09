#!/usr/bin/env python3
"""Zero-search Gate 08 R8-R0 candidate-depth reconstruction and sealing."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R7 = (
    ROOT
    / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r7/field-family-predictions.jsonl.gz"
)
R7_SEAL = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r7/prediction-seal.json"
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r8-r0"
DEPTHS = (5, 10, 20, 40, 50)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    seal = json.loads(R7_SEAL.read_text())
    if not seal.get("sealed") or seal["prediction_sha256"] != sha(R7):
        raise RuntimeError("r7_seal_invalid")
    records = []
    lengths = []
    with gzip.open(R7, "rt", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            full = source["r7_full_pool"]
            keys = [item["candidate_key"] for item in full]
            if len(keys) != len(set(keys)):
                raise RuntimeError(f"duplicate_candidate:{source['case_id']}")
            lengths.append(len(full))
            records.append(
                {
                    "case_id": source["case_id"],
                    "sealed_pool_count": len(full),
                    **{f"candidate_pool_{depth}": full[:depth] for depth in DEPTHS},
                }
            )
    if len(records) != 72:
        raise RuntimeError("prediction_count_drift")
    OUT.mkdir(parents=True, exist_ok=True)
    prediction = OUT / "candidate-depth-predictions.jsonl.gz"
    with prediction.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as output:
            for record in sorted(records, key=lambda item: item["case_id"]):
                output.write(
                    (
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                )
    protocol = {
        "gate": "pdf_retrieval_v4_gate_08_r8_r0",
        "depths": list(DEPTHS),
        "bm25_reruns": 0,
        "dense_reruns": 0,
        "embedding_reruns": 0,
        "bridge_changes": 0,
        "query_changes": 0,
        "query_plan_changes": 0,
        "gold_reads_before_seal": 0,
        "reranker_calls": 0,
        "production_switch_allowed": False,
    }
    manifest = {
        "prediction_count": 72,
        "prediction_sha256": sha(prediction),
        "r7_prediction_sha256": sha(R7),
        "r7_seal_sha256": sha(R7_SEAL),
        "sealed_pool_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "count_below_50": sum(length < 50 for length in lengths),
        },
    }
    write("protocol.json", protocol)
    write("input-integrity.json", manifest)
    write("prediction-manifest.json", manifest)
    write("prediction-seal.json", {**protocol, **manifest, "sealed": True})
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
