"""Gate 03 R2: Seal the Full-corpus Financial Semantic Graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT))

GATE03_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-03-r2"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Seal Gate 03 R2 semantic graph")
    parser.add_argument("--backend-root", type=str, default=str(ROOT))
    parser.add_argument("--output-dir", type=str, default=str(GATE03_OUT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    metrics_path = output_dir / "semantic-graph-metrics.json"
    equivalence_path = output_dir / "semantic-equivalence.json"

    if not metrics_path.exists():
        print(f"ERROR: metrics file not found: {metrics_path}", file=sys.stderr)
        return 1

    metrics = _read_json(metrics_path)
    equivalence = _read_json(equivalence_path) if equivalence_path.exists() else {}

    prediction_hash = hashlib.sha256(
        json.dumps(metrics, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    if not metrics.get("all_passed", False):
        print("ERROR: not all gates passed (all_passed != true)", file=sys.stderr)
        return 1

    safety_counters = {
        "question_reads": metrics.get("question_reads", 0),
        "gold_reads_before_seal": metrics.get("gold_reads_before_seal", 0),
        "governance_reads_before_seal": metrics.get("governance_reads_before_seal", 0),
        "candidate_bridge_builds": metrics.get("candidate_bridge_builds", 0),
        "index_builds": metrics.get("index_builds", 0),
        "retrieval_runs": metrics.get("retrieval_runs", 0),
        "production_switch_allowed": metrics.get("production_switch_allowed", False),
    }

    expected_safety = {
        "question_reads": 0,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "candidate_bridge_builds": 0,
        "index_builds": 0,
        "retrieval_runs": 0,
        "production_switch_allowed": False,
    }

    for key, expected in expected_safety.items():
        actual = safety_counters[key]
        if actual != expected:
            print(
                f"ERROR: safety counter '{key}' = {actual!r}, expected {expected!r}",
                file=sys.stderr,
            )
            return 1

    seal_manifest: dict[str, Any] = {
        "documents": 8,
        "pages": 1348,
        "logical_tables": metrics.get("logical_tables"),
        "semantic_rows": metrics.get("semantic_rows"),
        "atomic_facts": metrics.get("atomic_facts"),
        "comparison_facts": metrics.get("comparison_facts"),
        "bucket_facts": metrics.get("bucket_facts"),
        "row_matrices": metrics.get("row_matrices"),
        "narrative_evidence": metrics.get("narrative_evidence"),
        "question_reads": 0,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "candidate_bridge_builds": 0,
        "index_builds": 0,
        "retrieval_runs": 0,
        "prediction_hash": prediction_hash,
        "sealed": True,
        "equivalence_summary": {
            "equivalent_sets": equivalence.get("equivalent_sets"),
            "total_equivalent_groups": equivalence.get("total_equivalent_groups"),
        },
    }

    seal_path = output_dir / "seal-manifest.json"
    _write_json(seal_path, seal_manifest)

    print(f"Seal manifest written: {seal_path}")
    print(f"prediction_hash: {prediction_hash}")
    print("Gate 03 R2 sealed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
