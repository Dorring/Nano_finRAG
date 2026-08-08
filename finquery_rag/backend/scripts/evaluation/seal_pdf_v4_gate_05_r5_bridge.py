#!/usr/bin/env python3
"""Gate 05 R5 — Bridge Seal.

Seals the Candidate Evidence Bridge by computing deterministic hashes
over bridge results and structured views.

Usage:
    python3 scripts/evaluation/seal_pdf_v4_gate_05_r5_bridge.py

Outputs:
    seal-manifest.json — sealed bridge manifest with hashes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.pdf_retrieval_v4.candidate_bridge_models import (  # noqa: E402
    build_bridge_manifest_hash,
    build_candidate_view_hash,
)

OUTPUT_DIR = BACKEND_DIR / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5"


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    print("=" * 70)
    print("Gate 05 R5 — Bridge Seal")
    print("=" * 70)

    # Load bridge results
    results_path = OUTPUT_DIR / "bridge-results.jsonl"
    views_path = OUTPUT_DIR / "structured-views.jsonl"
    summary_path = OUTPUT_DIR / "bridge-summary.json"
    validation_path = OUTPUT_DIR / "bridge-validation.json"

    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run the bridge builder first.")
        return 1

    # Load results
    results = list(read_jsonl(results_path))
    print(f"Loaded {len(results)} bridge results")

    # Load views
    views = list(read_jsonl(views_path))
    print(f"Loaded {len(views)} structured views")

    # Load summary
    with open(summary_path) as f:
        summary = json.load(f)

    # Load validation
    with open(validation_path) as f:
        validation = json.load(f)

    # Compute hashes
    view_hash = build_candidate_view_hash(views)
    bridge_hash = build_bridge_manifest_hash(
        candidate_count=summary["total_candidates"],
        grade_a_count=summary["grade_a_count"],
        grade_b_count=summary["grade_b_count"],
        unmapped_count=summary["unmapped_count"],
        view_hash=view_hash,
    )

    # Build seal manifest
    manifest = {
        "schema": "pdf-retrieval-v4/gate-05-r5/seal/v1",
        "candidate_count": summary["total_candidates"],
        "structured_eligible_count": summary["total_candidates"]
        - summary.get("failure_stage_counts", {}).get("candidate_type_unsupported", 0),
        "grade_a_bridge_count": summary["grade_a_count"],
        "grade_b_bridge_count": summary["grade_b_count"],
        "unmapped_count": summary["unmapped_count"],
        "grade_counts": summary["grade_counts"],
        # Governance: no access to Question/Gold/Governance
        "question_reads": 0,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        # No retrieval/index operations
        "bm25_builds": 0,
        "dense_builds": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
        # Validation status
        "pre_seal_validation_passed": validation["passed"],
        "violation_count": len(validation["violations"]),
        # Hashes
        "bridge_manifest_hash": bridge_hash,
        "candidate_view_hash": view_hash,
        # Seal
        "sealed": True,
    }

    # Write seal manifest
    seal_path = OUTPUT_DIR / "seal-manifest.json"
    with open(seal_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nSeal Manifest:")
    print(f"  candidate_count:           {manifest['candidate_count']}")
    print(f"  grade_a_bridge_count:      {manifest['grade_a_bridge_count']}")
    print(f"  grade_b_bridge_count:      {manifest['grade_b_bridge_count']}")
    print(f"  unmapped_count:            {manifest['unmapped_count']}")
    print(f"  pre_seal_validation_passed: {manifest['pre_seal_validation_passed']}")
    print(f"  bridge_manifest_hash:      {manifest['bridge_manifest_hash'][:32]}...")
    print(f"  candidate_view_hash:       {manifest['candidate_view_hash'][:32]}...")
    print(f"  sealed:                    {manifest['sealed']}")

    print(f"\nSeal manifest written to: {seal_path}")
    print("=" * 70)

    return 0 if manifest["pre_seal_validation_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
