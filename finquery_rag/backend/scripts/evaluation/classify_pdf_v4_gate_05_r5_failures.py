#!/usr/bin/env python3
"""Gate 05 R5 — Classify First Failure Stages.

For each unmapped Gold candidate, determines the precise first_failure_stage
from the bridge results.

Failure stages:
    candidate_type_unsupported
    candidate_bbox_missing
    candidate_text_signature_mismatch
    numeric_signature_mismatch
    metric_signature_mismatch
    period_signature_mismatch
    multirow_required
    narrative_bridge_missing
    multiple_equal_matches
    semantic_evidence_fanout_ambiguous
    legacy_candidate_granularity_mismatch

Usage:
    python3 scripts/evaluation/classify_pdf_v4_gate_05_r5_failures.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

OUTPUT_DIR = BACKEND_DIR / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5"
GOLD_MAP_PATH = (
    BACKEND_DIR
    / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r1/gold-structural-map.json"
)


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    print("=" * 70)
    print("Gate 05 R5 — Classify First Failure Stages")
    print("=" * 70)

    # Load gold structural map
    with open(GOLD_MAP_PATH) as f:
        gold_map = json.load(f)
    matches = gold_map["matches"]

    # Load bridge results
    bridge_results = list(read_jsonl(OUTPUT_DIR / "bridge-results.jsonl"))
    bridge_by_key = {r["candidate_key"]: r for r in bridge_results}

    # Load universe scoring for context
    universe_path = OUTPUT_DIR / "universe-scoring.json"
    universe = None
    if universe_path.exists():
        with open(universe_path) as f:
            universe = json.load(f)

    # Classify each unmapped gold
    failures: list[dict] = []
    failure_counts: Counter = Counter()

    for match in matches:
        gold_key = match.get("gold_candidate_key") or ""
        case_id = match.get("case_id") or ""

        bridge_result = bridge_by_key.get(gold_key)

        if bridge_result is None:
            failure_stage = "candidate_not_found"
        elif bridge_result.get("grade") == "unmapped":
            failure_stage = bridge_result.get("failure_stage") or "unknown"
        elif bridge_result.get("grade") == "B_ambiguous":
            failure_stage = "multiple_equal_matches"
        else:
            # Mapped or Grade-A — no failure
            continue

        failure_counts[failure_stage] += 1
        failures.append(
            {
                "case_id": case_id,
                "gold_candidate_key": gold_key[:80],
                "gold_document_id": match.get("gold_document_id"),
                "gold_page": match.get("gold_page"),
                "gold_metric": match.get("gold_metric"),
                "gold_period": match.get("gold_period"),
                "gold_row_label": match.get("gold_row_label"),
                "first_failure_stage": failure_stage,
                "bridge_grade": bridge_result.get("grade") if bridge_result else None,
                "bridge_reasons": bridge_result.get("bridge_reasons", [])
                if bridge_result
                else [],
            }
        )

    # Build result
    result = {
        "total_unmapped_gold": len(failures),
        "failure_stage_counts": dict(failure_counts.most_common()),
        "failures": failures,
    }

    # Add context from universe scoring if available
    if universe:
        result["universe_context"] = {
            "total_gold": universe["total_gold"],
            "mapped": universe["mapped"],
            "unmapped": universe["unmapped"],
            "ambiguous": universe["ambiguous"],
            "raw_only": universe["raw_only"],
        }

    # Write result
    output_path = OUTPUT_DIR / "failure-classification.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\nTotal unmapped gold: {len(failures)}")
    print("\nFailure Stage Breakdown:")
    for stage, count in failure_counts.most_common():
        print(f"  {stage}: {count}")

    print(f"\nResult written to: {output_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
