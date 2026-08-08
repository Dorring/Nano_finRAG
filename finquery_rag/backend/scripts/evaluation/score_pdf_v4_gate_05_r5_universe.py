#!/usr/bin/env python3
"""Gate 05 R5 — Score 80 Strict Candidate Universe.

Evaluates whether Grade-A Structured Candidate Views cover the 80 Strict
Gold Identity records.

Usage:
    python3 scripts/evaluation/score_pdf_v4_gate_05_r5_universe.py

Gates:
    Minimum (next gate):  >= 68/80 (85%)
    Recommended:           >= 72/80 (90%)
    Strong:                >= 76/80 (95%)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.pdf_retrieval_v4.candidate_bridge_models import BridgeGrade  # noqa: E402

OUTPUT_DIR = BACKEND_DIR / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5"
GOLD_MAP_PATH = (
    BACKEND_DIR
    / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r1/gold-structural-map.json"
)

# Gate thresholds
MINIMUM_THRESHOLD = 68
RECOMMENDED_THRESHOLD = 72
STRONG_THRESHOLD = 76


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    print("=" * 70)
    print("Gate 05 R5 — Score 80 Strict Candidate Universe")
    print("=" * 70)

    # Load gold structural map
    with open(GOLD_MAP_PATH) as f:
        gold_map = json.load(f)

    total_gold = gold_map["total_gold"]
    baseline_in_universe = gold_map["in_structured_universe"]
    matches = gold_map["matches"]

    print(f"\nGold records: {total_gold}")
    print(f"Baseline (pre-bridge): {baseline_in_universe}/{total_gold}")

    # Load bridge results and structured views
    bridge_results = list(read_jsonl(OUTPUT_DIR / "bridge-results.jsonl"))
    structured_views = list(read_jsonl(OUTPUT_DIR / "structured-views.jsonl"))

    # Build lookup: candidate_key → bridge result
    bridge_by_key = {r["candidate_key"]: r for r in bridge_results}

    # Build lookup: candidate_key → structured view
    view_by_key = {v["candidate_key"]: v for v in structured_views}

    # Score each gold record
    mapped = 0
    unmapped = 0
    ambiguous = 0
    raw_only = 0
    details: list[dict] = []

    for match in matches:
        gold_key = match.get("gold_candidate_key") or ""
        case_id = match.get("case_id") or ""
        was_in_universe = match.get("in_structured_universe", False)
        old_method = match.get("mapping_method") or ""

        bridge_result = bridge_by_key.get(gold_key)
        view = view_by_key.get(gold_key)

        if bridge_result is None:
            # Candidate not found in bridge results
            unmapped += 1
            status = "unmapped"
            new_method = "candidate_not_found"
        elif view is not None:
            # Has structured view → mapped
            mapped += 1
            status = "mapped"
            new_method = f"bridge_{bridge_result.get('grade', 'unknown')}"
        elif bridge_result.get("grade") == BridgeGrade.B_AMBIGUOUS.value:
            ambiguous += 1
            status = "ambiguous"
            new_method = "bridge_ambiguous"
        elif BridgeGrade.is_grade_a(bridge_result.get("grade", "")):
            # Grade-A but no view (shouldn't happen, but handle)
            mapped += 1
            status = "mapped"
            new_method = f"bridge_{bridge_result['grade']}_no_view"
        else:
            raw_only += 1
            status = "raw_only"
            failure = bridge_result.get("failure_stage") or "unknown"
            new_method = f"bridge_unmapped_{failure}"

        details.append(
            {
                "case_id": case_id,
                "gold_candidate_key": gold_key[:80],
                "gold_document_id": match.get("gold_document_id"),
                "gold_page": match.get("gold_page"),
                "gold_metric": match.get("gold_metric"),
                "gold_period": match.get("gold_period"),
                "was_in_structured_universe": was_in_universe,
                "old_mapping_method": old_method,
                "new_status": status,
                "new_mapping_method": new_method,
                "bridge_grade": bridge_result.get("grade") if bridge_result else None,
                "bridge_failure_stage": bridge_result.get("failure_stage")
                if bridge_result
                else None,
            }
        )

    # Compute coverage
    coverage = mapped
    coverage_rate = coverage / total_gold if total_gold > 0 else 0.0

    # Determine gate status
    if coverage >= STRONG_THRESHOLD:
        gate_status = "strong_pass"
    elif coverage >= RECOMMENDED_THRESHOLD:
        gate_status = "recommended_pass"
    elif coverage >= MINIMUM_THRESHOLD:
        gate_status = "minimum_pass"
    else:
        gate_status = "fail"

    # Build result
    result = {
        "total_gold": total_gold,
        "baseline_in_universe": baseline_in_universe,
        "mapped": mapped,
        "unmapped": unmapped,
        "ambiguous": ambiguous,
        "raw_only": raw_only,
        "coverage": coverage,
        "coverage_rate": round(coverage_rate, 4),
        "improvement": coverage - baseline_in_universe,
        "gate_status": gate_status,
        "thresholds": {
            "minimum": MINIMUM_THRESHOLD,
            "recommended": RECOMMENDED_THRESHOLD,
            "strong": STRONG_THRESHOLD,
        },
        "details": details,
    }

    # Write result
    output_path = OUTPUT_DIR / "universe-scoring.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\nStrict Candidate Universe Coverage:")
    print(f"  Mapped (Grade-A):   {mapped}/{total_gold}")
    print(f"  Unmapped:           {unmapped}/{total_gold}")
    print(f"  Ambiguous:          {ambiguous}/{total_gold}")
    print(f"  Raw-only:           {raw_only}/{total_gold}")
    print(f"  Coverage:           {coverage}/{total_gold} ({coverage_rate:.1%})")
    print(
        f"  Improvement:        +{coverage - baseline_in_universe} (from {baseline_in_universe})"
    )
    print(f"\n  Gate Status:        {gate_status}")
    print(
        f"  Thresholds:         min={MINIMUM_THRESHOLD}, rec={RECOMMENDED_THRESHOLD}, strong={STRONG_THRESHOLD}"
    )

    print(f"\nResult written to: {output_path}")
    print("=" * 70)

    return 0 if gate_status != "fail" else 2


if __name__ == "__main__":
    sys.exit(main())
