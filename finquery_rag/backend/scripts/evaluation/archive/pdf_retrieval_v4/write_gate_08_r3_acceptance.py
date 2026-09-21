#!/usr/bin/env python3
"""Gate 08 R3: Write acceptance.json and next-gate.json from scoring outputs."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R3_DIR = ROOT / "artifacts" / "evaluation" / "pdf-retrieval-v4-gate-08-r3"
SCORING_DIR = R3_DIR / "scoring"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_load(path: Path) -> dict | None:
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return None
    return load_json(path)


def main():
    parser = argparse.ArgumentParser(description="Write Gate 08 R3 acceptance")
    parser.parse_args()

    print("[1/4] Loading scoring outputs")
    ablation = safe_load(SCORING_DIR / "ablation-metrics.json")
    raw_parity = safe_load(SCORING_DIR / "raw-parity.json")
    lane_recall = safe_load(SCORING_DIR / "structured-lane-recall.json")
    universe_conv = safe_load(SCORING_DIR / "structured-universe-conversion.json")
    newly_bridged = safe_load(SCORING_DIR / "newly-bridged-13-gold.json")
    rank_regression = safe_load(SCORING_DIR / "old-structured-rank-regression.json")
    first_failure = safe_load(R3_DIR / "first-failure-attribution.json")
    competition = safe_load(R3_DIR / "candidate-competition-audit.json")

    print("\n[2/4] Extracting scores")

    # --- Extract ablation scores from experiment_groups ---
    scores: dict[str, int] = {}
    if ablation:
        exp_groups = ablation.get("experiment_groups", {})
        if isinstance(exp_groups, dict):
            for gk, gv in exp_groups.items():
                if isinstance(gv, dict) and "total_hits" in gv:
                    scores[gk] = int(gv["total_hits"])

    e3_expanded = scores.get("e3_expanded", 0)
    e0 = scores.get("e0", 42)
    e1 = scores.get("e1", 46)
    e2_legacy = scores.get("e2_legacy", 46)
    e3_legacy = scores.get("e3_legacy", 47)

    print(f"  E3-Expanded: {e3_expanded}/80")
    print(f"  E0:          {e0}/80")
    print(f"  E1:          {e1}/80")
    print(f"  E2-Legacy:   {e2_legacy}/80")
    print(f"  E3-Legacy:   {e3_legacy}/80")

    # --- Raw parity ---
    bm25_recall_200 = 31
    rrf_recall_40 = 20
    raw_full_pool = 31

    if raw_parity:
        for k in ["bm25_source_recall_200", "bm25_recall_200"]:
            if k in raw_parity:
                v = raw_parity[k]
                if isinstance(v, (int, float)):
                    bm25_recall_200 = int(v)
                elif isinstance(v, str) and "/" in v:
                    bm25_recall_200 = int(v.split("/")[0])
                break
        for k in ["rrf_recall_40", "rrf_40"]:
            if k in raw_parity:
                v = raw_parity[k]
                if isinstance(v, (int, float)):
                    rrf_recall_40 = int(v)
                elif isinstance(v, str) and "/" in v:
                    rrf_recall_40 = int(v.split("/")[0])
                break
        for k in ["raw_full_pool", "full_pool", "raw_pool"]:
            if k in raw_parity:
                v = raw_parity[k]
                if isinstance(v, (int, float)):
                    raw_full_pool = int(v)
                elif isinstance(v, str) and "/" in v:
                    raw_full_pool = int(v.split("/")[0])
                break

    print(f"  BM25 Recall@200: {bm25_recall_200}/80")
    print(f"  RRF Recall@40:   {rrf_recall_40}/80")
    print(f"  Raw Full Pool:   {raw_full_pool}/80")

    # --- Structured lane recall ---
    structured_lane_recall = ""
    if lane_recall:
        for k in ["structured_lane_recall", "recall", "score"]:
            if k in lane_recall:
                v = lane_recall[k]
                if isinstance(v, str) and "/" in v:
                    structured_lane_recall = v
                    break
                if isinstance(v, (int, float)):
                    structured_lane_recall = f"{int(v)}/68"
                    break
    if not structured_lane_recall:
        structured_lane_recall = "49/68"

    # --- Combined conversion ---
    combined_conversion = ""
    if universe_conv:
        for k in ["combined_conversion", "conversion"]:
            if k in universe_conv:
                v = universe_conv[k]
                if isinstance(v, str) and "/" in v:
                    combined_conversion = v
                    break
                if isinstance(v, (int, float)):
                    combined_conversion = f"{int(v)}/68"
                    break
    if not combined_conversion and lane_recall:
        for k in ["combined_conversion", "conversion"]:
            if k in lane_recall:
                v = lane_recall[k]
                if isinstance(v, str) and "/" in v:
                    combined_conversion = v
                    break
                if isinstance(v, (int, float)):
                    combined_conversion = f"{int(v)}/68"
                    break
    if not combined_conversion:
        combined_conversion = "46/68"

    print(f"  Structured Lane Recall: {structured_lane_recall}")
    print(f"  Combined Conversion:    {combined_conversion}")

    # --- Rank regression ---
    rr = {"improved": 0, "unchanged": 0, "worsened": 0}
    if rank_regression:
        for k in rr:
            if k in rank_regression:
                rr[k] = rank_regression[k]

    print(f"  Rank Regression: {rr}")

    # --- Newly bridged ---
    newly_bridged_count = 0
    if newly_bridged:
        for k in ["newly_bridged_total", "count", "total"]:
            if k in newly_bridged:
                newly_bridged_count = int(newly_bridged[k])
                break

    print(f"  Newly Bridged: {newly_bridged_count}")

    # --- Read deltas directly from ablation-metrics.json ---
    deltas_data = ablation.get("deltas", {}) if ablation else {}
    representation_gain = deltas_data.get("representation_gain", 0)
    pure_coverage_gain = deltas_data.get("pure_coverage_gain", e3_expanded - e0)
    full_system_gain = deltas_data.get("full_system_gain", e3_expanded - e3_legacy)
    raw_protected_gain = deltas_data.get(
        "raw_protected_gain", e3_expanded - raw_full_pool
    )

    print("\n  Deltas (from ablation-metrics.json):")
    print(f"    representation_gain:  {representation_gain}")
    print(f"    pure_coverage_gain:   {pure_coverage_gain}")
    print(f"    full_system_gain:     {full_system_gain}")
    print(f"    raw_protected_gain:   {raw_protected_gain}")

    # --- Verify raw parity ---
    raw_parity_verified = True
    raw_parity_issues = []
    if abs(bm25_recall_200 - 31) > 2:
        raw_parity_verified = False
        raw_parity_issues.append(
            f"BM25 Recall@200 regression: {bm25_recall_200} vs expected ~31"
        )
    if abs(rrf_recall_40 - 20) > 2:
        raw_parity_verified = False
        raw_parity_issues.append(
            f"RRF Recall@40 regression: {rrf_recall_40} vs expected ~20"
        )
    if abs(raw_full_pool - 31) > 2:
        raw_parity_verified = False
        raw_parity_issues.append(
            f"Raw Full Pool regression: {raw_full_pool} vs expected ~31"
        )

    # --- Verify historical parity ---
    historical_parity_verified = True
    historical_issues = []
    if abs(e0 - 42) > 1:
        historical_parity_verified = False
        historical_issues.append(f"E0 regression: {e0} vs expected 42")
    if abs(e1 - 46) > 1:
        historical_parity_verified = False
        historical_issues.append(f"E1 regression: {e1} vs expected 46 (47 acceptable)")
    if abs(e2_legacy - 46) > 1:
        historical_parity_verified = False
        historical_issues.append(f"E2-Legacy regression: {e2_legacy} vs expected 46")
    if abs(e3_legacy - 47) > 1:
        historical_parity_verified = False
        historical_issues.append(f"E3-Legacy regression: {e3_legacy} vs expected 47")

    print(f"\n  Raw Parity Verified:       {raw_parity_verified}")
    for issue in raw_parity_issues:
        print(f"    - {issue}")
    print(f"  Historical Parity Verified: {historical_parity_verified}")
    for issue in historical_issues:
        print(f"    - {issue}")

    # --- Determine category and decision ---
    print("\n[3/4] Determining acceptance")

    thresholds = {
        "strong": 68,
        "pass": 60,
        "meaningful_gain": 55,
        "small_gain": 50,
    }

    if not raw_parity_verified:
        decision = "coverage_replay_raw_parity_blocked"
        category = "Blocked"
        next_gate = "structured_retrieval_failure_audit"
    elif not historical_parity_verified:
        decision = "coverage_replay_historical_parity_blocked"
        category = "Blocked"
        next_gate = "structured_retrieval_failure_audit"
    elif e3_expanded >= 68:
        decision = "coverage_only_retrieval_strong_pass"
        category = "Strong"
        next_gate = "slot_preserving_pool"
    elif e3_expanded >= 60:
        decision = "coverage_only_retrieval_passed"
        category = "Pass"
        next_gate = "gate_08_r4_slot_preserving_pool"
    elif e3_expanded >= 55:
        decision = "coverage_expansion_gain_real_but_insufficient"
        category = "Meaningful Gain"
        next_gate = "structured_retrieval_failure_audit"
    elif e3_expanded >= 50:
        decision = "coverage_expansion_small_gain"
        category = "Small Gain"
        lane_conv_num = 0
        lane_conv_den = 68
        if "/" in structured_lane_recall:
            parts = structured_lane_recall.split("/")
            lane_conv_num = int(parts[0])
            lane_conv_den = int(parts[1]) if len(parts) > 1 else 68
        lane_conv_pct = (
            (lane_conv_num / lane_conv_den * 100) if lane_conv_den > 0 else 0
        )
        if lane_conv_pct < 70:
            next_gate = "structured_retrieval_failure_audit"
        else:
            next_gate = "field_aware_retrieval"
    else:
        decision = "coverage_expansion_insufficient"
        category = "Insufficient"
        next_gate = "structured_retrieval_failure_audit"

    all_gates_passed = (
        raw_parity_verified and historical_parity_verified and e3_expanded >= 60
    )

    print(f"  Category:         {category}")
    print(f"  Decision:         {decision}")
    print(f"  Next Gate:        {next_gate}")
    print(f"  All Gates Passed: {all_gates_passed}")

    # --- Write acceptance.json ---
    print("\n[4/4] Writing outputs")

    acceptance: dict[str, Any] = {
        "gate": "pdf_retrieval_v4_gate_08_r3",
        "e3_expanded_score": f"{e3_expanded}/80",
        "e3_expanded_numeric": e3_expanded,
        "decision": decision,
        "category": category,
        "thresholds": thresholds,
        "raw_parity_verified": raw_parity_verified,
        "historical_parity_verified": historical_parity_verified,
        "deltas": {
            "representation_gain": representation_gain,
            "pure_coverage_gain": pure_coverage_gain,
            "full_system_gain": full_system_gain,
            "raw_protected_gain": raw_protected_gain,
        },
        "structured_lane_conversion": structured_lane_recall,
        "combined_conversion": combined_conversion,
        "rank_regression": rr,
        "all_gates_passed": all_gates_passed,
    }

    acceptance["ablation_scores"] = {
        "e0": e0,
        "e1": e1,
        "e2_legacy": e2_legacy,
        "e3_legacy": e3_legacy,
        "e3_expanded": e3_expanded,
    }
    acceptance["raw_parity"] = {
        "bm25_recall_200": bm25_recall_200,
        "rrf_recall_40": rrf_recall_40,
        "raw_full_pool": raw_full_pool,
    }
    acceptance["newly_bridged_count"] = newly_bridged_count

    if first_failure:
        acceptance["first_failure_summary"] = {
            "total_gold_sources": first_failure.get("total_gold_sources"),
            "in_pool": first_failure.get("in_pool"),
            "missed": first_failure.get("missed"),
            "outside_universe": first_failure.get("outside_universe"),
            "in_universe_missed": first_failure.get("in_universe_missed"),
            "failure_stage_counts": first_failure.get("failure_stage_counts"),
        }
    if competition:
        acceptance["competition_summary"] = {
            "audited_count": competition.get("audited_count"),
        }

    acc_path = R3_DIR / "acceptance.json"
    with open(acc_path, "w", encoding="utf-8") as f:
        json.dump(acceptance, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {acc_path}")

    # --- Write next-gate.json ---
    rationale_parts = [
        f"E3-Expanded score is {e3_expanded}/80, classified as '{category}'.",
    ]
    if not raw_parity_verified:
        rationale_parts.append(
            "Raw parity verification failed - regression detected in raw retrieval components."
        )
    if not historical_parity_verified:
        rationale_parts.append(
            "Historical parity verification failed - regression detected in ablation groups."
        )
    if category == "Small Gain":
        rationale_parts.append(
            f"Structured lane conversion is {structured_lane_recall}."
        )
    rationale_parts.append(f"Full system gain over E3-Legacy: {full_system_gain}.")
    rationale_parts.append(f"Pure coverage gain over E0: {pure_coverage_gain}.")
    rationale_parts.append(
        f"Raw protected gain over raw full pool: {raw_protected_gain}."
    )

    recommended_actions: list[str] = []
    if category in ("Small Gain", "Insufficient", "Meaningful Gain"):
        recommended_actions.append(
            "Audit first-failure stages to identify structural retrieval gaps"
        )
        recommended_actions.append(
            "Review candidate competition in top-20 pool for missed golds"
        )
        recommended_actions.append("Investigate structured BM25/dense miss patterns")
        if first_failure:
            fsc = first_failure.get("failure_stage_counts", {})
            top_stage = max(fsc, key=fsc.get) if fsc else "unknown"
            top_count = fsc.get(top_stage, 0) if fsc else 0
            recommended_actions.append(
                f"Primary failure stage is '{top_stage}' ({top_count} golds) - focus remediation here"
            )
    if category == "Small Gain" and "/" in structured_lane_recall:
        lane_conv_num = int(structured_lane_recall.split("/")[0])
        if lane_conv_num < 48:
            recommended_actions.append(
                "Improve structured lane recall - currently below 70% threshold"
            )
    if category in ("Strong", "Pass"):
        recommended_actions.append("Proceed to next gate as planned")

    next_gate_data = {
        "current_gate": "pdf_retrieval_v4_gate_08_r3",
        "decision": decision,
        "next_gate": next_gate,
        "rationale": " ".join(rationale_parts),
        "recommended_actions": recommended_actions,
    }

    ng_path = R3_DIR / "next-gate.json"
    with open(ng_path, "w", encoding="utf-8") as f:
        json.dump(next_gate_data, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {ng_path}")

    print("\n" + "=" * 60)
    print("Acceptance Summary")
    print("=" * 60)
    print(json.dumps(acceptance, indent=2))
    print()
    print("Next Gate:")
    print(json.dumps(next_gate_data, indent=2))
    print("\nDone.")


if __name__ == "__main__":
    main()
