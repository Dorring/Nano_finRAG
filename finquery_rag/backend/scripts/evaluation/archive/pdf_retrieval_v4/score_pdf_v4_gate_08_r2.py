"""Gate 08 R2: Score sealed R2 predictions against R1.1 B-class classification.

Loads the sealed R2 predictions, R1.1 gold-coverage-classification, and
Gate 08 raw predictions, then computes:

  1. Raw Full Pool Recall (31/80 baseline)
  2. Existing Structured Recall (25/80 baseline)
  3. Candidate Direct Recall (B-class in direct pool)
  4. Combined Strict Recall (42 + B-class recovered)
  5. B-class 22专项: recovered count, still missing, rank distribution
  6. Multi-evidence: complete structural/strict evidence

No predictions are re-run.  Gold/labels are read only AFTER the R2 seal
is verified.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_R2_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2"
DEFAULT_GATE08_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08"
DEFAULT_R11_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r1-1"
DEFAULT_LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2/scoring"

A_CLASS = "recovered_strict"
B_CLASS = "strict_mapped_not_retrieved"
C_CLASS = "structural_present_strict_unmapped"
D_CLASS = "structurally_absent"


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_predictions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if value.get("stream") != "header":
                    records.append(value)
    return records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _case_gold_sources(label: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in label.get("expected_sources") or []
        if item.get("candidate_key")
    ]


def _rank_bucket(rank: int) -> str:
    if rank <= 5:
        return "1-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    if rank <= 40:
        return "21-40"
    return "41+"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-out", type=Path, default=DEFAULT_R2_OUT)
    parser.add_argument("--gate08-out", type=Path, default=DEFAULT_GATE08_OUT)
    parser.add_argument("--r11-out", type=Path, default=DEFAULT_R11_OUT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Verify R2 seal and load sealed data
    # ------------------------------------------------------------------
    r2_seal_path = args.r2_out / "prediction-seal.json"
    r2_predictions_path = args.r2_out / "predictions.jsonl.gz"
    r2_seal = json.loads(r2_seal_path.read_text(encoding="utf-8"))
    if (
        not r2_seal.get("sealed")
        or r2_seal.get("gold_reads_before_seal") != 0
        or r2_seal.get("governance_reads_before_seal") != 0
    ):
        raise RuntimeError("r2_prediction_seal_invalid")

    r2_predictions_list = load_predictions(r2_predictions_path)
    r2_predictions = {
        str(item["case_id"]): item for item in r2_predictions_list
    }

    # Load Gate 08 raw predictions (for raw pool + structured pool).
    gate08_predictions_path = args.gate08_out / "retrieval-predictions.jsonl.gz"
    gate08_predictions_list = load_predictions(gate08_predictions_path)
    gate08_predictions = {
        str(item["case_id"]): item for item in gate08_predictions_list
    }

    # Load R1.1 gold-coverage-classification.
    r11_classification_path = args.r11_out / "gold-coverage-classification.json"
    r11_classification = json.loads(
        r11_classification_path.read_text(encoding="utf-8")
    )
    r11_rows = r11_classification.get("rows") or []
    r11_by_identity = {
        str(row["gold_source_identity"]): row for row in r11_rows
    }

    # Load labels.
    labels_list = load_jsonl(args.labels)
    labels = {str(item["case_id"]): item for item in labels_list}

    if set(r2_predictions) != set(labels):
        raise RuntimeError("prediction_label_case_set_mismatch")

    # ------------------------------------------------------------------
    # 2. Build per-gold-source analysis
    # ------------------------------------------------------------------
    raw_hits = 0
    structured_hits = 0
    combined_hits = 0
    b_class_total = 0
    b_class_recovered = 0
    b_class_rows: list[dict[str, Any]] = []
    b_class_rank_distribution: Counter[str] = Counter()
    multi_records: list[dict[str, Any]] = []

    total_gold = 0
    coverage_class_counts: Counter[str] = Counter()

    for case_id in sorted(labels):
        label = labels[case_id]
        sources = _case_gold_sources(label)
        r2_pred = r2_predictions[case_id]
        gate08_pred = gate08_predictions[case_id]

        raw_keys = {
            str(item.get("candidate_key"))
            for item in gate08_pred.get("raw_full_rrf_candidates") or []
            if item.get("candidate_key")
        }
        structured_keys = {
            str(item.get("original_candidate_identity"))
            for item in gate08_pred.get("structured_strict_source_pool") or []
            if item.get("original_candidate_identity")
        }
        direct_keys = {
            str(item.get("candidate_key"))
            for item in r2_pred.get("candidate_direct_pool") or []
            if item.get("candidate_key")
        }
        direct_rank_map = {
            str(item.get("candidate_key")): int(item.get("rank") or 0)
            for item in r2_pred.get("candidate_direct_pool") or []
            if item.get("candidate_key")
        }
        combined_keys = {
            str(item.get("candidate_key"))
            for item in r2_pred.get("combined_pool") or []
            if item.get("candidate_key")
        }

        is_multi_slot = bool(r2_pred.get("is_multi_slot"))
        slot_pool_keys: set[str] = set()
        for slot_hits in (r2_pred.get("slot_pools") or {}).values():
            for item in slot_hits:
                if item.get("candidate_key"):
                    slot_pool_keys.add(str(item["candidate_key"]))

        case_gold_count = 0
        case_gold_in_combined = 0
        case_gold_in_slot_pools = 0
        case_complete = True

        for idx, source in enumerate(sources):
            gold_key = str(source.get("candidate_key"))
            identity = f"{case_id}#{idx}"
            r11_row = r11_by_identity.get(identity, {})
            coverage_class = str(r11_row.get("coverage_class") or "")

            total_gold += 1
            coverage_class_counts[coverage_class] += 1

            in_raw = gold_key in raw_keys
            in_structured = gold_key in structured_keys
            in_direct = gold_key in direct_keys
            in_combined = gold_key in combined_keys

            if in_raw:
                raw_hits += 1
            if in_structured:
                structured_hits += 1
            if in_combined:
                combined_hits += 1

            case_gold_count += 1
            if in_combined:
                case_gold_in_combined += 1
            else:
                case_complete = False
            if gold_key in slot_pool_keys:
                case_gold_in_slot_pools += 1

            if coverage_class == B_CLASS:
                b_class_total += 1
                if in_direct:
                    b_class_recovered += 1
                    rank = direct_rank_map.get(gold_key)
                    if rank:
                        b_class_rank_distribution[_rank_bucket(rank)] += 1
                    b_class_rows.append(
                        {
                            "gold_source_identity": identity,
                            "case_id": case_id,
                            "gold_candidate_key": gold_key,
                            "recovered": True,
                            "rank_in_candidate_direct_pool": rank,
                            "rank_bucket": _rank_bucket(rank) if rank else None,
                        }
                    )
                else:
                    b_class_rows.append(
                        {
                            "gold_source_identity": identity,
                            "case_id": case_id,
                            "gold_candidate_key": gold_key,
                            "recovered": False,
                            "rank_in_candidate_direct_pool": None,
                            "rank_bucket": None,
                        }
                    )

        if is_multi_slot and case_gold_count > 0:
            multi_records.append(
                {
                    "case_id": case_id,
                    "required_slot_count": len(r2_pred.get("slot_pools") or {}),
                    "gold_source_count": case_gold_count,
                    "gold_in_combined_pool": case_gold_in_combined,
                    "gold_in_slot_pools": case_gold_in_slot_pools,
                    "complete_evidence_available": case_complete,
                }
            )

    # ------------------------------------------------------------------
    # 3. Compute metrics
    # ------------------------------------------------------------------
    a_class_count = coverage_class_counts.get(A_CLASS, 0)
    combined_strict_recall_a_plus_b = a_class_count + b_class_recovered
    complete_multi = sum(
        1 for r in multi_records if r["complete_evidence_available"]
    )

    scoring_report = {
        "gate": "pdf_retrieval_v4_gate_08_r2",
        "prediction_seal_verified": True,
        "prediction_count": len(r2_predictions),
        "gold_source_count": total_gold,
        "gold_reads_before_seal": int(
            r2_seal.get("gold_reads_before_seal", -1)
        ),
        "governance_reads_before_seal": int(
            r2_seal.get("governance_reads_before_seal", -1)
        ),
        "raw_full_pool_recall": f"{raw_hits}/{total_gold}",
        "existing_structured_recall": f"{structured_hits}/{total_gold}",
        "candidate_direct_recall_b_class": f"{b_class_recovered}/{b_class_total}",
        "combined_strict_recall": f"{combined_hits}/{total_gold}",
        "combined_strict_recall_a_plus_b": f"{combined_strict_recall_a_plus_b}/{total_gold}",
        "coverage_class_counts": dict(coverage_class_counts),
        "a_class_recovered": a_class_count,
        "b_class_total": b_class_total,
        "b_class_recovered": b_class_recovered,
        "multi_evidence": {
            "multi_evidence_case_count": len(multi_records),
            "complete_evidence_availability": (
                f"{complete_multi}/{len(multi_records)}"
                if multi_records
                else "not_evaluable"
            ),
            "records": multi_records,
        },
    }

    b_class_detail = {
        "b_class_total": b_class_total,
        "b_class_recovered": b_class_recovered,
        "b_class_still_missing": b_class_total - b_class_recovered,
        "rank_distribution": dict(b_class_rank_distribution),
        "rows": b_class_rows,
    }

    rank_distribution = {
        "b_class_rank_distribution": dict(b_class_rank_distribution),
        "rank_bucket_definitions": {
            "1-5": "rank 1 to 5",
            "6-10": "rank 6 to 10",
            "11-20": "rank 11 to 20",
            "21-40": "rank 21 to 40",
            "41+": "rank 41 and above",
        },
    }

    # ------------------------------------------------------------------
    # 4. Acceptance and next-gate (per spec sections 14.1–14.4)
    # ------------------------------------------------------------------
    raw_gold_retained = raw_hits == 31  # 31 raw gold from baseline
    raw_gold_retained_ok = raw_gold_retained

    # 14.2 Strong pass: B ≥ 21/22 AND Combined ≥ 63/80
    strong_pass = (
        b_class_recovered >= 21
        and combined_hits >= 63
        and raw_gold_retained_ok
    )
    # 14.1 Direct pass: B ≥ 18/22 AND Combined ≥ 60/80
    direct_pass = (
        b_class_recovered >= 18
        and combined_hits >= 60
        and raw_gold_retained_ok
    )
    # 14.3 Effective but insufficient: B 8–17/22 AND Combined 50–59/80
    effective_but_insufficient = (
        8 <= b_class_recovered <= 17
        and 50 <= combined_hits <= 59
    )
    # 14.4 Insufficient: B < 8/22 OR Combined < 50/80
    insufficient = (
        b_class_recovered < 8 or combined_hits < 50
    )

    if strong_pass:
        decision = "candidate_aligned_direct_retrieval_strong_pass"
        next_gate_name = "slot_normalized_evidence_pool"
        gate_passed = True
    elif direct_pass:
        decision = "candidate_aligned_direct_retrieval_passed"
        next_gate_name = "slot_normalized_evidence_pool"
        gate_passed = True
    elif effective_but_insufficient:
        decision = "candidate_aligned_direct_gain_real_but_insufficient"
        next_gate_name = "candidate_retrieval_failure_slice_repair"
        gate_passed = False
    else:
        decision = "candidate_aligned_direct_retrieval_insufficient"
        next_gate_name = "gate_08_r1_2_independent_structural_presence_audit"
        gate_passed = False

    acceptance = {
        "gate": "pdf_retrieval_v4_gate_08_r2",
        "prediction_seal_verified": True,
        "prediction_rerun": False,
        "prediction_count": len(r2_predictions),
        "gold_source_count": total_gold,
        "gold_reads_before_seal": int(
            r2_seal.get("gold_reads_before_seal", -1)
        ),
        "governance_reads_before_seal": int(
            r2_seal.get("governance_reads_before_seal", -1)
        ),
        "reranker_calls": 0,
        "calculator_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "parameter_scan": False,
        "per_query_oracle": False,
        "raw_full_pool_recall": scoring_report["raw_full_pool_recall"],
        "existing_structured_recall": scoring_report["existing_structured_recall"],
        "candidate_direct_recall_b_class": scoring_report[
            "candidate_direct_recall_b_class"
        ],
        "combined_strict_recall": scoring_report["combined_strict_recall"],
        "combined_strict_recall_a_plus_b": scoring_report[
            "combined_strict_recall_a_plus_b"
        ],
        "multi_evidence_complete": scoring_report["multi_evidence"][
            "complete_evidence_availability"
        ],
        "b_class_recovered": b_class_recovered,
        "b_class_total": b_class_total,
        "b_class_still_missing": b_class_total - b_class_recovered,
        "raw_gold_retained": raw_hits,
        "raw_gold_retained_ok": raw_gold_retained_ok,
        "gate_criteria": {
            "strong_pass": strong_pass,
            "direct_pass": direct_pass,
            "effective_but_insufficient": effective_but_insufficient,
            "insufficient": insufficient,
        },
        "gate_passed": gate_passed,
        "decision": decision,
        "next_gate": next_gate_name,
        "production_switch_allowed": False,
    }

    next_gate = {
        "current_gate": "pdf_retrieval_v4_gate_08_r2",
        "decision": acceptance["decision"],
        "gate_passed": gate_passed,
        "next_gate": acceptance["next_gate"],
        "b_class_recovered": b_class_recovered,
        "b_class_total": b_class_total,
        "b_class_still_missing": b_class_total - b_class_recovered,
        "production_switch_allowed": False,
    }

    # ------------------------------------------------------------------
    # 5. Write outputs
    # ------------------------------------------------------------------
    write(args.out_dir / "scoring-report.json", scoring_report)
    write(args.out_dir / "b-class-detail.json", b_class_detail)
    write(args.out_dir / "rank-distribution.json", rank_distribution)
    write(args.out_dir / "acceptance.json", acceptance)
    write(args.out_dir / "next-gate.json", next_gate)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("Gate 08 R2 scoring complete.")
    print(f"  Total Gold:                  {total_gold}")
    print(f"  Raw Full Pool Recall:        {scoring_report['raw_full_pool_recall']}")
    print(f"  Existing Structured Recall:  {scoring_report['existing_structured_recall']}")
    print(f"  Candidate Direct Recall (B): {scoring_report['candidate_direct_recall_b_class']}")
    print(f"  Combined Strict Recall:      {scoring_report['combined_strict_recall']}")
    print(f"  Combined Recall (A+B):       {scoring_report['combined_strict_recall_a_plus_b']}")
    print(f"  B-class total:               {b_class_total}")
    print(f"  B-class recovered:           {b_class_recovered}")
    print(f"  B-class still missing:       {b_class_total - b_class_recovered}")
    print(f"  B-class rank distribution:   {dict(b_class_rank_distribution)}")
    print(f"  Raw Gold Retained:           {raw_hits}/31")
    print(f"  Gate passed:                 {gate_passed}")
    print(f"  Decision:                    {acceptance['decision']}")
    print(f"  Next gate:                   {acceptance['next_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
