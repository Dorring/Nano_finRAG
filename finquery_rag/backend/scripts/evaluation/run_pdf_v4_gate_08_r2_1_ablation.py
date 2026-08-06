"""Gate 08 R2.1: Lane Contribution and Coverage Audit.

Offline ablation using sealed Gate 08 R2 lane hits.  No re-encoding,
no re-indexing, no re-retrieval.  Re-combines existing lane hits to
answer:

  - Do the 5 newly recovered Gold come from Raw or Structured lanes?
  - Do the 17 unrecovered Gold have Structured Views?
  - Is the problem Structured View coverage or Structured retrieval?

Four ablation groups:
  E0: raw_production + existing_structured             = 42/80 (baseline)
  E1: E0 + candidate_raw (bm25 + dense RRF Top-40)     = ?
  E2: E0 + candidate_structured (bm25 + dense RRF Top-40) = ?
  E3: E0 + candidate_raw + candidate_structured        = 47/80 (R2 result)

Seal verification: R2 prediction seal must be valid (sealed=true,
gold_reads=0).  Gold labels are read ONLY after R2 seal verification.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.lane_ablation import (  # noqa: E402
    LANE_K,
    POOL_K,
    RRF_K,
    RAW_LANES,
    STRUCTURED_LANES,
    build_combined_pool_keys,
    build_e0_pool,
    build_raw_pool_keys,
    classify_lane_support,
    find_rank_in_lane,
    find_rrf_rank,
    rrf_fuse,
)

DEFAULT_R2_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2"
DEFAULT_GATE08_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2-1"
DEFAULT_R11_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r1-1"
DATA = ROOT / "benchmarks/financial_rag_v1/data"


# ------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidate_metadata(index_dir: Path) -> dict[str, dict[str, Any]]:
    """Load bridge_grade for each candidate_key from metadata DB."""
    db_path = index_dir / "candidate-metadata.sqlite"
    uri = f"file:{db_path.absolute().as_posix()}?mode=ro"
    metadata: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            "SELECT candidate_key, metadata_json FROM view_metadata"
        ).fetchall()
    for candidate_key, metadata_json in rows:
        if candidate_key not in metadata:
            data = json.loads(metadata_json or "{}")
            metadata[candidate_key] = {
                "bridge_grade": data.get("bridge_grade", "raw_only"),
                "has_structured_mapping": data.get(
                    "has_structured_mapping", False
                ),
            }
    return metadata


# ------------------------------------------------------------------
# Main ablation
# ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-out", type=Path, default=DEFAULT_R2_OUT)
    parser.add_argument("--gate08-out", type=Path, default=DEFAULT_GATE08_OUT)
    parser.add_argument("--r1-1-out", type=Path, default=DEFAULT_R11_OUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--labels", type=Path, default=DATA / "labels.golden.jsonl")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Verify R2 seal (must be sealed, gold_reads=0)
    # ------------------------------------------------------------------
    r2_seal = load_json(args.r2_out / "prediction-seal.json")
    if (
        not r2_seal.get("sealed")
        or r2_seal.get("gold_reads_before_seal") != 0
        or r2_seal.get("governance_reads_before_seal") != 0
    ):
        raise RuntimeError("r2_seal_invalid")

    # ------------------------------------------------------------------
    # 2. Load sealed data (no Gold read yet)
    # ------------------------------------------------------------------
    r2_predictions_list = load_predictions(
        args.r2_out / "predictions.jsonl.gz"
    )
    r2_predictions = {
        str(p["case_id"]): p for p in r2_predictions_list
    }

    b_class_detail = load_json(
        args.r2_out / "scoring" / "b-class-detail.json"
    )
    failure_attr = load_json(
        args.r2_out / "failure-classification" / "failure-attribution.json"
    )

    raw_parity = load_json(args.gate08_out / "raw-parity.json")
    raw_by_case = {
        str(c["case_id"]): c for c in raw_parity["raw_cases"]
    }

    gate08_preds_list = load_predictions(
        args.gate08_out / "retrieval-predictions.jsonl.gz"
    )
    gate08_preds = {
        str(p["case_id"]): p for p in gate08_preds_list
    }

    candidate_metadata = load_candidate_metadata(
        args.r2_out / "candidate-indexes"
    )

    # R1.1 coverage classification (for D-class list)
    r11_coverage_path = args.r1_1_out / "coverage-classification.json"
    r11_coverage = (
        load_json(r11_coverage_path) if r11_coverage_path.is_file() else {}
    )

    # ------------------------------------------------------------------
    # 3. Write protocol (before any Gold read)
    # ------------------------------------------------------------------
    protocol = {
        "gate": "pdf_retrieval_v4_gate_08_r2_1",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "purpose": "lane_contribution_and_coverage_audit",
        "prediction_rerun": False,
        "re_encoding": False,
        "re_indexing": False,
        "re_retrieval": False,
        "r2_seal_verified": True,
        "r2_gold_reads_before_seal": int(
            r2_seal.get("gold_reads_before_seal", -1)
        ),
        "r2_governance_reads_before_seal": int(
            r2_seal.get("governance_reads_before_seal", -1)
        ),
        "r2_prediction_hash": r2_seal.get("prediction_hash"),
        "inputs": [
            "sealed_r2_predictions",
            "sealed_r2_prediction_seal",
            "r2_b_class_detail",
            "r2_failure_attribution",
            "r2_scoring_report",
            "r2_candidate_metadata",
            "gate_08_raw_parity",
            "gate_08_sealed_predictions",
            "r1_1_coverage_classification",
            "labels_golden",
        ],
        "forbidden_inputs": [
            "expected_value",
            "reference_answer",
            "gate_1_governance_fields",
        ],
        "ablation_groups": {
            "E0": "raw_production + existing_structured",
            "E1": "E0 + candidate_raw (bm25+dense RRF Top-40)",
            "E2": "E0 + candidate_structured (bm25+dense RRF Top-40)",
            "E3": "E0 + candidate_raw + candidate_structured (R2 combined)",
        },
        "rrf_k": RRF_K,
        "lane_k": LANE_K,
        "pool_k": POOL_K,
        "raw_lanes": list(RAW_LANES),
        "structured_lanes": list(STRUCTURED_LANES),
    }
    write(args.out_dir / "lane-ablation-protocol.json", protocol)

    # ------------------------------------------------------------------
    # 4. Gold read begins here (after seal verification + protocol write)
    # ------------------------------------------------------------------
    labels_list = load_jsonl(args.labels)
    labels = {str(item["case_id"]): item for item in labels_list}

    if set(r2_predictions) != set(labels):
        raise RuntimeError("prediction_label_case_set_mismatch")

    # ------------------------------------------------------------------
    # 5. Per-Gold lane contribution analysis (B-class 22)
    # ------------------------------------------------------------------
    b_class_rows = b_class_detail.get("rows", [])
    failure_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in failure_attr.get("rows", []):
        failure_by_key[
            (str(row["case_id"]), str(row["gold_candidate_key"]))
        ] = row

    lane_contribution: list[dict[str, Any]] = []
    recovered_support: list[dict[str, Any]] = []
    unrecovered_coverage: list[dict[str, Any]] = []

    for gold in b_class_rows:
        case_id = str(gold["case_id"])
        gold_key = str(gold["gold_candidate_key"])
        recovered = bool(gold.get("recovered"))

        pred = r2_predictions[case_id]
        lane_hits = pred.get("lane_hits", {})
        rrf_hits = pred.get("rrf_hits", [])

        # Lane ranks
        raw_bm25_rank = find_rank_in_lane(
            lane_hits.get("candidate_raw_bm25", []), gold_key
        )
        raw_dense_rank = find_rank_in_lane(
            lane_hits.get("candidate_raw_dense", []), gold_key
        )
        structured_bm25_rank = find_rank_in_lane(
            lane_hits.get("candidate_structured_bm25", []), gold_key
        )
        structured_dense_rank = find_rank_in_lane(
            lane_hits.get("candidate_structured_dense", []), gold_key
        )
        candidate_rrf_rank = find_rrf_rank(rrf_hits, gold_key)

        # Bridge grade from candidate metadata
        meta = candidate_metadata.get(gold_key, {})
        bridge_grade = str(meta.get("bridge_grade", "raw_only"))
        has_structured_view = bridge_grade != "raw_only"

        # Recovery source classification
        lane_support = classify_lane_support(
            recovered=recovered,
            raw_bm25_rank=raw_bm25_rank,
            raw_dense_rank=raw_dense_rank,
            structured_bm25_rank=structured_bm25_rank,
            structured_dense_rank=structured_dense_rank,
        )
        recovered_by_raw_lane = lane_support["recovered_by_raw_lane"]
        recovered_by_structured_lane = lane_support[
            "recovered_by_structured_lane"
        ]
        recovered_by_fusion_only = lane_support["recovered_by_fusion_only"]

        # Failure info
        fail = failure_by_key.get((case_id, gold_key), {})
        first_failure_stage = str(
            fail.get("first_failure_stage", "")
        )
        in_top50 = bool(fail.get("in_top50", False))
        in_top40 = bool(fail.get("in_top40", False))
        best_rank = fail.get("best_rank")

        record = {
            "gold_candidate_key": gold_key,
            "case_id": case_id,
            "gold_source_identity": gold.get("gold_source_identity"),
            "recovered": recovered,
            "has_raw_view": True,
            "has_structured_view": has_structured_view,
            "bridge_grade": bridge_grade,
            "raw_bm25_rank": raw_bm25_rank,
            "raw_dense_rank": raw_dense_rank,
            "structured_bm25_rank": structured_bm25_rank,
            "structured_dense_rank": structured_dense_rank,
            "candidate_rrf_rank": candidate_rrf_rank,
            "recovered_by_raw_lane": recovered_by_raw_lane,
            "recovered_by_structured_lane": recovered_by_structured_lane,
            "recovered_by_fusion_only": recovered_by_fusion_only,
            "first_failure_stage": first_failure_stage,
            "in_top50": in_top50,
            "in_top40": in_top40,
            "best_rank": best_rank,
        }
        lane_contribution.append(record)

        if recovered:
            recovered_support.append(record)
        else:
            unrecovered_coverage.append(record)

    write(
        args.out_dir / "lane-contribution-by-gold.json",
        {
            "gate": "pdf_retrieval_v4_gate_08_r2_1",
            "b_class_total": len(b_class_rows),
            "records": lane_contribution,
        },
    )

    # ------------------------------------------------------------------
    # 6. Recovered gold support breakdown
    # ------------------------------------------------------------------
    raw_only_support = sum(
        1 for r in recovered_support if r["recovered_by_raw_lane"]
        and not r["recovered_by_structured_lane"]
    )
    structured_only_support = sum(
        1 for r in recovered_support if r["recovered_by_structured_lane"]
        and not r["recovered_by_raw_lane"]
    )
    both_support = sum(
        1 for r in recovered_support
        if r["recovered_by_raw_lane"]
        and r["recovered_by_structured_lane"]
    )
    fusion_only_support = sum(
        1 for r in recovered_support if r["recovered_by_fusion_only"]
    )

    write(
        args.out_dir / "recovered-gold-support.json",
        {
            "gate": "pdf_retrieval_v4_gate_08_r2_1",
            "recovered_count": len(recovered_support),
            "support_breakdown": {
                "raw_only": raw_only_support,
                "structured_only": structured_only_support,
                "raw_and_structured": both_support,
                "fusion_only": fusion_only_support,
            },
            "records": recovered_support,
        },
    )

    # ------------------------------------------------------------------
    # 7. Unrecovered structured coverage
    # ------------------------------------------------------------------
    unrecovered_with_structured = sum(
        1 for r in unrecovered_coverage if r["has_structured_view"]
    )
    unrecovered_raw_only = sum(
        1 for r in unrecovered_coverage if not r["has_structured_view"]
    )
    unrecovered_in_structured_top50 = sum(
        1 for r in unrecovered_coverage
        if r["structured_bm25_rank"] is not None
        or r["structured_dense_rank"] is not None
    )
    unrecovered_structured_lane_missed = sum(
        1 for r in unrecovered_coverage
        if r["has_structured_view"]
        and r["structured_bm25_rank"] is None
        and r["structured_dense_rank"] is None
    )

    write(
        args.out_dir / "unrecovered-structured-coverage.json",
        {
            "gate": "pdf_retrieval_v4_gate_08_r2_1",
            "unrecovered_count": len(unrecovered_coverage),
            "structured_view_exists": unrecovered_with_structured,
            "raw_only": unrecovered_raw_only,
            "structured_lane_top50": unrecovered_in_structured_top50,
            "structured_lane_completely_missed": unrecovered_structured_lane_missed,
            "records": unrecovered_coverage,
        },
    )

    # ------------------------------------------------------------------
    # 8. Structured view coverage metrics (global)
    # ------------------------------------------------------------------
    total_b_class = len(b_class_rows)
    b_class_with_structured = sum(
        1 for r in lane_contribution if r["has_structured_view"]
    )
    b_class_raw_only = total_b_class - b_class_with_structured

    # D-class from R1.1
    d_class_rows = []
    if r11_coverage:
        for row in r11_coverage.get("rows", []):
            if str(row.get("coverage_class")) == "structurally_absent":
                d_class_rows.append(row)
    d_class_total = len(d_class_rows)

    write(
        args.out_dir / "structured-view-coverage-metrics.json",
        {
            "gate": "pdf_retrieval_v4_gate_08_r2_1",
            "global_candidate_universe": {
                "total_candidates": 38319,
                "with_structured_view": 628,
                "raw_only": 37691,
                "structured_coverage_pct": round(
                    628 / 38319 * 100, 2
                ),
            },
            "b_class_coverage": {
                "total": total_b_class,
                "with_structured_view": b_class_with_structured,
                "raw_only": b_class_raw_only,
            },
            "d_class_count": d_class_total,
            "bridge_grade_distribution": dict(
                Counter(r["bridge_grade"] for r in lane_contribution)
            ),
        },
    )

    # ------------------------------------------------------------------
    # 9. Pool truncation audit
    # ------------------------------------------------------------------
    truncation_cases: list[dict[str, Any]] = []
    for r in lane_contribution:
        if not r["recovered"]:
            stage = r["first_failure_stage"]
            if stage in (
                "candidate_rank_41_to_50",
                "multi_slot_budget_truncated",
            ):
                truncation_cases.append(r)

    write(
        args.out_dir / "pool-truncation-audit.json",
        {
            "gate": "pdf_retrieval_v4_gate_08_r2_1",
            "truncation_count": len(truncation_cases),
            "stages": dict(
                Counter(
                    r["first_failure_stage"]
                    for r in truncation_cases
                )
            ),
            "records": truncation_cases,
        },
    )

    # ------------------------------------------------------------------
    # 10. Ablation recall computation (all 80 Gold)
    # ------------------------------------------------------------------
    e0_hits = 0
    e1_hits = 0
    e2_hits = 0
    e3_hits = 0
    raw_gold_retained = 0
    total_gold = 0

    b_class_keys = {
        (r["case_id"], r["gold_candidate_key"]) for r in lane_contribution
    }
    e0_b = e1_b = e2_b = e3_b = 0

    for case_id in sorted(labels):
        label = labels[case_id]
        sources = [
            s for s in label.get("expected_sources", []) if s.get("candidate_key")
        ]
        pred = r2_predictions[case_id]
        lane_hits = pred.get("lane_hits", {})

        raw_case = raw_by_case.get(case_id, {})
        gate08_pred = gate08_preds.get(case_id, {})

        # E0: raw + structured
        e0_pool = build_e0_pool(raw_case, gate08_pred)

        # E1: E0 + raw-only RRF Top-40
        raw_rrf = rrf_fuse(lane_hits, RAW_LANES)
        e1_pool = e0_pool | set(raw_rrf)

        # E2: E0 + structured-only RRF Top-40
        structured_rrf = rrf_fuse(lane_hits, STRUCTURED_LANES)
        e2_pool = e0_pool | set(structured_rrf)

        # E3: R2 combined_pool (full)
        e3_pool = build_combined_pool_keys(pred)

        raw_pool_keys = build_raw_pool_keys(raw_case)

        for source in sources:
            gold_key = str(source.get("candidate_key"))
            total_gold += 1

            if gold_key in raw_pool_keys:
                raw_gold_retained += 1

            in_e0 = gold_key in e0_pool
            in_e1 = gold_key in e1_pool
            in_e2 = gold_key in e2_pool
            in_e3 = gold_key in e3_pool

            if in_e0:
                e0_hits += 1
            if in_e1:
                e1_hits += 1
            if in_e2:
                e2_hits += 1
            if in_e3:
                e3_hits += 1

            if (case_id, gold_key) in b_class_keys:
                if in_e0:
                    e0_b += 1
                if in_e1:
                    e1_b += 1
                if in_e2:
                    e2_b += 1
                if in_e3:
                    e3_b += 1

    # ------------------------------------------------------------------
    # 11. Acceptance and next-gate
    # ------------------------------------------------------------------
    # Determine primary bottleneck per spec section 3.5
    if unrecovered_raw_only > unrecovered_with_structured:
        primary_bottleneck = "structured_view_coverage_insufficient"
        next_gate = "gate_08_r1_2_independent_structural_presence_audit"
    elif unrecovered_structured_lane_missed > 0:
        primary_bottleneck = "structured_retrieval_text_or_dense_mismatch"
        next_gate = "gate_08_r3_field_aware_candidate_retrieval"
    elif len(truncation_cases) > 0:
        primary_bottleneck = "pool_and_slot_budget_contract"
        next_gate = "gate_08_r4_slot_preserving_pool_construction"
    else:
        primary_bottleneck = "mixed"
        next_gate = "gate_08_r1_2_independent_structural_presence_audit"

    acceptance = {
        "gate": "pdf_retrieval_v4_gate_08_r2_1",
        "r2_seal_verified": True,
        "ablation_recall": {
            "E0_raw_plus_structured": f"{e0_hits}/{total_gold}",
            "E1_E0_plus_candidate_raw": f"{e1_hits}/{total_gold}",
            "E2_E0_plus_candidate_structured": f"{e2_hits}/{total_gold}",
            "E3_E0_plus_full_candidate": f"{e3_hits}/{total_gold}",
        },
        "b_class_ablation_recall": {
            "E0": f"{e0_b}/{total_b_class}",
            "E1_raw_only": f"{e1_b}/{total_b_class}",
            "E2_structured_only": f"{e2_b}/{total_b_class}",
            "E3_full": f"{e3_b}/{total_b_class}",
        },
        "raw_gold_retained": f"{raw_gold_retained}/31",
        "recovered_gold_support": {
            "raw_only": raw_only_support,
            "structured_only": structured_only_support,
            "raw_and_structured": both_support,
            "fusion_only": fusion_only_support,
        },
        "unrecovered_analysis": {
            "structured_view_exists": unrecovered_with_structured,
            "raw_only": unrecovered_raw_only,
            "structured_lane_top50": unrecovered_in_structured_top50,
            "structured_lane_completely_missed": unrecovered_structured_lane_missed,
        },
        "pool_truncation_count": len(truncation_cases),
        "primary_bottleneck": primary_bottleneck,
        "decision": "lane_contribution_audit_complete",
        "next_gate": next_gate,
        "production_switch_allowed": False,
    }
    write(args.out_dir / "acceptance.json", acceptance)

    write(
        args.out_dir / "next-gate.json",
        {
            "current_gate": "pdf_retrieval_v4_gate_08_r2_1",
            "decision": "lane_contribution_audit_complete",
            "primary_bottleneck": primary_bottleneck,
            "next_gate": next_gate,
            "ablation_summary": {
                "E0": e0_hits,
                "E1": e1_hits,
                "E2": e2_hits,
                "E3": e3_hits,
                "total_gold": total_gold,
            },
            "b_class_summary": {
                "E0": e0_b,
                "E1": e1_b,
                "E2": e2_b,
                "E3": e3_b,
                "total": total_b_class,
            },
            "production_switch_allowed": False,
        },
    )

    # ------------------------------------------------------------------
    # 12. Print summary
    # ------------------------------------------------------------------
    print("Gate 08 R2.1 lane contribution audit complete.")
    print(f"  Total Gold:                {total_gold}")
    print(f"  E0 (raw+structured):       {e0_hits}/{total_gold}")
    print(f"  E1 (E0+candidate_raw):     {e1_hits}/{total_gold}")
    print(f"  E2 (E0+candidate_struct):  {e2_hits}/{total_gold}")
    print(f"  E3 (E0+full candidate):    {e3_hits}/{total_gold}")
    print(f"  Raw Gold Retained:         {raw_gold_retained}/31")
    print(f"  B-class total:             {total_b_class}")
    print(f"  B-class E1 (raw only):     {e1_b}/{total_b_class}")
    print(f"  B-class E2 (struct only):  {e2_b}/{total_b_class}")
    print(f"  B-class E3 (full):         {e3_b}/{total_b_class}")
    print("  Recovered support:")
    print(f"    raw_only:                {raw_only_support}")
    print(f"    structured_only:         {structured_only_support}")
    print(f"    raw+structured:          {both_support}")
    print(f"    fusion_only:             {fusion_only_support}")
    print(f"  Unrecovered ({len(unrecovered_coverage)}):")
    print(f"    structured_view_exists:  {unrecovered_with_structured}")
    print(f"    raw_only:                {unrecovered_raw_only}")
    print(f"    structured_lane_top50:   {unrecovered_in_structured_top50}")
    print(f"    structured_lane_missed:  {unrecovered_structured_lane_missed}")
    print(f"  Pool truncation:           {len(truncation_cases)}")
    print(f"  Primary bottleneck:        {primary_bottleneck}")
    print(f"  Next gate:                 {next_gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
