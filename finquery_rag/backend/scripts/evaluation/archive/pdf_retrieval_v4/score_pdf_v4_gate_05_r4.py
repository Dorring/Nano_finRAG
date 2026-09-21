"""Score V4 Gate 05 R4 only after the temporal-binding prediction seal."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r4"
DEFAULT_R3 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r3"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_stream(path: Path) -> list[dict[str, Any]]:
    result = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if index == 0 and value.get("stream") == "header":
                continue
            result.append(value)
    return result


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _temporal_matches(binding: dict[str, Any] | None, expected_period: str | None) -> bool:
    if not binding or not expected_period:
        return False
    kind = binding.get("kind")
    if kind in {"point", "duration"}:
        return binding.get("period") == expected_period
    if kind == "comparison":
        return expected_period in {binding.get("base_period"), binding.get("current_period")}
    if kind == "bucket":
        return binding.get("reporting_period") == expected_period
    if kind == "period_set":
        return expected_period in set(binding.get("periods", []))
    return False


def _metric_matches(expected: str | None, actual: str | None) -> bool:
    target = _tokens(expected)
    observed = _tokens(actual)
    return bool(target) and target <= observed


def _canonical_metric_identity(value: Any) -> str:
    tokens = []
    for token in re.findall(r"[a-z0-9]+", str(value or "").lower()):
        if token.endswith("ies") and len(token) > 3:
            token = token[:-3] + "y"
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--r3", type=Path, default=DEFAULT_R3)
    args = parser.parse_args()
    out = args.out
    seal = json.loads((out / "temporal-binding-seal.json").read_text(encoding="utf-8"))
    if not seal.get("predictions_sealed"):
        raise RuntimeError("gate_05_r4_prediction_not_sealed")
    if seal.get("protocol_hash") != _sha(out / "gate-05-r4-protocol.json") or seal.get("input_hash") != _sha(out / "input-integrity.json") or seal.get("prediction_hash") != _sha(out / "temporal-binding-predictions.jsonl.gz"):
        raise RuntimeError("gate_05_r4_prediction_seal_invalid")
    records = _load_stream(out / "temporal-binding-predictions.jsonl.gz")
    by_fact = {str(record.get("fact_id")): record for record in records}
    by_cell = {str(record.get("cell_id")): record for record in records}
    r3_scoring = json.loads((args.r3 / "evidence-unit-scoring.json").read_text(encoding="utf-8"))
    oracle_results = json.loads((args.r3 / "oracle-evidence-audit.json").read_text(encoding="utf-8")).get("record_results", [])
    oracle_audit: list[dict[str, Any]] = []
    false_temporal: list[dict[str, Any]] = []
    false_metric: list[dict[str, Any]] = []
    for result in oracle_results:
        selected = result.get("selected") or {}
        fact_id = selected.get("fact_id")
        record = by_fact.get(str(fact_id)) if fact_id else by_cell.get(str(selected.get("cell_id")))
        fact_id = record.get("fact_id") if record else fact_id
        temporal_match = _temporal_matches(record.get("temporal_binding") if record else None, result.get("expected_period"))
        baseline_metric = _canonical_metric_identity(selected.get("normalized_metric_path"))
        r4_metric = _canonical_metric_identity(record.get("normalized_metric") if record else None)
        metric_match = bool(record and baseline_metric and baseline_metric == r4_metric)
        row = {"oracle_record_id": result.get("oracle_record_id"), "unique_source_identity": result.get("unique_source_identity"), "fact_id": fact_id, "expected_period": result.get("expected_period"), "temporal_binding": record.get("temporal_binding") if record else None, "temporal_match": temporal_match, "expected_metric": result.get("expected_metric"), "normalized_metric": record.get("normalized_metric") if record else None, "metric_match": metric_match}
        oracle_audit.append(row)
        if record and not temporal_match:
            false_temporal.append(row)
        if record and not metric_match:
            false_metric.append(row)
    counts = {
        "oracle_record_count": len(oracle_audit),
        "temporal_match_count": sum(row["temporal_match"] for row in oracle_audit),
        "metric_match_count": sum(row["metric_match"] for row in oracle_audit),
        "false_temporal_binding_count": len(false_temporal),
        "false_metric_binding_count": len(false_metric),
        "missing_fact_mapping_count": sum(row["fact_id"] is None or row["fact_id"] not in by_fact for row in oracle_audit),
    }
    gap = json.loads((out / "complete-predicate-gap-audit.json").read_text(encoding="utf-8"))
    gap_ids = {str(row.get("fact_id")) for row in gap.get("records", [])}
    gap_records = [by_fact[fact_id] for fact_id in gap_ids if fact_id in by_fact]
    gap_safe = sum(record.get("admission_status", "").startswith("admitted_") for record in gap_records)
    gap_atomic = sum(record.get("fact_semantic_type") == "atomic_fact" for record in gap_records)
    gap_audit_records = []
    for original in gap.get("records", []):
        record = by_fact.get(str(original.get("fact_id")))
        gap_audit_records.append({**original, "recovery_status": "safe_typed_evidence" if record and record.get("admission_status", "").startswith("admitted_") else "blocked", "recovered_semantic_type": record.get("fact_semantic_type") if record else None, "temporal_binding": record.get("temporal_binding") if record else None})
    gap["safe_typed_recovery_count"] = gap_safe
    gap["atomic_recovery_count"] = gap_atomic
    gap["records"] = gap_audit_records
    _write(out / "complete-predicate-gap-audit.json", gap)
    typed_kinds = {"atomic_fact", "comparison_fact", "bucket_fact", "row_matrix_evidence"}
    eligible_financial = sum(record.get("eligibility_class") != "non_fact_numeric" for record in records)
    typed_admitted = sum(record.get("fact_semantic_type") in typed_kinds for record in records)
    typed_rate = typed_admitted / max(1, eligible_financial)
    atomic = json.loads((out / "atomic-fact-audit.json").read_text(encoding="utf-8"))
    comparison = json.loads((out / "comparison-fact-audit.json").read_text(encoding="utf-8"))
    bucket = json.loads((out / "bucket-fact-audit.json").read_text(encoding="utf-8"))
    row_matrix = json.loads((out / "row-matrix-evidence-audit.json").read_text(encoding="utf-8"))
    r3_metrics = r3_scoring.get("record_metrics", {})
    oracle_regression = {
        "r3_oracle_record_count": len(oracle_results),
        "r3_metrics": r3_metrics,
        "r3_all_required_thresholds": all(values[0] >= 21 for values in r3_metrics.values() if isinstance(values, list) and len(values) == 2),
        "r4_temporal_match_count": counts["temporal_match_count"],
        "r4_metric_match_count": counts["metric_match_count"],
        "r4_false_temporal_binding_count": counts["false_temporal_binding_count"],
        "r4_false_metric_binding_count": counts["false_metric_binding_count"],
        "record_audit": oracle_audit,
    }
    _write(out / "oracle-regression.json", oracle_regression)
    _write(out / "false-binding-audit.json", {"false_temporal_binding_count": len(false_temporal), "false_metric_binding_count": len(false_metric), "missing_fact_mapping_count": counts["missing_fact_mapping_count"], "prediction_stage_oracle_reads": 0, "posthoc_oracle_records_read": len(oracle_audit), "temporal_records": false_temporal, "metric_records": false_metric})
    thresholds = {
        "complete_predicate_gap_classified": gap.get("classified_count") == 47,
        "complete_predicate_gap_safe_recovery": gap_safe >= 40,
        "temporally_typed_admission": typed_rate >= 0.90,
        "atomic_fact_admission": float(atomic.get("admission_rate", 0.0)) >= 0.85,
        "comparison_fact_admission": float(comparison.get("admission_rate", 0.0)) >= 0.90,
        "bucket_fact_admission": float(bucket.get("admission_rate", 0.0)) >= 0.90,
        "false_temporal_binding": len(false_temporal) == 0,
        "false_metric_binding": len(false_metric) == 0,
        "oracle_regression": oracle_regression["r3_all_required_thresholds"] and counts["missing_fact_mapping_count"] == 0,
    }
    passed = all(thresholds.values())
    if passed:
        decision, next_gate = "financial_temporal_binding_graph_passed", "multi_granularity_shadow_index_r2"
    elif not thresholds["complete_predicate_gap_safe_recovery"]:
        decision, next_gate = "complete_predicate_gap_recovery_insufficient", "stop_and_fix_complete_predicate_gap"
    elif not thresholds["temporally_typed_admission"]:
        decision, next_gate = "temporally_typed_admission_insufficient", "stop_and_fix_table_schema"
    elif not thresholds["oracle_regression"] or not thresholds["false_temporal_binding"] or not thresholds["false_metric_binding"]:
        decision, next_gate = "temporal_binding_unsafe", "stop_and_fix_table_schema"
    else:
        decision, next_gate = "temporal_binding_schema_gate_blocked", "stop_and_fix_table_schema"
    scoring = {
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "prediction_seal_verified": True,
        "record_count": len(records),
        "eligible_financial_evidence_count": eligible_financial,
        "typed_admitted_count": typed_admitted,
        "temporally_typed_evidence_admission_rate": typed_rate,
        "atomic_fact": {"count": atomic.get("count"), "eligible": atomic.get("eligible_atomic_candidate_count"), "admission_rate": atomic.get("admission_rate")},
        "comparison_fact": {"count": comparison.get("count"), "eligible": comparison.get("eligible_comparison_candidate_count"), "admission_rate": comparison.get("admission_rate")},
        "bucket_fact": {"count": bucket.get("count"), "eligible": bucket.get("eligible_bucket_candidate_count"), "admission_rate": bucket.get("admission_rate")},
        "row_matrix_evidence": {"count": row_matrix.get("count"), "eligible": row_matrix.get("eligible_row_matrix_candidate_count"), "admission_rate": row_matrix.get("admission_rate")},
        "complete_predicate_gap": {"classified": gap.get("classified_count"), "safe_typed_recovery": gap_safe, "atomic_recovery": gap_atomic},
        "oracle_regression": oracle_regression,
        "thresholds": thresholds,
        "runtime_gold_reads": 0,
        "posthoc_oracle_records_read": len(oracle_audit),
        "question_reads": 0,
        "production_index_writes": 0,
    }
    _write(out / "gate-05-r4-scoring.json", scoring)
    _write(out / "acceptance.json", {
        "gate": "pdf_retrieval_v4_gate_05_r4",
        "gate_passed": passed,
        "decision": decision,
        "next_gate": next_gate,
        "temporally_typed_evidence_admission_rate": typed_rate,
        "atomic_fact_admission_rate": atomic.get("admission_rate"),
        "comparison_fact_admission_rate": comparison.get("admission_rate"),
        "bucket_fact_admission_rate": bucket.get("admission_rate"),
        "row_matrix_evidence_count": row_matrix.get("count"),
        "complete_predicate_gap_safe_recovery": gap_safe,
        "false_temporal_binding_count": len(false_temporal),
        "false_metric_binding_count": len(false_metric),
        "prediction_seal_verified": True,
        "posthoc_oracle_records_read": len(oracle_audit),
        "runtime_gold_reads": 0,
        "question_reads": 0,
        "index_builds": 0,
        "retrieval_runs": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
    })
    _write(out / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    print(json.dumps({"decision": decision, "next_gate": next_gate, "typed_rate": typed_rate, "gap_safe": gap_safe, "false_temporal": len(false_temporal), "false_metric": len(false_metric), "thresholds": thresholds}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
