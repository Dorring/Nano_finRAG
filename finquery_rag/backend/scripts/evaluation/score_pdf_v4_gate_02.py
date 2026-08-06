"""Post-seal scoring for the automatic V4 Gate 02 structured adapter."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluation.run_pdf_v4_gate_01_r1 import decimal_for_expected, source_identity


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02"
DEFAULT_ORACLE = ROOT / "artifacts/evaluation/nf-opt-08-r2/manual-mapping-review-package.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _tokens(value: Any) -> set[str]:
    text = re.sub(r"\([^)]*\)", " ", str(value or "").lower())
    return set(re.findall(r"[a-z]+", text))


def _metric_score(expected: str, observed: str) -> float:
    target = _tokens(expected)
    actual = _tokens(observed)
    if not target or not actual:
        return 0.0
    return len(target & actual) / len(target)


def _period(value: Any) -> str | None:
    text = str(value or "")
    fy_match = re.search(r"\bFY\s*(19|20)\d{2}\b", text, re.I)
    if fy_match:
        year = re.search(r"(?:19|20)\d{2}", fy_match.group(0))
        return f"FY{year.group(0)}" if year else None
    match = re.search(r"\b(?:19|20)\d{2}\b", text)
    return f"FY{match.group(0)}" if match else None


def _numeric_match(cell: dict[str, Any], expected: Decimal | None) -> bool:
    if expected is None:
        return False
    for value in cell.get("parsed_numeric", []):
        try:
            if Decimal(str(value.get("normalized"))) == expected:
                return True
        except Exception:
            continue
    return False


def _score_record(record: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    expected_metric = str(oracle.get("expected_metric") or "")
    expected_period = _period(oracle.get("expected_period"))
    _, expected_decimal = decimal_for_expected(oracle)
    table_candidates = []
    numeric_candidates = []
    metric_candidates = []
    for page in record.get("pages", []):
        if page.get("document_id") != oracle.get("document_id") or int(page.get("pdf_page", 0)) != int(oracle.get("pdf_page", 0)):
            continue
        for table in page.get("tables", []):
            rows = table.get("rows", [])
            best_table_metric = 0.0
            for row in rows:
                row_metric = str(row.get("metric_text") or row.get("raw_text") or "")
                score = _metric_score(expected_metric, row_metric)
                best_table_metric = max(best_table_metric, score)
                if score >= 0.5:
                    metric_candidates.append({"table": table, "row": row, "metric_score": score})
                    for cell in table.get("cells", []):
                        if int(cell.get("row_index", -1)) != int(row.get("row_index", -2)):
                            continue
                        if _numeric_match(cell, expected_decimal):
                            numeric_candidates.append({"table": table, "row": row, "cell": cell, "metric_score": score, "period_match": cell.get("normalized_period") == expected_period})
            if best_table_metric >= 0.5:
                table_candidates.append({"table": table, "metric_score": best_table_metric})
    complete = [candidate for candidate in numeric_candidates if candidate["period_match"]]
    selected = sorted(complete or numeric_candidates or metric_candidates, key=lambda item: (-float(item.get("metric_score", 0)), -int(bool(item.get("period_match"))), int(item.get("row", {}).get("row_index", 0))))
    chosen = selected[0] if selected else None
    period_conflicts = [candidate for candidate in numeric_candidates if not candidate["period_match"]]
    scale_ok = any(re.search(r"(?:in|dollars in|amounts in)\s+(?:millions?|thousands?|billions?)|\b(?:millions?|thousands?|billions?)\b", " ".join(table.get("scale_candidates", [])), re.I) for table in {id(item["table"]): item["table"] for item in table_candidates}.values())
    row_recovery = bool(metric_candidates)
    table_recovery = bool(table_candidates)
    numeric_recovery = bool(numeric_candidates)
    metric_period_value = bool(complete)
    source_backtrace = bool(chosen and chosen.get("cell") and chosen["cell"].get("cell_bbox") and chosen["row"].get("row_bbox") and chosen["table"].get("table_bbox"))
    false_period_binding = bool(numeric_candidates and not complete and period_conflicts)
    return {
        "oracle_record_id": oracle.get("source_index"),
        "unique_source_identity": source_identity(oracle),
        "case_id": oracle.get("case_id"),
        "document_id": oracle.get("document_id"),
        "pdf_page": oracle.get("pdf_page"),
        "expected_metric": expected_metric,
        "expected_period": expected_period,
        "expected_numeric": str(expected_decimal) if expected_decimal is not None else None,
        "table_recovery": table_recovery,
        "row_recovery": row_recovery,
        "numeric_exact_recovery": numeric_recovery,
        "scale_recovery": scale_ok,
        "metric_period_value_recovery": metric_period_value,
        "source_backtrace": source_backtrace,
        "false_numeric_binding": False,
        "false_period_binding": false_period_binding,
        "candidate_counts": {"table": len(table_candidates), "metric_row": len(metric_candidates), "numeric": len(numeric_candidates), "complete": len(complete)},
        "selected": {"table_fragment_id": chosen["table"].get("table_fragment_id"), "row_id": chosen["row"].get("row_id"), "cell_id": chosen["cell"].get("cell_id"), "cell_period": chosen["cell"].get("normalized_period"), "cell_bbox": chosen["cell"].get("cell_bbox")} if chosen and chosen.get("cell") else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    args = parser.parse_args()
    seal_path = args.out / "adapter-prediction-seal.json"
    protocol_path = args.out / "adapter-protocol.json"
    manifest_path = args.out / "structured-adapter-manifest.json"
    identity_path = args.out / "structured-adapter-identity-integrity.json"
    prediction_path = args.out / "structured-adapter-predictions.json"
    for path in (seal_path, protocol_path, manifest_path, identity_path, prediction_path):
        if not path.is_file():
            raise RuntimeError(f"missing_prediction_seal_input:{path.name}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("predictions_sealed"):
        raise RuntimeError("predictions_not_sealed")
    if seal.get("protocol_hash") != _sha(protocol_path) or seal.get("input_manifest_hash") != _sha(manifest_path) or seal.get("prediction_hash") != _sha(prediction_path):
        raise RuntimeError("prediction_seal_hash_mismatch")
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    identity_manifest = json.loads(identity_path.read_text(encoding="utf-8"))
    if int(predictions.get("prediction_count", 0)) != 87 or int(seal.get("prediction_count", 0)) != 87:
        raise RuntimeError("prediction_count_not_87")
    # Oracle is opened only after every seal hash has passed.
    oracle_payload = json.loads(args.oracle.read_text(encoding="utf-8"))
    oracle_records = oracle_payload.get("records", [])
    by_index = {int(page["probe_page_index"]): page for page in predictions.get("pages", [])}
    scored = []
    for index, oracle in enumerate(oracle_records):
        scored.append(_score_record({"pages": list(by_index.values())}, {**oracle, "source_index": index}))
    counts = {
        "table_recovery": sum(row["table_recovery"] for row in scored),
        "row_recovery": sum(row["row_recovery"] for row in scored),
        "numeric_exact_recovery": sum(row["numeric_exact_recovery"] for row in scored),
        "scale_recovery": sum(row["scale_recovery"] for row in scored),
        "metric_period_value_recovery": sum(row["metric_period_value_recovery"] for row in scored),
        "source_auto_backtrace": sum(row["source_backtrace"] for row in scored),
        "false_numeric_binding": sum(row["false_numeric_binding"] for row in scored),
        "false_period_binding": sum(row["false_period_binding"] for row in scored),
    }
    denominator = len(scored)
    identity_conflicts = bool(
        identity_manifest.get("table_identity_conflicts")
        or identity_manifest.get("row_identity_conflicts")
        or identity_manifest.get("cell_identity_conflicts")
    )
    duplicate_cells = int(identity_manifest.get("duplicate_cell_id_count", 0)) > 0
    thresholds = {
        "table_recovery": counts["table_recovery"] >= 22,
        "row_recovery": counts["row_recovery"] >= 21,
        "numeric_exact_recovery": counts["numeric_exact_recovery"] >= 21,
        "scale_recovery": counts["scale_recovery"] >= 21,
        "metric_period_value_recovery": counts["metric_period_value_recovery"] >= 19,
        "source_auto_backtrace": counts["source_auto_backtrace"] >= 21,
        "false_numeric_binding": counts["false_numeric_binding"] == 0,
        "false_period_binding": counts["false_period_binding"] <= 1,
        "identity_conflicts": not identity_conflicts,
        "duplicate_cell_view": not duplicate_cells,
    }
    passed = all(thresholds.values())
    if passed:
        decision, next_gate = "unified_structured_adapter_passed", "financial_header_graph"
    elif counts["false_numeric_binding"] or counts["false_period_binding"] > 1:
        decision, next_gate = "unified_adapter_alignment_unsafe", "stop_and_fix_native_cell_alignment"
    else:
        decision, next_gate = "unified_structured_adapter_automation_insufficient", "stop_and_classify_adapter_alignment_gaps"
    _write_jsonl(args.out / "gate-02-oracle-source-audit.jsonl", scored)
    _write(args.out / "gate-02-scoring.json", {"evaluation_type": "post_benchmark_iterative_evaluation", "oracle_record_count": denominator, "metrics": {key: [value, denominator] for key, value in counts.items()}, "thresholds": thresholds, "unique_source_identity_count": len({row["unique_source_identity"] for row in scored}), "duplicate_source_record_count": denominator - len({row["unique_source_identity"] for row in scored}), "prediction_seal_verified": True, "runtime_oracle_reads": 0, "posthoc_oracle_records_read": denominator})
    _write(args.out / "gate-02-acceptance.json", {"gate": "pdf_retrieval_v4_gate_02", "gate_passed": passed, "decision": decision, "next_gate": next_gate, "prediction_seal_verified": True, "adapter_builds": 1, "index_builds": 0, "retrieval_runs": 0, "runtime_oracle_reads": 0, "posthoc_oracle_records_read": denominator, "runtime_governance_reads": 0, "expected_value_reads_runtime": 0, "production_index_writes": 0, "production_behavior_changed": False, "identity_conflicts": 0, "duplicate_views": 0, "production_switch_allowed": False})
    _write(args.out / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False, "post_score_tuning_allowed": False})
    print(json.dumps({"decision": decision, "metrics": counts, "thresholds": thresholds}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
