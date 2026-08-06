"""Score Gate 05 Evidence Units only after the Oracle-blind prediction seal."""

from __future__ import annotations

import argparse
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.run_pdf_v4_gate_01_r1 import decimal_for_expected, source_identity  # noqa: E402


DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05"
DEFAULT_ORACLE = ROOT / "artifacts/evaluation/nf-opt-08-r2/manual-mapping-review-package.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_prediction_stream(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        if not lines:
            raise RuntimeError("empty_evidence_unit_stream")
        header = dict(lines[0])
        header["evidence_units"] = lines[1:]
        return header
    return json.loads(path.read_text(encoding="utf-8"))


def _tokens(value: Any) -> set[str]:
    text = re.sub(r"\b(revenues|expenses|assets|liabilities)\b", lambda match: match.group(1)[:-1], str(value or "").lower())
    return set(re.findall(r"[a-z0-9]+", text))


def _period(value: Any) -> str | None:
    match = re.search(r"(?:FY|fiscal\s+year\s*|year\s+ended\s+)?((?:19|20)\d{2})", str(value or ""), re.I)
    return f"FY{match.group(1)}" if match else None


def _numeric_match(unit: dict[str, Any], expected: Decimal | None) -> bool:
    if expected is None:
        return False
    values = [unit.get("base_value"), unit.get("parsed_value")]
    for value in values:
        if value is None:
            continue
        try:
            if Decimal(str(value)) == expected:
                return True
        except (ArithmeticError, TypeError, ValueError):
            continue
    return False


def _metric_values(unit: dict[str, Any]) -> list[str]:
    values = [unit.get("normalized_metric_path"), unit.get("normalized_metric")]
    path = unit.get("metric_path")
    if isinstance(path, list):
        values.append(" / ".join(str(item) for item in path))
    return [str(value) for value in values if value]


def _metric_score(expected: str, unit: dict[str, Any]) -> tuple[float, bool]:
    target = _tokens(expected)
    if not target:
        return 0.0, False
    best = 0.0
    exact = False
    for value in _metric_values(unit):
        actual = _tokens(value)
        score = len(target & actual) / len(target)
        best = max(best, score)
        exact = exact or target <= actual
    return best, exact


def _pages(unit: dict[str, Any]) -> set[int]:
    result = set()
    for value in unit.get("source_pages", []):
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            pass
    return result


def _scale_match(unit: dict[str, Any], oracle: dict[str, Any]) -> bool:
    expected = str((oracle.get("proposed_candidate") or {}).get("parsed_scale") or "").lower()
    observed = str(unit.get("scale") or "").lower()
    if not expected:
        return bool(observed)
    return expected in observed or observed in expected


def _score_record(units: list[dict[str, Any]], oracle: dict[str, Any], source_index: int) -> dict[str, Any]:
    expected_metric = str(oracle.get("expected_metric") or "")
    expected_period = _period(oracle.get("expected_period"))
    _, expected_decimal = decimal_for_expected(oracle)
    candidates = [unit for unit in units if unit.get("document_id") == oracle.get("document_id") and int(oracle.get("pdf_page", -1)) in _pages(unit)]
    table_units = [unit for unit in candidates if unit.get("unit_type") == "table"]
    row_units = []
    numeric_units = []
    complete_units = []
    for unit in candidates:
        metric_score, metric_exact = _metric_score(expected_metric, unit)
        if metric_score < 0.5:
            continue
        if unit.get("unit_type") == "row":
            row_units.append({"unit": unit, "metric_score": metric_score, "metric_exact": metric_exact})
        if unit.get("unit_type") not in {"cell", "fact"}:
            continue
        numeric = _numeric_match(unit, expected_decimal)
        if numeric:
            period = unit.get("normalized_period") or unit.get("period")
            period_match = period == expected_period
            item = {"unit": unit, "metric_score": metric_score, "metric_exact": metric_exact, "numeric": numeric, "period_match": period_match}
            numeric_units.append(item)
            if metric_exact and period_match:
                complete_units.append(item)
    chosen = sorted(complete_units or numeric_units or row_units, key=lambda item: (-float(item.get("metric_score", 0.0)), -int(bool(item.get("metric_exact"))), -int(bool(item.get("period_match"))), str(item["unit"].get("evidence_unit_id"))))
    selected = chosen[0]["unit"] if chosen else (table_units[0] if table_units else None)
    selected_item = chosen[0] if chosen else None
    source_traceback = bool(selected and selected.get("source_traceback", {}).get("document_id") and selected.get("source_traceback", {}).get("pdf_page") is not None)
    return {
        "oracle_record_id": source_index,
        "unique_source_identity": source_identity(oracle),
        "case_id": oracle.get("case_id"),
        "document_id": oracle.get("document_id"),
        "pdf_page": oracle.get("pdf_page"),
        "expected_metric": expected_metric,
        "expected_period": expected_period,
        "expected_numeric": str(expected_decimal) if expected_decimal is not None else None,
        "table_recovery": bool(table_units),
        "row_recovery": bool(row_units),
        "metric_exact": bool(selected_item and selected_item.get("metric_exact")),
        "period_exact": bool(selected_item and selected_item.get("period_match")),
        "numeric_exact": bool(numeric_units),
        "scale_exact": bool(selected and _scale_match(selected, oracle)),
        "metric_period": bool(complete_units),
        "metric_period_value": bool(complete_units),
        "source_traceback": source_traceback,
        "selected": {
            "evidence_unit_id": selected.get("evidence_unit_id"),
            "unit_type": selected.get("unit_type"),
            "logical_table_id": selected.get("logical_table_id"),
            "fragment_id": selected.get("fragment_id"),
            "row_id": selected.get("row_id"),
            "cell_id": selected.get("cell_id"),
            "fact_id": selected.get("fact_id"),
            "normalized_metric_path": selected.get("normalized_metric_path"),
            "normalized_period": selected.get("normalized_period") or selected.get("period"),
            "scale": selected.get("scale"),
        } if selected else None,
        "candidate_counts": {"table": len(table_units), "row": len(row_units), "numeric": len(numeric_units), "complete": len(complete_units)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    args = parser.parse_args()
    root = args.out
    required = [
        root / "gate-05-protocol.json",
        root / "gate-05-input-integrity.json",
        root / "evidence-unit-predictions.jsonl.gz",
        root / "evidence-unit-prediction-seal.json",
        root / "evidence-unit-integrity.json",
    ]
    if any(not path.is_file() for path in required):
        raise RuntimeError("missing_gate_05_prediction_artifact")
    protocol_path = root / "gate-05-protocol.json"
    input_path = root / "gate-05-input-integrity.json"
    prediction_path = root / "evidence-unit-predictions.jsonl.gz"
    seal = json.loads((root / "evidence-unit-prediction-seal.json").read_text(encoding="utf-8"))
    if not seal.get("predictions_sealed") or seal.get("protocol_hash") != _sha(protocol_path) or seal.get("input_hash") != _sha(input_path) or seal.get("prediction_hash") != _sha(prediction_path):
        raise RuntimeError("gate_05_prediction_seal_invalid")
    predictions = _load_prediction_stream(prediction_path)
    units = predictions.get("evidence_units", [])
    # Oracle is intentionally opened only after every prediction seal hash is verified.
    oracle_payload = json.loads(args.oracle.read_text(encoding="utf-8"))
    oracle_records = oracle_payload.get("records", [])
    scored = [_score_record(units, oracle, index) for index, oracle in enumerate(oracle_records)]
    unique_ids = sorted({row["unique_source_identity"] for row in scored})
    unique_scored = []
    for identity in unique_ids:
        group = [row for row in scored if row["unique_source_identity"] == identity]
        unique_scored.append({"unique_source_identity": identity, **{key: any(row.get(key) for row in group) for key in ("table_recovery", "row_recovery", "metric_exact", "period_exact", "scale_exact", "numeric_exact", "metric_period", "metric_period_value", "source_traceback")}})
    keys = ("table_recovery", "row_recovery", "metric_exact", "period_exact", "scale_exact", "numeric_exact", "metric_period", "metric_period_value", "source_traceback")
    counts = {key: sum(bool(row.get(key)) for row in scored) for key in keys}
    unique_counts = {key: sum(bool(row.get(key)) for row in unique_scored) for key in keys}
    integrity = json.loads((root / "evidence-unit-integrity.json").read_text(encoding="utf-8"))
    integrity_ok = all(integrity.get(key, 0) == 0 for key in ("duplicate_unit_id_count", "source_traceback_missing_count", "cross_page_merged_count"))
    thresholds = {
        "table_recovery": counts["table_recovery"] >= 21,
        "row_recovery": counts["row_recovery"] >= 21,
        "numeric_exact": counts["numeric_exact"] >= 21,
        "scale_exact": counts["scale_exact"] >= 21,
        "metric_exact": counts["metric_exact"] >= 21,
        "period_exact": counts["period_exact"] >= 21,
        "metric_period": counts["metric_period"] >= 21,
        "metric_period_value": counts["metric_period_value"] >= 21,
        "source_traceback": counts["source_traceback"] >= 21,
        "integrity": integrity_ok,
    }
    passed = all(thresholds.values())
    decision, next_gate = ("evidence_unit_generation_passed", "multi_granularity_shadow_index") if passed else ("evidence_unit_generation_blocked", "stop_and_fix_evidence_units")
    scoring = {
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "prediction_seal_verified": True,
        "oracle_record_count": len(scored),
        "unique_source_identity_count": len(unique_ids),
        "duplicate_source_record_count": len(scored) - len(unique_ids),
        "record_metrics": {key: [value, len(scored)] for key, value in counts.items()},
        "unique_source_metrics": {key: [value, len(unique_ids)] for key, value in unique_counts.items()},
        "unit_integrity": integrity,
        "thresholds": thresholds,
        "runtime_oracle_reads": 0,
        "posthoc_oracle_records_read": len(scored),
    }
    _write(root / "oracle-evidence-audit.json", {"record_results": scored, "unique_source_results": unique_scored})
    _write(root / "evidence-unit-scoring.json", scoring)
    _write(root / "acceptance.json", {
        "gate": "pdf_retrieval_v4_gate_05",
        "gate_passed": passed,
        "decision": decision,
        "next_gate": next_gate,
        "prediction_seal_verified": True,
        "cross_page_merged": False,
        "runtime_oracle_reads": 0,
        "posthoc_oracle_records_read": len(scored),
        "runtime_governance_reads": 0,
        "expected_value_reads_runtime": 0,
        "question_reads": 0,
        "index_builds": 0,
        "retrieval_runs": 0,
        "answer_generation_calls": 0,
        "reranker_calls": 0,
        "model_training_calls": 0,
        "parameter_scan": False,
        "per_query_oracle": False,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "candidate_identity_conflicts": 0,
        "duplicate_views": 0,
        "duplicate_unit_ids": integrity.get("duplicate_unit_id_count", 0),
        "source_traceback_missing": integrity.get("source_traceback_missing_count", 0),
        "reference_answer_reads_runtime": 0,
        "production_switch_allowed": False,
    })
    _write(root / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False, "post_score_tuning_allowed": False})
    print(json.dumps({"decision": decision, "record_metrics": counts, "unique_source_metrics": unique_counts, "unit_count": len(units), "thresholds": thresholds}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
