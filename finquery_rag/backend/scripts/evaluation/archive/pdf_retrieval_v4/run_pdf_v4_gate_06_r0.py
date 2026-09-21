"""Audit and close Gate 05 Evidence Unit stream/manifest counts before indexing."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE05 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-06"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_stream(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    physical = nonempty = blank = invalid = 0
    parsed: list[dict[str, Any]] = []
    header: dict[str, Any] | None = None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            physical += 1
            if not line.strip():
                blank += 1
                continue
            nonempty += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if line_number == 1:
                if not isinstance(value, dict) or value.get("format") != "evidence_unit_jsonl_v1":
                    invalid += 1
                else:
                    header = value
                continue
            if isinstance(value, dict):
                parsed.append(value)
            else:
                invalid += 1
    if header is None:
        raise RuntimeError("evidence_unit_stream_header_missing")
    return header, parsed, {"physical_line_count": physical, "nonempty_line_count": nonempty, "blank_line_count": blank, "invalid_json_count": invalid}


def _audit(gate05: Path) -> dict[str, Any]:
    stream_path = gate05 / "evidence-units.jsonl.gz"
    manifest_path = gate05 / "evidence-units-manifest.json"
    seal_path = gate05 / "evidence-unit-prediction-seal.json"
    metrics_path = gate05 / "evidence-unit-metrics.json"
    for path in (stream_path, manifest_path, seal_path, metrics_path):
        if not path.is_file():
            raise RuntimeError(f"missing_gate05_count_input:{path.name}")
    header, units, line_counts = _load_stream(stream_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ids = [unit.get("evidence_unit_id") for unit in units]
    type_counts: dict[str, int] = {}
    for unit in units:
        unit_type = unit.get("unit_type")
        type_counts[unit_type] = type_counts.get(unit_type, 0) + 1
    expected_types = {"section": 117, "table": 117, "row": 1581, "cell": 7136, "fact": 4254}
    unknown_type_count = sum(count for unit_type, count in type_counts.items() if unit_type not in expected_types)
    duplicate_unit_id_count = len(ids) - len({value for value in ids if value})
    missing_unit_id_count = sum(not value for value in ids)
    canonical_payload = {"prediction_count": header.get("prediction_count"), "unit_count": len(units), "cross_page_merged": bool(header.get("cross_page_merged")), "evidence_units": units}
    uncompressed_hash = _hash(canonical_payload)
    compressed_hash = _sha(stream_path)
    manifest_matches = manifest.get("record_count") == len(units) and manifest.get("compressed_sha256") == compressed_hash and manifest.get("uncompressed_sha256") == uncompressed_hash
    metrics_count_matches = metrics.get("unit_count") == len(units) and {key: metrics.get(f"{key}_count") for key in expected_types} == expected_types
    prediction_header, prediction_units, _ = _load_stream(gate05 / "evidence-unit-predictions.jsonl.gz")
    prediction_payload_hash = _hash({"prediction_count": prediction_header.get("prediction_count"), "unit_count": len(prediction_units), "cross_page_merged": bool(prediction_header.get("cross_page_merged")), "evidence_units": prediction_units})
    audit = {
        "stream_path": stream_path.name,
        "gzip_physical_line_count": line_counts["physical_line_count"],
        "header_line_count": 1,
        "nonempty_line_count": line_counts["nonempty_line_count"],
        "nonempty_record_count": len(units),
        "parsed_record_count": len(units),
        "blank_line_count": line_counts["blank_line_count"],
        "invalid_json_count": line_counts["invalid_json_count"],
        "unknown_type_count": unknown_type_count,
        "missing_unit_id_count": missing_unit_id_count,
        "duplicate_unit_id_count": duplicate_unit_id_count,
        "type_counts": type_counts,
        "expected_type_counts": expected_types,
        "five_type_sum": sum(type_counts.get(key, 0) for key in expected_types),
        "metrics_unit_count": metrics.get("unit_count"),
        "manifest_record_count": manifest.get("record_count"),
        "manifest_matches_parsed_records": manifest_matches,
        "metrics_matches_parsed_records": metrics_count_matches,
        "compressed_sha256": compressed_hash,
        "manifest_compressed_sha256": manifest.get("compressed_sha256"),
        "uncompressed_sha256": uncompressed_hash,
        "manifest_uncompressed_sha256": manifest.get("uncompressed_sha256"),
        "prediction_seal_storage": seal.get("prediction_storage"),
        "prediction_seal_verified": bool(seal.get("predictions_sealed")),
        "prediction_stream_hash": _sha(gate05 / "evidence-unit-predictions.jsonl.gz"),
        "prediction_stream_uncompressed_sha256": prediction_payload_hash,
        "prediction_evidence_stream_equal": prediction_payload_hash == uncompressed_hash,
        "manifest_deterministic_flag": bool(manifest.get("deterministic")),
    }
    audit["gate_passed"] = all((
        audit["parsed_record_count"] == 13205,
        audit["five_type_sum"] == 13205,
        audit["unknown_type_count"] == 0,
        audit["invalid_json_count"] == 0,
        audit["duplicate_unit_id_count"] == 0,
        audit["missing_unit_id_count"] == 0,
        audit["manifest_matches_parsed_records"],
        audit["metrics_matches_parsed_records"],
        audit["prediction_evidence_stream_equal"],
    ))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate05", type=Path, default=DEFAULT_GATE05)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    audit = _audit(args.gate05)
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {"gate": "pdf_retrieval_v4_gate_06_r0", "evaluation_type": "post_benchmark_iterative_evaluation", "input_gate": "pdf_retrieval_v4_gate_05", "question_reads": 0, "runtime_gold_reads": 0, "runtime_governance_reads": 0, "retrieval_runs": 0, "index_builds": 0, "production_index_writes": 0, "production_switch_allowed": False}
    input_integrity = {"gate05_stream_sha256": audit["compressed_sha256"], "gate05_manifest_sha256": _sha(args.gate05 / "evidence-units-manifest.json"), "gate05_metrics_sha256": _sha(args.gate05 / "evidence-unit-metrics.json"), "gate05_prediction_seal_sha256": _sha(args.gate05 / "evidence-unit-prediction-seal.json")}
    decision = "evidence_unit_manifest_integrity_passed" if audit["gate_passed"] else "evidence_unit_manifest_integrity_blocked"
    next_gate = "multi_granularity_shadow_index" if audit["gate_passed"] else "stop_and_fix_gate_05_manifest"
    _write(args.out / "gate-06-protocol.json", protocol)
    _write(args.out / "gate-06-input-integrity.json", input_integrity)
    _write(args.out / "evidence-unit-count-audit.json", audit)
    _write(args.out / "acceptance.json", {"gate": "pdf_retrieval_v4_gate_06_r0", "gate_passed": audit["gate_passed"], "decision": decision, "next_gate": next_gate, "question_reads": 0, "runtime_gold_reads": 0, "runtime_governance_reads": 0, "retrieval_runs": 0, "production_index_writes": 0, "production_switch_allowed": False})
    _write(args.out / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    print(json.dumps({"decision": decision, "next_gate": next_gate, **{key: audit[key] for key in ("gzip_physical_line_count", "parsed_record_count", "blank_line_count", "invalid_json_count", "unknown_type_count", "duplicate_unit_id_count", "manifest_record_count", "five_type_sum")}}, ensure_ascii=False))
    return 0 if audit["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
