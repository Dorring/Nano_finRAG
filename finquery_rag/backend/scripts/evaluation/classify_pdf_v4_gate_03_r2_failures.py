"""Gate 03 R2: Failure classification for failed gold records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT))

GATE03_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-03-r2"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 03 R2 failure classification")
    parser.add_argument("--backend-root", type=str, default=str(ROOT))
    parser.add_argument("--gate03-out", type=str, default=str(GATE03_OUT))
    args = parser.parse_args()

    gate03_out = Path(args.gate03_out)

    scoring_path = gate03_out / "problem-gold-scoring.json"
    if not scoring_path.exists():
        print(f"ERROR: scoring file not found: {scoring_path}", file=sys.stderr)
        return 1

    scoring = _read_json(scoring_path)

    category_counts = {
        "metric_graph_missing": 0,
        "temporal_binding_missing": 0,
        "dimension_binding_missing": 0,
        "scale_resolution_missing": 0,
        "row_matrix_missing": 0,
        "narrative_evidence_missing": 0,
    }

    classified_records: list[dict[str, Any]] = []

    def _classify(class_label: str, records: list[dict[str, Any]]) -> None:
        for rec in records:
            if rec.get("all_correct"):
                continue
            failures: list[str] = []

            if not rec.get("metric_path_correct"):
                failures.append("metric_graph_missing")
            if not rec.get("temporal_binding_correct"):
                failures.append("temporal_binding_missing")
            if not rec.get("value_present"):
                failures.append("dimension_binding_missing")
            if not rec.get("scale_correct_or_recoverable"):
                failures.append("scale_resolution_missing")
            if not rec.get("typed_evidence_present"):
                failures.append("row_matrix_missing")
            if not rec.get("candidate_compatible_typed_evidence"):
                failures.append("narrative_evidence_missing")

            for f in failures:
                category_counts[f] = category_counts.get(f, 0) + 1

            classified_records.append(
                {
                    "record_id": rec.get("record_id"),
                    "class": class_label,
                    "document_id": rec.get("document_id"),
                    "gold_metric_label": rec.get("gold_metric_label"),
                    "matched_row_ids": rec.get("matched_row_ids"),
                    "failures": failures,
                    "checks": {
                        "metric_path_correct": rec.get("metric_path_correct"),
                        "temporal_binding_correct": rec.get("temporal_binding_correct"),
                        "value_present": rec.get("value_present"),
                        "scale_correct_or_recoverable": rec.get(
                            "scale_correct_or_recoverable"
                        ),
                        "typed_evidence_present": rec.get("typed_evidence_present"),
                        "candidate_compatible_typed_evidence": rec.get(
                            "candidate_compatible_typed_evidence"
                        ),
                    },
                }
            )

    b_records = scoring.get("b_class", {}).get("records", [])
    d_records = scoring.get("d_class", {}).get("records", [])

    _classify("B", b_records)
    _classify("D", d_records)

    total_failed = len(classified_records)

    result = {
        "total_failed": total_failed,
        "category_counts": category_counts,
        "records": classified_records,
    }

    out_path = gate03_out / "failure-classification.json"
    _write_json(out_path, result)

    print(f"Failure classification written: {out_path}")
    print(f"Total failed records: {total_failed}")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
