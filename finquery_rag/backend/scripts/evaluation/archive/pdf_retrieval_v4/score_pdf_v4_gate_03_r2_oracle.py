"""Gate 03 R2: Post-seal Oracle regression scoring (22 records)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT))

GATE03_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-03-r2"
R3_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r3"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_for_match(value: Any) -> str:
    """Normalize for case-insensitive, space-insensitive substring matching."""
    return (
        _coerce_str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 03 R2 oracle regression scoring")
    parser.add_argument("--backend-root", type=str, default=str(ROOT))
    parser.add_argument("--gate03-out", type=str, default=str(GATE03_OUT))
    parser.add_argument("--r3-out", type=str, default=str(R3_OUT))
    args = parser.parse_args()

    gate03_out = Path(args.gate03_out)
    r3_out = Path(args.r3_out)

    oracle_path = r3_out / "post-seal-oracle-regression.json"
    if not oracle_path.exists():
        print(f"ERROR: oracle file not found: {oracle_path}", file=sys.stderr)
        return 1

    oracle_data = _read_json(oracle_path)
    per_record = oracle_data.get("per_record", [])

    atomic_facts = _read_jsonl(gate03_out / "atomic-facts.jsonl")
    row_matrices = _read_jsonl(gate03_out / "row-matrices.jsonl")
    scale_resolutions = _read_jsonl(gate03_out / "scale-resolutions.jsonl")

    # Build table-level scale lookup: table_fragment_id -> (scale, scale_unit)
    table_scale_map: dict[str, tuple[Any, str | None]] = {}
    for sr in scale_resolutions:
        tfid = _coerce_str(sr.get("table_fragment_id"))
        if tfid and sr.get("scale_status") == "resolved":
            table_scale_map[tfid] = (sr.get("scale"), sr.get("scale_unit"))

    # Build table_fragment_id -> document_id map from atomic facts
    tfid_to_doc: dict[str, str] = {}
    for f in atomic_facts:
        tfid = _coerce_str(f.get("table_fragment_id"))
        doc = _coerce_str(f.get("document_id"))
        if tfid and doc:
            tfid_to_doc[tfid] = doc

    # Build document-level dominant scale: document_id -> (scale, scale_unit)
    doc_scale_counter: dict[str, Counter] = {}
    for sr in scale_resolutions:
        tfid = _coerce_str(sr.get("table_fragment_id"))
        doc = tfid_to_doc.get(tfid, "")
        status = sr.get("scale_status", "")
        scale = sr.get("scale")
        if doc and status == "resolved" and scale is not None:
            doc_scale_counter.setdefault(doc, Counter())[scale] += 1

    doc_scale_map: dict[str, tuple[Any, str | None]] = {}
    for doc, counter in doc_scale_counter.items():
        if counter:
            dominant_scale, _ = counter.most_common(1)[0]
            unit = (
                "millions"
                if dominant_scale == 1000000.0
                else (
                    "billions"
                    if dominant_scale == 1000000000.0
                    else ("thousands" if dominant_scale == 1000.0 else None)
                )
            )
            doc_scale_map[doc] = (dominant_scale, unit)

    per_record_results: list[dict[str, Any]] = []
    numeric_count = 0
    period_count = 0
    scale_count = 0
    source_count = 0
    mpv_count = 0

    def _check_scale(fact_or_rm: dict[str, Any], doc_id: str) -> bool:
        """Check scale from fact/matrix, then table-level, then document-level."""
        scale = fact_or_rm.get("scale")
        scale_unit = fact_or_rm.get("scale_unit")
        if scale is not None or scale_unit is not None:
            return True
        tbl = _coerce_str(fact_or_rm.get("table_fragment_id"))
        if tbl in table_scale_map:
            return True
        if doc_id in doc_scale_map:
            return True
        return False

    def _check_metric_match(expected_metric: str, metric_path: str) -> bool:
        """Check if expected_metric matches metric_path (bidirectional, normalized)."""
        if not expected_metric or not metric_path:
            return False
        exp_norm = _normalize_for_match(expected_metric)
        mp_norm = _normalize_for_match(metric_path)
        if not exp_norm or not mp_norm:
            return False
        # Bidirectional substring: either is a substring of the other
        return exp_norm in mp_norm or mp_norm in exp_norm

    for rec in per_record:
        record_id = rec.get("record_id") or rec.get("id")
        expected_numeric = _coerce_str(rec.get("expected_numeric"))
        expected_period = _coerce_str(rec.get("expected_period"))
        expected_metric = _coerce_str(rec.get("expected_metric"))
        document_id = _coerce_str(rec.get("document_id"))

        numeric_present = False
        period_correct = False
        scale_recoverable = False
        source_traceback = False
        metric_period_value = False

        def _value_matches(value_raw: str, value_normalized: str) -> bool:
            if not expected_numeric:
                return False
            return expected_numeric in value_raw or expected_numeric in value_normalized

        def _period_matches(normalized_period: str) -> bool:
            if not expected_period:
                return False
            if normalized_period != expected_period:
                return False
            if document_id:
                return True  # document_id check done at caller level
            return True

        # Check atomic facts
        for fact in atomic_facts:
            value_raw = _coerce_str(fact.get("value_raw"))
            value_normalized = _coerce_str(fact.get("value_normalized"))
            normalized_period = _coerce_str(fact.get("normalized_period"))
            fact_doc_id = _coerce_str(fact.get("document_id"))
            metric_path = _coerce_str(fact.get("metric_path"))
            src_tb = fact.get("source_traceback") or {}

            doc_matches = not document_id or fact_doc_id == document_id

            if _value_matches(value_raw, value_normalized):
                numeric_present = True

                if _period_matches(normalized_period) and doc_matches:
                    period_correct = True

                if _check_scale(fact, fact_doc_id):
                    scale_recoverable = True

                if (
                    src_tb
                    and src_tb.get("document_id")
                    and src_tb.get("pdf_page") is not None
                ):
                    source_traceback = True

                if (
                    _check_metric_match(expected_metric, metric_path)
                    and _period_matches(normalized_period)
                    and doc_matches
                ):
                    metric_period_value = True

            # Also check period without value match (for period_correct)
            if expected_period and normalized_period == expected_period and doc_matches:
                period_correct = True

        # Check row matrices for metric_period_value, scale, value, period
        for rm in row_matrices:
            rm_doc_id = _coerce_str(rm.get("document_id"))
            if document_id and rm_doc_id != document_id:
                continue
            rm_metric_path = _coerce_str(rm.get("metric_path"))

            dims = rm.get("dimensions") or []
            for dim in dims:
                dim_vr = _coerce_str(dim.get("value_raw"))
                dim_vn = _coerce_str(dim.get("value_normalized"))
                dim_period = _coerce_str(dim.get("normalized_period"))

                if _value_matches(dim_vr, dim_vn):
                    numeric_present = True

                    if _period_matches(dim_period):
                        period_correct = True

                    if _check_scale(rm, rm_doc_id):
                        scale_recoverable = True

                    rm_src = rm.get("source_traceback") or {}
                    if (
                        rm_src
                        and rm_src.get("document_id")
                        and rm_src.get("pdf_page") is not None
                    ):
                        source_traceback = True

                    if _check_metric_match(
                        expected_metric, rm_metric_path
                    ) and _period_matches(dim_period):
                        metric_period_value = True

                if expected_period and dim_period == expected_period:
                    period_correct = True

        if numeric_present:
            numeric_count += 1
        if period_correct:
            period_count += 1
        if scale_recoverable:
            scale_count += 1
        if source_traceback:
            source_count += 1
        if metric_period_value:
            mpv_count += 1

        per_record_results.append(
            {
                "record_id": record_id,
                "document_id": document_id,
                "expected_numeric": expected_numeric,
                "expected_period": expected_period,
                "expected_metric": expected_metric,
                "numeric_present": numeric_present,
                "period_correct": period_correct,
                "scale_recoverable": scale_recoverable,
                "source_traceback": source_traceback,
                "metric_period_value": metric_period_value,
            }
        )

    total = len(per_record_results)
    result = {
        "total_records": total,
        "numeric": {
            "pass": numeric_count,
            "total": total,
            "ratio": numeric_count / total if total else 0,
        },
        "period": {
            "pass": period_count,
            "total": total,
            "ratio": period_count / total if total else 0,
        },
        "scale": {
            "pass": scale_count,
            "total": total,
            "ratio": scale_count / total if total else 0,
        },
        "source_traceback": {
            "pass": source_count,
            "total": total,
            "ratio": source_count / total if total else 0,
        },
        "metric_period_value": {
            "pass": mpv_count,
            "total": total,
            "ratio": mpv_count / total if total else 0,
        },
        "per_record": per_record_results,
    }

    out_path = gate03_out / "oracle-regression.json"
    _write_json(out_path, result)

    print(f"Oracle regression written: {out_path}")
    print(f"Numeric:               {numeric_count}/{total}")
    print(f"Period:                {period_count}/{total}")
    print(f"Scale:                 {scale_count}/{total}")
    print(f"Source Traceback:      {source_count}/{total}")
    print(f"Metric x Period x Value: {mpv_count}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
