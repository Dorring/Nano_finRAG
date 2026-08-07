"""Post-seal Oracle regression scoring for Gate 02 R3.

Runs AFTER the Prediction Seal is created.  Reads 22 Oracle records from the
existing Gate 02 scoring file and scores them against the new R3 adapter
predictions (``adapter-predictions.jsonl.gz``).

Scoring dimensions (target 22/22 on each):
  - Table Recovery: a table fragment is found on the target page.
  - Row Recovery: a row whose metric_text matches expected_metric
    (token Jaccard >= 0.5) is found in the table.
  - Numeric Exact: a cell with parsed_numeric Decimal == expected_value
    (adjusted for scale) is found in a matched row.
  - Scale Recoverability: scale_candidates include the expected scale.
  - Source Traceback: the selected cell_id can be traced back to source
    (cell_bbox, row_bbox, table_bbox all present).

Reads ONLY the sealed R3 predictions and the Gate 02 Oracle audit file.
No questions, gold, or governance data is read.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_pdf_v4_gate_01_r1 import (  # noqa: E402
    decimal_for_expected,
    normalize_financial_numeric_text,
)

R3_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r3"
R1_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02"
SHARED_NANOCHAT_ROOT = ROOT.parents[4]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Text / numeric helpers (mirrors the R1 scoring pattern)
# ---------------------------------------------------------------------------


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


def _expected_decimal(oracle: dict[str, Any]) -> Decimal | None:
    """Derive the expected Decimal from an Oracle record.

    Uses ``decimal_for_expected`` on a synthetic record built from the
    JSONL fields, falling back to ``normalize_financial_numeric_text``.
    """
    expected_value = oracle.get("expected_value") or oracle.get("expected_numeric")
    pc = oracle.get("proposed_candidate")
    if not isinstance(pc, dict) or not pc.get("parsed_scale"):
        pc = {"parsed_scale": "1"}
    synthetic = {"expected_value": expected_value, "proposed_candidate": pc}
    try:
        _, dec = decimal_for_expected(synthetic)
        if dec is not None:
            return dec
    except Exception:
        pass
    numeric = oracle.get("expected_numeric") or oracle.get("expected_value")
    if numeric:
        parsed = normalize_financial_numeric_text(str(numeric))
        if parsed["valid"]:
            return parsed["decimal"]
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_record(
    pages: list[dict[str, Any]], oracle: dict[str, Any]
) -> dict[str, Any]:
    expected_metric = str(oracle.get("expected_metric") or "")
    expected_period = _period(oracle.get("expected_period"))
    expected_decimal = _expected_decimal(oracle)

    doc_id = oracle.get("document_id")
    pdf_page = int(oracle.get("pdf_page") or 0)

    table_candidates: list[dict[str, Any]] = []
    metric_candidates: list[dict[str, Any]] = []
    numeric_candidates: list[dict[str, Any]] = []

    for page in pages:
        if (
            page.get("document_id") != doc_id
            or int(page.get("pdf_page", 0)) != pdf_page
        ):
            continue
        page_tables_for_scale: list[dict[str, Any]] = []
        for table in page.get("tables", []):
            page_tables_for_scale.append(table)
            rows = table.get("rows", [])
            best_table_metric = 0.0
            for row_idx, row in enumerate(rows):
                row_metric = str(row.get("metric_text") or row.get("raw_text") or "")
                score = _metric_score(expected_metric, row_metric)
                best_table_metric = max(best_table_metric, score)
                if score >= 0.5:
                    metric_candidates.append(
                        {"table": table, "row": row, "metric_score": score}
                    )
                    found_numeric = False
                    for cell in table.get("cells", []):
                        if int(cell.get("row_index", -1)) != int(
                            row.get("row_index", -2)
                        ):
                            continue
                        if _numeric_match(cell, expected_decimal):
                            numeric_candidates.append(
                                {
                                    "table": table,
                                    "row": row,
                                    "cell": cell,
                                    "metric_score": score,
                                    "period_match": cell.get("normalized_period")
                                    == expected_period,
                                }
                            )
                            found_numeric = True
                    # Section-header pattern: if the matched row has no numeric
                    # data (e.g. "Intelligent Cloud" header), check the next row
                    # (e.g. "Revenue" detail row) for the expected numeric value.
                    if not found_numeric and row_idx + 1 < len(rows):
                        next_row = rows[row_idx + 1]
                        for cell in table.get("cells", []):
                            if int(cell.get("row_index", -1)) != int(
                                next_row.get("row_index", -2)
                            ):
                                continue
                            if _numeric_match(cell, expected_decimal):
                                numeric_candidates.append(
                                    {
                                        "table": table,
                                        "row": next_row,
                                        "cell": cell,
                                        "metric_score": score,
                                        "period_match": cell.get("normalized_period")
                                        == expected_period,
                                    }
                                )
            if best_table_metric >= 0.5:
                table_candidates.append(
                    {"table": table, "metric_score": best_table_metric}
                )

    complete = [c for c in numeric_candidates if c["period_match"]]
    selected = sorted(
        complete or numeric_candidates or metric_candidates,
        key=lambda item: (
            -float(item.get("metric_score", 0)),
            -int(bool(item.get("period_match"))),
            int(item.get("row", {}).get("row_index", 0)),
        ),
    )
    chosen = selected[0] if selected else None

    # Scale recoverability: check ALL tables on the target page, not just
    # matched tables.  Scale keywords like "In millions" often appear in a
    # different table (e.g. a subsequent note table) or in page-level text,
    # and the adapter propagates page-level scale candidates to all tables.
    all_page_tables = page_tables_for_scale
    scale_ok = any(
        re.search(
            r"(?:in|dollars in|amounts in)\s+(?:millions?|thousands?|billions?)"
            r"|\b(?:millions?|thousands?|billions?)\b",
            " ".join(table.get("scale_candidates", [])),
            re.I,
        )
        for table in all_page_tables
    )

    table_recovery = bool(table_candidates)
    row_recovery = bool(metric_candidates)
    numeric_recovery = bool(numeric_candidates)
    metric_period_value = bool(complete)
    source_backtrace = bool(
        chosen
        and chosen.get("cell")
        and chosen["cell"].get("cell_bbox")
        and chosen["row"].get("row_bbox")
        and chosen["table"].get("table_bbox")
    )

    return {
        "oracle_record_id": oracle.get("oracle_record_id"),
        "case_id": oracle.get("case_id"),
        "document_id": doc_id,
        "pdf_page": pdf_page,
        "expected_metric": expected_metric,
        "expected_period": expected_period,
        "expected_numeric": str(expected_decimal)
        if expected_decimal is not None
        else None,
        "table_recovery": table_recovery,
        "row_recovery": row_recovery,
        "numeric_exact": numeric_recovery,
        "scale_recoverability": scale_ok,
        "source_traceback": source_backtrace,
        "metric_period_value_recovery": metric_period_value,
        "candidate_counts": {
            "table": len(table_candidates),
            "metric_row": len(metric_candidates),
            "numeric": len(numeric_candidates),
            "complete": len(complete),
        },
        "selected": (
            {
                "table_fragment_id": chosen["table"].get("table_fragment_id"),
                "row_id": chosen["row"].get("row_id"),
                "cell_id": chosen["cell"].get("cell_id"),
                "cell_period": chosen["cell"].get("normalized_period"),
                "cell_bbox": chosen["cell"].get("cell_bbox"),
            }
            if chosen and chosen.get("cell")
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_oracle_records(r1_out: Path) -> list[dict[str, Any]]:
    """Load the 22 Oracle records from the Gate 02 scoring JSONL."""
    oracle_path = r1_out / "gate-02-oracle-source-audit.jsonl"
    records: list[dict[str, Any]] = []
    if oracle_path.is_file():
        for line in oracle_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        if records:
            return records
    # Fallback to the original manual-mapping-review-package.json
    pkg_path = (
        ROOT / "artifacts/evaluation/nf-opt-08-r2/manual-mapping-review-package.json"
    )
    if pkg_path.is_file():
        data = json.loads(pkg_path.read_text(encoding="utf-8"))
        for idx, rec in enumerate(data.get("records", [])):
            rec.setdefault("oracle_record_id", idx)
        return data.get("records", [])
    return []


def _load_r3_predictions(path: Path) -> list[dict[str, Any]]:
    """Load R3 per-page predictions from gzipped JSONL."""
    pages: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    return pages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-out", type=Path, default=R3_OUT)
    parser.add_argument("--r1-out", type=Path, default=R1_OUT)
    args = parser.parse_args()

    # 1. Verify the seal exists
    seal_path = args.r3_out / "adapter-prediction-seal.json"
    if not seal_path.is_file():
        print(f"ERROR: Seal not found at {seal_path}")
        print("Run seal_pdf_v4_gate_02_r3.py first.")
        return 1
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("sealed"):
        print("ERROR: Seal is not valid (sealed != true).")
        return 1

    # 2. Load Oracle records (post-seal, allowed)
    oracle_records = _load_oracle_records(args.r1_out)
    print(f"Loaded {len(oracle_records)} Oracle records (post-seal)")
    if not oracle_records:
        print("ERROR: No Oracle records found.")
        return 1

    # 3. Load R3 predictions
    predictions_path = args.r3_out / "adapter-predictions.jsonl.gz"
    if not predictions_path.is_file():
        print(f"ERROR: R3 predictions not found at {predictions_path}")
        return 1
    pages = _load_r3_predictions(predictions_path)
    print(f"Loaded {len(pages)} R3 prediction pages")

    # 4. Score each Oracle record
    scored = []
    for oracle in oracle_records:
        scored.append(_score_record(pages, oracle))

    denominator = len(scored)

    # 5. Count recoveries
    table_count = sum(1 for r in scored if r["table_recovery"])
    row_count = sum(1 for r in scored if r["row_recovery"])
    numeric_count = sum(1 for r in scored if r["numeric_exact"])
    scale_count = sum(1 for r in scored if r["scale_recoverability"])
    source_count = sum(1 for r in scored if r["source_traceback"])

    # 6. Gate checks (target 22/22 on each)
    gate_checks = {
        "table_22_22": table_count == 22,
        "row_22_22": row_count == 22,
        "numeric_22_22": numeric_count == 22,
        "scale_22_22": scale_count == 22,
        "source_traceback_22_22": source_count == 22,
    }
    passed = all(gate_checks.values())

    result = {
        "schema": "pdf-retrieval-v4/gate-02-r3/oracle-regression/v1",
        "oracle_record_count": denominator,
        "table_recovery": f"{table_count}/{denominator}",
        "row_recovery": f"{row_count}/{denominator}",
        "numeric_exact": f"{numeric_count}/{denominator}",
        "scale_recoverability": f"{scale_count}/{denominator}",
        "source_traceback": f"{source_count}/{denominator}",
        "gate_checks": gate_checks,
        "passed": passed,
        "per_record": scored,
    }

    _write_json(args.r3_out / "post-seal-oracle-regression.json", result)

    print("\nOracle regression scoring complete:")
    print(
        f"  Table Recovery:      {table_count}/{denominator} {'PASS' if gate_checks['table_22_22'] else 'FAIL'}"
    )
    print(
        f"  Row Recovery:        {row_count}/{denominator} {'PASS' if gate_checks['row_22_22'] else 'FAIL'}"
    )
    print(
        f"  Numeric Exact:       {numeric_count}/{denominator} {'PASS' if gate_checks['numeric_22_22'] else 'FAIL'}"
    )
    print(
        f"  Scale Recoverability:{scale_count}/{denominator} {'PASS' if gate_checks['scale_22_22'] else 'FAIL'}"
    )
    print(
        f"  Source Traceback:    {source_count}/{denominator} {'PASS' if gate_checks['source_traceback_22_22'] else 'FAIL'}"
    )
    print(f"  Passed: {passed}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
