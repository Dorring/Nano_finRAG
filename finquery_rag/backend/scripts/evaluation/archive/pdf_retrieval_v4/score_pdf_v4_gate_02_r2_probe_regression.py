"""Gate 02 R2: Post-seal probe regression scoring.

After the seal is verified, reads the 22 Oracle records from Gate 01/02
and checks whether the full-corpus MinerU output maintains the same
structural capability on the original 87-page probe pages.

Reports:
  - Table Recovery (22/22)
  - Row Recovery (22/22)
  - Period Header Availability (22/22)
  - Raw Numeric Recovery (>= 10/22)
  - Raw Scale Recovery (>= 18/22)

This is a post-hoc scoring on already-sealed MinerU output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS_PATH = ROOT / "benchmarks/financial_rag_v1/corpus.json"
DEFAULT_GATE01_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-01"
DEFAULT_GATE02_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2"
SHARED_NANOCHAT_ROOT = ROOT.parents[4]
DEFAULT_MINERU_OUTPUT = SHARED_NANOCHAT_ROOT / ".runtime/pdf-retrieval-v4-gate-02-r2/mineru"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def digits_only(value: Any) -> str:
    return re.sub(r"[^0-9]", "", str(value or ""))


def iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def _find_json(output_dir: Path, pattern: str) -> Path | None:
    matches = sorted(output_dir.rglob(pattern))
    return matches[0] if matches else None


def _extract_middle_pages(middle_path: Path) -> dict[int, dict[str, Any]]:
    """Extract page data from middle.json."""
    if not middle_path or not middle_path.is_file():
        return {}
    try:
        payload = json.loads(middle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    page_info = payload.get("pdf_info", []) if isinstance(payload, dict) else []
    pages: dict[int, dict[str, Any]] = {}
    for page_idx, page_payload in enumerate(page_info):
        tables: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for block in iter_dicts(page_payload):
            table_html = None
            for key in ("html", "table_body"):
                val = block.get(key)
                if isinstance(val, str) and "<table" in val.lower():
                    table_html = val
                    break
            if not table_html:
                continue
            rows: list[list[str]] = []
            for row_match in re.finditer(
                r"<tr[^>]*>(.*?)</tr>", table_html,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                cells: list[str] = []
                for cell_match in re.finditer(
                    r"<t[dh][^>]*>(.*?)</t[dh]>", row_match.group(1),
                    flags=re.IGNORECASE | re.DOTALL,
                ):
                    cell = re.sub(r"<[^>]+>", " ", cell_match.group(1))
                    cells.append(re.sub(r"\s+", " ", cell).strip())
                if cells:
                    rows.append(cells)
            bbox = block.get("bbox") or block.get("img_bbox")
            key = (normalize_text(table_html), json.dumps(bbox, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            tables.append({
                "bbox": bbox,
                "row_count": len(rows),
                "rows": rows,
                "text": " ".join(" ".join(row) for row in rows),
            })
        pages[page_idx] = {"tables": tables}
    return pages


def _extract_content_texts(content_path: Path) -> dict[int, str]:
    """Extract page texts from content_list.json."""
    if not content_path or not content_path.is_file():
        return {}
    try:
        content = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    pages: defaultdict[int, list[str]] = defaultdict(list)
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        page_idx = block.get("page_idx", 0)
        if block.get("type") in {"text", "title", "discarded"} and block.get("text"):
            pages[int(page_idx)].append(str(block["text"]))
    return {page: " ".join(texts) for page, texts in pages.items()}


def _page_metrics(middle: dict[str, Any], page_text: str) -> dict[str, Any]:
    """Compute page metrics for oracle matching."""
    tables = middle.get("tables", [])
    table_text = " ".join(str(t.get("text") or "") for t in tables)
    all_text = f"{page_text} {table_text}"
    return {
        "table_count": len(tables),
        "all_text": all_text,
        "tables": tables,
    }


def _oracle_page_match(
    oracle_record: dict[str, Any],
    page: dict[str, Any],
) -> dict[str, bool]:
    """Match an oracle record against a parsed page."""
    all_text = normalize_text(page.get("all_text"))
    metric = normalize_text(
        oracle_record.get("expected_metric")
        or oracle_record.get("proposed_candidate", {}).get("normalized_metric")
    )
    metric_tokens = [
        t for t in re.findall(r"[a-z0-9]+", metric) if len(t) > 2
    ]
    metric_match = bool(metric_tokens) and all(
        t in all_text for t in metric_tokens
    )

    expected_raw = str(
        oracle_record.get("proposed_candidate", {}).get("raw_cell_text")
        or oracle_record.get("expected_numeric")
        or oracle_record.get("expected_value")
        or ""
    )
    expected_digits = digits_only(expected_raw)
    expected_base = digits_only(
        oracle_record.get("proposed_candidate", {}).get("normalized_base_value")
        or oracle_record.get("expected_numeric")
        or oracle_record.get("expected_value")
    )
    page_digits = digits_only(page.get("all_text"))
    numeric_match = bool(expected_digits and expected_digits in page_digits)
    if not numeric_match and expected_base:
        numeric_match = expected_base in page_digits

    period = normalize_text(
        oracle_record.get("expected_period")
        or oracle_record.get("proposed_candidate", {}).get("normalized_period")
    )
    year_match = bool(period) and any(
        year in all_text for year in re.findall(r"\d{4}", period)
    )

    scale_match = bool(
        re.search(r"\b(?:million|millions|thousand|thousands|billion|billions)\b", all_text)
    )

    return {
        "table_detected": bool(page.get("table_count")),
        "row_text_recovered": metric_match and bool(page.get("table_count")),
        "metric_text_recovered": metric_match,
        "numeric_text_recovered": numeric_match,
        "numeric_text_accurate": numeric_match,
        "period_header_available": year_match,
        "scale_header_available": scale_match,
    }


def load_oracle_records(gate02_out: Path) -> list[dict[str, Any]]:
    """Load the 22 Oracle records from Gate 02 oracle source audit."""
    oracle_path = gate02_out / "gate-02-oracle-source-audit.jsonl"
    if not oracle_path.is_file():
        # Fallback to Gate 01 oracle package
        oracle_path = ROOT / "artifacts/evaluation/nf-opt-08-r2/manual-mapping-review-package.json"
        if oracle_path.is_file():
            data = json.loads(oracle_path.read_text(encoding="utf-8"))
            return data.get("records", [])
        return []
    records: list[dict[str, Any]] = []
    for line in oracle_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate01-out", type=Path, default=DEFAULT_GATE01_OUT)
    parser.add_argument("--gate02-out", type=Path, default=DEFAULT_GATE02_OUT)
    parser.add_argument("--mineru-output", type=Path, default=DEFAULT_MINERU_OUTPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # Verify seal
    seal_path = args.out_dir / "full-corpus-ingestion-seal.json"
    if not seal_path.is_file():
        print("ERROR: Seal not found. Run seal_pdf_v4_gate_02_r2_outputs.py first.")
        return 1
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("sealed"):
        print("ERROR: Seal not valid.")
        return 1

    # Load oracle records (post-seal, allowed)
    oracle_records = load_oracle_records(args.gate02_out)
    print(f"Loaded {len(oracle_records)} Oracle records (post-seal)")

    # Load probe input manifest to get page mappings
    probe_manifest = json.loads(
        (args.gate01_out / "probe-input-manifest.json").read_text(encoding="utf-8")
    )
    probe_records = probe_manifest.get("records", [])

    # Build (document_id, pdf_page) -> page_index mapping
    page_map: dict[tuple[str, int], int] = {}
    for rec in probe_records:
        doc_id = str(rec.get("document_id") or "")
        pdf_page = int(rec.get("pdf_page") or 0)
        page_map[(doc_id, pdf_page)] = pdf_page - 1  # 0-based

    # For each oracle record, find the corresponding page in the full-corpus output
    per_oracle: list[dict[str, Any]] = []
    counters: defaultdict[str, int] = defaultdict(int)
    missing: list[dict[str, Any]] = []

    for oracle_rec in oracle_records:
        doc_id = str(oracle_rec.get("document_id") or "")
        pdf_page = int(oracle_rec.get("pdf_page") or 0)
        page_idx = pdf_page - 1  # 0-based

        # Find the document's MinerU output
        doc_output = args.mineru_output / doc_id
        if not doc_output.is_dir():
            missing.append({"document_id": doc_id, "pdf_page": pdf_page, "reason": "no_output_dir"})
            continue

        middle_path = _find_json(doc_output, "*_middle.json")
        content_path = _find_json(doc_output, "*_content_list.json")

        middle_pages = _extract_middle_pages(middle_path) if middle_path else {}
        content_texts = _extract_content_texts(content_path) if content_path else {}

        page_middle = middle_pages.get(page_idx, {"tables": []})
        page_text = content_texts.get(page_idx, "")
        page = _page_metrics(page_middle, page_text)

        match = _oracle_page_match(oracle_rec, page)
        for name, value in match.items():
            counters[name] += int(value)

        per_oracle.append({
            "oracle_record_id": oracle_rec.get("oracle_record_id"),
            "document_id": doc_id,
            "pdf_page": pdf_page,
            "expected_metric": oracle_rec.get("expected_metric"),
            "expected_numeric": oracle_rec.get("expected_numeric"),
            "expected_period": oracle_rec.get("expected_period"),
            "observed": match,
        })

    denominator = len(oracle_records)
    if denominator == 0:
        print("ERROR: No oracle records found.")
        return 1

    def ratio(name: str) -> float:
        return counters[name] / denominator

    metrics = {
        "table_recovery": ratio("table_detected"),
        "row_recovery": ratio("row_text_recovered"),
        "period_header_availability": ratio("period_header_available"),
        "raw_numeric_recovery": ratio("numeric_text_recovered"),
        "raw_scale_recovery": ratio("scale_header_available"),
    }

    # Count raw numbers
    table_count = counters["table_detected"]
    row_count = counters["row_text_recovered"]
    period_count = counters["period_header_available"]
    numeric_count = counters["numeric_text_recovered"]
    scale_count = counters["scale_header_available"]

    # Gate checks
    table_passed = table_count >= 22
    row_passed = row_count >= 22
    period_passed = period_count >= 22
    numeric_passed = numeric_count >= 10
    scale_passed = scale_count >= 18

    # Structural regression check
    structural_regression = not (table_passed and row_passed and period_passed)

    result = {
        "gate": "pdf_retrieval_v4_gate_02_r2",
        "oracle_record_count": denominator,
        "oracle_records_scored": len(per_oracle),
        "missing_oracle_pages": missing,
        "metrics": metrics,
        "raw_counts": {
            "table_recovery": f"{table_count}/{denominator}",
            "row_recovery": f"{row_count}/{denominator}",
            "period_header_availability": f"{period_count}/{denominator}",
            "raw_numeric_recovery": f"{numeric_count}/{denominator}",
            "raw_scale_recovery": f"{scale_count}/{denominator}",
        },
        "gate_checks": {
            "table_recovery_22_22": table_passed,
            "row_recovery_22_22": row_passed,
            "period_22_22": period_passed,
            "raw_numeric_ge_10": numeric_passed,
            "raw_scale_ge_18": scale_passed,
        },
        "structural_regression": structural_regression,
        "oracle_records_read_after_seal": True,
        "oracle_scoring_posthoc": True,
        "per_oracle": per_oracle,
    }

    write_json(args.out_dir / "post-seal-probe-regression.json", result)

    print("Probe regression scoring complete:")
    print(f"  Table Recovery:       {table_count}/{denominator} {'PASS' if table_passed else 'FAIL'}")
    print(f"  Row Recovery:         {row_count}/{denominator} {'PASS' if row_passed else 'FAIL'}")
    print(f"  Period Header:        {period_count}/{denominator} {'PASS' if period_passed else 'FAIL'}")
    print(f"  Raw Numeric:          {numeric_count}/{denominator} {'PASS' if numeric_passed else 'FAIL'}")
    print(f"  Raw Scale:            {scale_count}/{denominator} {'PASS' if scale_passed else 'FAIL'}")
    print(f"  Structural regression: {structural_regression}")

    if structural_regression:
        print("\nBLOCKED: Structural regression detected.")
        return 1

    print("\nProbe regression PASSED (no structural regression).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
