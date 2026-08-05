"""Audit Financial Table Structure Resolver coverage on independent PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

from src.evaluation.pdf_source_representation_v2 import (
    YEAR_RE,
    extract_scale,
    normalize_text,
    parse_number,
    resolve_period_headers,
    row_label,
    stable_identity,
    statement_from_lines,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-source-representation-v2-gate-a"
SOURCES = (
    {
        "document_id": "walmart_fy2026_pdf_dev",
        "issuer": "Walmart Inc.",
        "url": "https://www.sec.gov/Archives/edgar/data/104169/000010416926000091/wmtfy26annualreport-final.pdf",
    },
    {
        "document_id": "adobe_fy2025_pdf_dev",
        "issuer": "Adobe Inc.",
        "url": "https://www.sec.gov/Archives/edgar/data/796343/000079634326000045/adbe2025annualreporta.pdf",
    },
    {
        "document_id": "salesforce_fy2026_pdf_dev",
        "issuer": "Salesforce, Inc.",
        "url": "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001108524/124f523f-1f0b-4e13-8076-23cfd73951cf.pdf",
    },
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _download(source: dict[str, str], runtime_dir: Path, user_agent: str) -> Path:
    path = runtime_dir / f"{source['document_id']}.pdf"
    if path.exists() and path.stat().st_size:
        return path
    request = Request(source["url"], headers={"User-Agent": user_agent, "Accept": "application/pdf"})
    content = b""
    for attempt in range(3):
        try:
            with urlopen(request, timeout=300) as response:  # nosec B310: frozen official/issuer PDF URLs
                content = response.read()
            break
        except TimeoutError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    if not content.startswith(b"%PDF"):
        raise ValueError(f"source is not a PDF: {source['document_id']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _candidate_pages(document: object) -> list[int]:
    selected = []
    for index, page in enumerate(document):
        text = page.get_text("text")
        lowered = text.casefold()
        if "consolidated" in lowered and "statement" in lowered and len(YEAR_RE.findall(text)) >= 2:
            selected.append(index)
    return selected


def _page_lines_above(page: object, table_bbox: tuple[float, float, float, float]) -> list[str]:
    lines = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[:5]
        if y1 <= table_bbox[1] and table_bbox[1] - y1 <= 180 and x1 >= table_bbox[0] and x0 <= table_bbox[2]:
            lines.extend(str(text).splitlines())
    return lines


def _table_record(
    document_id: str,
    page_number: int,
    page: object,
    table: object,
    table_index: int,
    *,
    parser_name: str,
    parser_flavor: str | None = None,
) -> dict[str, object]:
    matrix = [[normalize_text(str(cell or "")) for cell in row] for row in table.extract()]
    width = max((len(row) for row in matrix), default=0)
    periods = resolve_period_headers(matrix, width)
    bbox = tuple(round(float(value), 3) for value in table.bbox)
    nearby_lines = _page_lines_above(page, bbox)
    matrix_context = " ".join(cell for row in matrix[:8] for cell in row)
    raw_scale, scale = extract_scale(matrix_context)
    scale_source = "table_header" if scale else None
    if not scale:
        raw_scale, scale = extract_scale(" ".join(nearby_lines))
        scale_source = "bbox_nearby_text" if scale else None
    statement = statement_from_lines(nearby_lines + [cell for row in matrix[:5] for cell in row])
    table_id = stable_identity("pdf-table-v2", document_id, page_number, bbox, matrix)
    rows = []
    numeric_cell_count = 0
    period_cell_count = 0
    geometry_cell_count = 0
    for row_index, matrix_row in enumerate(matrix):
        label = row_label(matrix_row)
        row_geometry = table.rows[row_index].cells if row_index < len(table.rows) else []
        cells = []
        for column_index, raw in enumerate(matrix_row):
            parsed = parse_number(raw)
            if parsed is None:
                continue
            numeric_cell_count += 1
            period = periods[column_index] if column_index < len(periods) else None
            period_cell_count += int(period is not None)
            cell_bbox = row_geometry[column_index] if column_index < len(row_geometry) else None
            geometry_valid = bool(cell_bbox) and bbox[0] - 1 <= cell_bbox[0] <= cell_bbox[2] <= bbox[2] + 1 and bbox[1] - 1 <= cell_bbox[1] <= cell_bbox[3] <= bbox[3] + 1
            geometry_cell_count += int(geometry_valid)
            cells.append(
                {
                    "cell_id": stable_identity("pdf-cell-v2", table_id, row_index, column_index, raw),
                    "column_index": column_index,
                    "period": period,
                    "raw_value_hash": hashlib.sha256(raw.encode()).hexdigest(),
                    "bbox": [round(float(value), 3) for value in cell_bbox] if cell_bbox else None,
                    "geometry_valid": geometry_valid,
                }
            )
        if cells:
            rows.append(
                {
                    "row_id": stable_identity("pdf-row-v2", table_id, row_index, matrix_row),
                    "row_index": row_index,
                    "metric": label,
                    "metric_present": bool(label),
                    "cells": cells,
                }
            )
    return {
        "table_id": table_id,
        "document_id": document_id,
        "pdf_page": page_number,
        "table_index": table_index,
        "parser_name": parser_name,
        "parser_flavor": parser_flavor,
        "bbox": bbox,
        "statement": statement,
        "scale": scale,
        "scale_context_hash": hashlib.sha256((raw_scale or "").encode()).hexdigest() if raw_scale else None,
        "scale_source": scale_source,
        "period_headers": periods,
        "row_count": len(rows),
        "numeric_cell_count": numeric_cell_count,
        "period_cell_count": period_cell_count,
        "geometry_cell_count": geometry_cell_count,
        "rows": rows,
    }


def _camelot_adapter(item: object, *, page_height: float) -> object:
    matrix = [[normalize_text(str(cell or "")) for cell in row] for row in item.df.values.tolist()]

    def convert(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = (float(value) for value in bbox)
        return x0, page_height - y1, x1, page_height - y0

    rows = []
    for row in item.cells:
        rows.append(SimpleNamespace(cells=[convert((cell.x1, cell.y1, cell.x2, cell.y2)) for cell in row]))
    return SimpleNamespace(
        extract=lambda: matrix,
        rows=rows,
        bbox=convert(tuple(getattr(item, "_bbox"))),
    )


def _camelot_fallback(pdf: Path, page_number: int, page: object) -> tuple[list[object], str | None]:
    import camelot

    parsed = list(camelot.read_pdf(str(pdf), pages=str(page_number), flavor="stream", edge_tol=50, row_tol=10))
    flavor = "stream"
    usable = [item for item in parsed if item.df.shape[0] >= 2 and item.df.shape[1] >= 2]
    if not usable:
        parsed = list(camelot.read_pdf(str(pdf), pages=str(page_number), flavor="lattice"))
        flavor = "lattice"
        usable = [item for item in parsed if item.df.shape[0] >= 2 and item.df.shape[1] >= 2]
    return [_camelot_adapter(item, page_height=float(page.rect.height)) for item in usable], flavor if usable else None


def run(args: argparse.Namespace) -> int:
    import pymupdf

    tables = []
    documents = []
    for source in SOURCES:
        path = _download(source, args.runtime_dir, args.user_agent)
        with pymupdf.open(path) as document:
            pages = _candidate_pages(document)
            document_tables = []
            for page_index in pages:
                page = document[page_index]
                found = list(page.find_tables().tables)
                parser_name = "pymupdf"
                parser_flavor = None
                if not found:
                    found, parser_flavor = _camelot_fallback(path, page_index + 1, page)
                    parser_name = "camelot" if found else parser_name
                document_tables.extend(
                    _table_record(
                        source["document_id"],
                        page_index + 1,
                        page,
                        table,
                        table_index,
                        parser_name=parser_name,
                        parser_flavor=parser_flavor,
                    )
                    for table_index, table in enumerate(found)
                )
        tables.extend(document_tables)
        documents.append(
            {
                "document_id": source["document_id"],
                "issuer": source["issuer"],
                "pdf_sha256": _sha(path),
                "pdf_bytes": path.stat().st_size,
                "selected_page_count": len(pages),
                "table_count": len(document_tables),
                "runtime_pdf_committed": False,
            }
        )
    rows = [row for table in tables for row in table["rows"]]
    cells = [cell for row in rows for cell in row["cells"]]
    metric_rows = sum(row["metric_present"] for row in rows)
    period_rows = sum(any(cell["period"] for cell in row["cells"]) for row in rows)
    statement_rows = sum(bool(table["statement"]) * table["row_count"] for table in tables)
    scale_rows = sum(bool(table["scale"]) * table["row_count"] for table in tables)
    geometry_cells = sum(cell["geometry_valid"] for cell in cells)
    runtime_structure = args.runtime_dir / "resolved-table-structures.json"
    _write(runtime_structure, {"tables": tables})
    coverage = {
        "document_count": len(documents),
        "table_count": len(tables),
        "pymupdf_table_count": sum(table["parser_name"] == "pymupdf" for table in tables),
        "camelot_stream_table_count": sum(table["parser_name"] == "camelot" and table["parser_flavor"] == "stream" for table in tables),
        "camelot_lattice_table_count": sum(table["parser_name"] == "camelot" and table["parser_flavor"] == "lattice" for table in tables),
        "row_count": len(rows),
        "numeric_cell_count": len(cells),
        "metric_row_count": metric_rows,
        "metric_row_coverage": metric_rows / len(rows) if rows else 0,
        "period_row_count": period_rows,
        "period_row_coverage": period_rows / len(rows) if rows else 0,
        "metric_period_row_count": sum(row["metric_present"] and any(cell["period"] for cell in row["cells"]) for row in rows),
        "metric_period_row_coverage": sum(row["metric_present"] and any(cell["period"] for cell in row["cells"]) for row in rows) / len(rows) if rows else 0,
        "statement_row_count": statement_rows,
        "statement_row_coverage": statement_rows / len(rows) if rows else 0,
        "scale_row_count": scale_rows,
        "scale_row_coverage": scale_rows / len(rows) if rows else 0,
        "geometry_cell_count": geometry_cells,
        "geometry_cell_coverage": geometry_cells / len(cells) if cells else 0,
    }
    thresholds = {
        "metric_row_coverage": 0.80,
        "period_row_coverage": 0.70,
        "metric_period_row_coverage": 0.70,
        "statement_row_coverage": 0.70,
        "geometry_cell_coverage": 0.70,
    }
    threshold_results = {
        name: coverage[name] >= minimum for name, minimum in thresholds.items()
    }
    gate_passed = all(threshold_results.values())
    acceptance = {
        "schema": "pdf-source-representation-v2/gate-a/acceptance/v1",
        "development_pdf_count": len(documents),
        "frozen_72_question_reads": 0,
        "gold_label_reads": 0,
        "model_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "resolver_structure_runtime_sha256": _sha(runtime_structure),
        "coverage_thresholds": thresholds,
        "coverage_threshold_results": threshold_results,
        "gate_passed": gate_passed,
        "decision": (
            "pdf_sr_v2_structure_coverage_validated"
            if gate_passed
            else "pdf_sr_v2_structure_resolver_coverage_blocked"
        ),
        "next_gate": (
            "pdf_sr_v2_row_cell_candidate_construction"
            if gate_passed
            else "stop_and_analyze_header_statement_lineage"
        ),
    }
    _write(args.out_dir / "development-pdf-manifest.json", {"documents": documents})
    _write(args.out_dir / "structure-coverage-report.json", coverage)
    _write(args.out_dir / "resolver-schema.json", {"table_fields": [key for key in tables[0] if key != "rows"] if tables else [], "runtime_structures_committed": False})
    _write(args.out_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": acceptance["next_gate"], "production_switch_allowed": False})
    _write(args.out_dir / "pdf-sr-v2-gate-a-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--user-agent", default="nano-finance-research contact@example.com")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
