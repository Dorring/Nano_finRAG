"""Attribute PDF SR-V2 header and statement-lineage failures without changing resolution."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from scripts.evaluation.run_pdf_sr_v2_gate_a import (
    DEFAULT_OUT as GATE_A_OUT,
    SOURCES,
    _camelot_fallback,
    _candidate_pages,
    _download,
    _page_lines_above,
    _write,
)
from src.evaluation.pdf_source_representation_v2 import (
    YEAR_RE,
    normalize_text,
    parse_number,
    resolve_period_headers,
    row_label,
    statement_from_lines,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-source-representation-v2-gate-a-r1"
EXTENDED_STATEMENT_RE = re.compile(
    r"(?i)(?:statements?|schedules?)\s+(?:of\s+)?(?:comprehensive\s+income|financial\s+position|changes\s+in\s+equity|revenues?)|segment\s+information|debt\s+maturities|consolidated\s+financial\s+statements"
)
NOTE_HEADING_RE = re.compile(r"(?i)^\s*note\s+\d+")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(table: object) -> list[list[str]]:
    if hasattr(table, "extract"):
        values = table.extract()
    else:
        values = table.df.values.tolist()
    return [[normalize_text(str(cell or "")) for cell in row] for row in values]


def _numeric_rows(matrix: list[list[str]]) -> int:
    return sum(any(parse_number(cell) is not None for cell in row) for row in matrix)


def _period_rows(matrix: list[list[str]]) -> int:
    width = max((len(row) for row in matrix), default=0)
    periods = resolve_period_headers(matrix, width)
    return sum(
        any(parse_number(cell) is not None and periods[index] for index, cell in enumerate(row))
        for row in matrix
    )


def _bbox_text(page: object, bbox: tuple[float, float, float, float]) -> str:
    return normalize_text(page.get_text("text", clip=bbox))


def _period_failure(matrix: list[list[str]], page: object, bbox: tuple[float, float, float, float], previous_text: str) -> str:
    matrix_text = " ".join(cell for row in matrix for cell in row)
    matrix_years = YEAR_RE.findall(matrix_text)
    if matrix_years:
        header_rows = [row for row in matrix[:12] if any(YEAR_RE.search(cell) for cell in row)]
        if len(header_rows) > 1:
            return "period_multilevel_header"
        if len(set(matrix_years)) == 1:
            return "period_single_year_global_header"
        return "period_year_in_matrix_unresolved"
    if YEAR_RE.search(_bbox_text(page, bbox)):
        return "period_year_in_table_bbox_text"
    nearby = " ".join(_page_lines_above(page, bbox))
    if YEAR_RE.search(nearby):
        return "period_year_in_nearby_page_text"
    if YEAR_RE.search(previous_text):
        return "period_header_on_previous_page"
    page_text = page.get_text("text")
    if re.search(r"(?i)(?:three|six|nine|twelve)\s+months?\s+ended|year\s+ended|as\s+of", page_text):
        return "period_non_annual_date"
    if not YEAR_RE.search(page_text):
        return "period_text_absent_from_native_pdf"
    return "period_unknown"


def _statement_failure(matrix: list[list[str]], page: object, bbox: tuple[float, float, float, float], previous_text: str) -> str:
    page_text = page.get_text("text")
    nearby = _page_lines_above(page, bbox)
    table_head = [cell for row in matrix[:8] for cell in row]
    if statement_from_lines(page_text.splitlines()):
        if not statement_from_lines(nearby + table_head):
            return "statement_marker_on_page_but_outside_180"
    table_text = " ".join(table_head)
    if EXTENDED_STATEMENT_RE.search(table_text):
        return "statement_marker_in_table_header"
    if EXTENDED_STATEMENT_RE.search(page_text):
        return "statement_unrecognized_title"
    if statement_from_lines(previous_text.splitlines()) or EXTENDED_STATEMENT_RE.search(previous_text):
        return "statement_on_previous_page"
    if re.search(r"(?i)continued", page_text):
        return "statement_continuation_table"
    if any(NOTE_HEADING_RE.search(line) for line in page_text.splitlines()):
        return "statement_note_heading"
    if not re.search(r"(?i)statement|schedule|segment|maturit|note\s+\d+", page_text):
        return "statement_text_absent_from_native_pdf"
    return "statement_unknown"


def _parser_summary(tables: list[object]) -> dict[str, int | float]:
    matrices = [_matrix(table) for table in tables]
    numeric_rows = sum(_numeric_rows(matrix) for matrix in matrices)
    period_rows = sum(_period_rows(matrix) for matrix in matrices)
    metric_rows = sum(
        sum(bool(row_label(row)) and any(parse_number(cell) is not None for cell in row) for row in matrix)
        for matrix in matrices
    )
    return {
        "table_count": len(tables),
        "numeric_row_count": numeric_rows,
        "metric_row_count": metric_rows,
        "period_row_count": period_rows,
        "period_row_coverage": period_rows / numeric_rows if numeric_rows else 0,
    }


def run(args: argparse.Namespace) -> int:
    import camelot
    import pymupdf

    gate_a_acceptance = GATE_A_OUT / "pdf-sr-v2-gate-a-acceptance.json"
    baseline = json.loads(gate_a_acceptance.read_text(encoding="utf-8"))
    if baseline["decision"] != "pdf_sr_v2_structure_resolver_coverage_blocked":
        raise RuntimeError("Gate A must be blocked before R1 attribution")

    period_counts: Counter[str] = Counter()
    statement_counts: Counter[str] = Counter()
    by_document: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: {"period": Counter(), "statement": Counter()})
    by_parser: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: {"period": Counter(), "statement": Counter()})
    comparison_records = []
    failed_page_count = 0

    for source in SOURCES:
        pdf = _download(source, args.runtime_dir, args.user_agent)
        with pymupdf.open(pdf) as document:
            for page_index in _candidate_pages(document):
                page = document[page_index]
                primary = list(page.find_tables().tables)
                baseline_tables = primary
                parser = "pymupdf"
                if not baseline_tables:
                    baseline_tables, flavor = _camelot_fallback(pdf, page_index + 1, page)
                    parser = f"camelot_{flavor}" if flavor else "none"
                previous_text = document[page_index - 1].get_text("text") if page_index else ""
                page_failed = False
                for table in baseline_tables:
                    matrix = _matrix(table)
                    bbox = tuple(float(value) for value in table.bbox)
                    numeric_rows = _numeric_rows(matrix)
                    if not numeric_rows:
                        continue
                    missing_period_rows = numeric_rows - _period_rows(matrix)
                    if missing_period_rows:
                        reason = _period_failure(matrix, page, bbox, previous_text)
                        period_counts[reason] += missing_period_rows
                        by_document[source["document_id"]]["period"][reason] += missing_period_rows
                        by_parser[parser]["period"][reason] += missing_period_rows
                        page_failed = True
                    nearby = _page_lines_above(page, bbox)
                    if not statement_from_lines(nearby + [cell for row in matrix[:5] for cell in row]):
                        reason = _statement_failure(matrix, page, bbox, previous_text)
                        statement_counts[reason] += numeric_rows
                        by_document[source["document_id"]]["statement"][reason] += numeric_rows
                        by_parser[parser]["statement"][reason] += numeric_rows
                        page_failed = True
                if not page_failed:
                    continue
                failed_page_count += 1
                stream = list(camelot.read_pdf(str(pdf), pages=str(page_index + 1), flavor="stream", edge_tol=50, row_tol=10))
                stream = [item for item in stream if item.df.shape[0] >= 2 and item.df.shape[1] >= 2]
                lattice = list(camelot.read_pdf(str(pdf), pages=str(page_index + 1), flavor="lattice"))
                lattice = [item for item in lattice if item.df.shape[0] >= 2 and item.df.shape[1] >= 2]
                comparison_records.append(
                    {
                        "document_id": source["document_id"],
                        "pdf_page": page_index + 1,
                        "pymupdf": _parser_summary(primary),
                        "camelot_stream": _parser_summary(stream),
                        "camelot_lattice": _parser_summary(lattice),
                    }
                )

    period_total = sum(period_counts.values())
    statement_total = sum(statement_counts.values())
    native_absent = period_counts["period_text_absent_from_native_pdf"] + statement_counts["statement_text_absent_from_native_pdf"]
    missing_total = period_total + statement_total
    mineru_candidate_rate = native_absent / missing_total if missing_total else 0
    acceptance = {
        "schema": "pdf-source-representation-v2/gate-a-r1/acceptance/v1",
        "source_gate_a_sha256": _sha(gate_a_acceptance),
        "failed_page_count": failed_page_count,
        "period_failure_row_count": period_total,
        "statement_failure_row_count": statement_total,
        "native_text_absence_rate": mineru_candidate_rate,
        "primary_bottleneck": ["header_resolution", "statement_lineage", "parser_result_arbitration"],
        "mineru_required_now": False,
        "mineru_targeted_fallback_possible": mineru_candidate_rate >= 0.10,
        "resolver_behavior_changed": False,
        "recall_evaluation_run": False,
        "frozen_72_question_reads": 0,
        "gold_label_reads": 0,
        "model_calls": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
        "decision": "pdf_sr_v2_native_resolver_failure_attributed",
        "next_gate": "header_grid_and_statement_lineage_resolver",
    }
    _write(args.out_dir / "period-failure-attribution.json", {"counts": dict(period_counts), "by_document": {key: dict(value["period"]) for key, value in by_document.items()}, "by_parser": {key: dict(value["period"]) for key, value in by_parser.items()}})
    _write(args.out_dir / "statement-failure-attribution.json", {"counts": dict(statement_counts), "by_document": {key: dict(value["statement"]) for key, value in by_document.items()}, "by_parser": {key: dict(value["statement"]) for key, value in by_parser.items()}})
    parser_totals: dict[str, Counter[str]] = defaultdict(Counter)
    for record in comparison_records:
        for parser_name in ("pymupdf", "camelot_stream", "camelot_lattice"):
            summary = record[parser_name]
            parser_totals[parser_name]["page_with_table_count"] += int(summary["table_count"] > 0)
            for field in ("table_count", "numeric_row_count", "metric_row_count", "period_row_count"):
                parser_totals[parser_name][field] += int(summary[field])
    _write(
        args.out_dir / "failed-page-parser-comparison.json",
        {
            "failed_page_count": failed_page_count,
            "parser_totals": {key: dict(value) for key, value in parser_totals.items()},
            "records": comparison_records,
        },
    )
    _write(args.out_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": acceptance["next_gate"], "production_switch_allowed": False})
    _write(args.out_dir / "pdf-sr-v2-gate-a-r1-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--user-agent", default="nano-finance-research contact@example.com")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
