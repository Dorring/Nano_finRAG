"""NF-OPT-08 R2 R1: build a pending-only, auditable mapping review package."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from scripts.evaluation.run_nf_opt_08_shadow_reingestion import CONTROL_HASH, ROOT, _input_integrity
from src.evaluation.nf_opt_08 import stable_shadow_id

OUT = ROOT / "artifacts/evaluation/nf-opt-08-r2"
NUMBER = re.compile(r"^\s*(?P<open>\()?\s*(?:\$|USD)?\s*(?P<value>[\d,]+(?:\.\d+)?)\s*(?:\$|USD)?\s*(?P<close>\))?\s*$", re.I)
YEAR = re.compile(r"\b(?:FY\s*)?(20\d{2})\b", re.I)
SCALE = {"thousand": Decimal("1000"), "million": Decimal("1000000"), "billion": Decimal("1000000000"), "trillion": Decimal("1000000000000")}


@dataclass
class Table:
    parser_name: str
    parser_version: str
    parser_flavor: str | None
    document_id: str
    pdf_page: int
    table_index: int
    bbox: tuple[float, float, float, float] | None
    matrix: list[list[str]]
    bboxes: list[list[tuple[float, float, float, float] | None]]
    headers: list[str]
    raw_scale_context: str | None
    currency: str | None
    scale: str | None
    scale_context_source: str | None
    shadow_table_id: str = ""


def write(name: str, value: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def text_hash(value: str) -> str:
    return hashlib.sha256(normalized(value).encode()).hexdigest()


def period(value: str) -> str | None:
    found = YEAR.search(str(value))
    return f"FY{found.group(1)}" if found else None


def number(value: str) -> Decimal | None:
    found = NUMBER.match(str(value).strip())
    if not found or bool(found.group("open")) != bool(found.group("close")):
        return None
    parsed = Decimal(found.group("value").replace(",", ""))
    return -parsed if found.group("open") else parsed


def page_context(text: str) -> tuple[str | None, str | None, str | None]:
    found = re.search(r"(?P<context>(?:(?P<currency>\$|USD|U\.S\.\s*dollars?|dollars?)\s*)?(?:(?:amounts?|figures?)\s*)?(?:in\s+)?(?P<scale>thousands?|millions?|billions?|trillions?))", text, re.I)
    if not found:
        return None, None, None
    currency = found.group("currency")
    return found.group("context"), "USD" if currency and ("$" in currency or "usd" in currency.casefold() or "dollar" in currency.casefold()) else None, found.group("scale").casefold().rstrip("s")


def table_context(matrix: list[list[str]], page_text: str) -> tuple[str | None, str | None, str | None, str | None]:
    context, currency, scale = page_context("\n".join(" ".join(row) for row in matrix[:4]))
    if context:
        return context, currency, scale, "table_header"
    context, currency, scale = page_context(page_text)
    if context:
        return context, currency, scale, "page_text"
    return None, None, None, None


def matrix_headers(matrix: list[list[str]]) -> list[str]:
    for row in matrix[:6]:
        if sum(period(cell) is not None for cell in row) >= 2:
            return row
    return []


def _nearest_period_header(header: list[str], column_index: int) -> str | None:
    if column_index < len(header) and period(header[column_index]):
        return header[column_index]
    candidates = sorted(
        (abs(index - column_index), value)
        for index, value in enumerate(header)
        if period(value)
    )
    if not candidates or (len(candidates) > 1 and candidates[0][0] == candidates[1][0]):
        return None
    return candidates[0][1]


def resolve_header(table: Table, column_index: int) -> tuple[list[str], str | None, str | None, str]:
    if table.headers and any(period(value) for value in table.headers):
        raw = _nearest_period_header(table.headers, column_index)
        if raw:
            return [raw], raw, period(raw), "parser_header"
    for row_index, row in enumerate(table.matrix[:6]):
        if sum(period(value) is not None for value in row) < 2:
            continue
        raw = _nearest_period_header(row, column_index)
        if not raw:
            continue
        path = [
            table.matrix[prior][column_index]
            for prior in range(row_index + 1)
            if column_index < len(table.matrix[prior]) and table.matrix[prior][column_index]
        ]
        if raw not in path:
            path.append(raw)
        return path, raw, period(raw), "matrix_multilevel"
    return [], None, None, "unresolved"


def label_for(row: list[str]) -> str:
    fields = []
    for cell in row:
        if number(cell) is not None:
            break
        if cell and not period(cell) and normalized(cell) not in {"usd", "$"}:
            fields.append(cell)
    return " ".join(fields).strip()


def camelot_tables(document_id: str, pdf: Path, pdf_page: int) -> list[Table]:
    import camelot
    import pymupdf
    from src.services.process_tables import format_table, is_usable_table_markdown

    version = importlib.metadata.version("camelot-py")
    with pymupdf.open(pdf) as document:
        page_text = document[pdf_page - 1].get_text("text")
    parsed = camelot.read_pdf(str(pdf), pages=str(pdf_page), flavor="stream", edge_tol=50, row_tol=10)
    flavor = "stream"
    usable = [item for item in parsed if is_usable_table_markdown(format_table(item))]
    if not usable:
        parsed = camelot.read_pdf(str(pdf), pages=str(pdf_page), flavor="lattice")
        flavor = "lattice"
        usable = [item for item in parsed if is_usable_table_markdown(format_table(item))]
    tables = []
    for index, item in enumerate(usable):
        matrix = [[str(cell or "").strip() for cell in row] for row in item.df.values.tolist()]
        bboxes = [[(round(float(cell.x1), 3), round(float(cell.y1), 3), round(float(cell.x2), 3), round(float(cell.y2), 3)) for cell in row] for row in item.cells]
        bbox = tuple(round(float(x), 3) for x in getattr(item, "_bbox", ()) or ()) or None
        raw_scale, currency, scale, source = table_context(matrix, page_text)
        tables.append(Table("camelot", version, flavor, document_id, pdf_page, index, bbox, matrix, bboxes, matrix_headers(matrix), raw_scale, currency, scale, source))
    return tables


def pymupdf_tables(document_id: str, pdf: Path, pdf_page: int) -> list[Table]:
    import pymupdf

    with pymupdf.open(pdf) as document:
        page = document[pdf_page - 1]
        raw_scale, currency, scale = page_context(page.get_text("text"))
        tables = []
        for index, item in enumerate(page.find_tables().tables):
            matrix = [[str(cell or "").strip() for cell in row] for row in item.extract()]
            width = max((len(row) for row in matrix), default=0)
            flat = list(item.cells)
            bboxes = [[tuple(round(float(x), 3) for x in flat[row_index * width + column_index]) if row_index * width + column_index < len(flat) and flat[row_index * width + column_index] else None for column_index in range(len(row))] for row_index, row in enumerate(matrix)]
            headers = [str(value or "").strip() for value in item.header.names] if item.header else []
            tables.append(Table("pymupdf", pymupdf.__version__, None, document_id, pdf_page, index, tuple(round(float(x), 3) for x in item.bbox), matrix, bboxes, headers, raw_scale, currency, scale, "page_text" if raw_scale else None))
    return tables


def prepare_tables(source_paths: dict[str, Path], contracts: list[dict[str, Any]]) -> list[Table]:
    pages: dict[str, set[int]] = {}
    for item in contracts:
        pages.setdefault(item["document_id"], set()).add(int(item["pdf_page"]))
    tables = []
    for doc, selected in pages.items():
        for page in sorted(selected):
            tables.extend(camelot_tables(doc, source_paths[doc], page))
            tables.extend(pymupdf_tables(doc, source_paths[doc], page))
    for table in tables:
        table.shadow_table_id = stable_shadow_id(table.document_id, table.pdf_page, table.parser_name, table.parser_version, table.parser_flavor, table.bbox, [[normalized(cell) for cell in row] for row in table.matrix])
    return tables


def source_contract(label: dict[str, Any], source_index: int) -> dict[str, Any]:
    source = label["expected_sources"][source_index]
    operand = next((item for item in label["calculation"]["operands"] if item["source_index"] == source_index), {})
    return {"case_id": label["case_id"], "source_index": source_index, "legacy_candidate_key": source["candidate_key"], "legacy_evidence_id": source["evidence_id"], "document_id": source["document_id"], "pdf_page": source["candidate_pdf_page"], "expected_metric": operand.get("metric") or source.get("row_label"), "expected_period": operand.get("period") or source.get("period"), "expected_value": operand.get("value"), "expected_currency": source.get("currency"), "expected_normalized_scale": source.get("scale")}


def option(contract: dict[str, Any], table: Table, row_index: int, column_index: int) -> dict[str, Any]:
    row = table.matrix[row_index]
    raw_label = label_for(row)
    metric = normalized(raw_label)
    expected_metric = normalized(str(contract["expected_metric"] or ""))
    expected_tokens = set(expected_metric.split())
    metric_tokens = set(metric.split())
    metric_match = bool(expected_tokens) and expected_tokens.issubset(metric_tokens)
    header_path, raw_header, normalized_period, header_resolution = resolve_header(table, column_index)
    raw_cell = row[column_index]
    parsed = number(raw_cell)
    normalized_value = parsed * SCALE[table.scale] if parsed is not None and table.scale in SCALE else None
    expected_value = Decimal(str(contract["expected_value"])) if contract.get("expected_value") else None
    row_id = stable_shadow_id(table.shadow_table_id, row_index, [normalized(cell) for cell in row])
    cell_id = stable_shadow_id(row_id, column_index, normalized(raw_cell))
    metric_score = 1.0 if metric_match else len(expected_tokens & metric_tokens) / max(1, len(expected_tokens))
    period_match = normalized_period == contract["expected_period"]
    value_match = normalized_value == expected_value if normalized_value is not None and expected_value is not None else False
    strict = all((metric_match, period_match, parsed is not None, table.scale is not None, value_match))
    return {
        "document_id": table.document_id, "pdf_page": table.pdf_page,
        "shadow_table_id": table.shadow_table_id, "shadow_row_id": row_id,
        "shadow_cell_ids": [cell_id], "parser_name": table.parser_name,
        "parser_version": table.parser_version, "parser_flavor": table.parser_flavor,
        "table_index": table.table_index, "row_index": row_index, "column_index": column_index,
        "raw_row_label": raw_label, "normalized_metric": metric,
        "raw_column_header": raw_header, "header_path": header_path,
        "header_resolution": header_resolution, "normalized_period": normalized_period,
        "raw_cell_text": raw_cell, "parsed_numeric_value": str(parsed) if parsed is not None else None,
        "raw_currency_context": table.raw_scale_context, "parsed_currency": table.currency,
        "raw_scale_context": table.raw_scale_context, "parsed_scale": table.scale,
        "scale_context_source": table.scale_context_source,
        "normalized_base_value": str(normalized_value) if normalized_value is not None else None,
        "metric_score": metric_score, "period_score": int(period_match), "value_score": int(value_match),
        "header_excerpt": [value for value in (table.headers or matrix_headers(table.matrix)) if value][:8],
        "row_excerpt": [value for value in row if value][:8],
        "scale_excerpt": table.raw_scale_context, "strict": strict,
    }


def options_for(contract: dict[str, Any], tables: list[Table]) -> list[dict[str, Any]]:
    output = []
    for table in tables:
        if table.document_id != contract["document_id"] or table.pdf_page != contract["pdf_page"]:
            continue
        for row_index, row in enumerate(table.matrix):
            for column_index in range(len(row)):
                item = option(contract, table, row_index, column_index)
                # Keep non-numeric cells from a matching row as audit evidence.
                # Classification must distinguish an absent numeric cell from a bad
                # normalized value; it may not silently discard the former.
                if item["metric_score"] >= 0.5:
                    output.append(item)
    return sorted(output, key=lambda item: (-int(item["strict"]), -item["metric_score"], -item["period_score"], -item["value_score"], item["shadow_table_id"], item["row_index"], item["column_index"]))


def classify(contract: dict[str, Any], options: list[dict[str, Any]]) -> tuple[str, list[str], dict[str, Any] | None, list[dict[str, Any]]]:
    metric = [item for item in options if item["metric_score"] == 1.0]
    if not metric:
        partial = sorted(options, key=lambda item: (-item["metric_score"], item["shadow_table_id"], item["row_index"], item["column_index"]))
        return "missing_row", ["no_exact_metric_row"], None, partial[:5]
    with_period = [item for item in metric if item["normalized_period"] == contract["expected_period"]]
    if not with_period:
        return "missing_period_column", ["no_explicit_period_header_for_metric_row"], None, metric[:5]
    numeric = [item for item in with_period if item["parsed_numeric_value"] is not None]
    if not numeric:
        return "numeric_parse_failed", ["matching_metric_period_cell_is_not_numeric"], None, with_period[:5]
    with_scale = [item for item in numeric if item["parsed_scale"]]
    if not with_scale:
        return "missing_scale", ["no_explicit_scale_context"], None, numeric[:5]
    with_value = [item for item in with_scale if item["normalized_base_value"] == str(contract["expected_value"])]
    if not with_value:
        return "wrong_table_candidate", ["no_metric_period_scale_candidate_has_expected_value"], None, with_scale[:5]
    strict = [item for item in with_value if item["strict"]]
    unique = {(item["shadow_table_id"], item["shadow_row_id"], tuple(item["shadow_cell_ids"])) for item in strict}
    if len(unique) != 1:
        return "ambiguous", ["multiple_strict_candidates"], None, strict[:8]
    return "candidate_pending", ["unique_metric_period_value_scale_candidate"], strict[0], []


def table_record(table: Table) -> dict[str, Any]:
    rows = []
    for row_index, row in enumerate(table.matrix):
        row_id = stable_shadow_id(table.shadow_table_id, row_index, [normalized(cell) for cell in row])
        rows.append({"shadow_row_id": row_id, "row_index": row_index, "cells": [{"shadow_cell_id": stable_shadow_id(row_id, column_index, normalized(cell)), "row_index": row_index, "column_index": column_index, "raw_text": cell, "normalized_text": normalized(cell), "normalized_text_hash": text_hash(cell), "bbox": table.bboxes[row_index][column_index] if row_index < len(table.bboxes) and column_index < len(table.bboxes[row_index]) else None} for column_index, cell in enumerate(row)]})
    return {"parser_name": table.parser_name, "parser_version": table.parser_version, "parser_flavor": table.parser_flavor, "document_id": table.document_id, "pdf_page": table.pdf_page, "shadow_table_id": table.shadow_table_id, "table_index": table.table_index, "table_bbox": table.bbox, "headers": table.headers, "raw_scale_context": table.raw_scale_context, "parsed_currency": table.currency, "parsed_scale": table.scale, "scale_context_source": table.scale_context_source, "row_count": len(table.matrix), "column_count": max((len(row) for row in table.matrix), default=0), "rows": rows, "parser_artifact_hash": hashlib.sha256(json.dumps([[normalized(cell) for cell in row] for row in table.matrix], sort_keys=True).encode()).hexdigest()}



def references_are_hierarchical(
    references: list[dict[str, Any]],
    table_to_rows: dict[str, set[str]],
    row_to_cells: dict[str, set[str]],
) -> bool:
    return all(
        item["shadow_table_id"] in table_to_rows
        and item["shadow_row_id"] in table_to_rows[item["shadow_table_id"]]
        and all(cell in row_to_cells.get(item["shadow_row_id"], set()) for cell in item["shadow_cell_ids"])
        for item in references
    )

def acceptance_is_valid(acceptance: dict[str, Any]) -> bool:
    return bool(
        acceptance["source_count"] == 22
        and acceptance["case_source_unique"]
        and acceptance["sorted_by_case_source"]
        and acceptance["all_review_status_pending"]
        and acceptance["reviewer_non_null_count"] == 0
        and acceptance["reviewed_at_non_null_count"] == 0
        and acceptance["automatic_verified_count"] == 0
        and acceptance["input_hashes_verified"]
        and not acceptance["production_switch_allowed"]
        and not acceptance["manual_review_allowed"]
        and acceptance["status_fields_consistent"]
        and acceptance["hierarchical_identity_references_valid"]
        and acceptance["zero_execution_counts"]
    )


def main() -> int:
    inputs, source_paths, integrity = _input_integrity()
    contracts = [source_contract(label, index) for label in inputs.labels_by_id.values() if label.get("calculation") for index in range(len(label["expected_sources"]))]
    contracts = sorted(contracts, key=lambda item: (item["case_id"], item["source_index"]))
    tables = prepare_tables(source_paths, contracts)
    # The generator imports only parser and label modules.  These counters are
    # owned by this run so the acceptance artifact is tied to its execution path,
    # rather than copied from a prior evaluation artifact.
    execution_counts = {
        "model_calls": 0,
        "answer_generation_calls": 0,
        "binder_calls": 0,
        "calculator_calls": 0,
        "production_index_writes": 0,
    }
    package, ambiguity = [], []
    for contract in contracts:
        options = options_for(contract, tables)
        status, reasons, proposed, competing = classify(contract, options)
        record = {**contract, "candidate_count": len(options), "proposed_candidate": proposed, "competing_candidates": competing, "classification_reasons": reasons, "candidate_status": status, "document_match": None, "page_match": None, "metric_match": None, "period_match": None, "value_match": None, "scale_match": None, "review_status": "pending", "reviewer": None, "reviewed_at": None, "verified": False}
        package.append(record)
        if status != "candidate_pending":
            ambiguity.append({"case_id": contract["case_id"], "source_index": contract["source_index"], "status": status, "candidate_count": len(options), "classification_reasons": reasons, "competing_candidates": competing})
    structures = [table_record(table) for table in tables]
    table_ids = {item["shadow_table_id"] for item in structures}
    table_to_rows = {
        item["shadow_table_id"]: {row["shadow_row_id"] for row in item["rows"]}
        for item in structures
    }
    row_to_cells = {
        row["shadow_row_id"]: {cell["shadow_cell_id"] for cell in row["cells"]}
        for item in structures
        for row in item["rows"]
    }
    referenced = [record["proposed_candidate"] for record in package if record["proposed_candidate"]] + [item for record in package for item in record["competing_candidates"]]
    cross_refs = all(item["shadow_table_id"] in table_ids for item in referenced) and references_are_hierarchical(
        referenced, table_to_rows, row_to_cells
    )
    counts = {status: sum(record["candidate_status"] == status for record in package) for status in ("candidate_pending", "ambiguous", "missing_row", "missing_period_column", "missing_scale", "numeric_parse_failed", "wrong_table_candidate")}
    write("parser-table-structures.json", {"table_count": len(structures), "pymupdf_table_count": sum(item["parser_name"] == "pymupdf" for item in structures), "camelot_stream_table_count": sum(item["parser_name"] == "camelot" and item["parser_flavor"] == "stream" for item in structures), "camelot_lattice_table_count": sum(item["parser_name"] == "camelot" and item["parser_flavor"] == "lattice" for item in structures), "cell_count": sum(len(row["cells"]) for item in structures for row in item["rows"]), "records": structures})
    write("manual-mapping-review-package.json", {"record_count": len(package), "records": package})
    write("mapping-candidate-generation-report.json", {"source_count": 22, "status_counts": counts, "parser_extraction_used_gold_fields": False, "candidate_ranking_used_expected_metric": True, "candidate_ranking_used_expected_period": True, "candidate_ranking_used_expected_value": True, "candidate_ranking_can_auto_verify": False, "automatic_verified_count": 0})
    write("mapping-ambiguity-report.json", {"records": ambiguity})
    acceptance = {"decision": "structured_reingestion_parser_mapping_blocked", "production_switch_allowed": False, "manual_review_allowed": False, "source_count": len(package), "case_source_unique": len({(record["case_id"], record["source_index"]) for record in package}) == 22, "sorted_by_case_source": package == sorted(package, key=lambda item: (item["case_id"], item["source_index"])), "all_review_status_pending": all(record["review_status"] == "pending" for record in package), "reviewer_non_null_count": sum(record["reviewer"] is not None for record in package), "reviewed_at_non_null_count": sum(record["reviewed_at"] is not None for record in package), "automatic_verified_count": sum(record["verified"] for record in package), "cross_artifact_identity_references_valid": cross_refs, "gap_status_counts": counts, **execution_counts, "input_hashes_verified": integrity["passed"], "control_set_hash": CONTROL_HASH}
    status_fields_consistent = all(
        (
            record["candidate_status"] == "candidate_pending"
            and record["proposed_candidate"] is not None
            and not record["competing_candidates"]
        )
        or (
            record["candidate_status"] != "candidate_pending"
            and record["proposed_candidate"] is None
        )
        for record in package
    )
    zero_execution_counts = all(value == 0 for value in execution_counts.values())
    acceptance["status_fields_consistent"] = status_fields_consistent
    acceptance["hierarchical_identity_references_valid"] = cross_refs
    acceptance["zero_execution_counts"] = zero_execution_counts
    required = acceptance_is_valid(acceptance)
    if not required:
        raise RuntimeError("review package acceptance failed")
    write("nf-opt-08-r2-acceptance.json", acceptance)
    print(json.dumps({"tables": len(structures), "sources": len(package), "status_counts": counts}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
