"""Build pending-only NF-OPT-08 R2 manual mapping review material."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from scripts.evaluation.run_nf_opt_08_shadow_reingestion import (
    CONTROL_HASH,
    ROOT,
    _input_integrity,
)
from src.evaluation.nf_opt_08 import stable_shadow_id

OUT = ROOT / "artifacts/evaluation/nf-opt-08-r2"
SCALE = {"thousand": Decimal("1000"), "million": Decimal("1000000"), "billion": Decimal("1000000000"), "trillion": Decimal("1000000000000")}
NUMBER = re.compile(r"^\(?\$?\s*([\d,]+(?:\.\d+)?)\)?$")
YEAR = re.compile(r"\b(?:FY\s*)?(20\d{2})\b", re.I)


@dataclass
class Table:
    parser: str
    document_id: str
    page: int
    index: int
    bbox: tuple[float, float, float, float] | None
    matrix: list[list[str]]
    headers: list[str]
    scale_text: str | None
    currency: str | None
    scale: str | None
    table_id: str = ""


def write(name: str, value: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def period(value: str) -> str | None:
    found = YEAR.search(value)
    return f"FY{found.group(1)}" if found else None


def numeric(value: str) -> Decimal | None:
    found = NUMBER.match(value.strip())
    if not found:
        return None
    amount = Decimal(found.group(1).replace(",", ""))
    return -amount if value.strip().startswith("(") else amount


def context(page_text: str) -> tuple[str | None, str | None, str | None]:
    match = re.search(r"(?P<currency>\$|USD)?\s*(?:in|\$ in)\s+(?P<scale>thousands?|millions?|billions?|trillions?)", page_text, re.I)
    if not match:
        return None, None, None
    raw = match.group(0)
    scale = match.group("scale").casefold().rstrip("s")
    return raw, "USD" if match.group("currency") == "$" else None, scale


def row_label(row: list[str]) -> str:
    labels = []
    for cell in row:
        if numeric(cell) is not None:
            break
        if cell.strip() and not period(cell):
            labels.append(cell.strip())
    return " ".join(labels) or (row[0].strip() if row else "")


def headers_from_matrix(matrix: list[list[str]]) -> list[str]:
    for row in matrix[:3]:
        if sum(period(cell) is not None for cell in row) >= 2:
            return row
    return []


def parse_tables(document_id: str, pdf_path: Path, pages: list[int]) -> list[Table]:
    import camelot
    import pymupdf
    from src.services.process_tables import format_table, is_usable_table_markdown

    out: list[Table] = []
    wanted = ",".join(str(page) for page in sorted(set(pages)))
    camelot_tables = camelot.read_pdf(str(pdf_path), pages=wanted, flavor="stream", edge_tol=50, row_tol=10)
    if len(camelot_tables) == 0:
        camelot_tables = camelot.read_pdf(str(pdf_path), pages=wanted, flavor="lattice")
    by_page: dict[int, int] = {}
    for table in camelot_tables:
        markdown = format_table(table)
        if not is_usable_table_markdown(markdown):
            continue
        page = int(table.page)
        matrix = [[str(cell or "").strip() for cell in row] for row in table.df.values.tolist()]
        index = by_page.get(page, 0)
        by_page[page] = index + 1
        bbox = tuple(round(float(x), 3) for x in getattr(table, "_bbox", ()) or ()) or None
        out.append(Table("camelot", document_id, page, index, bbox, matrix, headers_from_matrix(matrix), None, None, None))
    with pymupdf.open(pdf_path) as pdf:
        for page_no in sorted(set(pages)):
            page = pdf[page_no - 1]
            raw_scale, currency, scale = context(page.get_text("text"))
            for index, table in enumerate(page.find_tables().tables):
                matrix = [[str(cell or "").strip() for cell in row] for row in table.extract()]
                headers = [str(name or "").strip() for name in table.header.names] if table.header else []
                bbox = tuple(round(float(x), 3) for x in table.bbox)
                out.append(Table("pymupdf", document_id, page_no, index, bbox, matrix, headers, raw_scale, currency, scale))
    for table in out:
        matrix = [[norm(cell) for cell in row] for row in table.matrix]
        table.table_id = stable_shadow_id(table.document_id, table.page, table.parser, table.bbox, matrix)
    return out


def source_contract(label: dict[str, Any], source_index: int) -> dict[str, Any]:
    source = label["expected_sources"][source_index]
    operand = next((x for x in label["calculation"]["operands"] if x["source_index"] == source_index), {})
    return {
        "case_id": label["case_id"], "source_index": source_index,
        "legacy_candidate_key": source["candidate_key"], "legacy_evidence_id": source["evidence_id"],
        "document_id": source["document_id"], "pdf_page": source["candidate_pdf_page"],
        "expected_metric": operand.get("metric") or source.get("row_label"),
        "expected_period": operand.get("period") or source.get("period"),
        "expected_value": operand.get("value"), "expected_currency": source.get("currency"),
        "expected_normalized_scale": source.get("scale"),
    }


def candidates(contract: dict[str, Any], tables: list[Table]) -> list[dict[str, Any]]:
    expected = norm(str(contract["expected_metric"] or ""))
    expected_tokens = set(expected.split())
    expected_period = str(contract["expected_period"] or "")
    expected_value = Decimal(str(contract["expected_value"])) if contract.get("expected_value") else None
    options = []
    for table in tables:
        if table.document_id != contract["document_id"] or table.page != contract["pdf_page"]:
            continue
        headers = table.headers or headers_from_matrix(table.matrix)
        for row_index, row in enumerate(table.matrix):
            label = row_label(row)
            tokens = set(norm(label).split())
            metric_score = len(tokens & expected_tokens) / max(1, len(expected_tokens))
            if metric_score < 0.5:
                continue
            for column_index, cell in enumerate(row):
                raw = cell.strip()
                value = numeric(raw)
                if value is None:
                    continue
                header = headers[column_index] if column_index < len(headers) else ""
                found_period = period(header)
                normalized = value * SCALE[table.scale] if table.scale in SCALE else None
                value_score = int(normalized == expected_value) if normalized is not None and expected_value is not None else 0
                period_score = int(found_period == expected_period)
                row_id = stable_shadow_id(table.table_id, row_index, [norm(x) for x in row])
                cell_id = stable_shadow_id(row_id, column_index, norm(raw))
                options.append({
                    "score": metric_score + period_score + value_score,
                    "shadow_table_id": table.table_id, "shadow_row_id": row_id, "shadow_cell_ids": [cell_id],
                    "parser_name": table.parser, "table_index": table.index, "row_index": row_index, "column_index": column_index,
                    "raw_row_label": label, "normalized_metric": norm(label),
                    "raw_column_header": header, "normalized_period": found_period,
                    "raw_cell_text": raw, "parsed_numeric_value": str(value),
                    "raw_currency_context": table.scale_text, "parsed_currency": table.currency,
                    "raw_scale_context": table.scale_text, "parsed_scale": table.scale,
                    "normalized_base_value": str(normalized) if normalized is not None else None,
                    "header_excerpt": [x for x in headers if x][:6],
                    "row_excerpt": [x for x in row if x][:8],
                    "scale_excerpt": table.scale_text,
                })
    return sorted(options, key=lambda x: (-x["score"], x["parser_name"], x["shadow_table_id"], x["row_index"], x["column_index"]))


def main() -> int:
    inputs, source_paths, integrity = _input_integrity()
    pages: dict[str, list[int]] = {}
    contracts = []
    for label in inputs.labels_by_id.values():
        if label.get("calculation"):
            for index in range(len(label["expected_sources"])):
                item = source_contract(label, index)
                contracts.append(item)
                pages.setdefault(item["document_id"], []).append(int(item["pdf_page"]))
    tables = [table for doc, selected in pages.items() for table in parse_tables(doc, source_paths[doc], selected)]
    by_source = {}
    package = []
    ambiguities = []
    for contract in sorted(contracts, key=lambda x: (x["case_id"], x["source_index"])):
        options = candidates(contract, tables)
        best = options[0] if options else {}
        status = "candidate_pending" if len(options) == 1 or (len(options) > 1 and options[0]["score"] > options[1]["score"]) else ("ambiguous" if options else "missing_row")
        record = {
            **contract, **best,
            "document_match": None, "page_match": None, "metric_match": None, "period_match": None, "value_match": None, "scale_match": None,
            "candidate_status": status, "review_status": "pending", "reviewer": None, "reviewed_at": None, "verified": False,
        }
        package.append(record)
        if status != "candidate_pending":
            ambiguities.append({"case_id": contract["case_id"], "source_index": contract["source_index"], "status": status, "candidate_count": len(options)})
        by_source[(contract["case_id"], contract["source_index"])] = best
    structures = []
    selected_ids = {r.get("shadow_table_id") for r in package if r.get("shadow_table_id")}
    for table in tables:
        if table.table_id not in selected_ids:
            continue
        structures.append({
            "parser_name": table.parser, "parser_version": "1.26.6" if table.parser == "pymupdf" else "camelot-current",
            "document_id": table.document_id, "pdf_page": table.page, "shadow_table_id": table.table_id,
            "table_index": table.index, "table_bbox": table.bbox, "row_count": len(table.matrix),
            "column_count": max((len(row) for row in table.matrix), default=0),
            "header_excerpt": [x for x in (table.headers or headers_from_matrix(table.matrix)) if x][:8],
            "row_identity_count": len(table.matrix),
            "parser_artifact_hash": hashlib.sha256(json.dumps([[norm(c) for c in r] for r in table.matrix], sort_keys=True).encode()).hexdigest(),
        })
    write("parser-table-structures.json", {"table_count": len(structures), "records": sorted(structures, key=lambda x: (x["document_id"], x["pdf_page"], x["parser_name"], x["table_index"]))})
    write("manual-mapping-review-package.json", {"record_count": len(package), "records": package})
    write("mapping-candidate-generation-report.json", {"source_count": 22, "candidate_pending_count": sum(x["candidate_status"] == "candidate_pending" for x in package), "gap_or_ambiguity_count": len(ambiguities), "automatic_verified_count": 0, "parser_inputs_contained_gold_answers": False})
    write("mapping-ambiguity-report.json", {"records": ambiguities})
    write("nf-opt-08-r2-acceptance.json", {"decision": "structured_reingestion_parser_mapping_blocked", "source_count": len(package), "case_source_unique": len({(x["case_id"],x["source_index"]) for x in package}) == 22, "all_review_status_pending": all(x["review_status"] == "pending" for x in package), "automatic_verified_count": 0, "model_calls": 0, "answer_generation_calls": 0, "binder_calls": 0, "calculator_calls": 0, "production_index_writes": 0, "input_hashes_verified": integrity["passed"], "control_set_hash": CONTROL_HASH})
    print(json.dumps({"records": len(package), "tables": len(structures), "pending": sum(x["candidate_status"] == "candidate_pending" for x in package), "gaps": len(ambiguities)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
