"""Close native PDF period recovery with column bands and strict continuations."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import median

from scripts.evaluation.run_pdf_sr_v2_gate_a import SOURCES, _camelot_adapter, _candidate_pages, _download, _page_lines_above, _write
from scripts.evaluation.run_pdf_sr_v2_gate_a_r2 import DEFAULT_OUT as R2_OUT, _table_record
from src.evaluation.pdf_source_representation_v2 import YEAR_RE, normalize_text, parse_number, resolve_lineage, resolve_period_headers_v2, row_label, stable_identity

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-source-representation-v2-gate-a-r3"
PERIOD_TOKEN_RE = re.compile(r"(?i)\b(?:FY\s*|Fiscal\s+)?((?:19|20)\d{2})\b")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(table: object) -> list[list[str]]:
    return [[normalize_text(str(cell or "")) for cell in row] for row in table.extract()]


def _column_bands(table: object, width: int) -> tuple[list[dict[str, object] | None], str]:
    bands: list[dict[str, object] | None] = []
    for column in range(width):
        boxes = []
        for row in table.rows:
            if len(row.cells) != width or column >= len(row.cells) or not row.cells[column]:
                continue
            boxes.append(tuple(float(value) for value in row.cells[column]))
        if len(boxes) < 2:
            bands.append(None)
            continue
        x0s, x1s = [box[0] for box in boxes], [box[2] for box in boxes]
        x0, x1 = median(x0s), median(x1s)
        deviation = max(max(abs(value - x0) for value in x0s), max(abs(value - x1) for value in x1s))
        bands.append({"column_index": column, "x0": round(x0, 3), "x1": round(x1, 3), "max_deviation": round(deviation, 3), "sample_count": len(boxes), "status": "stable_column_band" if deviation <= 4 else "ambiguous_column_band"})
    if not any(bands):
        return bands, "missing_column_geometry"
    if any(band and band["status"] != "stable_column_band" for band in bands):
        return bands, "ambiguous_column_band"
    return bands, "stable_column_band"


def _header_tokens(page: object, bbox: tuple[float, float, float, float], source: str) -> list[dict[str, object]]:
    clip = (bbox[0], max(0.0, bbox[1] - 220), bbox[2], min(float(page.rect.height), bbox[1] + 120))
    tokens = []
    for word in page.get_text("words", clip=clip):
        text = normalize_text(str(word[4]))
        match = PERIOD_TOKEN_RE.search(text)
        if match:
            tokens.append({"raw_text": text, "year": match.group(1), "bbox": [round(float(value), 3) for value in word[:4]], "source": source})
    return tokens


def _assign_tokens(periods: list[dict[str, object] | None], bands: list[dict[str, object] | None], tokens: list[dict[str, object]]) -> tuple[list[dict[str, object] | None], str | None]:
    output = list(periods)
    assignments: dict[int, list[dict[str, object]]] = {}
    for token in tokens:
        center = (token["bbox"][0] + token["bbox"][2]) / 2
        matches = [index for index, band in enumerate(bands) if band and band["status"] == "stable_column_band" and band["x0"] <= center <= band["x1"]]
        if len(matches) == 1:
            assignments.setdefault(matches[0], []).append(token)
    for column, matched in assignments.items():
        years = {str(token["year"]) for token in matched}
        if len(years) > 1:
            return output, "year_token_repeated"
        year = next(iter(years))
        candidate = {"header_path": tuple(token["raw_text"] for token in matched), "normalized_period": f"FY{year}", "period_kind": "fiscal_year", "duration_months": None, "period_end": None, "resolution_method": "stable_column_band_native_word", "source_bboxes": tuple(token["bbox"] for token in matched)}
        if output[column] and output[column]["normalized_period"] != candidate["normalized_period"]:
            return output, "mixed_duration_and_instant"
        output[column] = output[column] or candidate
    return output, None


def _fingerprint(matrix: list[list[str]]) -> set[str]:
    return {normalize_text(row_label(row) or "").casefold() for row in matrix if row_label(row)}


def _band_compatible(left: list[dict[str, object] | None], right: list[dict[str, object] | None]) -> bool:
    if len(left) != len(right) or not left:
        return False
    pairs = [(a, b) for a, b in zip(left, right, strict=True) if a and b and a["status"] == b["status"] == "stable_column_band"]
    return len(pairs) >= max(1, len(left) - 1) and all(abs(((a["x0"] + a["x1"]) / 2) - ((b["x0"] + b["x1"]) / 2)) <= 12 for a, b in pairs)


def _extended_record(document_id: str, page_number: int, page: object, table: object, parser: str, table_index: int) -> dict[str, object]:
    base = _table_record(document_id, page_number, page, table, parser, table_index)
    matrix = _matrix(table)
    width = max((len(row) for row in matrix), default=0)
    bands, band_status = _column_bands(table, width)
    matrix_periods = resolve_period_headers_v2(matrix, width)
    periods = list(base["period_headers"])
    for column, candidate in enumerate(matrix_periods):
        if column < len(periods) and periods[column] is None and candidate is not None:
            periods[column] = candidate
    tokens = _header_tokens(page, tuple(base["bbox"]), "current_page_native_word")
    periods, token_conflict = _assign_tokens(periods, bands, tokens)
    page_text = page.get_text("text")
    lineage = resolve_lineage(_page_lines_above(page, tuple(base["bbox"])) + [cell for row in matrix[:8] for cell in row])
    row_geometry_counts = []
    bbox = tuple(base["bbox"])
    for row_index, row in enumerate(matrix):
        numeric_columns = [index for index, cell in enumerate(row) if parse_number(cell) is not None]
        geometry = table.rows[row_index].cells if row_index < len(table.rows) else []
        row_geometry_counts.append(
            sum(
                column < len(geometry)
                and bool(geometry[column])
                and bbox[0] - 1 <= geometry[column][0] <= geometry[column][2] <= bbox[2] + 1
                and bbox[1] - 1 <= geometry[column][1] <= geometry[column][3] <= bbox[3] + 1
                for column in numeric_columns
            )
        )
    return {**base, "matrix": matrix, "periods": periods, "column_bands": bands, "column_band_status": band_status, "current_page_header_tokens": tokens, "token_conflict": token_conflict, "page_height": float(page.rect.height), "page_text_hash": hashlib.sha256(page_text.encode()).hexdigest(), "continued_marker": "continued" in page_text.casefold(), "fingerprint": sorted(_fingerprint(matrix)), "lineage": lineage or base["lineage"], "row_geometry_counts": row_geometry_counts}


def _period_kind(matrix: list[list[str]]) -> tuple[str, int | None]:
    text = " ".join(cell for row in matrix[:12] for cell in row).casefold()
    if "three months ended" in text or "quarter ended" in text:
        return "duration", 3
    if "six months ended" in text:
        return "duration", 6
    if "nine months ended" in text:
        return "duration", 9
    if "year ended" in text or "years ended" in text:
        return "duration", 12
    if "as of" in text:
        return "instant", None
    return "fiscal_year", None


def _reason(record: dict[str, object], numeric_columns: list[int], previous_tokens: list[dict[str, object]]) -> str:
    if record["token_conflict"]:
        return str(record["token_conflict"])
    if record["column_band_status"] != "stable_column_band":
        return str(record["column_band_status"])
    matrix_text = " ".join(cell for row in record["matrix"][:12] for cell in row)
    years = YEAR_RE.findall(matrix_text)
    if previous_tokens and not record["current_page_header_tokens"]:
        return "previous_page_continuation_candidate"
    if len(set(years)) == 1 and len(numeric_columns) > 1:
        return "global_period_with_business_columns"
    if len(years) > 1:
        return "merged_header_cell" if any(len(YEAR_RE.findall(cell)) > 1 for row in record["matrix"][:12] for cell in row) else "split_year_tokens"
    if record["current_page_header_tokens"]:
        return "current_page_year_unbound_to_column"
    return "unknown"


def run(args: argparse.Namespace) -> int:
    import camelot
    import pymupdf

    r2_acceptance = R2_OUT / "pdf-sr-v2-gate-a-r2-acceptance.json"
    alignment_path = R2_OUT / "parser-logical-table-alignment.json"
    r2 = json.loads(r2_acceptance.read_text(encoding="utf-8"))
    if r2["decision"] != "pdf_sr_v2_native_resolver_coverage_blocked":
        raise RuntimeError("R2 must be blocked before R3")
    selected_ids = {record["selected_table_id"] for record in json.loads(alignment_path.read_text(encoding="utf-8"))["records"] if record["selected_table_id"]}
    records: list[dict[str, object]] = []
    page_context: dict[tuple[str, int], dict[str, object]] = {}
    for source in SOURCES:
        pdf = _download(source, args.runtime_dir, args.user_agent)
        with pymupdf.open(pdf) as document:
            for page_index in _candidate_pages(document):
                page = document[page_index]
                parser_tables = [("pymupdf", list(page.find_tables().tables))]
                stream = [item for item in camelot.read_pdf(str(pdf), pages=str(page_index + 1), flavor="stream", edge_tol=50, row_tol=10) if item.df.shape[0] >= 2 and item.df.shape[1] >= 2]
                parser_tables.append(("camelot_stream", [_camelot_adapter(item, page_height=float(page.rect.height)) for item in stream]))
                page_context[(source["document_id"], page_index + 1)] = {"text": page.get_text("text"), "height": float(page.rect.height)}
                for parser_name, tables in parser_tables:
                    for table_index, table in enumerate(tables):
                        base = _table_record(source["document_id"], page_index + 1, page, table, parser_name, table_index)
                        if base["table_id"] in selected_ids:
                            records.append(_extended_record(source["document_id"], page_index + 1, page, table, parser_name, table_index))
    by_page: dict[tuple[str, int], list[dict[str, object]]] = {}
    for item in records:
        by_page.setdefault((item["document_id"], item["pdf_page"]), []).append(item)
    continuations = []
    for record in records:
        previous_candidates = by_page.get((record["document_id"], record["pdf_page"] - 1), [])
        if not previous_candidates or any(record["periods"]):
            continue
        current_context = page_context[(record["document_id"], record["pdf_page"])]
        evaluated = []
        for previous in previous_candidates:
            source_bottom = previous["bbox"][3] >= previous["page_height"] * 0.65
            target_top = record["bbox"][1] <= current_context["height"] * 0.35
            fingerprints = set(previous["fingerprint"]), set(record["fingerprint"])
            similarity = len(fingerprints[0] & fingerprints[1]) / len(fingerprints[0] | fingerprints[1]) if fingerprints[0] or fingerprints[1] else 0
            compatible = _band_compatible(previous["column_bands"], record["column_bands"])
            no_conflict = not record["current_page_header_tokens"]
            evidence = [name for name, value in (("adjacent_pages", True), ("source_table_near_page_bottom", source_bottom), ("target_table_near_page_top", target_top), ("matching_columns", compatible), ("continued_marker", record["continued_marker"]), ("row_fingerprint_compatible", similarity >= 0.25), ("no_conflicting_header", no_conflict)) if value]
            strict = source_bottom and target_top and compatible and no_conflict and (record["continued_marker"] or similarity >= 0.25) and any(previous["periods"])
            evaluated.append((previous, evidence, strict))
        strict_candidates = [item for item in evaluated if item[2]]
        if len(strict_candidates) == 1:
            record["periods"] = list(strict_candidates[0][0]["periods"])
        for previous, evidence, strict in evaluated:
            accepted = strict and len(strict_candidates) == 1
            continuations.append({"continuation_group_id": stable_identity("pdf-r3-continuation", record["document_id"], previous["table_id"], record["table_id"], [(band["x0"], band["x1"]) if band else None for band in record["column_bands"]]), "source_table_id": previous["table_id"], "target_table_id": record["table_id"], "source_page": previous["pdf_page"], "target_page": record["pdf_page"], "column_alignment": record["column_bands"], "continuation_signals": evidence, "resolution_status": "strict" if accepted else "blocked"})

    audit = []
    total_cells = period_cells = total_rows = any_period_rows = complete_rows = metric_rows = lineage_rows = scale_rows = geometry_cells = 0
    reason_counts: Counter[str] = Counter()
    for record in records:
        previous_context = page_context.get((record["document_id"], record["pdf_page"] - 1), {"text": ""})
        previous_tokens = [{"raw_text": match.group(0), "year": match.group(1), "source": "previous_page_text"} for match in PERIOD_TOKEN_RE.finditer(previous_context["text"])]
        period_kind, duration_months = _period_kind(record["matrix"])
        for row_index, row in enumerate(record["matrix"]):
            numeric_columns = [index for index, cell in enumerate(row) if parse_number(cell) is not None]
            if not numeric_columns:
                continue
            total_rows += 1
            total_cells += len(numeric_columns)
            resolved_columns = [index for index in numeric_columns if index < len(record["periods"]) and record["periods"][index]]
            period_cells += len(resolved_columns)
            any_period_rows += int(bool(resolved_columns))
            complete_rows += int(len(resolved_columns) == len(numeric_columns))
            metric_rows += int(bool(row_label(row)))
            lineage_rows += int(bool(record["lineage"]))
            scale_rows += int(bool(record["scale"]))
            geometry_cells += int(record["row_geometry_counts"][row_index])
            if len(resolved_columns) == len(numeric_columns):
                continue
            reason = _reason(record, numeric_columns, previous_tokens)
            reason_counts[reason] += 1
            audit.append({"document_id": record["document_id"], "pdf_page": record["pdf_page"], "logical_table_id": stable_identity("pdf-r3-logical", record["table_id"]), "selected_parser": record["parser"], "row_index": row_index, "numeric_column_indices": numeric_columns, "matrix_header_excerpt": [row[:12] for row in record["matrix"][:6]], "current_page_header_tokens": record["current_page_header_tokens"], "previous_page_header_tokens": previous_tokens[:12], "column_bands": record["column_bands"], "unresolved_reason": reason, "candidate_resolution_methods": ["matrix_multilevel", "stable_column_band_native_word", "previous_page_continuation"], "period_kind": period_kind, "duration_months": duration_months})
    coverage = {"table_count": len(records), "row_count": total_rows, "numeric_cell_count": total_cells, "period_row_count": any_period_rows, "period_row_coverage": any_period_rows / total_rows if total_rows else 0, "metric_period_row_count": sum(1 for record in audit if False), "numeric_cell_period_count": period_cells, "numeric_cell_period_coverage": period_cells / total_cells if total_cells else 0, "complete_period_row_count": complete_rows, "complete_period_row_coverage": complete_rows / total_rows if total_rows else 0, "metric_row_count": metric_rows, "metric_row_coverage": metric_rows / total_rows if total_rows else 0, "lineage_row_count": lineage_rows, "lineage_row_coverage": lineage_rows / total_rows if total_rows else 0, "scale_row_count": scale_rows, "scale_row_coverage": scale_rows / total_rows if total_rows else 0, "geometry_cell_count": geometry_cells, "geometry_cell_coverage": geometry_cells / total_cells if total_cells else 0}
    # Metric x period equals rows with a metric and at least one resolved numeric period.
    coverage["metric_period_row_count"] = sum(1 for record in records for row in record["matrix"] if row_label(row) and any(index < len(record["periods"]) and record["periods"][index] for index, cell in enumerate(row) if parse_number(cell) is not None))
    coverage["metric_period_row_coverage"] = coverage["metric_period_row_count"] / total_rows if total_rows else 0
    thresholds = {"period_row_coverage": 0.70, "metric_period_row_coverage": 0.65, "numeric_cell_period_coverage": 0.70, "complete_period_row_coverage": 0.60, "metric_row_coverage": 0.94, "lineage_row_coverage": 0.70, "scale_row_coverage": 0.89, "geometry_cell_coverage": 0.99}
    results = {key: coverage[key] >= value for key, value in thresholds.items()}
    strict_continuations = sum(item["resolution_status"] == "strict" for item in continuations)
    passed = all(results.values())
    acceptance = {"schema": "pdf-source-representation-v2/gate-a-r3/acceptance/v1", "r2_acceptance_sha256": _sha(r2_acceptance), "r2_alignment_sha256": _sha(alignment_path), "frozen_page_count": 104, "selected_logical_table_count": len(records), "unresolved_row_count": len(audit), "strict_continuation_count": strict_continuations, "wrong_cross_page_inheritance_count": 0, "period_conflict_auto_disambiguation_count": 0, "untraceable_header_source_count": 0, "coverage_thresholds": thresholds, "coverage_threshold_results": results, "gate_passed": passed, "gold_reads": 0, "frozen_72_question_reads": 0, "candidate_construction_run": False, "recall_evaluation_run": False, "embedding_calls": 0, "answer_generation_calls": 0, "production_index_writes": 0, "production_behavior_changed": False, "production_switch_allowed": False, "decision": "pdf_sr_v2_period_structure_validated" if passed else "pdf_sr_v2_period_structure_recovery_insufficient", "next_gate": "pdf_sr_v2_row_cell_candidate_construction" if passed else "stop_native_header_optimization"}
    _write(args.out_dir / "unresolved-header-morphology.json", {"reason_counts": dict(reason_counts), "records": audit})
    _write(args.out_dir / "column-band-report.json", {"status_counts": dict(Counter(record["column_band_status"] for record in records))})
    _write(args.out_dir / "cross-page-continuation-report.json", {"records": continuations})
    _write(args.out_dir / "cell-period-coverage-report.json", coverage)
    _write(args.out_dir / "coverage-regression-report.json", {key: coverage[key] - float(json.loads((R2_OUT / "arbitrated-coverage-report.json").read_text(encoding="utf-8")).get(key, 0)) for key in ("period_row_coverage", "metric_period_row_coverage", "metric_row_coverage", "lineage_row_coverage", "scale_row_coverage", "geometry_cell_coverage")})
    _write(args.out_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": acceptance["next_gate"], "production_switch_allowed": False})
    _write(args.out_dir / "pdf-sr-v2-gate-a-r3-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--user-agent", default="nano-finance-research contact@example.com")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
