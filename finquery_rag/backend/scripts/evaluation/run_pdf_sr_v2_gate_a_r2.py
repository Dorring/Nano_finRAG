"""Replay native resolvers, then shadow-align and arbitrate parser tables."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path

from scripts.evaluation.run_pdf_sr_v2_gate_a import (
    DEFAULT_OUT as GATE_A_OUT,
    SOURCES,
    _camelot_adapter,
    _candidate_pages,
    _download,
    _page_lines_above,
    _write,
)
from scripts.evaluation.run_pdf_sr_v2_gate_a_r1 import DEFAULT_OUT as R1_OUT
from src.evaluation.pdf_source_representation_v2 import (
    YEAR_RE,
    extract_scale,
    normalize_text,
    parse_number,
    resolve_lineage,
    resolve_period_headers_v2,
    row_label,
    stable_identity,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-source-representation-v2-gate-a-r2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(table: object) -> list[list[str]]:
    return [[normalize_text(str(cell or "")) for cell in row] for row in table.extract()]


def _bbox(table: object) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in table.bbox)


def _all_lines_above(page: object, bbox: tuple[float, float, float, float]) -> list[str]:
    candidates = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            line_bbox = tuple(float(value) for value in line["bbox"])
            if line_bbox[3] > bbox[1] or line_bbox[2] < bbox[0] or line_bbox[0] > bbox[2]:
                continue
            text = normalize_text(" ".join(span["text"] for span in line["spans"]))
            if not text:
                continue
            max_size = max(float(span["size"]) for span in line["spans"])
            bold = any("bold" in str(span.get("font", "")).casefold() for span in line["spans"])
            centered = abs(((line_bbox[0] + line_bbox[2]) / 2) - ((bbox[0] + bbox[2]) / 2)) <= max(36, (bbox[2] - bbox[0]) * 0.2)
            if bold or centered or max_size >= 9 or "statement" in text.casefold() or text.casefold().startswith("note "):
                candidates.append((line_bbox[1], text))
    return [text for _, text in sorted(candidates)]


def _native_periods(page: object, table: object, matrix: list[list[str]]) -> list[dict[str, object] | None]:
    width = max((len(row) for row in matrix), default=0)
    resolved = resolve_period_headers_v2(matrix, width)
    unresolved = [index for index, value in enumerate(resolved) if value is None]
    if not unresolved or not table.rows:
        return resolved
    bbox = _bbox(table)
    header_clip = (bbox[0], max(0.0, bbox[1] - 160), bbox[2], min(float(page.rect.height), bbox[1] + 100))
    words = [word for word in page.get_text("words", clip=header_clip) if YEAR_RE.fullmatch(normalize_text(str(word[4])))]
    if not words:
        return resolved
    geometry_rows = [row.cells for row in table.rows if row.cells and len(row.cells) >= width]
    if not geometry_rows:
        return resolved
    columns = geometry_rows[0]
    assignments: dict[int, set[str]] = {index: set() for index in unresolved}
    boxes: dict[int, list[tuple[float, float, float, float]]] = {index: [] for index in unresolved}
    for word in words:
        center = (float(word[0]) + float(word[2])) / 2
        matches = [index for index in unresolved if columns[index] and float(columns[index][0]) <= center <= float(columns[index][2])]
        if len(matches) == 1:
            index = matches[0]
            assignments[index].add(normalize_text(str(word[4])))
            boxes[index].append(tuple(float(value) for value in word[:4]))
    for index, years in assignments.items():
        if len(years) == 1:
            year = next(iter(years))
            resolved[index] = {
                "header_path": (year,),
                "normalized_period": f"FY{year}",
                "period_kind": "fiscal_year",
                "resolution_method": "native_word_column_alignment",
                "source_bboxes": boxes[index],
            }
    return resolved


def _table_record(document_id: str, page_number: int, page: object, table: object, parser: str, index: int) -> dict[str, object]:
    matrix = _matrix(table)
    bbox = _bbox(table)
    periods = _native_periods(page, table, matrix)
    headings = _all_lines_above(page, bbox)
    lineage = resolve_lineage(headings + [cell for row in matrix[:8] for cell in row])
    matrix_text = " ".join(cell for row in matrix[:8] for cell in row)
    raw_scale, scale = extract_scale(matrix_text)
    if not scale:
        raw_scale, scale = extract_scale(" ".join(_page_lines_above(page, bbox)))
    rows = []
    geometry_cells = 0
    numeric_cells = 0
    for row_index, row in enumerate(matrix):
        numeric = [(column, parse_number(cell)) for column, cell in enumerate(row) if parse_number(cell) is not None]
        if not numeric:
            continue
        geometry = table.rows[row_index].cells if row_index < len(table.rows) else []
        valid = sum(
            column < len(geometry)
            and bool(geometry[column])
            and bbox[0] - 1 <= geometry[column][0] <= geometry[column][2] <= bbox[2] + 1
            and bbox[1] - 1 <= geometry[column][1] <= geometry[column][3] <= bbox[3] + 1
            for column, _ in numeric
        )
        numeric_cells += len(numeric)
        geometry_cells += valid
        rows.append(
            {
                "metric_present": bool(row_label(row)),
                "period_present": any(column < len(periods) and periods[column] for column, _ in numeric),
                "numeric_cell_count": len(numeric),
                "geometry_cell_count": valid,
            }
        )
    fingerprint = sorted({normalize_text(row_label(row) or "").casefold() for row in matrix if row_label(row)})
    numeric_fingerprint = sorted({normalize_text(cell).replace(",", "") for row in matrix for cell in row if parse_number(cell) is not None})
    table_id = stable_identity("pdf-r2-table", document_id, page_number, parser, bbox, matrix)
    return {
        "table_id": table_id,
        "document_id": document_id,
        "pdf_page": page_number,
        "parser": parser,
        "table_index": index,
        "bbox": bbox,
        "row_count": len(rows),
        "rows": rows,
        "numeric_cell_count": numeric_cells,
        "geometry_cell_count": geometry_cells,
        "lineage": lineage,
        "scale": scale,
        "scale_context_hash": hashlib.sha256((raw_scale or "").encode()).hexdigest() if raw_scale else None,
        "period_headers": periods,
        "row_fingerprint": fingerprint,
        "numeric_fingerprint": numeric_fingerprint,
    }


def _coverage(tables: list[dict[str, object]]) -> dict[str, int | float]:
    rows = [row for table in tables for row in table["rows"]]
    numeric_cells = sum(int(table["numeric_cell_count"]) for table in tables)
    geometry_cells = sum(int(table["geometry_cell_count"]) for table in tables)
    metric = sum(bool(row["metric_present"]) for row in rows)
    period = sum(bool(row["period_present"]) for row in rows)
    metric_period = sum(bool(row["metric_present"] and row["period_present"]) for row in rows)
    lineage = sum(bool(table["lineage"]) * int(table["row_count"]) for table in tables)
    scale = sum(bool(table["scale"]) * int(table["row_count"]) for table in tables)
    denominator = len(rows)
    return {
        "table_count": len(tables),
        "row_count": denominator,
        "metric_row_count": metric,
        "metric_row_coverage": metric / denominator if denominator else 0,
        "period_row_count": period,
        "period_row_coverage": period / denominator if denominator else 0,
        "metric_period_row_count": metric_period,
        "metric_period_row_coverage": metric_period / denominator if denominator else 0,
        "lineage_row_count": lineage,
        "lineage_row_coverage": lineage / denominator if denominator else 0,
        "scale_row_count": scale,
        "scale_row_coverage": scale / denominator if denominator else 0,
        "numeric_cell_count": numeric_cells,
        "geometry_cell_count": geometry_cells,
        "geometry_cell_coverage": geometry_cells / numeric_cells if numeric_cells else 0,
    }


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0


def _jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 0


def _quality(table: dict[str, object]) -> tuple[float, float, float, int, float, int]:
    rows = table["rows"]
    count = len(rows)
    geometry = int(table["geometry_cell_count"]) / int(table["numeric_cell_count"]) if table["numeric_cell_count"] else 0
    period = sum(row["period_present"] for row in rows) / count if count else 0
    metric = sum(row["metric_present"] for row in rows) / count if count else 0
    consistency = 1.0 if count and table["numeric_cell_count"] else 0.0
    return geometry, period, metric, int(bool(table["lineage"])), consistency, 0


def run(args: argparse.Namespace) -> int:
    import camelot
    import pymupdf

    gate_a = GATE_A_OUT / "pdf-sr-v2-gate-a-acceptance.json"
    r1 = R1_OUT / "pdf-sr-v2-gate-a-r1-acceptance.json"
    if not gate_a.exists() or not r1.exists():
        raise RuntimeError("Gate A and R1 inputs are required")
    resolver_tables = []
    all_parser_tables = []
    page_count = 0
    for source in SOURCES:
        pdf = _download(source, args.runtime_dir, args.user_agent)
        with pymupdf.open(pdf) as document:
            pages = _candidate_pages(document)
            page_count += len(pages)
            for page_index in pages:
                page = document[page_index]
                pymupdf_tables = list(page.find_tables().tables)
                stream_raw = list(camelot.read_pdf(str(pdf), pages=str(page_index + 1), flavor="stream", edge_tol=50, row_tol=10))
                stream_raw = [item for item in stream_raw if item.df.shape[0] >= 2 and item.df.shape[1] >= 2]
                lattice_raw = list(camelot.read_pdf(str(pdf), pages=str(page_index + 1), flavor="lattice"))
                lattice_raw = [item for item in lattice_raw if item.df.shape[0] >= 2 and item.df.shape[1] >= 2]
                stream = [_camelot_adapter(item, page_height=float(page.rect.height)) for item in stream_raw]
                lattice = [_camelot_adapter(item, page_height=float(page.rect.height)) for item in lattice_raw]
                baseline = pymupdf_tables or stream or lattice
                baseline_parser = "pymupdf" if pymupdf_tables else "camelot_stream" if stream else "camelot_lattice"
                resolver_tables.extend(_table_record(source["document_id"], page_index + 1, page, table, baseline_parser, index) for index, table in enumerate(baseline))
                for parser_name, tables in (("pymupdf", pymupdf_tables), ("camelot_stream", stream), ("camelot_lattice", lattice)):
                    all_parser_tables.extend(_table_record(source["document_id"], page_index + 1, page, table, parser_name, index) for index, table in enumerate(tables))

    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for table in all_parser_tables:
        grouped.setdefault((str(table["document_id"]), int(table["pdf_page"])), []).append(table)
    alignments = []
    selected = []
    ambiguous = 0
    for (document_id, pdf_page), page_tables in grouped.items():
        remaining = {str(table["table_id"]): table for table in page_tables}
        pairs = []
        ids = list(remaining)
        for left_index, left_id in enumerate(ids):
            left = remaining[left_id]
            for right_id in ids[left_index + 1 :]:
                right = remaining[right_id]
                if left["parser"] == right["parser"]:
                    continue
                bbox_iou = _iou(tuple(left["bbox"]), tuple(right["bbox"]))
                row_similarity = _jaccard(left["row_fingerprint"], right["row_fingerprint"])
                numeric_similarity = _jaccard(left["numeric_fingerprint"], right["numeric_fingerprint"])
                if bbox_iou >= 0.45 and (row_similarity >= 0.35 or numeric_similarity >= 0.5):
                    pairs.append((bbox_iou + row_similarity + numeric_similarity, left_id, right_id, bbox_iou, row_similarity, numeric_similarity))
        adjacency: Counter[str] = Counter()
        for _, left_id, right_id, *_ in pairs:
            adjacency[left_id] += 1
            adjacency[right_id] += 1
        ambiguous_ids = {table_id for table_id, degree in adjacency.items() if degree > 1}
        ambiguous += len(ambiguous_ids)
        used = set()
        for _score, left_id, right_id, bbox_iou, row_similarity, numeric_similarity in sorted(pairs, reverse=True):
            if left_id in used or right_id in used:
                continue
            if left_id in ambiguous_ids or right_id in ambiguous_ids:
                continue
            members = [remaining[left_id], remaining[right_id]]
            winner = max(members, key=_quality)
            logical_id = stable_identity("pdf-r2-logical-table", document_id, pdf_page, sorted([left_id, right_id]))
            selected.append(winner)
            used.update((left_id, right_id))
            alignments.append({"logical_table_id": logical_id, "member_table_ids": [left_id, right_id], "selected_table_id": winner["table_id"], "match_status": "unique", "bbox_iou": bbox_iou, "row_fingerprint_similarity": row_similarity, "numeric_fingerprint_similarity": numeric_similarity, "quality": {table["parser"]: _quality(table) for table in members}})
        for table_id, table in remaining.items():
            if table_id in ambiguous_ids:
                alignments.append({"logical_table_id": None, "member_table_ids": [table_id], "selected_table_id": None, "match_status": "parser_arbitration_ambiguous", "quality": {table["parser"]: _quality(table)}})
            elif table_id not in used:
                selected.append(table)
                alignments.append({"logical_table_id": stable_identity("pdf-r2-logical-table", document_id, pdf_page, table_id), "member_table_ids": [table_id], "selected_table_id": table_id, "match_status": "unmatched", "quality": {table["parser"]: _quality(table)}})

    resolver_coverage = _coverage(resolver_tables)
    arbitrated_coverage = _coverage(selected)
    thresholds = {"period_row_coverage": 0.70, "metric_period_row_coverage": 0.65, "lineage_row_coverage": 0.70, "metric_row_coverage": 0.94, "scale_row_coverage": 0.89, "geometry_cell_coverage": 0.99}
    threshold_results = {key: arbitrated_coverage[key] >= value for key, value in thresholds.items()}
    gate_passed = page_count == 104 and all(threshold_results.values())
    acceptance = {
        "schema": "pdf-source-representation-v2/gate-a-r2/acceptance/v1",
        "gate_a_sha256": _sha(gate_a),
        "r1_sha256": _sha(r1),
        "frozen_page_count": page_count,
        "resolver_only_table_count": len(resolver_tables),
        "arbitrated_logical_table_count": len(selected),
        "parser_arbitration_ambiguity_count": ambiguous,
        "duplicate_logical_table_id_count": len(selected) - len({item["logical_table_id"] for item in alignments if item["logical_table_id"] is not None}),
        "coverage_thresholds": thresholds,
        "coverage_threshold_results": threshold_results,
        "gate_passed": gate_passed,
        "resolver_gold_reads": 0,
        "parser_arbitration_gold_reads": 0,
        "frozen_72_question_reads": 0,
        "recall_evaluation_run": False,
        "candidate_construction_run": False,
        "embedding_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "decision": "pdf_sr_v2_native_resolver_gate_passed" if gate_passed else "pdf_sr_v2_native_resolver_coverage_blocked",
        "next_gate": "pdf_sr_v2_row_cell_candidate_construction" if gate_passed else "stop_and_review_unresolved_native_structure",
    }
    _write(args.out_dir / "header-grid-schema.json", {"fields": ["column_index", "header_path", "normalized_period", "period_kind", "resolution_method", "source_bboxes"], "gold_fields_used": False})
    _write(args.out_dir / "statement-lineage-schema.json", {"lineage_types": ["primary_financial_statement", "financial_statement_continuation", "note_section", "segment_section", "financial_schedule", "other_financial_section", "unknown"], "gold_fields_used": False})
    _write(args.out_dir / "resolver-only-coverage-report.json", resolver_coverage)
    _write(args.out_dir / "parser-logical-table-alignment.json", {"records": alignments})
    _write(args.out_dir / "parser-arbitration-report.json", {"all_parser_table_count": len(all_parser_tables), "logical_table_count": len(selected), "alignment_count": sum(item["match_status"] == "unique" for item in alignments), "ambiguity_count": ambiguous, "selected_parser_counts": dict(Counter(str(table["parser"]) for table in selected))})
    _write(args.out_dir / "arbitrated-coverage-report.json", arbitrated_coverage)
    _write(args.out_dir / "unresolved-header-report.json", {"row_count": arbitrated_coverage["row_count"] - arbitrated_coverage["period_row_count"]})
    _write(args.out_dir / "unresolved-lineage-report.json", {"row_count": arbitrated_coverage["row_count"] - arbitrated_coverage["lineage_row_count"]})
    _write(args.out_dir / "coverage-regression-report.json", {key: arbitrated_coverage[key] - resolver_coverage[key] for key in thresholds})
    _write(args.out_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": acceptance["next_gate"], "production_switch_allowed": False})
    _write(args.out_dir / "pdf-sr-v2-gate-a-r2-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--user-agent", default="nano-finance-research contact@example.com")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
