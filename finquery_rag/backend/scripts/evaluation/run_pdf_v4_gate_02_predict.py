"""Build the automatic MinerU + PyMuPDF structured adapter (V4 Gate 02).

Prediction is deliberately Oracle-blind.  It reads only the sealed Hybrid
MinerU output, the probe page manifest and the original PDFs.  Oracle records
are loaded by the separate scoring script after the prediction seal exists.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from html.parser import HTMLParser
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluation.run_pdf_v4_gate_01_r1 import (
    DEFAULT_PDF_ROOT,
    DEFAULT_RUNTIME,
    extract_bbox,
    iter_dicts,
    normalize_financial_numeric_text,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBE = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-01"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _norm(value: Any) -> str:
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(value))


def _tokens(value: Any) -> list[str]:
    value = html.unescape(str(value or "")).lower().replace("−", "-")
    return re.findall(r"[a-z]+|\(?[-+]?\d[\d,]*(?:\.\d+)?\)?%?", value)


class _TableHTMLParser(HTMLParser):
    """Small deterministic table parser preserving rowspan/colspan."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, Any]]] = []
        self._row: list[dict[str, Any]] | None = None
        self._cell: dict[str, Any] | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"td", "th"} and self._row is not None:
            attrs_dict = dict(attrs)
            self._cell = {
                "raw_text": "", "rowspan": max(1, int(attrs_dict.get("rowspan") or 1)),
                "colspan": max(1, int(attrs_dict.get("colspan") or 1)),
                "header": tag.lower() == "th",
            }
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._cell["raw_text"] = re.sub(r"\s+", " ", " ".join(self._buffer)).strip()
            self._row.append(self._cell)
            self._cell = None
            self._buffer = []
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def parse_table_html(table_html: str) -> dict[str, Any]:
    parser = _TableHTMLParser()
    parser.feed(table_html)
    source_rows = parser.rows
    grid: list[list[dict[str, Any] | None]] = []
    cell_records: list[dict[str, Any]] = []
    for row_index, source_row in enumerate(source_rows):
        while len(grid) <= row_index:
            grid.append([])
        col = 0
        for source_cell in source_row:
            while col < len(grid[row_index]) and grid[row_index][col] is not None:
                col += 1
            record = {
                "source_row": row_index,
                "source_col": col,
                "rowspan": int(source_cell["rowspan"]),
                "colspan": int(source_cell["colspan"]),
                "raw_text": source_cell["raw_text"],
                "header": bool(source_cell["header"]),
            }
            cell_records.append(record)
            for rr in range(row_index, row_index + record["rowspan"]):
                while len(grid) <= rr:
                    grid.append([])
                while len(grid[rr]) < col + record["colspan"]:
                    grid[rr].append(None)
                for cc in range(col, col + record["colspan"]):
                    if grid[rr][cc] is None:
                        grid[rr][cc] = record
            col += record["colspan"]
    width = max((len(row) for row in grid), default=0)
    for row in grid:
        row.extend([None] * (width - len(row)))
    unique_records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row_index, row in enumerate(grid):
        for col_index, record in enumerate(row):
            if record is None:
                continue
            ident = id(record)
            if ident not in seen:
                record["grid_row"] = row_index
                record["grid_col"] = col_index
                unique_records.append(record)
                seen.add(ident)
    return {"grid": grid, "cells": unique_records, "row_count": len(grid), "column_count": width}


def _period_from_text(text: str) -> str | None:
    match = re.search(r"\b(?:fy|fiscal\s+year\s*)?(19|20)\d{2}\b", text, re.I)
    if match:
        year = re.search(r"(?:19|20)\d{2}", match.group(0))
        return f"FY{year.group(0)}" if year else None
    match = re.search(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2},?\s+(19|20)\d{2}\b", text, re.I)
    if not match:
        return None
    year_match = re.search(r"(?:19|20)\d{2}", match.group(0))
    return f"FY{year_match.group(0)}" if year_match else None


def _period_kind(text: str) -> str | None:
    lowered = text.lower()
    if "as of" in lowered:
        return "instant"
    if "ended" in lowered or "year" in lowered:
        return "duration"
    return None


def _numeric_values(text: str) -> list[dict[str, Any]]:
    values = []
    grouped_number = r"(?:\d{1,3}(?:\s\d{3})+(?:\.\d+)?|\d[\d,]*(?:\.\d+)?)"
    pattern = rf"(?<![A-Za-z0-9])(?:\(\s*(?:[$€£¥₹₽₩]?\s*)?[-+−]?\s*{grouped_number}\s*\)|(?:[$€£¥₹₽₩]?\s*)?[-+−]?\s*{grouped_number}%?)"
    for match in re.finditer(pattern, html.unescape(text or "")):
        token = normalize_financial_numeric_text(match.group(0))
        if token["valid"]:
            values.append({"raw": token["raw"], "normalized": token["normalized"], "percent": token["percent"]})
    return values


def _page_words(pdf_path: Path, page_number: int) -> list[dict[str, Any]]:
    import fitz

    document = fitz.open(pdf_path)
    page = document[page_number - 1]
    return [{"index": i, "text": item[4], "bbox": [float(x) for x in item[:4]], "line": item[6], "block": item[5]} for i, item in enumerate(page.get_text("words"))]


def _inside(word: dict[str, Any], bbox: list[float], margin: float = 0.0) -> bool:
    x0, y0, x1, y1 = bbox
    wx0, wy0, wx1, wy1 = word["bbox"]
    return wx1 >= x0 - margin and wx0 <= x1 + margin and wy1 >= y0 - margin and wy0 <= y1 + margin


def _union_bbox(words: list[dict[str, Any]]) -> list[float] | None:
    if not words:
        return None
    return [min(w["bbox"][0] for w in words), min(w["bbox"][1] for w in words), max(w["bbox"][2] for w in words), max(w["bbox"][3] for w in words)]


def _column_bands(bbox: list[float], width: int) -> list[list[float]]:
    if width <= 0:
        return []
    x0, y0, x1, y1 = bbox
    step = (x1 - x0) / width
    return [[x0 + index * step, y0, x0 + (index + 1) * step, y1] for index in range(width)]


def _match_native_words(cell_text: str, words: list[dict[str, Any]], row_index: int, row_count: int, table_bbox: list[float], used: set[int]) -> list[dict[str, Any]]:
    target = [_compact(token) for token in _tokens(cell_text) if _compact(token)]
    if not target:
        return []
    y0, y1 = table_bbox[1], table_bbox[3]
    expected_y = y0 + (row_index + 0.5) * (y1 - y0) / max(row_count, 1)
    candidates = [w for w in words if w["index"] not in used and abs((w["bbox"][1] + w["bbox"][3]) / 2 - expected_y) <= max(18.0, (y1 - y0) / max(row_count, 1) * 1.5)]
    selected: list[dict[str, Any]] = []
    last_x = -1.0

    def token_matches(target_token: str, word_text: str) -> bool:
        word_token = _compact(word_text)
        if not target_token or not word_token:
            return False
        if target_token == word_token:
            return True
        # Do not let a one-digit word such as a change percentage satisfy a
        # multi-digit financial value merely because it is a substring.
        return (len(target_token) >= 4 and target_token in word_token) or (len(word_token) >= 4 and word_token in target_token)

    for token in target:
        matching = [w for w in candidates if w["index"] not in {x["index"] for x in selected} and w["bbox"][0] >= last_x and token_matches(token, w["text"])]
        if not matching:
            # Retry globally within the table for values/labels split over a
            # line; the row estimate still makes the choice deterministic.
            matching = [w for w in words if w["index"] not in used and w["index"] not in {x["index"] for x in selected} and token_matches(token, w["text"])]
        if not matching:
            continue
        chosen = min(matching, key=lambda w: (abs((w["bbox"][1] + w["bbox"][3]) / 2 - expected_y), w["bbox"][0]))
        selected.append(chosen)
        last_x = chosen["bbox"][2]

    # Some SEC PDFs split the final digit of a table value into a tiny text
    # fragment directly below the main word (for example ``281,72`` + ``4``).
    # Rejoin only deterministic numeric continuations with tight vertical and
    # horizontal geometry; never infer a digit from an expected value.
    for anchor in list(selected):
        if not re.fullmatch(r"[$€£¥₹₽₩]?[-+−()\d,\.\s]+", anchor["text"].strip()):
            continue
        current = anchor
        while True:
            ax0, ay0, ax1, ay1 = current["bbox"]
            continuations = [
                word
                for word in words
                if word["index"] not in used
                and word["index"] not in {item["index"] for item in selected}
                and re.fullmatch(r"[-+−()\d,\.]+", word["text"].strip())
                and 0.0 <= word["bbox"][1] - ay1 <= 3.0
                and (abs(word["bbox"][0] - ax0) <= 3.0 or abs(word["bbox"][2] - ax1) <= 3.0)
                and word["bbox"][2] - word["bbox"][0] <= max(8.0, (ax1 - ax0) * 0.5)
            ]
            if not continuations:
                break
            continuation = min(continuations, key=lambda word: (word["bbox"][1], word["bbox"][0]))
            continuation["_numeric_continuation"] = True
            selected.append(continuation)
            current = continuation
    for word in selected:
        used.add(word["index"])
    return selected


def _extract_middle_tables(middle_path: Path) -> dict[int, list[dict[str, Any]]]:
    payload = json.loads(middle_path.read_text(encoding="utf-8"))
    pages: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for page_index, page in enumerate(payload.get("pdf_info", [])):
        seen: set[str] = set()
        for block in iter_dicts(page):
            raw = block.get("html")
            if not isinstance(raw, str) or "<table" not in raw.lower():
                continue
            bbox = extract_bbox(block)
            key = _hash_payload([raw, bbox])
            if key in seen:
                continue
            seen.add(key)
            pages[page_index].append({"html": raw, "bbox": bbox, "parsed": parse_table_html(raw)})
    return dict(pages)


def _resolve_pdf(source_path: str, pdf_root: Path, shared_root: Path) -> Path | None:
    direct = Path(source_path)
    if direct.is_file():
        return direct
    for root in (pdf_root, shared_root, shared_root / "finquery_rag/backend/runtime/benchmark/financial_rag_v1/review-package/pdfs"):
        candidate = root / direct.name
        if candidate.is_file():
            return candidate
        if root.is_dir():
            matches = sorted(root.rglob(direct.name))
            if matches:
                return matches[0]
    return None


def _build_table(table: dict[str, Any], document_id: str, pdf_page: int, table_index: int, words: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = table["parsed"]
    bbox = table["bbox"] or [0.0, 0.0, 0.0, 0.0]
    grid = parsed["grid"]
    header_texts = [cell["raw_text"] for cell in parsed["cells"] if cell.get("header")]
    # Header cells are not consistently tagged by MinerU, so include the
    # first rows containing dates/years as deterministic header candidates.
    header_rows: list[int] = []
    for row_index, row in enumerate(grid[: min(8, len(grid))]):
        text = " ".join(cell["raw_text"] for cell in row if cell)
        if _period_from_text(text) or re.search(r"\b(?:year|quarter|month|as of|ended)\b", text, re.I):
            header_rows.append(row_index)
    if not header_rows and grid:
        header_rows = [0]
    header_by_col: defaultdict[int, list[str]] = defaultdict(list)
    period_by_col: dict[int, str] = {}
    for row_index in header_rows:
        for col_index, cell in enumerate(grid[row_index]):
            if not cell:
                continue
            text = str(cell["raw_text"])
            if text and text not in header_by_col[col_index]:
                header_by_col[col_index].append(text)
            period = _period_from_text(text)
            if period:
                for col in range(cell["grid_col"], cell["grid_col"] + cell["colspan"]):
                    period_by_col[col] = period
    table_words = [word for word in words if _inside(word, bbox, 0.5)]
    local_words = [word for word in words if _inside(word, bbox, 80.0)]
    used: set[int] = set()
    bands = _column_bands(bbox, parsed["column_count"])
    adapter_cells: list[dict[str, Any]] = []
    for cell in parsed["cells"]:
        row_index = int(cell["grid_row"])
        col_index = int(cell["grid_col"])
        native_words = _match_native_words(cell["raw_text"], table_words, row_index, parsed["row_count"], bbox, used)
        cell_bbox = _union_bbox(native_words)
        if cell_bbox is None:
            span = bands[col_index : col_index + int(cell["colspan"])]
            cell_bbox = [span[0][0], bbox[1] + row_index * (bbox[3] - bbox[1]) / max(parsed["row_count"], 1), span[-1][2], bbox[1] + (row_index + int(cell["rowspan"])) * (bbox[3] - bbox[1]) / max(parsed["row_count"], 1)] if span else None
        header_path = []
        for header_row in header_rows:
            if header_row >= len(grid):
                continue
            header_cell = grid[header_row][min(col_index, len(grid[header_row]) - 1)] if grid[header_row] else None
            if header_cell and header_cell["raw_text"] and header_cell["raw_text"] not in header_path:
                header_path.append(header_cell["raw_text"])
        period = period_by_col.get(col_index)
        if period is None:
            for col in range(col_index, col_index + int(cell["colspan"])):
                period = period_by_col.get(col)
                if period:
                    break
        native_parts: list[str] = []
        for word in sorted(native_words, key=lambda item: (item["bbox"][1], item["bbox"][0])):
            if word.get("_numeric_continuation") and native_parts:
                native_parts[-1] += word["text"]
            else:
                native_parts.append(word["text"])
        native_text = " ".join(native_parts)
        identity_base = [document_id, pdf_page, table_index, row_index, col_index, cell["raw_text"], cell["rowspan"], cell["colspan"]]
        cell_id = "cell:" + hashlib.sha256(json.dumps(identity_base, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        adapter_cells.append({
            "cell_id": cell_id, "row_index": row_index, "column_index": col_index,
            "rowspan": cell["rowspan"], "colspan": cell["colspan"],
            "raw_text": cell["raw_text"], "normalized_text": _norm(cell["raw_text"]),
            "header_path": header_path, "normalized_period": period,
            "period_kind": _period_kind(" ".join(header_path)), "cell_bbox": cell_bbox,
            "mineru_text": cell["raw_text"], "native_words": native_words,
            "native_text": native_text, "resolved_text": native_text or cell["raw_text"],
            "text_source": "pymupdf_native" if native_words else "mineru_table_text",
            "alignment_confidence": round(min(1.0, len(native_words) / max(1, len(_tokens(cell["raw_text"])))), 4),
            "parsed_numeric": _numeric_values(native_text or cell["raw_text"]),
            "scale_candidates": sorted(set(re.findall(r"(?:in|dollars in|amounts in)\s+(?:millions?|thousands?|billions?)|\b(?:millions?|thousands?|billions?)\b", " ".join(header_path), re.I))),
        })
    table_signature = [document_id, pdf_page, bbox, [[c["raw_text"], c["row_index"], c["column_index"]] for c in adapter_cells]]
    table_id = "table:" + hashlib.sha256(json.dumps(table_signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rows: list[dict[str, Any]] = []
    for row_index in range(parsed["row_count"]):
        row_cells = [cell for cell in adapter_cells if cell["row_index"] == row_index]
        row_bbox = _union_bbox([{"bbox": cell["cell_bbox"]} for cell in row_cells if cell.get("cell_bbox")])
        row_text = " | ".join(cell["resolved_text"] for cell in sorted(row_cells, key=lambda item: item["column_index"]))
        row_id = "row:" + hashlib.sha256(json.dumps([table_id, row_index, _norm(row_text)], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        for cell in row_cells:
            cell["table_fragment_id"] = table_id
            cell["row_id"] = row_id
        rows.append({"row_id": row_id, "row_index": row_index, "row_bbox": row_bbox, "raw_text": row_text, "cell_ids": [cell["cell_id"] for cell in sorted(row_cells, key=lambda item: item["column_index"])], "metric_text": row_cells[0]["resolved_text"] if row_cells else ""})
    table_text = " ".join(str(cell.get("raw_text") or "") for cell in parsed["cells"])
    native_local_text = " ".join(word["text"] for word in local_words)
    table_scale_candidates = sorted(set(re.findall(r"(?:in|dollars in|amounts in)\s+(?:millions?|thousands?|billions?)|\b(?:millions?|thousands?|billions?)\b", table_text + " " + native_local_text, re.I)))
    return {"table_fragment_id": table_id, "document_id": document_id, "pdf_page": pdf_page, "table_index": table_index, "table_bbox": bbox, "parser_backend": "mineru_hybrid_high", "rows": rows, "cells": adapter_cells, "header_texts": header_texts, "periods": sorted({cell["normalized_period"] for cell in adapter_cells if cell.get("normalized_period")}), "scale_candidates": table_scale_candidates, "source_lineage": {"document_id": document_id, "pdf_page": pdf_page, "table_bbox": bbox}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    probe_manifest = json.loads((args.probe / "probe-input-manifest.json").read_text(encoding="utf-8"))
    middle_path = args.runtime / "hybrid_high" / "probe-input-87-pages" / "hybrid_auto" / "probe-input-87-pages_middle.json"
    if not middle_path.is_file():
        matches = sorted((args.runtime / "hybrid_high").rglob("*_middle.json"))
        if not matches:
            raise FileNotFoundError("sealed_hybrid_middle_missing")
        middle_path = matches[0]
    pages = _extract_middle_tables(middle_path)
    shared_root = ROOT.parents[4]
    predictions: list[dict[str, Any]] = []
    unresolved_pages = []
    table_count = row_count = cell_count = 0
    seen_tables: set[str] = set()
    seen_rows: set[str] = set()
    seen_cells: set[str] = set()
    for manifest_row in sorted(probe_manifest.get("records", []), key=lambda item: int(item["probe_page_index"])):
        page_index = int(manifest_row["probe_page_index"])
        source_path = _resolve_pdf(str(manifest_row.get("source_path", "")), args.pdf_root, shared_root)
        page_tables: list[dict[str, Any]] = []
        if source_path is not None:
            words = _page_words(source_path, int(manifest_row["pdf_page"]))
            for table_index, table in enumerate(pages.get(page_index, [])):
                if not table.get("bbox"):
                    continue
                page_tables.append(_build_table(table, str(manifest_row["document_id"]), int(manifest_row["pdf_page"]), table_index, words))
        if not page_tables:
            unresolved_pages.append(page_index)
        for table in page_tables:
            table_count += 1
            seen_tables.add(table["table_fragment_id"])
            row_count += len(table["rows"])
            cell_count += len(table["cells"])
            seen_rows.update(row["row_id"] for row in table["rows"])
            seen_cells.update(cell["cell_id"] for cell in table["cells"])
        predictions.append({"probe_page_index": page_index, "document_id": manifest_row["document_id"], "pdf_page": manifest_row["pdf_page"], "source_path": str(source_path) if source_path else None, "tables": page_tables})
    all_tables = [table for page in predictions for table in page["tables"]]
    all_cells = [cell for table in all_tables for cell in table["cells"]]
    manifest = {"prediction_page_count": len(predictions), "table_count": table_count, "row_count": row_count, "cell_count": cell_count, "unresolved_page_count": len(unresolved_pages), "unresolved_pages": unresolved_pages, "structural_backend": "mineru_hybrid_high", "text_source_priority": ["pymupdf_native", "mineru_table_text", "mineru_ocr"], "table_identity_hash": _hash_payload(sorted(seen_tables)), "row_identity_hash": _hash_payload(sorted(seen_rows)), "cell_identity_hash": _hash_payload(sorted(seen_cells)), "duplicate_table_id_count": table_count - len(seen_tables), "duplicate_row_id_count": row_count - len(seen_rows), "duplicate_cell_id_count": cell_count - len(seen_cells), "production_index_writes": 0}
    protocol = {"gate": "pdf_retrieval_v4_gate_02", "evaluation_type": "post_benchmark_iterative_evaluation", "code_commit": args.code_commit, "structural_backend": "mineru_hybrid_high", "adapter": "automatic_mineru_html_pymupdf_native_alignment", "binding_precedence": ["native_word_center_or_overlap", "same_text_line_and_column_order", "mineru_text_fallback"], "input_middle_sha256": sha256_file(middle_path), "probe_manifest_sha256": sha256_file(args.probe / "probe-input-manifest.json"), "runtime_oracle_reads": 0, "runtime_question_reads": 0, "expected_value_reads": 0, "adapter_builds": 1, "index_builds": 0, "retrieval_runs": 0, "production_index_writes": 0, "per_record_backend_selection": False, "forbidden": ["oracle", "gold", "expected_value", "question", "case_id", "retrieval", "index", "reranker", "answer_generation"]}
    _write(args.out / "adapter-protocol.json", protocol)
    _write(args.out / "adapter-input-integrity.json", {"probe_manifest_sha256": sha256_file(args.probe / "probe-input-manifest.json"), "hybrid_middle_sha256": sha256_file(middle_path), "prediction_page_count": len(predictions), "source_pdf_hashes": {str(page["source_path"]): sha256_file(Path(page["source_path"])) for page in predictions if page.get("source_path")}})
    _write(args.out / "structured-adapter-manifest.json", manifest)
    _write(args.out / "structured-adapter-identity-integrity.json", {"table_identity_conflicts": 0, "row_identity_conflicts": 0, "cell_identity_conflicts": 0, "duplicate_table_ids": manifest["duplicate_table_id_count"], "duplicate_row_ids": manifest["duplicate_row_id_count"], "duplicate_cell_ids": manifest["duplicate_cell_id_count"], "cells_without_source_lineage": sum(not cell.get("cell_bbox") for cell in all_cells), "native_numeric_loss_count": sum(not cell.get("parsed_numeric") for cell in all_cells)})
    prediction_payload = {"prediction_count": len(predictions), "pages": predictions}
    _write(args.out / "structured-adapter-predictions.json", prediction_payload)
    prediction_path = args.out / "structured-adapter-predictions.json"
    seal = {"prediction_count": len(predictions), "protocol_hash": sha256_file(args.out / "adapter-protocol.json"), "input_manifest_hash": sha256_file(args.out / "structured-adapter-manifest.json"), "prediction_hash": sha256_file(prediction_path), "predictions_sealed": True, "runtime_oracle_reads": 0, "runtime_governance_reads": 0, "expected_value_reads": 0, "labels_read_before_seal": 0, "index_builds": 0, "retrieval_runs": 0}
    _write(args.out / "adapter-prediction-seal.json", seal)
    _write(args.out / "adapter-acceptance.json", {"gate": "pdf_retrieval_v4_gate_02", "prediction_sealed": True, "decision": "pending_posthoc_scoring", "next_gate": "score_unified_structured_adapter", "adapter_builds": 1, "index_builds": 0, "retrieval_runs": 0, "runtime_oracle_reads": 0, "runtime_governance_reads": 0, "production_index_writes": 0, "production_behavior_changed": False, "production_switch_allowed": False})
    print(json.dumps({"prediction_pages": len(predictions), "tables": table_count, "rows": row_count, "cells": cell_count, "unresolved_pages": unresolved_pages}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
