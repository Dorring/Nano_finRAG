"""Generate Oracle-blind adjacent-page continuation candidates for V4 Gate 04A."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-03"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-04"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _norm(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\b(?:continued|cont\.?|page\s+\d+)\b", " ", text)
    text = re.sub(r"[^a-z0-9%/\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _norm(value)))


def _text_fragments(table: dict[str, Any]) -> list[str]:
    values: list[str] = []
    context = table.get("table_context", {})
    values.extend([context.get("title"), context.get("statement")])
    values.extend(context.get("section_path") or [])
    values.extend(table.get("column_header_paths", {}).values())
    for node in table.get("header_nodes", []):
        values.append(node.get("raw_text"))
    for row in table.get("rows", []):
        values.extend([row.get("raw_label"), row.get("raw_text")])
    return [str(value) for value in values if str(value or "").strip()]


def _title(table: dict[str, Any]) -> str:
    return str(table.get("table_context", {}).get("title") or "").strip()


def _statement(table: dict[str, Any]) -> str:
    return str(table.get("table_context", {}).get("statement") or "").strip()


def _periods(table: dict[str, Any]) -> set[str]:
    return {str(cell.get("normalized_period")) for cell in table.get("cells", []) if cell.get("normalized_period")}


def _currency(table: dict[str, Any]) -> str | None:
    return table.get("table_context", {}).get("table_currency")


def _scale(table: dict[str, Any]) -> str | None:
    return table.get("table_context", {}).get("table_scale")


def _column_count(table: dict[str, Any]) -> int:
    columns = {int(index) for index in table.get("column_header_paths", {})}
    columns.update(int(cell.get("column_index", 0)) for cell in table.get("cells", []))
    return max(columns) + 1 if columns else 0


def _column_bands(table: dict[str, Any]) -> list[tuple[float, float]]:
    grouped: defaultdict[int, list[tuple[float, float]]] = defaultdict(list)
    for cell in table.get("cells", []):
        bbox = cell.get("cell_bbox")
        if not bbox or len(bbox) < 4:
            continue
        try:
            grouped[int(cell.get("column_index", 0))].append((float(bbox[0]), float(bbox[2])))
        except (TypeError, ValueError):
            continue
    table_bbox = table.get("table_context", {}).get("table_bbox")
    if table_bbox and len(table_bbox) >= 4:
        origin, end = float(table_bbox[0]), float(table_bbox[2])
    else:
        all_x = [coord for pair in grouped.values() for coord in pair for coord in pair]
        origin, end = (min(all_x), max(all_x)) if all_x else (0.0, 1.0)
    width = max(end - origin, 1.0)
    result = []
    for column in sorted(grouped):
        left = median(pair[0] for pair in grouped[column])
        right = median(pair[1] for pair in grouped[column])
        result.append(((left - origin) / width, (right - origin) / width))
    return result


def _band_similarity(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    errors = []
    for a, b in zip(left, right):
        errors.extend([abs(a[0] - b[0]), abs(a[1] - b[1])])
    return max(0.0, min(1.0, 1.0 - sum(errors) / max(len(errors), 1)))


def _header_fingerprint(table: dict[str, Any]) -> str:
    paths = [[str(part).strip().lower() for part in path if str(part).strip()] for _, path in sorted(table.get("column_header_paths", {}).items(), key=lambda item: int(item[0]))]
    period_types = sorted({str(cell.get("period_type")) for cell in table.get("cells", []) if cell.get("period_type")})
    value_kinds = sorted({str(cell.get("value_kind")) for cell in table.get("cells", []) if cell.get("value_kind")})
    return _payload_hash({"paths": paths, "period_types": period_types, "value_kinds": value_kinds})


def _row_style(table: dict[str, Any]) -> dict[str, Any]:
    rows = table.get("rows", [])
    numeric_by_row = defaultdict(int)
    for cell in table.get("cells", []):
        if cell.get("parsed_numeric"):
            numeric_by_row[int(cell.get("row_index", -1))] += 1
    return {"indent_levels": sorted(int(row.get("indent_level", 0)) for row in rows), "numeric_columns": sorted(numeric_by_row.values()), "roles": sorted(str(row.get("row_role")) for row in rows), "metric_depths": sorted(len(row.get("metric_path", [])) for row in rows)}


def _continued_marker(table: dict[str, Any]) -> bool:
    return any(re.search(r"\b(?:continued|cont\.?)\b", text, re.I) for text in _text_fragments(table))


def _title_compatible(left: str, right: str) -> bool:
    if not left or not right:
        return True
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return True
    incompatible = {"balance", "operations", "income", "cash", "flow", "equity"}
    return bool(a & b) and not bool((a ^ b) & incompatible)


def _page_position(table: dict[str, Any], page_tables: list[dict[str, Any]]) -> tuple[float, float]:
    bbox = table.get("table_context", {}).get("table_bbox")
    if not bbox or len(bbox) < 4:
        return 0.5, 0.5
    bottoms = [float(item["table_context"]["table_bbox"][3]) for item in page_tables if item.get("table_context", {}).get("table_bbox")]
    tops = [float(item["table_context"]["table_bbox"][1]) for item in page_tables if item.get("table_context", {}).get("table_bbox")]
    page_bottom, page_top = max(bottoms, default=float(bbox[3])), min(tops, default=float(bbox[1]))
    span = max(page_bottom - page_top, 1.0)
    return (float(bbox[3]) - page_top) / span, (float(bbox[1]) - page_top) / span


def _features(left: dict[str, Any], right: dict[str, Any], left_page_tables: list[dict[str, Any]], right_page_tables: list[dict[str, Any]]) -> dict[str, Any]:
    left_title, right_title = _title(left), _title(right)
    left_statement, right_statement = _statement(left), _statement(right)
    left_periods, right_periods = _periods(left), _periods(right)
    left_bottom, _ = _page_position(left, left_page_tables)
    _, right_top = _page_position(right, right_page_tables)
    scale_ok = not (_scale(left) and _scale(right) and _scale(left) != _scale(right))
    currency_ok = not (_currency(left) and _currency(right) and _currency(left) != _currency(right))
    period_ok = not (left_periods and right_periods and left_periods != right_periods)
    title_ok = _title_compatible(left_title, right_title)
    statement_ok = not (left_statement and right_statement and _norm(left_statement) != _norm(right_statement))
    columns_ok = _column_count(left) == _column_count(right)
    bands = _band_similarity(_column_bands(left), _column_bands(right))
    row_style_left, row_style_right = _row_style(left), _row_style(right)
    row_style_ok = row_style_left["roles"] == row_style_right["roles"] or bool(set(row_style_left["roles"]) & set(row_style_right["roles"]))
    header_equal = _header_fingerprint(left) == _header_fingerprint(right)
    repeated = header_equal or bool(_periods(left) & _periods(right))
    blockers: list[str] = []
    if not title_ok:
        blockers.append("conflicting_title")
    if not statement_ok:
        blockers.append("conflicting_statement")
    if not columns_ok:
        blockers.append("column_count_conflict")
    if bands == 0.0 and _column_bands(left) and _column_bands(right):
        blockers.append("column_order_or_geometry_conflict")
    if not period_ok:
        blockers.append("period_structure_conflict")
    if not scale_ok:
        blockers.append("scale_conflict")
    if not currency_ok:
        blockers.append("currency_conflict")
    right_first_row = (right.get("rows") or [{}])[0]
    return {"continued_marker": _continued_marker(left) or _continued_marker(right), "same_or_compatible_title": title_ok, "same_statement": statement_ok, "same_section": left.get("table_context", {}).get("section_path", []) == right.get("table_context", {}).get("section_path", []), "column_count_compatible": columns_ok, "column_band_similarity": round(bands, 6), "header_fingerprint_equal": header_equal, "repeated_header_detected": repeated, "scale_compatible": scale_ok, "currency_compatible": currency_ok, "period_set_compatible": period_ok, "row_label_style_compatible": row_style_ok, "left_near_page_bottom": left_bottom >= 0.5, "right_near_page_top": right_top <= 0.5, "left_row_count": len(left.get("rows", [])), "right_row_count": len(right.get("rows", [])), "right_first_row_role": right_first_row.get("row_role"), "right_first_row_label": right_first_row.get("raw_label"), "left_title": left_title, "right_title": right_title, "left_periods": sorted(left_periods), "right_periods": sorted(right_periods), "left_scale": _scale(left), "right_scale": _scale(right), "left_currency": _currency(left), "right_currency": _currency(right), "hard_blockers": blockers}


def _flatten(source: dict[str, Any]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    by_page: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for page in source.get("pages", []):
        document, pdf_page = str(page.get("document_id")), int(page.get("pdf_page", page.get("probe_page_index", 0)))
        for table in page.get("tables", []):
            item = dict(table)
            by_page[(document, pdf_page)].append(item)
    return dict(by_page)


def build_candidates(source: dict[str, Any]) -> list[dict[str, Any]]:
    by_page = _flatten(source)
    candidates: list[dict[str, Any]] = []
    for (document, page), left_tables in sorted(by_page.items()):
        right_tables = by_page.get((document, page + 1), [])
        for left in left_tables:
            for right in right_tables:
                features = _features(left, right, left_tables, right_tables)
                pair_id = "continuation-candidate:" + _payload_hash([document, left["table_fragment_id"], right["table_fragment_id"]])
                candidates.append({"candidate_pair_id": pair_id, "document_id": document, "left_fragment_id": left["table_fragment_id"], "left_page": page, "right_fragment_id": right["table_fragment_id"], "right_page": page + 1, "page_gap": 1, "features": features, "hard_blockers": features["hard_blockers"]})
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    required = ["header-graph-predictions.json", "header-graph-prediction-seal.json", "graph-integrity.json", "header-graph-input-integrity.json"]
    for name in required:
        if not (args.input / name).is_file():
            raise RuntimeError(f"missing_gate_03_input:{name}")
    prediction_path = args.input / "header-graph-predictions.json"
    seal = json.loads((args.input / "header-graph-prediction-seal.json").read_text(encoding="utf-8"))
    if not seal.get("predictions_sealed") or seal.get("prediction_hash") != _sha(prediction_path):
        raise RuntimeError("gate_03_prediction_seal_invalid")
    source = json.loads(prediction_path.read_text(encoding="utf-8"))
    candidates = build_candidates(source)
    protocol = {"gate": "pdf_retrieval_v4_gate_04a", "evaluation_type": "post_benchmark_iterative_evaluation", "code_commit": args.code_commit, "input_gate": "pdf_retrieval_v4_gate_03", "gate_03_prediction_hash": _sha(prediction_path), "page_gap": 1, "candidate_generation_recall_target": 1.0, "automatic_merge_thresholds": {"column_band_similarity": 0.90}, "mineru_reruns": 0, "ocr_calls": 0, "index_builds": 0, "retrieval_runs": 0, "reranker_calls": 0, "production_index_writes": 0, "production_switch_allowed": False}
    input_integrity = {"gate_03_prediction_sha256": _sha(prediction_path), "gate_03_seal_sha256": _sha(args.input / "header-graph-prediction-seal.json"), "graph_integrity_sha256": _sha(args.input / "graph-integrity.json"), "header_graph_input_integrity_sha256": _sha(args.input / "header-graph-input-integrity.json"), "table_identity_hash": _payload_hash(sorted(table["table_fragment_id"] for page in source.get("pages", []) for table in page.get("tables", []))), "row_identity_hash": _payload_hash(sorted(row["row_id"] for page in source.get("pages", []) for table in page.get("tables", []) for row in table.get("rows", []))), "cell_identity_hash": _payload_hash(sorted(cell["cell_id"] for page in source.get("pages", []) for table in page.get("tables", []) for cell in table.get("cells", []))), "fact_identity_hash": _payload_hash(sorted(fact["fact_id"] for page in source.get("pages", []) for table in page.get("tables", []) for fact in table.get("facts", [])))}
    review = [{"candidate_pair_id": item["candidate_pair_id"], "same_logical_table": None, "continuation_direction_correct": None, "header_inheritance_allowed": None, "scale_inheritance_allowed": None, "row_split_across_pages": None, "review_class": None, "review_status": "pending", "reviewer": None, "reviewed_at": None, "review_notes": None, "verified": False} for item in candidates]
    _write(args.out / "gate-04-protocol.json", protocol)
    _write(args.out / "gate-04a-protocol.json", protocol)
    _write(args.out / "gate-04-input-integrity.json", input_integrity)
    _write(args.out / "gate-04a-input-integrity.json", input_integrity)
    _write(args.out / "continuation-candidates.json", {"candidate_count": len(candidates), "candidates": candidates})
    _write(args.out / "continuation-review-package.json", {"candidate_count": len(review), "reviews": review, "review_instructions": ["inspect_adjacent_page_table_regions", "compare_title_header_scale_currency_and_row_pattern", "ambiguous_is_not_positive"]})
    _write(args.out / "acceptance.json", {"gate": "pdf_retrieval_v4_gate_04a", "candidate_count": len(candidates), "review_pending": len(review), "decision": "pending_manual_review", "mineru_reruns": 0, "ocr_calls": 0, "index_builds": 0, "retrieval_runs": 0, "production_index_writes": 0, "production_switch_allowed": False})
    _write(args.out / "next-gate.json", {"decision": "pending_manual_review", "next_gate": "pdf_retrieval_v4_gate_04b"})
    print(json.dumps({"candidate_count": len(candidates), "document_count": len({item["document_id"] for item in candidates}), "pending_reviews": len(review)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
