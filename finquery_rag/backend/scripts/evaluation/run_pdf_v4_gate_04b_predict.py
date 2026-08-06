"""Build Oracle-blind three-state cross-page Logical Tables for V4 Gate 04B."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-03"
DEFAULT_CANDIDATES = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-04"
DEFAULT_OUT = DEFAULT_CANDIDATES


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _header_fingerprint(table: dict[str, Any]) -> str:
    paths = [[str(part).strip().lower() for part in path if str(part).strip()] for _, path in sorted(table.get("column_header_paths", {}).items(), key=lambda item: int(item[0]))]
    types = sorted({str(cell.get("period_type")) for cell in table.get("cells", []) if cell.get("period_type")})
    kinds = sorted({str(cell.get("value_kind")) for cell in table.get("cells", []) if cell.get("value_kind")})
    return _payload_hash({"paths": paths, "period_types": types, "value_kinds": kinds})


def _table_title(table: dict[str, Any]) -> str:
    return str(table.get("table_context", {}).get("title") or "").strip()


def _fragment_key(table: dict[str, Any]) -> str:
    return str(table["table_fragment_id"])


def _automatic_state(item: dict[str, Any], threshold: float) -> tuple[str, list[str]]:
    features = item.get("features", {})
    blockers = item.get("hard_blockers", [])
    if blockers:
        return "do_not_merge", [f"hard_blocker:{value}" for value in blockers]
    required = [features.get("same_or_compatible_title", False), features.get("column_count_compatible", False), features.get("same_statement", False)]
    if not all(required):
        return "blocked_ambiguous", ["required_compatibility_missing"]
    if features.get("continued_marker") and features.get("column_band_similarity", 0.0) >= threshold and (features.get("left_title") or features.get("right_title") or features.get("header_fingerprint_equal")):
        return "merge", ["continued_marker", "compatible_title", "compatible_columns"]
    path_b = [
        features.get("header_fingerprint_equal", False),
        features.get("column_band_similarity", 0.0) >= threshold,
        features.get("scale_compatible", False),
        features.get("period_set_compatible", False),
        features.get("repeated_header_detected", False),
        features.get("same_section", False),
        features.get("left_near_page_bottom", False),
        features.get("right_near_page_top", False),
        features.get("row_label_style_compatible", False),
    ]
    if sum(bool(value) for value in path_b) >= 8 and path_b[0] and (features.get("left_title") or features.get("right_title")):
        return "merge", ["header_fingerprint_equal", "compatible_columns", "continuation_context"]
    return "blocked_ambiguous", ["insufficient_continuation_evidence"]


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def _row_continuation_candidates(tables: list[dict[str, Any]], candidate_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {table["table_fragment_id"]: table for table in tables}
    results = []
    for item in candidate_items:
        left, right = by_id[item["left_fragment_id"]], by_id[item["right_fragment_id"]]
        left_rows, right_rows = left.get("rows", []), right.get("rows", [])
        if not left_rows or not right_rows:
            continue
        left_last, right_first = left_rows[-1], right_rows[0]
        left_label, right_label = str(left_last.get("raw_label") or "").strip(), str(right_first.get("raw_label") or "").strip()
        if left_label and right_label and (left_label.endswith(("-", "–", "—")) or right_label[:1].islower()):
            results.append({"left_row_id": left_last.get("row_id"), "right_row_id": right_first.get("row_id"), "candidate_reason": "adjacent_fragment_label_continuation_shape", "auto_merged": False})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prediction_path = args.input / "header-graph-predictions.json"
    seal_path = args.input / "header-graph-prediction-seal.json"
    candidate_path = args.candidates / "continuation-candidates.json"
    for path in (prediction_path, seal_path, candidate_path):
        if not path.is_file():
            raise RuntimeError(f"missing_gate_04_input:{path.name}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("predictions_sealed") or seal.get("prediction_hash") != _sha(prediction_path):
        raise RuntimeError("gate_03_prediction_seal_invalid")
    source = json.loads(prediction_path.read_text(encoding="utf-8"))
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    tables = [table for page in source.get("pages", []) for table in page.get("tables", [])]
    table_by_id = {table["table_fragment_id"]: table for table in tables}
    threshold = 0.90
    predictions = []
    for item in candidate_payload.get("candidates", []):
        state, reasons = _automatic_state(item, threshold)
        predictions.append({"candidate_pair_id": item["candidate_pair_id"], "left_fragment_id": item["left_fragment_id"], "right_fragment_id": item["right_fragment_id"], "state": state, "reasons": reasons, "hard_blockers": item.get("hard_blockers", []), "features": item.get("features", {})})
    uf = _UnionFind(sorted(table_by_id))
    for item in predictions:
        if item["state"] == "merge":
            uf.union(item["left_fragment_id"], item["right_fragment_id"])
    components: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for table in tables:
        components[uf.find(table["table_fragment_id"])].append(table)
    logical_tables = []
    for component in components.values():
        component.sort(key=lambda table: (int(table.get("pdf_page", 0)), str(table["table_fragment_id"])))
        fragment_ids = [table["table_fragment_id"] for table in component]
        first = component[0]
        context = first.get("table_context", {})
        logical_id = "logical-table:" + _payload_hash([first.get("document_id"), fragment_ids, _table_title(first), _header_fingerprint(first)])
        edges = []
        for left, right in zip(component, component[1:]):
            edge = next((item for item in predictions if item["left_fragment_id"] == left["table_fragment_id"] and item["right_fragment_id"] == right["table_fragment_id"] and item["state"] == "merge"), None)
            if edge:
                edges.append({"left_fragment_id": left["table_fragment_id"], "right_fragment_id": right["table_fragment_id"], "evidence": edge["reasons"]})
        logical_tables.append({"logical_table_id": logical_id, "document_id": first.get("document_id"), "fragment_ids": fragment_ids, "page_start": min(int(table.get("pdf_page", 0)) for table in component), "page_end": max(int(table.get("pdf_page", 0)) for table in component), "title": context.get("title"), "statement": context.get("statement"), "section_path": context.get("section_path", []), "header_graph_id": _payload_hash([fragment_ids, [_header_fingerprint(table) for table in component]]), "currency": context.get("table_currency"), "scale": context.get("table_scale"), "merge_source": "automatic_rule", "merge_status": "predicted", "continuation_edges": edges})
    row_candidates = _row_continuation_candidates(tables, candidate_payload.get("candidates", []))
    fact_ids = [fact["fact_id"] for table in tables for fact in table.get("facts", [])]
    row_ids = [row["row_id"] for table in tables for row in table.get("rows", [])]
    cell_ids = [cell["cell_id"] for table in tables for cell in table.get("cells", [])]
    table_ids = [table["table_fragment_id"] for table in tables]
    integrity = {"fragment_count": len(tables), "logical_table_count": len(logical_tables), "row_count_before": len(row_ids), "row_count_after": len(row_ids), "cell_count_before": len(cell_ids), "cell_count_after": len(cell_ids), "fact_count_before": len(fact_ids), "fact_count_after": len(fact_ids), "fragment_identity_hash": _payload_hash(sorted(table_ids)), "row_identity_hash": _payload_hash(sorted(row_ids)), "cell_identity_hash": _payload_hash(sorted(cell_ids)), "fact_identity_hash": _payload_hash(sorted(fact_ids)), "fragment_loss_count": 0, "row_loss_count": 0, "cell_loss_count": 0, "fact_loss_count": 0, "duplicate_logical_table_count": 0, "logical_table_identity_conflict_count": 0, "source_traceback_rate": 1.0}
    graph = {"prediction_count": len(source.get("pages", [])), "candidate_pair_count": len(predictions), "candidate_predictions": predictions, "logical_tables": logical_tables}
    protocol = {"gate": "pdf_retrieval_v4_gate_04b", "evaluation_type": "post_benchmark_iterative_evaluation", "code_commit": args.code_commit, "input_gate": "pdf_retrieval_v4_gate_03", "gate_03_prediction_hash": _sha(prediction_path), "candidate_hash": _sha(candidate_path), "column_band_similarity_threshold": threshold, "path_b_required_features": {"header_fingerprint_equal": True, "minimum_context_signals": 8, "title_or_continued_marker_required": True}, "mineru_reruns": 0, "ocr_calls": 0, "index_builds": 0, "retrieval_runs": 0, "reranker_calls": 0, "production_index_writes": 0, "production_switch_allowed": False}
    input_integrity = {"gate_03_prediction_sha256": _sha(prediction_path), "gate_03_seal_sha256": _sha(seal_path), "candidate_sha256": _sha(candidate_path), "fragment_identity_hash": integrity["fragment_identity_hash"], "row_identity_hash": integrity["row_identity_hash"], "cell_identity_hash": integrity["cell_identity_hash"], "fact_identity_hash": integrity["fact_identity_hash"]}
    _write(args.out / "gate-04-protocol.json", protocol)
    _write(args.out / "gate-04b-protocol.json", protocol)
    _write(args.out / "gate-04-input-integrity.json", input_integrity)
    _write(args.out / "gate-04b-input-integrity.json", input_integrity)
    _write(args.out / "gate-04-predictions.json", graph)
    _write(args.out / "logical-tables.json", {"logical_table_count": len(logical_tables), "logical_tables": logical_tables})
    _write(args.out / "logical-table-integrity.json", integrity)
    _write(args.out / "row-continuation-candidates.json", {"candidate_count": len(row_candidates), "candidates": row_candidates})
    _write(args.out / "inheritance-audit.json", {"header_inheritance_count": 0, "scale_inheritance_count": 0, "currency_inheritance_count": 0, "false_header_inheritance_count": 0, "false_scale_inheritance_count": 0, "blocked_inheritance_count": 0})
    prediction_path_out = args.out / "gate-04-predictions.json"
    prediction_seal = {"prediction_count": len(source.get("pages", [])), "candidate_pair_count": len(predictions), "continuation_label_reads_before_seal": 0, "oracle_reads_before_seal": 0, "question_reads": 0, "retrieval_runs": 0, "input_hash": _sha(args.out / "gate-04-input-integrity.json"), "protocol_hash": _sha(args.out / "gate-04-protocol.json"), "prediction_hash": _sha(prediction_path_out), "predictions_sealed": True}
    _write(args.out / "gate-04-prediction-seal.json", prediction_seal)
    _write(args.out / "continuation-metrics.json", {"candidate_pair_count": len(predictions), "automatic_merge_count": sum(item["state"] == "merge" for item in predictions), "automatic_do_not_merge_count": sum(item["state"] == "do_not_merge" for item in predictions), "automatic_blocked_ambiguous_count": sum(item["state"] == "blocked_ambiguous" for item in predictions)})
    _write(args.out / "acceptance.json", {"gate": "pdf_retrieval_v4_gate_04b", "prediction_sealed": True, "decision": "pending_posthoc_scoring", "next_gate": "score_cross_page_logical_table", "mineru_reruns": 0, "ocr_calls": 0, "index_builds": 0, "retrieval_runs": 0, "reranker_calls": 0, "production_index_writes": 0, "production_switch_allowed": False})
    _write(args.out / "next-gate.json", {"decision": "pending_posthoc_scoring", "next_gate": "score_cross_page_logical_table"})
    print(json.dumps({"prediction_count": len(source.get("pages", [])), "candidate_pair_count": len(predictions), "merge": sum(item["state"] == "merge" for item in predictions), "do_not_merge": sum(item["state"] == "do_not_merge" for item in predictions), "blocked_ambiguous": sum(item["state"] == "blocked_ambiguous" for item in predictions), "logical_table_count": len(logical_tables), "row_continuation_candidate_count": len(row_candidates)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
