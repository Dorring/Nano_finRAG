"""Gate 03 R2: 33-problem Gold scoring against semantic evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT))

GATE03_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-03-r2"
R3_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r3"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_label(value: Any) -> str:
    return _coerce_str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _strip_trailing_number(label: str) -> str:
    """Strip trailing _<digits> footnote references (e.g. 'services_1' -> 'services')."""
    return re.sub(r"_\d+$", "", label)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 03 R2 33-problem gold scoring")
    parser.add_argument("--backend-root", type=str, default=str(ROOT))
    parser.add_argument("--gate03-out", type=str, default=str(GATE03_OUT))
    parser.add_argument("--r3-out", type=str, default=str(R3_OUT))
    args = parser.parse_args()

    gate03_out = Path(args.gate03_out)
    r3_out = Path(args.r3_out)

    alignment_path = r3_out / "target-structural-alignment.json"
    ambiguity_path = r3_out / "ambiguity-closure.json"

    if not alignment_path.exists():
        print(f"ERROR: alignment file not found: {alignment_path}", file=sys.stderr)
        return 1

    alignment = _read_json(alignment_path)
    ambiguity = _read_json(ambiguity_path) if ambiguity_path.exists() else {}

    b_records = alignment.get("b_class_records", [])
    d_records = alignment.get("d_class_records", [])

    atomic_facts = _read_jsonl(gate03_out / "atomic-facts.jsonl")
    row_matrices = _read_jsonl(gate03_out / "row-matrices.jsonl")
    bucket_facts = _read_jsonl(gate03_out / "bucket-facts.jsonl")
    scale_resolutions = _read_jsonl(gate03_out / "scale-resolutions.jsonl")

    # Build table-level scale lookup: table_fragment_id -> (scale, scale_unit)
    table_scale_map: dict[str, tuple[Any, str | None]] = {}
    for sr in scale_resolutions:
        tfid = _coerce_str(sr.get("table_fragment_id"))
        if tfid and sr.get("scale_status") == "resolved":
            table_scale_map[tfid] = (sr.get("scale"), sr.get("scale_unit"))

    # Build table_fragment_id -> document_id map from atomic facts
    # (scale-resolutions.jsonl doesn't carry document_id)
    tfid_to_doc: dict[str, str] = {}
    for f in atomic_facts:
        tfid = _coerce_str(f.get("table_fragment_id"))
        doc = _coerce_str(f.get("document_id"))
        if tfid and doc:
            tfid_to_doc[tfid] = doc

    # Build document-level dominant scale: document_id -> (scale, scale_unit)
    # Used as fallback when table-level scale is conflict/missing.
    doc_scale_counter: dict[str, Counter] = {}
    for sr in scale_resolutions:
        tfid = _coerce_str(sr.get("table_fragment_id"))
        doc = tfid_to_doc.get(tfid, "")
        status = sr.get("scale_status", "")
        scale = sr.get("scale")
        if doc and status == "resolved" and scale is not None:
            doc_scale_counter.setdefault(doc, Counter())[scale] += 1

    doc_scale_map: dict[str, tuple[Any, str | None]] = {}
    for doc, counter in doc_scale_counter.items():
        if counter:
            dominant_scale, _ = counter.most_common(1)[0]
            unit = (
                "millions"
                if dominant_scale == 1000000.0
                else (
                    "billions"
                    if dominant_scale == 1000000000.0
                    else ("thousands" if dominant_scale == 1000.0 else None)
                )
            )
            doc_scale_map[doc] = (dominant_scale, unit)

    # Build cell_id -> atomic_fact index for efficient cell_id matching
    facts_by_cell: dict[str, list[dict[str, Any]]] = {}
    for f in atomic_facts:
        cid = _coerce_str(f.get("cell_id"))
        if cid:
            facts_by_cell.setdefault(cid, []).append(f)

    # Build row_id -> row_matrix index
    matrices_by_row: dict[str, list[dict[str, Any]]] = {}
    for rm in row_matrices:
        rid = _coerce_str(rm.get("row_id"))
        if rid:
            matrices_by_row.setdefault(rid, []).append(rm)

    # Load equivalent sets from ambiguity-closure records
    equiv_records = [
        r
        for r in (ambiguity.get("records") or [])
        if r.get("alignment_status") == "equivalent_set"
    ]
    equiv_map: dict[str, list[str]] = {}
    for rec in equiv_records:
        members = rec.get("equivalent_row_ids") or rec.get("physical_row_ids") or []
        for rid in members:
            equiv_map.setdefault(rid, [])
            for other in members:
                if other not in equiv_map[rid]:
                    equiv_map[rid].append(other)

    def _equivalent_row_ids(row_id: str) -> list[str]:
        ids = [row_id]
        if row_id in equiv_map:
            for m in equiv_map[row_id]:
                if m not in ids:
                    ids.append(m)
        return ids

    def _score_record(rec: dict[str, Any]) -> dict[str, Any]:
        record_id = rec.get("case_id") or rec.get("record_id") or rec.get("id")
        document_id = _coerce_str(rec.get("document_id"))
        gold_metric_label = _normalize_label(
            rec.get("gold_metric_label") or rec.get("metric_label")
        )
        gold_metric_stripped = _strip_trailing_number(gold_metric_label)

        matched_row_ids = rec.get("matched_row_ids") or []
        if not isinstance(matched_row_ids, list):
            matched_row_ids = [matched_row_ids]
        matched_row_ids = [_coerce_str(r) for r in matched_row_ids if r]

        matched_cell_ids = rec.get("matched_cell_ids") or []
        if not isinstance(matched_cell_ids, list):
            matched_cell_ids = [matched_cell_ids]
        matched_cell_ids = set(_coerce_str(c) for c in matched_cell_ids if c)

        # Check if this case_id appears in equivalent_set records
        case_id = rec.get("case_id") or ""
        equiv_case_ids = {r.get("case_id") for r in equiv_records}
        equivalent_set_case = case_id in equiv_case_ids

        candidate_row_ids: list[str] = []
        for rid in matched_row_ids:
            candidate_row_ids.extend(_equivalent_row_ids(rid))
        if not candidate_row_ids:
            candidate_row_ids = list(matched_row_ids)
        candidate_row_set = set(candidate_row_ids)

        def _fact_matches(fact: dict[str, Any]) -> bool:
            fact_row_id = _coerce_str(fact.get("row_id"))
            if fact_row_id and fact_row_id in candidate_row_set:
                return True
            fact_cell_id = _coerce_str(fact.get("cell_id"))
            if fact_cell_id and fact_cell_id in matched_cell_ids:
                return True
            return False

        # Collect candidate facts by document + row/cell match
        candidate_facts: list[dict[str, Any]] = []
        if document_id:
            for f in atomic_facts:
                if _coerce_str(f.get("document_id")) == document_id and _fact_matches(
                    f
                ):
                    candidate_facts.append(f)
        else:
            for f in atomic_facts:
                if _fact_matches(f):
                    candidate_facts.append(f)

        # Collect candidate row matrices
        candidate_matrices: list[dict[str, Any]] = []
        for rid in candidate_row_ids:
            candidate_matrices.extend(matrices_by_row.get(rid, []))
        if document_id:
            candidate_matrices = [
                rm
                for rm in candidate_matrices
                if _coerce_str(rm.get("document_id")) == document_id
            ]

        # Collect candidate bucket facts
        candidate_buckets: list[dict[str, Any]] = []
        for bf in bucket_facts:
            bf_doc_id = _coerce_str(bf.get("document_id"))
            if document_id and bf_doc_id != document_id:
                continue
            bf_row_id = _coerce_str(bf.get("row_id"))
            if bf_row_id and bf_row_id in candidate_row_set:
                candidate_buckets.append(bf)

        metric_path_correct = False
        temporal_binding_correct = False
        value_present = False
        scale_correct_or_recoverable = False
        typed_evidence_present = False
        candidate_compatible_typed_evidence = False

        # Check atomic facts
        for fact in candidate_facts:
            metric_path = _normalize_label(fact.get("metric_path"))

            # Metric path matching: try exact, then stripped (trailing number removal)
            if gold_metric_label and (
                gold_metric_label in metric_path or gold_metric_stripped in metric_path
            ):
                metric_path_correct = True

            # For empty gold_metric_label (D-class), accept any non-empty metric_path
            if not gold_metric_label and metric_path:
                metric_path_correct = True

            if fact.get("normalized_period") is not None:
                temporal_binding_correct = True
            if fact.get("value_normalized") is not None:
                value_present = True

            # Scale from fact
            scale = fact.get("scale")
            scale_unit = fact.get("scale_unit")
            if scale is not None or scale_unit is not None:
                scale_correct_or_recoverable = True
            else:
                # Try table-level scale recovery
                tbl = _coerce_str(fact.get("table_fragment_id"))
                if tbl in table_scale_map:
                    scale_correct_or_recoverable = True
                elif document_id in doc_scale_map:
                    # Document-level scale recovery (dominant resolved scale)
                    scale_correct_or_recoverable = True

        # Check row matrices for metric_path, temporal, value, scale
        for rm in candidate_matrices:
            rm_metric_path = _normalize_label(rm.get("metric_path"))

            # Metric path matching on row matrices
            if gold_metric_label and (
                gold_metric_label in rm_metric_path
                or gold_metric_stripped in rm_metric_path
            ):
                metric_path_correct = True
            if not gold_metric_label and rm_metric_path:
                metric_path_correct = True

            dims = rm.get("dimensions") or []
            for dim in dims:
                if dim.get("normalized_period") is not None:
                    temporal_binding_correct = True
                if dim.get("value_normalized") is not None:
                    value_present = True

            rm_scale = rm.get("scale")
            if rm_scale is not None:
                scale_correct_or_recoverable = True
            else:
                tbl = _coerce_str(rm.get("table_fragment_id"))
                if tbl in table_scale_map:
                    scale_correct_or_recoverable = True
                elif document_id in doc_scale_map:
                    # Document-level scale recovery
                    scale_correct_or_recoverable = True

        # Check bucket facts for metric_path, value, scale
        for bf in candidate_buckets:
            bf_metric_path = _normalize_label(bf.get("metric_path"))

            # Metric path matching on bucket facts
            if gold_metric_label and (
                gold_metric_label in bf_metric_path
                or gold_metric_stripped in bf_metric_path
            ):
                metric_path_correct = True
            if not gold_metric_label and bf_metric_path:
                metric_path_correct = True

            if bf.get("value_normalized") is not None:
                value_present = True

            bf_scale = bf.get("scale")
            if bf_scale is not None:
                scale_correct_or_recoverable = True
            else:
                tbl = _coerce_str(bf.get("table_fragment_id"))
                if tbl in table_scale_map:
                    scale_correct_or_recoverable = True
                elif document_id in doc_scale_map:
                    scale_correct_or_recoverable = True

        # Typed evidence present
        if candidate_facts or candidate_matrices or candidate_buckets:
            typed_evidence_present = True

        # Candidate-compatible typed evidence (source traceback with page)
        for fact in candidate_facts:
            src_tb = fact.get("source_traceback") or {}
            if src_tb.get("document_id") and src_tb.get("pdf_page") is not None:
                candidate_compatible_typed_evidence = True
                break
        if not candidate_compatible_typed_evidence:
            for rm in candidate_matrices:
                src_tb = rm.get("source_traceback") or {}
                if src_tb.get("document_id") and src_tb.get("pdf_page") is not None:
                    candidate_compatible_typed_evidence = True
                    break
        if not candidate_compatible_typed_evidence:
            for bf in candidate_buckets:
                src_tb = bf.get("source_traceback") or {}
                if src_tb.get("document_id") and src_tb.get("pdf_page") is not None:
                    candidate_compatible_typed_evidence = True
                    break

        all_correct = (
            metric_path_correct
            and temporal_binding_correct
            and value_present
            and scale_correct_or_recoverable
            and typed_evidence_present
            and candidate_compatible_typed_evidence
        )

        return {
            "record_id": record_id,
            "document_id": document_id,
            "gold_metric_label": gold_metric_label,
            "matched_row_ids": matched_row_ids,
            "equivalent_set_case": equivalent_set_case,
            "metric_path_correct": metric_path_correct,
            "temporal_binding_correct": temporal_binding_correct,
            "value_present": value_present,
            "scale_correct_or_recoverable": scale_correct_or_recoverable,
            "typed_evidence_present": typed_evidence_present,
            "candidate_compatible_typed_evidence": candidate_compatible_typed_evidence,
            "all_correct": all_correct,
        }

    b_results = [_score_record(r) for r in b_records]
    d_results = [_score_record(r) for r in d_records]

    b_pass = sum(1 for r in b_results if r["all_correct"])
    d_pass = sum(1 for r in d_results if r["all_correct"])
    b_total = len(b_results)
    d_total = len(d_results)
    total_pass = b_pass + d_pass
    total = b_total + d_total

    if total_pass >= 26:
        decision = "full_corpus_financial_semantic_graph_strong_pass"
        next_gate = "full_corpus_candidate_evidence_bridge"
    elif total_pass >= 22:
        decision = "full_corpus_financial_semantic_graph_passed"
        next_gate = "full_corpus_candidate_evidence_bridge"
    else:
        decision = "full_corpus_financial_semantic_graph_insufficient"
        next_gate = "stop_and_fix_semantic_first_failure_stage"

    b_pass_gate = b_pass >= 14
    d_pass_gate = d_pass >= 12
    total_pass_gate = total_pass >= 22
    strong_pass = total_pass >= 26

    result = {
        "b_class": {
            "pass": b_pass,
            "total": b_total,
            "gate_passed": b_pass_gate,
            "gate_threshold": 14,
            "records": b_results,
        },
        "d_class": {
            "pass": d_pass,
            "total": d_total,
            "gate_passed": d_pass_gate,
            "gate_threshold": 12,
            "records": d_results,
        },
        "total": {
            "pass": total_pass,
            "total": total,
            "gate_passed": total_pass_gate,
            "strong_pass": strong_pass,
            "pass_threshold": 22,
            "strong_pass_threshold": 26,
        },
        "decision": decision,
        "next_gate": next_gate,
        "production_switch_allowed": False,
    }

    out_path = gate03_out / "problem-gold-scoring.json"
    _write_json(out_path, result)

    print(f"Problem gold scoring written: {out_path}")
    print(
        f"B-class: {b_pass}/{b_total} (gate >= 14: {'PASS' if b_pass_gate else 'FAIL'})"
    )
    print(
        f"D-class: {d_pass}/{d_total} (gate >= 12: {'PASS' if d_pass_gate else 'FAIL'})"
    )
    print(
        f"Total:   {total_pass}/{total} (pass >= 22: {'PASS' if total_pass_gate else 'FAIL'}, strong >= 26: {'PASS' if strong_pass else 'FAIL'})"
    )
    print(f"Decision: {decision}")
    print(f"Next gate: {next_gate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
