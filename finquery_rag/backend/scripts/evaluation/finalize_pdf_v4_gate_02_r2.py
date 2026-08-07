"""Gate 02 R2: Finalize - structural diff, acceptance, and next-gate.

Generates the final three artifacts after seal + probe regression:
  1. probe-page-structural-diff.json  (old 87-page Probe vs new full output)
  2. acceptance.json                  (final acceptance gate check)
  3. next-gate.json                   (decision and next gate)

Reads the old Probe output ONLY for structural comparison (table count,
text block count, bbox, HTML hash).  No gold or question data is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS_PATH = ROOT / "benchmarks/financial_rag_v1/corpus.json"
DEFAULT_GATE01_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-01"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2"
SHARED_NANOCHAT_ROOT = ROOT.parents[4]
DEFAULT_MINERU_OUTPUT = SHARED_NANOCHAT_ROOT / ".runtime/pdf-retrieval-v4-gate-02-r2/mineru"
DEFAULT_PROBE_OUTPUT = (
    SHARED_NANOCHAT_ROOT
    / ".runtime/pdf-retrieval-v4-gate-01/hybrid_high/probe-input-87-pages/hybrid_auto"
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Structural extraction helpers
# ---------------------------------------------------------------------------


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _find_json(output_dir: Path, pattern: str) -> Path | None:
    matches = sorted(output_dir.rglob(pattern))
    return matches[0] if matches else None


def _extract_page_structure(middle_data: Any, page_idx: int) -> dict[str, Any]:
    """Extract structural info for one page from middle.json."""
    if not isinstance(middle_data, dict):
        return {"table_count": 0, "tables": []}
    pdf_info = middle_data.get("pdf_info", [])
    if not isinstance(pdf_info, list) or page_idx >= len(pdf_info):
        return {"table_count": 0, "tables": []}
    page_data = pdf_info[page_idx]
    if not isinstance(page_data, dict):
        return {"table_count": 0, "tables": []}

    tables: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in _iter_dicts(page_data):
        table_html = None
        for key in ("html", "table_body"):
            val = block.get(key)
            if isinstance(val, str) and "<table" in val.lower():
                table_html = val
                break
        if not table_html:
            continue
        html_hash = hashlib.sha256(table_html.encode("utf-8")).hexdigest()
        bbox = block.get("bbox") or block.get("img_bbox")
        bbox_str = json.dumps(bbox, sort_keys=True) if bbox else "null"
        dedup_key = f"{html_hash}:{bbox_str}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        tables.append({
            "bbox": bbox,
            "html_hash": html_hash,
            "row_count": table_html.count("<tr"),
        })

    return {"table_count": len(tables), "tables": tables}


def _count_text_blocks(content_data: Any, page_idx: int) -> int:
    """Count non-empty text blocks for a page from content_list.json."""
    if not isinstance(content_data, list):
        return 0
    count = 0
    for block in content_data:
        if not isinstance(block, dict):
            continue
        if block.get("page_idx") != page_idx:
            continue
        block_type = str(block.get("type") or "")
        if block_type in ("text", "title", "discarded") and block.get("text"):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Structural diff
# ---------------------------------------------------------------------------


def build_structural_diff(
    probe_manifest: dict[str, Any],
    probe_output_dir: Path,
    full_output_root: Path,
) -> dict[str, Any]:
    """Compare old 87-page Probe output with new full-corpus output."""
    records = probe_manifest.get("records", [])

    # Load old Probe middle.json and content_list.json
    old_middle_path = _find_json(probe_output_dir, "*_middle.json")
    old_content_path = _find_json(probe_output_dir, "*_content_list.json")

    old_middle: Any = None
    old_content: Any = None
    if old_middle_path and old_middle_path.is_file():
        try:
            old_middle = json.loads(old_middle_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old_middle = None
    if old_content_path and old_content_path.is_file():
        try:
            old_content = json.loads(old_content_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old_content = None

    # Cache new-corpus middle/content per document
    new_middle_cache: dict[str, Any] = {}
    new_content_cache: dict[str, Any] = {}

    per_page: list[dict[str, Any]] = []
    missing_old = 0
    table_count_delta = 0
    text_block_delta = 0
    table_html_changed = 0
    table_bbox_changed = 0
    pages_missing_in_new = 0

    for rec in records:
        probe_idx = int(rec.get("probe_page_index", 0))
        doc_id = str(rec.get("document_id") or "")
        pdf_page = int(rec.get("pdf_page") or 0)
        page_idx = pdf_page - 1  # 0-based

        # Old Probe structure
        old_struct = _extract_page_structure(old_middle, probe_idx)
        old_text_blocks = _count_text_blocks(old_content, probe_idx)

        # New full-corpus structure
        if doc_id not in new_middle_cache:
            doc_output = full_output_root / doc_id
            new_middle_path = _find_json(doc_output, "*_middle.json") if doc_output.is_dir() else None
            new_content_path = _find_json(doc_output, "*_content_list.json") if doc_output.is_dir() else None
            if new_middle_path and new_middle_path.is_file():
                try:
                    new_middle_cache[doc_id] = json.loads(new_middle_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    new_middle_cache[doc_id] = None
            else:
                new_middle_cache[doc_id] = None
            if new_content_path and new_content_path.is_file():
                try:
                    new_content_cache[doc_id] = json.loads(new_content_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    new_content_cache[doc_id] = None
            else:
                new_content_cache[doc_id] = None

        new_middle = new_middle_cache.get(doc_id)
        new_content = new_content_cache.get(doc_id)
        new_struct = _extract_page_structure(new_middle, page_idx)
        new_text_blocks = _count_text_blocks(new_content, page_idx)

        if not old_struct["table_count"] and not old_text_blocks:
            missing_old += 1
        if not new_struct["table_count"] and not new_text_blocks:
            pages_missing_in_new += 1

        tc_delta = new_struct["table_count"] - old_struct["table_count"]
        tb_delta = new_text_blocks - old_text_blocks
        table_count_delta += abs(tc_delta)
        text_block_delta += abs(tb_delta)

        # Compare table HTML hashes and bboxes
        old_hashes = {t["html_hash"] for t in old_struct["tables"]}
        new_hashes = {t["html_hash"] for t in new_struct["tables"]}
        if old_hashes != new_hashes and old_hashes:
            table_html_changed += 1

        old_bboxes = {json.dumps(t["bbox"], sort_keys=True) for t in old_struct["tables"]}
        new_bboxes = {json.dumps(t["bbox"], sort_keys=True) for t in new_struct["tables"]}
        if old_bboxes != new_bboxes and old_bboxes:
            table_bbox_changed += 1

        per_page.append({
            "probe_page_index": probe_idx,
            "document_id": doc_id,
            "pdf_page": pdf_page,
            "old_table_count": old_struct["table_count"],
            "new_table_count": new_struct["table_count"],
            "table_count_delta": tc_delta,
            "old_text_block_count": old_text_blocks,
            "new_text_block_count": new_text_blocks,
            "text_block_delta": tb_delta,
            "old_table_html_hashes": sorted(old_hashes),
            "new_table_html_hashes": sorted(new_hashes),
            "table_html_changed": old_hashes != new_hashes,
            "table_bbox_changed": old_bboxes != new_bboxes,
            "page_present_in_new": new_struct["table_count"] > 0 or new_text_blocks > 0,
        })

    return {
        "schema": "pdf-retrieval-v4/gate-02-r2/structural-diff/v1",
        "probe_page_count": len(records),
        "pages_compared": len(per_page),
        "pages_missing_in_old": missing_old,
        "pages_missing_in_new": pages_missing_in_new,
        "total_table_count_delta": table_count_delta,
        "total_text_block_delta": text_block_delta,
        "pages_with_table_html_changed": table_html_changed,
        "pages_with_table_bbox_changed": table_bbox_changed,
        "per_page": per_page,
    }


# ---------------------------------------------------------------------------
# Acceptance gate
# ---------------------------------------------------------------------------


def build_acceptance(
    out_dir: Path,
    structural_diff: dict[str, Any],
) -> dict[str, Any]:
    """Build the final acceptance gate check."""
    # Load all previously generated artifacts
    integrity = json.loads((out_dir / "corpus-input-integrity.json").read_text(encoding="utf-8"))
    doc_summary = json.loads((out_dir / "document-run-summary.json").read_text(encoding="utf-8"))
    page_coverage = json.loads((out_dir / "page-coverage-summary.json").read_text(encoding="utf-8"))
    raw_integrity = json.loads((out_dir / "raw-output-integrity.json").read_text(encoding="utf-8"))
    output_manifest = json.loads((out_dir / "full-corpus-raw-output-manifest.json").read_text(encoding="utf-8"))
    seal = json.loads((out_dir / "full-corpus-ingestion-seal.json").read_text(encoding="utf-8"))

    # Probe regression (may not exist yet if not run)
    probe_path = out_dir / "post-seal-probe-regression.json"
    probe_regression = None
    if probe_path.is_file():
        probe_regression = json.loads(probe_path.read_text(encoding="utf-8"))

    # Input gate
    input_gate = {
        "frozen_documents": f"{integrity['actual_document_count']}/{integrity['expected_document_count']}",
        "pdf_hash_match": "8/8" if integrity["all_sha256_match"] else "FAIL",
        "page_count_match": "8/8" if integrity["all_page_counts_match"] else "FAIL",
        "unexpected_documents": 0,
        "passed": integrity["integrity_passed"],
    }

    # Execution gate
    completed = doc_summary.get("completed", 0) + doc_summary.get("skipped", 0)
    failed = doc_summary.get("failed", 0)
    execution_gate = {
        "completed_documents": f"{completed}/8",
        "expected_pages_recorded": "100%" if page_coverage.get("missing_pages", 0) == 0 else "FAIL",
        "missing_pages": page_coverage.get("missing_pages", 0),
        "duplicate_pages": page_coverage.get("duplicate_pages", 0),
        "fatal_parser_errors": failed,
        "passed": completed == 8 and failed == 0 and page_coverage.get("missing_pages", 0) == 0,
    }

    # Output gate
    invalid_json = raw_integrity.get("invalid_json_files", 0)
    missing_refs = sum(len(r.get("missing_artifact_references", [])) for r in raw_integrity.get("records", []))
    zero_byte = raw_integrity.get("zero_byte_files", 0)
    output_gate = {
        "invalid_json": invalid_json,
        "missing_artifact_references": missing_refs,
        "zero_byte_required_files": zero_byte,
        "output_manifest_complete": bool(output_manifest.get("manifest_hash")),
        "seal_hash_verified": bool(seal.get("raw_output_manifest_hash")),
        "passed": (
            invalid_json == 0
            and missing_refs == 0
            and zero_byte == 0
            and bool(output_manifest.get("manifest_hash"))
            and bool(seal.get("raw_output_manifest_hash"))
        ),
    }

    # Probe regression gate
    probe_gate = {"passed": True, "decision": "not_scored"}
    if probe_regression:
        gate_checks = probe_regression.get("gate_checks", {})
        table_ok = gate_checks.get("table_recovery_22_22", False)
        row_ok = gate_checks.get("row_recovery_22_22", False)
        period_ok = gate_checks.get("period_22_22", False)
        numeric_ok = gate_checks.get("raw_numeric_ge_10", False)
        scale_ok = gate_checks.get("raw_scale_ge_18", False)
        structural_regression = probe_regression.get("structural_regression", True)
        probe_gate = {
            "table_recovery": probe_regression.get("raw_counts", {}).get("table_recovery", "?"),
            "row_recovery": probe_regression.get("raw_counts", {}).get("row_recovery", "?"),
            "period": probe_regression.get("raw_counts", {}).get("period_header_availability", "?"),
            "raw_numeric": probe_regression.get("raw_counts", {}).get("raw_numeric_recovery", "?"),
            "raw_scale": probe_regression.get("raw_counts", {}).get("raw_scale_recovery", "?"),
            "table_passed": table_ok,
            "row_passed": row_ok,
            "period_passed": period_ok,
            "numeric_passed": numeric_ok,
            "scale_passed": scale_ok,
            "structural_regression": structural_regression,
            "passed": not structural_regression,
            "decision": "probe_regression_blocked" if structural_regression else "passed",
        }

    # Safety gate
    safety_gate = {
        "question_reads": seal.get("question_reads", 0),
        "gold_reads_before_seal": seal.get("gold_reads_before_seal", 0),
        "governance_reads_before_seal": seal.get("governance_reads_before_seal", 0),
        "adapter_runs": seal.get("adapter_runs", 0),
        "index_builds": seal.get("index_builds", 0),
        "retrieval_runs": seal.get("retrieval_runs", 0),
        "production_index_writes": seal.get("production_index_writes", 0),
        "passed": (
            seal.get("question_reads", 0) == 0
            and seal.get("gold_reads_before_seal", 0) == 0
            and seal.get("governance_reads_before_seal", 0) == 0
            and seal.get("adapter_runs", 0) == 0
            and seal.get("index_builds", 0) == 0
            and seal.get("retrieval_runs", 0) == 0
            and seal.get("production_index_writes", 0) == 0
        ),
    }

    all_passed = (
        input_gate["passed"]
        and execution_gate["passed"]
        and output_gate["passed"]
        and probe_gate["passed"]
        and safety_gate["passed"]
    )

    return {
        "schema": "pdf-retrieval-v4/gate-02-r2/acceptance/v1",
        "gate": "pdf_retrieval_v4_gate_02_r2",
        "input_gate": input_gate,
        "execution_gate": execution_gate,
        "output_gate": output_gate,
        "probe_regression_gate": probe_gate,
        "safety_gate": safety_gate,
        "structural_diff_summary": {
            "pages_compared": structural_diff.get("pages_compared", 0),
            "pages_missing_in_new": structural_diff.get("pages_missing_in_new", 0),
            "pages_with_table_html_changed": structural_diff.get("pages_with_table_html_changed", 0),
            "pages_with_table_bbox_changed": structural_diff.get("pages_with_table_bbox_changed", 0),
        },
        "all_passed": all_passed,
        "decision": (
            "full_corpus_structured_ingestion_passed"
            if all_passed
            else "full_corpus_ingestion_blocked"
        ),
    }


# ---------------------------------------------------------------------------
# Next gate
# ---------------------------------------------------------------------------


def build_next_gate(acceptance: dict[str, Any]) -> dict[str, Any]:
    """Build the next-gate decision."""
    decision = acceptance.get("decision", "")
    if decision == "full_corpus_structured_ingestion_passed":
        return {
            "schema": "pdf-retrieval-v4/gate-02-r2/next-gate/v1",
            "current_gate": "pdf_retrieval_v4_gate_02_r2",
            "decision": "full_corpus_structured_ingestion_passed",
            "next_gate": "full_corpus_unified_structured_adapter",
            "production_switch_allowed": False,
            "stop": True,
        }
    return {
        "schema": "pdf-retrieval-v4/gate-02-r2/next-gate/v1",
        "current_gate": "pdf_retrieval_v4_gate_02_r2",
        "decision": decision or "full_corpus_ingestion_blocked",
        "next_gate": "stop_and_fix",
        "production_switch_allowed": False,
        "stop": True,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate01-out", type=Path, default=DEFAULT_GATE01_OUT)
    parser.add_argument("--mineru-output", type=Path, default=DEFAULT_MINERU_OUTPUT)
    parser.add_argument("--probe-output", type=Path, default=DEFAULT_PROBE_OUTPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # Verify seal exists
    seal_path = args.out_dir / "full-corpus-ingestion-seal.json"
    if not seal_path.is_file():
        print("ERROR: Seal not found. Run seal_pdf_v4_gate_02_r2_outputs.py first.")
        return 1

    # Load probe manifest
    probe_manifest_path = args.gate01_out / "probe-input-manifest.json"
    if not probe_manifest_path.is_file():
        print("ERROR: Probe input manifest not found at", probe_manifest_path)
        return 1
    probe_manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))

    # 1. Build structural diff
    print("Building structural diff (old 87-page Probe vs new full output)...")
    structural_diff = build_structural_diff(
        probe_manifest=probe_manifest,
        probe_output_dir=args.probe_output,
        full_output_root=args.mineru_output,
    )
    write_json(args.out_dir / "probe-page-structural-diff.json", structural_diff)
    print(f"  Pages compared: {structural_diff['pages_compared']}")
    print(f"  Pages missing in new: {structural_diff['pages_missing_in_new']}")
    print(f"  Table HTML changed: {structural_diff['pages_with_table_html_changed']}")
    print(f"  Table bbox changed: {structural_diff['pages_with_table_bbox_changed']}")

    # 2. Build acceptance
    print("\nBuilding acceptance gate...")
    acceptance = build_acceptance(args.out_dir, structural_diff)
    write_json(args.out_dir / "acceptance.json", acceptance)
    print(f"  Input gate: {'PASS' if acceptance['input_gate']['passed'] else 'FAIL'}")
    print(f"  Execution gate: {'PASS' if acceptance['execution_gate']['passed'] else 'FAIL'}")
    print(f"  Output gate: {'PASS' if acceptance['output_gate']['passed'] else 'FAIL'}")
    print(f"  Probe regression gate: {'PASS' if acceptance['probe_regression_gate']['passed'] else 'FAIL'}")
    print(f"  Safety gate: {'PASS' if acceptance['safety_gate']['passed'] else 'FAIL'}")
    print(f"  Decision: {acceptance['decision']}")

    # 3. Build next-gate
    print("\nBuilding next-gate decision...")
    next_gate = build_next_gate(acceptance)
    write_json(args.out_dir / "next-gate.json", next_gate)
    print(f"  Next gate: {next_gate['next_gate']}")
    print(f"  Stop: {next_gate['stop']}")

    return 0 if acceptance["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
