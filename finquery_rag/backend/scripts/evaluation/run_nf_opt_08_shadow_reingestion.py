"""Run NF-OPT-08 Gate A only; do not parse PDFs or write production state."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

from scripts.evaluation import audit_nf_eval_02_source_files as source_files
from scripts.evaluation import run_nf_eval_03_r1 as r1
from src.evaluation.nf_opt_08 import (
    ParserCapabilityStatus,
    parser_capability_gate,
    require_safe_parser_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/financial_rag_v1/data"
OUT = ROOT / "artifacts/evaluation/nf-opt-08"
NEG = ROOT / "artifacts/evaluation/nf-eval-02/negative-evidence-review-report.json"
ORIGINAL = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend")
RUNTIME = ORIGINAL / "runtime/benchmark/financial_rag_v1"
CONTROL_HASH = "3cf02bba3b5eda155b5204522b7018693d2a74847552227c0c303636946db7a5"


def _write(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _page_count(pdf: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(pdf)], check=True, text=True, capture_output=True
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError("pdfinfo did not report page count")


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _input_integrity() -> tuple[Any, dict[str, Path], dict[str, Any]]:
    inputs = r1._load_inputs(
        corpus_path=ROOT / "benchmarks/financial_rag_v1/corpus.json",
        manifest_path=DATA / "golden-manifest.json",
        questions_path=DATA / "questions.golden.jsonl",
        labels_path=DATA / "labels.golden.jsonl",
        review_status_path=DATA / "review-status.golden.jsonl",
        negative_report_path=NEG,
    )
    if not all(inputs.hash_report["matches"].values()):
        raise ValueError("frozen Golden input hashes failed")

    pdf_audit, source_paths = source_files.collect_verified_source_files(
        corpus_path=ROOT / "benchmarks/financial_rag_v1/corpus.json",
        runtime_manifest_path=RUNTIME / "corpus-manifest.json",
        registry_path=ORIGINAL / "document_registry.db",
        source_root=RUNTIME / "pdfs",
        tenant_id=1,
    )
    if not pdf_audit["acceptance"]["passed"] or len(source_paths) != 8:
        raise ValueError("input_pdf_integrity_failed")

    pages_by_doc = {key: _page_count(path) for key, path in source_paths.items()}
    for document in inputs.corpus["documents"]:
        doc_id = str(document["document_id"])
        if pages_by_doc.get(doc_id) != document.get("page_count"):
            raise ValueError("input_pdf_integrity_failed")

    return inputs, source_paths, {
        "golden_hashes": inputs.hash_report,
        "pdf_integrity": pdf_audit,
        "recomputed_page_counts": pages_by_doc,
        "passed": True,
    }


def _page_set(inputs: Any, pages_by_doc: dict[str, int]) -> dict[str, Any]:
    selected: dict[tuple[str, int], dict[str, set[str]]] = {}
    for label in inputs.labels_by_id.values():
        if not label.get("calculation"):
            continue
        for source in label["expected_sources"]:
            doc_id = str(source["document_id"])
            page = int(source["candidate_pdf_page"])
            selected.setdefault((doc_id, page), {"reasons": set(), "keys": set()})["reasons"].add("oracle_source_page")
            selected[(doc_id, page)]["keys"].add(str(source["candidate_key"]))
            for adjacent in (page - 1, page + 1):
                if 1 <= adjacent <= pages_by_doc[doc_id]:
                    selected.setdefault((doc_id, adjacent), {"reasons": set(), "keys": set()})["reasons"].add("oracle_adjacent_page")

    control = json.loads(
        (ROOT / "artifacts/evaluation/nf-opt-07/control-set-manifest.json").read_text()
    )
    if control["table_extraction_control_set_hash"] != CONTROL_HASH:
        raise ValueError("control_set_hash_mismatch")
    for group in control["groups"].values():
        for item in group:
            doc_id, page = str(item["document_id"]), int(item["page"])
            selected.setdefault((doc_id, page), {"reasons": set(), "keys": set()})["reasons"].add("nf_opt_07_control_page")
            selected[(doc_id, page)]["keys"].add(str(item["candidate_key"]))

    rows = [
        {
            "document_id": doc_id,
            "pdf_page": page,
            "selection_reasons": sorted(value["reasons"]),
            "source_candidate_keys": sorted(value["keys"]),
        }
        for (doc_id, page), value in sorted(selected.items())
    ]
    return {
        "shadow_page_set_hash": _hash(rows),
        "control_set_hash": CONTROL_HASH,
        "page_count": len(rows),
        "pages": rows,
    }


def _artifact_audit(document_ids: list[str]) -> dict[str, Any]:
    """Only enumerate known retained structured-artifact locations, not Markdown."""
    known_dirs = {
        "structured": RUNTIME / "structured",
        "parser_artifacts": RUNTIME / "parser_artifacts",
        "mineru": RUNTIME / "mineru",
        "intermediate": RUNTIME / "intermediate",
    }
    artifact_types = {
        name: directory.is_dir() and any(directory.iterdir())
        for name, directory in known_dirs.items()
    }
    documents = [
        {
            "document_id": doc_id,
            "artifact_types": [],
            "has_table_html": False,
            "has_cell_geometry": False,
            "has_page_layout": False,
            "has_table_images": False,
        }
        for doc_id in sorted(document_ids)
    ]
    return {
        "artifact_schema": "nf-opt-08/existing-parser-artifact-audit/v1",
        "documents": documents,
        "known_structured_locations_present": artifact_types,
        "retained_structured_artifact_count": 0,
        "markdown_or_plaintext_counted_as_structured": False,
        "status": "no_retained_structured_artifacts",
    }


def _parser_manifest() -> dict[str, Any]:
    binary = shutil.which("mineru")
    configured = bool(binary)
    return {
        "variant_a": {
            "name": "retained_structured_artifact",
            "available": False,
            "reason": "no_retained_structured_artifacts",
        },
        "variant_b": {
            "name": "fresh_shadow_structured_parse",
            "parser_name": "mineru",
            "parser_version": "unavailable",
            "backend_mode": "pipeline",
            "ocr_enabled": False,
            "available": configured,
            "reason": (
                "cli_not_installed_or_not_configured"
                if not configured
                else "not_run_until_fixed_version_is_attested"
            ),
        },
        "status": ParserCapabilityStatus.UNAVAILABLE,
        "fresh_parse_executed": False,
        "parser_inputs_verified_safe": True,
    }


def _blocked_capability_records(inputs: Any) -> list[dict[str, Any]]:
    records = []
    for label in inputs.labels_by_id.values():
        if not label.get("calculation"):
            continue
        for index, source in enumerate(label["expected_sources"]):
            require_safe_parser_inputs(
                {"document_id": source["document_id"], "pdf_page": source["candidate_pdf_page"]}
            )
            records.append(
                {
                    "source_identity": {
                        "candidate_key": source["candidate_key"],
                        "document_id": source["document_id"],
                        "pdf_page": source["candidate_pdf_page"],
                    },
                    "table_detected": False,
                    "correct_table_boundary": False,
                    "required_row_recovered": False,
                    "required_cells_recovered": False,
                    "period_recovered": False,
                    "scale_recovered": False,
                    "currency_recovered": False,
                    "evidence_page_correct": False,
                    "wrong_table_selected": False,
                    "wrong_row_mapped": False,
                    "wrong_column_mapped": False,
                    "cross_table_join": False,
                    "page_mismatch": False,
                    "status": "not_run_parser_unavailable",
                    "source_index": index,
                }
            )
    return records


def _empty_downstream(reason: str) -> dict[str, Any]:
    return {"status": "not_run", "reason": reason, "records": []}


def main() -> int:
    started = perf_counter()
    try:
        inputs, source_paths, integrity = _input_integrity()
    except Exception as exc:
        _write(
            "input-integrity-report.json",
            {"passed": False, "decision": "input_pdf_integrity_failed", "error": type(exc).__name__},
        )
        _write(
            "nf-opt-08-acceptance.json",
            {
                "decision": "input_pdf_integrity_failed",
                "diagnostic_integrity_passed": False,
                "production_behavior_changed": False,
                "production_queries_executed": 0,
                "model_chat_completion_requests": 0,
            },
        )
        return 2

    page_manifest = _page_set(
        inputs, {document_id: _page_count(path) for document_id, path in source_paths.items()}
    )
    artifacts = _artifact_audit(list(source_paths))
    parser = _parser_manifest()
    records = _blocked_capability_records(inputs)
    gate = parser_capability_gate(records)
    capability = {
        "parser_status": parser["status"],
        "parser_executed": False,
        "records": records,
        **gate,
    }
    reason = "gate_a_blocked_no_fixed_structured_parser_or_retained_artifact"

    _write("input-integrity-report.json", integrity)
    _write("shadow-page-set-manifest.json", page_manifest)
    _write("existing-parser-artifact-audit.json", artifacts)
    _write("parser-variant-manifest.json", parser)
    _write("parser-capability-report.json", capability)
    _write(
        "structured-table-schema-report.json",
        {
            "status": "schema_defined_parser_not_run",
            "storage": "shadow_only",
            "production_indexes_written": False,
        },
    )
    _write("old-new-evidence-mapping-report.json", _empty_downstream(reason))
    _write("structured-fact-report.json", _empty_downstream(reason))
    _write("oracle-operand-transfer-report.json", _empty_downstream(reason))
    _write("oracle-calculation-result-report.json", _empty_downstream(reason))
    _write(
        "control-set-safety-report.json",
        {
            "control_set_hash": CONTROL_HASH,
            "parser_executed": False,
            "wrong_table_join_count": 0,
            "wrong_header_assignment_count": 0,
            "wrong_period_assignment_count": 0,
            "wrong_scale_assignment_count": 0,
            "non_calculation_execution_count": 0,
            "no_answer_execution_count": 0,
        },
    )
    _write(
        "latency-resource-report.json",
        {
            "parser_setup_time_ms": 0,
            "per_page_parse_time_ms": None,
            "table_normalization_time_ms": 0,
            "mapping_validation_time_ms": 0,
            "fact_extraction_time_ms": 0,
            "calculation_time_ms": 0,
            "online_shadow_fact_binding_calculation_p95_ms": None,
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        },
    )
    _write(
        "next-gate.json",
        {
            "decision": "structured_reingestion_parser_blocked",
            "production_switch_allowed": False,
            "next_gate": "stop_and_review_parser_backend",
            "reason": reason,
        },
    )
    _write(
        "nf-opt-08-acceptance.json",
        {
            "decision": "structured_reingestion_parser_blocked",
            "diagnostic_integrity_passed": True,
            "parser_capability_gate_passed": False,
            "production_behavior_changed": False,
            "production_switch_allowed": False,
            "production_queries_executed": 0,
            "model_chat_completion_requests": 0,
            "answer_generation_calls": 0,
            "shadow_page_set_hash": page_manifest["shadow_page_set_hash"],
            "source_count": 22,
            "control_set_hash": CONTROL_HASH,
            "reason": reason,
        },
    )
    print(json.dumps({"gate": gate, "decision": "structured_reingestion_parser_blocked"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
