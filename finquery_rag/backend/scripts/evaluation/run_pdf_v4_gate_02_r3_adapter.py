"""Gate 02 R3: Full-corpus Unified Structured Adapter.

Converts Gate 02 R2 sealed MinerU output for all 8 frozen benchmark PDFs
(1,348 pages) into a unified Document → Page → Table Fragment → Row → Cell
structure with PyMuPDF native word alignment.

This gate ONLY builds structure.  It does NOT:
  - Build header graphs or metric hierarchies
  - Build evidence units or candidate views
  - Build indexes (BM25 / Dense)
  - Run retrieval, RRF, reranker, or answer generation
  - Read questions, gold, governance, or expected values before seal

Outputs evaluation artifacts to:
  artifacts/evaluation/pdf-retrieval-v4-gate-02-r3/

Large per-cell prediction data is stored as:
  adapter-predictions.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.adapter_integrity import (  # noqa: E402
    check_bbox_integrity,
    check_identity_integrity,
    check_page_integrity,
    check_text_integrity,
)
from src.pdf_retrieval_v4.full_corpus_adapter import (  # noqa: E402
    collect_document_metrics,
    collect_structure_metrics,
    run_full_corpus_adapter,
)

CORPUS_PATH = ROOT / "benchmarks/financial_rag_v1/corpus.json"
SHARED_NANOCHAT_ROOT = ROOT.parents[4]
DEFAULT_PDF_DIR = ROOT.parents[3] / "backend/runtime/benchmark/financial_rag_v1/review-package/pdfs"
DEFAULT_MINERU_OUTPUT = SHARED_NANOCHAT_ROOT / ".runtime/pdf-retrieval-v4-gate-02-r2/mineru"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r3"
DEFAULT_R2_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_corpus(corpus_path: Path) -> list[dict[str, Any]]:
    """Load frozen corpus manifest and return sorted document list."""
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    documents = data.get("documents") or data.get("corpus") or []
    return sorted(documents, key=lambda d: str(d.get("document_id") or ""))


def _build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    """Build the adapter protocol."""
    return {
        "gate": "pdf_retrieval_v4_gate_02_r3",
        "evaluation_type": "full_corpus_unified_structured_adapter",
        "code_commit": args.code_commit,
        "structural_backend": "mineru_hybrid_high",
        "adapter": "automatic_mineru_html_pymupdf_native_alignment",
        "binding_precedence": [
            "native_word_center_or_overlap",
            "same_text_line_and_column_order",
            "mineru_text_fallback",
        ],
        "input_r2_commit": args.r2_commit,
        "forbidden": [
            "oracle", "gold", "expected_value", "question", "case_id",
            "retrieval", "index", "reranker", "answer_generation",
            "header_graph", "evidence_unit", "candidate_view",
            "temporal_binding", "metric_hierarchy",
        ],
        "runtime_oracle_reads": 0,
        "runtime_question_reads": 0,
        "runtime_governance_reads": 0,
        "expected_value_reads": 0,
        "header_graph_runs": 0,
        "evidence_unit_builds": 0,
        "index_builds": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_config_modified": False,
        "production_switch_allowed": False,
        "bbox_tolerance": {
            "row_bbox_fallback": "estimated_from_table_bbox_and_row_index",
            "cell_bbox_fallback": "estimated_from_column_bands_and_row_height",
            "description": "Rows/cells without native word bbox use table geometry fallback",
        },
    }


def _build_input_integrity(
    documents: list[dict[str, Any]],
    mineru_output: Path,
    r2_out: Path,
) -> dict[str, Any]:
    """Build input integrity record."""
    doc_records = []
    for doc in documents:
        doc_id = str(doc["document_id"])
        doc_output = mineru_output / doc_id
        middle_path = None
        if doc_output.is_dir():
            matches = sorted(doc_output.rglob("*_middle.json"))
            if matches:
                middle_path = matches[0]

        record = {
            "document_id": doc_id,
            "expected_pdf_sha256": doc.get("pdf_sha256") or doc.get("sha256") or "",
            "expected_page_count": int(doc.get("page_count") or 0),
            "mineru_output_present": doc_output.is_dir(),
            "middle_json_present": middle_path is not None,
            "middle_json_sha256": _sha256_file(middle_path) if middle_path else "",
        }
        doc_records.append(record)

    # Load R2 seal for input manifest hash
    r2_seal_path = r2_out / "full-corpus-ingestion-seal.json"
    r2_seal_hash = ""
    if r2_seal_path.is_file():
        r2_seal_hash = _sha256_file(r2_seal_path)

    return {
        "documents": doc_records,
        "document_count": len(documents),
        "all_documents_present": all(r["mineru_output_present"] for r in doc_records),
        "all_middle_json_present": all(r["middle_json_present"] for r in doc_records),
        "r2_seal_sha256": r2_seal_hash,
    }


def _write_predictions_gz(pages: list[dict[str, Any]], path: Path) -> str:
    """Write per-page predictions as gzipped JSONL. Returns SHA-256 of file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for page in pages:
            f.write(json.dumps(page, ensure_ascii=False, separators=(",", ":")) + "\n")
    return _sha256_file(path)


def _build_native_alignment_audit(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Build native alignment audit."""
    total_cells = 0
    native_aligned = 0
    mineru_fallback = 0
    unresolved = 0
    by_document: dict[str, dict[str, int]] = {}

    for page in pages:
        doc_id = page["document_id"]
        if doc_id not in by_document:
            by_document[doc_id] = {
                "total_cells": 0,
                "native_aligned": 0,
                "mineru_fallback": 0,
                "unresolved": 0,
            }
        for table in page.get("tables", []):
            for cell in table.get("cells", []):
                total_cells += 1
                by_document[doc_id]["total_cells"] += 1
                source = cell.get("text_source", "")
                if source == "pymupdf_native":
                    native_aligned += 1
                    by_document[doc_id]["native_aligned"] += 1
                elif source in ("mineru_table_text", "mineru_ocr"):
                    mineru_fallback += 1
                    by_document[doc_id]["mineru_fallback"] += 1
                else:
                    unresolved += 1
                    by_document[doc_id]["unresolved"] += 1

    return {
        "total_cells": total_cells,
        "native_aligned": native_aligned,
        "mineru_fallback": mineru_fallback,
        "unresolved": unresolved,
        "by_document": dict(sorted(by_document.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--mineru-output", type=Path, default=DEFAULT_MINERU_OUTPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--r2-out", type=Path, default=DEFAULT_R2_OUT)
    parser.add_argument("--r2-commit", default="6f5990f")
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load frozen corpus
    documents = _load_corpus(args.corpus)
    total_expected_pages = sum(int(d.get("page_count") or 0) for d in documents)
    print(f"Loaded {len(documents)} documents, {total_expected_pages} expected pages")

    # 2. Build and write protocol
    protocol = _build_protocol(args)
    _write_json(args.out_dir / "gate-02-r3-protocol.json", protocol)

    # 3. Build and write input integrity
    input_integrity = _build_input_integrity(documents, args.mineru_output, args.r2_out)
    _write_json(args.out_dir / "input-integrity.json", input_integrity)
    if not input_integrity["all_middle_json_present"]:
        print("ERROR: Not all middle.json files are present")
        return 1

    # 4. Run full corpus adapter
    print("Running full-corpus adapter...")
    pages = run_full_corpus_adapter(
        documents=documents,
        mineru_output_root=args.mineru_output,
        pdf_dir=args.pdf_dir,
        shared_root=SHARED_NANOCHAT_ROOT,
    )
    print(f"  Pages: {len(pages)}")

    # 5. Write predictions as gzipped JSONL
    predictions_path = args.out_dir / "adapter-predictions.jsonl.gz"
    predictions_hash = _write_predictions_gz(pages, predictions_path)
    print(f"  Predictions hash: {predictions_hash[:16]}...")

    # 6. Collect structure metrics
    structure_metrics = collect_structure_metrics(pages)
    doc_metrics = collect_document_metrics(pages)
    print(f"  Tables: {structure_metrics['table_count']}")
    print(f"  Rows: {structure_metrics['row_count']}")
    print(f"  Cells: {structure_metrics['cell_count']}")

    # 7. Write structure metrics
    _write_json(args.out_dir / "full-corpus-structure-metrics.json", structure_metrics)
    _write_json(args.out_dir / "document-structure-metrics.json", doc_metrics)

    # 8. Write prediction manifest
    manifest = {
        "prediction_page_count": len(pages),
        "table_count": structure_metrics["table_count"],
        "row_count": structure_metrics["row_count"],
        "cell_count": structure_metrics["cell_count"],
        "predictions_hash": predictions_hash,
        "predictions_file": "adapter-predictions.jsonl.gz",
        "protocol_hash": _sha256_file(args.out_dir / "gate-02-r3-protocol.json"),
        "input_integrity_hash": _sha256_file(args.out_dir / "input-integrity.json"),
        "structure_metrics_hash": _sha256_file(args.out_dir / "full-corpus-structure-metrics.json"),
        "duplicate_table_id_count": structure_metrics["duplicate_table_id_count"],
        "duplicate_row_id_count": structure_metrics["duplicate_row_id_count"],
        "duplicate_cell_id_count": structure_metrics["duplicate_cell_id_count"],
        "table_identity_hash": structure_metrics["table_identity_hash"],
        "row_identity_hash": structure_metrics["row_identity_hash"],
        "cell_identity_hash": structure_metrics["cell_identity_hash"],
    }
    _write_json(args.out_dir / "adapter-prediction-manifest.json", manifest)

    # 9. Write native alignment audit
    native_audit = _build_native_alignment_audit(pages)
    _write_json(args.out_dir / "native-alignment-audit.json", native_audit)

    # 10. Write identity integrity
    page_integrity = check_page_integrity(pages, total_expected_pages)
    identity_integrity = check_identity_integrity(pages)
    bbox_integrity = check_bbox_integrity(pages)
    text_integrity = check_text_integrity(pages)
    _write_json(args.out_dir / "identity-integrity.json", {
        "page_integrity": page_integrity,
        "identity_integrity": identity_integrity,
        "bbox_integrity": bbox_integrity,
        "text_integrity": text_integrity,
        "all_passed": (
            page_integrity["passed"]
            and identity_integrity["passed"]
        ),
    })

    print("\nAdapter complete:")
    print(f"  Page integrity: {'PASS' if page_integrity['passed'] else 'FAIL'}")
    print(f"  Identity integrity: {'PASS' if identity_integrity['passed'] else 'FAIL'}")
    print(f"  BBox integrity: {'PASS' if bbox_integrity['passed'] else 'FAIL'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
