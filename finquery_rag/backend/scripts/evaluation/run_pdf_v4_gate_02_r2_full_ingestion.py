"""Gate 02 R2: Full benchmark corpus structured ingestion.

Runs MinerU Hybrid High on all 8 frozen benchmark PDFs (1348 pages),
producing deterministic per-document output with checkpoint support.

This gate ONLY runs MinerU.  It does NOT:
  - Build adapters
  - Build header graphs
  - Build evidence units
  - Build indexes
  - Run retrieval
  - Read questions, gold, or governance before seal

Outputs evaluation artifacts to:
  artifacts/evaluation/pdf-retrieval-v4-gate-02-r2/

Large raw MinerU output is stored in:
  artifacts/runtime/pdf-retrieval-v4-gate-02-r2/mineru/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.deterministic_output_manifest import (  # noqa: E402
    build_output_manifest,
)
from src.pdf_retrieval_v4.frozen_corpus_manifest import (  # noqa: E402
    load_corpus_manifest,
    verify_corpus_integrity,
)
from src.pdf_retrieval_v4.mineru_full_corpus_runner import (  # noqa: E402
    MinerUConfig,
    capture_runtime_environment,
    nvidia_snapshot,
    run_full_corpus,
)
from src.pdf_retrieval_v4.page_coverage import (  # noqa: E402
    build_page_status,
    summarize_page_coverage,
    write_page_status_gzip,
)

CORPUS_PATH = ROOT / "benchmarks/financial_rag_v1/corpus.json"
SHARED_NANOCHAT_ROOT = ROOT.parents[4]
DEFAULT_PDF_DIR = ROOT.parents[3] / "backend/runtime/benchmark/financial_rag_v1/review-package/pdfs"
DEFAULT_MINERU = SHARED_NANOCHAT_ROOT / ".runtime/mineru-venv-cu126/bin/mineru"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2"
DEFAULT_RUNTIME = SHARED_NANOCHAT_ROOT / ".runtime/pdf-retrieval-v4-gate-02-r2"
DEFAULT_MINERU_OUTPUT = DEFAULT_RUNTIME / "mineru"
RUNBOOK_PATH = ROOT / "docs/operations/runtime-environment-runbook.md"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--mineru", type=Path, default=DEFAULT_MINERU)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--mineru-output", type=Path, default=DEFAULT_MINERU_OUTPUT)
    parser.add_argument("--cuda-visible-devices", default="5")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--skip-mineru", action="store_true",
                        help="Skip MinerU execution (for testing artifacts only)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.mineru_output.mkdir(parents=True, exist_ok=True)
    tmpdir = args.runtime_dir / "finquery_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    config = MinerUConfig()

    # ------------------------------------------------------------------
    # 1. Write protocol
    # ------------------------------------------------------------------
    protocol = {
        "schema": "pdf-retrieval-v4/gate-02-r2/protocol/v1",
        "gate": "pdf_retrieval_v4_gate_02_r2",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "purpose": "full_benchmark_corpus_structured_ingestion",
        "mineru_only": True,
        "adapter_runs": 0,
        "header_graph_runs": 0,
        "evidence_unit_builds": 0,
        "index_builds": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "answer_generation_calls": 0,
        "question_reads": 0,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "expected_value_reads_before_seal": 0,
        "reference_answer_reads_before_seal": 0,
        "parameter_scan": False,
        "per_document_backend_selection": False,
        "per_page_backend_selection": False,
        "production_index_writes": 0,
        "production_config_modified": False,
        "production_switch_allowed": False,
        "fixed_config": config.to_dict(),
        "corpus_manifest": str(args.corpus),
        "runbook": {
            "path": str(RUNBOOK_PATH),
            "isolated_mineru_environment": str(args.mineru.parent.parent),
            "project_tmpdir": str(tmpdir),
            "cuda_visible_devices": args.cuda_visible_devices,
        },
    }
    write_json(args.out_dir / "gate-02-r2-protocol.json", protocol)

    # ------------------------------------------------------------------
    # 2. Verify corpus integrity
    # ------------------------------------------------------------------
    print("Step 1: Verifying frozen corpus integrity...")
    integrity = verify_corpus_integrity(args.corpus, args.pdf_dir)
    write_json(args.out_dir / "corpus-input-integrity.json", integrity.to_dict())

    if not integrity.integrity_passed:
        print("ERROR: Corpus integrity check FAILED.")
        print(f"  Documents: {integrity.document_count_actual}/{integrity.document_count_expected}")
        print(f"  Pages: {integrity.total_pages_actual}/{integrity.total_pages_expected}")
        print(f"  SHA256 match: {integrity.all_sha256_match}")
        print(f"  Page count match: {integrity.all_page_counts_match}")
        return 1

    print(f"  OK: {integrity.document_count_actual} documents, "
          f"{integrity.total_pages_actual} pages verified")

    # Load manifest documents for downstream use
    manifest = load_corpus_manifest(args.corpus)
    documents = sorted(
        manifest.get("documents", []),
        key=lambda d: str(d.get("document_id") or ""),
    )
    # Add resolved pdf_path
    for doc in documents:
        doc_id = str(doc["document_id"])
        filename = str(doc["filename"])
        doc["pdf_path"] = str(args.pdf_dir / filename)

    # ------------------------------------------------------------------
    # 3. Capture runtime environment
    # ------------------------------------------------------------------
    print("Step 2: Capturing runtime environment...")
    runtime_env = capture_runtime_environment(
        args.mineru, config, args.cuda_visible_devices,
    )
    runtime_env["nvidia_smi"] = nvidia_snapshot()
    runtime_env["runbook_path"] = str(RUNBOOK_PATH)
    runtime_env["pip_freeze_hash"] = None  # Filled below if possible
    write_json(args.out_dir / "runtime-environment-manifest.json", runtime_env)
    print(f"  MinerU: {runtime_env.get('mineru_version')}")
    print(f"  Torch: {runtime_env.get('torch_version')}")
    print(f"  CUDA available: {runtime_env.get('torch_cuda_available')}")

    # ------------------------------------------------------------------
    # 4. Run MinerU on all documents
    # ------------------------------------------------------------------
    if args.skip_mineru:
        print("Step 3: SKIPPING MinerU execution (--skip-mineru)")
        doc_results = []
        for doc in documents:
            doc_results.append({
                "document_id": doc["document_id"],
                "status": "skipped",
                "input_pdf_sha256": doc["file_sha256"],
                "config_hash": config.config_hash,
                "output_manifest_hash": None,
                "processed_page_count": 0,
                "started_at": "",
                "completed_at": "",
                "elapsed_seconds": 0,
                "return_code": None,
                "error": None,
                "output_dir": str(args.mineru_output / doc["document_id"]),
            })
    else:
        print(f"Step 3: Running MinerU Hybrid High on {len(documents)} documents...")
        print(f"  GPU: {args.cuda_visible_devices}")
        print(f"  Output: {args.mineru_output}")
        results = run_full_corpus(
            mineru_bin=args.mineru,
            documents=documents,
            output_root=args.mineru_output,
            config=config,
            cuda_visible_devices=args.cuda_visible_devices,
            tmpdir=tmpdir,
            timeout_seconds=args.timeout_seconds,
        )
        doc_results = [r.to_dict() for r in results]

    write_json(
        args.out_dir / "document-run-summary.json",
        {
            "document_count": len(doc_results),
            "completed": sum(1 for r in doc_results if r["status"] == "completed"),
            "failed": sum(1 for r in doc_results if r["status"] == "failed"),
            "skipped": sum(1 for r in doc_results if r["status"] == "skipped"),
            "results": doc_results,
        },
    )

    # Check for failures
    failed_docs = [r for r in doc_results if r["status"] == "failed"]
    if failed_docs and not args.skip_mineru:
        print(f"ERROR: {len(failed_docs)} documents failed:")
        for r in failed_docs:
            print(f"  {r['document_id']}: {r.get('error')}")
        return 1

    # ------------------------------------------------------------------
    # 5. Build page coverage
    # ------------------------------------------------------------------
    print("Step 4: Building page coverage records...")
    all_statuses = []
    for doc in documents:
        doc_id = str(doc["document_id"])
        expected_pc = int(doc.get("page_count", 0))
        doc_output = args.mineru_output / doc_id
        statuses = build_page_status(
            document_id=doc_id,
            expected_page_count=expected_pc,
            output_dir=doc_output,
        )
        all_statuses.extend(statuses)

    page_status_path = args.out_dir / "page-processing-status.jsonl.gz"
    write_page_status_gzip(page_status_path, all_statuses)

    coverage = summarize_page_coverage(all_statuses, documents)
    write_json(args.out_dir / "page-coverage-summary.json", coverage.to_dict())
    print(f"  Total pages: {coverage.total_pages}")
    print(f"  Processed: {coverage.processed}")
    print(f"  Processed (no table): {coverage.processed_no_table}")
    print(f"  Failed: {coverage.failed}")

    # ------------------------------------------------------------------
    # 6. Build deterministic output manifest
    # ------------------------------------------------------------------
    print("Step 5: Building deterministic output manifest...")
    output_manifest = build_output_manifest(args.mineru_output, documents)
    write_json(
        args.out_dir / "full-corpus-raw-output-manifest.json",
        output_manifest,
    )
    print(f"  Files: {len(output_manifest['files'])}")
    print(f"  Manifest hash: {output_manifest['manifest_hash']}")

    print("\nGate 02 R2 ingestion complete. Run audit, seal, and probe regression next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
