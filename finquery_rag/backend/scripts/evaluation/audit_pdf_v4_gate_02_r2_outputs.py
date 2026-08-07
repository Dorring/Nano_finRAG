"""Gate 02 R2: Audit raw MinerU outputs for integrity.

Checks each document's MinerU output for:
  - middle.json / content_list.json / model.json parseable
  - Page indices unique and in-range
  - No zero-byte required files
  - No missing artifact references

Reads ONLY the raw MinerU output.  No gold, questions, or governance.
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

from src.pdf_retrieval_v4.frozen_corpus_manifest import (  # noqa: E402
    load_corpus_manifest,
)
from src.pdf_retrieval_v4.mineru_output_integrity import (  # noqa: E402
    audit_full_corpus,
)

CORPUS_PATH = ROOT / "benchmarks/financial_rag_v1/corpus.json"
SHARED_NANOCHAT_ROOT = ROOT.parents[4]
DEFAULT_MINERU_OUTPUT = SHARED_NANOCHAT_ROOT / ".runtime/pdf-retrieval-v4-gate-02-r2/mineru"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--mineru-output", type=Path, default=DEFAULT_MINERU_OUTPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = load_corpus_manifest(args.corpus)
    documents = sorted(
        manifest.get("documents", []),
        key=lambda d: str(d.get("document_id") or ""),
    )

    print("Auditing MinerU output integrity...")
    results = audit_full_corpus(
        output_root=args.mineru_output,
        documents=documents,
    )

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r.integrity_passed)
    failed = total - passed

    invalid_json = sum(len(r.invalid_json_files) for r in results)
    zero_byte = sum(len(r.zero_byte_files) for r in results)
    duplicates = sum(len(r.duplicate_page_indices) for r in results)
    out_of_range = sum(len(r.out_of_range_page_indices) for r in results)

    summary = {
        "gate": "pdf_retrieval_v4_gate_02_r2",
        "documents_total": total,
        "documents_passed": passed,
        "documents_failed": failed,
        "invalid_json_files": invalid_json,
        "zero_byte_files": zero_byte,
        "duplicate_page_indices": duplicates,
        "out_of_range_page_indices": out_of_range,
        "all_passed": failed == 0,
        "records": [r.to_dict() for r in results],
    }

    write_json(args.out_dir / "raw-output-integrity.json", summary)

    print(f"  Documents: {passed}/{total} passed")
    print(f"  Invalid JSON: {invalid_json}")
    print(f"  Zero-byte files: {zero_byte}")
    print(f"  Duplicate pages: {duplicates}")
    print(f"  Out-of-range pages: {out_of_range}")

    if failed > 0:
        print("\nFAILED documents:")
        for r in results:
            if not r.integrity_passed:
                print(f"  {r.document_id}: page_count={r.page_count}/{r.expected_page_count}")
        return 1

    print("\nIntegrity audit PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
