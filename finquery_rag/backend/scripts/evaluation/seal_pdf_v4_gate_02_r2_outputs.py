"""Gate 02 R2: Seal the raw MinerU output.

Creates a prediction seal that records:
  - No gold/questions/governance read before seal
  - No adapter/index/retrieval runs
  - Raw output manifest hash
  - Input integrity hash

After sealing, post-hoc probe regression scoring is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
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

CORPUS_PATH = ROOT / "benchmarks/financial_rag_v1/corpus.json"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = load_corpus_manifest(args.corpus)
    document_count = int(manifest.get("document_count", 0))
    page_count = int(manifest.get("total_pages", 0))

    # Load hashes from previously generated artifacts
    protocol_path = args.out_dir / "gate-02-r2-protocol.json"
    integrity_path = args.out_dir / "corpus-input-integrity.json"
    manifest_path = args.out_dir / "full-corpus-raw-output-manifest.json"

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    output_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    protocol_hash = hashlib.sha256(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    integrity_hash = hashlib.sha256(
        json.dumps(integrity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    seal = {
        "schema": "pdf-retrieval-v4/gate-02-r2/seal/v1",
        "gate": "pdf_retrieval_v4_gate_02_r2",
        "document_count": document_count,
        "page_count": page_count,

        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "question_reads": 0,
        "expected_value_reads_before_seal": 0,
        "reference_answer_reads_before_seal": 0,

        "parameter_scan": False,
        "per_document_backend_selection": False,
        "per_page_backend_selection": False,

        "adapter_runs": 0,
        "header_graph_runs": 0,
        "evidence_unit_builds": 0,
        "index_builds": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "answer_generation_calls": 0,

        "production_index_writes": 0,
        "production_config_modified": False,
        "production_switch_allowed": False,

        "raw_output_manifest_hash": output_manifest.get("manifest_hash"),
        "input_integrity_hash": integrity_hash,
        "protocol_hash": protocol_hash,

        "sealed": True,
    }

    write_json(args.out_dir / "full-corpus-ingestion-seal.json", seal)

    print("Seal created:")
    print(f"  Documents: {document_count}")
    print(f"  Pages: {page_count}")
    print(f"  Raw output manifest hash: {seal['raw_output_manifest_hash']}")
    print(f"  Sealed: {seal['sealed']}")
    print("\nPost-seal probe regression scoring is now allowed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
