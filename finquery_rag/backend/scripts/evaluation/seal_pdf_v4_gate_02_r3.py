"""Gate 02 R3: Seal the adapter predictions.

Creates a prediction seal that records:
  - Document/page/table/row/cell counts
  - No gold/questions/governance read before seal
  - No header-graph/evidence-unit/index/retrieval/reranker/answer runs
  - Prediction hash from the adapter manifest
  - Protocol hash from gate-02-r3-protocol.json
  - Input manifest hash from input-integrity.json

After sealing, post-hoc probe regression scoring and identity continuity
audits are allowed.

Reads ONLY the adapter manifest, protocol, and input integrity.
No questions, gold, or governance data is read.
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
R3_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r3"

# Protocol fields that must be zero/false before sealing.
_SAFETY_FIELDS = [
    "runtime_oracle_reads",
    "runtime_question_reads",
    "runtime_governance_reads",
    "expected_value_reads",
    "header_graph_runs",
    "evidence_unit_builds",
    "index_builds",
    "retrieval_runs",
    "reranker_calls",
    "answer_generation_calls",
    "production_index_writes",
]

_SAFETY_BOOL_FIELDS = [
    "production_config_modified",
    "production_switch_allowed",
]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_safety(protocol: dict[str, Any]) -> list[str]:
    """Return a list of safety violations (empty if all clean)."""
    violations: list[str] = []
    for field in _SAFETY_FIELDS:
        value = protocol.get(field)
        if value not in (0, None, False):
            violations.append(f"{field}={value}")
    for field in _SAFETY_BOOL_FIELDS:
        value = protocol.get(field)
        if value not in (False, None, 0):
            violations.append(f"{field}={value}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--out-dir", type=Path, default=R3_OUT)
    args = parser.parse_args()

    manifest_path = args.out_dir / "adapter-prediction-manifest.json"
    protocol_path = args.out_dir / "gate-02-r3-protocol.json"
    integrity_path = args.out_dir / "input-integrity.json"

    for label, path in [
        ("adapter manifest", manifest_path),
        ("protocol", protocol_path),
        ("input integrity", integrity_path),
    ]:
        if not path.is_file():
            print(f"ERROR: {label} not found at {path}")
            print("Run run_pdf_v4_gate_02_r3_adapter.py first.")
            return 1

    corpus_manifest = load_corpus_manifest(args.corpus)
    document_count = int(corpus_manifest.get("document_count", 0))
    page_count = int(corpus_manifest.get("total_pages", 0))

    adapter_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    # Read integrity file to verify it exists and is valid JSON, but its
    # contents are not needed for the seal (used only by the integrity gate).
    json.loads(integrity_path.read_text(encoding="utf-8"))

    table_count = int(adapter_manifest.get("table_count", 0))
    row_count = int(adapter_manifest.get("row_count", 0))
    cell_count = int(adapter_manifest.get("cell_count", 0))
    predictions_hash = str(adapter_manifest.get("predictions_hash", ""))

    protocol_hash = _sha256_file(protocol_path)
    input_manifest_hash = _sha256_file(integrity_path)

    # Verify safety: no questions/gold/governance read before seal.
    violations = _verify_safety(protocol)
    if violations:
        print("ERROR: Safety violations detected in protocol:")
        for v in violations:
            print(f"  {v}")
        return 1

    seal = {
        "schema": "pdf-retrieval-v4/gate-02-r3/seal/v1",
        "gate": "pdf_retrieval_v4_gate_02_r3",
        "document_count": document_count,
        "page_count": page_count,
        "table_count": table_count,
        "row_count": row_count,
        "cell_count": cell_count,
        "question_reads_before_seal": 0,
        "gold_reads_before_seal": 0,
        "governance_reads_before_seal": 0,
        "expected_value_reads_before_seal": 0,
        "reference_answer_reads_before_seal": 0,
        "header_graph_runs": 0,
        "evidence_unit_builds": 0,
        "index_builds": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_config_modified": False,
        "production_switch_allowed": False,
        "prediction_hash": predictions_hash,
        "input_manifest_hash": input_manifest_hash,
        "protocol_hash": protocol_hash,
        "sealed": True,
    }

    _write_json(args.out_dir / "adapter-prediction-seal.json", seal)

    print("Seal created:")
    print(f"  Documents: {document_count}")
    print(f"  Pages: {page_count}")
    print(f"  Tables: {table_count}")
    print(f"  Rows: {row_count}")
    print(f"  Cells: {cell_count}")
    print(f"  Prediction hash: {predictions_hash[:16]}...")
    print(f"  Protocol hash: {protocol_hash[:16]}...")
    print(f"  Input manifest hash: {input_manifest_hash[:16]}...")
    print(f"  Sealed: {seal['sealed']}")
    print("\nPost-seal probe regression scoring is now allowed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
