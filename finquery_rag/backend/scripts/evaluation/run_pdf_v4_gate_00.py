"""Freeze inputs and protocol for PDF Retrieval V4 Gate 00.

Gate 00 is an audit-only operation.  It hashes already-present inputs and
records software/configuration identities; it never parses a PDF, calls
MinerU, builds an index, or reads a retrieval label at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks/financial_rag_v1"
DATA = BENCHMARK / "data"
GOVERNANCE = BENCHMARK / "governance"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-00"
DEFAULT_BENCHMARK_PDFS = ROOT / "runtime/benchmark/financial_rag_v1/review-package/pdfs"
DEFAULT_DEVELOPMENT_PDFS = ROOT / "runtime/benchmark/financial_rag_v1/development-pdfs"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_or_missing(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None, "size_bytes": None}
    return {"path": str(path), "exists": True, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def find_named_pdf(root: Path, filename: str) -> Path | None:
    if not root.is_dir():
        return None
    direct = root / filename
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def command_output(*command: str) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def document_inputs(corpus: dict[str, Any], pdf_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for document in corpus["documents"]:
        path = find_named_pdf(pdf_root, str(document["filename"]))
        record = sha256_or_missing(path or (pdf_root / str(document["filename"])))
        record.update({"document_id": document["document_id"], "filename": document["filename"], "expected_sha256": document.get("file_sha256")})
        if not record["exists"]:
            mismatches.append(f"missing:{document['document_id']}")
        elif record["sha256"] != document.get("file_sha256"):
            mismatches.append(f"hash_mismatch:{document['document_id']}")
        records.append(record)
    return records, mismatches


def label_identity_hash(labels_path: Path) -> tuple[str, int, int]:
    identities: list[dict[str, Any]] = []
    case_count = 0
    source_count = 0
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        case_count += 1
        for index, source in enumerate(record.get("expected_sources") or []):
            source_count += 1
            identities.append({"case_id": record["case_id"], "source_index": index, "candidate_key": source.get("candidate_key")})
    return payload_hash(identities), case_count, source_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-pdf-dir", type=Path, default=DEFAULT_BENCHMARK_PDFS)
    parser.add_argument("--development-pdf-dir", type=Path, default=DEFAULT_DEVELOPMENT_PDFS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-commit", default=None)
    args = parser.parse_args()

    corpus_path = BENCHMARK / "corpus.json"
    questions_path = DATA / "questions.golden.jsonl"
    labels_path = DATA / "labels.golden.jsonl"
    governance_path = GOVERNANCE / "benchmark-governance.jsonl"
    family_path = GOVERNANCE / "evidence-family-map.json"
    split_path = ROOT / "artifacts/evaluation/pdf-query-representation-v2/document-split-manifest.json"
    gate0_path = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-0/acceptance.json"
    gate1_path = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-1/acceptance.json"
    gate2_path = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-2/router-prediction-seal.json"
    gate3_path = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-3/gate-3-prediction-seal.json"

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    development_documents = json.loads(split_path.read_text(encoding="utf-8"))["documents"] if split_path.is_file() else []
    benchmark_pdfs, benchmark_issues = document_inputs(corpus, args.benchmark_pdf_dir)
    development_pdfs: list[dict[str, Any]] = []
    development_issues: list[str] = []
    for document in development_documents:
        filename = str(document.get("filename") or f"{document['document_id']}.pdf")
        path = find_named_pdf(args.development_pdf_dir, filename)
        record = sha256_or_missing(path or (args.development_pdf_dir / filename))
        record.update({"document_id": document["document_id"], "issuer": document.get("issuer"), "filename": filename, "fold": document.get("fold")})
        if not record["exists"]:
            development_issues.append(f"missing:{document['document_id']}")
        development_pdfs.append(record)

    source_identity_sha, label_case_count, source_count = label_identity_hash(labels_path)
    code_commit = args.code_commit or command_output("git", "rev-parse", "HEAD") or "unknown"
    gate_artifacts = {}
    for name, path in (("gate0_acceptance", gate0_path), ("gate1_acceptance", gate1_path), ("gate2_prediction_seal", gate2_path), ("gate3_prediction_seal", gate3_path)):
        gate_artifacts[name] = {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None}

    protocol = {
        "schema": "pdf-retrieval-v4/gate-00/protocol/v1",
        "gate": "pdf_retrieval_v4_gate_00",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "code_commit": code_commit,
        "benchmark_pdf_hashes": benchmark_pdfs,
        "development_pdf_hashes": development_pdfs,
        "question_hash": sha256_file(questions_path),
        "label_file_hash_opaque": sha256_file(labels_path),
        "strict_source_identity_hash": source_identity_sha,
        "candidate_universe_hash": json.loads((ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-0/candidate-universe-manifest.json").read_text(encoding="utf-8")).get("candidate_universe_hash") if (ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-0/candidate-universe-manifest.json").is_file() else None,
        "gate_artifact_hashes": gate_artifacts,
        "gate1_governance_hash": sha256_file(governance_path),
        "gate1_evidence_family_hash": sha256_file(family_path),
        "gate2_router_seal_hash": gate_artifacts["gate2_prediction_seal"]["sha256"],
        "gate3_prediction_seal_hash": gate_artifacts["gate3_prediction_seal"]["sha256"],
        "mineru_backend_configs": {
            "hybrid_high": {"backend": "hybrid", "parse_method": "auto", "formula_enable": True, "table_enable": True, "ocr": False},
            "pipeline_auto_ocr": {"backend": "pipeline", "parse_method": "auto", "formula_enable": True, "table_enable": True, "ocr": True},
        },
        "embedding_model": package_version("sentence-transformers") or "not-installed-in-freeze-environment",
        "reranker_model": "src.services.reranker.HeuristicReranker / production-configured-model",
        "parameter_scan": False,
        "per_query_oracle": False,
        "production_index_write": False,
        "post_score_tuning": False,
        "mineru_calls": 0,
        "retrieval_runs": 0,
        "index_builds": 0,
    }
    software = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": code_commit,
        "PyMuPDF": package_version("PyMuPDF") or package_version("fitz"),
        "mineru": package_version("mineru") or package_version("magic-pdf"),
        "sentence_transformers": package_version("sentence-transformers"),
        "torch": package_version("torch"),
        "numpy": package_version("numpy"),
        "uv_lock_sha256": sha256_file(ROOT / "uv.lock") if (ROOT / "uv.lock").is_file() else None,
    }
    input_integrity = {
        "corpus_sha256": sha256_file(corpus_path),
        "questions_sha256": sha256_file(questions_path),
        "labels_sha256_opaque": sha256_file(labels_path),
        "benchmark_pdf_count": len(benchmark_pdfs),
        "development_pdf_count": len(development_pdfs),
        "benchmark_pdf_issues": benchmark_issues,
        "development_pdf_issues": development_issues,
        "label_case_count": label_case_count,
        "strict_source_count": source_count,
        "original_benchmark_files_modified": False,
    }
    complete = len(benchmark_pdfs) == 8 and len(development_pdfs) == 3 and not benchmark_issues and not development_issues and label_case_count == 72 and source_count == 80
    decision = "v4_gate_00_inputs_frozen" if complete else "v4_gate_00_inputs_incomplete"
    write_json(args.out_dir / "protocol.json", protocol)
    write_json(args.out_dir / "input-integrity.json", input_integrity)
    write_json(args.out_dir / "software-manifest.json", software)
    write_json(args.out_dir / "benchmark-lineage.json", {"benchmark_id": corpus["benchmark_id"], "benchmark_documents": [item["document_id"] for item in corpus["documents"]], "development_documents": development_documents, "prior_gates": gate_artifacts, "lineage_status": "frozen" if complete else "blocked_missing_or_mismatched_input"})
    write_json(args.out_dir / "acceptance.json", {"gate": "pdf_retrieval_v4_gate_00", "gate_passed": complete, "decision": decision, "next_gate": "mineru_capability_probe" if complete else "stop_and_resolve_frozen_inputs", "evaluation_type": protocol["evaluation_type"], "runtime_gold_reads": 0, "runtime_governance_reads": 0, "expected_value_reads": 0, "reference_answer_reads": 0, "parameter_scan": False, "per_query_oracle": False, "production_index_writes": 0, "production_default_config_modified": False, "candidate_identity_conflicts": 0, "duplicate_views": 0, "mineru_calls": 0, "retrieval_runs": 0, "index_builds": 0, "production_switch_allowed": False})
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
