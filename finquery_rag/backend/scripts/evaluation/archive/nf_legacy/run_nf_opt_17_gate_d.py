"""NF-OPT-17 Gate D: materialize a dev-only Shadow Candidate Corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.nf_opt_17 import build_shadow_candidate_corpus

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/nf-opt-17-gate-d"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_annotations(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate_manifest(candidate: dict[str, Any]) -> dict[str, Any]:
    """Commit identity and content hashes, not raw candidate text."""
    return {
        key: candidate[key]
        for key in (
            "candidate_key",
            "issuer",
            "source_document",
            "xbrl_concept",
            "context_id",
            "period_end",
            "period_kind",
            "table_index",
            "row_index",
            "content_sha256",
        )
    }


def run(args: argparse.Namespace) -> int:
    annotations = _read_annotations(args.annotations)
    if len(annotations) != args.expected_annotations:
        raise ValueError(f"expected {args.expected_annotations} annotations, got {len(annotations)}")
    candidates, lineage = build_shadow_candidate_corpus(annotations)
    candidate_keys = {str(candidate["candidate_key"]) for candidate in candidates}
    missing_lineage = [row for row in lineage if str(row["candidate_key"]) not in candidate_keys]
    if missing_lineage:
        raise ValueError("annotation lineage references missing shadow candidates")
    if any(not str(candidate["candidate_key"]).startswith("dev:sec:") for candidate in candidates):
        raise ValueError("non-development candidate identity leaked into shadow corpus")
    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_output.write_text(
        "".join(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n" for candidate in candidates),
        encoding="utf-8",
    )
    raw_candidate_bytes = args.candidate_output.read_bytes()
    issuer_counts = Counter(str(candidate["issuer"]) for candidate in candidates)
    role_counts = Counter(str(row["role"]) for row in lineage)
    manifest = {
        "schema": "nf-opt-17/shadow-candidate-corpus-manifest/v1",
        "candidate_count": len(candidates),
        "lineage_record_count": len(lineage),
        "runtime_candidate_corpus_sha256": _sha(raw_candidate_bytes),
        "runtime_candidate_corpus_committed": False,
        "candidate_records": [_candidate_manifest(candidate) for candidate in candidates],
    }
    lineage_report = {
        "annotation_count": len(annotations),
        "candidate_count": len(candidates),
        "lineage_record_count": len(lineage),
        "missing_candidate_lineage_count": len(missing_lineage),
        "issuer_counts": dict(sorted(issuer_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "production_candidate_identity_count": 0,
    }
    acceptance = {
        "schema": "nf-opt-17/gate-d/acceptance/v1",
        "annotation_count": len(annotations),
        "shadow_candidate_count": len(candidates),
        "lineage_record_count": len(lineage),
        "missing_candidate_lineage_count": len(missing_lineage),
        "frozen_benchmark_question_or_label_reads": 0,
        "production_candidate_identity_count": 0,
        "model_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "decision": "hard_negative_shadow_candidate_corpus_ready",
        "next_gate": "nf-opt-17-gate-e-frozen-reranker-training-config",
    }
    _write(args.out_dir / "shadow-candidate-corpus-manifest.json", manifest)
    _write(args.out_dir / "annotation-candidate-lineage-report.json", lineage_report)
    _write(args.out_dir / "next-gate.json", {
        "decision": acceptance["decision"],
        "next_gate": acceptance["next_gate"],
        "production_switch_allowed": False,
    })
    _write(args.out_dir / "nf-opt-17-gate-d-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-annotations", type=int, default=80)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
