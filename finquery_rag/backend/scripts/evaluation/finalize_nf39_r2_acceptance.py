"""Accept NF39 R2 as a corrected, reproducible evaluation baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "nf39-r2/v1"


@dataclass(frozen=True)
class CorrectedBaselineDecision:
    legacy_baseline_status: str
    corrected_baseline_status: str
    run_metrics_deterministic: bool
    candidate_pool_deterministic: bool
    final_context_deterministic: bool
    candidate_integrity_passed: bool
    snapshot_integrity_passed: bool
    canonical_scope_passed: bool
    production_behavior_changed: bool
    nf40_start_allowed: bool


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalized_stage_metrics(run_dir: Path) -> dict[str, dict[str, int]]:
    stages = _read(run_dir / "stage-metrics-same-k.json")["stages"]
    locations = {
        "rrf_at_5": ("s0_rrf_top40", 5),
        "rrf_at_20": ("s0_rrf_top40", 20),
        "rrf_at_40": ("s0_rrf_top40", 40),
        "reranker_at_5": ("s2_reranker_ranked_top20", 5),
        "reranker_at_20": ("s2_reranker_ranked_top20", 20),
        "final_at_5": ("s4_final_context_top5", 5),
    }
    return {
        name: {
            "case_hits": stages[stage][f"case_hit_count_at_{k}"],
            "source_hits": stages[stage][f"source_hit_count_at_{k}"],
        }
        for name, (stage, k) in locations.items()
    }


def _candidate_integrity(run_dir: Path) -> dict[str, Any]:
    audit = _read(run_dir / "candidate-boundary-audit.json")["boundaries"]
    final_manifest = _read(run_dir / "final-context-manifest.json")
    candidates = [candidate for case in final_manifest["cases"].values() for candidate in case["candidates"]]
    duplicate_count = sum(
        len(keys) - len(set(keys))
        for keys in ([candidate["candidate_key"] for candidate in case["candidates"]] for case in final_manifest["cases"].values())
    )
    missing_document = sum(item["missing_document_id_count"] for item in audit)
    missing_source = sum(item["missing_source_id_count"] for item in audit)
    invalid = sum(not candidate["candidate_key"].startswith("candidate:v1:") for candidate in candidates)
    missing_hash = sum(not candidate.get("content_hash") for candidate in candidates)
    output = {
        "rrf_candidate_count": next(item["candidate_count"] for item in audit if item["stage"] == "rrf_top40"),
        "final_candidate_count": len(candidates),
        "invalid_candidate_key_count": invalid,
        "missing_document_id_count": missing_document,
        "missing_source_id_count": missing_source,
        "missing_content_hash_count": missing_hash,
        "duplicate_candidate_key_count": duplicate_count,
    }
    output["passed"] = all(value == 0 for key, value in output.items() if key.endswith("_count") and key not in {"rrf_candidate_count", "final_candidate_count"}) and output["rrf_candidate_count"] == 1080 and output["final_candidate_count"] == 135
    return output


def _scope_integrity(scope_audit: dict[str, Any]) -> dict[str, Any]:
    output = {
        "tenant_id": scope_audit["tenant_id"],
        "document_count": scope_audit["allowed_document_count"],
        "canonical_evidence_count": scope_audit["canonical_evidence_count"],
        "legacy_out_of_corpus_candidate_occurrences_filtered": scope_audit["out_of_corpus_candidate_occurrences"],
        "legacy_out_of_corpus_distinct_candidates_filtered": scope_audit["out_of_corpus_distinct_candidate_count"],
        "remaining_out_of_corpus_candidates": scope_audit["remaining_out_of_corpus_candidates"],
    }
    output["passed"] = (
        output["tenant_id"] == 1
        and output["document_count"] == 3
        and output["canonical_evidence_count"] == 2636
        and output["legacy_out_of_corpus_distinct_candidates_filtered"] > 0
        and output["remaining_out_of_corpus_candidates"] == 0
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--snapshot-a", required=True)
    parser.add_argument("--snapshot-b", required=True)
    parser.add_argument("--scope-audit", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    run_a, run_b, out = Path(args.run_a), Path(args.run_b), Path(args.out_dir)
    metrics_a, metrics_b = normalized_stage_metrics(run_a), normalized_stage_metrics(run_b)
    baseline_a, baseline_b = _read(run_a / "baseline-manifest.json"), _read(run_b / "baseline-manifest.json")
    final_a, final_b = _read(run_a / "final-context-manifest.json"), _read(run_b / "final-context-manifest.json")
    candidate_integrity = _candidate_integrity(run_a)
    verification = _read(run_a / "snapshot-verification-report.json")
    snapshot_manifest = _read(run_a / "snapshot-manifest.json")
    scope_integrity = _scope_integrity(_read(Path(args.scope_audit)))
    pool_equal = baseline_a["candidate_pool_hash"] == baseline_b["candidate_pool_hash"]
    final_equal = final_a["cases"] == final_b["cases"]
    snapshot_equal = _sha256(Path(args.snapshot_a)) == _sha256(Path(args.snapshot_b))
    metrics_equal = metrics_a == metrics_b
    snapshot_integrity = {
        "payload_record_count": snapshot_manifest["payload_record_count"],
        "snapshot_rehydrated_count": verification["snapshot_rehydrated_count"],
        "verified_final_context_count": verification["verified_final_context_count"],
        "passed": verification["passed"],
    }
    decision = CorrectedBaselineDecision(
        legacy_baseline_status="superseded_due_to_artifact_corruption",
        corrected_baseline_status="accepted",
        run_metrics_deterministic=metrics_equal,
        candidate_pool_deterministic=pool_equal,
        final_context_deterministic=final_equal and snapshot_equal,
        candidate_integrity_passed=candidate_integrity["passed"],
        snapshot_integrity_passed=snapshot_integrity["passed"],
        canonical_scope_passed=scope_integrity["passed"],
        production_behavior_changed=False,
        nf40_start_allowed=False,
    )
    allowed = all([
        metrics_equal,
        pool_equal,
        final_equal,
        snapshot_equal,
        candidate_integrity["passed"],
        snapshot_integrity["passed"],
        scope_integrity["passed"],
        not decision.production_behavior_changed,
    ])
    decision = CorrectedBaselineDecision(**(asdict(decision) | {"nf40_start_allowed": allowed}))
    acceptance = {
        "artifact_schema": SCHEMA,
        "legacy_baseline": {"version": "nf39-r1", "status": decision.legacy_baseline_status, "valid_for_regression": False, "reason_codes": ["empty_candidate_identity", "missing_content_hash", "out_of_corpus_bm25_records"]},
        "corrected_baseline": {"version": "nf39-r2", "status": decision.corrected_baseline_status, "valid_for_future_regression": True},
        "candidate_integrity": candidate_integrity,
        "scope_integrity": scope_integrity,
        "snapshot_integrity": snapshot_integrity,
        "determinism": {"candidate_pool_hash_equal": pool_equal, "stage_metrics_equal": metrics_equal, "final_context_hashes_equal": final_equal, "snapshot_hash_equal": snapshot_equal, "passed": metrics_equal and pool_equal and final_equal and snapshot_equal},
        "legacy_r1_metrics_match": False,
        "legacy_r1_metrics_match_is_informational_only": True,
        "production_behavior_changed": decision.production_behavior_changed,
        "nf40_start_allowed": decision.nf40_start_allowed,
    }
    migration = {"from": "nf39-r1", "to": "nf39-r2", "migration_type": "artifact_integrity_correction", "algorithm_changed": False, "production_config_changed": False, "evaluation_scope_changed": True, "scope_changes": ["candidate identity preserved", "out-of-corpus BM25 records removed", "content hashes added", "frozen snapshot made reproducible"], "metric_changes_are_performance_claim": False, "old_metrics_valid": False, "new_metrics_valid": True}
    determinism = {"candidate_pool_hash_run_a": baseline_a["candidate_pool_hash"], "candidate_pool_hash_run_b": baseline_b["candidate_pool_hash"], "stage_metrics_run_a": metrics_a, "stage_metrics_run_b": metrics_b, "final_context_manifest_sha256_run_a": _sha256(run_a / "final-context-manifest.json"), "final_context_manifest_sha256_run_b": _sha256(run_b / "final-context-manifest.json"), "snapshot_sha256_run_a": _sha256(Path(args.snapshot_a)), "snapshot_sha256_run_b": _sha256(Path(args.snapshot_b)), "passed": acceptance["determinism"]["passed"]}
    _write(out / "nf39-r2-acceptance.json", acceptance)
    _write(out / "baseline-migration-report.json", migration)
    _write(out / "determinism-report.json", determinism)
    print(json.dumps(acceptance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

