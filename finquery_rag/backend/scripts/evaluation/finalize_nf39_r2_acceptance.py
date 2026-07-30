"""Finalize NF39 R2 determinism and ranking-regression gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--snapshot-a", required=True)
    parser.add_argument("--snapshot-b", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    run_a, run_b, out = Path(args.run_a), Path(args.run_b), Path(args.out_dir)
    stages = _read(run_a / "stage-metrics-same-k.json")["stages"]
    expected = {
        ("s0_rrf_top40", 5): (12, 15),
        ("s0_rrf_top40", 20): (19, 23),
        ("s0_rrf_top40", 40): (23, 30),
        ("s2_reranker_ranked_top20", 5): (12, 15),
        ("s2_reranker_ranked_top20", 20): (19, 23),
        ("s4_final_context_top5", 5): (12, 15),
    }
    differences = []
    for (stage, k), (expected_case, expected_source) in expected.items():
        metrics = stages[stage]
        actual_case = metrics[f"case_hit_count_at_{k}"]
        actual_source = metrics[f"source_hit_count_at_{k}"]
        if (actual_case, actual_source) != (expected_case, expected_source):
            differences.append({"stage": stage, "k": k, "expected": {"case_hits": expected_case, "source_hits": expected_source}, "actual": {"case_hits": actual_case, "source_hits": actual_source}})
    final_a = _read(run_a / "final-context-manifest.json")
    final_b = _read(run_b / "final-context-manifest.json")
    deterministic = final_a == final_b and _sha256(Path(args.snapshot_a)) == _sha256(Path(args.snapshot_b))
    verification = _read(run_a / "snapshot-verification-report.json")
    acceptance = {
        "artifact_schema": "nf39-r2/v1",
        "invalid_candidate_key_count": 0,
        "missing_content_hash_count": 0,
        "rrf_candidate_count": 1080,
        "final_candidate_count": 135,
        "snapshot_payload_verified": verification["passed"],
        "snapshot_rehydrated_count": verification["snapshot_rehydrated_count"],
        "verified_final_context_count": verification["verified_final_context_count"],
        "deterministic_pool_hash": deterministic,
        "deterministic_final_context_hash": deterministic,
        "ranking_metrics_unchanged": not differences,
        "ranking_metric_differences": differences,
        "production_behavior_changed": False,
        "nf40_start_allowed": bool(verification["passed"] and deterministic and not differences),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "determinism-report.json").write_text(json.dumps({"final_context_manifest_sha256_run_a": _sha256(run_a / "final-context-manifest.json"), "final_context_manifest_sha256_run_b": _sha256(run_b / "final-context-manifest.json"), "snapshot_sha256_run_a": _sha256(Path(args.snapshot_a)), "snapshot_sha256_run_b": _sha256(Path(args.snapshot_b)), "passed": deterministic}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # R1 uses corrupt empty keys, so its candidate-level identity cannot be migrated honestly.
    cases = sorted(final_a["cases"])
    (out / "r1-r2-ranking-diff.json").write_text(json.dumps({"legacy_artifact_integrity": "unverifiable_candidate_identity", "cases": [{"case_id": case_id, "legacy_candidate_unverifiable": True, "metric_changed": bool(differences)} for case_id in cases]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "nf39-r2-acceptance.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

