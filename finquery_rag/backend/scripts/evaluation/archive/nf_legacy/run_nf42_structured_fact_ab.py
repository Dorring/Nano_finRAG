"""Evaluate generic structured-fact extraction on frozen NF39 R2 contexts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.evaluation.evaluation import load_jsonl_cases
from src.evaluation.nf40_frozen_context import load_frozen_contexts
from src.evaluation.nf40_start_gate import require_verified_nf39_r2_inputs
from src.evaluation.nf42_structured_fact_extraction import extract_structured_facts, fact_matches_case


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--nf40-case-attribution", required=True, type=Path)
    parser.add_argument("--nf41-case-attribution", required=True, type=Path)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--final-context-manifest", required=True, type=Path)
    parser.add_argument("--frozen-payload-path", required=True, type=Path)
    parser.add_argument("--expected-payload-sha256", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _facts_payload(facts) -> list[dict]:
    return [{
        "fact_id": fact.fact_id, "candidate_key": fact.candidate_key,
        "candidate_rank": fact.candidate_rank, "document_id": fact.document_id,
        "page": fact.page, "value_expression": fact.value_expression,
        "canonical_value": fact.canonical_value, "value_type": fact.value_type,
        "currency": fact.currency, "period": fact.period,
        "extraction_kind": fact.extraction_kind,
    } for fact in facts]


def main() -> None:
    args = _args()
    require_verified_nf39_r2_inputs(
        acceptance_path=args.acceptance, snapshot_manifest_path=args.snapshot_manifest,
        frozen_payload_path=args.frozen_payload_path, expected_payload_sha256=args.expected_payload_sha256,
    )
    cases = load_jsonl_cases(args.cases)
    if len(cases) != 27:
        raise ValueError(f"NF42 requires 27 labeled cases, got {len(cases)}")
    contexts = load_frozen_contexts(args.frozen_payload_path, args.final_context_manifest)
    nf40 = {item["case_id"]: item for item in json.loads(args.nf40_case_attribution.read_text(encoding="utf-8"))["cases"]}
    nf41 = {item["case_id"]: item for item in json.loads(args.nf41_case_attribution.read_text(encoding="utf-8"))["cases"]}
    eligible = [case for case in cases if nf40[case.case_id]["context_coverage"] == "all_gold_in_final"]
    records = []
    baseline_hits = 0
    variant_hits = 0
    regressions = []
    improvements = []
    all_public_facts = []
    for case in eligible:
        production_failure = nf41[case.case_id]["production_failure"]
        baseline_hit = production_failure in {
            "correct", "production_fact_available_not_selected", "production_fact_selected_render_wrong",
        }
        facts = extract_structured_facts(contexts[case.case_id])
        variant_hit = any(fact_matches_case(fact, case) for fact in facts)
        baseline_hits += int(baseline_hit)
        variant_hits += int(variant_hit)
        if baseline_hit and not variant_hit:
            regressions.append(case.case_id)
        if not baseline_hit and variant_hit:
            improvements.append(case.case_id)
        records.append({
            "case_id": case.case_id,
            "baseline_production_fact_available": baseline_hit,
            "structured_fact_available": variant_hit,
            "structured_fact_count": len(facts),
        })
        all_public_facts.append({"case_id": case.case_id, "facts": _facts_payload(facts)})
    denominator = len(eligible)
    comparison = {
        "eligible_all_gold_case_count": denominator,
        "baseline_production_fact_coverage": {"count": baseline_hits, "denominator": denominator, "rate": baseline_hits / denominator if denominator else 1.0},
        "structured_fact_coverage": {"count": variant_hits, "denominator": denominator, "rate": variant_hits / denominator if denominator else 1.0},
        "improved_cases": improvements,
        "regressed_cases": regressions,
        "gate_passed": variant_hits > baseline_hits and not regressions,
    }
    artifact_seed = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    _write(args.out_dir / "baseline-manifest.json", {
        "artifact_schema": "nf42/v1", "case_count": len(cases),
        "eligible_all_gold_case_count": denominator,
        "frozen_payload_hash": args.expected_payload_sha256,
        "candidate_fact_report_hash": hashlib.sha256(artifact_seed.encode("utf-8")).hexdigest(),
        "production_behavior_changed": False, "retrieval_calls": 0, "model_chat_completion_requests": 0,
    })
    _write(args.out_dir / "structured-fact-report.json", {"artifact_schema": "nf42/v1", "cases": all_public_facts})
    _write(args.out_dir / "coverage-comparison.json", comparison)
    _write(args.out_dir / "case-diff.json", {"cases": records})
    _write(args.out_dir / "nf42-acceptance.json", {
        "artifact_schema": "nf42/v1", "gate_passed": comparison["gate_passed"],
        "production_switch_allowed": False, "production_behavior_changed": False,
        "retrieval_calls": 0, "model_chat_completion_requests": 0,
    })


if __name__ == "__main__":
    main()
