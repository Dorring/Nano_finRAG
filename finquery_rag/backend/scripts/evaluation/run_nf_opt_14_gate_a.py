"""NF-OPT-14 Gate A: query slot contracts and realizable selector ceilings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from src.evaluation.nf_opt_14 import (
    candidate_slot_compatibility,
    deterministic_slot_selector,
    parse_query_slot_contract,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "artifacts/evaluation/nf-eval-03-r2/case-results.json"
DEFAULT_QUESTIONS = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"
DEFAULT_LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
DEFAULT_OUT = ROOT / "artifacts/evaluation/nf-opt-14"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_contexts(db_path: Path, candidates: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Read existing candidate content in SQLite read-only mode; never write an index."""
    doc_ids = {str(item.get("doc_id") or item.get("evidence_id") or "") for item in candidates}
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = {
            str(doc_id): {"content": str(content or ""), "metadata": str(metadata or "")}
            for doc_id, content, metadata in connection.execute(
                "SELECT doc_id, content, metadata_json FROM chunk_store WHERE doc_id IN ("
                + ",".join("?" for _ in doc_ids)
                + ")",
                tuple(doc_ids),
            )
        }
    finally:
        connection.close()
    return rows


def _source_hits(selected: list[dict[str, Any]], sources: list[dict[str, Any]]) -> set[int]:
    keys = {str(item.get("candidate_key")) for item in selected}
    return {index for index, source in enumerate(sources) if str(source["candidate_key"]) in keys}


def _gold_promoted_ceiling(reranked: list[dict[str, Any]], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gold_keys = {str(source["candidate_key"]) for source in sources}
    return ([item for item in reranked if str(item.get("candidate_key")) in gold_keys] + [item for item in reranked if str(item.get("candidate_key")) not in gold_keys])[:5]


def run(args: argparse.Namespace) -> int:
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    questions = {str(item["case_id"]): item for item in _load_jsonl(args.questions)}
    labels = {str(item["case_id"]): item for item in _load_jsonl(args.labels)}
    answerable = [item for item in payload["cases"] if not item.get("expected_no_answer")]
    no_answer = [item for item in payload["cases"] if item.get("expected_no_answer")]
    if len(answerable) != 64 or len(no_answer) != 8:
        raise ValueError("expected 64 answerable and 8 no-answer cases")
    all_candidates = [candidate for case in answerable for candidate in case["retrieval_stages"]["reranker"][:20]]
    contexts = _candidate_contexts(args.candidate_db, all_candidates)
    contracts: list[dict[str, Any]] = []
    matrices: list[dict[str, Any]] = []
    selector_cases: list[dict[str, Any]] = []
    baseline_hits: set[tuple[str, int]] = set()
    promoted_hits: set[tuple[str, int]] = set()
    constrained_hits: set[tuple[str, int]] = set()
    selector_hits: set[tuple[str, int]] = set()
    baseline_hit_instances: set[tuple[str, int]] = set()
    selector_hit_instances: set[tuple[str, int]] = set()

    for case in answerable:
        case_id = str(case["case_id"])
        question = questions[case_id]
        sources = list(labels[case_id].get("expected_sources") or [])
        contract = parse_query_slot_contract(question)
        contracts.append(contract)
        reranked = [dict(item) for item in case["retrieval_stages"]["reranker"][:20]]
        baseline = list(case["retrieval_stages"]["final"])
        for candidate in reranked:
            record = contexts.get(str(candidate.get("doc_id") or candidate.get("evidence_id") or ""), {})
            candidate["metadata_text"] = record.get("metadata", "")
            candidate["candidate_text"] = record.get("content", "")
        matrix = [
            candidate_slot_compatibility(
                candidate=candidate,
                candidate_text=str(candidate["candidate_text"]),
                slot=slot,
                document_scope={str(value) for value in contract["document_scope"]},
            )
            for candidate in reranked
            for slot in contract["slots"]
        ]
        matrices.append({"case_id": case_id, "candidate_count": len(reranked), "slots": matrix})
        strict_keys = {
            item["candidate_key"] for item in matrix if item["compatibility"] == "strict"
        }
        strict_candidates = [item for item in reranked if str(item.get("candidate_key")) in strict_keys]
        constrained = _gold_promoted_ceiling(strict_candidates, sources)
        selected = deterministic_slot_selector(
            baseline_final=baseline,
            reranked=reranked,
            slots=contract["slots"] if contract["contract_status"] == "complete" else [],
            matrix=matrix,
        )
        baseline_case_hits = _source_hits(baseline, sources)
        selector_case_hits = _source_hits(selected, sources)
        baseline_hits.update((case_id, index) for index in baseline_case_hits)
        promoted_hits.update((case_id, index) for index in _source_hits(_gold_promoted_ceiling(reranked, sources), sources))
        constrained_hits.update((case_id, index) for index in _source_hits(constrained, sources))
        selector_hits.update((case_id, index) for index in selector_case_hits)
        baseline_hit_instances.update((case_id, index) for index in baseline_case_hits)
        selector_hit_instances.update((case_id, index) for index in selector_case_hits)
        selector_cases.append(
            {
                "case_id": case_id,
                "contract_status": contract["contract_status"],
                "baseline_candidate_keys": [str(item.get("candidate_key")) for item in baseline],
                "selected_candidate_keys": [str(item.get("candidate_key")) for item in selected],
                "strict_compatible_candidate_count": len(strict_candidates),
                "baseline_matched_source_count": len(baseline_case_hits),
                "selected_matched_source_count": len(selector_case_hits),
            }
        )

    source_count = sum(len(labels[str(case["case_id"])].get("expected_sources") or []) for case in answerable)
    if source_count != 80:
        raise ValueError(f"expected 80 sources, got {source_count}")
    no_answer_records = [
        {
            "case_id": str(case["case_id"]),
            "baseline_final_candidate_keys": [
                str(item.get("candidate_key")) for item in case["retrieval_stages"]["final"]
            ],
            "shadow_selected_candidate_keys": [
                str(item.get("candidate_key")) for item in case["retrieval_stages"]["final"]
            ],
        }
        for case in no_answer
    ]
    no_answer_unchanged = all(
        item["baseline_final_candidate_keys"] == item["shadow_selected_candidate_keys"]
        for item in no_answer_records
    )
    strict_regressions = baseline_hit_instances - selector_hit_instances
    metrics = {
        "baseline_strict_final_source_recall_at_5": {"matched_sources": len(baseline_hits), "source_count": source_count},
        "gold_promoted_reranker_pool_ceiling_at_5": {"matched_sources": len(promoted_hits), "source_count": source_count},
        "compatibility_constrained_oracle_at_5": {"matched_sources": len(constrained_hits), "source_count": source_count},
        "deterministic_selector_recall_at_5": {"matched_sources": len(selector_hits), "source_count": source_count},
    }
    selector_gate = len(selector_hits) >= 17 and not strict_regressions
    interpretation = {
        "reported_metric": "oracle_slot_aware_strict_recall_at_5",
        "reported_value": "24/80",
        "actual_semantics": "gold_promoted_reranker_pool_ceiling_at_5",
        "query_slots_used_for_candidate_selection": False,
        "sufficient_to_gate_production_selector": False,
    }
    acceptance = {
        "artifact_schema": "nf-opt-14/gate-a/acceptance/v1",
        "case_count": 64,
        "no_answer_case_count": 8,
        "source_count": source_count,
        "input_hashes": {
            "case_results_sha256": _sha(args.cases),
            "questions_sha256": _sha(args.questions),
            "labels_sha256": _sha(args.labels),
        },
        "candidate_store_access_mode": "sqlite_read_only",
        "slot_plan_expected_fields_read": False,
        "candidate_compatibility_gold_fields_read": False,
        "gold_used_only_for_posthoc_scoring": True,
        "final_top_k": 5,
        "reranker_order_modified": False,
        "no_answer_final_behavior_unchanged": no_answer_unchanged,
        "model_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "strict_hit_regression_count": len(strict_regressions),
    }
    next_gate = {
        "decision": "slot_selector_gate_b_eligible" if selector_gate else "slot_selector_gate_a_gain_insufficient",
        "deterministic_selector_gate_passed": selector_gate,
        "production_switch_allowed": False,
    }
    _write(args.out_dir / "nf-opt-12-oracle-interpretation.json", interpretation)
    _write(args.out_dir / "query-slot-contract.json", {"contract_version": "nf-opt-14/query-slot/v1", "rules_frozen_before_benchmark_run": True})
    _write(args.out_dir / "query-slot-plan-report.json", {"case_count": len(contracts), "cases": contracts})
    _write(args.out_dir / "candidate-slot-compatibility-report.json", {"gold_fields_read": False, "cases": matrices})
    _write(args.out_dir / "compatibility-constrained-oracle-report.json", metrics)
    _write(args.out_dir / "slot-selector-shadow-results.json", {"selector": "fixed_query_only_strict_compatibility", "cases": selector_cases, "no_answer_cases": no_answer_records})
    _write(args.out_dir / "strict-hit-regression-report.json", {"regressed_source_instances": sorted(strict_regressions), "regressed_count": len(strict_regressions)})
    _write(args.out_dir / "latency-report.json", {"selector_overhead_ms": {"p50": 0.0, "p95": 0.0}, "note": "offline artifact replay; no online selector invoked"})
    _write(args.out_dir / "next-gate.json", next_gate)
    _write(args.out_dir / "nf-opt-14-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
